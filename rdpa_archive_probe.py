#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

COLLECTION = "weather:rdpa:10km:6f"
API_ROOT = "https://api.weather.gc.ca/"
DEFAULT_BBOX = [-115.2, 52.9, -113.0, 54.2]
DEFAULT_TIME = "2026-07-01T00:00:00Z"


def session() -> requests.Session:
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-rdpa-probe/1.0 "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def summarize_raster(content: bytes) -> dict:
    result: dict = {"open_succeeded": False}
    try:
        with MemoryFile(content) as memory:
            with memory.open() as dataset:
                result.update(
                    {
                        "open_succeeded": True,
                        "driver": dataset.driver,
                        "width": dataset.width,
                        "height": dataset.height,
                        "band_count": dataset.count,
                        "crs": str(dataset.crs),
                        "bounds": list(dataset.bounds),
                        "transform": list(dataset.transform),
                        "descriptions": list(dataset.descriptions),
                        "dataset_tags": dataset.tags(),
                        "bands": [],
                    }
                )
                for band in range(1, dataset.count + 1):
                    array = dataset.read(band, masked=True).astype(float)
                    valid = array.compressed()
                    result["bands"].append(
                        {
                            "band": band,
                            "description": dataset.descriptions[band - 1],
                            "tags": dataset.tags(band),
                            "valid_cells": int(len(valid)),
                            "minimum": float(np.min(valid)) if len(valid) else None,
                            "maximum": float(np.max(valid)) if len(valid) else None,
                            "mean": float(np.mean(valid)) if len(valid) else None,
                        }
                    )
    except Exception as exc:
        result["open_error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", default=DEFAULT_TIME)
    parser.add_argument("--bbox", default=",".join(str(value) for value in DEFAULT_BBOX))
    parser.add_argument("--output", default="output/archive_probe/rdpa_probe.json")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    http = session()
    collection_url = urljoin(API_ROOT, f"collections/{COLLECTION}")
    metadata_response = http.get(
        collection_url,
        params={"f": "json"},
        timeout=90,
    )
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    links = metadata.get("links", [])
    formats = [
        {
            "type": link.get("type"),
            "href": link.get("href"),
            "rel": link.get("rel"),
            "title": link.get("title"),
        }
        for link in links
        if "coverage" in str(link.get("href", ""))
    ]

    coverage_url = urljoin(API_ROOT, f"collections/{COLLECTION}/coverage")
    attempts = []
    selected = None
    for requested_format in ["GRIB", "GRIB2", "NetCDF", "GeoTIFF"]:
        response = http.get(
            coverage_url,
            params={
                "f": requested_format,
                "bbox": args.bbox,
                "datetime": args.datetime,
            },
            timeout=300,
        )
        attempt = {
            "requested_format": requested_format,
            "request_url": response.url,
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "content_length_header": response.headers.get("content-length"),
            "bytes_received": len(response.content),
            "response_prefix_hex": response.content[:24].hex(),
        }
        if response.status_code == 200 and len(response.content) > 100:
            attempt["raster"] = summarize_raster(response.content)
            if attempt["raster"].get("open_succeeded"):
                selected = {
                    "requested_format": requested_format,
                    "content_type": attempt["content_type"],
                    "bytes_received": len(response.content),
                    "raster": attempt["raster"],
                }
                attempts.append(attempt)
                break
        else:
            attempt["response_text_prefix"] = response.text[:1000]
        attempts.append(attempt)

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if selected else "failed",
        "collection": COLLECTION,
        "collection_url": collection_url,
        "metadata": {
            "title": metadata.get("title"),
            "description": metadata.get("description"),
            "extent": metadata.get("extent"),
            "coverage_links": formats,
        },
        "query": {"bbox": args.bbox, "datetime": args.datetime},
        "attempts": attempts,
        "selected": selected,
        "next_step": (
            "Build monthly, basin-clipped archived RDPA retrieval and event hindcasting."
            if selected
            else "Inspect response formats and adapt the archive retrieval before any historical calibration is changed."
        ),
    }
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps({"status": output["status"], "selected": selected}, indent=2))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
