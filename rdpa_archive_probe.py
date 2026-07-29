#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin

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
SUPPORTED_FORMATS = ["GRIB", "GTiff", "NetCDF", "json"]


def session() -> requests.Session:
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-rdpa-probe/1.1 "
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


def safe_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


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
                        "bounds": [float(value) for value in dataset.bounds],
                        "transform": [float(value) for value in dataset.transform],
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


def response_record(response: requests.Response, requested_format: str) -> dict:
    record = {
        "requested_format": requested_format,
        "request_url": response.url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "content_length_header": response.headers.get("content-length"),
        "bytes_received": len(response.content),
        "response_prefix_hex": response.content[:24].hex(),
    }
    if response.status_code != 200 or len(response.content) <= 100:
        record["response_text_prefix"] = response.text[:2000]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", default=DEFAULT_TIME)
    parser.add_argument(
        "--bbox", default=",".join(str(value) for value in DEFAULT_BBOX)
    )
    parser.add_argument("--output", default="output/archive_probe/rdpa_probe.json")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc)
    encoded_collection = quote(COLLECTION, safe="")
    collection_url = urljoin(API_ROOT, f"collections/{encoded_collection}")
    coverage_url = urljoin(API_ROOT, f"collections/{encoded_collection}/coverage")
    output = {
        "generated_utc": generated.isoformat(),
        "status": "failed",
        "collection": COLLECTION,
        "collection_url": collection_url,
        "coverage_url": coverage_url,
        "query": {
            "bbox": args.bbox,
            "datetime": args.datetime,
            "properties": "1",
        },
        "metadata_request": {},
        "metadata": {},
        "attempts": [],
        "selected": None,
        "fatal_error": None,
        "next_step": "Inspect the saved request diagnostics and adapt archive retrieval before changing calibration.",
    }

    try:
        http = session()
        metadata_response = http.get(
            collection_url,
            params={"f": "json"},
            timeout=90,
        )
        output["metadata_request"] = response_record(metadata_response, "json")
        if metadata_response.status_code == 200:
            try:
                metadata = metadata_response.json()
                links = metadata.get("links", [])
                output["metadata"] = {
                    "title": metadata.get("title"),
                    "description": metadata.get("description"),
                    "extent": metadata.get("extent"),
                    "coverage_links": [
                        {
                            "type": link.get("type"),
                            "href": link.get("href"),
                            "rel": link.get("rel"),
                            "title": link.get("title"),
                        }
                        for link in links
                        if "coverage" in str(link.get("href", ""))
                    ],
                }
            except Exception as exc:
                output["metadata_parse_error"] = (
                    f"{exc.__class__.__name__}: {exc}"
                )
        else:
            output["metadata_request"]["response_text_prefix"] = (
                metadata_response.text[:2000]
            )

        for requested_format in SUPPORTED_FORMATS:
            try:
                response = http.get(
                    coverage_url,
                    params={
                        "f": requested_format,
                        "bbox": args.bbox,
                        "datetime": args.datetime,
                        "properties": "1",
                    },
                    timeout=300,
                )
                attempt = response_record(response, requested_format)
                if response.status_code == 200 and len(response.content) > 100:
                    if requested_format == "json":
                        try:
                            payload = response.json()
                            attempt["json_top_level_keys"] = sorted(payload.keys())
                            attempt["json_preview"] = payload
                            output["selected"] = {
                                "requested_format": requested_format,
                                "content_type": attempt["content_type"],
                                "bytes_received": len(response.content),
                                "coverage_json": payload,
                            }
                            output["attempts"].append(attempt)
                            break
                        except Exception as exc:
                            attempt["json_parse_error"] = (
                                f"{exc.__class__.__name__}: {exc}"
                            )
                    else:
                        attempt["raster"] = summarize_raster(response.content)
                        if attempt["raster"].get("open_succeeded"):
                            output["selected"] = {
                                "requested_format": requested_format,
                                "content_type": attempt["content_type"],
                                "bytes_received": len(response.content),
                                "raster": attempt["raster"],
                            }
                            output["attempts"].append(attempt)
                            break
                output["attempts"].append(attempt)
            except Exception as exc:
                output["attempts"].append(
                    {
                        "requested_format": requested_format,
                        "request_error": f"{exc.__class__.__name__}: {exc}",
                    }
                )

        if output["selected"] is not None:
            output["status"] = "passed"
            output["next_step"] = (
                "Build basin-clipped archived RDPA retrieval and multi-event hindcasting."
            )
    except Exception as exc:
        output["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        out.write_text(json.dumps(output, indent=2, default=safe_json))
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "selected_format": (
                        output["selected"].get("requested_format")
                        if output["selected"]
                        else None
                    ),
                    "fatal_error": output["fatal_error"],
                },
                indent=2,
            )
        )

    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
