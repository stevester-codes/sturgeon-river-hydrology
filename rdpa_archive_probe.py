#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin

import numpy as np
import requests
from rasterio.io import MemoryFile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

COLLECTION = "weather:rdpa:10km:6f"
OGC_API_ROOT = "https://api.weather.gc.ca/"
WCS_URL = "https://geo.weather.gc.ca/geomet"
WCS_LAYER = "RDPA.6F_PR"
DEFAULT_BBOX = [-115.2, 52.9, -113.0, 54.2]
DEFAULT_TIME = "2026-07-01T00:00:00Z"


def session() -> requests.Session:
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-rdpa-probe/1.2 "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=3,
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


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 4:
        raise ValueError("bbox must contain minx,miny,maxx,maxy")
    minx, miny, maxx, maxy = values
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox bounds are not ordered")
    return minx, miny, maxx, maxy


def try_raster_response(
    response: requests.Response,
    label: str,
    attempts: list[dict],
):
    attempt = response_record(response, label)
    if response.status_code == 200 and len(response.content) > 100:
        attempt["raster"] = summarize_raster(response.content)
        attempts.append(attempt)
        if attempt["raster"].get("open_succeeded"):
            return {
                "source": label,
                "content_type": attempt["content_type"],
                "bytes_received": len(response.content),
                "raster": attempt["raster"],
            }
    else:
        attempts.append(attempt)
    return None


def wcs_parameters(
    bbox: tuple[float, float, float, float],
    timestamp: str,
    output_format: str,
    include_resolution: bool,
):
    minx, miny, maxx, maxy = bbox
    params = [
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", WCS_LAYER),
        ("SUBSETTINGCRS", "EPSG:4326"),
        ("OUTPUTCRS", "EPSG:4326"),
        ("SUBSET", f"x({minx},{maxx})"),
        ("SUBSET", f"y({miny},{maxy})"),
        ("FORMAT", output_format),
        ("TIME", timestamp),
    ]
    if include_resolution:
        params.extend(
            [
                ("RESOLUTION", "x(0.09)"),
                ("RESOLUTION", "y(0.09)"),
                ("INTERPOLATION", "NEAREST"),
            ]
        )
    return params


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
    bbox = parse_bbox(args.bbox)
    encoded_collection = quote(COLLECTION, safe="")
    collection_url = urljoin(OGC_API_ROOT, f"collections/{encoded_collection}")
    default_coverage_url = urljoin(
        OGC_API_ROOT, f"collections/{encoded_collection}/coverage"
    )
    output = {
        "generated_utc": generated.isoformat(),
        "status": "failed",
        "collection": COLLECTION,
        "collection_url": collection_url,
        "ogc_coverage_url": default_coverage_url,
        "wcs_url": WCS_URL,
        "wcs_layer": WCS_LAYER,
        "query": {"bbox": args.bbox, "datetime": args.datetime},
        "metadata_request": {},
        "metadata": {},
        "ogc_attempts": [],
        "wcs_attempts": [],
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
        metadata = {}
        if metadata_response.status_code == 200:
            try:
                metadata = metadata_response.json()
                links = metadata.get("links", [])
                coverage_links = [
                    {
                        "type": link.get("type"),
                        "href": link.get("href"),
                        "rel": link.get("rel"),
                        "title": link.get("title"),
                    }
                    for link in links
                    if "coverage" in str(link.get("href", ""))
                ]
                output["metadata"] = {
                    "title": metadata.get("title"),
                    "description": metadata.get("description"),
                    "extent": metadata.get("extent"),
                    "coverage_links": coverage_links,
                }
            except Exception as exc:
                output["metadata_parse_error"] = (
                    f"{exc.__class__.__name__}: {exc}"
                )
        else:
            output["metadata_request"]["response_text_prefix"] = (
                metadata_response.text[:2000]
            )

        # Try only formats actually advertised by the collection and use its
        # exact unescaped link. The field filter is omitted because this RDPA
        # collection has exhibited server-side 500s when properties=1 is used.
        advertised = output.get("metadata", {}).get("coverage_links", [])
        ogc_candidates = []
        for link in advertised:
            href = str(link.get("href", ""))
            media_type = str(link.get("type", ""))
            if "application/x-grib" in media_type:
                ogc_candidates.append(("OGC_API_GRIB", href, "GRIB"))
            elif "coverage+json" in media_type:
                ogc_candidates.append(("OGC_API_JSON", href, "json"))
        if not ogc_candidates:
            ogc_candidates = [
                ("OGC_API_GRIB", default_coverage_url, "GRIB"),
                ("OGC_API_JSON", default_coverage_url, "json"),
            ]

        for label, href, requested_format in ogc_candidates:
            try:
                response = http.get(
                    href,
                    params={
                        "f": requested_format,
                        "bbox": args.bbox,
                        "datetime": args.datetime,
                    },
                    timeout=300,
                )
                if requested_format == "json":
                    attempt = response_record(response, label)
                    if response.status_code == 200 and len(response.content) > 100:
                        try:
                            payload = response.json()
                            attempt["json_top_level_keys"] = sorted(payload.keys())
                            output["selected"] = {
                                "source": label,
                                "content_type": attempt["content_type"],
                                "bytes_received": len(response.content),
                                "coverage_json_top_level_keys": sorted(payload.keys()),
                            }
                        except Exception as exc:
                            attempt["json_parse_error"] = (
                                f"{exc.__class__.__name__}: {exc}"
                            )
                    output["ogc_attempts"].append(attempt)
                else:
                    selected = try_raster_response(
                        response, label, output["ogc_attempts"]
                    )
                    if selected:
                        output["selected"] = selected
                if output["selected"]:
                    break
            except Exception as exc:
                output["ogc_attempts"].append(
                    {
                        "source": label,
                        "request_error": f"{exc.__class__.__name__}: {exc}",
                    }
                )

        # Official WCS 2.0.1 fallback. Try native resolution first, then an
        # explicit 0.09-degree output grid, and finally NetCDF.
        if output["selected"] is None:
            wcs_requests = [
                ("WCS_TIFF_NATIVE", "image/tiff", False),
                ("WCS_TIFF_0.09DEG", "image/tiff", True),
                ("WCS_NETCDF_NATIVE", "image/netcdf", False),
            ]
            for label, output_format, include_resolution in wcs_requests:
                try:
                    response = http.get(
                        WCS_URL,
                        params=wcs_parameters(
                            bbox, args.datetime, output_format, include_resolution
                        ),
                        timeout=300,
                    )
                    selected = try_raster_response(
                        response, label, output["wcs_attempts"]
                    )
                    if selected:
                        output["selected"] = selected
                        break
                except Exception as exc:
                    output["wcs_attempts"].append(
                        {
                            "source": label,
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
                    "selected_source": (
                        output["selected"].get("source")
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
