#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from requests.adapters import HTTPAdapter
from shapely.geometry import mapping
from urllib3.util.retry import Retry

from historical_gauge_analysis import STATION, TARGET_Q
from historical_rdpa_pairing import (
    build_hourly,
    fit_and_validate,
    gauge_only_recession,
    rolling_rain_at_points,
    valid_times_for_candidates,
)
from rdpa_archive_probe import WCS_URL, wcs_parameters
from spatial_qpf import http as basin_http, load_basin_polygons, union_station

OUT_DEFAULT = Path("output/archive_probe/historical_rdpa_pairing.json")
PAIRS_DEFAULT = Path("output/archive_probe/historical_rdpa_pairs.csv")
CACHE_DEFAULT = Path("archive_cache/historical_rdpa_05EA002.csv")
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is not None:
        return value
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-historical-rdpa-wcs/1.0 "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    value.mount("https://", HTTPAdapter(max_retries=retry))
    THREAD_LOCAL.session = value
    return value


def floor_6h(timestamp: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.floor("6h")


def official_basin_4326():
    basins, metadata = load_basin_polygons(basin_http())
    geometry = union_station(basins, STATION)
    if geometry is None or geometry.is_empty:
        raise RuntimeError(f"Official basin polygon unavailable for {STATION}")
    geometry = gpd.GeoSeries([geometry], crs=basins.crs).to_crs("EPSG:4326").iloc[0]
    minx, miny, maxx, maxy = geometry.bounds
    pad = 0.05
    return geometry, (minx - pad, miny - pad, maxx + pad, maxy + pad), metadata


def fetch_wcs(valid: pd.Timestamp, geometry, bbox) -> dict:
    timestamp = floor_6h(valid)
    try:
        response = session().get(
            WCS_URL,
            params=wcs_parameters(
                bbox,
                timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "image/tiff",
                True,
            ),
            timeout=120,
        )
        response.raise_for_status()
        if len(response.content) <= 100:
            raise RuntimeError(f"WCS response too small: {len(response.content)} bytes")
        with MemoryFile(response.content) as memory:
            with memory.open() as dataset:
                clipped, _ = rio_mask(
                    dataset,
                    [mapping(geometry)],
                    crop=True,
                    filled=False,
                    indexes=1,
                    all_touched=False,
                )
                values = np.ma.asarray(clipped, dtype=float)
                values = np.ma.masked_where(
                    ~np.isfinite(values) | (values < -0.01) | (values > 1000),
                    values,
                )
                valid_values = values.compressed()
                if not len(valid_values):
                    raise RuntimeError("No valid RDPA cells intersect the official basin")
                return {
                    "valid_utc": timestamp.isoformat(),
                    "status": "retrieved",
                    "station": STATION,
                    "precip_mm": float(np.mean(valid_values)),
                    "source_type": "wcs_official_basin_clip",
                    "source_url": response.url,
                    "valid_cells": int(len(valid_values)),
                    "error": None,
                }
    except Exception as exc:
        return {
            "valid_utc": timestamp.isoformat(),
            "status": "missing",
            "station": STATION,
            "precip_mm": None,
            "source_type": "wcs_official_basin_clip",
            "source_url": None,
            "valid_cells": 0,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 20:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "valid_utc" not in frame.columns:
        return pd.DataFrame()
    frame["valid_utc"] = pd.to_datetime(frame.valid_utc, utc=True)
    return frame.sort_values("valid_utc").drop_duplicates("valid_utc", keep="last").set_index("valid_utc")


def save_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(path, index=False)


def retrieve_series(
    valid_times: list[pd.Timestamp], workers: int, cache_path: Path
) -> tuple[pd.DataFrame, dict]:
    requested = pd.DatetimeIndex([floor_6h(value) for value in valid_times]).drop_duplicates().sort_values()
    cached = load_cache(cache_path)
    cache_hits = 0
    if not cached.empty and "status" in cached.columns:
        cache_hits = int(cached.reindex(requested).status.eq("retrieved").sum())
    need = [
        value
        for value in requested
        if cached.empty
        or value not in cached.index
        or str(cached.loc[value].get("status")) != "retrieved"
    ]
    geometry, bbox, basin_metadata = official_basin_4326()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_wcs, value, geometry, bbox): value for value in need}
        for future in as_completed(futures):
            rows.append(future.result())
    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        fresh["valid_utc"] = pd.to_datetime(fresh.valid_utc, utc=True)
        fresh = fresh.sort_values("valid_utc").drop_duplicates("valid_utc", keep="last").set_index("valid_utc")
    combined = pd.concat([cached, fresh]) if not cached.empty else fresh
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    save_cache(cache_path, combined)
    selected = combined.reindex(requested)
    metadata = {
        "cache_path": str(cache_path),
        "cache_hits": cache_hits,
        "periods_fetched_this_run": int(len(need)),
        "bbox_epsg4326": [float(value) for value in bbox],
        "basin_source": basin_metadata,
        "method": "WCS 10 km RDPA clipped to the official WSC 05EA002 drainage-basin polygon",
    }
    return selected, metadata


def model_summary(frame: pd.DataFrame, current_q: float) -> dict:
    return {
        "gauge_only": fit_and_validate(frame, "gauge_only_with_rdpa_metrics", current_q),
        "rdpa_strict": fit_and_validate(
            frame[
                frame.complete_168h_coverage
                & (frame.rain_24h_mm <= 0.5)
                & (frame.rain_72h_mm <= 1.5)
                & (frame.rain_168h_mm <= 5.0)
            ],
            "rdpa_strict_dry_screen",
            current_q,
        ),
        "rdpa_moderate": fit_and_validate(
            frame[
                frame.complete_168h_coverage
                & (frame.rain_24h_mm <= 1.0)
                & (frame.rain_72h_mm <= 3.0)
                & (frame.rain_168h_mm <= 10.0)
            ],
            "rdpa_moderate_dry_screen",
            current_q,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lookback-hours", type=int, default=168)
    parser.add_argument("--cache", default=str(CACHE_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument("--pairs-output", default=str(PAIRS_DEFAULT))
    args = parser.parse_args()

    output_path = Path(args.output)
    pairs_path = Path(args.pairs_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    hourly, retrieval_records = build_hourly(args.months)
    gauge = gauge_only_recession(hourly)
    if gauge.empty:
        raise RuntimeError("No gauge-only recession points were identified")
    valid_times = valid_times_for_candidates(gauge.index, args.lookback_hours)
    rdpa, retrieval_method = retrieve_series(valid_times, args.workers, Path(args.cache))
    rain = rolling_rain_at_points(gauge.index, rdpa, (24, 72, 168))
    paired = gauge.join(rain)
    paired["complete_168h_coverage"] = paired.rain_168h_coverage >= 0.999

    current_q = float(hourly.discharge_m3s.iloc[-1])
    models = model_summary(paired, current_q)
    strict = models["rdpa_strict"]
    gauge_model = models["gauge_only"]
    strict_cv = strict.get("event_block_cross_validation", {}).get("aggregate")
    gauge_cv = gauge_model.get("event_block_cross_validation", {}).get("aggregate")

    requested = int(len(rdpa))
    retrieved = int(rdpa.status.eq("retrieved").sum())
    coverage = retrieved / requested if requested else 0.0
    reasons: list[str] = []
    if coverage < 0.90:
        reasons.append("archived_rdpa_coverage_below_90_percent")
    if strict.get("events", 0) < 3:
        reasons.append("fewer_than_three_strict_dry_recession_events")
    if strict.get("points", 0) < 200:
        reasons.append("fewer_than_200_strict_dry_recession_points")
    if strict_cv is None:
        reasons.append("strict_event_block_cross_validation_unavailable")
    if strict_cv and gauge_cv and strict_cv["rmse_per_day"] > gauge_cv["rmse_per_day"]:
        reasons.append("strict_rdpa_screen_does_not_improve_event_block_rmse")

    paired.to_csv(pairs_path, index_label="date_utc")
    generated = datetime.now(timezone.utc).replace(microsecond=0)
    output = {
        "generated_utc": generated.isoformat(),
        "status": "historical_rdpa_pairing_complete",
        "mode": "shadow_only_no_automatic_promotion",
        "station": STATION,
        "target_discharge_m3s": TARGET_Q,
        "requested_months": args.months,
        "wateroffice_retrieval": {
            "chunks": retrieval_records,
            "hourly_points": int(len(hourly)),
            "first_utc": hourly.index.min().isoformat(),
            "last_utc": hourly.index.max().isoformat(),
        },
        "rdpa_retrieval": {
            "product": "RDPA 6-hour 10 km WCS",
            "requested_periods": requested,
            "retrieved_periods": retrieved,
            "missing_periods": requested - retrieved,
            "coverage_fraction": coverage,
            "first_valid_utc": rdpa.index.min().isoformat() if requested else None,
            "last_valid_utc": rdpa.index.max().isoformat() if requested else None,
            "retrieval_method": retrieval_method,
            "missing_examples": rdpa[rdpa.status != "retrieved"].head(20).reset_index().to_dict("records"),
        },
        "pairing": {
            "gauge_only_candidate_points": int(len(paired)),
            "complete_168h_coverage_points": int(paired.complete_168h_coverage.sum()),
            "strict_dry_points": int(models["rdpa_strict"].get("points", 0)),
            "moderate_dry_points": int(models["rdpa_moderate"].get("points", 0)),
            "strict_thresholds_mm": {"24h": 0.5, "72h": 1.5, "168h": 5.0},
            "moderate_thresholds_mm": {"24h": 1.0, "72h": 3.0, "168h": 10.0},
            "pairs_csv": str(pairs_path),
        },
        "models": models,
        "promotion_screen": {
            "automatic_promotion_enabled": False,
            "candidate_passes_minimum_screen": not reasons,
            "reasons_not_to_promote": reasons,
            "requirements": [
                "at least 90 percent archived RDPA coverage",
                "at least three distinct strict dry recession events",
                "at least 200 strict dry recession points",
                "event-block cross-validation available",
                "strict dry screening does not worsen event-block RMSE",
                "manual engineering review before operational promotion",
            ],
        },
        "interpretation": (
            "The historical direct-discharge recession has been paired with basin-clipped "
            "archived RDPA. Strict and moderate dry-screen fits remain independent shadow "
            "sensitivities until the promotion screen and engineering review are satisfied."
        ),
        "limitations": [
            "The 10 km RDPA grid may miss sub-grid convective detail.",
            "WSC discharge remains provisional and rating-derived from stage.",
            "Hourly points are autocorrelated, so event-block validation is emphasized.",
            "A seven-day rainfall screen may not fully represent longer lake and wetland storage memory.",
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(
        json.dumps(
            {
                "status": output["status"],
                "coverage_fraction": coverage,
                "strict_points": strict.get("points"),
                "strict_events": strict.get("events"),
                "strict_projection_days": (strict.get("projection") or {}).get("days"),
                "moderate_projection_days": (models["rdpa_moderate"].get("projection") or {}).get("days"),
                "promotion_screen": output["promotion_screen"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
