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

from historical_gauge_analysis import fit_recession, predict_rate, rate_metrics
from historical_rdpa_pairing import build_hourly, floor_6h
from historical_rdpa_wcs_pairing import retrieve_series
from rdpa_archive_probe import WCS_URL, wcs_parameters
from spatial_qpf import build_subareas, http as basin_http, load_basin_polygons

PAIRING_DEFAULT = Path("output/archive_probe/historical_rdpa_pairing.json")
PAIRS_DEFAULT = Path("output/archive_probe/historical_rdpa_pairs.csv")
BASIN_CACHE_DEFAULT = Path("archive_cache/historical_rdpa_05EA002.csv")
GRID_CACHE_DEFAULT = Path("archive_cache/historical_rdpa_spatial_grids")
OUT_DEFAULT = Path("output/historical_event_backfill/historical_spatial_event_backfill.json")
EVENTS_DEFAULT = Path("output/historical_event_backfill/historical_spatial_events.csv")
MODEL_DEFAULT = Path("output/historical_event_backfill/historical_spatial_response_model.json")

TARGET = "05EA002"
FEATURES = [
    "basin_mm",
    "lower_ratio",
    "upper_ratio",
    "duration_h",
    "antecedent_168h_mm",
    "basin_pct_gt_10mm",
    "pre_stage_m",
]
SPATIAL_MAP = {
    "basin": "basin_to_05EA002",
    "lower": "lower_incremental_05EA005_to_05EA002",
    "upper": "upper_lake_chain_isle_lac_ste_anne",
    "middle": "lac_ste_anne_to_villeneuve_mainstem",
    "atim": "atim_creek_big_lake_tributary",
    "carrot": "carrot_creek",
    "local": "direct_big_lake_and_local_to_05EA002",
}
THREAD_LOCAL = threading.local()


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def safe_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is not None:
        return value
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-historical-spatial-event-backfill/1.0 "
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


def continuous_6h(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = floor_6h(start)
    last = floor_6h(end)
    return list(pd.date_range(first, last, freq="6h", tz="UTC"))


def detect_rain_events(
    rdpa: pd.DataFrame,
    minimum_6h_mm: float = 0.25,
    minimum_event_mm: float = 1.5,
    join_gap_hours: int = 18,
) -> list[dict]:
    work = rdpa.copy()
    work["precip_mm"] = pd.to_numeric(work.get("precip_mm"), errors="coerce")
    work = work[work.get("status").eq("retrieved") & work.precip_mm.notna()].sort_index()
    wet = work[work.precip_mm >= minimum_6h_mm]
    if wet.empty:
        return []
    groups = (
        wet.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(3600)
        .fillna(join_gap_hours + 1)
        .gt(join_gap_hours)
        .cumsum()
    )
    events: list[dict] = []
    for _, group in wet.groupby(groups):
        valid_start = group.index.min()
        valid_end = group.index.max()
        periods = work.loc[(work.index >= valid_start) & (work.index <= valid_end)]
        total = float(periods.precip_mm.sum())
        peak = float(periods.precip_mm.max())
        if total < minimum_event_mm and peak < 1.0:
            continue
        events.append(
            {
                "rain_valid_start_utc": valid_start,
                "rain_start_utc": valid_start - pd.Timedelta(hours=6),
                "rain_end_utc": valid_end,
                "rain_valid_end_utc": valid_end,
                "basin_total_preliminary_mm": total,
                "basin_peak_6h_preliminary_mm": peak,
                "period_count": int(len(periods)),
            }
        )
    return events


def grid_path(cache_dir: Path, valid: pd.Timestamp) -> Path:
    return cache_dir / f"rdpa_{floor_6h(valid):%Y%m%d%H}.tif"


def basin_bbox_and_subareas():
    basins, metadata = load_basin_polygons(basin_http())
    subareas, subarea_metadata = build_subareas(basins)
    subareas = subareas.to_crs("EPSG:4326")
    basin_row = subareas[subareas.subarea == "basin_to_05EA002"]
    if basin_row.empty:
        raise RuntimeError("Derived 05EA002 basin subarea is unavailable")
    minx, miny, maxx, maxy = basin_row.iloc[0].geometry.bounds
    pad = 0.05
    return subareas, (minx - pad, miny - pad, maxx + pad, maxy + pad), {
        "official_basins": metadata,
        "derived_subareas": subarea_metadata,
    }


def fetch_grid(valid: pd.Timestamp, bbox, cache_dir: Path) -> dict:
    timestamp = floor_6h(valid)
    path = grid_path(cache_dir, timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100:
        return {
            "valid_utc": timestamp.isoformat(),
            "status": "cache_hit",
            "path": str(path),
            "bytes": path.stat().st_size,
            "error": None,
        }
    try:
        response = session().get(
            WCS_URL,
            params=wcs_parameters(
                bbox,
                timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "image/tiff",
                True,
            ),
            timeout=180,
        )
        response.raise_for_status()
        if len(response.content) <= 100:
            raise RuntimeError(f"WCS response too small: {len(response.content)} bytes")
        with MemoryFile(response.content) as memory:
            with memory.open() as dataset:
                if dataset.count < 1:
                    raise RuntimeError("WCS raster contains no bands")
                _ = dataset.read(1, masked=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(response.content)
        tmp.replace(path)
        return {
            "valid_utc": timestamp.isoformat(),
            "status": "retrieved",
            "path": str(path),
            "bytes": path.stat().st_size,
            "error": None,
        }
    except Exception as exc:
        return {
            "valid_utc": timestamp.isoformat(),
            "status": "missing",
            "path": None,
            "bytes": 0,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def ensure_grids(
    times: list[pd.Timestamp], bbox, cache_dir: Path, workers: int
) -> tuple[dict[pd.Timestamp, Path], dict]:
    requested = (
        pd.DatetimeIndex([floor_6h(value) for value in times])
        .drop_duplicates()
        .sort_values()
    )
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_grid, value, bbox, cache_dir): value
            for value in requested
        }
        for future in as_completed(futures):
            rows.append(future.result())
    paths: dict[pd.Timestamp, Path] = {}
    for row in rows:
        timestamp = pd.Timestamp(row["valid_utc"])
        if row["status"] in {"retrieved", "cache_hit"} and row["path"]:
            paths[timestamp] = Path(row["path"])
    return paths, {
        "requested_periods": int(len(requested)),
        "available_periods": int(len(paths)),
        "coverage_fraction": float(len(paths) / len(requested)) if len(requested) else 0.0,
        "retrieved_this_run": int(sum(row["status"] == "retrieved" for row in rows)),
        "cache_hits": int(sum(row["status"] == "cache_hit" for row in rows)),
        "missing_examples": [row for row in rows if row["status"] == "missing"][:20],
        "cache_dir": str(cache_dir),
        "method": "WCS 10 km RDPA rasters clipped to official WSC-derived subareas",
    }


def accumulated_subarea_features(
    paths: list[Path], subareas: gpd.GeoDataFrame
) -> tuple[dict, float]:
    totals: dict[str, np.ma.MaskedArray] = {}
    successful = 0
    for path in paths:
        try:
            with MemoryFile(path.read_bytes()) as memory:
                with memory.open() as dataset:
                    for _, area in subareas.iterrows():
                        geometry = (
                            gpd.GeoSeries([area.geometry], crs=subareas.crs)
                            .to_crs(dataset.crs)
                            .iloc[0]
                        )
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
                        name = str(area.subarea)
                        if name not in totals:
                            totals[name] = values.copy()
                        elif totals[name].shape == values.shape:
                            totals[name] = np.ma.add(totals[name], values)
                        else:
                            raise RuntimeError(f"Grid shape changed for {name}")
            successful += 1
        except Exception:
            continue
    features: dict[str, dict] = {}
    for name, values in totals.items():
        valid = values.compressed()
        record = {
            "mean_mm": float(np.mean(valid)) if len(valid) else None,
            "max_mm": float(np.max(valid)) if len(valid) else None,
            "valid_cells": int(len(valid)),
        }
        for threshold in (5, 10, 20, 30, 50):
            record[f"pct_gt_{threshold}mm"] = (
                float(np.mean(valid > threshold) * 100) if len(valid) else None
            )
        features[name] = record
    coverage = successful / len(paths) if paths else 0.0
    return features, coverage


def dynamic_baseline(
    value0: float, index: pd.DatetimeIndex, fit: dict
) -> pd.Series:
    values: list[float] = []
    value = float(value0)
    previous = None
    for timestamp in index:
        if previous is not None:
            hours = max(0.0, (timestamp - previous).total_seconds() / 3600.0)
            rate = min(
                -0.001,
                float(fit["intercept_per_day"])
                + float(fit["coefficient_per_day"]) * value,
            )
            value += rate * hours / 24.0
        values.append(value)
        previous = timestamp
    return pd.Series(values, index=index, dtype=float)


def first_sustained(condition: pd.Series, hours: int):
    sustained = condition.astype(int).rolling(hours, min_periods=hours).sum() >= hours
    hits = sustained[sustained].index
    return hits[0] if len(hits) else None


def historical_recession_fits(pairs_path: Path) -> tuple[dict, dict, dict]:
    pairs = pd.read_csv(pairs_path)
    date_column = "date_utc" if "date_utc" in pairs.columns else pairs.columns[0]
    pairs[date_column] = pd.to_datetime(pairs[date_column], utc=True)
    pairs = pairs.set_index(date_column).sort_index()
    coverage_flag = (
        pairs["complete_168h_coverage"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
        if "complete_168h_coverage" in pairs.columns
        else pd.Series(False, index=pairs.index)
    )
    moderate = pairs[
        coverage_flag
        & (pd.to_numeric(pairs.get("rain_24h_mm"), errors="coerce") <= 1.0)
        & (pd.to_numeric(pairs.get("rain_72h_mm"), errors="coerce") <= 3.0)
        & (pd.to_numeric(pairs.get("rain_168h_mm"), errors="coerce") <= 10.0)
    ].copy()
    q_fit = fit_recession(moderate, "discharge_m3s", "q_rate_m3s_per_day")
    stage_fit = fit_recession(moderate, "stage_m", "stage_rate_m_per_day")
    if q_fit is None or stage_fit is None:
        raise RuntimeError(
            "Unable to fit historical moderate-dry stage and discharge recessions"
        )
    return q_fit, stage_fit, {
        "moderate_dry_points": int(len(moderate)),
        "first_utc": moderate.index.min().isoformat(),
        "last_utc": moderate.index.max().isoformat(),
    }


def classify_event(record: dict) -> str:
    basin = finite(record.get("basin_mm"), 0.0)
    lower = finite(record.get("lower_mean_mm"), 0.0)
    upper = finite(record.get("upper_mean_mm"), 0.0)
    middle = finite(record.get("middle_mean_mm"), 0.0)
    atim = finite(record.get("atim_mean_mm"), 0.0)
    carrot = finite(record.get("carrot_mean_mm"), 0.0)
    local = finite(record.get("local_mean_mm"), 0.0)
    pct10 = finite(record.get("basin_pct_gt_10mm"), 0.0)
    pct5 = finite(record.get("basin_pct_gt_5mm"), 0.0)
    if pct10 >= 70 or pct5 >= 90:
        return "widespread_basin"
    if basin > 0 and lower >= 1.35 * basin and upper <= 0.85 * basin:
        return "lower_basin_concentrated"
    if lower > 0 and upper >= 1.35 * lower:
        return "upper_lake_chain_concentrated"
    if max(atim, carrot, local) >= max(2.0, 1.4 * basin):
        dominant = max(
            [
                (atim, "atim_big_lake"),
                (carrot, "carrot_creek"),
                (local, "direct_local"),
            ]
        )[1]
        return f"tributary_localized_{dominant}"
    if middle >= max(2.0, 1.25 * upper, 1.25 * lower):
        return "middle_mainstem_concentrated"
    return "mixed_or_weak"


def nearest(series: pd.Series, timestamp: pd.Timestamp):
    if series.empty:
        return None
    location = series.index.get_indexer([timestamp], method="nearest")[0]
    if location < 0:
        return None
    return finite(series.iloc[location])


def event_record(
    event_id: int,
    event: dict,
    next_start: pd.Timestamp | None,
    hourly: pd.DataFrame,
    rdpa: pd.DataFrame,
    q_fit: dict,
    stage_fit: dict,
    paths: dict[pd.Timestamp, Path],
    subareas: gpd.GeoDataFrame,
) -> dict | None:
    start = pd.Timestamp(event["rain_start_utc"])
    end = pd.Timestamp(event["rain_end_utc"])
    candidate_end = end + pd.Timedelta(days=14)
    dataset_end = hourly.index.max()
    analysis_end = min(candidate_end, dataset_end)
    truncation_reason = "analysis_window_limit"
    if next_start is not None and next_start < analysis_end:
        analysis_end = next_start
        truncation_reason = "next_rain_event"
    elif dataset_end < candidate_end:
        truncation_reason = "dataset_right_edge"

    pre_candidates = hourly.index[hourly.index <= start]
    if not len(pre_candidates):
        return None
    t0 = pre_candidates[-1]
    observed = hourly.loc[t0:analysis_end].dropna(
        subset=["stage_m", "discharge_m3s"]
    )
    if len(observed) < 12:
        return None

    q_base = dynamic_baseline(
        float(observed.discharge_m3s.iloc[0]), observed.index, q_fit
    )
    stage_base = dynamic_baseline(
        float(observed.stage_m.iloc[0]), observed.index, stage_fit
    )
    q_departure = observed.discharge_m3s - q_base
    stage_departure = observed.stage_m - stage_base
    post = observed.index >= start
    q_post = q_departure.loc[post]
    stage_post = stage_departure.loc[post]
    if q_post.empty:
        return None

    peak_time = q_post.idxmax()
    peak_q_departure = float(q_post.max())
    peak_stage_departure = (
        float(stage_post.loc[:peak_time].max())
        if len(stage_post.loc[:peak_time])
        else None
    )
    onset_q = first_sustained(q_post >= 0.20, 3)
    onset_stage = first_sustained(stage_post >= 0.01, 3)
    onset_candidates = [
        value for value in (onset_q, onset_stage) if value is not None
    ]
    onset = min(onset_candidates) if onset_candidates else None

    baseline_q_at_peak = float(q_base.loc[peak_time])
    rate_q_at_peak = abs(
        min(
            -0.001,
            float(q_fit["intercept_per_day"])
            + float(q_fit["coefficient_per_day"]) * baseline_q_at_peak,
        )
    )
    days_lost = (
        peak_q_departure / rate_q_at_peak if rate_q_at_peak > 0.001 else None
    )

    after_peak = observed.index >= peak_time
    recovery_condition = (
        q_departure.loc[after_peak]
        <= max(0.15, 0.15 * max(0.0, peak_q_departure))
    ) & (
        stage_departure.loc[after_peak]
        <= max(0.015, 0.15 * max(0.0, peak_stage_departure or 0.0))
    )
    recovery = first_sustained(recovery_condition, 12)
    peak_near_end = (analysis_end - peak_time).total_seconds() <= 12 * 3600
    censored = bool(
        truncation_reason
        and (
            recovery is None
            or peak_near_end
            or float(q_post.iloc[-1])
            > max(0.15, 0.15 * max(0.0, peak_q_departure))
        )
    )

    valid_times = continuous_6h(
        pd.Timestamp(event["rain_valid_start_utc"]),
        pd.Timestamp(event["rain_valid_end_utc"]),
    )
    event_paths = [paths[value] for value in valid_times if value in paths]
    spatial, processing_coverage = accumulated_subarea_features(
        event_paths, subareas
    )
    retrieval_coverage = len(event_paths) / len(valid_times) if valid_times else 0.0
    spatial_coverage = processing_coverage * retrieval_coverage

    antecedent = {}
    for hours in (24, 72, 168):
        end_antecedent = floor_6h(start)
        start_antecedent = end_antecedent - pd.Timedelta(hours=hours - 6)
        antecedent[hours] = float(
            pd.to_numeric(
                rdpa.loc[
                    (rdpa.index >= start_antecedent)
                    & (rdpa.index <= end_antecedent),
                    "precip_mm",
                ],
                errors="coerce",
            ).sum()
        )

    record: dict = {
        "event_id": int(event_id),
        "rain_start_utc": start.isoformat(),
        "rain_end_utc": end.isoformat(),
        "rain_duration_h": float((end - start).total_seconds() / 3600.0),
        "analysis_end_utc": analysis_end.isoformat(),
        "truncation_reason": truncation_reason,
        "response_censored": censored,
        "pre_stage_m": nearest(hourly.stage_m, start),
        "pre_discharge_m3s": nearest(hourly.discharge_m3s, start),
        "antecedent_24h_mm": antecedent[24],
        "antecedent_72h_mm": antecedent[72],
        "antecedent_168h_mm": antecedent[168],
        "response_onset_utc": onset.isoformat() if onset is not None else None,
        "lag_to_onset_h": (
            float((onset - start).total_seconds() / 3600.0)
            if onset is not None
            else None
        ),
        "departure_peak_utc": peak_time.isoformat(),
        "q_departure_peak_m3s": peak_q_departure,
        "stage_departure_peak_m": peak_stage_departure,
        "estimated_recession_days_lost": (
            float(days_lost) if days_lost is not None else None
        ),
        "recovery_utc": recovery.isoformat() if recovery is not None else None,
        "recovery_duration_h": (
            float((recovery - peak_time).total_seconds() / 3600.0)
            if recovery is not None
            else None
        ),
        "spatial_periods_requested": int(len(valid_times)),
        "spatial_periods_available": int(len(event_paths)),
        "spatial_coverage_fraction": spatial_coverage,
    }
    for short, subarea in SPATIAL_MAP.items():
        values = spatial.get(subarea, {})
        record[f"{short}_mean_mm"] = finite(values.get("mean_mm"))
        record[f"{short}_max_mm"] = finite(values.get("max_mm"))
        for threshold in (5, 10, 20, 30, 50):
            record[f"{short}_pct_gt_{threshold}mm"] = finite(
                values.get(f"pct_gt_{threshold}mm")
            )
    record["basin_mm"] = record.get("basin_mean_mm")
    basin = finite(record.get("basin_mm"), 0.0)
    record["lower_ratio"] = (
        finite(record.get("lower_mean_mm"), 0.0) / basin if basin > 0 else 0.0
    )
    record["upper_ratio"] = (
        finite(record.get("upper_mean_mm"), 0.0) / basin if basin > 0 else 0.0
    )
    record["duration_h"] = record["rain_duration_h"]
    record["basin_pct_gt_10mm"] = record.get("basin_pct_gt_10mm")
    record["storm_type"] = classify_event(record)
    record["isolated_event"] = bool(
        antecedent[24] <= 0.5 and antecedent[72] <= 3.0
    )

    response_quality = "complete"
    if spatial_coverage < 0.80:
        response_quality = "insufficient_spatial_coverage"
    elif not record["isolated_event"]:
        response_quality = "antecedent_rain_overlap"
    elif censored:
        response_quality = "censored_lower_bound"
    elif peak_q_departure < 0.20 or days_lost is None or days_lost <= 0:
        response_quality = "weak_or_no_detectable_response"
    elif recovery is None:
        response_quality = "peak_observed_recovery_not_observed"
    record["response_quality"] = response_quality
    record["eligible_for_point_training"] = bool(
        not censored
        and spatial_coverage >= 0.80
        and peak_q_departure >= 0.20
        and days_lost is not None
        and 0 < days_lost <= 30
        and record["isolated_event"]
    )
    record["eligible_for_recovery_training"] = bool(
        record["eligible_for_point_training"] and recovery is not None
    )
    return record


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    penalty: float = 1.0,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-9] = 1.0
    x = (train_x - mean) / scale
    test = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    test_design = np.column_stack([np.ones(len(test)), test])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ train_y
    )
    return test_design @ coefficients


def ridge_model_parameters(
    train_x: np.ndarray,
    train_y: np.ndarray,
    feature_names: list[str],
    penalty: float = 1.0,
) -> dict:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-9] = 1.0
    x = (train_x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ train_y
    )
    return {
        "type": "standardized_ridge",
        "penalty": penalty,
        "features": feature_names,
        "feature_means": {
            name: float(value) for name, value in zip(feature_names, mean)
        },
        "feature_scales": {
            name: float(value) for name, value in zip(feature_names, scale)
        },
        "intercept": float(coefficients[0]),
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(feature_names, coefficients[1:])
        },
    }


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-9] = 1.0
    x = (train_x - mean) / scale
    test = (test_x - mean) / scale
    predictions = []
    for row in test:
        distance = np.sqrt(np.sum((x - row) ** 2, axis=1))
        order = np.argsort(distance)[: max(1, min(k, len(distance)))]
        weights = 1.0 / np.maximum(distance[order], 0.25)
        predictions.append(float(np.average(train_y[order], weights=weights)))
    return np.asarray(predictions)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_days": float(np.sqrt(np.mean(error**2))),
        "mae_days": float(np.mean(np.abs(error))),
        "bias_days": float(np.mean(error)),
    }


def validate_models(frame: pd.DataFrame) -> dict:
    eligible = frame[frame.eligible_for_point_training.astype(bool)].copy()
    eligible = eligible.dropna(subset=FEATURES + ["estimated_recession_days_lost"])
    result = {
        "eligible_events": int(len(eligible)),
        "storm_type_count": (
            int(eligible.storm_type.nunique()) if len(eligible) else 0
        ),
        "models": {},
        "preferred_candidate": None,
        "promotion_screen": {},
    }
    if len(eligible) < 4:
        result["promotion_screen"] = {
            "candidate_passes_minimum_screen": False,
            "reasons_not_to_promote": [
                "fewer_than_four_uncensored_spatial_events"
            ],
        }
        return result

    x_all = eligible[FEATURES].to_numpy(float)
    amount_features = ["basin_mm", "antecedent_168h_mm", "pre_stage_m"]
    x_amount = eligible[amount_features].to_numpy(float)
    y = eligible.estimated_recession_days_lost.to_numpy(float)
    model_predictions = {
        "median_baseline": [],
        "amount_only_ridge": [],
        "spatial_ridge": [],
        "spatial_knn": [],
    }
    fold_rows = []
    for i in range(len(eligible)):
        keep = np.arange(len(eligible)) != i
        model_predictions["median_baseline"].append(float(np.median(y[keep])))
        model_predictions["amount_only_ridge"].append(
            float(ridge_predict(x_amount[keep], y[keep], x_amount[[i]], 1.0)[0])
        )
        model_predictions["spatial_ridge"].append(
            float(ridge_predict(x_all[keep], y[keep], x_all[[i]], 1.0)[0])
        )
        model_predictions["spatial_knn"].append(
            float(knn_predict(x_all[keep], y[keep], x_all[[i]], 3)[0])
        )
        fold_rows.append(
            {
                "held_out_event_id": int(eligible.iloc[i].event_id),
                "observed_days_lost": float(y[i]),
            }
        )
    for name, values in model_predictions.items():
        prediction = np.asarray(values)
        score = metrics(y, prediction)
        score["predictions"] = [
            {
                **fold_rows[i],
                "predicted_days_lost": float(prediction[i]),
            }
            for i in range(len(prediction))
        ]
        result["models"][name] = score

    preferred_name = min(
        ("amount_only_ridge", "spatial_ridge", "spatial_knn"),
        key=lambda name: result["models"][name]["rmse_days"],
    )
    result["preferred_candidate"] = preferred_name
    if preferred_name == "spatial_ridge":
        result["fitted_shadow_model"] = ridge_model_parameters(
            x_all, y, FEATURES, 1.0
        )
    elif preferred_name == "amount_only_ridge":
        result["fitted_shadow_model"] = ridge_model_parameters(
            x_amount, y, amount_features, 1.0
        )
    else:
        result["fitted_shadow_model"] = {
            "type": "standardized_knn",
            "k": 3,
            "features": FEATURES,
            "training_events": [
                {
                    "event_id": int(row.event_id),
                    "storm_type": str(row.storm_type),
                    "target_days_lost": float(
                        row.estimated_recession_days_lost
                    ),
                    "features": {
                        feature: float(getattr(row, feature))
                        for feature in FEATURES
                    },
                }
                for row in eligible.itertuples(index=False)
            ],
        }
    median_rmse = result["models"]["median_baseline"]["rmse_days"]
    amount_rmse = result["models"]["amount_only_ridge"]["rmse_days"]
    preferred_rmse = result["models"][preferred_name]["rmse_days"]
    gain_vs_median = (
        (median_rmse - preferred_rmse) / median_rmse * 100.0
        if median_rmse > 0
        else None
    )
    gain_vs_amount = (
        (amount_rmse - preferred_rmse) / amount_rmse * 100.0
        if amount_rmse > 0
        else None
    )
    reasons = []
    if len(eligible) < 10:
        reasons.append("fewer_than_ten_uncensored_spatial_events")
    if eligible.storm_type.nunique() < 3:
        reasons.append("fewer_than_three_observed_storm_types")
    if gain_vs_median is None or gain_vs_median < 15.0:
        reasons.append(
            "preferred_model_improves_median_baseline_by_less_than_15_percent"
        )
    if preferred_name.startswith("spatial") and (
        gain_vs_amount is None or gain_vs_amount < 10.0
    ):
        reasons.append(
            "spatial_features_improve_amount_only_model_by_less_than_10_percent"
        )
    if not preferred_name.startswith("spatial"):
        reasons.append("spatial_model_is_not_cross_validated_best")
    result["promotion_screen"] = {
        "candidate_passes_minimum_screen": not reasons,
        "automatic_promotion_enabled": False,
        "reasons_not_to_promote": reasons,
        "skill_improvement_vs_median_pct": gain_vs_median,
        "skill_improvement_vs_amount_only_pct": gain_vs_amount,
        "requirements": [
            "at least ten uncensored events with at least 80 percent spatial coverage",
            "at least three observed storm-pattern classes",
            "leave-one-event-out validation available",
            "preferred model improves median baseline by at least 15 percent",
            "spatial model improves amount-only model by at least 10 percent",
            "manual engineering review and operational hindcast before promotion",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pairing", default=str(PAIRING_DEFAULT))
    parser.add_argument("--pairs", default=str(PAIRS_DEFAULT))
    parser.add_argument("--basin-cache", default=str(BASIN_CACHE_DEFAULT))
    parser.add_argument("--grid-cache", default=str(GRID_CACHE_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument("--events-output", default=str(EVENTS_DEFAULT))
    parser.add_argument("--model-output", default=str(MODEL_DEFAULT))
    args = parser.parse_args()

    output_path = Path(args.output)
    events_path = Path(args.events_output)
    model_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    hourly, wateroffice_records = build_hourly(args.months)
    valid_times = continuous_6h(hourly.index.min(), hourly.index.max())
    rdpa, basin_retrieval = retrieve_series(
        valid_times, args.workers, Path(args.basin_cache)
    )
    coverage = (
        float(rdpa.status.eq("retrieved").mean()) if len(rdpa) else 0.0
    )
    if coverage < 0.90:
        raise RuntimeError(
            f"Continuous archived RDPA coverage is only {coverage:.1%}"
        )

    detected = detect_rain_events(rdpa)
    if not detected:
        raise RuntimeError("No historical rainfall events were detected")

    subareas, bbox, spatial_metadata = basin_bbox_and_subareas()
    all_event_times = []
    for event in detected:
        all_event_times.extend(
            continuous_6h(
                pd.Timestamp(event["rain_valid_start_utc"]),
                pd.Timestamp(event["rain_valid_end_utc"]),
            )
        )
    paths, grid_retrieval = ensure_grids(
        all_event_times, bbox, Path(args.grid_cache), args.workers
    )

    q_fit, stage_fit, fit_support = historical_recession_fits(
        Path(args.pairs)
    )
    records = []
    for position, event in enumerate(detected):
        next_start = (
            pd.Timestamp(detected[position + 1]["rain_start_utc"])
            if position + 1 < len(detected)
            else None
        )
        record = event_record(
            position + 1,
            event,
            next_start,
            hourly,
            rdpa,
            q_fit,
            stage_fit,
            paths,
            subareas,
        )
        if record is not None:
            records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(
            "Historical event-response extraction produced no records"
        )
    frame.to_csv(events_path, index=False)

    validation = validate_models(frame)
    model_payload = {
        "generated_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "historical_spatial_response_shadow_model_evaluated",
        "mode": "shadow_only_manual_promotion_required",
        "feature_names": FEATURES,
        "target": "estimated_recession_days_lost",
        "validation": validation,
        "limitations": [
            "Historical WCS RDPA is 10 km, while live HRDPS and recent-event HRDPA features are 2.5 km.",
            "WaterOffice discharge is rating-derived and provisional.",
            "Event response is measured relative to a precipitation-screened empirical recession model, not a routed physical watershed model.",
            "Overlapping storms are censored and are not used as point-training events.",
            "No automatic promotion is permitted.",
        ],
    }
    model_path.write_text(
        json.dumps(model_payload, indent=2, default=safe_json)
    )

    result = {
        "generated_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "historical_spatial_event_backfill_complete",
        "mode": "shadow_only_no_automatic_promotion",
        "requested_months": args.months,
        "station": TARGET,
        "wateroffice_retrieval": {
            "chunks": wateroffice_records,
            "hourly_points": int(len(hourly)),
            "first_utc": hourly.index.min().isoformat(),
            "last_utc": hourly.index.max().isoformat(),
        },
        "continuous_rdpa_retrieval": {
            "requested_periods": int(len(rdpa)),
            "retrieved_periods": int(rdpa.status.eq("retrieved").sum()),
            "coverage_fraction": coverage,
            "retrieval_method": basin_retrieval,
        },
        "spatial_grid_retrieval": grid_retrieval,
        "spatial_metadata": spatial_metadata,
        "event_detection": {
            "minimum_6h_basin_mm": 0.25,
            "minimum_event_basin_mm": 1.5,
            "join_gap_hours": 18,
            "detected_events": int(len(detected)),
            "extracted_response_events": int(len(frame)),
            "uncensored_point_training_events": int(
                frame.eligible_for_point_training.sum()
            ),
            "complete_recovery_events": int(
                frame.eligible_for_recovery_training.sum()
            ),
            "storm_type_counts_all": frame.storm_type.value_counts().to_dict(),
            "storm_type_counts_training": frame.loc[
                frame.eligible_for_point_training.astype(bool), "storm_type"
            ]
            .value_counts()
            .to_dict(),
        },
        "recession_baseline": {
            "source": "moderate precipitation-screened historical recession points",
            "support": fit_support,
            "discharge_fit": q_fit,
            "stage_fit": stage_fit,
        },
        "model_evaluation": validation,
        "outputs": {
            "events_csv": str(events_path),
            "model_json": str(model_path),
        },
        "interpretation": (
            "The 18-month archive has been mined for rainfall events, each event has been "
            "summarized over official WSC-derived subareas, and the observed 05EA002 response "
            "has been measured against the precipitation-screened historical recession curve. "
            "All models remain shadow-only until event-level validation demonstrates material skill."
        ),
        "limitations": model_payload["limitations"],
    }
    output_path.write_text(json.dumps(result, indent=2, default=safe_json))
    print(
        json.dumps(
            {
                "status": result["status"],
                "events": result["event_detection"],
                "preferred_model": validation.get("preferred_candidate"),
                "promotion_screen": validation.get("promotion_screen"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
