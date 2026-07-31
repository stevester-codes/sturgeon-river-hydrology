#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

from qpf_forecast_v2 import download, get_links, http, parse_run_time, summarize

ROOT = Path("sturgeon_pipeline_output")
SPATIAL = ROOT / "spatial"
SUBAREAS = SPATIAL / "derived_subareas.geojson"
CACHE = Path("grid_cache") / "medium_range"

GDPS_BASE = "https://dd.weather.gc.ca/today/model_gdps/15km"
GEPS_BASE = "https://dd.weather.gc.ca/today/ensemble/geps/grib2/raw"
GDPS_HOURS = list(range(6, 241, 6))
GDPS_SUMMARY_HORIZONS = {72, 120, 168, 240}
GEPS_HORIZONS = [24, 48, 72, 96, 120, 144, 168, 192, 240, 288, 336, 384]
GEPS_RUN_RE = re.compile(r"_(\d{10})_P\d{3}_allmbrs\.grib2$", re.I)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gdps_candidate(links: Iterable[str]) -> str | None:
    candidates: list[str] = []
    for name in links:
        lower = name.lower()
        if not lower.endswith(".grib2"):
            continue
        if "_msc_gdps_apcp-accum6h_" not in lower:
            continue
        if any(token in lower for token in ["-prob_", "snow", "freezing", "convective"]):
            continue
        candidates.append(name)
    return sorted(candidates, key=len)[0] if candidates else None


def latest_gdps_run(session) -> tuple[str, datetime, str] | None:
    found: list[tuple[datetime, str, str]] = []
    for hour in ["00", "12"]:
        directory = f"{GDPS_BASE}/{hour}/006/"
        try:
            candidate = gdps_candidate(get_links(session, directory))
        except Exception:
            continue
        if candidate:
            run_time = parse_run_time(candidate)
            if run_time:
                found.append((run_time, hour, candidate))
    if not found:
        return None
    run_time, hour, candidate = max(found, key=lambda item: item[0])
    return hour, run_time, candidate


def geps_candidate(links: Iterable[str]) -> str | None:
    candidates = []
    for name in links:
        lower = name.lower()
        if not lower.endswith(".grib2"):
            continue
        if "cmc_geps-raw_apcp_sfc_0_" not in lower:
            continue
        if not lower.endswith("_allmbrs.grib2"):
            continue
        candidates.append(name)
    return sorted(candidates, key=len)[0] if candidates else None


def geps_run_time(filename: str) -> datetime | None:
    match = GEPS_RUN_RE.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def latest_geps_run(session) -> tuple[str, datetime, str] | None:
    """Select the newest GEPS cycle that is complete at every required horizon.

    A newer PT024 file may appear hours before PT384. During that publication
    window, continue using the previous complete cycle rather than failing the
    live forecast or delaying a gauge-driven update.
    """
    found: list[tuple[datetime, str, str]] = []
    for hour in ["00", "12"]:
        run_times: list[datetime] = []
        probe: str | None = None
        complete = True
        for horizon in GEPS_HORIZONS:
            directory = f"{GEPS_BASE}/{hour}/{horizon:03d}/"
            try:
                candidate = geps_candidate(get_links(session, directory))
            except Exception:
                complete = False
                break
            if not candidate:
                complete = False
                break
            run_time = geps_run_time(candidate)
            if not run_time:
                complete = False
                break
            run_times.append(run_time)
            if horizon == GEPS_HORIZONS[0]:
                probe = candidate
        if complete and probe and len(run_times) == len(GEPS_HORIZONS):
            if all(value == run_times[0] for value in run_times):
                found.append((run_times[0], hour, probe))
    if not found:
        return None
    run_time, hour, candidate = max(found, key=lambda item: item[0])
    return hour, run_time, candidate


def clip_one_band(path: Path, geometry, geometry_crs) -> np.ma.MaskedArray:
    with rasterio.open(path) as dataset:
        projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(dataset.crs).iloc[0]
        values, _ = mask(dataset, [mapping(projected)], crop=True, filled=False, indexes=1)
    array = np.ma.asarray(values, dtype=float)
    return np.ma.masked_where(~np.isfinite(array) | (array < -0.01) | (array > 2000), array)


def clip_all_band_means(dataset, geometry, geometry_crs) -> list[float]:
    projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(dataset.crs).iloc[0]
    indexes = list(range(1, dataset.count + 1))
    values, _ = mask(dataset, [mapping(projected)], crop=True, filled=False, indexes=indexes)
    array = np.ma.asarray(values, dtype=float)
    array = np.ma.masked_where(~np.isfinite(array) | (array < -0.01) | (array > 5000), array)
    means: list[float] = []
    for band in range(array.shape[0]):
        valid = array[band].compressed()
        means.append(float(np.mean(valid)) if len(valid) else float("nan"))
    return means


def process_gdps(session, subareas: gpd.GeoDataFrame) -> tuple[list[dict], list[dict], dict, list[str]]:
    warnings: list[str] = []
    discovered = latest_gdps_run(session)
    if discovered is None:
        return [], [], {"model": "GDPS", "status": "no run discovered"}, [
            "GDPS: no APCP-Accum6h run discovered"
        ]
    run_hour, run_time, probe = discovered
    files: list[tuple[int, Path, str]] = []
    candidates: dict[str, str | None] = {}
    for forecast_hour in GDPS_HOURS:
        directory = f"{GDPS_BASE}/{run_hour}/{forecast_hour:03d}/"
        try:
            candidate = gdps_candidate(get_links(session, directory))
            candidates[str(forecast_hour)] = candidate
            if not candidate:
                warnings.append(f"GDPS PT{forecast_hour:03d}: APCP-Accum6h file missing")
                continue
            path = download(session, directory + candidate, CACHE / "gdps" / candidate)
            files.append((forecast_hour, path, candidate))
        except Exception as exc:
            warnings.append(f"GDPS PT{forecast_hour:03d}: {exc}")

    interval_rows: list[dict] = []
    horizon_rows: list[dict] = []
    accumulators: dict[str, np.ma.MaskedArray] = {}
    for forecast_hour, path, filename in files:
        for _, area in subareas.iterrows():
            try:
                values = clip_one_band(path, area.geometry, subareas.crs)
                row = {
                    "model": "GDPS",
                    "run_time_utc": run_time.isoformat(),
                    "forecast_hour_start": forecast_hour - 6,
                    "forecast_hour_end": forecast_hour,
                    "interval_hours": 6,
                    "accumulation_semantics": "independent_6h_interval",
                    "subarea": area.subarea,
                    "source_file": filename,
                }
                row.update(summarize(values))
                interval_rows.append(row)
                name = str(area.subarea)
                if name not in accumulators:
                    accumulators[name] = values.copy()
                else:
                    accumulators[name] = accumulators[name] + values
                if forecast_hour in GDPS_SUMMARY_HORIZONS:
                    total = summarize(accumulators[name])
                    total.update({
                        "model": "GDPS",
                        "run_time_utc": run_time.isoformat(),
                        "horizon_h": forecast_hour,
                        "subarea": name,
                        "intervals_used": forecast_hour // 6,
                        "complete_horizon": len([f for f, _, _ in files if f <= forecast_hour]) == forecast_hour // 6,
                    })
                    horizon_rows.append(total)
            except Exception as exc:
                warnings.append(f"GDPS {filename} {area.subarea}: {exc}")

    metadata = {
        "model": "GDPS",
        "status": "processed" if interval_rows else "no rows",
        "run_time_utc": run_time.isoformat(),
        "run_hour_utc": run_hour,
        "probe_file": probe,
        "files_processed": len(files),
        "expected_files": len(GDPS_HOURS),
        "candidate_count": sum(value is not None for value in candidates.values()),
        "horizons_h": sorted(GDPS_SUMMARY_HORIZONS),
        "accumulation_semantics": "Independent APCP-Accum6h fields summed to each horizon.",
    }
    return interval_rows, horizon_rows, metadata, warnings


def process_geps(session, subareas: gpd.GeoDataFrame) -> tuple[list[dict], dict, list[str]]:
    warnings: list[str] = []
    discovered = latest_geps_run(session)
    if discovered is None:
        return [], {"model": "GEPS", "status": "no run discovered"}, [
            "GEPS: no raw APCP all-members run discovered"
        ]
    run_hour, run_time, probe = discovered
    rows: list[dict] = []
    band_counts: dict[str, int] = {}
    band_metadata_examples: list[dict] = []
    member_series: dict[str, dict[int, list[float]]] = {}
    candidates: dict[str, str | None] = {}

    for horizon in GEPS_HORIZONS:
        directory = f"{GEPS_BASE}/{run_hour}/{horizon:03d}/"
        try:
            candidate = geps_candidate(get_links(session, directory))
            candidates[str(horizon)] = candidate
            if not candidate:
                warnings.append(f"GEPS PT{horizon:03d}: raw APCP all-members file missing")
                continue
            path = download(session, directory + candidate, CACHE / "geps" / candidate)
            with rasterio.open(path) as dataset:
                band_counts[str(horizon)] = int(dataset.count)
                if not band_metadata_examples:
                    for band in range(1, min(dataset.count, 5) + 1):
                        band_metadata_examples.append({
                            "band": band,
                            "description": dataset.descriptions[band - 1],
                            "tags": dataset.tags(band),
                        })
                for _, area in subareas.iterrows():
                    means = np.asarray(
                        clip_all_band_means(dataset, area.geometry, subareas.crs),
                        dtype=float,
                    )
                    means = means[np.isfinite(means)]
                    if not len(means):
                        continue
                    name = str(area.subarea)
                    member_series.setdefault(name, {})[horizon] = means.tolist()
                    row = {
                        "model": "GEPS",
                        "run_time_utc": run_time.isoformat(),
                        "horizon_h": horizon,
                        "subarea": name,
                        "source_file": candidate,
                        "member_count": int(len(means)),
                        "accumulation_semantics": "cumulative_from_model_start",
                        "mean_mm": float(np.mean(means)),
                        "min_mm": float(np.min(means)),
                        "p10_mm": float(np.percentile(means, 10)),
                        "p25_mm": float(np.percentile(means, 25)),
                        "p50_mm": float(np.percentile(means, 50)),
                        "p75_mm": float(np.percentile(means, 75)),
                        "p90_mm": float(np.percentile(means, 90)),
                        "max_mm": float(np.max(means)),
                    }
                    for threshold in [5, 10, 20, 30, 50]:
                        row[f"prob_ge_{threshold}mm"] = float(np.mean(means >= threshold) * 100)
                    rows.append(row)
        except Exception as exc:
            warnings.append(f"GEPS PT{horizon:03d}: {exc}")

    monotonic_violations = 0
    comparisons = 0
    for _, horizons in member_series.items():
        ordered = sorted(horizons)
        for previous, current in zip(ordered, ordered[1:]):
            a = np.asarray(horizons[previous], dtype=float)
            b = np.asarray(horizons[current], dtype=float)
            n = min(len(a), len(b))
            if not n:
                continue
            comparisons += n
            monotonic_violations += int(np.sum(b[:n] < a[:n] - 0.25))

    minimum_bands = min(band_counts.values()) if band_counts else 0
    validation_passed = bool(rows) and minimum_bands >= 20 and monotonic_violations == 0
    metadata = {
        "model": "GEPS",
        "status": "processed" if rows else "no rows",
        "run_time_utc": run_time.isoformat(),
        "run_hour_utc": run_hour,
        "probe_file": probe,
        "horizons_requested_h": GEPS_HORIZONS,
        "horizons_processed_h": sorted(int(key) for key in band_counts),
        "band_counts": band_counts,
        "minimum_band_count": minimum_bands,
        "band_metadata_examples": band_metadata_examples,
        "validation": {
            "passed": validation_passed,
            "expected_members_minimum": 20,
            "monotonic_comparisons": comparisons,
            "cumulative_monotonic_violations": monotonic_violations,
            "interpretation": "Raw APCP allmbrs files are treated as cumulative precipitation from model initialization.",
        },
    }
    if monotonic_violations:
        warnings.append(f"GEPS cumulative validation: {monotonic_violations} member/subarea decreases detected")
    if minimum_bands and minimum_bands < 20:
        warnings.append(f"GEPS member validation: only {minimum_bands} bands found")
    return rows, metadata, warnings


def main() -> None:
    SPATIAL.mkdir(parents=True, exist_ok=True)
    if not SUBAREAS.exists():
        raise RuntimeError("Derived subareas missing; historical spatial calibration must exist")
    subareas = gpd.read_file(SUBAREAS)
    session = http()

    gdps_intervals, gdps_horizons, gdps_meta, gdps_warnings = process_gdps(session, subareas)
    geps_rows, geps_meta, geps_warnings = process_geps(session, subareas)
    warnings = gdps_warnings + geps_warnings

    write_csv(gdps_intervals, SPATIAL / "gdps_interval_qpf_by_subarea.csv")
    write_csv(gdps_horizons, SPATIAL / "gdps_horizon_qpf_by_subarea.csv")
    write_csv(geps_rows, SPATIAL / "geps_qpf_by_subarea.csv")

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "short_range": "HRDPS 0-48 h handled by qpf_forecast_v2.py",
            "medium_range": "GDPS deterministic through 240 h",
            "ensemble_range": "GEPS probabilistic through 384 h",
            "beyond_16_days": "No event-scale water-level prediction; extended products are outlook/anomaly only.",
        },
        "gdps": gdps_meta,
        "geps": geps_meta,
        "row_counts": {
            "gdps_intervals": len(gdps_intervals),
            "gdps_horizons": len(gdps_horizons),
            "geps": len(geps_rows),
        },
        "warning_count": len(warnings),
    }
    (SPATIAL / "medium_range_qpf.json").write_text(json.dumps(result, indent=2))
    (SPATIAL / "medium_range_qpf_warnings.log").write_text(
        "\n".join(warnings) if warnings else "No warnings.\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
