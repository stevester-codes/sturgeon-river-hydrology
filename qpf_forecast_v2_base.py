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
import requests
from rasterio.mask import mask
from requests.adapters import HTTPAdapter
from shapely.geometry import mapping
from urllib3.util.retry import Retry

ROOT = Path("sturgeon_pipeline_output")
SPATIAL = ROOT / "spatial"
CACHE = Path("grid_cache") / "qpf_v2"
SUBAREAS = SPATIAL / "derived_subareas.geojson"
HREF = re.compile(r'href="([^"?]+)"', re.I)
RUN_RE = re.compile(r"^(\d{8})T(\d{2})Z_")

MODEL_CONFIG = {
    "HRDPS": {
        "base": "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km",
        "max_hour": 48,
        "grid": "2.5km",
    },
    "RDPS": {
        "base": "https://dd.weather.gc.ca/today/model_rdps/10km",
        "max_hour": 84,
        "grid": "10km",
    },
}
REPS_BASE = "https://dd.weather.gc.ca/today/ensemble/reps/10km/grib2"


def http() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = (
        "sturgeon-river-hydrology/3.0 "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_links(session: requests.Session, url: str) -> list[str]:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return HREF.findall(response.text)


def download(session: requests.Session, url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100:
        return path
    temp = path.with_suffix(path.suffix + ".part")
    with session.get(url, timeout=300, stream=True) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp.replace(path)
    return path


def parse_run_time(filename: str) -> datetime | None:
    match = RUN_RE.match(filename)
    if not match:
        return None
    try:
        return datetime.strptime(
            match.group(1) + match.group(2), "%Y%m%d%H"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def deterministic_candidate(links: Iterable[str], model: str) -> str | None:
    candidates = []
    token = f"_MSC_{model}_APCP-Accum6h_".lower()
    for name in links:
        lower = name.lower()
        if not lower.endswith(".grib2"):
            continue
        if token not in lower:
            continue
        if "-prob_" in lower or any(
            bad in lower for bad in ["convective", "freezing", "snow", "solid"]
        ):
            continue
        candidates.append(name)
    return sorted(candidates, key=len)[0] if candidates else None


def latest_deterministic_run(
    session: requests.Session, model: str
) -> tuple[str, str, datetime] | None:
    config = MODEL_CONFIG[model]
    found: list[tuple[datetime, str, str]] = []
    for hour in ["00", "06", "12", "18"]:
        url = f"{config['base']}/{hour}/006/"
        try:
            candidate = deterministic_candidate(get_links(session, url), model)
        except Exception:
            continue
        if candidate:
            run_time = parse_run_time(candidate)
            if run_time:
                found.append((run_time, hour, candidate))
    if not found:
        return None
    run_time, hour, candidate = max(found, key=lambda item: item[0])
    return hour, candidate, run_time


def clip_band(
    path: Path,
    geometry,
    geometry_crs,
    band: int = 1,
) -> np.ma.MaskedArray:
    with rasterio.open(path) as dataset:
        projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(
            dataset.crs
        ).iloc[0]
        values, _ = mask(
            dataset,
            [mapping(projected)],
            crop=True,
            filled=False,
            indexes=band,
        )
        array = np.ma.asarray(values, dtype=float)
        return np.ma.masked_where(
            ~np.isfinite(array) | (array < -0.01) | (array > 1000), array
        )


def summarize(array: np.ma.MaskedArray) -> dict:
    values = array.compressed()
    if not len(values):
        return {
            "mean_mm": None,
            "max_mm": None,
            "valid_cells": 0,
            "pct_gt_5mm": None,
            "pct_gt_10mm": None,
            "pct_gt_20mm": None,
            "pct_gt_30mm": None,
            "pct_gt_50mm": None,
        }
    result = {
        "mean_mm": float(np.mean(values)),
        "max_mm": float(np.max(values)),
        "valid_cells": int(len(values)),
    }
    for threshold in [5, 10, 20, 30, 50]:
        result[f"pct_gt_{threshold}mm"] = float(
            np.mean(values > threshold) * 100
        )
    return result


def process_deterministic(
    session: requests.Session,
    model: str,
    subareas: gpd.GeoDataFrame,
) -> tuple[list[dict], dict, list[str]]:
    config = MODEL_CONFIG[model]
    warnings: list[str] = []
    discovered = latest_deterministic_run(session, model)
    if discovered is None:
        return [], {"model": model, "status": "no run discovered"}, [
            f"{model}: no six-hour APCP run discovered"
        ]
    run_hour, probe_file, run_time = discovered
    files: list[tuple[int, Path, str]] = []
    candidates: dict[str, str | None] = {}
    for forecast_hour in range(6, int(config["max_hour"]) + 1, 6):
        directory = f"{config['base']}/{run_hour}/{forecast_hour:03d}/"
        try:
            candidate = deterministic_candidate(
                get_links(session, directory), model
            )
            candidates[str(forecast_hour)] = candidate
            if not candidate:
                warnings.append(
                    f"{model} PT{forecast_hour:03d}: APCP-Accum6h file missing"
                )
                continue
            path = CACHE / model.lower() / candidate
            files.append(
                (
                    forecast_hour,
                    download(session, directory + candidate, path),
                    candidate,
                )
            )
        except Exception as exc:
            warnings.append(f"{model} PT{forecast_hour:03d}: {exc}")
    rows: list[dict] = []
    for forecast_hour, path, filename in files:
        for _, area in subareas.iterrows():
            try:
                row = {
                    "model": model,
                    "run_time_utc": run_time.isoformat(),
                    "forecast_hour_start": forecast_hour - 6,
                    "forecast_hour_end": forecast_hour,
                    "interval_hours": 6,
                    "accumulation_semantics": "independent_6h_interval",
                    "subarea": area.subarea,
                    "source_file": filename,
                }
                row.update(
                    summarize(
                        clip_band(path, area.geometry, subareas.crs, band=1)
                    )
                )
                rows.append(row)
            except Exception as exc:
                warnings.append(
                    f"{model} {filename} {area.subarea}: {exc}"
                )
    metadata = {
        "model": model,
        "status": "processed" if rows else "no rows",
        "run_time_utc": run_time.isoformat(),
        "run_hour_utc": run_hour,
        "probe_file": probe_file,
        "files_processed": len(files),
        "expected_files": int(config["max_hour"]) // 6,
        "candidates": candidates,
        "accumulation_semantics": (
            "Each APCP-Accum6h field is an independent six-hour interval; "
            "forecast totals are the sum of intervals."
        ),
    }
    return rows, metadata, warnings


def reps_candidate(
    links: Iterable[str], accumulation_hours: int, probability: bool
) -> str | None:
    wanted = f"TPRATE-Accum{accumulation_hours}h"
    candidates = []
    for name in links:
        if not name.endswith(".grib2") or wanted not in name:
            continue
        has_probability = "-Prob_" in name
        if has_probability != probability:
            continue
        candidates.append(name)
    return sorted(candidates, key=len)[0] if candidates else None


def latest_reps_run(
    session: requests.Session,
) -> tuple[str, datetime] | None:
    found: list[tuple[datetime, str]] = []
    for hour in ["00", "06", "12", "18"]:
        directory = f"{REPS_BASE}/{hour}/024/"
        try:
            candidate = reps_candidate(get_links(session, directory), 24, False)
        except Exception:
            continue
        if candidate:
            run_time = parse_run_time(candidate)
            if run_time:
                found.append((run_time, hour))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1], max(
        found, key=lambda item: item[0]
    )[0]


def band_metadata(dataset: rasterio.DatasetReader, band: int) -> dict:
    return {
        "band": band,
        "description": dataset.descriptions[band - 1],
        "tags": dataset.tags(band),
    }


def process_reps(
    session: requests.Session,
    subareas: gpd.GeoDataFrame,
) -> tuple[list[dict], dict, list[str]]:
    warnings: list[str] = []
    discovered = latest_reps_run(session)
    if discovered is None:
        return [], {"model": "REPS", "status": "no run discovered"}, [
            "REPS: no ensemble precipitation run discovered"
        ]
    run_hour, run_time = discovered
    rows: list[dict] = []
    file_metadata: list[dict] = []
    for horizon in [24, 48]:
        directory = f"{REPS_BASE}/{run_hour}/{horizon:03d}/"
        try:
            links = get_links(session, directory)
            candidate = reps_candidate(links, horizon, False)
            if not candidate:
                warnings.append(
                    f"REPS PT{horizon:03d}: non-probability Accum{horizon}h file missing"
                )
                continue
            path = download(
                session,
                directory + candidate,
                CACHE / "reps" / candidate,
            )
            with rasterio.open(path) as dataset:
                metadata_sample = [
                    band_metadata(dataset, band)
                    for band in range(1, min(dataset.count, 25) + 1)
                ]
                file_metadata.append(
                    {
                        "horizon_h": horizon,
                        "source_file": candidate,
                        "band_count": dataset.count,
                        "band_metadata_sample": metadata_sample,
                    }
                )
                for _, area in subareas.iterrows():
                    member_means: list[float] = []
                    for band in range(1, dataset.count + 1):
                        try:
                            values = clip_band(
                                path, area.geometry, subareas.crs, band=band
                            ).compressed()
                            if len(values):
                                member_means.append(float(np.mean(values)))
                        except Exception as exc:
                            warnings.append(
                                f"REPS {candidate} band {band} {area.subarea}: {exc}"
                            )
                    if not member_means:
                        continue
                    values = np.asarray(member_means, dtype=float)
                    row = {
                        "model": "REPS",
                        "run_time_utc": run_time.isoformat(),
                        "horizon_h": horizon,
                        "subarea": area.subarea,
                        "source_file": candidate,
                        "member_or_stat_band_count": int(len(values)),
                        "mean_mm": float(np.mean(values)),
                        "min_mm": float(np.min(values)),
                        "p10_mm": float(np.percentile(values, 10)),
                        "p25_mm": float(np.percentile(values, 25)),
                        "p50_mm": float(np.percentile(values, 50)),
                        "p75_mm": float(np.percentile(values, 75)),
                        "p90_mm": float(np.percentile(values, 90)),
                        "max_mm": float(np.max(values)),
                        "prob_ge_5mm": float(np.mean(values >= 5) * 100),
                        "prob_ge_10mm": float(np.mean(values >= 10) * 100),
                        "prob_ge_20mm": float(np.mean(values >= 20) * 100),
                        "prob_ge_30mm": float(np.mean(values >= 30) * 100),
                    }
                    rows.append(row)
        except Exception as exc:
            warnings.append(f"REPS PT{horizon:03d}: {exc}")
    metadata = {
        "model": "REPS",
        "status": "processed" if rows else "no rows",
        "run_time_utc": run_time.isoformat(),
        "run_hour_utc": run_hour,
        "files": file_metadata,
        "interpretation_warning": (
            "REPS GRIB files may contain ensemble members and statistical bands. "
            "Band metadata is preserved so member/statistic identification can be "
            "verified before operational probability claims."
        ),
    }
    return rows, metadata, warnings


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


def main() -> None:
    SPATIAL.mkdir(parents=True, exist_ok=True)
    if not SUBAREAS.exists():
        raise RuntimeError(
            "Derived subareas are missing; run spatial_qpf.py before qpf_forecast_v2.py"
        )
    subareas = gpd.read_file(SUBAREAS)
    session = http()
    deterministic_rows: list[dict] = []
    deterministic_metadata: list[dict] = []
    warnings: list[str] = []
    for model in ["HRDPS", "RDPS"]:
        rows, metadata, model_warnings = process_deterministic(
            session, model, subareas
        )
        deterministic_rows.extend(rows)
        deterministic_metadata.append(metadata)
        warnings.extend(model_warnings)
    write_csv(
        deterministic_rows, SPATIAL / "deterministic_qpf_by_subarea.csv"
    )
    reps_rows, reps_metadata, reps_warnings = process_reps(session, subareas)
    warnings.extend(reps_warnings)
    write_csv(reps_rows, SPATIAL / "ensemble_qpf_by_subarea.csv")
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_models": deterministic_metadata,
        "deterministic_rows": len(deterministic_rows),
        "reps": reps_metadata,
        "reps_rows": len(reps_rows),
        "warning_count": len(warnings),
        "validation": {
            "deterministic_accumulation": (
                "APCP-Accum6h fields are summed as independent six-hour intervals; "
                "no cumulative differencing is used."
            ),
            "units": "APCP/TPRATE accumulated water equivalent treated as millimetres.",
        },
    }
    (SPATIAL / "qpf_v2.json").write_text(json.dumps(result, indent=2))
    (SPATIAL / "qpf_v2_warnings.log").write_text(
        "\n".join(warnings) if warnings else "No warnings.\n"
    )
    print(
        json.dumps(
            {
                "deterministic_rows": len(deterministic_rows),
                "reps_rows": len(reps_rows),
                "warnings": len(warnings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
