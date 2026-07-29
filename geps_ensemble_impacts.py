#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from forecast_impacts_v2 import (
    BASE_CAL,
    CAL_V2,
    FEATURES,
    OUT,
    analog_predict,
    cross_validate,
    current_antecedent_rain,
    finite,
    rain_free_stage,
    training_frame,
    uncertainty_value,
)
from medium_range_qpf import CACHE, clip_all_band_means

ROOT = Path("sturgeon_pipeline_output")
SPATIAL = ROOT / "spatial"
GEPS_SUMMARY = SPATIAL / "geps_qpf_by_subarea.csv"
GEPS_META = SPATIAL / "medium_range_qpf.json"
SUBAREAS = SPATIAL / "derived_subareas.geojson"
FORECAST = OUT / "forecast_impacts_v2.json"
MEMBER_CSV = SPATIAL / "geps_member_qpf_by_subarea.csv"
ENSEMBLE_OUT = OUT / "ensemble_paths_v2.json"

REQUIRED_SUBAREAS = [
    "basin_to_05EA002",
    "lower_incremental_05EA005_to_05EA002",
    "upper_lake_chain_isle_lac_ste_anne",
]
WINDOW_ENDPOINTS = [48, 120, 240, 384]
SCENARIO_QUANTILES = {"dry": 0.10, "central": 0.50, "wet": 0.90}


def load_geps_members() -> pd.DataFrame:
    metadata = json.loads(GEPS_META.read_text())
    geps_meta = metadata.get("geps", {})
    if not geps_meta.get("validation", {}).get("passed"):
        raise RuntimeError("GEPS validation has not passed")
    summary = pd.read_csv(GEPS_SUMMARY)
    if summary.empty:
        raise RuntimeError("GEPS summary is empty")
    subareas = gpd.read_file(SUBAREAS)
    selected = subareas[subareas.subarea.isin(REQUIRED_SUBAREAS)].copy()
    missing = sorted(set(REQUIRED_SUBAREAS) - set(selected.subarea.astype(str)))
    if missing:
        raise RuntimeError(f"Required subareas missing: {missing}")

    rows: list[dict] = []
    horizons = sorted(
        pd.to_numeric(summary.horizon_h, errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for horizon in horizons:
        basin_rows = summary[
            (summary.horizon_h == horizon)
            & (summary.subarea == "basin_to_05EA002")
        ]
        if basin_rows.empty:
            continue
        source_file = str(basin_rows.iloc[0].source_file)
        path = CACHE / "geps" / source_file
        if not path.exists():
            raise RuntimeError(f"Cached GEPS file missing: {path}")
        with rasterio.open(path) as dataset:
            expected_count = int(dataset.count)
            by_subarea: dict[str, np.ndarray] = {}
            for _, area in selected.iterrows():
                means = np.asarray(
                    clip_all_band_means(dataset, area.geometry, selected.crs),
                    dtype=float,
                )
                if len(means) != expected_count:
                    raise RuntimeError(
                        f"{source_file} {area.subarea}: expected "
                        f"{expected_count} members, got {len(means)}"
                    )
                by_subarea[str(area.subarea)] = means
            for member_index in range(expected_count):
                for subarea, values in by_subarea.items():
                    rows.append(
                        {
                            "model": "GEPS",
                            "run_time_utc": basin_rows.iloc[0].run_time_utc,
                            "horizon_h": int(horizon),
                            "member_index": int(member_index + 1),
                            "subarea": subarea,
                            "cumulative_mm": float(values[member_index]),
                            "source_file": source_file,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No GEPS member rows produced")
    frame.to_csv(MEMBER_CSV, index=False)
    return frame


def member_value(
    frame: pd.DataFrame,
    member_index: int,
    horizon: int,
    subarea: str,
) -> float:
    rows = frame[
        (frame.member_index == member_index)
        & (frame.horizon_h == horizon)
        & (frame.subarea == subarea)
    ]
    if rows.empty:
        return 0.0
    return max(0.0, finite(rows.iloc[0].cumulative_mm, 0.0))


def nearest_quantile_member(
    frame: pd.DataFrame,
    horizon: int,
    quantile: float,
) -> tuple[int, float]:
    basin = frame[
        (frame.horizon_h == horizon)
        & (frame.subarea == "basin_to_05EA002")
    ][["member_index", "cumulative_mm"]].dropna()
    if basin.empty:
        raise RuntimeError(f"No basin members at {horizon} h")
    target = float(np.quantile(basin.cumulative_mm.to_numpy(float), quantile))
    index = int((basin.cumulative_mm - target).abs().idxmin())
    row = basin.loc[index]
    return int(row.member_index), float(row.cumulative_mm)


def short_range_delay(forecast: dict, scenario: str) -> tuple[float, dict]:
    candidates = [
        item
        for item in forecast.get("deterministic_scenarios", [])
        if str(item.get("model")) == "HRDPS"
        and int(item.get("horizon_h", 0)) == 48
    ]
    if not candidates:
        return 0.0, {"status": "HRDPS 48 h impact unavailable"}
    item = candidates[0]
    delay_range = item.get("estimated_days_lost_range", [0.0, 0.0])
    central = finite(
        item.get("analog_prediction", {}).get("days_lost"),
        finite(item.get("projected_1_70_days_central"), 0.0)
        - finite(forecast.get("rain_free_days_to_1_70"), 0.0),
    )
    if scenario == "dry":
        delay = finite(delay_range[0], central)
    elif scenario == "wet":
        delay = finite(delay_range[1], central)
    else:
        delay = central
    return max(0.0, delay), {
        "status": "HRDPS 48 h quantified impact",
        "delay_days": max(0.0, delay),
        "source": item,
    }


def window_prediction(
    training: pd.DataFrame,
    cv: dict,
    stage_now: float,
    recession_model: dict,
    antecedent: float,
    start_h: int,
    end_h: int,
    basin_mm: float,
    lower_mm: float,
    upper_mm: float,
    prior_delay_days: float,
) -> dict:
    if basin_mm < 0.5:
        return {
            "start_h": start_h,
            "end_h": end_h,
            "basin_mm": basin_mm,
            "lower_mm": lower_mm,
            "upper_mm": upper_mm,
            "days_lost_central": 0.0,
            "days_lost_rmse": 0.0,
            "status": "negligible",
        }
    effective_start_h = max(0.0, start_h - prior_delay_days * 24.0)
    pre_stage = rain_free_stage(stage_now, recession_model, effective_start_h)
    duration_h = float(min(72, max(24, end_h - start_h)))
    feature_values = np.asarray(
        [
            basin_mm,
            lower_mm / basin_mm if basin_mm > 0 else 1.0,
            upper_mm / basin_mm if basin_mm > 0 else 1.0,
            duration_h,
            antecedent,
            0.0,
            pre_stage,
        ],
        dtype=float,
    )
    if len(training) < 2:
        return {
            "start_h": start_h,
            "end_h": end_h,
            "basin_mm": basin_mm,
            "lower_mm": lower_mm,
            "upper_mm": upper_mm,
            "status": "insufficient training",
            "days_lost_central": 0.0,
            "days_lost_rmse": None,
        }
    prediction = analog_predict(training, feature_values, k=min(3, len(training)))
    error = uncertainty_value(
        cv,
        "days_lost",
        max(
            1.0,
            float(training.days_lost.std(ddof=0))
            if len(training) > 1
            else 1.5,
        ),
    )
    return {
        "start_h": start_h,
        "end_h": end_h,
        "basin_mm": basin_mm,
        "lower_mm": lower_mm,
        "upper_mm": upper_mm,
        "duration_proxy_h": duration_h,
        "pre_stage_m": pre_stage,
        "feature_vector": dict(zip(FEATURES, [float(v) for v in feature_values])),
        "analog_prediction": prediction,
        "days_lost_central": max(0.0, finite(prediction.get("days_lost"), 0.0)),
        "days_lost_rmse": max(0.0, error),
        "status": "quantified with low-confidence event analogues",
    }


def build_path(
    stage_now: float,
    recession_model: dict,
    projected_days: float,
    delays: list[dict],
) -> list[dict]:
    horizon_days = max(16, int(math.ceil(projected_days)) + 2)
    rows = []
    for day in range(0, horizon_days + 1):
        accrued = sum(
            finite(item.get("applied_delay_days"), 0.0)
            for item in delays
            if finite(item.get("start_h"), 0.0) <= day * 24.0
        )
        effective_days = max(0.0, day - accrued)
        rows.append(
            {
                "day": day,
                "stage_05EA002_m": rain_free_stage(
                    stage_now,
                    recession_model,
                    effective_days * 24.0,
                ),
                "accrued_delay_days": accrued,
                "effective_recession_days": effective_days,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    members = load_geps_members()
    base = json.loads(BASE_CAL.read_text())
    forecast = json.loads(FORECAST.read_text())
    events = pd.read_csv(CAL_V2)
    _, training = training_frame(events)
    cv = cross_validate(training)
    stage_now = finite(base.get("latest_stage_m"))
    recession_model = base.get("master_recession", {})
    rain_free_days = finite(
        base.get("rain_free_projection_to_1_70", {}).get("days")
    )
    antecedent = current_antecedent_rain(168)

    available_horizons = sorted(
        pd.to_numeric(members.horizon_h, errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    selection_horizon = max(h for h in available_horizons if h <= 384)
    endpoints = [h for h in WINDOW_ENDPOINTS if h in available_horizons]
    if 48 not in endpoints:
        raise RuntimeError("GEPS 48 h horizon missing")

    scenarios: dict[str, dict] = {}
    for name, quantile in SCENARIO_QUANTILES.items():
        member_index, selected_total = nearest_quantile_member(
            members,
            selection_horizon,
            quantile,
        )
        short_delay, short_detail = short_range_delay(forecast, name)
        total_delay = short_delay
        projected_days = rain_free_days + total_delay
        delays = [
            {
                "start_h": 0,
                "end_h": 48,
                "applied_delay_days": short_delay,
                "source": "HRDPS 48 h impact range",
                "detail": short_detail,
            }
        ]
        windows = []
        previous_h = 48
        previous_values = {
            area: member_value(members, member_index, 48, area)
            for area in REQUIRED_SUBAREAS
        }
        cumulative_before = previous_values["basin_to_05EA002"]

        for end_h in [h for h in endpoints if h > 48]:
            current_values = {
                area: member_value(members, member_index, end_h, area)
                for area in REQUIRED_SUBAREAS
            }
            increments = {
                area: max(0.0, current_values[area] - previous_values[area])
                for area in REQUIRED_SUBAREAS
            }
            prediction = window_prediction(
                training=training,
                cv=cv,
                stage_now=stage_now,
                recession_model=recession_model,
                antecedent=antecedent + cumulative_before,
                start_h=previous_h,
                end_h=end_h,
                basin_mm=increments["basin_to_05EA002"],
                lower_mm=increments[
                    "lower_incremental_05EA005_to_05EA002"
                ],
                upper_mm=increments[
                    "upper_lake_chain_isle_lac_ste_anne"
                ],
                prior_delay_days=total_delay,
            )
            apply_window = (previous_h / 24.0) <= projected_days
            applied = (
                finite(prediction.get("days_lost_central"), 0.0)
                if apply_window
                else 0.0
            )
            prediction["applied_before_threshold"] = bool(apply_window)
            prediction["applied_delay_days"] = applied
            windows.append(prediction)
            if applied > 0:
                delays.append(
                    {
                        "start_h": previous_h,
                        "end_h": end_h,
                        "applied_delay_days": applied,
                        "source": "GEPS member window analogue",
                        "detail": prediction,
                    }
                )
                total_delay += applied
                projected_days = rain_free_days + total_delay
            previous_h = end_h
            previous_values = current_values
            cumulative_before = current_values["basin_to_05EA002"]

        model_rmse = math.sqrt(
            sum(
                finite(item.get("days_lost_rmse"), 0.0) ** 2
                for item in windows
                if item.get("applied_before_threshold")
            )
        )
        projected_date = (
            datetime.now(timezone.utc) + timedelta(days=projected_days)
        ).date().isoformat()
        scenarios[name] = {
            "meteorological_quantile": quantile,
            "selected_member_index": member_index,
            "selection_horizon_h": selection_horizon,
            "selected_member_basin_total_mm": selected_total,
            "cumulative_rainfall_mm": {
                str(h): {
                    area: member_value(members, member_index, h, area)
                    for area in REQUIRED_SUBAREAS
                }
                for h in endpoints
            },
            "short_range": short_detail,
            "later_windows": windows,
            "total_applied_delay_days": total_delay,
            "model_delay_rmse_days": model_rmse,
            "projected_1_70_days": projected_days,
            "projected_1_70_date_utc": projected_date,
            "path_daily": build_path(
                stage_now,
                recession_model,
                projected_days,
                delays,
            ),
        }

    official = {
        "status": "operational_geps_integrated",
        "primary_scenario": "central",
        "dry_days_to_1_70": scenarios["dry"]["projected_1_70_days"],
        "central_days_to_1_70": scenarios["central"]["projected_1_70_days"],
        "wet_days_to_1_70": scenarios["wet"]["projected_1_70_days"],
        "dry_date_utc": scenarios["dry"]["projected_1_70_date_utc"],
        "central_date_utc": scenarios["central"]["projected_1_70_date_utc"],
        "wet_date_utc": scenarios["wet"]["projected_1_70_date_utc"],
        "interpretation": (
            "HRDPS governs the first 48 hours. Actual GEPS members selected near "
            "the 10th, 50th and 90th percentiles govern later rainfall windows "
            "through 16 days. Only windows beginning before the projected 1.70 m "
            "crossing are allowed to delay that crossing."
        ),
        "confidence": "low",
    }

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_stage_m": stage_now,
        "rain_free_days_to_1_70": rain_free_days,
        "geps_run_time_utc": members.run_time_utc.iloc[0],
        "geps_member_count": int(members.member_index.nunique()),
        "geps_horizons_h": available_horizons,
        "official_outlook": official,
        "scenarios": scenarios,
        "limitations": [
            "Only two uncensored historical peak-response events are available for point prediction.",
            "No historical event has a completely observed recovery, so delay estimates remain weakly constrained.",
            "GEPS supplies cumulative precipitation and broad timing windows; exact storm timing inside each window is unresolved.",
            "The medium-range analogue uses a duration proxy capped at 72 hours and has no forecast spatial-coverage percentage feature.",
            "Daily paths represent recession-clock delays at 05EA002, not a hydraulic simulation or exact Starkey stage.",
        ],
    }
    ENSEMBLE_OUT.write_text(json.dumps(output, indent=2))
    forecast["ensemble_medium_range"] = output
    forecast["official_outlook"] = official
    forecast.setdefault("limitations", []).append(
        "Medium-range dry/central/wet paths are delay-based GEPS member scenarios, not deterministic hydraulic traces."
    )
    FORECAST.write_text(json.dumps(forecast, indent=2))
    print(
        json.dumps(
            {
                "geps_members": output["geps_member_count"],
                "horizons": len(available_horizons),
                "official_outlook": official,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
