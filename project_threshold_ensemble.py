#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_impacts_v2 import (
    BASE_CAL,
    CAL_V2,
    cross_validate,
    current_antecedent_rain,
    finite,
    model_rate,
    training_frame,
)
from forecast_starkey_wse import (
    FIELD_TARGET_Q,
    choose_current_limb,
    fit_rating,
    hourly_pairs,
    read_pairs,
    stage_from_q,
)
from run_geps_ensemble_impacts import (
    REQUIRED_SUBAREAS,
    WINDOW_ENDPOINTS,
    load_geps_members,
    member_value,
    short_range_delay,
    window_prediction,
)

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
FORECAST = ROOT / "forecast_v2" / "forecast_impacts_v2.json"
OUT_JSON = ROOT / "forecast_v2" / "project_threshold_ensemble.json"
OUT_CSV = ROOT / "forecast_v2" / "project_threshold_member_forecast.csv"
PROJECT_WSE_M = 650.20
PROJECT_RIVER_STATION = 18883
MAX_RECESSION_HOURS = 24 * 90


def rain_free_days_to_stage(stage_now: float, target_stage: float, recession_model: dict) -> float:
    if stage_now <= target_stage:
        return 0.0
    stage = float(stage_now)
    hours = 0
    while stage > target_stage and hours < MAX_RECESSION_HOURS:
        stage += model_rate(recession_model, stage) / 24.0
        hours += 1
    if stage > target_stage:
        raise RuntimeError("Rain-free recession did not reach the project threshold within 90 days")
    return hours / 24.0


def compact_short_range(detail: dict) -> dict:
    source = detail.get("source", {}) if isinstance(detail, dict) else {}
    return {
        "status": detail.get("status") if isinstance(detail, dict) else None,
        "delay_days": finite(detail.get("delay_days"), 0.0) if isinstance(detail, dict) else 0.0,
        "model": source.get("model"),
        "horizon_h": source.get("horizon_h"),
        "basin_mm": finite(source.get("basin_mm"), None),
        "lower_mm": finite(source.get("lower_mm"), None),
        "upper_mm": finite(source.get("upper_mm"), None),
        "storm_type": source.get("storm_type"),
        "estimated_days_lost_range": source.get("estimated_days_lost_range"),
    }


def date_from_days(generated: datetime, days: float) -> str:
    return (generated + timedelta(days=float(days))).date().isoformat()


def quantile_record(values: np.ndarray, generated: datetime, quantile: float) -> dict:
    days = float(np.quantile(values, quantile))
    return {
        "quantile": quantile,
        "days": days,
        "date_utc": date_from_days(generated, days),
    }


def main() -> None:
    for path in (BASE_CAL, CAL_V2, SUMMARY, FORECAST):
        if not path.exists():
            raise FileNotFoundError(path)

    generated = datetime.now(timezone.utc)
    base = json.loads(BASE_CAL.read_text())
    summary = json.loads(SUMMARY.read_text())
    forecast = json.loads(FORECAST.read_text())
    events = pd.read_csv(CAL_V2)
    _, training = training_frame(events)
    cv = cross_validate(training)
    members = load_geps_members()

    target = summary.get("target_05EA002", {})
    stage_now = finite(target.get("latest"))
    change_24h = finite(target.get("change_24h"), 0.0)
    if stage_now is None:
        raise RuntimeError("Current 05EA002 stage is unavailable")

    falling = change_24h < -0.005
    limb = "falling" if falling else ("rising" if change_24h > 0.005 else "approximately_flat")
    pairs = read_pairs()
    rating_rows = choose_current_limb(hourly_pairs(pairs), falling=falling)
    rating = fit_rating(rating_rows)
    target_stage = stage_from_q(FIELD_TARGET_Q, rating)

    recession_model = base.get("master_recession", {})
    base_days = rain_free_days_to_stage(stage_now, target_stage, recession_model)
    antecedent = current_antecedent_rain(168)
    short_delay, short_detail = short_range_delay(forecast, "central")

    available_horizons = sorted(
        pd.to_numeric(members.horizon_h, errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    endpoints = [h for h in WINDOW_ENDPOINTS if h in available_horizons]
    if 48 not in endpoints:
        raise RuntimeError("GEPS 48 h horizon is missing")

    member_rows: list[dict] = []
    member_indices = sorted(pd.to_numeric(members.member_index, errors="coerce").dropna().astype(int).unique())
    for member_index in member_indices:
        total_delay = max(0.0, float(short_delay))
        projected_days = base_days + total_delay
        previous_h = 48
        previous_values = {
            area: member_value(members, member_index, 48, area)
            for area in REQUIRED_SUBAREAS
        }
        cumulative_before = previous_values["basin_to_05EA002"]
        applied_windows: list[dict] = []
        window_rmse: list[float] = []

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
                lower_mm=increments["lower_incremental_05EA005_to_05EA002"],
                upper_mm=increments["upper_lake_chain_isle_lac_ste_anne"],
                prior_delay_days=total_delay,
            )

            apply_window = (previous_h / 24.0) <= projected_days
            applied_delay = (
                max(0.0, finite(prediction.get("days_lost_central"), 0.0))
                if apply_window
                else 0.0
            )
            prediction["applied_before_project_threshold"] = bool(apply_window)
            prediction["applied_delay_days"] = applied_delay
            applied_windows.append(prediction)
            if apply_window:
                window_rmse.append(max(0.0, finite(prediction.get("days_lost_rmse"), 0.0)))
            if applied_delay > 0:
                total_delay += applied_delay
                projected_days = base_days + total_delay

            previous_h = end_h
            previous_values = current_values
            cumulative_before = current_values["basin_to_05EA002"]

        model_rmse = math.sqrt(sum(value * value for value in window_rmse))
        row = {
            "member_index": int(member_index),
            "crossing_days": float(projected_days),
            "crossing_date_utc": date_from_days(generated, projected_days),
            "rain_free_days_to_project_threshold": float(base_days),
            "short_range_delay_days": float(short_delay),
            "total_applied_delay_days": float(total_delay),
            "model_delay_rmse_days": float(model_rmse),
            "basin_rain_48h_mm": member_value(members, member_index, 48, "basin_to_05EA002"),
            "basin_rain_120h_mm": member_value(members, member_index, 120, "basin_to_05EA002") if 120 in endpoints else None,
            "basin_rain_240h_mm": member_value(members, member_index, 240, "basin_to_05EA002") if 240 in endpoints else None,
            "basin_rain_384h_mm": member_value(members, member_index, 384, "basin_to_05EA002") if 384 in endpoints else None,
            "later_windows": applied_windows,
        }
        member_rows.append(row)

    if not member_rows:
        raise RuntimeError("No GEPS project-threshold member forecasts were produced")

    days = np.asarray([row["crossing_days"] for row in member_rows], dtype=float)
    quantiles = {
        name: quantile_record(days, generated, value)
        for name, value in {
            "p10": 0.10,
            "p25": 0.25,
            "p50": 0.50,
            "p75": 0.75,
            "p90": 0.90,
        }.items()
    }

    representative_members = {}
    for name, item in quantiles.items():
        target_days = item["days"]
        chosen = min(member_rows, key=lambda row: abs(row["crossing_days"] - target_days))
        representative_members[name] = {
            "member_index": chosen["member_index"],
            "crossing_days": chosen["crossing_days"],
            "crossing_date_utc": chosen["crossing_date_utc"],
            "basin_rain_384h_mm": chosen.get("basin_rain_384h_mm"),
        }

    cdf = []
    first_day = max(0, int(math.floor(float(np.min(days)))))
    last_day = int(math.ceil(float(np.max(days)))) + 3
    for offset in range(first_day, last_day + 1):
        cdf.append(
            {
                "days_from_run": offset,
                "date_utc": date_from_days(generated, offset),
                "probability_exposed_by_date_pct": float(np.mean(days <= offset) * 100.0),
            }
        )

    model_rmse_values = np.asarray([row["model_delay_rmse_days"] for row in member_rows], dtype=float)
    output = {
        "generated_utc": generated.isoformat(),
        "status": "operational_project_threshold_ensemble",
        "method": "All validated GEPS members are translated into project-threshold crossing dates. HRDPS supplies one common central 0-48 h impact; each GEPS member supplies coherent later rainfall windows through 16 days. A later window is applied only if it begins before that member's current projected RS18883 exposure time.",
        "project_target": {
            "river_station": PROJECT_RIVER_STATION,
            "wse_m_cgvd28": PROJECT_WSE_M,
            "target_05EA002_discharge_m3s": FIELD_TARGET_Q,
            "hydrograph_limb": limb,
            "equivalent_05EA002_stage_m": float(target_stage),
            "current_05EA002_stage_m": float(stage_now),
            "rain_free_days_to_threshold": float(base_days),
        },
        "short_range_hrdps": compact_short_range(short_detail),
        "antecedent_168h_basin_rain_mm": float(antecedent),
        "geps": {
            "member_count": len(member_rows),
            "horizons_h": available_horizons,
            "window_endpoints_h": endpoints,
        },
        "crossing_distribution": {
            "mean_days": float(np.mean(days)),
            "standard_deviation_days": float(np.std(days, ddof=0)),
            "earliest": {
                "days": float(np.min(days)),
                "date_utc": date_from_days(generated, float(np.min(days))),
            },
            "latest": {
                "days": float(np.max(days)),
                "date_utc": date_from_days(generated, float(np.max(days))),
            },
            "quantiles": quantiles,
            "representative_members_by_crossing_date": representative_members,
            "probability_exposed_by_date": cdf,
        },
        "model_uncertainty": {
            "median_later_window_delay_rmse_days": float(np.median(model_rmse_values)),
            "maximum_later_window_delay_rmse_days": float(np.max(model_rmse_values)),
            "interpretation": "The member spread is meteorological uncertainty only. Analogue-response uncertainty is additional and is summarized separately rather than pretending the GEPS member frequency is a fully calibrated probability of construction readiness.",
        },
        "members": member_rows,
        "limitations": [
            "Only two uncensored historical peak-response events and zero complete recoveries are available for rainfall-impact prediction.",
            "All members share the same deterministic HRDPS central impact during the first 48 hours; short-range hydrologic uncertainty is not represented as an ensemble.",
            "GEPS grid resolution is too coarse to resolve small subbasins reliably; basin totals and broad upper/lower contrasts are used.",
            "Member frequencies are raw ensemble probabilities and have not been statistically calibrated against historical project-threshold forecast errors.",
            "Final work release still requires direct confirmation of drainage and bearing capacity at RS18883.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2))
    csv_fields = [
        "member_index",
        "crossing_days",
        "crossing_date_utc",
        "rain_free_days_to_project_threshold",
        "short_range_delay_days",
        "total_applied_delay_days",
        "model_delay_rmse_days",
        "basin_rain_48h_mm",
        "basin_rain_120h_mm",
        "basin_rain_240h_mm",
        "basin_rain_384h_mm",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in member_rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    print(
        json.dumps(
            {
                "status": output["status"],
                "member_count": len(member_rows),
                "target_stage_m": target_stage,
                "rain_free_days": base_days,
                "p10": quantiles["p10"],
                "p50": quantiles["p50"],
                "p90": quantiles["p90"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
