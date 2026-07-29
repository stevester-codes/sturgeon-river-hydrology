#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_impacts_v2 import model_rate
from wateroffice_archive_probe import month_chunks, request_chunk, session

ROOT = Path("sturgeon_pipeline_output")
PROJECT = ROOT / "routing" / "forecast_starkey_wse.json"
BASE = ROOT / "calibration" / "calibration.json"
OUT_DEFAULT = Path("output/historical_gauge_analysis/analysis.json")
STATION = "05EA002"
TARGET_Q = 6.77


def season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "fall"
    return "winter"


def fit_rating(frame: pd.DataFrame) -> dict | None:
    if len(frame) < 24:
        return None
    stage = frame.stage_m.to_numpy(float)
    discharge = frame.discharge_m3s.to_numpy(float)
    slope, intercept = np.polyfit(stage, discharge, 1)
    slope = float(slope)
    intercept = float(intercept)
    if slope <= 0:
        return None
    prediction = slope * stage + intercept
    residual = prediction - discharge
    return {
        "n": int(len(frame)),
        "slope_m3s_per_m": slope,
        "intercept_m3s": intercept,
        "r2": float(np.corrcoef(stage, discharge)[0, 1] ** 2),
        "rmse_m3s": float(np.sqrt(np.mean(residual**2))),
        "stage_range_m": [float(np.min(stage)), float(np.max(stage))],
        "discharge_range_m3s": [float(np.min(discharge)), float(np.max(discharge))],
        "target_stage_m": (TARGET_Q - intercept) / slope,
    }


def summarize_target(frame: pd.DataFrame, tolerance: float) -> dict:
    rows = frame[(frame.discharge_m3s - TARGET_Q).abs() <= tolerance].copy()
    if rows.empty:
        return {"n": 0, "tolerance_m3s": tolerance}
    stages = rows.stage_m.to_numpy(float)
    result = {
        "n": int(len(rows)),
        "tolerance_m3s": tolerance,
        "discharge_range_m3s": [
            float(rows.discharge_m3s.min()),
            float(rows.discharge_m3s.max()),
        ],
        "stage_min_m": float(np.min(stages)),
        "stage_p10_m": float(np.quantile(stages, 0.10)),
        "stage_p25_m": float(np.quantile(stages, 0.25)),
        "stage_median_m": float(np.median(stages)),
        "stage_p75_m": float(np.quantile(stages, 0.75)),
        "stage_p90_m": float(np.quantile(stages, 0.90)),
        "stage_max_m": float(np.max(stages)),
        "stage_standard_deviation_m": float(np.std(stages, ddof=0)),
        "first_utc": rows.index.min().isoformat(),
        "last_utc": rows.index.max().isoformat(),
    }
    return result


def group_target_summaries(frame: pd.DataFrame, tolerance: float) -> list[dict]:
    rows = []
    group_fields = [
        ("limb", ["limb"]),
        ("season", ["season"]),
        ("season_limb", ["season", "limb"]),
        ("year_season_limb", ["year", "season", "limb"]),
    ]
    for grouping_name, columns in group_fields:
        grouper = columns[0] if len(columns) == 1 else columns
        for key, group in frame.groupby(grouper):
            key_values = (key,) if len(columns) == 1 else tuple(key)
            summary = summarize_target(group, tolerance)
            summary.update(
                {
                    "grouping": grouping_name,
                    **dict(zip(columns, key_values)),
                }
            )
            rows.append(summary)
    return rows


def fit_recession(frame: pd.DataFrame, value_column: str, rate_column: str) -> dict | None:
    clean = frame.dropna(subset=[value_column, rate_column])
    if len(clean) < 48:
        return None
    x_value = clean[value_column].to_numpy(float)
    y_rate = clean[rate_column].to_numpy(float)
    lo, hi = np.quantile(y_rate, [0.05, 0.95])
    keep = (y_rate >= lo) & (y_rate <= hi)
    x = np.column_stack([np.ones(int(np.sum(keep))), x_value[keep]])
    y = y_rate[keep]
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ coefficients
    residual = prediction - y
    return {
        "n": int(len(y)),
        "intercept_per_day": float(coefficients[0]),
        "coefficient_per_day": float(coefficients[1]),
        "value_range": [float(np.min(x_value[keep])), float(np.max(x_value[keep]))],
        "rate_rmse_per_day": float(np.sqrt(np.mean(residual**2))),
    }


def predict_rate(value: np.ndarray, fit: dict) -> np.ndarray:
    return np.minimum(
        -0.001,
        float(fit["intercept_per_day"])
        + float(fit["coefficient_per_day"]) * value,
    )


def rate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_per_day": float(np.sqrt(np.mean(error**2))),
        "mae_per_day": float(np.mean(np.abs(error))),
        "bias_per_day": float(np.mean(error)),
    }


def project_value(current: float, target: float, fit: dict, max_hours: int = 24 * 120):
    value = float(current)
    hours = 0
    path = []
    while value > target and hours < max_hours:
        rate = min(
            -0.001,
            float(fit["intercept_per_day"])
            + float(fit["coefficient_per_day"]) * value,
        )
        value += rate / 24.0
        hours += 1
        if hours % 24 == 0:
            path.append({"day": hours // 24, "value": value, "rate_per_day": rate})
    return {
        "reached": value <= target,
        "days": hours / 24.0 if value <= target else None,
        "path_daily": path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = (now - pd.DateOffset(months=args.months)).to_pydatetime()
    records = []
    frames = []
    http = session()
    for chunk_start, chunk_end in month_chunks(start, now):
        record, frame = request_chunk(http, STATION, chunk_start, chunk_end)
        records.append(record)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No historical unit-value data were retrieved")

    raw = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date_utc", "parameter_id"], keep="last")
        .sort_values(["date_utc", "parameter_id"])
    )
    pivot = raw.pivot_table(
        index="date_utc", columns="parameter_id", values="value", aggfunc="median"
    ).rename(columns={46: "stage_m", 47: "discharge_m3s"})
    hourly = pivot.sort_index().resample("1h").median().interpolate(limit=2)
    hourly = hourly.dropna(subset=["stage_m", "discharge_m3s"])
    hourly["stage_change_6h_m"] = hourly.stage_m.diff(6)
    hourly["stage_change_24h_m"] = hourly.stage_m.diff(24)
    hourly["q_change_6h_m3s"] = hourly.discharge_m3s.diff(6)
    hourly["q_change_24h_m3s"] = hourly.discharge_m3s.diff(24)
    hourly["stage_rate_m_per_day"] = hourly.stage_m.diff(6) / 6.0 * 24.0
    hourly["q_rate_m3s_per_day"] = hourly.discharge_m3s.diff(6) / 6.0 * 24.0
    hourly["limb"] = np.where(
        hourly.stage_change_6h_m >= 0.003,
        "rising",
        np.where(hourly.stage_change_6h_m <= -0.003, "falling", "approximately_flat"),
    )
    hourly["season"] = [season(timestamp.month) for timestamp in hourly.index]
    hourly["year"] = hourly.index.year

    # A strict gauge-only recession screen: at least 24 h of net decline,
    # current 6 h decline, and no positive 6 h movement greater than 0.003 m
    # during the previous 24 h. This does not prove absence of rain, so all
    # resulting models remain candidates until RDPA is paired.
    recent_max_rise = hourly.stage_change_6h_m.rolling(24, min_periods=24).max()
    recession = hourly[
        (hourly.stage_change_24h_m < -0.01)
        & (hourly.stage_change_6h_m < -0.001)
        & (recent_max_rise <= 0.003)
        & (hourly.stage_rate_m_per_day > -0.30)
        & (hourly.q_rate_m3s_per_day > -20.0)
    ].copy()

    stage_fit = fit_recession(recession, "stage_m", "stage_rate_m_per_day")
    q_fit = fit_recession(recession, "discharge_m3s", "q_rate_m3s_per_day")
    year_holdout = {}
    years = sorted(recession.year.unique()) if not recession.empty else []
    if len(years) >= 2:
        train_years = years[:-1]
        test_year = years[-1]
        train = recession[recession.year.isin(train_years)]
        test = recession[recession.year == test_year]
        train_stage = fit_recession(train, "stage_m", "stage_rate_m_per_day")
        train_q = fit_recession(train, "discharge_m3s", "q_rate_m3s_per_day")
        if train_stage and train_q and len(test) >= 24:
            year_holdout = {
                "train_years": [int(value) for value in train_years],
                "test_year": int(test_year),
                "test_points": int(len(test)),
                "stage_candidate": rate_metrics(
                    test.stage_rate_m_per_day.to_numpy(float),
                    predict_rate(test.stage_m.to_numpy(float), train_stage),
                ),
                "discharge_candidate": rate_metrics(
                    test.q_rate_m3s_per_day.to_numpy(float),
                    predict_rate(test.discharge_m3s.to_numpy(float), train_q),
                ),
            }

    project = json.loads(PROJECT.read_text()) if PROJECT.exists() else {}
    base = json.loads(BASE.read_text()) if BASE.exists() else {}
    current_stage = float(hourly.stage_m.iloc[-1])
    current_q = float(hourly.discharge_m3s.iloc[-1])
    current_limb = str(hourly.limb.iloc[-1])
    current_season = str(hourly.season.iloc[-1])
    operational_target_stage = project.get("construction_threshold", {}).get(
        "equivalent_05EA002_stage_on_current_limb_m"
    )

    tolerance_summaries = {
        "plus_minus_0_25_m3s": {
            "overall": summarize_target(hourly, 0.25),
            "groups": group_target_summaries(hourly, 0.25),
        },
        "plus_minus_0_50_m3s": {
            "overall": summarize_target(hourly, 0.50),
            "groups": group_target_summaries(hourly, 0.50),
        },
    }
    current_group = hourly[
        (hourly.season == current_season) & (hourly.limb == current_limb)
    ]
    current_group_summary = summarize_target(current_group, 0.50)
    current_limb_summary = summarize_target(
        hourly[hourly.limb == current_limb], 0.50
    )
    overall_summary = summarize_target(hourly, 0.50)
    if current_group_summary.get("n", 0) >= 20:
        empirical_target_stage = current_group_summary.get("stage_median_m")
        target_basis = "current_season_and_limb_near_target_observations"
        target_support = current_group_summary
    elif current_limb_summary.get("n", 0) >= 20:
        empirical_target_stage = current_limb_summary.get("stage_median_m")
        target_basis = "current_limb_near_target_observations"
        target_support = current_limb_summary
    else:
        empirical_target_stage = overall_summary.get("stage_median_m")
        target_basis = "all_near_target_observations"
        target_support = overall_summary

    rating_fits = {}
    for key, group in hourly.groupby(["season", "limb"]):
        if len(group) >= 24:
            fit = fit_rating(group)
            if fit:
                rating_fits[f"{key[0]}_{key[1]}"] = fit

    projections = {}
    if stage_fit and empirical_target_stage is not None:
        projections["historical_gauge_only_stage_recession"] = project_value(
            current_stage, float(empirical_target_stage), stage_fit
        )
    if q_fit:
        projections["historical_gauge_only_discharge_recession"] = project_value(
            current_q, TARGET_Q, q_fit
        )
    official_recession = base.get("master_recession", {})
    if empirical_target_stage is not None and official_recession:
        stage = current_stage
        hours = 0
        while stage > float(empirical_target_stage) and hours < 24 * 120:
            stage += model_rate(official_recession, stage) / 24.0
            hours += 1
        projections["official_recession_to_empirical_target_stage"] = {
            "reached": stage <= float(empirical_target_stage),
            "days": hours / 24.0 if stage <= float(empirical_target_stage) else None,
        }

    output = {
        "generated_utc": now.isoformat(),
        "status": "historical_gauge_candidate_ready_for_rdpa_pairing",
        "mode": "shadow_only_no_operational_promotion",
        "station": STATION,
        "requested_months": args.months,
        "retrieval": {
            "chunks": records,
            "hourly_paired_points": int(len(hourly)),
            "first_utc": hourly.index.min().isoformat(),
            "last_utc": hourly.index.max().isoformat(),
            "stage_range_m": [float(hourly.stage_m.min()), float(hourly.stage_m.max())],
            "discharge_range_m3s": [
                float(hourly.discharge_m3s.min()),
                float(hourly.discharge_m3s.max()),
            ],
        },
        "current": {
            "stage_m": current_stage,
            "discharge_m3s": current_q,
            "season": current_season,
            "limb": current_limb,
            "operational_target_stage_m": operational_target_stage,
        },
        "target_stage_empirical": {
            "target_discharge_m3s": TARGET_Q,
            "recommended_candidate_stage_m": empirical_target_stage,
            "basis": target_basis,
            "support": target_support,
            "difference_from_current_operational_target_m": (
                float(empirical_target_stage) - float(operational_target_stage)
                if empirical_target_stage is not None
                and operational_target_stage is not None
                else None
            ),
            "tolerance_summaries": tolerance_summaries,
            "rating_fits_by_season_and_limb": rating_fits,
        },
        "gauge_only_recession_screen": {
            "points": int(len(recession)),
            "stage_fit": stage_fit,
            "discharge_fit": q_fit,
            "year_holdout": year_holdout,
            "projections": projections,
        },
        "promotion_policy": {
            "automatic_promotion_enabled": False,
            "requirements": [
                "pair recession and response periods with archived RDPA to verify dry conditions and rainfall timing",
                "compare event-block hindcasts against the current calibration",
                "verify target-stage stability across season, limb and year",
                "retain the current target and recession model for rollback",
            ],
        },
        "limitations": [
            "WaterOffice discharge is provisional and rating-derived from stage.",
            "Gauge-only recession screening cannot prove that no delayed rainfall response is present.",
            "Seasonal station shutdown intervals are excluded rather than filled.",
            "An empirical stage distribution near Q=6.77 is stronger than a long linear extrapolation but still depends on the provisional WSC rating.",
        ],
    }
    out.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "hourly_points": len(hourly),
                "recession_points": len(recession),
                "current": output["current"],
                "target_stage_empirical": {
                    "stage_m": empirical_target_stage,
                    "basis": target_basis,
                    "support_n": target_support.get("n"),
                    "difference_from_operational_m": output["target_stage_empirical"]["difference_from_current_operational_target_m"],
                },
                "projections": projections,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
