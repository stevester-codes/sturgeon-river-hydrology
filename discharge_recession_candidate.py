#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_impacts_v2 import model_rate

ROOT = Path("sturgeon_pipeline_output")
GAUGE = ROOT / "raw" / "wateroffice" / "05EA002.csv"
PRECIP = ROOT / "processed" / "watershed_precip_06h.csv"
BASE = ROOT / "calibration" / "calibration.json"
PROJECT = ROOT / "routing" / "forecast_starkey_wse.json"
OUT = ROOT / "diagnostics" / "discharge_recession_candidate.json"
TIME_RE = re.compile(r"_(\d{10})_000_\d{2}\.dbf$")
TARGET_Q = 6.77
MAX_HOURS = 24 * 120


def parse_gauge() -> pd.DataFrame:
    frame = pd.read_csv(GAUGE, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = next(column for column in frame.columns if column.lower() == "date")
    parameter_column = next(
        column
        for column in frame.columns
        if "parameter" in column.lower() or "paramètre" in column.lower()
    )
    value_column = next(
        column
        for column in frame.columns
        if "value" in column.lower() or "valeur" in column.lower()
    )
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    frame[parameter_column] = pd.to_numeric(frame[parameter_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, parameter_column, value_column])
    pivot = frame.pivot_table(
        index=date_column,
        columns=parameter_column,
        values=value_column,
        aggfunc="median",
    ).rename(columns={46: "stage_m", 47: "discharge_m3s"})
    return pivot.sort_index().resample("1h").median().interpolate(limit=2)


def parse_precip(index: pd.DatetimeIndex) -> pd.Series:
    frame = pd.read_csv(PRECIP)
    if frame.empty:
        return pd.Series(0.0, index=index)

    def valid_time(filename: str):
        match = TIME_RE.search(str(filename))
        if not match:
            return pd.NaT
        return pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)

    frame["valid_utc"] = frame["_source_file"].map(valid_time)
    frame["PR_mm"] = pd.to_numeric(frame["PR_mm"], errors="coerce")
    target = frame[
        (frame.Station.astype(str) == "05EA002")
        & frame.valid_utc.notna()
        & frame.PR_mm.notna()
    ]
    return (
        target.groupby("valid_utc").PR_mm.mean().sort_index().reindex(index, fill_value=0.0)
    )


def fit_linear_rate(frame: pd.DataFrame) -> dict:
    q = frame.discharge_m3s.to_numpy(float)
    rate = frame.q_rate_m3s_per_day.to_numpy(float)
    x = np.column_stack([np.ones(len(frame)), q])
    coefficients, *_ = np.linalg.lstsq(x, rate, rcond=None)
    prediction = x @ coefficients
    residuals = prediction - rate
    return {
        "type": "linear_rate_in_discharge",
        "intercept_m3s_per_day": float(coefficients[0]),
        "coefficient_per_day": float(coefficients[1]),
        "n": int(len(frame)),
        "discharge_range_m3s": [float(np.min(q)), float(np.max(q))],
        "rate_rmse_m3s_per_day": float(np.sqrt(np.mean(residuals**2))),
    }


def rate_from_q(discharge: float, fit: dict) -> float:
    return min(
        -0.001,
        float(fit["intercept_m3s_per_day"])
        + float(fit["coefficient_per_day"]) * float(discharge),
    )


def stage_to_q(stage: float, rating: dict) -> float:
    return max(
        0.0,
        float(rating["slope_m3s_per_m"]) * float(stage)
        + float(rating["intercept_m3s"]),
    )


def official_change_6h(stage: float, recession: dict, rating: dict) -> float:
    future_stage = float(stage)
    for _ in range(6):
        future_stage += model_rate(recession, future_stage) / 24.0
    return stage_to_q(future_stage, rating) - stage_to_q(stage, rating)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_m3s": float(np.sqrt(np.mean(error**2))),
        "mae_m3s": float(np.mean(np.abs(error))),
        "bias_m3s": float(np.mean(error)),
    }


def project_q(current_q: float, target_q: float, fit: dict) -> dict:
    q = float(current_q)
    path = []
    hours = 0
    while q > target_q and hours < MAX_HOURS:
        q += rate_from_q(q, fit) / 24.0
        q = max(0.0, q)
        hours += 1
        if hours % 24 == 0:
            path.append(
                {
                    "day": hours // 24,
                    "discharge_m3s": q,
                    "rate_m3s_per_day": rate_from_q(q, fit),
                }
            )
    return {
        "reached": q <= target_q,
        "hours": hours if q <= target_q else None,
        "days": hours / 24.0 if q <= target_q else None,
        "path_daily": path,
    }


def main() -> None:
    for path in (GAUGE, PRECIP, BASE, PROJECT):
        if not path.exists():
            raise FileNotFoundError(path)

    generated = datetime.now(timezone.utc)
    gauge = parse_gauge().dropna(subset=["stage_m", "discharge_m3s"])
    rain = parse_precip(gauge.index)
    gauge["rain_48h_mm"] = rain.rolling(48, min_periods=1).sum()
    gauge["q_change_6h_m3s"] = gauge.discharge_m3s.shift(-6) - gauge.discharge_m3s
    gauge["q_rate_m3s_per_day"] = gauge.discharge_m3s.diff(6) / 6.0 * 24.0
    gauge["q_change_previous_6h_m3s"] = gauge.discharge_m3s.diff(6)

    dry = gauge[
        (gauge.rain_48h_mm <= 0.75)
        & (gauge.q_change_previous_6h_m3s < -0.01)
        & (gauge.q_rate_m3s_per_day > -10.0)
        & gauge.q_change_6h_m3s.notna()
        & gauge.q_rate_m3s_per_day.notna()
    ].copy()
    output = {
        "generated_utc": generated.isoformat(),
        "status": "insufficient_dry_discharge_observations",
        "mode": "shadow_only_no_effect_on_operational_forecast",
        "target_discharge_m3s": TARGET_Q,
        "dry_hourly_points": int(len(dry)),
        "data_range_utc": [gauge.index.min().isoformat(), gauge.index.max().isoformat()],
        "candidate_fit": None,
        "holdout": {},
        "current_projection": {},
        "promotion_recommendation": {
            "recommended": False,
            "reason": "Insufficient dry discharge observations.",
        },
        "limitations": [
            "WSC discharge is provisional and rating-derived from stage rather than independently measured each hour.",
            "The rolling operational dataset is short and the 6.77 m3/s target may lie below the observed fitting range.",
            "This cross-check excludes forecast rainfall and cannot replace the full rainfall-response model.",
        ],
    }

    if len(dry) >= 48:
        split = max(24, int(len(dry) * 0.70))
        split = min(split, len(dry) - 18)
        train = dry.iloc[:split]
        test = dry.iloc[split:]
        fit = fit_linear_rate(train)
        base = json.loads(BASE.read_text())
        project = json.loads(PROJECT.read_text())
        recession = base.get("master_recession", {})
        rating = project.get("current_event_rating_fit", {})

        observed = test.q_change_6h_m3s.to_numpy(float)
        candidate = np.asarray(
            [rate_from_q(value, fit) * 6.0 / 24.0 for value in test.discharge_m3s],
            dtype=float,
        )
        official = np.asarray(
            [
                official_change_6h(stage, recession, rating)
                for stage in test.stage_m.to_numpy(float)
            ],
            dtype=float,
        )
        candidate_metrics = metrics(observed, candidate)
        official_metrics = metrics(observed, official)
        rmse_improvement = 1.0 - candidate_metrics["rmse_m3s"] / official_metrics["rmse_m3s"]

        current_q = float(gauge.discharge_m3s.iloc[-1])
        projection = project_q(current_q, TARGET_Q, fit)
        official_target_stage = project.get("construction_threshold", {}).get(
            "equivalent_05EA002_stage_on_current_limb_m"
        )
        official_rain_free_days = None
        if official_target_stage is not None:
            stage = float(gauge.stage_m.iloc[-1])
            hours = 0
            while stage > float(official_target_stage) and hours < MAX_HOURS:
                stage += model_rate(recession, stage) / 24.0
                hours += 1
            if stage <= float(official_target_stage):
                official_rain_free_days = hours / 24.0

        target_outside = TARGET_Q < fit["discharge_range_m3s"][0] or TARGET_Q > fit["discharge_range_m3s"][1]
        improvement_pass = rmse_improvement >= 0.10
        bias_pass = abs(candidate_metrics["bias_m3s"]) <= max(
            0.05, abs(official_metrics["bias_m3s"])
        )
        recommended = bool(
            improvement_pass and bias_pass and not target_outside and projection["reached"]
        )
        output.update(
            {
                "status": (
                    "candidate_ready_for_extended_historical_testing"
                    if improvement_pass and bias_pass
                    else "not_ready_for_extended_historical_testing"
                ),
                "candidate_fit": fit,
                "holdout": {
                    "train_points": int(len(train)),
                    "test_points": int(len(test)),
                    "official_stage_recession_then_rating": official_metrics,
                    "direct_discharge_candidate": candidate_metrics,
                    "rmse_improvement_fraction": float(rmse_improvement),
                    "minimum_improvement_required": 0.10,
                },
                "current_projection": {
                    "current_discharge_m3s": current_q,
                    "candidate_rain_free": projection,
                    "official_stage_based_rain_free_days": official_rain_free_days,
                    "candidate_minus_official_days": (
                        projection["days"] - official_rain_free_days
                        if projection["days"] is not None
                        and official_rain_free_days is not None
                        else None
                    ),
                    "target_outside_candidate_fit_range": target_outside,
                },
                "promotion_recommendation": {
                    "recommended": recommended,
                    "automatic_promotion_enabled": False,
                    "holdout_improvement_check": improvement_pass,
                    "bias_check": bias_pass,
                    "target_inside_fit_range_check": not target_outside,
                    "reason": (
                        "Candidate passes current screening but still requires 18-month event-block hindcasting and rainfall integration."
                        if recommended
                        else "One or more holdout, bias, target-range or projection checks are not met."
                    ),
                },
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "dry_hourly_points": output["dry_hourly_points"],
                "holdout": output.get("holdout"),
                "current_projection": output.get("current_projection"),
                "promotion_recommended": output["promotion_recommendation"]["recommended"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
