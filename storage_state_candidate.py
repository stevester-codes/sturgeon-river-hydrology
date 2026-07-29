#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
RAW = ROOT / "raw" / "wateroffice"
PRECIP = ROOT / "processed" / "watershed_precip_06h.csv"
BASE = ROOT / "calibration" / "calibration.json"
OUT = ROOT / "diagnostics" / "storage_state_candidate.json"
TARGET = "05EA002"
STATIONS = ["05EA002", "05EA005", "05EA010", "05EA011", "05EA012"]
TIME_RE = re.compile(r"_(\d{10})_000_\d{2}\.dbf$")
FORECAST_HOURS = 6
RIDGE_LAMBDA = 4.0


def parse_gauge(station: str) -> pd.DataFrame:
    path = RAW / f"{station}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
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
    ).rename(columns={46: "stage_m", 47: "flow_m3s"})
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
        (frame.Station.astype(str) == TARGET)
        & frame.valid_utc.notna()
        & frame.PR_mm.notna()
    ]
    six_hour = target.groupby("valid_utc").PR_mm.mean().sort_index()
    # HRDPA six-hour totals are represented as impulses at each valid time.
    # Rolling hourly windows therefore sum each independent accumulation once.
    return six_hour.reindex(index, fill_value=0.0)


def recession_rate(model: dict, stage: pd.Series | float):
    intercept = float(model.get("intercept_m_per_day", -0.03))
    coefficient = float(model.get("stage_coefficient_per_day", -0.007))
    values = intercept + coefficient * stage
    if isinstance(values, pd.Series):
        return values.clip(upper=-0.001)
    return min(-0.001, float(values))


def build_dataset() -> tuple[pd.DataFrame, dict]:
    gauges = {station: parse_gauge(station) for station in STATIONS}
    target = gauges[TARGET].stage_m.dropna()
    common_index = target.index
    frame = pd.DataFrame(index=common_index)
    frame["target_stage_m"] = target
    frame["target_change_6h_m"] = target.shift(-FORECAST_HOURS) - target
    frame["target_change_24h_m"] = target - target.shift(24)

    for station in STATIONS[1:]:
        series = gauges[station].stage_m.reindex(common_index).interpolate(limit=2)
        frame[f"{station}_change_6h_m"] = series - series.shift(6)
        frame[f"{station}_change_24h_m"] = series - series.shift(24)

    rain = parse_precip(common_index)
    frame["rain_24h_mm"] = rain.rolling(24, min_periods=1).sum()
    frame["rain_72h_mm"] = rain.rolling(72, min_periods=1).sum()
    frame["rain_168h_mm"] = rain.rolling(168, min_periods=1).sum()

    # Exponential memory states approximate fast runoff, tributary/wetland
    # storage and slower lake/floodplain storage without claiming physical
    # reservoir volumes.
    for half_life_h in (12, 36, 96, 240):
        alpha = 1.0 - np.exp(np.log(0.5) / half_life_h)
        frame[f"rain_memory_{half_life_h}h_mm"] = rain.ewm(
            alpha=alpha, adjust=False
        ).mean() * 6.0

    base = json.loads(BASE.read_text())
    model = base.get("master_recession", {})
    frame["baseline_change_6h_m"] = (
        recession_rate(model, frame.target_stage_m) * FORECAST_HOURS / 24.0
    )
    frame["residual_change_6h_m"] = (
        frame.target_change_6h_m - frame.baseline_change_6h_m
    )
    metadata = {"recession_model": model, "gauges": gauges}
    return frame, metadata


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    identity = np.eye(x.shape[1], dtype=float)
    identity[0, 0] = 0.0  # do not penalize intercept
    return np.linalg.solve(x.T @ x + penalty * identity, x.T @ y)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    residual = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "mae_m": float(np.mean(np.abs(residual))),
        "bias_m": float(np.mean(residual)),
        "correlation": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if len(observed) >= 3
            and np.std(observed) > 0
            and np.std(predicted) > 0
            else None
        ),
    }


def main() -> None:
    for path in [BASE, PRECIP, *[RAW / f"{station}.csv" for station in STATIONS]]:
        if not path.exists():
            raise FileNotFoundError(path)

    frame, metadata = build_dataset()
    feature_columns = [
        "target_stage_m",
        "target_change_24h_m",
        "05EA005_change_6h_m",
        "05EA005_change_24h_m",
        "05EA010_change_6h_m",
        "05EA011_change_6h_m",
        "05EA012_change_6h_m",
        "05EA012_change_24h_m",
        "rain_24h_mm",
        "rain_72h_mm",
        "rain_memory_12h_mm",
        "rain_memory_36h_mm",
        "rain_memory_96h_mm",
        "rain_memory_240h_mm",
    ]
    clean = frame.dropna(
        subset=feature_columns
        + ["target_change_6h_m", "baseline_change_6h_m", "residual_change_6h_m"]
    ).copy()
    if len(clean) < 96:
        raise RuntimeError(
            f"Insufficient complete hourly rows for shadow storage model: {len(clean)}"
        )

    # The final 30 percent is held out chronologically. No future observations
    # are allowed into training, which makes this more meaningful than an
    # in-sample fit while still remaining only a short-period diagnostic.
    split = max(72, int(len(clean) * 0.70))
    split = min(split, len(clean) - 48)
    train = clean.iloc[:split]
    test = clean.iloc[split:]

    x_train_raw = train[feature_columns].to_numpy(float)
    x_test_raw = test[feature_columns].to_numpy(float)
    mean = x_train_raw.mean(axis=0)
    scale = x_train_raw.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_train = np.column_stack(
        [np.ones(len(train)), (x_train_raw - mean) / scale]
    )
    x_test = np.column_stack([np.ones(len(test)), (x_test_raw - mean) / scale])
    y_train = train.residual_change_6h_m.to_numpy(float)
    beta = ridge_fit(x_train, y_train, RIDGE_LAMBDA)

    baseline_prediction = test.baseline_change_6h_m.to_numpy(float)
    candidate_prediction = baseline_prediction + x_test @ beta
    observed = test.target_change_6h_m.to_numpy(float)
    baseline_metrics = metrics(observed, baseline_prediction)
    candidate_metrics = metrics(observed, candidate_prediction)
    rmse_improvement = 1.0 - candidate_metrics["rmse_m"] / baseline_metrics["rmse_m"]
    mae_improvement = 1.0 - candidate_metrics["mae_m"] / baseline_metrics["mae_m"]

    latest = clean.iloc[-1]
    latest_x = np.concatenate(
        [[1.0], (latest[feature_columns].to_numpy(float) - mean) / scale]
    )
    current_baseline = float(latest.baseline_change_6h_m)
    current_correction = float(latest_x @ beta)
    current_candidate = current_baseline + current_correction

    coefficients = [
        {
            "feature": "intercept",
            "standardized_coefficient_m_per_6h": float(beta[0]),
        }
    ]
    coefficients.extend(
        {
            "feature": feature,
            "standardized_coefficient_m_per_6h": float(value),
        }
        for feature, value in zip(feature_columns, beta[1:])
    )
    ranked = sorted(
        coefficients[1:],
        key=lambda item: abs(item["standardized_coefficient_m_per_6h"]),
        reverse=True,
    )

    improves = rmse_improvement >= 0.10 and mae_improvement >= 0.05
    stable_bias = abs(candidate_metrics["bias_m"]) <= max(
        0.005, abs(baseline_metrics["bias_m"])
    )
    candidate_status = (
        "promising_shadow_model"
        if improves and stable_bias
        else "not_ready_for_operational_promotion"
    )
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": candidate_status,
        "mode": "shadow_only_no_effect_on_operational_forecast",
        "method": (
            "A ridge-regression correction to the existing stage-dependent recession curve uses decaying rainfall-memory states and recent upstream gauge movement. "
            "The final 30 percent of observations are held out chronologically."
        ),
        "data": {
            "complete_hourly_rows": int(len(clean)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "start_utc": clean.index.min().isoformat(),
            "end_utc": clean.index.max().isoformat(),
            "forecast_horizon_h": FORECAST_HOURS,
            "ridge_lambda": RIDGE_LAMBDA,
            "features": feature_columns,
        },
        "holdout_performance": {
            "baseline_recession": baseline_metrics,
            "storage_candidate": candidate_metrics,
            "rmse_improvement_fraction": float(rmse_improvement),
            "mae_improvement_fraction": float(mae_improvement),
            "promotion_screen": {
                "minimum_rmse_improvement_fraction": 0.10,
                "minimum_mae_improvement_fraction": 0.05,
                "stable_bias_required": True,
                "performance_screen_passed": bool(improves and stable_bias),
            },
        },
        "current_state": {
            "timestamp_utc": clean.index[-1].isoformat(),
            "baseline_expected_change_next_6h_m": current_baseline,
            "storage_memory_correction_next_6h_m": current_correction,
            "candidate_expected_change_next_6h_m": current_candidate,
            "rain_24h_mm": float(latest.rain_24h_mm),
            "rain_72h_mm": float(latest.rain_72h_mm),
            "rain_168h_mm": float(latest.rain_168h_mm),
            "rain_memory_states_mm": {
                "12h": float(latest.rain_memory_12h_mm),
                "36h": float(latest.rain_memory_36h_mm),
                "96h": float(latest.rain_memory_96h_mm),
                "240h": float(latest.rain_memory_240h_mm),
            },
            "upstream_changes": {
                column: float(latest[column])
                for column in feature_columns
                if "change_" in column and column != "target_change_24h_m"
            },
        },
        "coefficients": coefficients,
        "largest_standardized_effects": ranked[:8],
        "promotion_policy": {
            "automatic_promotion_enabled": False,
            "additional_requirements": [
                "repeat improvement on multiple non-overlapping historical event blocks",
                "test across both rising and falling limbs",
                "verify coefficients and predictions remain physically plausible",
                "demonstrate improved project-threshold crossing hindcasts, not only six-hour stage changes",
                "retain current operational model for rollback and run both models in parallel first",
            ],
        },
        "limitations": [
            "The rolling operational dataset is short and may represent only one hydrologic regime.",
            "Gauge changes are empirical storage proxies, not physical reservoir volumes or routed inflows.",
            "Hourly WSC discharge is not used as a predictive feature because it is rating-derived from stage.",
            "The model predicts six-hour stage change and has not yet been validated for full threshold-crossing dates.",
            "A favorable holdout result is necessary but not sufficient for operational promotion.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": candidate_status,
                "rows": len(clean),
                "baseline_rmse_m": baseline_metrics["rmse_m"],
                "candidate_rmse_m": candidate_metrics["rmse_m"],
                "rmse_improvement_fraction": rmse_improvement,
                "performance_screen_passed": improves and stable_bias,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
