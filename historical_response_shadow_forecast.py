#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORECAST_DEFAULT = Path("output/latest/forecast_v2/forecast_impacts_v2.json")
MODEL_DEFAULT = Path(
    "output/historical_event_backfill/historical_censored_response_model.json"
)
OUT_DEFAULT = Path(
    "output/historical_event_backfill/current_historical_response_shadow.json"
)


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def hrdps_48h(forecast: dict) -> dict:
    candidates = [
        row
        for row in forecast.get("deterministic_scenarios", [])
        if str(row.get("model")) == "HRDPS"
        and int(row.get("horizon_h", 0)) == 48
        and bool(row.get("complete_horizon"))
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: str(row.get("run_time_utc") or ""),
    )


def transformed_feature(name: str, vector: dict) -> float | None:
    if name == "log_basin_mm":
        value = finite(vector.get("basin_mm"))
        return math.log1p(max(0.0, value)) if value is not None else None
    if name == "log_antecedent_168h_mm":
        value = finite(vector.get("antecedent_168h_mm"))
        return math.log1p(max(0.0, value)) if value is not None else None
    if name == "pre_stage_m":
        return finite(vector.get("pre_stage_m"))
    if name == "lower_ratio":
        return finite(vector.get("lower_ratio"))
    if name == "upper_ratio":
        return finite(vector.get("upper_ratio"))
    if name == "log_duration_h":
        value = finite(vector.get("duration_h"))
        return math.log1p(max(0.0, value)) if value is not None else None
    if name == "basin_pct_gt_10mm_fraction":
        value = finite(vector.get("basin_pct_gt_10mm"))
        return min(1.0, max(0.0, value / 100.0)) if value is not None else None
    return finite(vector.get(name))


def predict_days(model: dict, vector: dict) -> tuple[float | None, dict]:
    features = model.get("features", [])
    means = model.get("feature_means", {})
    scales = model.get("feature_scales", {})
    coefficients = model.get("standardized_coefficients", {})
    intercept = finite(model.get("intercept_log1p_days"))
    if intercept is None or not features:
        return None, {"status": "model_parameters_unavailable"}
    values = {}
    standardized = {}
    prediction_log = intercept
    for name in features:
        value = transformed_feature(name, vector)
        mean = finite(means.get(name))
        scale = finite(scales.get(name))
        coefficient = finite(coefficients.get(name))
        if value is None or mean is None or scale is None or coefficient is None or scale <= 0:
            return None, {
                "status": "feature_unavailable",
                "missing_feature": name,
            }
        z_value = (value - mean) / scale
        values[name] = value
        standardized[name] = z_value
        prediction_log += coefficient * z_value
    return max(0.0, float(np.expm1(prediction_log))), {
        "status": "prediction_available",
        "transformed_features": values,
        "standardized_features": standardized,
        "prediction_log1p_days": prediction_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", default=str(FORECAST_DEFAULT))
    parser.add_argument("--model", default=str(MODEL_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    forecast = load(Path(args.forecast))
    historical = load(Path(args.model))
    row = hrdps_48h(forecast)
    if not row:
        result = {
            "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "short_range_forecast_unavailable",
            "mode": "shadow_only_no_effect_on_operational_forecast",
            "historical_model_status": historical.get("status"),
            "historical_model_promotion_screen": historical.get("promotion_screen"),
        }
    else:
        basin_mm = finite(row.get("basin_mm"), 0.0)
        vector = dict(row.get("feature_vector") or {})
        if not vector:
            vector = {
                "basin_mm": basin_mm,
                "antecedent_168h_mm": finite(row.get("antecedent_168h_mm")),
                "pre_stage_m": finite(row.get("pre_stage_m")),
                "duration_h": finite(
                    row.get("forecast_timing", {}).get("duration_h")
                ),
                "basin_pct_gt_10mm": finite(row.get("basin_pct_gt_10mm")),
            }
        preferred = historical.get("preferred_candidate", {})
        full_model = preferred.get("full_model", {})
        analogue = finite(row.get("analog_prediction", {}).get("days_lost"))
        negligible = basin_mm < 0.10
        official_response = analogue if analogue is not None else (0.0 if negligible else None)
        prediction = 0.0 if negligible else None
        prediction_detail = {
            "status": "valid_negligible_short_range_qpf",
            "reason": "Complete HRDPS 48-hour basin rainfall is below 0.10 mm.",
        }
        if not negligible:
            prediction, prediction_detail = predict_days(full_model, vector)
        rmse = finite(
            preferred.get("exact_leave_one_event_out", {}).get("rmse_days")
        )
        difference = (
            prediction - official_response
            if prediction is not None and official_response is not None
            else None
        )
        result = {
            "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": (
                "historical_response_shadow_available"
                if prediction is not None
                else "historical_response_shadow_feature_unavailable"
            ),
            "mode": "shadow_only_no_effect_on_operational_forecast",
            "hrdps": {
                "run_time_utc": row.get("run_time_utc"),
                "horizon_h": row.get("horizon_h"),
                "complete_horizon": row.get("complete_horizon"),
                "basin_mm": basin_mm,
                "storm_type": row.get("storm_type"),
                "feature_vector": vector,
            },
            "official_analogue_response_days_lost": official_response,
            "historical_censored_model_days_lost": prediction,
            "historical_model_minus_official_days": difference,
            "historical_model_rmse_sensitivity_days": rmse,
            "historical_model_upper_rmse_sensitivity_days": (
                prediction + rmse
                if prediction is not None and rmse is not None
                else None
            ),
            "prediction_detail": prediction_detail,
            "historical_model": {
                "status": historical.get("status"),
                "target": historical.get("target"),
                "resolved_peak_events": historical.get("resolved_peak_events"),
                "censored_lower_bound_events": historical.get("censored_lower_bound_events"),
                "preferred_feature_set": preferred.get("feature_set"),
                "preferred_exact_rmse_days": rmse,
                "constraint_satisfaction_pct": preferred.get("censored_constraints", {}).get("constraint_satisfaction_pct"),
                "combined_screen_passed": historical.get("preferred_candidate_meets_combined_screen"),
                "promotion_screen": historical.get("promotion_screen"),
            },
            "interpretation": (
                "This is an independent historical response comparison using all resolved peaks and censored lower bounds. "
                "It does not alter the official forecast. A difference is diagnostic evidence, not a calibrated probability."
            ),
            "limitations": [
                "The historical candidate is amount-only because 10 km spatial features did not improve leave-one-event-out skill.",
                "The candidate satisfies only about 65 percent of censored lower bounds and has not passed promotion gates.",
                "The RMSE-based upper sensitivity is not a p90 confidence bound.",
            ],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
