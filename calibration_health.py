#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_impacts_v2 import FEATURES, current_antecedent_rain, training_frame

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
CAL_META = ROOT / "calibration_v2" / "calibration_v2.json"
CAL_EVENTS = ROOT / "calibration_v2" / "event_response_v2.csv"
FORECAST = ROOT / "forecast_v2" / "forecast_impacts_v2.json"
PROBABILITY = ROOT / "forecast_v2" / "project_threshold_ensemble.json"
PROJECT_WSE = ROOT / "routing" / "forecast_starkey_wse.json"
TRANSFER = ROOT / "routing" / "starkey_wse_transfer.json"
HISTORICAL_SELECTION = Path("output/archive_probe/historical_rdpa_model_selection.json")
OUT = ROOT / "diagnostics" / "calibration_health.json"


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def gauge_map(summary: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row.get("station")), str(row.get("metric"))): row
        for row in summary.get("gauges", [])
    }


def select_operational_scenario(forecast: dict) -> dict:
    candidates = [
        row
        for row in forecast.get("deterministic_scenarios", [])
        if str(row.get("model")) == "HRDPS" and row.get("feature_vector")
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: int(row.get("horizon_h", 0)))


def feature_coverage(training: pd.DataFrame, scenario: dict) -> dict:
    vector = scenario.get("feature_vector", {}) if scenario else {}
    rows = []
    outside = 0
    material = 0
    for feature in FEATURES:
        values = pd.to_numeric(training.get(feature, pd.Series(dtype=float)), errors="coerce").dropna()
        current = finite(vector.get(feature))
        if values.empty or current is None:
            rows.append({"feature": feature, "status": "unavailable", "current": current})
            continue
        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=0))
        span = max(maximum - minimum, 1e-9)
        if current < minimum:
            normalized_outside = (minimum - current) / span
        elif current > maximum:
            normalized_outside = (current - maximum) / span
        else:
            normalized_outside = 0.0
        status = "inside_training_range"
        if normalized_outside > 0:
            outside += 1
            status = "outside_training_range"
        if normalized_outside > 0.5:
            material += 1
            status = "material_extrapolation"
        z_score = None if standard_deviation <= 0 else (current - mean) / standard_deviation
        rows.append(
            {
                "feature": feature,
                "current": current,
                "training_min": minimum,
                "training_max": maximum,
                "training_mean": mean,
                "training_standard_deviation": standard_deviation,
                "z_score": z_score,
                "normalized_distance_outside_range": normalized_outside,
                "status": status,
            }
        )
    if material:
        status = "material_extrapolation"
    elif outside >= 2:
        status = "marginal_extrapolation"
    elif outside == 1:
        status = "minor_extrapolation"
    else:
        status = "within_observed_feature_envelope"
    return {
        "status": status,
        "outside_feature_count": outside,
        "materially_outside_feature_count": material,
        "nearest_analogue_distance": finite(
            scenario.get("analog_prediction", {}).get("nearest_distance") if scenario else None
        ),
        "storm_type": scenario.get("storm_type") if scenario else None,
        "horizon_h": scenario.get("horizon_h") if scenario else None,
        "features": rows,
    }


def storage_state(summary: dict) -> dict:
    gauges = gauge_map(summary)

    def value(station: str, metric: str, field: str):
        return finite(gauges.get((station, metric), {}).get(field))

    signals = []
    definitions = [
        ("upper_lake_chain_05EA012", "05EA012", "water_level_m"),
        ("atim_big_lake_05EA011", "05EA011", "water_level_m"),
        ("middle_basin_05EA005", "05EA005", "water_level_m"),
        ("lower_tributary_05EA010", "05EA010", "water_level_m"),
    ]
    elevated = 0
    for name, station, metric in definitions:
        change_24h = value(station, metric, "change_24h")
        change_72h = value(station, metric, "change_72h")
        rising_24h = change_24h is not None and change_24h > 0.01
        elevated_72h = change_72h is not None and change_72h > 0.05
        if rising_24h or elevated_72h:
            elevated += 1
        signals.append(
            {
                "name": name,
                "station": station,
                "change_24h_m": change_24h,
                "change_72h_m": change_72h,
                "rising_24h": rising_24h,
                "higher_than_72h_ago": elevated_72h,
            }
        )
    target_24h = value("05EA002", "water_level_m", "change_24h")
    if elevated >= 3:
        state = "high_residual_storage_signal"
    elif elevated >= 1:
        state = "elevated_residual_storage_signal"
    else:
        state = "low_residual_storage_signal"
    return {
        "status": state,
        "interpretation": (
            "This is a gauge-trend proxy for water still stored in upstream lakes, wetlands, "
            "tributaries and floodplain. It is diagnostic only and is not yet an explicit routed storage state."
        ),
        "target_change_24h_m": target_24h,
        "antecedent_rain_mm": {
            "24h": current_antecedent_rain(24),
            "72h": current_antecedent_rain(72),
            "168h": current_antecedent_rain(168),
        },
        "elevated_signal_count": elevated,
        "signals": signals,
    }


def rating_support(project: dict) -> dict:
    fit = project.get("current_event_rating_fit", {})
    stage_range = fit.get("stage_range_m", [None, None])
    minimum = finite(stage_range[0]) if len(stage_range) > 0 else None
    maximum = finite(stage_range[1]) if len(stage_range) > 1 else None
    target_stage = finite(
        project.get("construction_threshold", {}).get(
            "equivalent_05EA002_stage_on_current_limb_m"
        )
    )
    if minimum is None or maximum is None or target_stage is None:
        return {"status": "unavailable"}
    span = max(maximum - minimum, 1e-9)
    below = max(0.0, minimum - target_stage)
    above = max(0.0, target_stage - maximum)
    extrapolation = max(below, above)
    span_multiple = extrapolation / span
    if extrapolation == 0:
        status = "target_inside_current_limb_fit_range"
    elif span_multiple <= 0.5:
        status = "mild_target_extrapolation"
    else:
        status = "material_target_extrapolation"
    return {
        "status": status,
        "fit_r2": finite(fit.get("r2")),
        "fit_rmse_m3s": finite(fit.get("rmse_m3s")),
        "fit_hourly_points": fit.get("n_hourly_points"),
        "observed_stage_range_m": [minimum, maximum],
        "target_stage_m": target_stage,
        "extrapolation_distance_m": extrapolation,
        "extrapolation_as_observed_span": span_multiple,
        "warning": (
            "Reported WSC discharge is itself rating-derived; a tight stage-discharge fit is an operational mapping, "
            "not independent proof that the provisional discharge rating is physically correct."
        ),
    }


def transfer_support(transfer: dict, project: dict) -> dict:
    points = transfer.get("site_constraints", [])
    discharges = sorted(
        finite(point.get("discharge_m3s"))
        for point in points
        if finite(point.get("discharge_m3s")) is not None
    )
    design_points = [
        point for point in points if point.get("return_period_years") is not None
    ]
    current_q = finite(project.get("current", {}).get("observed_discharge_05EA002_m3s"))
    target_q = finite(
        project.get("construction_threshold", {}).get("calibrated_target_discharge_m3s")
    )
    first_design_q = min(
        (finite(point.get("discharge_m3s")) for point in design_points),
        default=None,
    )
    low_flow_gap = (
        first_design_q - target_q
        if first_design_q is not None and target_q is not None
        else None
    )
    if len(design_points) >= 10 and target_q is not None:
        status = "approximate_low_flow_anchor_plus_complete_design_profile"
        score = 8.0
    elif len(design_points) >= 3:
        status = "approximate_low_flow_anchor_plus_partial_design_profile"
        score = 6.0
    else:
        status = "sparse_project_transfer_support"
        score = 3.0
    return {
        "status": status,
        "score_out_of_10": score,
        "current_discharge_m3s": current_q,
        "target_discharge_m3s": target_q,
        "constraint_discharges_m3s": discharges,
        "design_profile_point_count": len(design_points),
        "first_design_discharge_m3s": first_design_q,
        "low_flow_anchor_to_first_design_gap_m3s": low_flow_gap,
        "working_wse_uncertainty_m": finite(
            transfer.get("transfer", {}).get("uncertainty_m"), 0.15
        ),
        "largest_constraint_gap_m3s": max(
            (b - a for a, b in zip(discharges, discharges[1:])), default=None
        ),
        "interpretation": (
            "The high-flow RS18883 transfer is now constrained by the complete "
            "2- to 1,000-year design profile. Remaining transfer uncertainty is "
            "concentrated in the reconstructed 6.77 m3/s low-flow anchor and the "
            "segment to the 14 m3/s two-year point."
        ),
    }


def historical_recession_validation() -> dict:
    if not HISTORICAL_SELECTION.exists():
        return {
            "status": "unavailable",
            "score_out_of_9": 0.0,
            "reason": "historical RDPA model-selection output is missing",
        }
    data = load_json(HISTORICAL_SELECTION)
    preferred = data.get("preferred_screened_candidate", {})
    cv = preferred.get("event_block_cross_validation", {}).get("aggregate", {})
    coverage = finite(data.get("rdpa_coverage_fraction"), 0.0)
    points = int(preferred.get("points") or 0)
    events = int(preferred.get("events") or 0)
    skill = finite(preferred.get("skill_improvement_vs_gauge_only_pct"), 0.0)
    rmse = finite(preferred.get("event_block_rmse_per_day"))
    score = 0.0
    score += 2.0 if coverage >= 0.90 else (1.0 if coverage >= 0.50 else 0.0)
    score += 1.0 if points >= 200 else 0.0
    score += 2.0 if events >= 8 else (1.0 if events >= 3 else 0.0)
    score += 2.0 if cv and rmse is not None else 0.0
    score += 2.0 if skill >= 5.0 else (1.0 if skill >= 0.0 and rmse is not None else 0.0)
    status = (
        "screened_event_block_validation_available"
        if score >= 6.0
        else "limited_historical_validation"
    )
    return {
        "status": status,
        "score_out_of_9": score,
        "preferred_model": preferred.get("name"),
        "rdpa_coverage_fraction": coverage,
        "screened_points": points,
        "independent_event_blocks": events,
        "rain_free_days_to_6_77_m3s": finite(
            preferred.get("rain_free_days_to_6_77_m3s")
        ),
        "event_block_rmse_per_day": rmse,
        "event_block_mae_per_day": finite(cv.get("mae_per_day")),
        "skill_improvement_vs_gauge_only_pct": skill,
        "operational_use": data.get("operational_use", {}),
        "promotion_recommendation": data.get("promotion_recommendation", {}),
        "interpretation": (
            "This is independent precipitation-screened validation of the direct-"
            "discharge recession timing. It improves confidence in schedule "
            "sensitivity, but remains shadow-only because skill gain and event "
            "diversity are not sufficient for promotion."
        ),
    }


def diagnostic_score(
    uncensored: int,
    recoveries: int,
    storm_types: int,
    coverage_status: str,
    rating_status: str,
    member_count: int,
    transfer_score: float,
    historical_score: float,
) -> dict:
    # Version 2 rebalances the same 100-point diagnostic to recognize the
    # complete RS18883 design profile and independent historical event-block
    # validation. It remains an engineering evidence score, not probability.
    components = {
        "uncensored_peak_events": min(20.0, uncensored * 7.0),
        "complete_recoveries": min(15.0, recoveries * 7.5),
        "storm_type_diversity": min(8.0, storm_types * 2.0),
        "forecast_feature_coverage": {
            "within_observed_feature_envelope": 12.0,
            "minor_extrapolation": 9.0,
            "marginal_extrapolation": 6.0,
            "material_extrapolation": 2.0,
        }.get(coverage_status, 0.0),
        "current_rating_support": {
            "target_inside_current_limb_fit_range": 10.0,
            "mild_target_extrapolation": 6.0,
            "material_target_extrapolation": 2.0,
        }.get(rating_status, 0.0),
        "meteorological_ensemble": 8.0 if member_count >= 20 else (4.0 if member_count else 0.0),
        "project_transfer_support": min(10.0, max(0.0, transfer_score)),
        "historical_recession_validation": min(9.0, max(0.0, historical_score)),
        "operational_integrity_controls": 8.0,
    }
    total = float(sum(components.values()))
    if total < 40:
        tier = "low"
    elif total < 65:
        tier = "low_to_moderate"
    elif total < 80:
        tier = "moderate"
    else:
        tier = "high"
    return {
        "score_version": 2,
        "score_out_of_100": total,
        "tier": tier,
        "components": components,
        "warning": (
            "This is a transparent engineering diagnostic score, not a statistically calibrated probability "
            "that a forecast date will be correct. Version 2 recognizes full-profile transfer support and "
            "historical event-block validation, so it is not directly comparable point-for-point with version 1."
        ),
    }


def main() -> None:
    summary = load_json(SUMMARY)
    calibration = load_json(CAL_META)
    forecast = load_json(FORECAST)
    project = load_json(PROJECT_WSE)
    transfer = load_json(TRANSFER)
    probability = load_json(PROBABILITY) if PROBABILITY.exists() else {}
    events = pd.read_csv(CAL_EVENTS)
    _, training = training_frame(events)

    scenario = select_operational_scenario(forecast)
    coverage = feature_coverage(training, scenario)
    storage = storage_state(summary)
    rating = rating_support(project)
    transfer_health = transfer_support(transfer, project)
    historical_health = historical_recession_validation()

    uncensored = int(calibration.get("uncensored_peak_events", len(training)))
    recoveries = int(calibration.get("complete_recovery_events", 0))
    storm_types = len(calibration.get("storm_type_counts", {}))
    member_count = int(probability.get("geps", {}).get("member_count", 0))
    score = diagnostic_score(
        uncensored,
        recoveries,
        storm_types,
        coverage.get("status", "unavailable"),
        rating.get("status", "unavailable"),
        member_count,
        finite(transfer_health.get("score_out_of_10"), 0.0),
        finite(historical_health.get("score_out_of_9"), 0.0),
    )

    crossing = probability.get("crossing_distribution", {})
    actions = [
        {
            "priority": 1,
            "action": "Survey at least one concurrent RS18883 water level near 650.20 m tied to CGVD28.",
            "reason": "The complete design profile improves the transfer above 14 m3/s, but the operational low-flow anchor remains reconstructed.",
        },
        {
            "priority": 2,
            "action": "Obtain at least one complete post-storm recovery without intervening rainfall.",
            "reason": "Zero complete recoveries leave live rainfall-delay estimates weakly constrained.",
        },
        {
            "priority": 3,
            "action": "Accumulate additional independent summer falling-limb dry recession blocks.",
            "reason": "The precipitation-screened direct-Q model has event-block validation, but only eight independent blocks and 1.87 percent skill gain.",
        },
        {
            "priority": 4,
            "action": "Assimilate new rainfall-response events only after censoring and backtest checks pass.",
            "reason": "Automatic promotion with only two clean response events could make the model less reliable rather than more reliable.",
        },
    ]

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_no_automatic_recalibration",
        "overall": score,
        "calibration_sample": {
            "all_event_constraints": int(calibration.get("event_count", len(events))),
            "uncensored_peak_training_events": uncensored,
            "complete_recovery_events": recoveries,
            "storm_type_counts": calibration.get("storm_type_counts", {}),
            "cross_validation_available": len(training) >= 3,
        },
        "current_forecast_feature_coverage": coverage,
        "hydrologic_memory_and_storage_proxies": storage,
        "current_limb_rating_support": rating,
        "project_wse_transfer_support": transfer_health,
        "historical_recession_validation": historical_health,
        "ensemble_date_spread": {
            "member_count": member_count,
            "standard_deviation_days": finite(crossing.get("standard_deviation_days")),
            "earliest": crossing.get("earliest"),
            "latest": crossing.get("latest"),
            "quantiles": crossing.get("quantiles"),
            "interpretation": "GEPS spread measures meteorological uncertainty; hydrologic-response and project-transfer errors are additional.",
        },
        "controlled_assimilation": {
            "automatic_promotion_enabled": False,
            "candidate_requirements": [
                "complete and quality-controlled observed precipitation coverage",
                "clearly observed river-response peak",
                "no later storm censoring the response window",
                "preferably a complete recovery",
                "candidate model does not worsen leave-one-event-out or hindcast performance",
                "previous calibration retained for rollback",
            ],
            "current_reason_not_automatic": "Only two uncensored peaks and zero complete recoveries are presently available.",
        },
        "priority_actions": actions,
        "limitations": [
            "Storage pressure is inferred from recent rainfall and upstream gauge trends, not simulated as physical reservoir volumes.",
            "The current event stage-discharge fit is based on provisional WSC stage and rating-derived discharge.",
            "The diagnostic score is intentionally transparent but is not a formal confidence interval.",
            "A model can pass operational integrity checks while still having limited scientific calibration confidence.",
            "The full design profile materially improves RS18883 interpolation above 14 m3/s but does not replace a surveyed low-flow project observation.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "score": score["score_out_of_100"],
                "tier": score["tier"],
                "feature_coverage": coverage["status"],
                "rating_support": rating["status"],
                "transfer_support": transfer_health["status"],
                "historical_validation": historical_health["status"],
                "storage_signal": storage["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
