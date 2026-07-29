#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import calibration_health
import hysteresis_diagnostics
import uncertainty_sensitivity

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
STARKEY = ROOT / "routing" / "forecast_starkey_wse.json"
PROBABILISTIC = ROOT / "forecast_v2" / "project_threshold_ensemble.json"
CALIBRATION_HEALTH = ROOT / "diagnostics" / "calibration_health.json"
HYSTERESIS = ROOT / "diagnostics" / "hysteresis_diagnostics.json"
UNCERTAINTY = ROOT / "diagnostics" / "uncertainty_sensitivity.json"
OUT = ROOT / "forecast_v2" / "construction_readiness.json"


def finite(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    for path in (SUMMARY, STARKEY, PROBABILISTIC):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(SUMMARY.read_text())
    starkey = json.loads(STARKEY.read_text())
    probabilistic = json.loads(PROBABILISTIC.read_text())
    if probabilistic.get("status") != "operational_project_threshold_ensemble":
        raise RuntimeError("Project-threshold ensemble output is not operational")

    # Diagnostics are part of the authoritative readiness product. A failure
    # must fail readiness rather than silently publishing an apparently
    # complete forecast.
    hysteresis_diagnostics.main()
    calibration_health.main()
    uncertainty_sensitivity.main()
    for path in (CALIBRATION_HEALTH, HYSTERESIS, UNCERTAINTY):
        if not path.exists():
            raise RuntimeError(f"Required diagnostic output was not generated: {path}")
    health = json.loads(CALIBRATION_HEALTH.read_text())
    hysteresis = json.loads(HYSTERESIS.read_text())
    uncertainty = json.loads(UNCERTAINTY.read_text())

    target = summary.get("target_05EA002", {})
    current = starkey.get("current", {})
    threshold = starkey.get("construction_threshold", {})
    scenarios = starkey.get("scenarios", {})
    distribution = probabilistic.get("crossing_distribution", {})

    schedule = {}
    for name in ("dry", "central", "wet"):
        item = scenarios.get(name, {})
        schedule[name] = {
            "days_to_main_floodplain_exposure": item.get("main_floodplain_crossing_days"),
            "date_utc": item.get("main_floodplain_crossing_date_utc"),
        }

    stage_now = finite(target.get("latest"))
    change_24h = finite(target.get("change_24h"), 0.0)
    limb = starkey.get("hydrograph_limb", "unknown")
    current_wse = finite(current.get("estimated_starkey_wse_m"))
    depth_main = finite(current.get("depth_over_main_floodplain_m"))

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_stage_utc": target.get("latest_utc"),
        "project_location": starkey.get("project_location", {}),
        "hydrograph_limb": limb,
        "current_conditions": {
            "stage_05EA002_m": stage_now,
            "stage_change_24h_m": change_24h,
            "observed_discharge_05EA002_m3s": current.get("observed_discharge_05EA002_m3s"),
            "estimated_starkey_wse_m": current_wse,
            "estimated_project_wse_m": current.get("estimated_project_wse_m", current_wse),
            "estimated_starkey_wse_range_m": current.get("estimated_starkey_wse_range_m"),
            "estimated_depth_over_main_floodplain_m": depth_main,
            "estimated_depth_over_low_pocket_m": current.get("depth_over_low_pocket_m"),
        },
        "authoritative_operational_threshold": {
            "main_floodplain_elevation_m": threshold.get("main_floodplain_wse_m", 650.20),
            "calibrated_05EA002_discharge_m3s": threshold.get("calibrated_target_discharge_m3s", 6.77),
            "equivalent_05EA002_stage_on_current_limb_m": threshold.get("equivalent_05EA002_stage_on_current_limb_m"),
            "basis": "The user observed the approximately 650.20 m project floodplain visible when the rising 05EA002 gauge crossed 1.700 m; the paired reported discharge was 6.77 m3/s. Current-limb stage is recalculated from discharge rather than fixed at 1.70 m.",
        },
        "forecast_main_floodplain_exposure": schedule,
        "probabilistic_exposure": {
            "status": probabilistic.get("status"),
            "member_count": probabilistic.get("geps", {}).get("member_count"),
            "quantiles": distribution.get("quantiles", {}),
            "earliest": distribution.get("earliest"),
            "latest": distribution.get("latest"),
            "mean_days": distribution.get("mean_days"),
            "standard_deviation_days": distribution.get("standard_deviation_days"),
            "probability_exposed_by_date": distribution.get("probability_exposed_by_date", []),
            "interpretation": "These are raw all-member GEPS meteorological probabilities translated to the RS18883 threshold. Analogue-response and site-transfer uncertainty are additional and are not hidden inside the percentages.",
        },
        "risk_adjusted_planning": {
            "status": uncertainty.get("status"),
            "planning_summary": uncertainty.get("planning_summary", {}),
            "scenarios": uncertainty.get("scenarios", {}),
            "uncertainty_components": uncertainty.get("uncertainty_components", {}),
            "interpretation": "This sensitivity envelope combines weather-member spread, rainfall-response error sensitivity and the working plus/minus 0.15 m RS18883 transfer allowance. It is more suitable for protecting the construction schedule than raw weather spread alone, but it is not a formal confidence interval.",
        },
        "model_health": {
            "status": health.get("status"),
            "overall": health.get("overall", {}),
            "calibration_sample": health.get("calibration_sample", {}),
            "forecast_feature_coverage": health.get("current_forecast_feature_coverage", {}),
            "hydrologic_memory_and_storage_proxies": health.get("hydrologic_memory_and_storage_proxies", {}),
            "current_limb_rating_support": health.get("current_limb_rating_support", {}),
            "project_wse_transfer_support": health.get("project_wse_transfer_support", {}),
            "controlled_assimilation": health.get("controlled_assimilation", {}),
            "priority_actions": health.get("priority_actions", []),
            "interpretation": "Operational integrity and scientific confidence are separate. This block reports whether the forecast is running correctly and how strongly its current prediction is supported by calibration data.",
        },
        "hysteresis_diagnostics": {
            "status": hysteresis.get("status"),
            "current_limb": hysteresis.get("current_limb"),
            "classification": hysteresis.get("classification", {}),
            "rising_fit": hysteresis.get("rising_fit"),
            "falling_fit": hysteresis.get("falling_fit"),
            "loop_comparison": hysteresis.get("loop_comparison", {}),
            "confidence_effect": hysteresis.get("confidence_effect"),
            "interpretation": hysteresis.get("interpretation"),
        },
        "secondary_field_observations": {
            "rising_limb_stage_m": 1.70,
            "spring_stage_m": 1.50,
            "interpretation": "These support material seasonal/limb hysteresis and are retained as checks, not universal release thresholds.",
        },
        "low_pocket": starkey.get("low_pocket", {}),
        "decision": {
            "status": "not_ready" if depth_main is None or depth_main > 0 else "inspect_now",
            "schedule_use": "Use the raw all-member p50 as the working hydrometeorological inspection date and the risk-adjusted conservative-sensitivity p90 as the protected schedule contingency. Retain dry/central/wet traces for explanation.",
            "release_rule": "Release floodplain work only after the estimated WSE is at or below 650.20 m and a direct project-site inspection confirms drainage, access bearing and no renewed rise.",
            "site_checks": [
                "main floodplain visibly drained",
                "no sustained renewed rise forecast",
                "access and working platform have acceptable bearing capacity",
                "rutting, pumping and dewatering are manageable",
                "confirm whether the former 649.60 m pocket has in fact been raised and drains with the main platform",
            ],
        },
        "uncertainty": {
            **starkey.get("uncertainty", {}),
            "all_member_model_uncertainty": probabilistic.get("model_uncertainty", {}),
            "calibration_health": health.get("overall", {}),
            "risk_adjusted_sensitivity": uncertainty.get("planning_summary", {}),
            "apparent_hysteresis": {
                "confidence_effect": hysteresis.get("confidence_effect"),
                "maximum_absolute_stage_separation_m": hysteresis.get("loop_comparison", {}).get("maximum_absolute_stage_separation_m"),
                "target_rising_minus_falling_stage_m": hysteresis.get("loop_comparison", {}).get("target_rising_minus_falling_stage_m"),
            },
        },
        "limitations": starkey.get("limitations", [])
        + probabilistic.get("limitations", [])
        + health.get("limitations", [])
        + hysteresis.get("limitations", [])
        + uncertainty.get("limitations", [])
        + [
            "The approximately 650.20 m field threshold and the 6.77 m3/s calibration are based on one 2026 observation rather than a surveyed concurrent project-site water level.",
            "Construction release remains a field decision, not an automatic model output.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
