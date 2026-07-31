#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import calibration_health
import discharge_recession_candidate
import hysteresis_diagnostics
import project_site_recession_shadow
import uncertainty_sensitivity

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
STARKEY = ROOT / "routing" / "forecast_starkey_wse.json"
PROBABILISTIC = ROOT / "forecast_v2" / "project_threshold_ensemble.json"
CALIBRATION_HEALTH = ROOT / "diagnostics" / "calibration_health.json"
HYSTERESIS = ROOT / "diagnostics" / "hysteresis_diagnostics.json"
UNCERTAINTY = ROOT / "diagnostics" / "uncertainty_sensitivity.json"
DISCHARGE_CANDIDATE = ROOT / "diagnostics" / "discharge_recession_candidate.json"
PROJECT_SITE_SHADOW = ROOT / "diagnostics" / "project_site_recession_shadow.json"
OUT = ROOT / "forecast_v2" / "construction_readiness.json"
MAX_OBSERVATION_AGE_HOURS = 6.0


def finite(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_utc(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def main() -> None:
    for path in (SUMMARY, STARKEY, PROBABILISTIC):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = json.loads(SUMMARY.read_text())
    starkey = json.loads(STARKEY.read_text())
    probabilistic = json.loads(PROBABILISTIC.read_text())
    if probabilistic.get("status") != "operational_project_threshold_ensemble":
        raise RuntimeError("Project-threshold ensemble output is not operational")

    target = summary.get("target_05EA002", {})
    generated_now = datetime.now(timezone.utc)
    observation_time = parse_utc(target.get("latest_utc"))
    if observation_time is None:
        raise RuntimeError("Latest 05EA002 observation timestamp is missing or invalid")
    observation_age_hours = (generated_now - observation_time).total_seconds() / 3600.0
    if observation_age_hours < -0.25:
        raise RuntimeError(
            f"Latest 05EA002 observation is in the future by {-observation_age_hours:.2f} hours"
        )
    if observation_age_hours > MAX_OBSERVATION_AGE_HOURS:
        raise RuntimeError(
            f"Latest 05EA002 observation is stale: {observation_age_hours:.2f} hours old"
        )

    # Diagnostics are regenerated as part of the authoritative readiness build.
    # Direct-discharge and site-recession models remain independent sensitivities
    # and cannot alter the official crossing date without later evidence and review.
    hysteresis_diagnostics.main()
    calibration_health.main()
    uncertainty_sensitivity.main()
    discharge_recession_candidate.main()
    project_site_recession_shadow.main()
    for path in (
        CALIBRATION_HEALTH,
        HYSTERESIS,
        UNCERTAINTY,
        DISCHARGE_CANDIDATE,
        PROJECT_SITE_SHADOW,
    ):
        if not path.exists():
            raise RuntimeError(f"Required diagnostic output was not generated: {path}")
    health = json.loads(CALIBRATION_HEALTH.read_text())
    hysteresis = json.loads(HYSTERESIS.read_text())
    uncertainty = json.loads(UNCERTAINTY.read_text())
    discharge_candidate = json.loads(DISCHARGE_CANDIDATE.read_text())
    project_site_shadow = json.loads(PROJECT_SITE_SHADOW.read_text())

    current = starkey.get("current", {})
    threshold = starkey.get("construction_threshold", {})
    scenarios = starkey.get("scenarios", {})
    distribution = probabilistic.get("crossing_distribution", {})

    schedule = {}
    for name in ("dry", "central", "wet"):
        item = scenarios.get(name, {})
        schedule[name] = {
            "days_to_main_floodplain_exposure": item.get(
                "main_floodplain_crossing_days"
            ),
            "date_utc": item.get("main_floodplain_crossing_date_utc"),
        }

    stage_now = finite(target.get("latest"))
    change_24h = finite(target.get("change_24h"), 0.0)
    limb = starkey.get("hydrograph_limb", "unknown")

    # The legacy current value is retained only as the steady-state design-profile
    # equivalent. It is no longer represented as actual 2026 site WSE or depth.
    design_profile_wse = finite(current.get("estimated_starkey_wse_m"))
    design_profile_depth_main = finite(current.get("depth_over_main_floodplain_m"))
    site_state = project_site_shadow.get("current_site_state", {})
    provisional_date_wse = finite(
        site_state.get("date_recession_estimated_project_wse_m")
    )
    provisional_date_depth = finite(
        site_state.get("date_recession_estimated_depth_over_650_20_m")
    )
    provisional_q_wse = finite(
        site_state.get("discharge_relation_estimated_project_wse_m")
    )
    provisional_q_depth = finite(
        site_state.get("discharge_relation_estimated_depth_over_650_20_m")
    )

    historical_sensitivities = uncertainty.get(
        "historical_direct_discharge_sensitivities", {}
    )
    planning_summary = uncertainty.get("planning_summary", {})
    project_transfer_health = dict(health.get("project_wse_transfer_support", {}))
    project_transfer_health.update(
        {
            "provisional_2026_field_conflict": True,
            "actual_current_site_depth_use": "not_supported",
            "field_evidence_status": project_site_shadow.get("status"),
            "interpretation_override": (
                "The complete flood-study profile remains useful for design-event context and "
                "threshold translation. Two provisional contractor site elevations are materially "
                "above that steady-state profile at comparable 2026 discharge, so it is not used "
                "as the actual current construction-site water surface."
            ),
        }
    )

    output = {
        "generated_utc": generated_now.isoformat(),
        "latest_stage_utc": observation_time.isoformat(),
        "observation_freshness": {
            "age_hours": observation_age_hours,
            "maximum_allowed_age_hours": MAX_OBSERVATION_AGE_HOURS,
            "status": "current",
        },
        "project_location": starkey.get("project_location", {}),
        "hydrograph_limb": limb,
        "current_conditions": {
            "stage_05EA002_m": stage_now,
            "stage_change_24h_m": change_24h,
            "observed_discharge_05EA002_m3s": current.get(
                "observed_discharge_05EA002_m3s"
            ),
            "estimated_starkey_wse_m": None,
            "estimated_project_wse_m": None,
            "estimated_starkey_wse_range_m": None,
            "estimated_depth_over_main_floodplain_m": None,
            "estimated_depth_over_low_pocket_m": None,
            "actual_2026_site_wse_status": (
                "not_directly_observed_current_value; "
                "steady_state_design_profile_conflicted_by_field_evidence"
            ),
            "steady_state_design_profile_equivalent_wse_m": design_profile_wse,
            "steady_state_design_profile_equivalent_wse_range_m": current.get(
                "estimated_starkey_wse_range_m"
            ),
            "steady_state_design_profile_equivalent_depth_over_650_20_m": (
                design_profile_depth_main
            ),
            "provisional_field_recession_estimated_project_wse_m": provisional_date_wse,
            "provisional_field_recession_estimated_depth_over_650_20_m": (
                provisional_date_depth
            ),
            "provisional_field_discharge_relation_estimated_project_wse_m": (
                provisional_q_wse
            ),
            "provisional_field_discharge_relation_estimated_depth_over_650_20_m": (
                provisional_q_depth
            ),
            "site_wse_interpretation": (
                "The steady-state flood-study transfer is retained as design-event context "
                "and threshold translation, not as the actual 2026 construction-site water "
                "surface. Contractor observations support separate provisional date-recession "
                "and discharge-relation estimates until datum, location and time are verified."
            ),
        },
        "authoritative_operational_threshold": {
            "main_floodplain_elevation_m": threshold.get(
                "main_floodplain_wse_m", 650.20
            ),
            "calibrated_05EA002_discharge_m3s": threshold.get(
                "calibrated_target_discharge_m3s", 6.77
            ),
            "equivalent_05EA002_stage_on_current_limb_m": threshold.get(
                "equivalent_05EA002_stage_on_current_limb_m"
            ),
            "basis": (
                "Elevation 650.20 m is the physical project work-area threshold. "
                "The corresponding 6.77 m3/s operational anchor was reconstructed "
                "from a later 2026 rising-limb condition near 1.70 m at 05EA002, "
                "not from a concurrent surveyed project WSE. The original project "
                "condition had been understood near 1.50 m. The two contractor "
                "date-level observations independently extrapolate close to this "
                "discharge anchor but remain provisional until fully documented."
            ),
        },
        "forecast_main_floodplain_exposure": schedule,
        "probabilistic_exposure": {
            "status": probabilistic.get("status"),
            "member_count": probabilistic.get("geps", {}).get("member_count"),
            "quantiles": distribution.get("quantiles", {}),
            "earliest": distribution.get("earliest"),
            "latest": distribution.get("latest"),
            "mean_days": distribution.get("mean_days"),
            "standard_deviation_days": distribution.get(
                "standard_deviation_days"
            ),
            "probability_exposed_by_date": distribution.get(
                "probability_exposed_by_date", []
            ),
            "interpretation": (
                "These are raw all-member GEPS meteorological probabilities translated "
                "to the 6.77 m3/s operational threshold. They are scheduling probabilities, "
                "not estimates of actual current 2026 project-site water depth."
            ),
        },
        "risk_adjusted_planning": {
            "status": uncertainty.get("status"),
            "planning_summary": planning_summary,
            "scenarios": uncertainty.get("scenarios", {}),
            "historical_direct_discharge_sensitivities": historical_sensitivities,
            "uncertainty_components": uncertainty.get("uncertainty_components", {}),
            "interpretation": (
                "The live discharge-threshold forecast remains official for timing. The "
                "precipitation-screened direct-Q forecast is the preferred independent timing "
                "check. Actual current site WSE is represented only by provisional contractor-"
                "observation diagnostics until a new verified survey is available."
            ),
        },
        "model_health": {
            "status": health.get("status"),
            "overall": health.get("overall", {}),
            "calibration_sample": health.get("calibration_sample", {}),
            "forecast_feature_coverage": health.get(
                "current_forecast_feature_coverage", {}
            ),
            "hydrologic_memory_and_storage_proxies": health.get(
                "hydrologic_memory_and_storage_proxies", {}
            ),
            "current_limb_rating_support": health.get(
                "current_limb_rating_support", {}
            ),
            "project_wse_transfer_support": project_transfer_health,
            "historical_recession_validation": health.get(
                "historical_recession_validation", {}
            ),
            "controlled_assimilation": health.get("controlled_assimilation", {}),
            "priority_actions": health.get("priority_actions", []),
            "interpretation": (
                "Operational integrity and scientific confidence are separate. The new "
                "field evidence specifically reduces confidence in absolute current site-depth "
                "estimates, while leaving the discharge-based threshold timing as a separate question."
            ),
        },
        "direct_discharge_redundancy_check": {
            "status": discharge_candidate.get("status"),
            "mode": discharge_candidate.get("mode"),
            "dry_hourly_points": discharge_candidate.get("dry_hourly_points"),
            "candidate_fit": discharge_candidate.get("candidate_fit"),
            "holdout": discharge_candidate.get("holdout", {}),
            "current_projection": discharge_candidate.get(
                "current_projection", {}
            ),
            "promotion_recommendation": discharge_candidate.get(
                "promotion_recommendation", {}
            ),
            "interpretation": (
                "This shorter-window shadow model forecasts the 6.77 m3/s threshold "
                "directly from discharge recession. It does not estimate actual site depth."
            ),
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
        "project_site_field_evidence": {
            "status": project_site_shadow.get("status"),
            "mode": project_site_shadow.get("mode"),
            "observations": project_site_shadow.get("observations", []),
            "date_recession_fit": project_site_shadow.get("date_recession_fit"),
            "discharge_wse_fit": project_site_shadow.get("discharge_wse_fit"),
            "current_site_state": project_site_shadow.get("current_site_state", {}),
            "comparison_with_design_profile": project_site_shadow.get(
                "comparison_with_design_profile", {}
            ),
            "operational_interpretation": project_site_shadow.get(
                "operational_interpretation"
            ),
            "limitations": project_site_shadow.get("limitations", []),
        },
        "secondary_field_observations": {
            "original_project_condition_stage_understood_m": 1.50,
            "later_reconstructed_rising_stage_m": 1.70,
            "later_reconstructed_rising_discharge_m3s": 6.77,
            "project_threshold_wse_m": 650.20,
            "concurrent_surveyed_triple_available": False,
            "contractor_date_level_site_wse_observations_m": {
                "2026-07-16": 651.748,
                "2026-07-23": 651.336,
            },
            "interpretation": (
                "The stage, discharge, threshold and contractor elevations are related "
                "pieces of evidence, not one simultaneous surveyed observation. The new "
                "site readings show that the 2026 hydraulic condition was materially above "
                "the steady-state design-profile WSE at comparable discharge."
            ),
        },
        "low_pocket": starkey.get("low_pocket", {}),
        "decision": {
            "status": (
                "not_ready"
                if provisional_date_depth is None or provisional_date_depth > 0
                else "inspect_now"
            ),
            "schedule_use": (
                "Use the nominal all-member p50 as the official working inspection date, "
                "compare it with the precipitation-screened direct-Q p50 and the provisional "
                "site-recession crossing, and retain the protected p90 for contingency."
            ),
            "release_rule": (
                "Release floodplain work only after a verified current project-site survey "
                "or direct drainage inspection confirms the work area is at or below 650.20 m, "
                "access has adequate bearing, and no renewed rise is forecast. Do not release "
                "from the steady-state design-profile equivalent WSE alone."
            ),
            "site_checks": [
                "obtain a current project-site water elevation tied to the project benchmark",
                "main floodplain visibly drained",
                "no sustained renewed rise forecast",
                "access and working platform have acceptable bearing capacity",
                "rutting, pumping and dewatering are manageable",
                "confirm whether the former 649.60 m pocket has in fact been raised and drains with the main platform",
            ],
        },
        "uncertainty": {
            **starkey.get("uncertainty", {}),
            "all_member_model_uncertainty": probabilistic.get(
                "model_uncertainty", {}
            ),
            "calibration_health": health.get("overall", {}),
            "risk_adjusted_sensitivity": planning_summary,
            "historical_direct_discharge_sensitivities": historical_sensitivities,
            "project_site_field_evidence": project_site_shadow.get(
                "comparison_with_design_profile", {}
            ),
            "apparent_hysteresis": {
                "confidence_effect": hysteresis.get("confidence_effect"),
                "maximum_absolute_stage_separation_m": hysteresis.get(
                    "loop_comparison", {}
                ).get("maximum_absolute_stage_separation_m"),
                "target_rising_minus_falling_stage_m": hysteresis.get(
                    "loop_comparison", {}
                ).get("target_rising_minus_falling_stage_m"),
            },
        },
        "limitations": starkey.get("limitations", [])
        + probabilistic.get("limitations", [])
        + health.get("limitations", [])
        + hysteresis.get("limitations", [])
        + uncertainty.get("limitations", [])
        + discharge_candidate.get("limitations", [])
        + project_site_shadow.get("limitations", [])
        + [
            "The 650.20 m threshold and 6.77 m3/s operational anchor are reconstructed from related evidence rather than one concurrent surveyed project-site water level.",
            "The precipitation-screened historical direct-Q check uses 10 km RDPA and only a limited number of independent dry event blocks.",
            "Contractor-reported July 16 and July 23 site elevations are provisional because exact survey time, datum, method and shot location remain undocumented.",
            "The steady-state RS18883 flood-study curve materially underpredicts the two provisional 2026 site observations and is no longer presented as actual current site depth.",
            "Construction release remains a field decision, not an automatic model output.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
