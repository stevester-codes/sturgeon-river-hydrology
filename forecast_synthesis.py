#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("sturgeon_pipeline_output")


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text())


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def hrdps_48h(forecast: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in forecast.get("deterministic_scenarios", [])
        if row.get("model") == "HRDPS"
        and int(row.get("horizon_h", 0)) == 48
        and bool(row.get("complete_horizon"))
    ]
    return max(rows, key=lambda row: str(row.get("run_time_utc") or "")) if rows else {}


def date_item(name: str, value: Any, role: str) -> dict[str, Any] | None:
    parsed = parse_date(value)
    if parsed is None:
        return None
    return {"name": name, "date_utc": parsed.isoformat(), "date": parsed, "role": role}


def rounded(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}{suffix}"


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    existing: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
            existing = list(reader)
    if any(item.get("run_id") == str(row.get("run_id")) for item in existing):
        return
    existing.append({key: "" if value is None else str(value) for key, value in row.items()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in existing:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)

    readiness_path = root / "forecast_v2" / "construction_readiness.json"
    forecast_path = root / "forecast_v2" / "forecast_impacts_v2.json"
    medium_path = root / "spatial" / "medium_range_qpf.json"
    health_path = root / "diagnostics" / "calibration_health.json"
    field_path = root / "diagnostics" / "project_site_recession_shadow.json"
    shadow_path = root / "diagnostics" / "historical_response_shadow_current.json"
    synthesis_path = root / "forecast_v2" / "forecast_synthesis.json"
    brief_path = root / "forecast_v2" / "forecast_brief.md"
    manifest_path = root / "run_manifest.json"
    history_path = root / "history" / "forecast_history.csv"

    previous = load_json(synthesis_path, required=False)
    readiness = load_json(readiness_path)
    forecast = load_json(forecast_path)
    medium = load_json(medium_path)
    health = load_json(health_path)
    field = load_json(field_path)
    shadow = load_json(shadow_path, required=False)

    generated = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ")
    hrdps = hrdps_48h(forecast)
    hrdps_run = hrdps.get("run_time_utc")
    geps_run = nested(medium, "geps", "run_time_utc")
    latest_stage_utc = readiness.get("latest_stage_utc")

    current = readiness.get("current_conditions", {})
    stage = finite(current.get("stage_05EA002_m"))
    discharge = finite(current.get("observed_discharge_05EA002_m3s"))
    stage_change_24h = finite(current.get("stage_change_24h_m"))
    freshness_age = finite(nested(readiness, "observation_freshness", "age_hours"))
    freshness_status = nested(readiness, "observation_freshness", "status")

    date_wse = finite(current.get("provisional_field_recession_estimated_project_wse_m"))
    q_wse = finite(current.get("provisional_field_discharge_relation_estimated_project_wse_m"))
    site_values = [value for value in (date_wse, q_wse) if value is not None]
    site_wse_low = min(site_values) if site_values else None
    site_wse_high = max(site_values) if site_values else None
    threshold_wse = finite(
        nested(readiness, "authoritative_operational_threshold", "main_floodplain_elevation_m"),
        650.20,
    )
    site_depth_low = site_wse_low - threshold_wse if site_wse_low is not None else None
    site_depth_high = site_wse_high - threshold_wse if site_wse_high is not None else None

    official_p50 = nested(readiness, "probabilistic_exposure", "quantiles", "p50", "date_utc")
    weather_p90 = nested(readiness, "probabilistic_exposure", "quantiles", "p90", "date_utc")
    screened_direct_q = nested(
        readiness,
        "risk_adjusted_planning",
        "planning_summary",
        "precipitation_screened_direct_q_p50",
        "date_utc",
    )
    contingency_date = nested(
        readiness,
        "risk_adjusted_planning",
        "planning_summary",
        "protected_schedule_p90",
        "date_utc",
    )
    field_crossing = nested(field, "date_recession_fit", "projected_threshold_crossing_local_date")

    consensus_inputs = [
        item
        for item in (
            date_item("official_weather_ensemble_median", official_p50, "primary"),
            date_item("precipitation_screened_direct_q", screened_direct_q, "independent_check"),
            date_item("contractor_site_rain_free_recession", field_crossing, "field_check"),
        )
        if item is not None
    ]
    consensus_dates = [item["date"] for item in consensus_inputs]
    if consensus_dates:
        consensus_start = min(consensus_dates)
        consensus_end = max(consensus_dates)
        consensus_spread = (consensus_end - consensus_start).days
    else:
        consensus_start = consensus_end = None
        consensus_spread = None

    if len(consensus_inputs) >= 3 and consensus_spread is not None and consensus_spread <= 2:
        agreement_status = "three_method_consensus"
        overall_confidence = "moderate"
    elif len(consensus_inputs) >= 2 and consensus_spread is not None and consensus_spread <= 2:
        agreement_status = "two_method_consensus"
        overall_confidence = "low_to_moderate"
    elif consensus_spread is not None and consensus_spread <= 4:
        agreement_status = "weak_consensus_widened_window"
        overall_confidence = "low_to_moderate"
    else:
        agreement_status = "material_method_disagreement"
        overall_confidence = "low"

    basin_mm = finite(hrdps.get("basin_mm"))
    lower_mm = finite(hrdps.get("lower_mm"))
    direct_local_mm = finite(hrdps.get("direct_local_mm"))
    short_range_delay = finite(nested(hrdps, "analog_prediction", "days_lost"))
    dry_days = finite(nested(readiness, "forecast_main_floodplain_exposure", "dry", "days_to_main_floodplain_exposure"))
    central_days = finite(nested(readiness, "forecast_main_floodplain_exposure", "central", "days_to_main_floodplain_exposure"))
    total_weather_shift = (
        central_days - dry_days
        if central_days is not None and dry_days is not None
        else None
    )

    shadow_run = nested(shadow, "hrdps", "run_time_utc")
    shadow_aligned = bool(hrdps_run and shadow_run and str(hrdps_run) == str(shadow_run))
    if shadow_aligned:
        shadow_status = "current_cycle_comparison_available"
        shadow_difference = finite(shadow.get("historical_model_minus_official_days"))
        shadow_historical_days = finite(shadow.get("historical_censored_model_days_lost"))
    elif shadow:
        shadow_status = "comparison_pending_current_weather_cycle"
        shadow_difference = None
        shadow_historical_days = None
    else:
        shadow_status = "historical_response_shadow_unavailable"
        shadow_difference = None
        shadow_historical_days = None

    material_shadow_disagreement = bool(
        shadow_aligned
        and shadow_difference is not None
        and abs(shadow_difference) >= 2.0
    )
    official_p50_days = finite(
        nested(readiness, "probabilistic_exposure", "quantiles", "p50", "days")
    )
    shadow_sensitivity_days = (
        official_p50_days + shadow_difference
        if material_shadow_disagreement
        and official_p50_days is not None
        and shadow_difference is not None
        else None
    )
    shadow_sensitivity_date = (
        (generated + timedelta(days=shadow_sensitivity_days)).date().isoformat()
        if shadow_sensitivity_days is not None
        else None
    )
    if material_shadow_disagreement and overall_confidence == "moderate":
        overall_confidence = "low_to_moderate"

    event_blocks = int(
        finite(nested(health, "historical_recession_validation", "independent_event_blocks"), 0)
        or 0
    )
    uncensored_events = int(
        finite(nested(health, "calibration_sample", "uncensored_peak_training_events"), 0)
        or 0
    )
    feature_status = nested(health, "current_forecast_feature_coverage", "status")
    geps_members = int(finite(nested(readiness, "probabilistic_exposure", "member_count"), 0) or 0)

    if freshness_status == "current" and freshness_age is not None and freshness_age <= 1.5:
        live_confidence = "high"
    elif freshness_status == "current" and freshness_age is not None and freshness_age <= 6:
        live_confidence = "moderate"
    else:
        live_confidence = "low"
    meteorology_confidence = "moderate" if hrdps and geps_members >= 20 else "low"
    recession_confidence = "moderate" if event_blocks >= 8 else "low"
    rain_response_confidence = (
        "low"
        if uncensored_events <= 2 or feature_status in {"material_extrapolation", "extrapolation"}
        else "low_to_moderate"
    )

    today = generated.date()
    consensus_start_days = (consensus_start - today).days if consensus_start else None
    if site_depth_low is not None and site_depth_low <= 0.10 and consensus_start_days is not None and consensus_start_days <= 2:
        decision_status = "inspection_window_approaching"
    elif site_depth_high is not None and site_depth_high <= 0:
        decision_status = "inspect_now_pending_field_release_checks"
    else:
        decision_status = "not_ready"

    prior_official = nested(previous, "working_forecast", "official_threshold_median_date")
    prior_hrdps_mm = finite(nested(previous, "weather", "hrdps_48h", "basin_mm"))
    prior_stage = finite(nested(previous, "current_state", "stage_05EA002_m"))
    prior_discharge = finite(nested(previous, "current_state", "discharge_05EA002_m3s"))
    official_change_days = None
    if parse_date(prior_official) and parse_date(official_p50):
        official_change_days = (parse_date(official_p50) - parse_date(prior_official)).days

    observed_changes = {
        "previous_run_available": bool(previous),
        "official_median_date_change_days": official_change_days,
        "stage_change_since_previous_run_m": (
            stage - prior_stage if stage is not None and prior_stage is not None else None
        ),
        "discharge_change_since_previous_run_m3s": (
            discharge - prior_discharge
            if discharge is not None and prior_discharge is not None
            else None
        ),
        "hrdps_48h_basin_rain_change_mm": (
            basin_mm - prior_hrdps_mm
            if basin_mm is not None and prior_hrdps_mm is not None
            else None
        ),
        "causal_attribution_available": False,
        "interpretation": (
            "These are observed input and output changes between issued forecasts. They are not a causal decomposition; "
            "counterfactual one-input-at-a-time attribution has not yet been implemented."
        ),
    }

    invalidation_conditions = [
        "05EA002 observations become older than six hours or the hydrograph reverses into a sustained rise.",
        "A newer HRDPS or GEPS cycle materially increases rainfall or moves the weather-ensemble upper date.",
        "The official median, screened direct-Q check and contractor-site recession projection diverge by more than two days.",
        "A verified site survey differs from the provisional field-informed WSE range by more than 0.15 m.",
        "The current-cycle historical response disagreement persists, grows, or is later validated for operational use.",
        "Access, drainage or bearing conditions remain unsuitable even after the water elevation reaches 650.20 m.",
    ]
    release_checklist = [
        "Verify current project-site water elevation or directly confirm the work area is drained to 650.20 m or lower.",
        "Confirm the river and project water surface are not beginning a renewed rise.",
        "Confirm construction access has adequate bearing capacity.",
        "Confirm the 649.60 m low pocket is drained, isolated or otherwise managed.",
        "Record the field observation time, exact location, method and vertical datum.",
    ]

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_utc": generated.isoformat(),
        "git": {
            "sha": os.getenv("GITHUB_SHA"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
            "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
            "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "trigger_reason": os.getenv("TRIGGER_REASON"),
        },
        "authoritative_inputs": {
            "construction_readiness": {
                "path": str(readiness_path),
                "generated_utc": readiness.get("generated_utc"),
            },
            "forecast_impacts": {
                "path": str(forecast_path),
                "generated_utc": forecast.get("generated_utc"),
                "hrdps_run_time_utc": hrdps_run,
            },
            "medium_range_qpf": {
                "path": str(medium_path),
                "generated_utc": medium.get("generated_utc"),
                "geps_run_time_utc": geps_run,
                "geps_member_count": geps_members,
            },
            "calibration_health": {
                "path": str(health_path),
                "generated_utc": health.get("generated_utc"),
                "score_version": nested(health, "overall", "score_version"),
            },
            "project_site_field_evidence": {
                "path": str(field_path),
                "generated_utc": field.get("generated_utc"),
                "status": field.get("status"),
            },
        },
        "shadow_inputs": {
            "historical_response": {
                "path": str(shadow_path),
                "generated_utc": shadow.get("generated_utc"),
                "hrdps_run_time_utc": shadow_run,
                "cycle_alignment_status": shadow_status,
                "included_in_current_comparison": shadow_aligned,
            }
        },
        "cycle_consistency": {
            "status": "consistent" if shadow_aligned or not shadow else "operational_consistent_shadow_pending",
            "latest_stage_utc": latest_stage_utc,
            "hrdps_run_time_utc": hrdps_run,
            "geps_run_time_utc": geps_run,
            "historical_shadow_hrdps_run_time_utc": shadow_run,
        },
    }

    synthesis = {
        "schema_version": 1,
        "status": "operational_forecast_synthesis",
        "run_id": run_id,
        "generated_utc": generated.isoformat(),
        "decision": {
            "status": decision_status,
            "headline": (
                "Not ready; timing methods agree on an inspection window, but the historical rain-response shadow is materially later. Retain the schedule contingency date."
                if decision_status == "not_ready" and material_shadow_disagreement
                else "Not ready; use the consensus inspection window and retain a separate schedule contingency date."
                if decision_status == "not_ready"
                else "Inspection window is approaching; field verification remains mandatory."
            ),
            "release_rule": nested(readiness, "decision", "release_rule"),
        },
        "working_forecast": {
            "consensus_status": agreement_status,
            "consensus_method_count": len(consensus_inputs),
            "consensus_inputs": [
                {key: value for key, value in item.items() if key != "date"}
                for item in consensus_inputs
            ],
            "consensus_spread_days": consensus_spread,
            "inspection_window_start_date": consensus_start.isoformat() if consensus_start else None,
            "inspection_window_end_date": consensus_end.isoformat() if consensus_end else None,
            "official_threshold_median_date": official_p50,
            "weather_ensemble_upper_date": weather_p90,
            "engineering_schedule_contingency_date": contingency_date,
            "historical_response_shadow_sensitivity_date": shadow_sensitivity_date,
            "terminology": {
                "weather_ensemble_upper_date": "Actual percentile of the same GEPS threshold distribution.",
                "engineering_schedule_contingency_date": "Sensitivity envelope for schedule protection; not a calibrated p90 probability.",
            },
        },
        "current_state": {
            "latest_stage_utc": latest_stage_utc,
            "observation_age_hours": freshness_age,
            "hydrograph_limb": readiness.get("hydrograph_limb"),
            "stage_05EA002_m": stage,
            "stage_change_24h_m": stage_change_24h,
            "discharge_05EA002_m3s": discharge,
            "provisional_site_wse_range_m": [site_wse_low, site_wse_high] if site_values else None,
            "provisional_depth_over_650_20_range_m": [site_depth_low, site_depth_high] if site_values else None,
            "actual_current_site_measurement_available": False,
        },
        "weather": {
            "hrdps_48h": {
                "run_time_utc": hrdps_run,
                "basin_mm": basin_mm,
                "lower_basin_mm": lower_mm,
                "direct_local_mm": direct_local_mm,
                "storm_type": hrdps.get("storm_type"),
                "official_short_range_response_days_lost": short_range_delay,
                "feature_support_status": feature_status,
            },
            "geps": {
                "run_time_utc": geps_run,
                "member_count": geps_members,
                "weather_ensemble_upper_date": weather_p90,
            },
            "central_total_weather_shift_from_dry_days": total_weather_shift,
        },
        "evidence_reconciliation": {
            "rule": (
                "The official GEPS median is primary. The precipitation-screened direct-Q date and contractor-site recession date are independent checks. "
                "When available methods agree within two days, their minimum-to-maximum range is reported as the consensus inspection window. Methods are not averaged."
            ),
            "historical_response_shadow": {
                "status": shadow_status,
                "official_analogue_response_days_lost": (
                    finite(shadow.get("official_analogue_response_days_lost")) if shadow_aligned else None
                ),
                "historical_censored_model_days_lost": shadow_historical_days,
                "historical_minus_official_days": shadow_difference,
                "shadow_adjusted_threshold_sensitivity_days": shadow_sensitivity_days,
                "shadow_adjusted_threshold_sensitivity_date": shadow_sensitivity_date,
                "material_disagreement": material_shadow_disagreement,
                "operational_effect": "none_shadow_only",
            },
        },
        "confidence": {
            "overall_inspection_timing": overall_confidence,
            "live_river_state": live_confidence,
            "meteorological_forecast": meteorology_confidence,
            "dry_recession_timing": recession_confidence,
            "rain_response_estimate": rain_response_confidence,
            "current_site_wse": "low",
            "threshold_translation": "low_to_moderate",
            "construction_release": "requires_field_verification_not_remotely_forecastable",
            "key_reasons": [
                f"{len(consensus_inputs)} timing methods span {consensus_spread} day(s)." if consensus_spread is not None else "Timing-method consensus is unavailable.",
                f"Rain-response calibration has {uncensored_events} uncensored live peak-training events.",
                f"Current forecast feature support is {feature_status}.",
                f"Screened direct-discharge validation contains {event_blocks} independent event blocks.",
                (
                    f"The current-cycle historical rain-response shadow is {abs(shadow_difference):.2f} days later than the official response."
                    if material_shadow_disagreement and shadow_difference is not None
                    else "No material current-cycle historical rain-response disagreement is available."
                ),
                "Only two date-level contractor site elevations are available and their datum, exact location and time remain unverified.",
            ],
        },
        "change_since_previous_issued_forecast": observed_changes,
        "conditions_that_invalidate_or_widen_the_forecast": invalidation_conditions,
        "field_release_checklist": release_checklist,
        "reporting_contract": {
            "official_working_date": "official_threshold_median_date",
            "inspection_window": "consensus range of agreeing independent methods",
            "weather_upper_date": "GEPS percentile for the same 650.20 m threshold",
            "schedule_contingency": "engineering sensitivity envelope, not a formal percentile",
            "actual_site_depth": "not reported without a current verified field observation",
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    synthesis_path.write_text(json.dumps(synthesis, indent=2))

    window = (
        consensus_start.isoformat()
        if consensus_start and consensus_end and consensus_start == consensus_end
        else f"{consensus_start.isoformat()} to {consensus_end.isoformat()}"
        if consensus_start and consensus_end
        else "unavailable"
    )
    site_range = (
        f"{site_wse_low:.3f}–{site_wse_high:.3f} m"
        if site_wse_low is not None and site_wse_high is not None
        else "unavailable"
    )
    depth_range = (
        f"{site_depth_low:.3f}–{site_depth_high:.3f} m"
        if site_depth_low is not None and site_depth_high is not None
        else "unavailable"
    )
    shadow_line = (
        f"Current-cycle historical shadow: {fmt(shadow_historical_days, 2, ' d')} versus official {fmt(short_range_delay, 2, ' d')}; difference {fmt(shadow_difference, 2, ' d')}. Shadow only."
        if shadow_aligned
        else "Historical response comparison is pending for the current HRDPS cycle and is excluded from the forecast synthesis."
    )
    brief = f"""# Sturgeon River Construction Forecast Brief

Generated: {generated.isoformat()}  
Run ID: `{run_id}`

## 1. Decision

**{decision_status.replace('_', ' ').title()}.** The river is {readiness.get('hydrograph_limb', 'unknown')}. Final construction release still requires a verified current site elevation or drainage inspection, suitable access and bearing capacity, and no renewed rise.

## 2. Working inspection window

- **Consensus inspection window:** **{window}**
- Official weather-ensemble median threshold date: **{official_p50 or 'unavailable'}**
- Independent precipitation-screened direct-Q date: **{screened_direct_q or 'unavailable'}**
- Contractor-site rain-free recession projection: **{field_crossing or 'unavailable'}**
- Weather-ensemble upper date: **{weather_p90 or 'unavailable'}**
- Engineering schedule contingency: **{contingency_date or 'unavailable'}** — sensitivity envelope, not a formal p90 probability.
- Historical response shadow sensitivity: **{shadow_sensitivity_date or 'not applicable'}** — shadow only and excluded from the consensus window.

## 3. Current river and site state

- 05EA002 stage: **{fmt(stage, 3, ' m')}**, 24-hour change **{fmt(stage_change_24h, 3, ' m')}**
- 05EA002 discharge: **{fmt(discharge, 2, ' m³/s')}**
- Observation age: **{fmt(freshness_age, 2, ' h')}**
- Provisional field-informed project WSE: **{site_range}**
- Provisional depth above 650.20 m: **{depth_range}**
- No direct current construction-site measurement is available.

## 4. Expected rainfall and added delay

- HRDPS cycle: **{hrdps_run or 'unavailable'}**
- 48-hour basin rain: **{fmt(basin_mm, 2, ' mm')}**
- Lower-basin rain: **{fmt(lower_mm, 2, ' mm')}**
- Direct-local rain: **{fmt(direct_local_mm, 2, ' mm')}**
- Official short-range response delay: **{fmt(short_range_delay, 2, ' days')}**
- Central forecast shift versus the dry trace: **{fmt(total_weather_shift, 2, ' days')}**

## 5. Evidence agreement

The official GEPS median is primary. Independent methods are not averaged. Because the available timing methods span **{consensus_spread if consensus_spread is not None else 'unavailable'} day(s)**, the synthesis status is **{agreement_status.replace('_', ' ')}**.

{shadow_line}

## 6. Confidence by component

- Overall inspection timing: **{overall_confidence}**
- Live river state: **{live_confidence}**
- Meteorological forecast: **{meteorology_confidence}**
- Dry recession timing: **{recession_confidence}**
- Rain-response estimate: **{rain_response_confidence}**
- Current site WSE: **low**
- Threshold translation: **low to moderate**
- Construction release: **requires field verification**

## 7. What changed since the previous issued forecast

- Official median date movement: **{fmt(finite(official_change_days), 0, ' day(s)')}**
- Stage change since previous run: **{fmt(observed_changes['stage_change_since_previous_run_m'], 3, ' m')}**
- Discharge change since previous run: **{fmt(observed_changes['discharge_change_since_previous_run_m3s'], 2, ' m³/s')}**
- HRDPS 48-hour basin-rain change: **{fmt(observed_changes['hrdps_48h_basin_rain_change_mm'], 2, ' mm')}**

These are observed changes, not a causal decomposition.

## 8. Conditions that would invalidate or widen the forecast

""" + "\n".join(f"- {item}" for item in invalidation_conditions) + """

## 9. Field release checklist

""" + "\n".join(f"- {item}" for item in release_checklist) + "\n"
    brief_path.write_text(brief)

    history_row = {
        "run_id": run_id,
        "generated_utc": generated.isoformat(),
        "latest_stage_utc": latest_stage_utc,
        "stage_05EA002_m": rounded(stage, 4),
        "discharge_05EA002_m3s": rounded(discharge, 3),
        "hydrograph_limb": readiness.get("hydrograph_limb"),
        "hrdps_run_time_utc": hrdps_run,
        "hrdps_basin_mm_48h": rounded(basin_mm, 3),
        "official_short_range_delay_days": rounded(short_range_delay, 3),
        "official_threshold_median_date": official_p50,
        "screened_direct_q_date": screened_direct_q,
        "field_recession_date": field_crossing,
        "consensus_window_start_date": consensus_start.isoformat() if consensus_start else None,
        "consensus_window_end_date": consensus_end.isoformat() if consensus_end else None,
        "weather_ensemble_upper_date": weather_p90,
        "engineering_schedule_contingency_date": contingency_date,
        "provisional_site_wse_low_m": rounded(site_wse_low, 4),
        "provisional_site_wse_high_m": rounded(site_wse_high, 4),
        "overall_confidence": overall_confidence,
        "decision_status": decision_status,
        "historical_shadow_cycle_aligned": shadow_aligned,
        "previous_official_threshold_median_date": prior_official,
        "official_median_date_change_days": official_change_days,
    }
    append_history(history_path, history_row)

    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()
