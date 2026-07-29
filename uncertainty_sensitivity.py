#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from forecast_impacts_v2 import model_rate
from starkey_wse_transfer import q_for_wse

ROOT = Path("sturgeon_pipeline_output")
BASE = ROOT / "calibration" / "calibration.json"
FORECAST = ROOT / "forecast_v2" / "forecast_impacts_v2.json"
PROBABILITY = ROOT / "forecast_v2" / "project_threshold_ensemble.json"
PROJECT_WSE = ROOT / "routing" / "forecast_starkey_wse.json"
HISTORICAL_GAUGE = Path("output/archive_probe/historical_gauge_analysis.json")
HISTORICAL_RDPA = Path("output/archive_probe/historical_rdpa_pairing.json")
HISTORICAL_SELECTION = Path("output/archive_probe/historical_rdpa_model_selection.json")
OUT = ROOT / "diagnostics" / "uncertainty_sensitivity.json"
MAIN_WSE = 650.20
SITE_UNCERTAINTY_M = 0.15
MAX_HOURS = 24 * 120


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


def load_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def stage_from_q(discharge: float, rating: dict) -> float:
    return (float(discharge) - float(rating["intercept_m3s"])) / float(
        rating["slope_m3s_per_m"]
    )


def rain_free_days(stage_now: float, target_stage: float, recession: dict) -> float:
    if stage_now <= target_stage:
        return 0.0
    stage = float(stage_now)
    hours = 0
    while stage > target_stage and hours < MAX_HOURS:
        stage += model_rate(recession, stage) / 24.0
        hours += 1
    if stage > target_stage:
        raise RuntimeError("Recession did not reach sensitivity target within 120 days")
    return hours / 24.0


def short_range_bounds(
    forecast: dict, central_fallback: float
) -> tuple[float, float, float]:
    candidates = [
        row
        for row in forecast.get("deterministic_scenarios", [])
        if str(row.get("model")) == "HRDPS" and int(row.get("horizon_h", 0)) == 48
    ]
    if not candidates:
        return central_fallback, central_fallback, central_fallback
    row = candidates[0]
    central = finite(row.get("analog_prediction", {}).get("days_lost"), central_fallback)
    bounds = row.get("estimated_days_lost_range", [central, central])
    low = finite(bounds[0], central) if len(bounds) > 0 else central
    high = finite(bounds[1], central) if len(bounds) > 1 else central
    return max(0.0, low), max(0.0, central), max(0.0, high)


def window_delay(window: dict, mode: str) -> float:
    central = max(0.0, finite(window.get("days_lost_central"), 0.0))
    error = max(0.0, finite(window.get("days_lost_rmse"), 0.0))
    if mode == "low":
        return max(0.0, central - error)
    if mode == "high":
        return central + error
    return central


def member_days(
    member: dict, base_days: float, short_delay: float, response_mode: str
) -> float:
    total_delay = max(0.0, float(short_delay))
    projected_days = base_days + total_delay
    for window in member.get("later_windows", []):
        start_h = max(0.0, finite(window.get("start_h"), 0.0))
        if start_h / 24.0 > projected_days:
            continue
        delay = window_delay(window, response_mode)
        if delay > 0:
            total_delay += delay
            projected_days = base_days + total_delay
    return projected_days


def date_record(generated: datetime, days: float) -> dict:
    return {
        "days": float(days),
        "date_utc": (generated + timedelta(days=float(days))).date().isoformat(),
    }


def summarize(values: np.ndarray, generated: datetime) -> dict:
    quantiles = {}
    for name, q in (
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
    ):
        quantiles[name] = date_record(generated, float(np.quantile(values, q)))
    return {
        "member_count": int(len(values)),
        "mean": date_record(generated, float(np.mean(values))),
        "earliest": date_record(generated, float(np.min(values))),
        "latest": date_record(generated, float(np.max(values))),
        "standard_deviation_days": float(np.std(values, ddof=0)),
        "quantiles": quantiles,
    }


def projection_days(model: dict) -> float | None:
    projection = model.get("projection") or {}
    return finite(projection.get("days")) if projection.get("reached") else None


def cv_rmse(model: dict) -> float | None:
    return finite(
        model.get("event_block_cross_validation", {})
        .get("aggregate", {})
        .get("rmse_per_day")
    )


def derive_preferred_screened(pairing: dict) -> dict | None:
    if pairing.get("status") != "historical_rdpa_pairing_complete":
        return None
    coverage = finite(pairing.get("rdpa_retrieval", {}).get("coverage_fraction"), 0.0)
    if coverage < 0.90:
        return None
    models = pairing.get("models", {})
    gauge_rmse = cv_rmse(models.get("gauge_only", {}))
    eligible = []
    for name in ("rdpa_strict", "rdpa_moderate"):
        model = models.get(name, {})
        days = projection_days(model)
        rmse = cv_rmse(model)
        points = int(model.get("points", 0) or 0)
        events = int(model.get("events", 0) or 0)
        if (
            days is not None
            and rmse is not None
            and gauge_rmse is not None
            and points >= 200
            and events >= 3
            and rmse <= gauge_rmse
        ):
            eligible.append(
                {
                    "name": name,
                    "points": points,
                    "events": events,
                    "rain_free_days_to_6_77_m3s": days,
                    "event_block_rmse_per_day": rmse,
                    "skill_improvement_vs_gauge_only_pct": (
                        (gauge_rmse - rmse) / gauge_rmse * 100.0
                        if gauge_rmse > 0
                        else None
                    ),
                    "fit": model.get("fit", {}),
                    "event_block_cross_validation": model.get(
                        "event_block_cross_validation", {}
                    ),
                }
            )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            float(item["event_block_rmse_per_day"]),
            -int(item["events"]),
            -int(item["points"]),
        ),
    )


def historical_direct_discharge_candidates() -> tuple[dict, dict]:
    gauge_data = load_optional(HISTORICAL_GAUGE)
    pairing = load_optional(HISTORICAL_RDPA)
    selection = load_optional(HISTORICAL_SELECTION)
    candidates: dict[str, dict] = {}

    if gauge_data:
        gauge_model = gauge_data.get("gauge_only_recession_screen", {})
        projection = (
            gauge_model.get("projections", {})
            .get("historical_gauge_only_discharge_recession", {})
        )
        gauge_days = finite(projection.get("days")) if projection.get("reached") else None
        if gauge_days is not None:
            candidates["gauge_only_18_month"] = {
                "name": "gauge_only_18_month",
                "source": "historical_gauge_analysis",
                "screening": "gauge decline only; not precipitation screened",
                "rain_free_days_to_6_77_m3s": gauge_days,
                "eligible_for_protected_schedule": True,
                "eligible_for_live_central_replacement": False,
                "support": {
                    "status": gauge_data.get("status"),
                    "generated_utc": gauge_data.get("generated_utc"),
                    "points": gauge_model.get("points"),
                    "fit": gauge_model.get("discharge_fit", {}),
                    "year_holdout": gauge_model.get("year_holdout", {}).get(
                        "discharge_candidate", {}
                    ),
                    "target_stage_support": gauge_data.get(
                        "target_stage_empirical", {}
                    ).get("support", {}),
                },
                "interpretation": "Conservative unscreened historical direct-Q check retained for schedule protection, not central forecasting.",
            }

    preferred = selection.get("preferred_screened_candidate") if selection else None
    if not preferred:
        preferred = derive_preferred_screened(pairing)
    if preferred:
        preferred_name = str(preferred.get("name"))
        pairing_model = pairing.get("models", {}).get(preferred_name, {})
        preferred_days = finite(preferred.get("rain_free_days_to_6_77_m3s"))
        if preferred_days is None:
            preferred_days = projection_days(pairing_model)
        if preferred_days is not None:
            candidates["rdpa_screened_preferred"] = {
                "name": "rdpa_screened_preferred",
                "source": preferred_name,
                "screening": "24 h, 72 h and 168 h basin-clipped RDPA dry-period screen",
                "rain_free_days_to_6_77_m3s": preferred_days,
                "eligible_for_protected_schedule": True,
                "eligible_for_live_central_replacement": False,
                "support": {
                    "pairing_status": pairing.get("status"),
                    "pairing_generated_utc": pairing.get("generated_utc"),
                    "rdpa_coverage_fraction": pairing.get("rdpa_retrieval", {}).get(
                        "coverage_fraction"
                    ),
                    "points": preferred.get("points", pairing_model.get("points")),
                    "events": preferred.get("events", pairing_model.get("events")),
                    "fit": preferred.get("fit", pairing_model.get("fit", {})),
                    "event_block_cross_validation": preferred.get(
                        "event_block_cross_validation",
                        pairing_model.get("event_block_cross_validation", {}),
                    ),
                    "event_block_rmse_per_day": preferred.get(
                        "event_block_rmse_per_day", cv_rmse(pairing_model)
                    ),
                    "skill_improvement_vs_gauge_only_pct": preferred.get(
                        "skill_improvement_vs_gauge_only_pct"
                    ),
                    "selection_status": selection.get("status") if selection else None,
                    "promotion_recommendation": selection.get(
                        "promotion_recommendation", {}
                    ),
                },
                "interpretation": "Preferred precipitation-screened direct-Q sensitivity. It is more defensible than the unscreened fit but remains shadow-only because the skill gain and independent event count are limited.",
            }

    overview = {
        "historical_gauge_status": gauge_data.get("status") if gauge_data else None,
        "rdpa_pairing_status": pairing.get("status") if pairing else None,
        "rdpa_coverage_fraction": pairing.get("rdpa_retrieval", {}).get(
            "coverage_fraction"
        )
        if pairing
        else None,
        "strict_dry_points": pairing.get("pairing", {}).get("strict_dry_points")
        if pairing
        else None,
        "moderate_dry_points": pairing.get("pairing", {}).get(
            "moderate_dry_points"
        )
        if pairing
        else None,
        "model_selection_status": selection.get("status") if selection else None,
        "preferred_screened_source": (
            preferred.get("name") if preferred else None
        ),
        "automatic_operational_promotion": False,
        "interpretation": "Historical direct-discharge fits are independent checks on the extrapolated current-limb target stage. The precipitation-screened candidate is preferred for interpretation, while the unscreened fit remains a conservative schedule sensitivity.",
    }
    return candidates, overview


def historical_distributions(
    candidates: dict,
    members: list[dict],
    generated: datetime,
    short_central: float,
    short_high: float,
) -> dict:
    results = {}
    for name, candidate in candidates.items():
        days = finite(candidate.get("rain_free_days_to_6_77_m3s"))
        if days is None:
            continue
        central_values = np.asarray(
            [member_days(member, days, short_central, "central") for member in members],
            dtype=float,
        )
        high_values = np.asarray(
            [member_days(member, days, short_high, "high") for member in members],
            dtype=float,
        )
        results[name] = {
            "status": "historical_direct_discharge_sensitivity_available",
            "rain_free_days_to_6_77_m3s": days,
            "central_response_distribution": summarize(central_values, generated),
            "upper_response_distribution": summarize(high_values, generated),
            "eligible_for_protected_schedule": bool(
                candidate.get("eligible_for_protected_schedule")
            ),
            "eligible_for_live_central_replacement": False,
            "source": candidate.get("source"),
            "screening": candidate.get("screening"),
            "support": candidate.get("support", {}),
            "interpretation": candidate.get("interpretation"),
        }
    return results


def main() -> None:
    generated = datetime.now(timezone.utc)
    base = load(BASE)
    forecast = load(FORECAST)
    probability = load(PROBABILITY)
    project = load(PROJECT_WSE)
    if probability.get("status") != "operational_project_threshold_ensemble":
        raise RuntimeError("Project-threshold ensemble is not operational")
    members = probability.get("members", [])
    if len(members) < 20:
        raise RuntimeError("Fewer than 20 project-threshold members are available")

    rating = project.get("current_event_rating_fit", {})
    stage_now = finite(project.get("current", {}).get("stage_05EA002_m"))
    if stage_now is None or finite(rating.get("slope_m3s_per_m")) is None:
        raise RuntimeError("Current stage or rating fit is unavailable")
    recession = base.get("master_recession", {})

    central_short = float(
        np.median(
            [
                max(0.0, finite(row.get("short_range_delay_days"), 0.0))
                for row in members
            ]
        )
    )
    short_low, short_central, short_high = short_range_bounds(
        forecast, central_short
    )

    configurations = {
        "optimistic_sensitivity": {
            "target_wse_m": MAIN_WSE + SITE_UNCERTAINTY_M,
            "short_delay_days": short_low,
            "response_mode": "low",
            "interpretation": "Higher allowable modelled WSE plus lower analogue-response delays. Sensitivity only.",
        },
        "central": {
            "target_wse_m": MAIN_WSE,
            "short_delay_days": short_central,
            "response_mode": "central",
            "interpretation": "Nominal 650.20 m threshold and central analogue-response delays.",
        },
        "conservative_sensitivity": {
            "target_wse_m": MAIN_WSE - SITE_UNCERTAINTY_M,
            "short_delay_days": short_high,
            "response_mode": "high",
            "interpretation": "Requires modelled WSE 0.15 m below nominal and applies upper response delays. Sensitivity only.",
        },
    }

    scenario_results = {}
    for name, configuration in configurations.items():
        target_q = q_for_wse(configuration["target_wse_m"])
        if target_q is None:
            raise RuntimeError(
                f"Unable to invert project WSE curve at {configuration['target_wse_m']} m"
            )
        target_stage = stage_from_q(target_q, rating)
        base_days = rain_free_days(stage_now, target_stage, recession)
        days = np.asarray(
            [
                member_days(
                    member,
                    base_days,
                    configuration["short_delay_days"],
                    configuration["response_mode"],
                )
                for member in members
            ],
            dtype=float,
        )
        scenario_results[name] = {
            "target_wse_m": configuration["target_wse_m"],
            "target_discharge_m3s": target_q,
            "equivalent_05EA002_stage_m": target_stage,
            "rain_free_days": base_days,
            "short_range_delay_days": configuration["short_delay_days"],
            "response_mode": configuration["response_mode"],
            "distribution": summarize(days, generated),
            "interpretation": configuration["interpretation"],
        }

    historical_candidates, historical_overview = (
        historical_direct_discharge_candidates()
    )
    historical_results = historical_distributions(
        historical_candidates,
        members,
        generated,
        short_central,
        short_high,
    )

    central_p50 = scenario_results["central"]["distribution"]["quantiles"]["p50"]
    transfer_conservative_p90 = scenario_results["conservative_sensitivity"][
        "distribution"
    ]["quantiles"]["p90"]
    optimistic_p10 = scenario_results["optimistic_sensitivity"]["distribution"][
        "quantiles"
    ]["p10"]

    protected_candidates = [
        ("transfer_and_response_sensitivity", transfer_conservative_p90)
    ]
    for name, result in historical_results.items():
        if result.get("eligible_for_protected_schedule"):
            protected_candidates.append(
                (
                    f"{name}_upper_response",
                    result["upper_response_distribution"]["quantiles"]["p90"],
                )
            )
    protected_source, protected_p90 = max(
        protected_candidates, key=lambda item: float(item[1]["days"])
    )

    screened_p50 = None
    screened_p90 = None
    if "rdpa_screened_preferred" in historical_results:
        screened_p50 = historical_results["rdpa_screened_preferred"][
            "central_response_distribution"
        ]["quantiles"]["p50"]
        screened_p90 = historical_results["rdpa_screened_preferred"][
            "upper_response_distribution"
        ]["quantiles"]["p90"]
    unscreened_p50 = None
    unscreened_p90 = None
    if "gauge_only_18_month" in historical_results:
        unscreened_p50 = historical_results["gauge_only_18_month"][
            "central_response_distribution"
        ]["quantiles"]["p50"]
        unscreened_p90 = historical_results["gauge_only_18_month"][
            "upper_response_distribution"
        ]["quantiles"]["p90"]

    planning_summary = {
        "optimistic_p10": optimistic_p10,
        "central_p50": central_p50,
        "precipitation_screened_direct_q_p50": screened_p50,
        "precipitation_screened_direct_q_upper_p90": screened_p90,
        "unscreened_direct_q_p50": unscreened_p50,
        "unscreened_direct_q_upper_p90": unscreened_p90,
        "transfer_conservative_p90": transfer_conservative_p90,
        "protected_schedule_p90": protected_p90,
        "protected_schedule_source": protected_source,
        "screened_direct_q_difference_from_live_central_days": (
            float(screened_p50["days"]) - float(central_p50["days"])
            if screened_p50 is not None
            else None
        ),
        "recommended_use": "Keep the nominal all-member p50 as the official working inspection forecast. Use the precipitation-screened direct-Q p50 as the preferred independent timing check, retain the unscreened direct-Q result as a conservative diagnostic, and protect the schedule with the latest upper-response/transfer p90. None of these sensitivities is a formal confidence limit.",
    }

    preferred_backward_compatible = (
        historical_results.get("rdpa_screened_preferred")
        or historical_results.get("gauge_only_18_month")
        or {"status": "unavailable"}
    )

    output = {
        "generated_utc": generated.isoformat(),
        "status": "operational_sensitivity_not_calibrated_probability",
        "method": "Full GEPS members are recomputed through project-transfer and rainfall-response sensitivities, plus independent gauge-only and precipitation-screened 18-month direct-discharge recession checks when available.",
        "uncertainty_components": {
            "meteorological": "Full validated GEPS member spread through 16 days.",
            "short_range_hydrologic": {
                "low_days": short_low,
                "central_days": short_central,
                "high_days": short_high,
            },
            "later_hydrologic": "Later rainfall windows use central days lost minus/plus analogue RMSE.",
            "project_transfer": {
                "nominal_wse_m": MAIN_WSE,
                "sensitivity_m": SITE_UNCERTAINTY_M,
            },
            "current_rating": "The short-window rating is retained for the official central scenario but is explicitly challenged by independent direct-discharge sensitivities.",
            "historical_direct_discharge": historical_overview,
        },
        "scenarios": scenario_results,
        "historical_direct_discharge_sensitivities": historical_results,
        "historical_direct_discharge_sensitivity": preferred_backward_compatible,
        "planning_summary": planning_summary,
        "limitations": [
            "Only two clean rainfall-response events are available for forecasting rainfall delays, so fallback response errors remain necessary.",
            "The plus/minus 0.15 m project-transfer allowance is not a fitted probability distribution.",
            "The preferred precipitation-screened direct-Q model uses 10 km RDPA, provisional rating-derived WSC discharge and a limited number of independent event blocks.",
            "The precipitation-screened direct-Q skill gain is small, so it remains a shadow timing check rather than a replacement central forecast.",
            "Local ponding and bearing capacity remain outside the hydrologic crossing-date calculation.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "central_p50": central_p50,
                "screened_direct_q_p50": screened_p50,
                "unscreened_direct_q_p50": unscreened_p50,
                "protected_schedule_p90": protected_p90,
                "protected_schedule_source": protected_source,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
