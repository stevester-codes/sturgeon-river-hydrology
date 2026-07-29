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


def short_range_bounds(forecast: dict, central_fallback: float) -> tuple[float, float, float]:
    candidates = [
        row
        for row in forecast.get("deterministic_scenarios", [])
        if str(row.get("model")) == "HRDPS"
        and int(row.get("horizon_h", 0)) == 48
    ]
    if not candidates:
        return central_fallback, central_fallback, central_fallback
    row = candidates[0]
    central = finite(
        row.get("analog_prediction", {}).get("days_lost"), central_fallback
    )
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
    member: dict,
    base_days: float,
    short_delay: float,
    response_mode: str,
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
    for name, q in (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90)):
        quantiles[name] = date_record(generated, float(np.quantile(values, q)))
    return {
        "member_count": int(len(values)),
        "mean": date_record(generated, float(np.mean(values))),
        "earliest": date_record(generated, float(np.min(values))),
        "latest": date_record(generated, float(np.max(values))),
        "standard_deviation_days": float(np.std(values, ddof=0)),
        "quantiles": quantiles,
    }


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
            [max(0.0, finite(row.get("short_range_delay_days"), 0.0)) for row in members]
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
            "interpretation": "Higher allowable modelled WSE plus lower analogue response delays. This is a sensitivity bound, not a stated probability.",
        },
        "central": {
            "target_wse_m": MAIN_WSE,
            "short_delay_days": short_central,
            "response_mode": "central",
            "interpretation": "Nominal 650.20 m threshold and central analogue response delays.",
        },
        "conservative_sensitivity": {
            "target_wse_m": MAIN_WSE - SITE_UNCERTAINTY_M,
            "short_delay_days": short_high,
            "response_mode": "high",
            "interpretation": "Requires the modelled WSE to fall 0.15 m below the nominal threshold and applies upper analogue-response delays. This is a planning sensitivity bound, not a calibrated confidence interval.",
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

    central_p50 = scenario_results["central"]["distribution"]["quantiles"]["p50"]
    conservative_p90 = scenario_results["conservative_sensitivity"]["distribution"]["quantiles"]["p90"]
    optimistic_p10 = scenario_results["optimistic_sensitivity"]["distribution"]["quantiles"]["p10"]
    output = {
        "generated_utc": generated.isoformat(),
        "status": "operational_sensitivity_not_calibrated_probability",
        "method": (
            "All GEPS member rainfall sequences are recomputed for three uncertainty configurations. "
            "The site-transfer sensitivity shifts the required modelled RS18883 WSE by plus or minus 0.15 m. "
            "Hydrologic response uses lower, central or upper short-range delays and later-window analogue RMSE sensitivity."
        ),
        "uncertainty_components": {
            "meteorological": "Full validated GEPS member spread through 16 days.",
            "short_range_hydrologic": {
                "low_days": short_low,
                "central_days": short_central,
                "high_days": short_high,
            },
            "later_hydrologic": "Each later rainfall window uses central days lost minus/plus its analogue RMSE for low/high sensitivity.",
            "project_transfer": {
                "nominal_wse_m": MAIN_WSE,
                "sensitivity_m": SITE_UNCERTAINTY_M,
            },
            "rating": "The current-limb stage-discharge fit is used in every configuration; uncertainty in the provisional WSC rating is described separately and is not assigned a false probability distribution.",
        },
        "scenarios": scenario_results,
        "planning_summary": {
            "optimistic_p10": optimistic_p10,
            "central_p50": central_p50,
            "conservative_p90": conservative_p90,
            "recommended_use": (
                "Use central p50 as the working inspection forecast and conservative-sensitivity p90 as the protected schedule contingency. "
                "These bounds deliberately include more than weather spread but remain sensitivity results, not formal confidence limits."
            ),
        },
        "limitations": [
            "Analogue RMSE is unavailable as formal cross-validation with only two clean training events, so fallback error magnitudes are used.",
            "The plus/minus 0.15 m project-transfer sensitivity is an engineering working allowance, not a fitted statistical distribution.",
            "The same current-limb WSC rating fit is used in all cases; provisional-rating error is not probabilistically quantified.",
            "Local ponding and bearing capacity remain outside the hydrologic crossing-date calculation.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "optimistic_p10": optimistic_p10,
                "central_p50": central_p50,
                "conservative_p90": conservative_p90,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
