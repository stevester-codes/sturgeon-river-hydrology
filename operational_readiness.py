#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("sturgeon_pipeline_output")
BASE = ROOT / "calibration" / "calibration.json"
SUMMARY = ROOT / "summary" / "summary.json"
ENSEMBLE = ROOT / "forecast_v2" / "ensemble_paths_v2.json"
OUT = ROOT / "forecast_v2" / "construction_readiness.json"

# These are field observations, not universal hydraulic equivalences.
THRESHOLDS = {
    "inspection_trigger": {
        "stage_m": 1.70,
        "basis": "Observed in 2026 on the rising limb when the Starkey floodplain was visible.",
        "use": "Mobilization/site-inspection trigger only; not an unconditional work release.",
    },
    "conservative_release_check": {
        "stage_m": 1.50,
        "basis": "Observed during spring 2026 as the approximate Starkey exposure threshold.",
        "use": "Conservative field-release check, still subject to site drainage and bearing-capacity inspection.",
    },
}


def finite(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def recession_rate(model: dict, stage_m: float) -> float:
    intercept = finite(model.get("intercept_m_per_day"), -0.03)
    coefficient = finite(model.get("stage_coefficient_per_day"), -0.007)
    return min(-0.001, intercept + coefficient * stage_m)


def rain_free_days(stage_now: float, target: float, model: dict) -> float:
    stage = stage_now
    hours = 0
    max_hours = 24 * 90
    while stage > target and hours < max_hours:
        stage += recession_rate(model, stage) / 24.0
        hours += 1
    return hours / 24.0


def scenario_delay_to_target(scenario: dict, base_days: float) -> tuple[float, list[dict]]:
    short = finite(scenario.get("short_range", {}).get("delay_days"), 0.0)
    total = max(0.0, short)
    applied = [{"start_h": 0, "end_h": 48, "delay_days": total, "source": "HRDPS"}]

    for window in sorted(scenario.get("later_windows", []), key=lambda x: finite(x.get("start_h"), 0.0)):
        start_h = finite(window.get("start_h"), 0.0)
        delay = max(0.0, finite(window.get("days_lost_central"), 0.0))
        if delay <= 0:
            continue
        crossing_days = base_days + total
        if start_h / 24.0 <= crossing_days:
            total += delay
            applied.append(
                {
                    "start_h": start_h,
                    "end_h": finite(window.get("end_h"), start_h),
                    "delay_days": delay,
                    "source": "GEPS member window analogue",
                }
            )
    return total, applied


def main() -> None:
    if not BASE.exists() or not SUMMARY.exists() or not ENSEMBLE.exists():
        raise FileNotFoundError("Required calibration, summary, or GEPS ensemble output is missing")

    base = json.loads(BASE.read_text())
    summary = json.loads(SUMMARY.read_text())
    ensemble = json.loads(ENSEMBLE.read_text())
    target = summary.get("target_05EA002", {})
    stage_now = finite(target.get("latest"))
    change_24h = finite(target.get("change_24h"), 0.0)
    stage_time = target.get("latest_utc")
    if stage_now is None or not stage_time:
        raise RuntimeError("Current 05EA002 stage is unavailable")

    if change_24h > 0.005:
        limb = "rising"
    elif change_24h < -0.005:
        limb = "falling"
    else:
        limb = "approximately_flat"

    recession_model = base.get("master_recession", {})
    generated = datetime.now(timezone.utc)
    results = {}

    for threshold_name, threshold in THRESHOLDS.items():
        target_stage = float(threshold["stage_m"])
        dry_days = rain_free_days(stage_now, target_stage, recession_model)
        scenarios = {}
        for scenario_name in ("dry", "central", "wet"):
            scenario = ensemble.get("scenarios", {}).get(scenario_name, {})
            delay_days, applied = scenario_delay_to_target(scenario, dry_days)
            total_days = dry_days + delay_days
            scenarios[scenario_name] = {
                "rain_free_days": dry_days,
                "forecast_rain_delay_days": delay_days,
                "projected_days": total_days,
                "projected_date_utc": (generated + timedelta(days=total_days)).date().isoformat(),
                "applied_rain_windows": applied,
            }
        results[threshold_name] = {**threshold, "scenarios": scenarios}

    output = {
        "generated_utc": generated.isoformat(),
        "latest_stage_utc": stage_time,
        "latest_stage_m": stage_now,
        "change_24h_m": change_24h,
        "hydrograph_limb": limb,
        "thresholds": results,
        "recommended_use": {
            "schedule_planning": "Use the 1.70 m forecast to schedule inspection and provisional mobilization.",
            "work_release": "Do not release floodplain earthworks solely at 1.70 m on a falling limb. Confirm field conditions; use the 1.50 m forecast as the conservative fallback threshold.",
            "site_checks": [
                "floodplain visibly drained",
                "no sustained renewed rise forecast",
                "access and working platform have acceptable bearing capacity",
                "rutting, pumping, and dewatering are manageable",
            ],
        },
        "limitations": [
            "The 1.70 m observation was made on a rising limb; the same gauge stage can correspond to different local inundation on a falling limb because of storage and drainage hysteresis.",
            "The 1.50 m observation was made in spring; summer vegetation, channel conveyance, and antecedent saturation may alter the relationship.",
            "Both thresholds are provisional field correlations from one year, not statistically calibrated hydraulic stage-transfer relationships.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
