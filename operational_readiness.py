#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
STARKEY = ROOT / "routing" / "forecast_starkey_wse.json"
OUT = ROOT / "forecast_v2" / "construction_readiness.json"


def finite(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    if not SUMMARY.exists() or not STARKEY.exists():
        raise FileNotFoundError("Current summary or Starkey WSE forecast is missing")

    summary = json.loads(SUMMARY.read_text())
    starkey = json.loads(STARKEY.read_text())
    target = summary.get("target_05EA002", {})
    current = starkey.get("current", {})
    threshold = starkey.get("construction_threshold", {})
    scenarios = starkey.get("scenarios", {})

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
        "hydrograph_limb": limb,
        "current_conditions": {
            "stage_05EA002_m": stage_now,
            "stage_change_24h_m": change_24h,
            "observed_discharge_05EA002_m3s": current.get("observed_discharge_05EA002_m3s"),
            "estimated_starkey_wse_m": current_wse,
            "estimated_starkey_wse_range_m": current.get("estimated_starkey_wse_range_m"),
            "estimated_depth_over_main_floodplain_m": depth_main,
            "estimated_depth_over_low_pocket_m": current.get("depth_over_low_pocket_m"),
        },
        "authoritative_operational_threshold": {
            "main_floodplain_elevation_m": threshold.get("main_floodplain_wse_m", 650.20),
            "calibrated_05EA002_discharge_m3s": threshold.get("calibrated_target_discharge_m3s", 6.77),
            "equivalent_05EA002_stage_on_current_limb_m": threshold.get("equivalent_05EA002_stage_on_current_limb_m"),
            "basis": "The user observed the approximately 650.20 m Starkey floodplain visible when the rising 05EA002 gauge crossed 1.700 m; the paired reported discharge was 6.77 m3/s. Current-limb stage is therefore recalculated from discharge rather than fixed at 1.70 m.",
        },
        "forecast_main_floodplain_exposure": schedule,
        "secondary_field_observations": {
            "rising_limb_stage_m": 1.70,
            "spring_stage_m": 1.50,
            "interpretation": "These support material seasonal/limb hysteresis and are retained as checks, not universal release thresholds.",
        },
        "low_pocket": starkey.get("low_pocket", {}),
        "decision": {
            "status": "not_ready" if depth_main is None or depth_main > 0 else "inspect_now",
            "schedule_use": "Use the dry/central/wet 650.20 m crossing dates for construction sequencing and provisional mobilization.",
            "release_rule": "Release floodplain work only after the estimated WSE is at or below 650.20 m and a direct Starkey inspection confirms drainage, access bearing and no renewed rise.",
            "site_checks": [
                "main floodplain visibly drained",
                "no sustained renewed rise forecast",
                "access and working platform have acceptable bearing capacity",
                "rutting, pumping and dewatering are manageable",
                "confirm whether the former 649.60 m pocket has in fact been raised and drains with the main platform",
            ],
        },
        "uncertainty": starkey.get("uncertainty", {}),
        "limitations": starkey.get("limitations", []) + [
            "The approximately 650.20 m field threshold and the 6.77 m3/s calibration are based on one 2026 observation rather than a surveyed concurrent Starkey water level.",
            "Construction release remains a field decision, not an automatic model output.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
