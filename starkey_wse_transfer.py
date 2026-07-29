#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
OUT = ROOT / "routing" / "starkey_wse_transfer.json"

# The project site is represented by HEC-RAS river station 18883, roughly
# 0.7 km upstream of the Starkey Road bridge sections near RS 18192/18178.
PROJECT_RIVER_STATION = 18883
BRIDGE_RIVER_STATIONS = [18192, 18178]

# Site-specific constraints at RS 18883.
# The low-flow point is an approximate field visibility threshold supplied
# by the user. The 1:20 and 1:100 values are project/design water levels and
# match the 2022 St. Albert Flood Hazard Study profile at RS 18883.
SITE_POINTS = [
    {
        "discharge_m3s": 6.77,
        "wse_m": 650.20,
        "source": "2026 user field observation at project site",
        "quality": "approximate floodplain-visibility threshold; not a concurrent surveyed WSE",
    },
    {
        "discharge_m3s": 52.0,
        "wse_m": 651.80,
        "source": "1:20-year project/design level at RS 18883",
        "quality": "modelled design WSE; 20-year discharge from 2022 flood-frequency study",
    },
    {
        "discharge_m3s": 90.0,
        "wse_m": 652.35,
        "source": "1:100-year project/design level at RS 18883",
        "quality": "modelled design WSE; 100-year discharge from 2022 flood-frequency study",
    },
]

FIELD_Q_M3S = SITE_POINTS[0]["discharge_m3s"]
FIELD_STARKEY_WSE_M = SITE_POINTS[0]["wse_m"]
MAIN_FLOODPLAIN_M = 650.20
LOW_POCKET_M = 649.60
TRANSFER_UNCERTAINTY_M = 0.15

# The 2018 high-water marks were collected at the Starkey Road bridge, not
# at the project section. They are retained only as a downstream check.
BRIDGE_HWM_2018_Q_M3S = 20.2
BRIDGE_HWM_2018_WSE_M = 651.0275


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def site_wse(q_m3s: float) -> float:
    """Quadratic interpolation of WSE versus ln(Q) through site constraints."""
    q = max(0.05, float(q_m3s))
    x = math.log(q)
    result = 0.0
    for i, point_i in enumerate(SITE_POINTS):
        xi = math.log(float(point_i["discharge_m3s"]))
        term = float(point_i["wse_m"])
        for j, point_j in enumerate(SITE_POINTS):
            if i == j:
                continue
            xj = math.log(float(point_j["discharge_m3s"]))
            term *= (x - xj) / (xi - xj)
        result += term
    return result


def q_for_wse(wse_m: float) -> float | None:
    """Invert the monotonic site curve by bisection over its useful range."""
    target = float(wse_m)
    low_q, high_q = 0.05, 200.0
    low_wse, high_wse = site_wse(low_q), site_wse(high_q)
    if not (low_wse <= target <= high_wse):
        return None
    for _ in range(100):
        mid_q = (low_q + high_q) / 2.0
        mid_wse = site_wse(mid_q)
        if mid_wse < target:
            low_q = mid_q
        else:
            high_q = mid_q
    return (low_q + high_q) / 2.0


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError("Current summary is missing")
    summary = json.loads(SUMMARY.read_text())
    gauges = summary.get("gauges", [])
    stage = next(
        (
            finite(row.get("latest"))
            for row in gauges
            if row.get("station") == "05EA002"
            and row.get("metric") == "water_level_m"
        ),
        None,
    )
    discharge = next(
        (
            finite(row.get("latest"))
            for row in gauges
            if row.get("station") == "05EA002"
            and row.get("metric") == "discharge_m3s"
        ),
        None,
    )
    if stage is None or discharge is None:
        raise RuntimeError("Current 05EA002 stage or discharge is unavailable")

    estimate = site_wse(discharge)
    bridge_check_prediction = site_wse(BRIDGE_HWM_2018_Q_M3S)
    historical_stage_at_q14 = 651.25 - 649.547

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "site_specific_rs18883_log_discharge_wse_curve",
        "project_location": {
            "river_station": PROJECT_RIVER_STATION,
            "description": "Trail and pedestrian bridge project section near Starkey",
            "downstream_bridge_sections": BRIDGE_RIVER_STATIONS,
            "note": "RS 18883 is the project section; bridge sections near RS 18192/18178 are not used as the project WSE location.",
        },
        "current_05EA002": {
            "stage_m": stage,
            "discharge_m3s": discharge,
        },
        "stage_discharge_detachment": {
            "historical_model_q14_stage_m": historical_stage_at_q14,
            "2026_rising_stage_m": 1.700,
            "2026_rising_discharge_m3s": FIELD_Q_M3S,
            "discharge_ratio_2026_to_historical": FIELD_Q_M3S / 14.0,
            "interpretation": "The same approximately 1.70 m gauge stage carried only about half the historical/model discharge; stage alone is not used to estimate project-site WSE.",
        },
        "site_constraints": SITE_POINTS,
        "transfer": {
            "type": "quadratic_lagrange_in_ln_discharge",
            "equation": "WSE is quadratic-interpolated against ln(Q) through the three RS 18883 site constraints",
            "points": SITE_POINTS,
            "operational_threshold_discharge_m3s": FIELD_Q_M3S,
            "design_check_discharge_range_m3s": [52.0, 90.0],
            "uncertainty_m": TRANSFER_UNCERTAINTY_M,
        },
        "downstream_bridge_check": {
            "river_stations": BRIDGE_RIVER_STATIONS,
            "discharge_m3s": BRIDGE_HWM_2018_Q_M3S,
            "observed_mean_wse_m": BRIDGE_HWM_2018_WSE_M,
            "project_curve_wse_m": bridge_check_prediction,
            "difference_project_curve_minus_bridge_hwm_m": bridge_check_prediction - BRIDGE_HWM_2018_WSE_M,
            "interpretation": "This is a location-mismatched reasonableness check only; bridge high-water marks are not a project-site calibration point.",
        },
        "current_starkey_estimate": {
            "central_wse_m": estimate,
            "range_m": [estimate - TRANSFER_UNCERTAINTY_M, estimate + TRANSFER_UNCERTAINTY_M],
            "depth_over_main_floodplain_m": estimate - MAIN_FLOODPLAIN_M,
            "depth_over_low_pocket_m": estimate - LOW_POCKET_M,
        },
        "site_thresholds": {
            "main_floodplain_elevation_m": MAIN_FLOODPLAIN_M,
            "main_floodplain_target_discharge_m3s": q_for_wse(MAIN_FLOODPLAIN_M),
            "low_pocket_elevation_m": LOW_POCKET_M,
            "low_pocket_curve_discharge_m3s": q_for_wse(LOW_POCKET_M),
            "low_pocket_note": "The 649.60 m pocket may remain controlled by local ponding or drainage and is not a reliable automatic work-release threshold. Raising it toward 650.20 m removes this separate low-point constraint.",
        },
        "limitations": [
            "Only one low-flow project-site field observation anchors the operational exposure threshold.",
            "The 1:20 and 1:100 project-site elevations are modelled design levels, not observed events.",
            "The 2026 field point is an approximate visibility threshold rather than a surveyed concurrent water surface.",
            "The curve between 6.77 and 52 m3/s is interpolated across a large data gap; approximately plus or minus 0.15 m remains the working uncertainty.",
            "Local blockage, downstream backwater, floodplain storage, and rising/falling-limb hysteresis can shift project-site WSE beyond the stated uncertainty.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
