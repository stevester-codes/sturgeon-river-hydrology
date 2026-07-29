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

# The physical work-area threshold is elevation 650.20 m. The original field
# understanding placed the corresponding 05EA002 condition near stage 1.50 m.
# A later 2026 reconstruction associated a particular rising-limb condition
# near stage 1.70 m with reported discharge 6.77 m3/s. These are contextual
# observations, not one concurrent surveyed stage/discharge/project-WSE triple.
FIELD_EVIDENCE = {
    "project_work_area_threshold_wse_m": 650.20,
    "original_gauge_stage_understood_m": 1.50,
    "later_reconstructed_rising_stage_m": 1.70,
    "later_reconstructed_discharge_m3s": 6.77,
    "surveyed_concurrent_project_wse": False,
    "interpretation": (
        "Use 650.20 m as the physical project threshold and 6.77 m3/s as the "
        "current operational discharge anchor. Do not treat either 1.50 m or "
        "1.70 m at 05EA002 as a universal release stage."
    ),
}

# RS18883 design-profile points combine the 2022 flood-frequency discharges at
# 05EA002 with the matching calibrated-profile WSEs at cross-section 71.
# The project drawing independently labels the 1:20 and 1:100 levels as
# 651.80 m and 652.35 m, confirming those two project design elevations.
SITE_POINTS = [
    {
        "discharge_m3s": 6.77,
        "wse_m": 650.20,
        "return_period_years": None,
        "source": "project threshold plus reconstructed 2026 field condition",
        "quality": (
            "approximate low-flow operational anchor; project elevation was "
            "not surveyed concurrently with the 05EA002 observation"
        ),
    },
    {
        "discharge_m3s": 14.0,
        "wse_m": 650.53,
        "return_period_years": 2,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 27.0,
        "wse_m": 651.21,
        "return_period_years": 5,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 39.0,
        "wse_m": 651.54,
        "return_period_years": 10,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 52.0,
        "wse_m": 651.80,
        "return_period_years": 20,
        "source": "project drawing and 2022 St. Albert Flood Hazard Study at RS18883",
        "quality": "modelled design point; project drawing confirmation",
    },
    {
        "discharge_m3s": 64.0,
        "wse_m": 652.00,
        "return_period_years": 35,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 72.0,
        "wse_m": 652.12,
        "return_period_years": 50,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 82.0,
        "wse_m": 652.25,
        "return_period_years": 75,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 90.0,
        "wse_m": 652.35,
        "return_period_years": 100,
        "source": "project drawing and 2022 St. Albert Flood Hazard Study at RS18883",
        "quality": "modelled design point; project drawing confirmation",
    },
    {
        "discharge_m3s": 110.0,
        "wse_m": 652.59,
        "return_period_years": 200,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 130.0,
        "wse_m": 652.81,
        "return_period_years": 350,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 140.0,
        "wse_m": 652.91,
        "return_period_years": 500,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 155.0,
        "wse_m": 653.06,
        "return_period_years": 750,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
    {
        "discharge_m3s": 166.0,
        "wse_m": 653.16,
        "return_period_years": 1000,
        "source": "2022 St. Albert Flood Hazard Study, 05EA002 flow and RS18883 profile",
        "quality": "modelled design point",
    },
]

FIELD_Q_M3S = 6.77
FIELD_STARKEY_WSE_M = 650.20
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


def _sorted_points() -> list[dict]:
    points = sorted(SITE_POINTS, key=lambda row: float(row["discharge_m3s"]))
    for left, right in zip(points, points[1:]):
        if float(right["discharge_m3s"]) <= float(left["discharge_m3s"]):
            raise RuntimeError("RS18883 discharge points must be strictly increasing")
        if float(right["wse_m"]) <= float(left["wse_m"]):
            raise RuntimeError("RS18883 WSE points must be strictly increasing")
    return points


def _segment_for_q(q_m3s: float) -> tuple[dict, dict]:
    points = _sorted_points()
    q = float(q_m3s)
    if q <= float(points[0]["discharge_m3s"]):
        return points[0], points[1]
    if q >= float(points[-1]["discharge_m3s"]):
        return points[-2], points[-1]
    for left, right in zip(points, points[1:]):
        if float(left["discharge_m3s"]) <= q <= float(right["discharge_m3s"]):
            return left, right
    raise RuntimeError("Unable to bracket RS18883 discharge")


def _segment_for_wse(wse_m: float) -> tuple[dict, dict]:
    points = _sorted_points()
    target = float(wse_m)
    if target <= float(points[0]["wse_m"]):
        return points[0], points[1]
    if target >= float(points[-1]["wse_m"]):
        return points[-2], points[-1]
    for left, right in zip(points, points[1:]):
        if float(left["wse_m"]) <= target <= float(right["wse_m"]):
            return left, right
    raise RuntimeError("Unable to bracket RS18883 WSE")


def site_wse(q_m3s: float) -> float:
    """Monotonic piecewise interpolation of WSE against ln(discharge)."""
    q = max(0.05, float(q_m3s))
    left, right = _segment_for_q(q)
    q0 = float(left["discharge_m3s"])
    q1 = float(right["discharge_m3s"])
    w0 = float(left["wse_m"])
    w1 = float(right["wse_m"])
    fraction = (math.log(q) - math.log(q0)) / (math.log(q1) - math.log(q0))
    return w0 + fraction * (w1 - w0)


def q_for_wse(wse_m: float) -> float:
    """Invert the monotonic piecewise ln(discharge)-WSE curve."""
    target = float(wse_m)
    left, right = _segment_for_wse(target)
    q0 = float(left["discharge_m3s"])
    q1 = float(right["discharge_m3s"])
    w0 = float(left["wse_m"])
    w1 = float(right["wse_m"])
    fraction = (target - w0) / (w1 - w0)
    return math.exp(math.log(q0) + fraction * (math.log(q1) - math.log(q0)))


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
    design_points = [row for row in SITE_POINTS if row.get("return_period_years")]

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "site_specific_rs18883_piecewise_log_discharge_wse_curve",
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
        "evidence_hierarchy": {
            "field_context": FIELD_EVIDENCE,
            "design_profile": {
                "point_count": len(design_points),
                "discharge_range_m3s": [
                    min(float(row["discharge_m3s"]) for row in design_points),
                    max(float(row["discharge_m3s"]) for row in design_points),
                ],
                "project_drawing_confirmed_levels": {
                    "20_year_wse_m": 651.80,
                    "100_year_wse_m": 652.35,
                },
                "interpretation": (
                    "The project drawing levels match the flood-study calibrated "
                    "profile at RS18883. The full flood-frequency table supplies "
                    "the matching design discharges."
                ),
            },
        },
        "stage_discharge_detachment": {
            "historical_model_q14_stage_m": historical_stage_at_q14,
            "original_project_condition_stage_understood_m": 1.50,
            "later_reconstructed_rising_stage_m": 1.700,
            "later_reconstructed_rising_discharge_m3s": FIELD_Q_M3S,
            "discharge_ratio_2026_to_historical": FIELD_Q_M3S / 14.0,
            "interpretation": (
                "The project condition was originally understood near 1.50 m, "
                "while a later reconstructed event placed a relevant rising-limb "
                "condition near 1.70 m but only 6.77 m3/s. Stage alone is therefore "
                "not used as a universal project-release threshold."
            ),
        },
        "site_constraints": SITE_POINTS,
        "transfer": {
            "type": "piecewise_linear_in_ln_discharge",
            "equation": (
                "WSE is linearly interpolated against ln(Q) between the low-flow "
                "anchor and every available RS18883 design-profile point"
            ),
            "points": SITE_POINTS,
            "operational_threshold_discharge_m3s": FIELD_Q_M3S,
            "design_check_discharge_range_m3s": [14.0, 166.0],
            "uncertainty_m": TRANSFER_UNCERTAINTY_M,
            "uncertainty_basis": (
                "Retained primarily for the unsurveyed low-flow anchor and local "
                "backwater/storage effects; the high-flow profile itself is now "
                "densely constrained by design points."
            ),
        },
        "downstream_bridge_check": {
            "river_stations": BRIDGE_RIVER_STATIONS,
            "discharge_m3s": BRIDGE_HWM_2018_Q_M3S,
            "observed_mean_wse_m": BRIDGE_HWM_2018_WSE_M,
            "project_curve_wse_m": bridge_check_prediction,
            "difference_project_curve_minus_bridge_hwm_m": bridge_check_prediction
            - BRIDGE_HWM_2018_WSE_M,
            "interpretation": "This is a location-mismatched reasonableness check only; bridge high-water marks are not a project-site calibration point.",
        },
        "current_starkey_estimate": {
            "central_wse_m": estimate,
            "range_m": [
                estimate - TRANSFER_UNCERTAINTY_M,
                estimate + TRANSFER_UNCERTAINTY_M,
            ],
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
            "The operational 650.20 m / 6.77 m3/s anchor is reconstructed from project and gauge evidence rather than one concurrent surveyed project-site water level.",
            "The original approximately 1.50 m and later approximately 1.70 m gauge-stage interpretations are contextual observations, not universal thresholds.",
            "Design points from 14 to 166 m3/s are modelled flood-profile values rather than observed project-site events.",
            "Below 6.77 m3/s the curve extrapolates the first log-discharge segment and should not control construction release without field inspection.",
            "Local blockage, downstream backwater, floodplain storage, and rising/falling-limb effects can shift project-site WSE beyond the stated uncertainty.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
