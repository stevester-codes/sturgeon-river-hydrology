#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("sturgeon_pipeline_output")
SUMMARY = ROOT / "summary" / "summary.json"
OUT = ROOT / "routing" / "starkey_wse_transfer.json"

# Approximate field calibration supplied by the user:
# at the 05EA002 rising-limb crossing of 1.700 m on 2026-06-24 05:25Z,
# reported discharge was 6.77 m3/s and the main Starkey floodplain at
# approximately El. 650.20 m was visible.
FIELD_Q_M3S = 6.77
FIELD_STARKEY_WSE_M = 650.20

# 2018 observed high-water marks near Starkey at Q=20.2 m3/s:
# 651.04, 651.02, 651.02 and 651.03 m (mean 651.0275 m).
HWM_2018_Q_M3S = 20.2
HWM_2018_STARKEY_WSE_M = 651.0275

# Independent HEC-RAS reasonableness check from the 2022 flood study:
# at Q=14 m3/s, modelled Starkey WSE is 650.49 m. The model underpredicted
# the 2018 Starkey high-water marks by about 0.14 m, giving a corrected
# check value near 650.63 m.
HECRAS_CHECK_Q_M3S = 14.0
HECRAS_CHECK_MODEL_WSE_M = 650.49
HECRAS_2018_BIAS_M = HWM_2018_STARKEY_WSE_M - 650.89

MAIN_FLOODPLAIN_M = 650.20
LOW_POCKET_M = 649.60
TRANSFER_UNCERTAINTY_M = 0.15


def finite(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def linear_transfer(q_m3s: float) -> float:
    slope = (HWM_2018_STARKEY_WSE_M - FIELD_STARKEY_WSE_M) / (
        HWM_2018_Q_M3S - FIELD_Q_M3S
    )
    intercept = FIELD_STARKEY_WSE_M - slope * FIELD_Q_M3S
    return intercept + slope * q_m3s


def q_for_wse(wse_m: float) -> float | None:
    slope = (HWM_2018_STARKEY_WSE_M - FIELD_STARKEY_WSE_M) / (
        HWM_2018_Q_M3S - FIELD_Q_M3S
    )
    intercept = FIELD_STARKEY_WSE_M - slope * FIELD_Q_M3S
    q = (wse_m - intercept) / slope
    return q if q >= 0 else None


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

    slope = (HWM_2018_STARKEY_WSE_M - FIELD_STARKEY_WSE_M) / (
        HWM_2018_Q_M3S - FIELD_Q_M3S
    )
    intercept = FIELD_STARKEY_WSE_M - slope * FIELD_Q_M3S
    estimate = linear_transfer(discharge)
    historical_stage_at_q14 = 651.25 - 649.547
    hecras_corrected = HECRAS_CHECK_MODEL_WSE_M + HECRAS_2018_BIAS_M

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "discharge_based_empirical_transfer_with_hec_ras_check",
        "current_05EA002": {
            "stage_m": stage,
            "discharge_m3s": discharge,
        },
        "stage_discharge_detachment": {
            "historical_model_q14_stage_m": historical_stage_at_q14,
            "2026_rising_stage_m": 1.700,
            "2026_rising_discharge_m3s": FIELD_Q_M3S,
            "discharge_ratio_2026_to_historical": FIELD_Q_M3S / HECRAS_CHECK_Q_M3S,
            "interpretation": "The same approximately 1.70 m gauge stage carried only about half the historical/model discharge; therefore stage alone is not used to estimate Starkey WSE.",
        },
        "calibration_points": [
            {
                "source": "2026 user field observation matched to 05EA002 rising crossing",
                "discharge_m3s": FIELD_Q_M3S,
                "starkey_wse_m": FIELD_STARKEY_WSE_M,
                "quality": "approximate field threshold",
            },
            {
                "source": "2018 surveyed Starkey high-water marks",
                "discharge_m3s": HWM_2018_Q_M3S,
                "starkey_wse_m": HWM_2018_STARKEY_WSE_M,
                "quality": "surveyed event mean",
            },
        ],
        "transfer": {
            "equation": "Starkey_WSE_m = intercept + slope * Q_05EA002_m3s",
            "intercept_m": intercept,
            "slope_m_per_m3s": slope,
            "intended_discharge_range_m3s": [FIELD_Q_M3S, HWM_2018_Q_M3S],
            "uncertainty_m": TRANSFER_UNCERTAINTY_M,
        },
        "hec_ras_check": {
            "discharge_m3s": HECRAS_CHECK_Q_M3S,
            "modelled_starkey_wse_m": HECRAS_CHECK_MODEL_WSE_M,
            "2018_bias_correction_m": HECRAS_2018_BIAS_M,
            "bias_corrected_wse_m": hecras_corrected,
            "empirical_transfer_wse_m": linear_transfer(HECRAS_CHECK_Q_M3S),
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
            "low_pocket_target_discharge_m3s": q_for_wse(LOW_POCKET_M),
            "low_pocket_note": "The 649.60 m pocket lies below the positive-flow intercept of this limited transfer and may remain ponded or hydraulically controlled independently; raising it toward 650.20 m materially improves drainage readiness.",
        },
        "limitations": [
            "Only two low-to-moderate-flow calibration points are available.",
            "The 2026 field point is an approximate visibility threshold rather than a surveyed water surface.",
            "The relationship is not extrapolated confidently below 6.77 or above 20.2 m3/s.",
            "Local blockage, downstream backwater, floodplain storage, and rising/falling limb hysteresis may shift Starkey WSE by more than the stated uncertainty.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
