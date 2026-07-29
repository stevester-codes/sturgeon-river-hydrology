#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
GAUGE = ROOT / "raw" / "wateroffice" / "05EA002.csv"
OUT = ROOT / "diagnostics" / "hysteresis_diagnostics.json"
TARGET_Q = 6.77


def parse_gauge() -> pd.DataFrame:
    frame = pd.read_csv(GAUGE, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = next(column for column in frame.columns if column.lower() == "date")
    parameter_column = next(
        column
        for column in frame.columns
        if "parameter" in column.lower() or "paramètre" in column.lower()
    )
    value_column = next(
        column
        for column in frame.columns
        if "value" in column.lower() or "valeur" in column.lower()
    )
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    frame[parameter_column] = pd.to_numeric(frame[parameter_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, parameter_column, value_column])
    pivot = frame.pivot_table(
        index=date_column,
        columns=parameter_column,
        values=value_column,
        aggfunc="median",
    ).rename(columns={46: "stage_m", 47: "flow_m3s"})
    return pivot.sort_index().resample("1h").median().interpolate(limit=2).dropna()


def fit_limb(frame: pd.DataFrame) -> dict | None:
    if len(frame) < 12:
        return None
    stage = frame.stage_m.to_numpy(float)
    discharge = frame.flow_m3s.to_numpy(float)
    slope, intercept = np.polyfit(stage, discharge, 1)
    slope = float(slope)
    intercept = float(intercept)
    if slope <= 0:
        return None
    predicted = slope * stage + intercept
    residuals = discharge - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((discharge - np.mean(discharge)) ** 2))
    return {
        "slope_m3s_per_m": slope,
        "intercept_m3s": intercept,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "rmse_m3s": float(np.sqrt(np.mean(residuals**2))),
        "n": int(len(frame)),
        "stage_range_m": [float(np.min(stage)), float(np.max(stage))],
        "discharge_range_m3s": [float(np.min(discharge)), float(np.max(discharge))],
        "start_utc": frame.index.min().isoformat(),
        "end_utc": frame.index.max().isoformat(),
    }


def stage_for_q(discharge: float, fit: dict) -> float:
    return (float(discharge) - float(fit["intercept_m3s"])) / float(
        fit["slope_m3s_per_m"]
    )


def main() -> None:
    if not GAUGE.exists():
        raise FileNotFoundError(GAUGE)
    frame = parse_gauge()
    if len(frame) < 24:
        raise RuntimeError("Insufficient paired stage-discharge observations")

    # Six-hour stage movement is used to avoid classifying single-hour noise as
    # a limb change. Nearly flat points are excluded from the loop comparison.
    frame["stage_change_6h_m"] = frame.stage_m.diff(6)
    rising = frame[frame.stage_change_6h_m >= 0.003].copy()
    falling = frame[frame.stage_change_6h_m <= -0.003].copy()
    flat = frame[(frame.stage_change_6h_m > -0.003) & (frame.stage_change_6h_m < 0.003)]

    rising_fit = fit_limb(rising)
    falling_fit = fit_limb(falling)
    comparison = {
        "status": "insufficient_rising_and_falling_support",
        "target_discharge_m3s": TARGET_Q,
    }

    if rising_fit and falling_fit:
        overlap_low = max(
            float(rising_fit["discharge_range_m3s"][0]),
            float(falling_fit["discharge_range_m3s"][0]),
        )
        overlap_high = min(
            float(rising_fit["discharge_range_m3s"][1]),
            float(falling_fit["discharge_range_m3s"][1]),
        )
        if overlap_high > overlap_low:
            q_values = np.linspace(overlap_low, overlap_high, 25)
            differences = np.asarray(
                [
                    stage_for_q(q, rising_fit) - stage_for_q(q, falling_fit)
                    for q in q_values
                ],
                dtype=float,
            )
            target_supported = overlap_low <= TARGET_Q <= overlap_high
            target_difference = (
                stage_for_q(TARGET_Q, rising_fit)
                - stage_for_q(TARGET_Q, falling_fit)
            )
            comparison = {
                "status": "apparent_loop_quantified",
                "overlap_discharge_range_m3s": [overlap_low, overlap_high],
                "median_rising_minus_falling_stage_m": float(np.median(differences)),
                "maximum_absolute_stage_separation_m": float(
                    np.max(np.abs(differences))
                ),
                "target_discharge_m3s": TARGET_Q,
                "target_inside_overlap": bool(target_supported),
                "target_rising_stage_m": stage_for_q(TARGET_Q, rising_fit),
                "target_falling_stage_m": stage_for_q(TARGET_Q, falling_fit),
                "target_rising_minus_falling_stage_m": target_difference,
            }
        else:
            comparison = {
                "status": "limb_fits_have_no_overlapping_discharge_range",
                "target_discharge_m3s": TARGET_Q,
                "rising_discharge_range_m3s": rising_fit["discharge_range_m3s"],
                "falling_discharge_range_m3s": falling_fit["discharge_range_m3s"],
            }

    latest_change = float(frame.stage_m.iloc[-1] - frame.stage_m.iloc[-7])
    current_limb = (
        "rising"
        if latest_change >= 0.003
        else ("falling" if latest_change <= -0.003 else "approximately_flat")
    )
    apparent_separation = comparison.get("maximum_absolute_stage_separation_m")
    if apparent_separation is None:
        confidence_effect = "unknown"
    elif apparent_separation >= 0.10:
        confidence_effect = "material_apparent_hysteresis"
    elif apparent_separation >= 0.05:
        confidence_effect = "moderate_apparent_hysteresis"
    else:
        confidence_effect = "small_apparent_hysteresis"

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only",
        "current_limb": current_limb,
        "observation_count": int(len(frame)),
        "classification": {
            "rising_points": int(len(rising)),
            "falling_points": int(len(falling)),
            "approximately_flat_points": int(len(flat)),
            "limb_threshold_6h_stage_change_m": 0.003,
        },
        "rising_fit": rising_fit,
        "falling_fit": falling_fit,
        "loop_comparison": comparison,
        "confidence_effect": confidence_effect,
        "interpretation": (
            "Separate recent rising and falling relationships are compared over their common discharge range. "
            "The result measures apparent loop separation in the provisional WSC stage-discharge series."
        ),
        "limitations": [
            "WSC discharge is rating-derived rather than an independent velocity-area measurement at every hour, so the apparent loop may partly reflect the station rating procedure.",
            "This diagnostic does not directly measure project-site RS18883 hysteresis, backwater or floodplain storage.",
            "A project-site logger tied to CGVD28 is required to calibrate actual rising-versus-falling WSE differences at the work site.",
            "The target-discharge comparison is extrapolative when 6.77 m3/s lies outside the overlap of recent rising and falling observations.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "current_limb": current_limb,
                "rising_points": len(rising),
                "falling_points": len(falling),
                "comparison_status": comparison.get("status"),
                "confidence_effect": confidence_effect,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
