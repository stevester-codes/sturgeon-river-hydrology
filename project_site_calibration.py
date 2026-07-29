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
OBSERVATIONS = Path("project_site_observations.csv")
OUT = ROOT / "diagnostics" / "project_site_calibration.json"
MAX_PAIR_MINUTES = 30
MIN_TOTAL = 6
MIN_PER_LIMB = 3


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
    ).rename(columns={46: "stage_m", 47: "discharge_m3s"})
    pivot = pivot.sort_index().resample("1h").median().interpolate(limit=2)
    pivot["stage_change_6h_m"] = pivot.stage_m.diff(6)
    pivot["limb"] = np.where(
        pivot.stage_change_6h_m >= 0.003,
        "rising",
        np.where(pivot.stage_change_6h_m <= -0.003, "falling", "approximately_flat"),
    )
    return pivot.dropna(subset=["stage_m", "discharge_m3s"])


def read_observations() -> pd.DataFrame:
    if not OBSERVATIONS.exists():
        raise FileNotFoundError(OBSERVATIONS)
    frame = pd.read_csv(OBSERVATIONS)
    if frame.empty:
        return frame
    required = ["observed_utc", "project_wse_m_cgvd28"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise RuntimeError(f"Project observation file missing columns: {missing}")
    frame["observed_utc"] = pd.to_datetime(frame.observed_utc, utc=True, errors="coerce")
    frame["project_wse_m_cgvd28"] = pd.to_numeric(
        frame.project_wse_m_cgvd28, errors="coerce"
    )
    frame["measurement_uncertainty_m"] = pd.to_numeric(
        frame.get("measurement_uncertainty_m", 0.03), errors="coerce"
    ).fillna(0.03).clip(lower=0.005)
    return frame.dropna(subset=required).sort_values("observed_utc")


def pair_observations(observations: pd.DataFrame, gauge: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, observation in observations.iterrows():
        when = pd.Timestamp(observation.observed_utc)
        location = gauge.index.get_indexer([when], method="nearest")[0]
        if location < 0:
            continue
        matched_time = gauge.index[location]
        difference_min = abs((matched_time - when).total_seconds()) / 60.0
        if difference_min > MAX_PAIR_MINUTES:
            continue
        matched = gauge.iloc[location]
        row = observation.to_dict()
        row.update(
            {
                "gauge_utc": matched_time,
                "pair_difference_minutes": difference_min,
                "stage_05EA002_m": float(matched.stage_m),
                "discharge_05EA002_m3s": float(matched.discharge_m3s),
                "stage_change_6h_m": float(matched.stage_change_6h_m)
                if np.isfinite(matched.stage_change_6h_m)
                else None,
                "hydrograph_limb": str(matched.limb),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def design_matrix(discharge: np.ndarray, degree: int) -> np.ndarray:
    x = np.log(np.maximum(discharge, 0.05))
    return np.column_stack([x**power for power in range(degree + 1)])


def fit_curve(frame: pd.DataFrame, degree: int = 1) -> dict | None:
    if len(frame) < degree + 3:
        return None
    q = frame.discharge_05EA002_m3s.to_numpy(float)
    y = frame.project_wse_m_cgvd28.to_numpy(float)
    uncertainty = frame.measurement_uncertainty_m.to_numpy(float)
    weights = 1.0 / np.maximum(uncertainty, 0.005) ** 2
    x = design_matrix(q, degree)
    weighted_x = x * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
    prediction = x @ coefficients
    residuals = prediction - y
    return {
        "degree_in_ln_q": degree,
        "coefficients": [float(value) for value in coefficients],
        "n": int(len(frame)),
        "discharge_range_m3s": [float(np.min(q)), float(np.max(q))],
        "wse_range_m": [float(np.min(y)), float(np.max(y))],
        "rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "mae_m": float(np.mean(np.abs(residuals))),
        "bias_m": float(np.mean(residuals)),
    }


def predict(discharge: float, fit: dict) -> float:
    x = math.log(max(float(discharge), 0.05))
    return sum(
        float(coefficient) * x**power
        for power, coefficient in enumerate(fit["coefficients"])
    )


def leave_one_out(frame: pd.DataFrame, degree: int) -> dict | None:
    if len(frame) < degree + 4:
        return None
    errors = []
    rows = []
    for index in frame.index:
        training = frame.drop(index=index)
        fit = fit_curve(training, degree)
        if fit is None:
            return None
        observation = frame.loc[index]
        prediction = predict(observation.discharge_05EA002_m3s, fit)
        error = prediction - float(observation.project_wse_m_cgvd28)
        errors.append(error)
        rows.append(
            {
                "observed_utc": pd.Timestamp(observation.observed_utc).isoformat(),
                "observed_wse_m": float(observation.project_wse_m_cgvd28),
                "predicted_wse_m": prediction,
                "error_m": error,
            }
        )
    values = np.asarray(errors, dtype=float)
    return {
        "n": int(len(values)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "mae_m": float(np.mean(np.abs(values))),
        "bias_m": float(np.mean(values)),
        "rows": rows,
    }


def main() -> None:
    generated = datetime.now(timezone.utc)
    observations = read_observations()
    output = {
        "generated_utc": generated.isoformat(),
        "status": "awaiting_project_observations",
        "mode": "shadow_only_no_automatic_transfer_update",
        "source_file": str(OBSERVATIONS),
        "minimum_requirements": {
            "total_paired_observations": MIN_TOTAL,
            "rising_observations_for_limb_curves": MIN_PER_LIMB,
            "falling_observations_for_limb_curves": MIN_PER_LIMB,
            "maximum_pair_difference_minutes": MAX_PAIR_MINUTES,
        },
        "counts": {
            "entered_observations": int(len(observations)),
            "paired_observations": 0,
            "rising": 0,
            "falling": 0,
            "approximately_flat": 0,
        },
        "paired_observations": [],
        "single_curve": None,
        "limb_curves": {},
        "comparison": {},
        "promotion_recommendation": {
            "recommended": False,
            "reason": "No paired project observations are available yet.",
        },
        "limitations": [
            "The operational RS18883 transfer remains authoritative until a candidate passes sample-size, datum, residual and leave-one-out checks.",
            "A fitted separation between limbs is not accepted unless it exceeds measurement uncertainty and repeats across observations.",
            "Project WSE measurement datum and sensor/barometric corrections must be independently verified.",
        ],
    }

    if not observations.empty:
        gauge = parse_gauge()
        paired = pair_observations(observations, gauge)
        counts = paired.hydrograph_limb.value_counts().to_dict() if not paired.empty else {}
        output["counts"] = {
            "entered_observations": int(len(observations)),
            "paired_observations": int(len(paired)),
            "rising": int(counts.get("rising", 0)),
            "falling": int(counts.get("falling", 0)),
            "approximately_flat": int(counts.get("approximately_flat", 0)),
        }
        if not paired.empty:
            serializable = paired.copy()
            for column in ["observed_utc", "gauge_utc"]:
                serializable[column] = pd.to_datetime(
                    serializable[column], utc=True, errors="coerce"
                ).map(lambda value: value.isoformat() if pd.notna(value) else None)
            output["paired_observations"] = serializable.to_dict(orient="records")

        if len(paired) >= MIN_TOTAL:
            single = fit_curve(paired, degree=1)
            if single is not None:
                single["leave_one_out"] = leave_one_out(paired, degree=1)
            output["single_curve"] = single
            for limb in ["rising", "falling"]:
                subset = paired[paired.hydrograph_limb == limb]
                if len(subset) >= MIN_PER_LIMB:
                    fit = fit_curve(subset, degree=1)
                    if fit is not None:
                        fit["leave_one_out"] = leave_one_out(subset, degree=1)
                    output["limb_curves"][limb] = fit

            rising = output["limb_curves"].get("rising")
            falling = output["limb_curves"].get("falling")
            comparison = {
                "separate_limb_curves_available": bool(rising and falling),
                "common_discharge_checks": [],
            }
            if rising and falling:
                low = max(
                    rising["discharge_range_m3s"][0],
                    falling["discharge_range_m3s"][0],
                )
                high = min(
                    rising["discharge_range_m3s"][1],
                    falling["discharge_range_m3s"][1],
                )
                if high > low:
                    for discharge in np.linspace(low, high, 5):
                        rising_wse = predict(discharge, rising)
                        falling_wse = predict(discharge, falling)
                        comparison["common_discharge_checks"].append(
                            {
                                "discharge_m3s": float(discharge),
                                "rising_wse_m": rising_wse,
                                "falling_wse_m": falling_wse,
                                "rising_minus_falling_m": rising_wse - falling_wse,
                            }
                        )
            output["comparison"] = comparison

            loo = single.get("leave_one_out") if single else None
            datum_ok = bool(
                observations.measurement_uncertainty_m.max() <= 0.10
            )
            cv_ok = bool(loo and loo.get("rmse_m", 1.0) <= 0.10)
            coverage_ok = bool(
                paired.project_wse_m_cgvd28.min() <= 650.20
                <= paired.project_wse_m_cgvd28.max()
            )
            recommended = bool(datum_ok and cv_ok and coverage_ok)
            output["status"] = (
                "candidate_transfer_ready_for_engineering_review"
                if recommended
                else "observations_available_but_promotion_checks_not_met"
            )
            output["promotion_recommendation"] = {
                "recommended": recommended,
                "datum_uncertainty_check": datum_ok,
                "leave_one_out_rmse_check": cv_ok,
                "threshold_bracketing_check": coverage_ok,
                "automatic_promotion_enabled": False,
                "reason": (
                    "Candidate meets screening checks but still requires engineering review and rollback-ready comparison."
                    if recommended
                    else "One or more sample, datum, validation or threshold-coverage checks are not met."
                ),
            }
        else:
            output["status"] = "insufficient_paired_project_observations"
            output["promotion_recommendation"] = {
                "recommended": False,
                "reason": f"Only {len(paired)} paired observations; at least {MIN_TOTAL} are required.",
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": output["status"],
                "counts": output["counts"],
                "promotion_recommended": output["promotion_recommendation"]["recommended"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
