#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path("sturgeon_pipeline_output")
GAUGE = ROOT / "raw" / "wateroffice" / "05EA002.csv"
SUMMARY = ROOT / "summary" / "summary.json"
ENSEMBLE = ROOT / "forecast_v2" / "ensemble_paths_v2.json"
TRANSFER = ROOT / "routing" / "starkey_wse_transfer.json"
OUT = ROOT / "routing" / "forecast_starkey_wse.json"

FIELD_TARGET_Q = 6.77
MAIN_FLOODPLAIN_WSE = 650.20
LOW_POCKET_WSE = 649.60


def finite(value, default=None):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def normalized(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_col(fields: list[str], needle: str) -> str | None:
    target = normalized(needle)
    for field in fields:
        if target in normalized(field):
            return field
    return None


def read_pairs() -> list[tuple[datetime, float, float]]:
    with GAUGE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    date_col = find_col(fields, "date")
    param_col = find_col(fields, "parameter")
    value_col = find_col(fields, "value")
    if not date_col or not param_col or not value_col:
        raise RuntimeError(f"Unrecognized WaterOffice columns: {fields}")
    values: dict[datetime, dict[str, float]] = {}
    for row in rows:
        timestamp = parse_dt(row.get(date_col, ""))
        value = finite(row.get(value_col))
        parameter = str(row.get(param_col, "")).strip()
        if timestamp is None or value is None or parameter not in {"46", "47"}:
            continue
        values.setdefault(timestamp, {})[parameter] = value
    return sorted(
        (timestamp, item["46"], item["47"])
        for timestamp, item in values.items()
        if "46" in item and "47" in item
    )


def hourly_pairs(pairs: list[tuple[datetime, float, float]]) -> list[tuple[datetime, float, float]]:
    buckets: dict[datetime, list[tuple[float, float]]] = {}
    for timestamp, stage, discharge in pairs:
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append((stage, discharge))
    return [
        (
            hour,
            float(np.median([x[0] for x in values])),
            float(np.median([x[1] for x in values])),
        )
        for hour, values in sorted(buckets.items())
    ]


def choose_current_limb(hourly: list[tuple[datetime, float, float]], falling: bool) -> list[tuple[datetime, float, float]]:
    if not hourly:
        return []
    cutoff = hourly[-1][0] - timedelta(days=7)
    recent = [row for row in hourly if row[0] >= cutoff]
    selected = []
    for index, row in enumerate(recent):
        if index == 0:
            continue
        stage_change = row[1] - recent[index - 1][1]
        same_limb = stage_change <= 0.003 if falling else stage_change >= -0.003
        if same_limb:
            selected.append(row)
    latest = recent[-1][0]
    short = [row for row in selected if row[0] >= latest - timedelta(hours=96)]
    if len(short) >= 24 and max(x[1] for x in short) - min(x[1] for x in short) >= 0.04:
        return short
    return selected


def fit_rating(rows: list[tuple[datetime, float, float]]) -> dict:
    if len(rows) < 12:
        raise RuntimeError("Insufficient paired observations for current-event rating fit")
    stage = np.asarray([x[1] for x in rows], dtype=float)
    discharge = np.asarray([x[2] for x in rows], dtype=float)
    coefficients = np.polyfit(stage, discharge, 1)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    prediction = slope * stage + intercept
    residual = discharge - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((discharge - np.mean(discharge)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    if slope <= 0:
        raise RuntimeError(f"Non-physical current-event rating slope: {slope}")
    return {
        "slope_m3s_per_m": slope,
        "intercept_m3s": intercept,
        "r2": r2,
        "rmse_m3s": rmse,
        "n_hourly_points": len(rows),
        "start_utc": rows[0][0].isoformat(),
        "end_utc": rows[-1][0].isoformat(),
        "stage_range_m": [float(np.min(stage)), float(np.max(stage))],
        "discharge_range_m3s": [float(np.min(discharge)), float(np.max(discharge))],
    }


def q_from_stage(stage: float, fit: dict) -> float:
    return max(0.0, fit["slope_m3s_per_m"] * stage + fit["intercept_m3s"])


def stage_from_q(discharge: float, fit: dict) -> float:
    return (discharge - fit["intercept_m3s"]) / fit["slope_m3s_per_m"]


def starkey_wse(discharge: float, transfer: dict) -> float:
    equation = transfer["transfer"]
    equation_type = str(equation.get("type", "linear"))
    if equation_type == "piecewise_linear_in_ln_discharge":
        points = sorted(
            equation.get("points", []),
            key=lambda row: float(row["discharge_m3s"]),
        )
        if len(points) < 2:
            raise RuntimeError("RS18883 transfer requires at least two site points")
        q = max(0.05, float(discharge))
        if q <= float(points[0]["discharge_m3s"]):
            left, right = points[0], points[1]
        elif q >= float(points[-1]["discharge_m3s"]):
            left, right = points[-2], points[-1]
        else:
            left = right = None
            for candidate_left, candidate_right in zip(points, points[1:]):
                if (
                    float(candidate_left["discharge_m3s"])
                    <= q
                    <= float(candidate_right["discharge_m3s"])
                ):
                    left, right = candidate_left, candidate_right
                    break
            if left is None or right is None:
                raise RuntimeError("Unable to bracket RS18883 discharge")
        q0 = float(left["discharge_m3s"])
        q1 = float(right["discharge_m3s"])
        w0 = float(left["wse_m"])
        w1 = float(right["wse_m"])
        fraction = (math.log(q) - math.log(q0)) / (math.log(q1) - math.log(q0))
        return w0 + fraction * (w1 - w0)
    if equation_type == "quadratic_lagrange_in_ln_discharge":
        points = equation.get("points", [])
        if len(points) != 3:
            raise RuntimeError("Legacy RS18883 transfer requires exactly three site points")
        q = max(0.05, float(discharge))
        x = math.log(q)
        result = 0.0
        for i, point_i in enumerate(points):
            xi = math.log(float(point_i["discharge_m3s"]))
            term = float(point_i["wse_m"])
            for j, point_j in enumerate(points):
                if i == j:
                    continue
                xj = math.log(float(point_j["discharge_m3s"]))
                term *= (x - xj) / (xi - xj)
            result += term
        return result
    return equation["intercept_m"] + equation["slope_m_per_m3s"] * discharge


def interpolate_crossing(rows: list[dict], target_q: float, fit: dict) -> float | None:
    points = []
    for row in rows:
        day = finite(row.get("day"))
        stage = finite(row.get("stage_05EA002_m"))
        if day is None or stage is None:
            continue
        points.append((day, q_from_stage(stage, fit)))
    for (day0, q0), (day1, q1) in zip(points, points[1:]):
        if q0 >= target_q >= q1:
            if q0 == q1:
                return day1
            fraction = (q0 - target_q) / (q0 - q1)
            return day0 + fraction * (day1 - day0)
    return None


def main() -> None:
    for path in (GAUGE, SUMMARY, ENSEMBLE, TRANSFER):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = json.loads(SUMMARY.read_text())
    ensemble = json.loads(ENSEMBLE.read_text())
    transfer = json.loads(TRANSFER.read_text())
    target = summary.get("target_05EA002", {})
    current_stage = finite(target.get("latest"))
    change_24h = finite(target.get("change_24h"), 0.0)
    if current_stage is None:
        raise RuntimeError("Current stage unavailable")
    falling = change_24h < -0.005
    limb = "falling" if falling else ("rising" if change_24h > 0.005 else "approximately_flat")

    pairs = read_pairs()
    hourly = hourly_pairs(pairs)
    fit_rows = choose_current_limb(hourly, falling=falling)
    fit = fit_rating(fit_rows)
    current_q_fit = q_from_stage(current_stage, fit)
    observed_current_q = pairs[-1][2]
    target_stage_current_limb = stage_from_q(FIELD_TARGET_Q, fit)
    transfer_uncertainty = finite(transfer.get("transfer", {}).get("uncertainty_m"), 0.15)

    scenarios = {}
    generated = datetime.now(timezone.utc)
    for name in ("dry", "central", "wet"):
        scenario = ensemble.get("scenarios", {}).get(name, {})
        path = scenario.get("path_daily", [])
        crossing_days = interpolate_crossing(path, FIELD_TARGET_Q, fit)
        enriched = []
        for row in path:
            day = finite(row.get("day"))
            stage = finite(row.get("stage_05EA002_m"))
            if day is None or stage is None:
                continue
            discharge = q_from_stage(stage, fit)
            wse = starkey_wse(discharge, transfer)
            enriched.append(
                {
                    "day": day,
                    "stage_05EA002_m": stage,
                    "estimated_discharge_m3s": discharge,
                    "estimated_project_wse_m": wse,
                    "estimated_starkey_wse_m": wse,
                    "depth_over_main_floodplain_m": wse - MAIN_FLOODPLAIN_WSE,
                }
            )
        scenarios[name] = {
            "main_floodplain_crossing_days": crossing_days,
            "main_floodplain_crossing_date_utc": None
            if crossing_days is None
            else (generated + timedelta(days=crossing_days)).date().isoformat(),
            "path_daily": enriched,
        }

    current_wse = starkey_wse(observed_current_q, transfer)
    project_location = transfer.get("project_location", {})
    output = {
        "generated_utc": generated.isoformat(),
        "method": "current_event_stage_discharge_fit_then_rs18883_multi_point_wse_curve",
        "project_location": project_location,
        "hydrograph_limb": limb,
        "current_event_rating_fit": fit,
        "current": {
            "stage_05EA002_m": current_stage,
            "observed_discharge_05EA002_m3s": observed_current_q,
            "fit_discharge_05EA002_m3s": current_q_fit,
            "estimated_starkey_wse_m": current_wse,
            "estimated_project_wse_m": current_wse,
            "estimated_starkey_wse_range_m": [
                current_wse - transfer_uncertainty,
                current_wse + transfer_uncertainty,
            ],
            "depth_over_main_floodplain_m": current_wse - MAIN_FLOODPLAIN_WSE,
            "depth_over_low_pocket_m": current_wse - LOW_POCKET_WSE,
        },
        "construction_threshold": {
            "main_floodplain_wse_m": MAIN_FLOODPLAIN_WSE,
            "calibrated_target_discharge_m3s": FIELD_TARGET_Q,
            "equivalent_05EA002_stage_on_current_limb_m": target_stage_current_limb,
            "interpretation": (
                "The physical project threshold is 650.20 m. The current-event "
                "stage corresponding to the operational 6.77 m3/s anchor is used "
                "instead of either the contextual 1.50 m or 1.70 m gauge-stage observations."
            ),
        },
        "scenarios": scenarios,
        "uncertainty": {
            "starkey_transfer_m": transfer_uncertainty,
            "rating_fit_rmse_m3s": fit["rmse_m3s"],
            "confidence": "low_to_moderate_for_scheduling; field verification required",
        },
        "low_pocket": {
            "elevation_m": LOW_POCKET_WSE,
            "status": "not reliably represented by river-stage transfer",
            "interpretation": "The 649.60 m pocket may be controlled by local ponding/drainage. Raising it toward 650.20 m removes this separate low-point constraint.",
        },
        "limitations": [
            "The current-event rating is fitted to recent provisional 05EA002 observations and can shift if backwater or conveyance changes.",
            "Forecast paths are daily recession-delay scenarios, not an unsteady hydraulic simulation.",
            "The 650.20 m / 6.77 m3/s low-flow anchor is reconstructed rather than one concurrent surveyed project-site water level.",
            "The RS18883 curve is now constrained by the complete 2- to 1,000-year design profile above 14 m3/s, but the segment from 6.77 to 14 m3/s still depends on the approximate low-flow anchor.",
            "Final construction release requires direct site inspection and bearing-capacity confirmation.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
