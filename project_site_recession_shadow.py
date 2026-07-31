#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
OBSERVATIONS = Path("project_site_date_observations.csv")
ARCHIVE_PAIRS = Path("output/archive_probe/historical_rdpa_pairs.csv")
TRANSFER = ROOT / "routing" / "starkey_wse_transfer.json"
OUT = ROOT / "diagnostics" / "project_site_recession_shadow.json"
LOCAL_TZ = ZoneInfo("America/Edmonton")
THRESHOLD_WSE_M = 650.20
EXISTING_TARGET_Q_M3S = 6.77


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def design_profile_wse(discharge_m3s: float, transfer: dict) -> float:
    equation = transfer.get("transfer", {})
    points = sorted(
        equation.get("points", []),
        key=lambda row: float(row["discharge_m3s"]),
    )
    if len(points) < 2:
        raise RuntimeError("RS18883 transfer requires at least two points")
    q = max(0.05, float(discharge_m3s))
    if q <= float(points[0]["discharge_m3s"]):
        left, right = points[0], points[1]
    elif q >= float(points[-1]["discharge_m3s"]):
        left, right = points[-2], points[-1]
    else:
        left = right = None
        for candidate_left, candidate_right in zip(points, points[1:]):
            if float(candidate_left["discharge_m3s"]) <= q <= float(candidate_right["discharge_m3s"]):
                left, right = candidate_left, candidate_right
                break
        if left is None or right is None:
            raise RuntimeError("Unable to bracket discharge on RS18883 curve")
    q0 = float(left["discharge_m3s"])
    q1 = float(right["discharge_m3s"])
    w0 = float(left["wse_m"])
    w1 = float(right["wse_m"])
    fraction = (math.log(q) - math.log(q0)) / (math.log(q1) - math.log(q0))
    return w0 + fraction * (w1 - w0)


def read_observations() -> pd.DataFrame:
    if not OBSERVATIONS.exists():
        raise FileNotFoundError(OBSERVATIONS)
    frame = pd.read_csv(OBSERVATIONS)
    frame["observed_local_date"] = pd.to_datetime(
        frame["observed_local_date"], errors="coerce"
    ).dt.date
    frame["reported_site_wse_m"] = pd.to_numeric(
        frame["reported_site_wse_m"], errors="coerce"
    )
    return frame.dropna(subset=["observed_local_date", "reported_site_wse_m"]).sort_values(
        "observed_local_date"
    )


def read_daily_gauge() -> pd.DataFrame:
    if not ARCHIVE_PAIRS.exists():
        raise FileNotFoundError(ARCHIVE_PAIRS)
    frame = pd.read_csv(ARCHIVE_PAIRS)
    frame["date_utc"] = pd.to_datetime(frame["date_utc"], utc=True, errors="coerce")
    frame["stage_m"] = pd.to_numeric(frame["stage_m"], errors="coerce")
    frame["discharge_m3s"] = pd.to_numeric(frame["discharge_m3s"], errors="coerce")
    frame = frame.dropna(subset=["date_utc", "stage_m", "discharge_m3s"])
    frame["observed_local_date"] = frame["date_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    grouped = frame.groupby("observed_local_date", as_index=False).agg(
        stage_median=("stage_m", "median"),
        stage_min=("stage_m", "min"),
        stage_max=("stage_m", "max"),
        discharge_median_m3s=("discharge_m3s", "median"),
        discharge_min_m3s=("discharge_m3s", "min"),
        discharge_max_m3s=("discharge_m3s", "max"),
        hourly_pair_count=("date_utc", "count"),
    )
    return grouped


def iso_local_date(value: date) -> str:
    return value.isoformat()


def main() -> None:
    generated = datetime.now(timezone.utc)
    observations = read_observations()
    daily = read_daily_gauge()
    transfer = json.loads(TRANSFER.read_text())
    paired = observations.merge(daily, on="observed_local_date", how="left")
    rows = []
    for _, row in paired.iterrows():
        q = finite(row.get("discharge_median_m3s"))
        design_wse = design_profile_wse(q, transfer) if q is not None else None
        observed_wse = float(row["reported_site_wse_m"])
        rows.append(
            {
                "observed_local_date": iso_local_date(row["observed_local_date"]),
                "reported_site_wse_m": observed_wse,
                "source": row.get("source"),
                "datum_status": row.get("datum_status"),
                "measurement_location_status": row.get("measurement_location_status"),
                "exact_time_status": row.get("exact_time_status"),
                "gauge_pairing_method": "median of available 05EA002 hourly pairs over the same America/Edmonton calendar date",
                "hourly_pair_count": int(row["hourly_pair_count"]) if finite(row.get("hourly_pair_count")) is not None else None,
                "stage_05EA002_median_m": finite(row.get("stage_median")),
                "stage_05EA002_range_m": [finite(row.get("stage_min")), finite(row.get("stage_max"))],
                "discharge_05EA002_median_m3s": q,
                "discharge_05EA002_range_m3s": [
                    finite(row.get("discharge_min_m3s")),
                    finite(row.get("discharge_max_m3s")),
                ],
                "steady_state_design_profile_wse_at_daily_median_q_m": design_wse,
                "observed_minus_design_profile_m": None if design_wse is None else observed_wse - design_wse,
                "notes": row.get("notes"),
            }
        )

    usable = [row for row in rows if row["discharge_05EA002_median_m3s"] is not None]
    output = {
        "generated_utc": generated.isoformat(),
        "status": "provisional_date_level_field_evidence_available" if len(usable) >= 2 else "insufficient_provisional_field_evidence",
        "mode": "shadow_only_no_automatic_threshold_or_transfer_replacement",
        "project_threshold_wse_m": THRESHOLD_WSE_M,
        "existing_operational_target_discharge_m3s": EXISTING_TARGET_Q_M3S,
        "observations": rows,
        "date_recession_fit": None,
        "discharge_wse_fit": None,
        "current_site_state": {},
        "comparison_with_design_profile": {},
        "operational_interpretation": (
            "The contractor elevations are treated as provisional direct site evidence. They alter how current site depth is reported, but they do not automatically replace the 6.77 m3/s scheduling anchor because exact survey time, datum and shot location have not yet been documented."
        ),
        "limitations": [
            "Both elevations are date-only observations, so gauge pairing uses the same-day median rather than a concurrent reading.",
            "The project datum and exact measurement location require confirmation before the observations can become authoritative calibration points.",
            "Two observations define an exact two-point line and cannot establish curvature, hysteresis or formal uncertainty.",
            "The date recession assumes the observed July 16 to July 23 decline continued without a new hydraulic control or rainfall response.",
            "This diagnostic does not authorize construction release; a direct site survey and bearing-capacity inspection remain required.",
        ],
    }

    if len(usable) >= 2:
        first = usable[0]
        last = usable[-1]
        first_date = date.fromisoformat(first["observed_local_date"])
        last_date = date.fromisoformat(last["observed_local_date"])
        elapsed_days = (last_date - first_date).days
        if elapsed_days <= 0:
            raise RuntimeError("Provisional site observations do not span a positive time interval")
        daily_slope = (
            float(last["reported_site_wse_m"]) - float(first["reported_site_wse_m"])
        ) / elapsed_days
        intercept = float(first["reported_site_wse_m"])
        threshold_day_from_first = (THRESHOLD_WSE_M - intercept) / daily_slope
        crossing_date = first_date + timedelta(days=float(threshold_day_from_first))
        current_local_date = generated.astimezone(LOCAL_TZ).date()
        current_day_from_first = (current_local_date - first_date).days
        date_estimate = intercept + daily_slope * current_day_from_first

        q = np.asarray([float(row["discharge_05EA002_median_m3s"]) for row in usable], dtype=float)
        y = np.asarray([float(row["reported_site_wse_m"]) for row in usable], dtype=float)
        q_slope, q_intercept = np.polyfit(q, y, 1)
        q_threshold = (THRESHOLD_WSE_M - q_intercept) / q_slope
        current_q = finite(transfer.get("current_05EA002", {}).get("discharge_m3s"))
        q_estimate = None if current_q is None else float(q_slope * current_q + q_intercept)
        design_current = None if current_q is None else design_profile_wse(current_q, transfer)

        residuals = [float(row["observed_minus_design_profile_m"]) for row in usable]
        output["date_recession_fit"] = {
            "observation_count": len(usable),
            "first_observation_local_date": first["observed_local_date"],
            "last_observation_local_date": last["observed_local_date"],
            "elapsed_days": elapsed_days,
            "wse_change_m": float(last["reported_site_wse_m"]) - float(first["reported_site_wse_m"]),
            "slope_m_per_day": daily_slope,
            "projected_threshold_crossing_local_date": crossing_date.isoformat(),
            "projected_threshold_crossing_basis": "linear continuation of the two contractor-reported site elevations",
        }
        output["discharge_wse_fit"] = {
            "observation_count": len(usable),
            "form": "reported_site_wse_m = slope_m_per_m3s * Q05EA002 + intercept_m",
            "slope_m_per_m3s": float(q_slope),
            "intercept_m": float(q_intercept),
            "extrapolated_q_at_650_20_m_m3s": float(q_threshold),
            "difference_from_existing_6_77_anchor_m3s": float(q_threshold - EXISTING_TARGET_Q_M3S),
            "interpretation": "The two date-level field observations independently extrapolate toward the current operational discharge anchor, but this is not a validated rating curve.",
        }
        output["current_site_state"] = {
            "current_05EA002_discharge_m3s": current_q,
            "direct_current_site_measurement_available": False,
            "date_recession_estimated_project_wse_m": float(date_estimate),
            "date_recession_estimated_depth_over_650_20_m": float(date_estimate - THRESHOLD_WSE_M),
            "discharge_relation_estimated_project_wse_m": q_estimate,
            "discharge_relation_estimated_depth_over_650_20_m": None if q_estimate is None else float(q_estimate - THRESHOLD_WSE_M),
            "steady_state_design_profile_equivalent_wse_m": design_current,
            "steady_state_design_profile_equivalent_depth_over_650_20_m": None if design_current is None else float(design_current - THRESHOLD_WSE_M),
            "status": "provisional_site_recession_estimate_not_direct_measurement",
        }
        output["comparison_with_design_profile"] = {
            "observed_minus_design_profile_residuals_m": residuals,
            "mean_observed_minus_design_profile_m": float(np.mean(residuals)),
            "minimum_observed_minus_design_profile_m": float(np.min(residuals)),
            "maximum_observed_minus_design_profile_m": float(np.max(residuals)),
            "classification": "material_2026_site_wse_above_steady_state_design_profile",
            "design_profile_current_use": "design_event_context_and_threshold_translation_only_not_actual_2026_site_depth",
            "interpretation": "The contractor elevations indicate that the 2026 site water surface was materially higher than the steady-state RS18883 design-profile transfer at comparable gauge discharge."
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
