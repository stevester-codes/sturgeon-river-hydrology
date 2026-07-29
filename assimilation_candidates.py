#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
GAUGE = ROOT / "raw" / "wateroffice" / "05EA002.csv"
PRECIP = ROOT / "processed" / "watershed_precip_06h.csv"
BASE = ROOT / "calibration" / "calibration.json"
EXISTING = ROOT / "calibration_v2" / "event_response_v2.csv"
OUT = ROOT / "diagnostics" / "assimilation_candidates.json"
TIME_RE = re.compile(r"_(\d{10})_000_\d{2}\.dbf$")


def parse_gauge() -> pd.Series:
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
    stage = frame[frame[parameter_column] == 46].set_index(date_column)[value_column]
    return stage.sort_index().resample("1h").median().interpolate(limit=2).dropna()


def parse_precip() -> pd.Series:
    frame = pd.read_csv(PRECIP)
    if frame.empty:
        return pd.Series(dtype=float)

    def valid_time(filename: str):
        match = TIME_RE.search(str(filename))
        if not match:
            return pd.NaT
        return pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)

    frame["valid_utc"] = frame["_source_file"].map(valid_time)
    frame["PR_mm"] = pd.to_numeric(frame["PR_mm"], errors="coerce")
    frame = frame[
        (frame.Station.astype(str) == "05EA002")
        & frame.valid_utc.notna()
        & frame.PR_mm.notna()
    ]
    return frame.groupby("valid_utc").PR_mm.mean().sort_index()


def merge_events(times: list[pd.Timestamp], gap_hours: int = 12):
    if not times:
        return []
    times = sorted(times)
    groups = []
    start = previous = times[0]
    for timestamp in times[1:]:
        if (timestamp - previous).total_seconds() <= gap_hours * 3600:
            previous = timestamp
        else:
            groups.append((start - pd.Timedelta(hours=6), previous + pd.Timedelta(hours=6)))
            start = previous = timestamp
    groups.append((start - pd.Timedelta(hours=6), previous + pd.Timedelta(hours=6)))
    return groups


def recession_rate(model: dict, stage: float) -> float:
    intercept = model.get("intercept_m_per_day")
    coefficient = model.get("stage_coefficient_per_day")
    if intercept is None or coefficient is None:
        return -0.02
    return min(-0.001, float(intercept) + float(coefficient) * float(stage))


def baseline(stage0: float, index: pd.DatetimeIndex, model: dict) -> pd.Series:
    values = []
    stage = float(stage0)
    previous = None
    for timestamp in index:
        if previous is not None:
            hours = max(0.0, (timestamp - previous).total_seconds() / 3600.0)
            stage += recession_rate(model, stage) * hours / 24.0
        values.append(stage)
        previous = timestamp
    return pd.Series(values, index=index, dtype=float)


def first_sustained_below(series: pd.Series, threshold: float, hours: int):
    condition = series <= threshold
    sustained = condition.rolling(hours, min_periods=hours).sum() >= hours
    hits = sustained[sustained].index
    return hits[0] if len(hits) else None


def overlaps_existing(start: pd.Timestamp, end: pd.Timestamp, existing: pd.DataFrame) -> bool:
    if existing.empty:
        return False
    starts = pd.to_datetime(existing.rain_start_utc, utc=True, errors="coerce")
    ends = pd.to_datetime(existing.rain_end_utc, utc=True, errors="coerce")
    return bool(((starts <= end) & (ends >= start)).any())


def main() -> None:
    for path in (GAUGE, PRECIP, BASE, EXISTING):
        if not path.exists():
            raise FileNotFoundError(path)

    generated = datetime.now(timezone.utc)
    stage = parse_gauge()
    precip = parse_precip()
    base = json.loads(BASE.read_text())
    model = base.get("master_recession", {})
    existing = pd.read_csv(EXISTING)

    if stage.empty or precip.empty:
        raise RuntimeError("Recent stage or precipitation data are unavailable")

    p24 = precip.rolling(4, min_periods=1).sum()
    wet_times = sorted(
        set(precip[precip >= 1.0].index.tolist() + p24[p24 >= 3.0].index.tolist())
    )
    events = merge_events(wet_times)
    candidates = []
    data_end = stage.index.max()

    for position, (start, end) in enumerate(events):
        next_start = events[position + 1][0] if position + 1 < len(events) else None
        nominal_end = end + pd.Timedelta(hours=168)
        analysis_end = min(nominal_end, data_end)
        censor_reason = None
        if next_start is not None and next_start < analysis_end:
            analysis_end = next_start
            censor_reason = "next_rain_event"
        elif data_end < nominal_end:
            censor_reason = "dataset_right_edge"

        before = stage.index[stage.index <= start]
        if not len(before):
            continue
        t0 = before[-1]
        observed = stage.loc[t0:analysis_end]
        if len(observed) < 12:
            continue
        expected = baseline(float(observed.iloc[0]), observed.index, model)
        departure = observed - expected
        response = departure.loc[departure.index >= start]
        if response.empty:
            continue

        peak_time = response.idxmax()
        peak_departure = float(response.max())
        departure_end = float(response.iloc[-1])
        hours_after_peak = max(0.0, (analysis_end - peak_time).total_seconds() / 3600.0)
        peak_decline_after = peak_departure - departure_end
        peak_complete = bool(hours_after_peak >= 24 and peak_decline_after >= 0.01)
        recovery_time = first_sustained_below(response.loc[peak_time:], 0.015, 12)
        recovery_complete = recovery_time is not None and recovery_time <= analysis_end
        already = overlaps_existing(start, end, existing)
        rain_total = float(precip.loc[(precip.index > start) & (precip.index <= end)].sum())

        if already:
            status = "already_in_calibration_dataset"
        elif end > data_end:
            status = "active_or_incomplete_rain_event"
        elif censor_reason == "next_rain_event" and not peak_complete:
            status = "censored_by_next_rain_before_clean_peak"
        elif recovery_complete:
            status = "recovery_candidate_ready_for_full_spatial_backfill"
        elif peak_complete:
            status = "peak_candidate_ready_for_full_spatial_backfill"
        else:
            status = "continue_observing_response"

        candidates.append(
            {
                "rain_start_utc": start.isoformat(),
                "rain_end_utc": end.isoformat(),
                "analysis_end_utc": analysis_end.isoformat(),
                "status": status,
                "already_calibrated": already,
                "censor_reason": censor_reason,
                "basin_rain_mm": rain_total,
                "pre_event_stage_m": float(stage.loc[t0]),
                "departure_peak_m": peak_departure,
                "departure_peak_utc": peak_time.isoformat(),
                "departure_at_analysis_end_m": departure_end,
                "hours_observed_after_peak": hours_after_peak,
                "peak_decline_after_peak_m": peak_decline_after,
                "clean_peak_observed": peak_complete,
                "complete_recovery_observed": bool(recovery_complete),
                "recovery_utc": recovery_time.isoformat() if recovery_time is not None else None,
                "full_grid_spatial_backfill_required": not already,
                "automatic_promotion_allowed": False,
            }
        )

    new_candidates = [row for row in candidates if not row["already_calibrated"]]
    peak_ready = [row for row in new_candidates if row["clean_peak_observed"]]
    recovery_ready = [row for row in new_candidates if row["complete_recovery_observed"]]
    output = {
        "generated_utc": generated.isoformat(),
        "status": "diagnostic_candidate_gate",
        "recent_data_range": {
            "stage_start_utc": stage.index.min().isoformat(),
            "stage_end_utc": stage.index.max().isoformat(),
            "precip_start_utc": precip.index.min().isoformat(),
            "precip_end_utc": precip.index.max().isoformat(),
        },
        "counts": {
            "detected_recent_events": len(candidates),
            "new_events_not_in_calibration": len(new_candidates),
            "clean_peak_candidates": len(peak_ready),
            "complete_recovery_candidates": len(recovery_ready),
        },
        "promotion_policy": {
            "automatic_promotion_enabled": False,
            "required_before_candidate_model": [
                "retrieve full HRDPA grid coverage for the complete event",
                "recompute the same spatial features used by the existing calibration",
                "confirm peak and censoring classifications against the longer historical window",
                "create a candidate calibration without overwriting the operational calibration",
                "run leave-one-event-out and hindcast comparisons where sample size permits",
                "promote only if performance and physical plausibility do not worsen",
                "retain the previous calibration and outputs for immediate rollback",
            ],
        },
        "recommended_action": (
            "Run the manual historical backfill and candidate-calibration comparison when a clean new peak or recovery candidate appears. "
            "Do not modify the live training set from the rolling 14-day operational files alone."
        ),
        "candidates": candidates,
        "limitations": [
            "The rolling operational dataset is only about 14 days and can truncate an event that began earlier.",
            "Operational precipitation contains watershed summaries but not all full-grid spatial features required by calibration_v2.",
            "Peak and recovery classifications are screening tests; final assimilation must use the longer historical backfill.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps({"status": output["status"], **output["counts"]}, indent=2))


if __name__ == "__main__":
    main()
