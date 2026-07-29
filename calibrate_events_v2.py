#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
RAW = ROOT / "raw" / "wateroffice"
CAL = ROOT / "calibration" / "calibration.json"
SPATIAL = ROOT / "spatial" / "observed_event_grid_coverage.csv"
PRECIP = ROOT / "processed" / "watershed_precip_06h.csv"
OUT = ROOT / "calibration_v2"
TARGET = "05EA002"
STATIONS = ["05EA002", "05EA005", "05EA006", "05EA010", "05EA011", "05EA012"]
TIME_RE = re.compile(r"_(\d{10})_000_\d{2}\.dbf$")


def parse_gauge(station: str) -> pd.DataFrame:
    path = RAW / f"{station}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
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
    return pivot.sort_index().resample("1h").median().interpolate(limit=2)


def parse_precip() -> pd.DataFrame:
    frame = pd.read_csv(PRECIP)
    if frame.empty:
        return frame

    def source_time(filename: str):
        match = TIME_RE.search(str(filename))
        if not match:
            return pd.NaT
        return pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)

    frame["valid_utc"] = frame["_source_file"].map(source_time)
    frame["PR_mm"] = pd.to_numeric(frame["PR_mm"], errors="coerce")
    return frame.dropna(subset=["valid_utc", "PR_mm", "Station"])


def model_rate(model: dict, stage: float) -> float:
    intercept = model.get("intercept_m_per_day")
    coefficient = model.get("stage_coefficient_per_day")
    if intercept is None or coefficient is None:
        return -0.02
    return min(-0.001, float(intercept) + float(coefficient) * float(stage))


def dynamic_baseline(stage0: float, index: pd.DatetimeIndex, model: dict) -> pd.Series:
    values = []
    stage = float(stage0)
    previous = None
    for timestamp in index:
        if previous is not None:
            hours = max(0.0, (timestamp - previous).total_seconds() / 3600.0)
            stage += model_rate(model, stage) * hours / 24.0
        values.append(stage)
        previous = timestamp
    return pd.Series(values, index=index, dtype=float)


def nearest(series: pd.Series, timestamp: pd.Timestamp) -> float:
    if series.empty:
        return np.nan
    location = series.index.get_indexer([timestamp], method="nearest")[0]
    return float(series.iloc[location]) if location >= 0 else np.nan


def first_sustained(series: pd.Series, threshold: float, hours: int, above: bool = True):
    if series.empty:
        return None
    condition = series >= threshold if above else series <= threshold
    sustained = condition.rolling(hours, min_periods=hours).sum() >= hours
    hits = sustained[sustained].index
    return hits[0] if len(hits) else None


def spatial_value(spatial: pd.DataFrame, event_id: int, subarea: str, column: str):
    rows = spatial[(spatial.event_id == event_id) & (spatial.subarea == subarea)]
    if rows.empty or column not in rows:
        return None
    value = pd.to_numeric(rows.iloc[0][column], errors="coerce")
    return float(value) if np.isfinite(value) else None


def classify_event(record: dict) -> str:
    basin = record.get("basin_mean_mm") or 0.0
    lower = record.get("lower_mean_mm") or 0.0
    upper = record.get("upper_mean_mm") or 0.0
    middle = record.get("middle_mean_mm") or 0.0
    atim = record.get("atim_mean_mm") or 0.0
    carrot = record.get("carrot_mean_mm") or 0.0
    local = record.get("local_mean_mm") or 0.0
    basin_pct10 = record.get("basin_pct_gt_10mm") or 0.0
    basin_pct5 = record.get("basin_pct_gt_5mm") or 0.0
    if basin_pct10 >= 70 or basin_pct5 >= 90:
        return "widespread_basin"
    if basin > 0 and lower >= 1.35 * basin and upper <= 0.85 * basin:
        return "lower_basin_concentrated"
    if lower > 0 and upper >= 1.35 * lower:
        return "upper_lake_chain_concentrated"
    if max(atim, carrot, local) >= max(2.0, 1.4 * basin):
        dominant = max(
            [(atim, "atim_big_lake"), (carrot, "carrot_creek"), (local, "direct_local")]
        )[1]
        return f"tributary_localized_{dominant}"
    if middle >= max(2.0, 1.25 * upper, 1.25 * lower):
        return "middle_mainstem_concentrated"
    return "mixed_or_weak"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(CAL.read_text())
    events = sorted(base.get("events", []), key=lambda event: event["rain_start_utc"])
    gauges = {station: parse_gauge(station) for station in STATIONS}
    target = gauges[TARGET]["stage_m"].dropna()
    model = base.get("master_recession", {})
    precip = parse_precip()
    precip_pivot = precip.pivot_table(
        index="valid_utc", columns="Station", values="PR_mm", aggfunc="mean"
    ).sort_index()
    spatial = pd.read_csv(SPATIAL) if SPATIAL.exists() and SPATIAL.stat().st_size else pd.DataFrame()
    if not spatial.empty:
        spatial["event_id"] = pd.to_numeric(spatial["event_id"], errors="coerce")
    records = []
    for position, event in enumerate(events):
        event_id = int(event["event_id"])
        start = pd.Timestamp(event["rain_start_utc"])
        end = pd.Timestamp(event["rain_end_utc"])
        next_start = (
            pd.Timestamp(events[position + 1]["rain_start_utc"])
            if position + 1 < len(events)
            else None
        )
        nominal_end = end + pd.Timedelta(hours=168)
        dataset_end = target.index[-1]
        analysis_end = min(nominal_end, dataset_end)
        truncation_reason = None
        if next_start is not None and next_start < analysis_end:
            analysis_end = next_start
            truncation_reason = "next_rain_event"
        elif dataset_end < nominal_end:
            truncation_reason = "dataset_right_edge"
        t0_candidates = target.index[target.index <= start]
        if not len(t0_candidates):
            continue
        t0 = t0_candidates[-1]
        observed = target.loc[t0:analysis_end]
        if len(observed) < 6:
            continue
        stage0 = float(observed.iloc[0])
        baseline = dynamic_baseline(stage0, observed.index, model)
        departure = observed - baseline
        event_departure = departure.loc[departure.index >= start]
        event_observed = observed.loc[observed.index >= start]
        if event_departure.empty:
            continue
        onset = first_sustained(event_departure, 0.01, 3, above=True)
        peak_time = event_departure.idxmax()
        peak_departure = float(event_departure.max())
        raw_peak_time = event_observed.idxmax()
        raw_peak_stage = float(event_observed.max())
        pre_stage = nearest(target, start)
        raw_rise = raw_peak_stage - pre_stage
        departure_at_end = float(event_departure.iloc[-1])
        peak_near_end = (analysis_end - peak_time).total_seconds() <= 12 * 3600
        censored = bool(truncation_reason and (departure_at_end > 0.02 or peak_near_end))
        censor_type = None
        if censored:
            censor_type = (
                "peak_or_recovery_censored_by_next_rain"
                if truncation_reason == "next_rain_event"
                else "peak_or_recovery_censored_by_dataset_end"
            )
        recovery_time = None
        recovery_slice = departure.loc[departure.index >= peak_time]
        candidate_recovery = first_sustained(recovery_slice, 0.015, 12, above=False)
        if candidate_recovery is not None and candidate_recovery <= analysis_end:
            recovery_time = candidate_recovery
        rate_at_peak = abs(model_rate(model, float(baseline.loc[peak_time])))
        days_lost = peak_departure / rate_at_peak if rate_at_peak > 0.001 else None
        target_precip = precip_pivot.get(TARGET, pd.Series(dtype=float)).fillna(0.0)
        antecedent_72 = float(target_precip.loc[(target_precip.index > start - pd.Timedelta(hours=72)) & (target_precip.index <= start)].sum())
        antecedent_168 = float(target_precip.loc[(target_precip.index > start - pd.Timedelta(hours=168)) & (target_precip.index <= start)].sum())
        record = {
            "event_id": event_id,
            "rain_start_utc": start.isoformat(),
            "rain_end_utc": end.isoformat(),
            "rain_duration_h": float((end - start).total_seconds() / 3600),
            "analysis_end_utc": analysis_end.isoformat(),
            "truncation_reason": truncation_reason,
            "response_censored": censored,
            "censor_type": censor_type,
            "pre_stage_m": pre_stage,
            "pre_model_recession_m_per_day": model_rate(model, pre_stage),
            "antecedent_72h_basin_rain_mm": antecedent_72,
            "antecedent_168h_basin_rain_mm": antecedent_168,
            "response_onset_utc": onset.isoformat() if onset is not None else None,
            "lag_to_onset_h": float((onset - start).total_seconds() / 3600) if onset is not None else None,
            "departure_peak_utc": peak_time.isoformat(),
            "departure_peak_m": peak_departure,
            "departure_at_analysis_end_m": departure_at_end,
            "lag_to_departure_peak_h": float((peak_time - start).total_seconds() / 3600),
            "raw_peak_utc": raw_peak_time.isoformat(),
            "raw_peak_stage_m": raw_peak_stage,
            "raw_stage_rise_m": raw_rise,
            "estimated_recession_days_lost": float(days_lost) if days_lost is not None else None,
            "recovery_utc": recovery_time.isoformat() if recovery_time is not None else None,
            "recovery_duration_h": float((recovery_time - peak_time).total_seconds() / 3600) if recovery_time is not None else None,
        }
        spatial_map = {
            "basin": "basin_to_05EA002",
            "lower": "lower_incremental_05EA005_to_05EA002",
            "upper": "upper_lake_chain_isle_lac_ste_anne",
            "middle": "lac_ste_anne_to_villeneuve_mainstem",
            "atim": "atim_creek_big_lake_tributary",
            "carrot": "carrot_creek",
            "local": "direct_big_lake_and_local_to_05EA002",
        }
        for short, subarea in spatial_map.items():
            record[f"{short}_mean_mm"] = spatial_value(spatial, event_id, subarea, "mean_mm") if not spatial.empty else None
            record[f"{short}_max_mm"] = spatial_value(spatial, event_id, subarea, "max_mm") if not spatial.empty else None
            for threshold in [5, 10, 20, 30, 50]:
                record[f"{short}_pct_gt_{threshold}mm"] = spatial_value(
                    spatial, event_id, subarea, f"pct_gt_{threshold}mm"
                ) if not spatial.empty else None
        record["storm_type"] = classify_event(record)
        for station in ["05EA005", "05EA006", "05EA010", "05EA011", "05EA012"]:
            series = gauges[station]["stage_m"].dropna().loc[start:analysis_end]
            if series.empty:
                continue
            starting = nearest(gauges[station]["stage_m"].dropna(), start)
            station_peak_time = series.idxmax()
            record[f"{station}_stage_change_m"] = float(series.max() - starting)
            record[f"{station}_peak_lag_h"] = float((station_peak_time - start).total_seconds() / 3600)
        response_quality = "complete"
        if censored:
            response_quality = "censored_lower_bound"
        elif peak_departure < 0.01:
            response_quality = "weak_or_no_detectable_response"
        elif recovery_time is None:
            response_quality = "peak_observed_recovery_not_observed"
        record["response_quality"] = response_quality
        record["eligible_for_peak_training"] = bool(not censored and peak_departure >= 0.01)
        record["eligible_for_recovery_training"] = bool(not censored and recovery_time is not None)
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "event_response_v2.csv", index=False)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(records),
        "uncensored_peak_events": int(frame.eligible_for_peak_training.sum()) if not frame.empty else 0,
        "complete_recovery_events": int(frame.eligible_for_recovery_training.sum()) if not frame.empty else 0,
        "storm_type_counts": frame.storm_type.value_counts().to_dict() if not frame.empty else {},
        "events": records,
        "method": {
            "baseline": "Dynamic hourly integration of the fitted stage-dependent rain-free recession model.",
            "onset": "Stage departure at least 0.01 m for three consecutive hourly values.",
            "recovery": "Stage departure at or below 0.015 m for 12 consecutive hourly values.",
            "censoring": "Response windows are truncated at the next rainfall event or dataset edge; unresolved peaks/recoveries are labelled as lower bounds.",
            "spatial_features": "Full-grid HRDPA means, maxima, and areal coverage above 5, 10, 20, 30, and 50 mm for derived subareas.",
        },
    }
    (OUT / "calibration_v2.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ["event_count", "uncensored_peak_events", "complete_recovery_events", "storm_type_counts"]}, indent=2))


if __name__ == "__main__":
    main()
