#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = ["05EA002", "05EA005", "05EA006", "05EA010", "05EA011", "05EA012"]
TARGET = "05EA002"
WO_DIR = Path("sturgeon_pipeline_output/raw/wateroffice")
P06 = Path("sturgeon_pipeline_output/processed/watershed_precip_06h.csv")
OUT = Path("sturgeon_pipeline_output/calibration")
FILENAME_TIME = re.compile(r"_(\d{10})_000_\d{2}\.dbf$")


def parse_wo(station: str) -> pd.DataFrame:
    path = WO_DIR / f"{station}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if c.lower() == "date")
    param_col = next(c for c in df.columns if "parameter" in c.lower() or "paramètre" in c.lower())
    value_col = next(c for c in df.columns if "value" in c.lower() or "valeur" in c.lower())
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[param_col] = pd.to_numeric(df[param_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col, param_col])
    piv = df.pivot_table(index=date_col, columns=param_col, values=value_col, aggfunc="median")
    piv = piv.rename(columns={46: "stage_m", 47: "flow_m3s"}).sort_index()
    return piv.resample("1h").median().interpolate(limit=2)


def parse_precip() -> pd.DataFrame:
    df = pd.read_csv(P06)
    if df.empty:
        return df
    def valid_time(name: str):
        m = FILENAME_TIME.search(str(name))
        return pd.to_datetime(m.group(1), format="%Y%m%d%H", utc=True) if m else pd.NaT
    df["valid_utc"] = df["_source_file"].map(valid_time)
    df["PR_mm"] = pd.to_numeric(df["PR_mm"], errors="coerce")
    df["Shp_Area"] = pd.to_numeric(df["Shp_Area"], errors="coerce")
    df["CFIA"] = pd.to_numeric(df["CFIA"], errors="coerce")
    return df.dropna(subset=["valid_utc", "PR_mm", "Station"])


def merge_events(times: list[pd.Timestamp], gap_hours: int = 12) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not times:
        return []
    times = sorted(times)
    groups = []
    start = prev = times[0]
    for t in times[1:]:
        if (t - prev).total_seconds() <= gap_hours * 3600:
            prev = t
        else:
            groups.append((start, prev + pd.Timedelta(hours=6)))
            start = prev = t
    groups.append((start, prev + pd.Timedelta(hours=6)))
    return groups


def fit_master_recession(stage: pd.Series, p_target: pd.Series) -> dict:
    # Six-hour rainfall is expanded to hourly totals. Dry means no meaningful
    # basin precipitation in the preceding 48 h and the river is falling.
    p_hour = p_target.reindex(stage.index, fill_value=0.0)
    p48 = p_hour.rolling(48, min_periods=1).sum()
    dh_day = stage.diff(6) / 6 * 24
    mask = (p48 <= 0.75) & (dh_day < -0.003) & (dh_day > -0.15) & stage.notna()
    x = stage[mask].to_numpy()
    y = dh_day[mask].to_numpy()
    if len(x) < 20:
        return {"n": int(len(x)), "intercept_m_per_day": None, "stage_coefficient_per_day": None, "rmse_m_per_day": None}
    # Trim slope outliers before ordinary least squares.
    lo, hi = np.nanpercentile(y, [10, 90])
    keep = (y >= lo) & (y <= hi)
    X = np.column_stack([np.ones(keep.sum()), x[keep]])
    beta, *_ = np.linalg.lstsq(X, y[keep], rcond=None)
    pred = X @ beta
    rmse = float(np.sqrt(np.mean((y[keep] - pred) ** 2)))
    return {
        "n": int(keep.sum()),
        "intercept_m_per_day": float(beta[0]),
        "stage_coefficient_per_day": float(beta[1]),
        "rmse_m_per_day": rmse,
    }


def expected_rate(model: dict, stage: float) -> float:
    a = model.get("intercept_m_per_day")
    b = model.get("stage_coefficient_per_day")
    if a is None or b is None:
        return -0.02
    return float(a + b * stage)


def first_sustained_positive(series: pd.Series, threshold: float = 0.002, hours: int = 3):
    d = series.diff()
    flag = d.rolling(hours).sum() >= threshold
    hits = flag[flag].index
    return hits[0] if len(hits) else None


def nearest_value(series: pd.Series, when: pd.Timestamp):
    if series.empty:
        return np.nan
    idx = series.index.get_indexer([when], method="nearest")[0]
    return float(series.iloc[idx]) if idx >= 0 else np.nan


def event_calibration(gauges: dict[str, pd.DataFrame], precip: pd.DataFrame, model: dict) -> list[dict]:
    pvt = precip.pivot_table(index="valid_utc", columns="Station", values="PR_mm", aggfunc="mean").sort_index()
    p002 = pvt.get(TARGET, pd.Series(dtype=float)).fillna(0)
    p24 = p002.rolling(4, min_periods=1).sum()
    wet_times = sorted(set(p002[p002 >= 1.0].index.tolist() + p24[p24 >= 3.0].index.tolist()))
    windows = merge_events(wet_times, 12)
    target_stage = gauges[TARGET]["stage_m"].dropna()
    rows = []
    for i, (rain_start, rain_end) in enumerate(windows, 1):
        # Include 24 h before first six-hour accumulation end.
        event_start = rain_start - pd.Timedelta(hours=6)
        search_end = rain_end + pd.Timedelta(hours=120)
        pre = target_stage.loc[event_start - pd.Timedelta(hours=30): event_start]
        if len(pre) < 6:
            continue
        # Robust pre-event rate from hourly medians over last 24 h.
        pre24 = pre.iloc[-25:]
        h = (pre24.index - pre24.index[0]).total_seconds() / 3600
        slope_h = float(np.polyfit(h, pre24.values, 1)[0]) if len(pre24) >= 6 else expected_rate(model, float(pre.iloc[-1])) / 24
        # Avoid using an active rise as a recession baseline.
        if slope_h >= -0.0001:
            slope_h = expected_rate(model, float(pre.iloc[-1])) / 24
        h0 = float(pre.iloc[-1])
        t0 = pre.index[-1]
        obs = target_stage.loc[t0:search_end]
        if obs.empty:
            continue
        elapsed_h = (obs.index - t0).total_seconds() / 3600
        baseline = pd.Series(h0 + slope_h * elapsed_h, index=obs.index)
        departure = obs - baseline
        peak_t = departure.idxmax()
        peak_departure = float(departure.max())
        onset_candidates = departure[(departure >= 0.01) & (departure.index >= event_start)]
        onset_t = onset_candidates.index[0] if len(onset_candidates) else first_sustained_positive(obs.loc[event_start:], 0.003, 3)
        actual_peak_t = obs.loc[event_start:search_end].idxmax()
        actual_peak = float(obs.loc[event_start:search_end].max())
        pre_stage = nearest_value(target_stage, event_start)
        actual_rise = actual_peak - pre_stage
        recovery = departure.loc[peak_t:]
        rec_hits = recovery[recovery <= 0.01]
        recovery_t = rec_hits.index[0] if len(rec_hits) else None
        rate_day = slope_h * 24
        days_lost = peak_departure / abs(rate_day) if rate_day < -0.001 else np.nan

        record = {
            "event_id": i,
            "rain_start_utc": event_start.isoformat(),
            "rain_end_utc": rain_end.isoformat(),
            "rain_duration_h": float((rain_end - event_start).total_seconds() / 3600),
            "pre_stage_m": pre_stage,
            "pre_recession_m_per_day": rate_day,
            "response_onset_utc": onset_t.isoformat() if onset_t is not None else None,
            "lag_to_onset_h": float((onset_t - event_start).total_seconds() / 3600) if onset_t is not None else None,
            "observed_peak_utc": actual_peak_t.isoformat(),
            "observed_peak_stage_m": actual_peak,
            "actual_stage_rise_m": actual_rise,
            "baseline_departure_peak_m": peak_departure,
            "baseline_departure_peak_utc": peak_t.isoformat(),
            "lag_to_departure_peak_h": float((peak_t - event_start).total_seconds() / 3600),
            "recovery_utc": recovery_t.isoformat() if recovery_t is not None else None,
            "recovery_duration_h": float((recovery_t - peak_t).total_seconds() / 3600) if recovery_t is not None else None,
            "estimated_recession_days_lost": float(days_lost) if np.isfinite(days_lost) else None,
        }
        # Rainfall totals by hydrometric watershed during the event.
        p_event = pvt.loc[(pvt.index > event_start) & (pvt.index <= rain_end)]
        for st in ["05EA002", "05EA005", "05EA010", "05EA011", "05EA012"]:
            record[f"rain_{st}_mm"] = float(p_event[st].sum()) if st in p_event else None
        # Verified nested area-weighted lower residual: 05EA005 watershed is
        # upstream and contained within 05EA002. This residual contains the
        # Big Lake/Atim/Carrot/direct lower-basin area and is not subdivided
        # further without polygon overlays.
        a2 = float(precip.loc[precip.Station == "05EA002", "Shp_Area"].dropna().iloc[0])
        a5 = float(precip.loc[precip.Station == "05EA005", "Shp_Area"].dropna().iloc[0])
        p2 = record.get("rain_05EA002_mm")
        p5 = record.get("rain_05EA005_mm")
        if p2 is not None and p5 is not None and a2 > a5:
            record["rain_incremental_05EA005_to_05EA002_mm"] = (p2 * a2 - p5 * a5) / (a2 - a5)
        # Nested gauge changes and lag to peak during the response window.
        for st in ["05EA005", "05EA006", "05EA010", "05EA011", "05EA012"]:
            s = gauges[st]["stage_m"].dropna().loc[event_start:search_end]
            if s.empty:
                continue
            s0 = nearest_value(gauges[st]["stage_m"].dropna(), event_start)
            pk_t = s.idxmax(); pk = float(s.max())
            record[f"{st}_stage_change_m"] = pk - s0
            record[f"{st}_peak_lag_h"] = float((pk_t - event_start).total_seconds() / 3600)
        rows.append(record)
    return rows


def project_to_target(stage_now: float, model: dict, target: float = 1.70) -> dict:
    dt = 1 / 24
    h = stage_now
    hours = 0
    path = []
    while h > target and hours < 24 * 90:
        rate = min(-0.001, expected_rate(model, h))
        h += rate * dt
        hours += 1
        if hours % 24 == 0:
            path.append({"day": hours // 24, "stage_m": h, "rate_m_per_day": rate})
    return {"hours": hours, "days": hours / 24, "path_daily": path}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gauges = {st: parse_wo(st) for st in STATIONS}
    precip = parse_precip()
    pvt = precip.pivot_table(index="valid_utc", columns="Station", values="PR_mm", aggfunc="mean").sort_index()
    p_target = pvt.get(TARGET, pd.Series(dtype=float)).reindex(gauges[TARGET].index, fill_value=0.0)
    model = fit_master_recession(gauges[TARGET]["stage_m"].dropna(), p_target)
    events = event_calibration(gauges, precip, model)
    current_stage = float(gauges[TARGET]["stage_m"].dropna().iloc[-1])
    projection = project_to_target(current_stage, model)
    latest_time = gauges[TARGET]["stage_m"].dropna().index[-1]
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_stage_utc": latest_time.isoformat(),
        "latest_stage_m": current_stage,
        "master_recession": model,
        "rain_free_projection_to_1_70": projection,
        "event_count": len(events),
        "events": events,
        "limitations": [
            "Empirical calibration based on provisional July 2026 WSC values and HRDPA watershed averages.",
            "05EA005-to-05EA002 incremental precipitation is area-weighted and valid only for that verified nested pair.",
            "Lake storage and routing are represented empirically through observed lags, not a formal reservoir-routing model.",
            "The rain-free projection excludes future precipitation and should be adjusted using quantitative forecast rainfall."
        ]
    }
    (OUT / "calibration.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(events).to_csv(OUT / "event_response_table.csv", index=False)
    pd.DataFrame(projection["path_daily"]).to_csv(OUT / "rain_free_projection.csv", index=False)
    print(json.dumps({"latest_stage_m": current_stage, "events": len(events), "projection_days": projection["days"], "model": model}, indent=2))


if __name__ == "__main__":
    main()
