#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import shapefile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from historical_gauge_analysis import (
    STATION,
    TARGET_Q,
    fit_recession,
    predict_rate,
    project_value,
    rate_metrics,
    season,
)
from wateroffice_archive_probe import month_chunks, request_chunk, session as wateroffice_session

OUT_DEFAULT = Path("output/archive_probe/historical_rdpa_pairing.json")
PAIRS_DEFAULT = Path("output/archive_probe/historical_rdpa_pairs.csv")
HOSTS = ("dd.weather.gc.ca", "dd.meteo.gc.ca")
RDPA_ACCUM = "006"
RDPA_CUTOFF = "0700"
RDPA_GROUP = "05"
THREAD_LOCAL = threading.local()


def rdpa_session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is not None:
        return value
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-historical-rdpa/1.0 "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=2,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    value.mount("https://", HTTPAdapter(max_retries=retry))
    THREAD_LOCAL.session = value
    return value


def floor_6h(timestamp: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("6h")


def valid_times_for_candidates(index: pd.DatetimeIndex, lookback_hours: int) -> list[pd.Timestamp]:
    values: set[pd.Timestamp] = set()
    steps = int(math.ceil(lookback_hours / 6.0))
    for timestamp in index:
        end = floor_6h(timestamp)
        for step in range(steps + 1):
            values.add(end - pd.Timedelta(hours=6 * step))
    return sorted(values)


def candidate_urls(valid: pd.Timestamp) -> Iterable[str]:
    valid = floor_6h(valid)
    filename = (
        f"CMC_HRDPA_WATERSHED-{RDPA_ACCUM}-{RDPA_CUTOFF}cutoff_"
        f"SFC_0_ps2.5km_{valid:%Y%m%d%H}_000_{RDPA_GROUP}.dbf"
    )
    # Final 0700-cutoff fields may be published in either the valid-date folder
    # or the following publication-date folder. Try the primary host first.
    for publication_day in (valid.normalize(), valid.normalize() + pd.Timedelta(days=1)):
        for host in HOSTS:
            yield (
                f"https://{host}/{publication_day:%Y%m%d}/WXO-DD/analysis/precip/"
                f"hrdpa_watershed/shapefile/06/{filename}"
            )


def parse_station_record(content: bytes, station: str) -> dict | None:
    reader = shapefile.Reader(dbf=io.BytesIO(content))
    fields = [field[0] for field in reader.fields[1:]]
    station_field = "Station" if "Station" in fields else next(
        (field for field in fields if "stat" in field.lower()), None
    )
    precip_field = "PR_mm" if "PR_mm" in fields else next(
        (field for field in fields if field.lower().startswith("pr")), None
    )
    if station_field is None or precip_field is None:
        return None
    for record in reader.records():
        row = record.as_dict()
        if str(row.get(station_field, "")).strip() == station:
            return {
                "station": station,
                "precip_mm": float(row.get(precip_field, 0.0) or 0.0),
                "cfia": (
                    float(row["CFIA"]) if row.get("CFIA") not in (None, "") else None
                ),
            }
    return None


def fetch_one(valid: pd.Timestamp, station: str) -> dict:
    errors: list[str] = []
    http = rdpa_session()
    for url in candidate_urls(valid):
        try:
            response = http.get(url, timeout=45)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            row = parse_station_record(response.content, station)
            if row is None:
                errors.append(f"station_or_precip_field_missing:{url}")
                continue
            return {
                "valid_utc": floor_6h(valid).isoformat(),
                "status": "retrieved",
                "source_url": url,
                **row,
            }
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}:{exc}")
    return {
        "valid_utc": floor_6h(valid).isoformat(),
        "status": "missing",
        "station": station,
        "precip_mm": None,
        "cfia": None,
        "source_url": None,
        "errors": errors[-4:],
    }


def fetch_rdpa(valid_times: list[pd.Timestamp], station: str, workers: int) -> pd.DataFrame:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_one, valid, station): valid for valid in valid_times}
        for future in as_completed(futures):
            rows.append(future.result())
    frame = pd.DataFrame(rows)
    frame["valid_utc"] = pd.to_datetime(frame.valid_utc, utc=True)
    frame = frame.sort_values("valid_utc").drop_duplicates("valid_utc", keep="last")
    return frame.set_index("valid_utc")


def build_hourly(months: int) -> tuple[pd.DataFrame, list[dict]]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = (pd.Timestamp(now) - pd.DateOffset(months=months)).to_pydatetime()
    records: list[dict] = []
    frames: list[pd.DataFrame] = []
    http = wateroffice_session()
    for chunk_start, chunk_end in month_chunks(start, now):
        record, frame = request_chunk(http, STATION, chunk_start, chunk_end)
        records.append(record)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No historical unit-value data were retrieved")
    raw = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date_utc", "parameter_id"], keep="last")
        .sort_values(["date_utc", "parameter_id"])
    )
    pivot = raw.pivot_table(
        index="date_utc", columns="parameter_id", values="value", aggfunc="median"
    ).rename(columns={46: "stage_m", 47: "discharge_m3s"})
    hourly = pivot.sort_index().resample("1h").median().interpolate(limit=2)
    hourly = hourly.dropna(subset=["stage_m", "discharge_m3s"])
    hourly["stage_change_6h_m"] = hourly.stage_m.diff(6)
    hourly["stage_change_24h_m"] = hourly.stage_m.diff(24)
    hourly["q_change_6h_m3s"] = hourly.discharge_m3s.diff(6)
    hourly["q_change_24h_m3s"] = hourly.discharge_m3s.diff(24)
    hourly["stage_rate_m_per_day"] = hourly.stage_m.diff(6) / 6.0 * 24.0
    hourly["q_rate_m3s_per_day"] = hourly.discharge_m3s.diff(6) / 6.0 * 24.0
    hourly["limb"] = np.where(
        hourly.stage_change_6h_m >= 0.003,
        "rising",
        np.where(hourly.stage_change_6h_m <= -0.003, "falling", "approximately_flat"),
    )
    hourly["season"] = [season(timestamp.month) for timestamp in hourly.index]
    hourly["year"] = hourly.index.year
    return hourly, records


def gauge_only_recession(hourly: pd.DataFrame) -> pd.DataFrame:
    recent_max_rise = hourly.stage_change_6h_m.rolling(24, min_periods=24).max()
    return hourly[
        (hourly.stage_change_24h_m < -0.01)
        & (hourly.stage_change_6h_m < -0.001)
        & (recent_max_rise <= 0.003)
        & (hourly.stage_rate_m_per_day > -0.30)
        & (hourly.q_rate_m3s_per_day > -20.0)
    ].copy()


def rolling_rain_at_points(
    points: pd.DatetimeIndex, rdpa: pd.DataFrame, windows: tuple[int, ...]
) -> pd.DataFrame:
    precipitation = rdpa.precip_mm.astype(float)
    available = rdpa.status.eq("retrieved").astype(int)
    output = pd.DataFrame(index=points)
    for hours in windows:
        sums: list[float] = []
        coverage: list[float] = []
        max_6h: list[float] = []
        expected = max(1, int(math.ceil(hours / 6.0)))
        for timestamp in points:
            end = floor_6h(timestamp)
            start = end - pd.Timedelta(hours=hours - 6)
            values = precipitation.loc[(precipitation.index >= start) & (precipitation.index <= end)]
            flags = available.loc[(available.index >= start) & (available.index <= end)]
            valid = values.dropna()
            sums.append(float(valid.sum()) if len(valid) else math.nan)
            max_6h.append(float(valid.max()) if len(valid) else math.nan)
            coverage.append(float(flags.sum()) / expected)
        output[f"rain_{hours}h_mm"] = sums
        output[f"rain_{hours}h_coverage"] = coverage
        output[f"rain_{hours}h_max6h_mm"] = max_6h
    return output


def assign_events(frame: pd.DataFrame, gap_hours: int = 12) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="int64", index=frame.index)
    gaps = frame.index.to_series().diff().dt.total_seconds().div(3600.0)
    return (gaps.isna() | (gaps > gap_hours)).cumsum().astype(int)


def fit_and_validate(frame: pd.DataFrame, label: str, current_q: float) -> dict:
    fit = fit_recession(frame, "discharge_m3s", "q_rate_m3s_per_day")
    result: dict = {
        "label": label,
        "points": int(len(frame)),
        "events": 0,
        "fit": fit,
        "projection": None,
        "event_block_cross_validation": {},
    }
    if frame.empty or fit is None:
        return result
    work = frame.sort_index().copy()
    work["event_id"] = assign_events(work)
    event_sizes = work.groupby("event_id").size()
    usable_events = [int(event) for event, size in event_sizes.items() if size >= 24]
    work = work[work.event_id.isin(usable_events)]
    result["events"] = len(usable_events)
    result["event_sizes_hours"] = {
        str(int(event)): int(size)
        for event, size in event_sizes.items()
        if int(event) in usable_events
    }
    result["projection"] = project_value(float(current_q), TARGET_Q, fit)
    observed_all: list[float] = []
    predicted_all: list[float] = []
    folds: list[dict] = []
    if len(usable_events) >= 3:
        for event in usable_events:
            train = work[work.event_id != event]
            test = work[work.event_id == event]
            train_fit = fit_recession(train, "discharge_m3s", "q_rate_m3s_per_day")
            if train_fit is None or len(test) < 24:
                continue
            observed = test.q_rate_m3s_per_day.to_numpy(float)
            predicted = predict_rate(test.discharge_m3s.to_numpy(float), train_fit)
            metrics = rate_metrics(observed, predicted)
            metrics.update(
                {
                    "held_out_event": int(event),
                    "train_points": int(len(train)),
                    "test_first_utc": test.index.min().isoformat(),
                    "test_last_utc": test.index.max().isoformat(),
                }
            )
            folds.append(metrics)
            observed_all.extend(observed.tolist())
            predicted_all.extend(predicted.tolist())
    result["event_block_cross_validation"] = {
        "available": bool(folds),
        "folds": folds,
        "aggregate": (
            rate_metrics(np.asarray(observed_all), np.asarray(predicted_all))
            if folds
            else None
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lookback-hours", type=int, default=168)
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument("--pairs-output", default=str(PAIRS_DEFAULT))
    args = parser.parse_args()

    out = Path(args.output)
    pairs_out = Path(args.pairs_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).replace(microsecond=0)

    hourly, retrieval_records = build_hourly(args.months)
    gauge = gauge_only_recession(hourly)
    if gauge.empty:
        raise RuntimeError("No gauge-only recession points were identified")

    valid_times = valid_times_for_candidates(gauge.index, args.lookback_hours)
    rdpa = fetch_rdpa(valid_times, STATION, args.workers)
    rain = rolling_rain_at_points(gauge.index, rdpa, (24, 72, 168))
    paired = gauge.join(rain)
    paired["complete_168h_coverage"] = paired.rain_168h_coverage >= 0.999

    # Two transparent rainfall screens. Neither is promoted automatically.
    strict = paired[
        paired.complete_168h_coverage
        & (paired.rain_24h_mm <= 0.5)
        & (paired.rain_72h_mm <= 1.5)
        & (paired.rain_168h_mm <= 5.0)
    ].copy()
    moderate = paired[
        paired.complete_168h_coverage
        & (paired.rain_24h_mm <= 1.0)
        & (paired.rain_72h_mm <= 3.0)
        & (paired.rain_168h_mm <= 10.0)
    ].copy()

    current_q = float(hourly.discharge_m3s.iloc[-1])
    gauge_model = fit_and_validate(paired, "gauge_only_with_rdpa_metrics", current_q)
    strict_model = fit_and_validate(strict, "rdpa_strict_dry_screen", current_q)
    moderate_model = fit_and_validate(moderate, "rdpa_moderate_dry_screen", current_q)

    required = int(len(rdpa))
    retrieved = int(rdpa.status.eq("retrieved").sum())
    coverage = retrieved / required if required else 0.0
    promotion_reasons: list[str] = []
    strict_cv = strict_model.get("event_block_cross_validation", {}).get("aggregate")
    gauge_cv = gauge_model.get("event_block_cross_validation", {}).get("aggregate")
    if coverage < 0.90:
        promotion_reasons.append("archived_rdpa_coverage_below_90_percent")
    if strict_model.get("events", 0) < 3:
        promotion_reasons.append("fewer_than_three_strict_dry_recession_events")
    if strict_model.get("points", 0) < 200:
        promotion_reasons.append("fewer_than_200_strict_dry_recession_points")
    if strict_cv is None:
        promotion_reasons.append("strict_event_block_cross_validation_unavailable")
    if strict_cv and gauge_cv and strict_cv["rmse_per_day"] > gauge_cv["rmse_per_day"]:
        promotion_reasons.append("strict_rdpa_screen_does_not_improve_event_block_rmse")

    columns = [
        "stage_m",
        "discharge_m3s",
        "stage_change_6h_m",
        "stage_change_24h_m",
        "q_change_6h_m3s",
        "q_change_24h_m3s",
        "stage_rate_m_per_day",
        "q_rate_m3s_per_day",
        "limb",
        "season",
        "year",
        "rain_24h_mm",
        "rain_24h_coverage",
        "rain_72h_mm",
        "rain_72h_coverage",
        "rain_168h_mm",
        "rain_168h_coverage",
        "complete_168h_coverage",
    ]
    paired[columns].to_csv(pairs_out, index_label="date_utc")

    output = {
        "generated_utc": generated.isoformat(),
        "status": "historical_rdpa_pairing_complete",
        "mode": "shadow_only_no_automatic_promotion",
        "station": STATION,
        "target_discharge_m3s": TARGET_Q,
        "requested_months": args.months,
        "wateroffice_retrieval": {
            "chunks": retrieval_records,
            "hourly_points": int(len(hourly)),
            "first_utc": hourly.index.min().isoformat(),
            "last_utc": hourly.index.max().isoformat(),
        },
        "rdpa_retrieval": {
            "product": "HRDPA watershed 6-hour final 0700-cutoff",
            "station_basin": STATION,
            "requested_periods": required,
            "retrieved_periods": retrieved,
            "missing_periods": required - retrieved,
            "coverage_fraction": coverage,
            "first_valid_utc": rdpa.index.min().isoformat() if required else None,
            "last_valid_utc": rdpa.index.max().isoformat() if required else None,
            "missing_examples": rdpa[rdpa.status != "retrieved"]
            .head(20)
            .reset_index()
            .to_dict("records"),
        },
        "pairing": {
            "gauge_only_candidate_points": int(len(paired)),
            "complete_168h_coverage_points": int(paired.complete_168h_coverage.sum()),
            "strict_dry_points": int(len(strict)),
            "moderate_dry_points": int(len(moderate)),
            "strict_thresholds_mm": {"24h": 0.5, "72h": 1.5, "168h": 5.0},
            "moderate_thresholds_mm": {"24h": 1.0, "72h": 3.0, "168h": 10.0},
            "pairs_csv": str(pairs_out),
        },
        "models": {
            "gauge_only": gauge_model,
            "rdpa_strict": strict_model,
            "rdpa_moderate": moderate_model,
        },
        "promotion_screen": {
            "automatic_promotion_enabled": False,
            "candidate_passes_minimum_screen": not promotion_reasons,
            "reasons_not_to_promote": promotion_reasons,
            "requirements": [
                "at least 90 percent archived RDPA coverage",
                "at least three distinct strict dry recession events",
                "at least 200 strict dry recession points",
                "event-block cross-validation available",
                "strict dry screening does not worsen event-block RMSE",
                "manual engineering review before any operational change",
            ],
        },
        "interpretation": (
            "The independent direct-discharge recession has now been paired with archived "
            "HRDPA watershed precipitation. Strict and moderate dry screens remain shadow "
            "sensitivities until event-block validation and engineering review support promotion."
        ),
        "limitations": [
            "HRDPA watershed precipitation is an areal analysis and may miss sub-grid convective detail.",
            "WSC discharge remains provisional and rating-derived from stage.",
            "Hourly recession points within one event are autocorrelated; event-block validation is therefore reported separately.",
            "Seven-day rainfall screening may not fully represent longer lake and wetland storage memory.",
        ],
    }
    out.write_text(json.dumps(output, indent=2, default=str))
    print(
        json.dumps(
            {
                "status": output["status"],
                "rdpa_coverage_fraction": coverage,
                "gauge_points": len(paired),
                "strict_points": len(strict),
                "moderate_points": len(moderate),
                "gauge_projection_days": (gauge_model.get("projection") or {}).get("days"),
                "strict_projection_days": (strict_model.get("projection") or {}).get("days"),
                "moderate_projection_days": (moderate_model.get("projection") or {}).get("days"),
                "promotion_screen": output["promotion_screen"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
