#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline"
DEFAULT_STATION = "05EA002"
TARGET_Q_M3S = 6.77


def session() -> requests.Session:
    value = requests.Session()
    value.headers["User-Agent"] = (
        "sturgeon-river-hydrology-wateroffice-probe/1.1 "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def month_chunks(start: datetime, end: datetime):
    current = start
    while current < end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        chunk_end = min(end, next_month)
        yield current, chunk_end
        current = chunk_end


def identify_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
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
    return date_column, parameter_column, value_column


def request_chunk(
    http: requests.Session,
    station: str,
    start: datetime,
    end: datetime,
) -> tuple[dict, pd.DataFrame]:
    params = [
        ("stations[]", station),
        ("parameters[]", "46"),
        ("parameters[]", "47"),
        ("start_date", start.strftime("%Y-%m-%d %H:%M:%S")),
        ("end_date", end.strftime("%Y-%m-%d %H:%M:%S")),
    ]
    response = http.get(URL, params=params, timeout=300)
    record = {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "request_url": response.url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "bytes_received": len(response.content),
        "availability": "request_failed" if response.status_code != 200 else "unknown",
        "rows": 0,
        "stage_rows": 0,
        "discharge_rows": 0,
        "first_timestamp_utc": None,
        "last_timestamp_utc": None,
        "error": None,
    }
    if response.status_code != 200:
        record["error"] = response.text[:1000]
        return record, pd.DataFrame()
    try:
        frame = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
        if frame.empty:
            record["availability"] = "no_station_data_returned"
            record["note"] = (
                "HTTP request succeeded but the station reported no unit values; "
                "this can represent seasonal gauge shutdown rather than a retrieval failure."
            )
            return record, frame
        date_column, parameter_column, value_column = identify_columns(frame)
        frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
        frame[parameter_column] = pd.to_numeric(frame[parameter_column], errors="coerce")
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=[date_column, parameter_column, value_column])
        record.update(
            {
                "availability": "data_returned",
                "rows": int(len(frame)),
                "stage_rows": int((frame[parameter_column] == 46).sum()),
                "discharge_rows": int((frame[parameter_column] == 47).sum()),
                "first_timestamp_utc": frame[date_column].min().isoformat(),
                "last_timestamp_utc": frame[date_column].max().isoformat(),
                "columns": list(frame.columns),
            }
        )
        normalized = frame[[date_column, parameter_column, value_column]].rename(
            columns={
                date_column: "date_utc",
                parameter_column: "parameter_id",
                value_column: "value",
            }
        )
        return record, normalized
    except Exception as exc:
        record["availability"] = "parse_failed"
        record["error"] = f"{exc.__class__.__name__}: {exc}"
        record["response_prefix"] = response.text[:2000]
        return record, pd.DataFrame()


def coverage_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"status": "no_data", "rows": 0}
    pivot = frame.pivot_table(
        index="date_utc",
        columns="parameter_id",
        values="value",
        aggfunc="median",
    ).rename(columns={46: "stage_m", 47: "discharge_m3s"})
    pivot = pivot.sort_index()
    hourly = pivot.resample("1h").median()
    expected_hours = int(
        (hourly.index.max() - hourly.index.min()).total_seconds() / 3600
    ) + 1
    stage_hours = int(hourly.get("stage_m", pd.Series(dtype=float)).notna().sum())
    discharge_hours = int(
        hourly.get("discharge_m3s", pd.Series(dtype=float)).notna().sum()
    )
    paired_hours = int(hourly.dropna(subset=["stage_m", "discharge_m3s"]).shape[0])
    available_columns = [
        column for column in ["stage_m", "discharge_m3s"] if column in hourly
    ]
    gaps = hourly[available_columns].isna().all(axis=1)
    groups = (gaps != gaps.shift()).cumsum()
    gap_lengths = gaps[gaps].groupby(groups[gaps]).size()
    stage_range = (
        [float(pivot.stage_m.min()), float(pivot.stage_m.max())]
        if "stage_m" in pivot
        else None
    )
    discharge_range = (
        [float(pivot.discharge_m3s.min()), float(pivot.discharge_m3s.max())]
        if "discharge_m3s" in pivot
        else None
    )
    return {
        "status": "data_available",
        "raw_rows": int(len(frame)),
        "first_timestamp_utc": pivot.index.min().isoformat(),
        "last_timestamp_utc": pivot.index.max().isoformat(),
        "calendar_span_days": float(
            (pivot.index.max() - pivot.index.min()).total_seconds() / 86400.0
        ),
        "expected_hourly_steps": expected_hours,
        "stage_hourly_values": stage_hours,
        "discharge_hourly_values": discharge_hours,
        "paired_hourly_values": paired_hours,
        "stage_hourly_coverage_pct": 100.0 * stage_hours / expected_hours,
        "discharge_hourly_coverage_pct": 100.0 * discharge_hours / expected_hours,
        "paired_hourly_coverage_pct": 100.0 * paired_hours / expected_hours,
        "longest_all_parameter_gap_h": int(gap_lengths.max()) if len(gap_lengths) else 0,
        "stage_range_m": stage_range,
        "discharge_range_m3s": discharge_range,
        "target_discharge_m3s": TARGET_Q_M3S,
        "target_discharge_inside_observed_range": bool(
            discharge_range
            and discharge_range[0] <= TARGET_Q_M3S <= discharge_range[1]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument(
        "--output", default="output/archive_probe/wateroffice_probe.json"
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = (now - pd.DateOffset(months=args.months)).to_pydatetime()
    output = {
        "generated_utc": now.isoformat(),
        "status": "failed",
        "service": URL,
        "station": args.station,
        "requested_start_utc": start.isoformat(),
        "requested_end_utc": now.isoformat(),
        "requested_months": args.months,
        "chunks": [],
        "coverage": {},
        "fatal_error": None,
        "next_step": "Inspect saved request diagnostics before expanding calibration.",
    }
    frames: list[pd.DataFrame] = []
    try:
        http = session()
        for chunk_start, chunk_end in month_chunks(start, now):
            record, frame = request_chunk(http, args.station, chunk_start, chunk_end)
            output["chunks"].append(record)
            if not frame.empty:
                frames.append(frame)
        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["date_utc", "parameter_id"], keep="last")
            .sort_values(["date_utc", "parameter_id"])
            if frames
            else pd.DataFrame()
        )
        coverage = coverage_summary(combined)
        output["coverage"] = coverage
        data_chunks = sum(
            record.get("availability") == "data_returned"
            for record in output["chunks"]
        )
        empty_chunks = sum(
            record.get("availability") == "no_station_data_returned"
            for record in output["chunks"]
        )
        failed_chunks = sum(
            record.get("availability") in {"request_failed", "parse_failed"}
            for record in output["chunks"]
        )
        output.update(
            {
                "data_chunks": data_chunks,
                "seasonal_no_data_chunks": empty_chunks,
                "failed_chunks": failed_chunks,
                "chunk_count": len(output["chunks"]),
                "availability_interpretation": (
                    "05EA002 is seasonally operated; successful header-only winter responses are retained as seasonal no-data intervals and are not silently filled."
                ),
            }
        )

        latest = pd.Timestamp(coverage.get("last_timestamp_utc")) if coverage.get("last_timestamp_utc") else None
        recent_enough = bool(
            latest is not None
            and (pd.Timestamp(now) - latest).total_seconds() <= 72 * 3600
        )
        criteria = {
            "no_http_or_parse_failures": failed_chunks == 0,
            "at_least_ten_months_with_data": data_chunks >= 10,
            "at_least_300_calendar_days_spanned": coverage.get("calendar_span_days", 0) >= 300,
            "at_least_60_percent_hourly_coverage_across_calendar_span": coverage.get("paired_hourly_coverage_pct", 0) >= 60,
            "target_discharge_inside_observed_range": bool(
                coverage.get("target_discharge_inside_observed_range")
            ),
            "latest_data_within_72_hours": recent_enough,
        }
        output["acceptance_criteria"] = criteria
        if combined.empty:
            output["status"] = "failed_no_data"
        elif all(criteria.values()):
            output["status"] = "passed_seasonal_open_water_archive"
            output["next_step"] = (
                "Retrieve all basin gauges for the same open-water periods and pair them with archived RDPA for multi-season event hindcasting."
            )
        else:
            output["status"] = "partial"
            output["next_step"] = (
                "Use only verified open-water intervals and resolve any failed acceptance criterion before candidate calibration."
            )
    except Exception as exc:
        output["fatal_error"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        out.write_text(json.dumps(output, indent=2))
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "data_chunks": output.get("data_chunks"),
                    "seasonal_no_data_chunks": output.get("seasonal_no_data_chunks"),
                    "failed_chunks": output.get("failed_chunks"),
                    "coverage": output.get("coverage"),
                    "acceptance_criteria": output.get("acceptance_criteria"),
                    "fatal_error": output["fatal_error"],
                },
                indent=2,
            )
        )

    if output["status"] != "passed_seasonal_open_water_archive":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
