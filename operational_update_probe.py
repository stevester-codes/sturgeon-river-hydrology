#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WATER_OFFICE = "https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline"
HRDPS_BASE = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"
GEPS_BASE = "https://dd.weather.gc.ca/today/ensemble/geps/grib2/raw"
HREF = re.compile(r'href="([^"?]+)"', re.I)
HRDPS_RUN = re.compile(r"^(\d{8})T(\d{2})Z_")
GEPS_RUN = re.compile(r"_(\d{10})_P\d{3}_allmbrs\.grib2$", re.I)


def utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def session() -> requests.Session:
    result = requests.Session()
    result.headers["User-Agent"] = (
        "sturgeon-river-hydrology/update-probe "
        "(stevester-codes@users.noreply.github.com)"
    )
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    result.mount("https://", HTTPAdapter(max_retries=retry))
    return result


def links(http: requests.Session, url: str) -> list[str]:
    response = http.get(url, timeout=30)
    response.raise_for_status()
    return HREF.findall(response.text)


def hrdps_candidate(names: list[str]) -> str | None:
    candidates = []
    for name in names:
        lower = name.lower()
        if not lower.endswith(".grib2"):
            continue
        if "_msc_hrdps_apcp-accum6h_" not in lower:
            continue
        if any(token in lower for token in ["-prob_", "snow", "freezing", "convective"]):
            continue
        candidates.append(name)
    return sorted(candidates, key=len)[0] if candidates else None


def hrdps_time(filename: str | None) -> datetime | None:
    if not filename:
        return None
    match = HRDPS_RUN.match(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H").replace(
        tzinfo=timezone.utc
    )


def latest_complete_hrdps(http: requests.Session) -> datetime | None:
    complete: list[datetime] = []
    for hour in ["00", "06", "12", "18"]:
        try:
            first = hrdps_candidate(links(http, f"{HRDPS_BASE}/{hour}/006/"))
            last = hrdps_candidate(links(http, f"{HRDPS_BASE}/{hour}/048/"))
        except Exception:
            continue
        first_time = hrdps_time(first)
        last_time = hrdps_time(last)
        if first_time and first_time == last_time:
            complete.append(first_time)
    return max(complete) if complete else None


def geps_candidate(names: list[str]) -> str | None:
    candidates = []
    for name in names:
        lower = name.lower()
        if (
            lower.endswith("_allmbrs.grib2")
            and "cmc_geps-raw_apcp_sfc_0_" in lower
        ):
            candidates.append(name)
    return sorted(candidates, key=len)[0] if candidates else None


def geps_time(filename: str | None) -> datetime | None:
    if not filename:
        return None
    match = GEPS_RUN.search(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def geps_publication_state(
    http: requests.Session,
) -> tuple[datetime | None, datetime | None]:
    """Return newest advertised PT024 run and newest fully published PT384 run."""
    advertised: list[datetime] = []
    complete: list[datetime] = []
    for hour in ["00", "12"]:
        try:
            first = geps_candidate(links(http, f"{GEPS_BASE}/{hour}/024/"))
            last = geps_candidate(links(http, f"{GEPS_BASE}/{hour}/384/"))
        except Exception:
            continue
        first_time = geps_time(first)
        last_time = geps_time(last)
        if first_time:
            advertised.append(first_time)
        if first_time and first_time == last_time:
            complete.append(first_time)
    return (
        max(advertised) if advertised else None,
        max(complete) if complete else None,
    )


def latest_gauge(http: requests.Session) -> tuple[datetime | None, float | None]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=2)
    params = [
        ("stations[]", "05EA002"),
        ("parameters[]", "46"),
        ("start_date", start.strftime("%Y-%m-%d %H:%M:%S")),
        ("end_date", end.strftime("%Y-%m-%d %H:%M:%S")),
    ]
    response = http.get(WATER_OFFICE, params=params, timeout=45)
    response.raise_for_status()
    rows = csv.DictReader(io.StringIO(response.text.lstrip("\ufeff")))
    latest_time: datetime | None = None
    latest_value: float | None = None
    for row in rows:
        observed = utc(row.get("Date"))
        try:
            value = float(row.get("Value/Valeur", ""))
        except ValueError:
            continue
        if observed and (latest_time is None or observed > latest_time):
            latest_time = observed
            latest_value = value
    return latest_time, latest_value


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def published_hrdps(root: Path) -> datetime | None:
    data = load(root / "forecast_v2/last_valid_hrdps.json")
    values = [
        utc(row.get("run_time_utc"))
        for row in data.get("deterministic_scenarios", [])
        if row.get("model") == "HRDPS" and row.get("complete_horizon")
    ]
    values = [value for value in values if value]
    return max(values) if values else None


def published_geps(root: Path) -> datetime | None:
    data = load(root / "spatial/medium_range_qpf.json")
    return utc(data.get("geps", {}).get("run_time_utc"))


def published_gauge(root: Path) -> tuple[datetime | None, float | None]:
    data = load(root / "forecast_v2/construction_readiness.json")
    observed = utc(data.get("latest_stage_utc"))
    value = data.get("current_conditions", {}).get("stage_05EA002_m")
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = None
    return observed, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-root", default="output/latest")
    parser.add_argument("--json-output", default="update_probe.json")
    parser.add_argument("--github-output")
    args = parser.parse_args()

    root = Path(args.published_root)
    http = session()
    errors: list[str] = []

    try:
        source_gauge_time, source_gauge_value = latest_gauge(http)
    except Exception as exc:
        source_gauge_time, source_gauge_value = None, None
        errors.append(f"gauge probe failed: {exc}")
    try:
        source_hrdps = latest_complete_hrdps(http)
    except Exception as exc:
        source_hrdps = None
        errors.append(f"HRDPS probe failed: {exc}")
    try:
        advertised_geps, source_geps = geps_publication_state(http)
    except Exception as exc:
        advertised_geps, source_geps = None, None
        errors.append(f"GEPS probe failed: {exc}")

    old_gauge_time, old_gauge_value = published_gauge(root)
    old_hrdps = published_hrdps(root)
    old_geps = published_geps(root)
    completed = None
    completed_path = root / "workflow_completed_utc.txt"
    if completed_path.exists():
        completed = utc(completed_path.read_text().strip())

    publication_block = bool(
        advertised_geps
        and (old_geps is None or advertised_geps > old_geps)
        and (source_geps is None or source_geps < advertised_geps)
    )

    candidate_reasons: list[str] = []
    if source_hrdps and (old_hrdps is None or source_hrdps > old_hrdps):
        candidate_reasons.append(f"complete HRDPS cycle {source_hrdps.isoformat()} is new")
    if source_geps and (old_geps is None or source_geps > old_geps):
        candidate_reasons.append(f"complete GEPS cycle {source_geps.isoformat()} is new")

    if source_gauge_time:
        age_since_published = (
            (source_gauge_time - old_gauge_time).total_seconds() / 60
            if old_gauge_time
            else 9999
        )
        stage_change = (
            abs(source_gauge_value - old_gauge_value)
            if source_gauge_value is not None and old_gauge_value is not None
            else None
        )
        if age_since_published >= 30 and (stage_change is None or stage_change >= 0.002):
            candidate_reasons.append(
                f"05EA002 is {age_since_published:.0f} minutes newer"
                + (f" and changed {stage_change:.3f} m" if stage_change is not None else "")
            )
        elif completed and (datetime.now(timezone.utc) - completed).total_seconds() >= 2 * 3600:
            candidate_reasons.append("published operational package is at least two hours old")

    reasons = [] if publication_block else candidate_reasons
    waiting_reason = None
    if publication_block:
        waiting_reason = (
            f"GEPS cycle {advertised_geps.isoformat()} has begun publishing but "
            "the PT384 all-member file is not yet available"
        )

    result = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dispatch_required": bool(reasons),
        "dispatch_blocked_for_incomplete_geps": publication_block,
        "waiting_reason": waiting_reason,
        "candidate_reasons": candidate_reasons,
        "reasons": reasons,
        "errors": errors,
        "source": {
            "gauge_time_utc": source_gauge_time.isoformat() if source_gauge_time else None,
            "gauge_stage_m": source_gauge_value,
            "complete_hrdps_run_utc": source_hrdps.isoformat() if source_hrdps else None,
            "advertised_geps_run_utc": advertised_geps.isoformat() if advertised_geps else None,
            "complete_geps_run_utc": source_geps.isoformat() if source_geps else None,
        },
        "published": {
            "gauge_time_utc": old_gauge_time.isoformat() if old_gauge_time else None,
            "gauge_stage_m": old_gauge_value,
            "hrdps_run_utc": old_hrdps.isoformat() if old_hrdps else None,
            "geps_run_utc": old_geps.isoformat() if old_geps else None,
            "workflow_completed_utc": completed.isoformat() if completed else None,
        },
        "interpretation": (
            "The detector runs every ten minutes. It dispatches for complete new weather "
            "cycles, meaningful newer gauge observations, or a two-hour safety refresh. "
            "It temporarily waits while a newer GEPS cycle is only partly published, then "
            "dispatches on the first check after PT384 becomes available."
        ),
    }
    Path(args.json_output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if args.github_output:
        reason_text = "; ".join(reasons).replace("\n", " ")
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"dispatch={'true' if reasons else 'false'}\n")
            handle.write(f"reason={reason_text}\n")


if __name__ == "__main__":
    main()
