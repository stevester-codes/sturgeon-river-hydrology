#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import medium_range_qpf_base as _base
from medium_range_qpf_base import *  # noqa: F401,F403 - compatibility re-export
from qpf_forecast_v2 import get_links, http, parse_run_time

MAX_GEPS_CARRY_FORWARD_HOURS = 24.0


def _bases(base: str) -> list[str]:
    now = datetime.now(timezone.utc)
    candidates = [base]
    for days_back in (0, 1, 2):
        date_bucket = (now - timedelta(days=days_back)).strftime("%Y%m%d")
        candidates.append(
            base.replace("/today/", f"/{date_bucket}/WXO-DD/", 1)
        )
    return list(dict.fromkeys(candidates))


def _select_gdps_base(session) -> str | None:
    found: list[tuple[datetime, str]] = []
    for base in _bases(_base.GDPS_BASE):
        for hour in ("00", "12"):
            try:
                first = _base.gdps_candidate(get_links(session, f"{base}/{hour}/006/"))
                last = _base.gdps_candidate(get_links(session, f"{base}/{hour}/240/"))
            except Exception:
                continue
            first_time = parse_run_time(first) if first else None
            last_time = parse_run_time(last) if last else None
            if first_time and first_time == last_time:
                found.append((first_time, base))
    return max(found, key=lambda item: item[0])[1] if found else None


def _select_complete_geps_base(session) -> str | None:
    found: list[tuple[datetime, str]] = []
    for base in _bases(_base.GEPS_BASE):
        for hour in ("00", "12"):
            run_times: list[datetime] = []
            complete = True
            for horizon in _base.GEPS_HORIZONS:
                try:
                    candidate = _base.geps_candidate(
                        get_links(session, f"{base}/{hour}/{horizon:03d}/")
                    )
                except Exception:
                    complete = False
                    break
                run_time = _base.geps_run_time(candidate) if candidate else None
                if run_time is None:
                    complete = False
                    break
                run_times.append(run_time)
            if complete and run_times and all(value == run_times[0] for value in run_times):
                found.append((run_times[0], base))
    return max(found, key=lambda item: item[0])[1] if found else None


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _carry_forward_valid_geps() -> dict | None:
    spatial = Path("sturgeon_pipeline_output") / "spatial"
    metadata_path = spatial / "medium_range_qpf.json"
    geps_path = spatial / "geps_qpf_by_subarea.csv"
    if not metadata_path.exists() or not geps_path.exists() or geps_path.stat().st_size <= 100:
        return None
    try:
        data = json.loads(metadata_path.read_text())
    except Exception:
        return None
    geps = data.get("geps", {})
    run_time = _parse_utc(geps.get("run_time_utc"))
    now = datetime.now(timezone.utc)
    age_hours = None if run_time is None else (now - run_time).total_seconds() / 3600.0
    horizons = sorted(int(value) for value in geps.get("horizons_processed_h", []))
    valid = (
        geps.get("status") == "processed"
        and geps.get("validation", {}).get("passed") is True
        and horizons == sorted(_base.GEPS_HORIZONS)
        and age_hours is not None
        and -0.25 <= age_hours <= MAX_GEPS_CARRY_FORWARD_HOURS
    )
    if not valid:
        return None
    geps["carry_forward"] = {
        "active": True,
        "reason": "No complete newer GEPS package was discoverable during ECCC publication or UTC-day rollover.",
        "source_run_time_utc": run_time.isoformat(),
        "age_hours_at_refresh": age_hours,
        "maximum_age_hours": MAX_GEPS_CARRY_FORWARD_HOURS,
        "interpretation": "The last complete ensemble is retained; missing source files are not interpreted as zero rainfall.",
    }
    data["generated_utc"] = now.isoformat()
    data["geps"] = geps
    data["warning_count"] = int(data.get("warning_count") or 0) + 1
    data["operational_source_state"] = "validated_geps_carry_forward"
    metadata_path.write_text(json.dumps(data, indent=2))
    warning_path = spatial / "medium_range_qpf_warnings.log"
    existing = warning_path.read_text() if warning_path.exists() else ""
    warning_path.write_text(
        existing.rstrip()
        + "\nCarried forward the last complete GEPS package because no complete newer source package was discoverable.\n"
    )
    return geps["carry_forward"]


def main() -> None:
    session = http()
    selected: dict[str, str] = {}
    gdps_base = _select_gdps_base(session)
    if gdps_base:
        _base.GDPS_BASE = gdps_base
        selected["GDPS"] = gdps_base
    geps_base = _select_complete_geps_base(session)
    if geps_base:
        _base.GEPS_BASE = geps_base
        selected["GEPS"] = geps_base
    if geps_base is None:
        carry = _carry_forward_valid_geps()
        if carry is not None:
            print(json.dumps({"rollover_selected_bases": selected, "geps_carry_forward": carry}, indent=2))
            return
    print({"rollover_selected_bases": selected})
    _base.main()


if __name__ == "__main__":
    main()
