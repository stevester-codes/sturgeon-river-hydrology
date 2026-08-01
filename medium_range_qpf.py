#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import medium_range_qpf_base as _base
from medium_range_qpf_base import *  # noqa: F401,F403 - compatibility re-export
from qpf_forecast_v2 import get_links, http, parse_run_time


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
    print({"rollover_selected_bases": selected})
    _base.main()


if __name__ == "__main__":
    main()
