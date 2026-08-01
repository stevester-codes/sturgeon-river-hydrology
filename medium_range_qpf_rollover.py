#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime

import medium_range_qpf as medium
from qpf_forecast_v2 import get_links, http, parse_run_time

DAY_BUCKETS = ("today", "yesterday")


def bases(base: str) -> list[str]:
    return [base.replace("/today/", f"/{bucket}/", 1) for bucket in DAY_BUCKETS]


def select_gdps_base(session) -> str | None:
    found: list[tuple[datetime, str]] = []
    for base in bases(medium.GDPS_BASE):
        for hour in ("00", "12"):
            try:
                first = medium.gdps_candidate(get_links(session, f"{base}/{hour}/006/"))
                last = medium.gdps_candidate(get_links(session, f"{base}/{hour}/240/"))
            except Exception:
                continue
            first_time = parse_run_time(first) if first else None
            last_time = parse_run_time(last) if last else None
            if first_time and first_time == last_time:
                found.append((first_time, base))
    return max(found, key=lambda item: item[0])[1] if found else None


def select_complete_geps_base(session) -> str | None:
    found: list[tuple[datetime, str]] = []
    for base in bases(medium.GEPS_BASE):
        for hour in ("00", "12"):
            run_times: list[datetime] = []
            complete = True
            for horizon in medium.GEPS_HORIZONS:
                try:
                    candidate = medium.geps_candidate(
                        get_links(session, f"{base}/{hour}/{horizon:03d}/")
                    )
                except Exception:
                    complete = False
                    break
                run_time = medium.geps_run_time(candidate) if candidate else None
                if run_time is None:
                    complete = False
                    break
                run_times.append(run_time)
            if complete and run_times and all(value == run_times[0] for value in run_times):
                found.append((run_times[0], base))
    return max(found, key=lambda item: item[0])[1] if found else None


def main() -> None:
    session = http()
    selected = {}
    gdps_base = select_gdps_base(session)
    if gdps_base:
        medium.GDPS_BASE = gdps_base
        selected["GDPS"] = gdps_base
    geps_base = select_complete_geps_base(session)
    if geps_base:
        medium.GEPS_BASE = geps_base
        selected["GEPS"] = geps_base
    print({"rollover_selected_bases": selected})
    medium.main()


if __name__ == "__main__":
    main()
