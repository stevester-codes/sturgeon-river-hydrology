#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime

import qpf_forecast_v2 as qpf

DAY_BUCKETS = ("today", "yesterday")


def bases(base: str) -> list[str]:
    return [base.replace("/today/", f"/{bucket}/", 1) for bucket in DAY_BUCKETS]


def select_deterministic_base(session, model: str) -> str | None:
    configured = qpf.MODEL_CONFIG[model]["base"]
    found: list[tuple[datetime, str]] = []
    for base in bases(configured):
        for hour in ("00", "06", "12", "18"):
            try:
                first = qpf.deterministic_candidate(
                    qpf.get_links(session, f"{base}/{hour}/006/"), model
                )
                last = qpf.deterministic_candidate(
                    qpf.get_links(
                        session,
                        f"{base}/{hour}/{int(qpf.MODEL_CONFIG[model]['max_hour']):03d}/",
                    ),
                    model,
                )
            except Exception:
                continue
            first_time = qpf.parse_run_time(first) if first else None
            last_time = qpf.parse_run_time(last) if last else None
            if first_time and first_time == last_time:
                found.append((first_time, base))
    return max(found, key=lambda item: item[0])[1] if found else None


def select_reps_base(session) -> str | None:
    found: list[tuple[datetime, str]] = []
    for base in bases(qpf.REPS_BASE):
        for hour in ("00", "06", "12", "18"):
            try:
                first = qpf.reps_candidate(
                    qpf.get_links(session, f"{base}/{hour}/024/"), 24, False
                )
                last = qpf.reps_candidate(
                    qpf.get_links(session, f"{base}/{hour}/048/"), 48, False
                )
            except Exception:
                continue
            first_time = qpf.parse_run_time(first) if first else None
            last_time = qpf.parse_run_time(last) if last else None
            if first_time and first_time == last_time:
                found.append((first_time, base))
    return max(found, key=lambda item: item[0])[1] if found else None


def main() -> None:
    session = qpf.http()
    selected = {}
    for model in ("HRDPS", "RDPS"):
        base = select_deterministic_base(session, model)
        if base:
            qpf.MODEL_CONFIG[model]["base"] = base
            selected[model] = base
    reps_base = select_reps_base(session)
    if reps_base:
        qpf.REPS_BASE = reps_base
        selected["REPS"] = reps_base
    print({"rollover_selected_bases": selected})
    qpf.main()


if __name__ == "__main__":
    main()
