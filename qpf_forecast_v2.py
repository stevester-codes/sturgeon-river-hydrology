#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import qpf_forecast_v2_base as _base
from qpf_forecast_v2_base import *  # noqa: F401,F403 - compatibility re-export


def _bases(base: str) -> list[str]:
    now = datetime.now(timezone.utc)
    candidates = [base]
    for days_back in (0, 1, 2):
        date_bucket = (now - timedelta(days=days_back)).strftime("%Y%m%d")
        candidates.append(
            base.replace("/today/", f"/{date_bucket}/WXO-DD/", 1)
        )
    return list(dict.fromkeys(candidates))


def _select_deterministic_base(session, model: str) -> str | None:
    configured = _base.MODEL_CONFIG[model]["base"]
    found: list[tuple[datetime, str]] = []
    final_hour = int(_base.MODEL_CONFIG[model]["max_hour"])
    for base in _bases(configured):
        for hour in ("00", "06", "12", "18"):
            try:
                first = _base.deterministic_candidate(
                    _base.get_links(session, f"{base}/{hour}/006/"), model
                )
                last = _base.deterministic_candidate(
                    _base.get_links(session, f"{base}/{hour}/{final_hour:03d}/"), model
                )
            except Exception:
                continue
            first_time = _base.parse_run_time(first) if first else None
            last_time = _base.parse_run_time(last) if last else None
            if first_time and first_time == last_time:
                found.append((first_time, base))
    return max(found, key=lambda item: item[0])[1] if found else None


def _select_reps_base(session) -> str | None:
    found: list[tuple[datetime, str]] = []
    for base in _bases(_base.REPS_BASE):
        for hour in ("00", "06", "12", "18"):
            try:
                first = _base.reps_candidate(
                    _base.get_links(session, f"{base}/{hour}/024/"), 24, False
                )
                last = _base.reps_candidate(
                    _base.get_links(session, f"{base}/{hour}/048/"), 48, False
                )
            except Exception:
                continue
            first_time = _base.parse_run_time(first) if first else None
            last_time = _base.parse_run_time(last) if last else None
            if first_time and first_time == last_time:
                found.append((first_time, base))
    return max(found, key=lambda item: item[0])[1] if found else None


def main() -> None:
    session = _base.http()
    selected: dict[str, str] = {}
    for model in ("HRDPS", "RDPS"):
        base = _select_deterministic_base(session, model)
        if base:
            _base.MODEL_CONFIG[model]["base"] = base
            selected[model] = base
    reps_base = _select_reps_base(session)
    if reps_base:
        _base.REPS_BASE = reps_base
        selected["REPS"] = reps_base
    print({"rollover_selected_bases": selected})
    _base.main()


if __name__ == "__main__":
    main()
