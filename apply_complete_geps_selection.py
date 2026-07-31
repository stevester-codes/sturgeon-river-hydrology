#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


def patch_medium_range() -> None:
    path = Path('medium_range_qpf.py')
    text = path.read_text()
    pattern = re.compile(
        r"def latest_geps_run\(session\) -> tuple\[str, datetime, str\] \| None:\n.*?\n\ndef clip_one_band",
        re.S,
    )
    replacement = '''def latest_geps_run(session) -> tuple[str, datetime, str] | None:
    """Select the newest GEPS cycle that is complete at every required horizon.

    A newer PT024 file may appear hours before PT384. During that publication
    window, continue using the previous complete cycle rather than failing the
    live forecast or delaying a gauge-driven update.
    """
    found: list[tuple[datetime, str, str]] = []
    for hour in ["00", "12"]:
        run_times: list[datetime] = []
        probe: str | None = None
        complete = True
        for horizon in GEPS_HORIZONS:
            directory = f"{GEPS_BASE}/{hour}/{horizon:03d}/"
            try:
                candidate = geps_candidate(get_links(session, directory))
            except Exception:
                complete = False
                break
            if not candidate:
                complete = False
                break
            run_time = geps_run_time(candidate)
            if not run_time:
                complete = False
                break
            run_times.append(run_time)
            if horizon == GEPS_HORIZONS[0]:
                probe = candidate
        if complete and probe and len(run_times) == len(GEPS_HORIZONS):
            if all(value == run_times[0] for value in run_times):
                found.append((run_times[0], hour, probe))
    if not found:
        return None
    run_time, hour, candidate = max(found, key=lambda item: item[0])
    return hour, run_time, candidate


def clip_one_band'''
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError('latest_geps_run function block was not patched')
    path.write_text(updated)


def patch_probe() -> None:
    path = Path('operational_update_probe.py')
    text = path.read_text()
    old = '    reasons = [] if publication_block else candidate_reasons\n'
    new = '    # Gauge and HRDPS updates may proceed using the last complete GEPS cycle.\n    reasons = candidate_reasons\n'
    if old not in text:
        raise RuntimeError('probe publication-block assignment was not found')
    text = text.replace(old, new, 1)
    text = text.replace(
        '"It temporarily waits while a newer GEPS cycle is only partly published, then "\n            "dispatches on the first check after PT384 becomes available."',
        '"A partially published newer GEPS cycle is reported but does not block gauge or "\n            "HRDPS updates; the operational retriever uses the previous complete cycle."',
        1,
    )
    path.write_text(text)


def main() -> None:
    patch_medium_range()
    patch_probe()
    print('Complete-cycle GEPS selection and nonblocking detector patch applied.')


if __name__ == '__main__':
    main()
