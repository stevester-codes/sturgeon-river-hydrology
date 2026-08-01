#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# qpf_forecast_v2.py: search both ECCC day buckets and preserve seeded QPF
# when an upstream listing is temporarily unavailable.
# ---------------------------------------------------------------------------
path = Path("qpf_forecast_v2.py")
text = path.read_text()
text = replace_once(
    text,
    'REPS_BASE = "https://dd.weather.gc.ca/today/ensemble/reps/10km/grib2"\n',
    'REPS_BASE = "https://dd.weather.gc.ca/today/ensemble/reps/10km/grib2"\nDAY_BUCKETS = ("today", "yesterday")\n\n\ndef source_bases(base: str) -> list[str]:\n    """Search the current and previous UTC-day directories.\n\n    ECCC rotates completed late-day cycles from /today/ to /yesterday/ at\n    00 UTC, before a new-day cycle is necessarily available.\n    """\n    return [base.replace("/today/", f"/{bucket}/", 1) for bucket in DAY_BUCKETS]\n',
    "qpf day buckets",
)
old = '''def latest_deterministic_run(\n    session: requests.Session, model: str\n) -> tuple[str, str, datetime] | None:\n    config = MODEL_CONFIG[model]\n    found: list[tuple[datetime, str, str]] = []\n    for hour in ["00", "06", "12", "18"]:\n        url = f"{config['base']}/{hour}/006/"\n        try:\n            candidate = deterministic_candidate(get_links(session, url), model)\n        except Exception:\n            continue\n        if candidate:\n            run_time = parse_run_time(candidate)\n            if run_time:\n                found.append((run_time, hour, candidate))\n    if not found:\n        return None\n    run_time, hour, candidate = max(found, key=lambda item: item[0])\n    return hour, candidate, run_time\n'''
new = '''def latest_deterministic_run(\n    session: requests.Session, model: str\n) -> tuple[str, str, str, datetime] | None:\n    config = MODEL_CONFIG[model]\n    found: list[tuple[datetime, str, str, str]] = []\n    for run_base in source_bases(config["base"]):\n        for hour in ["00", "06", "12", "18"]:\n            url = f"{run_base}/{hour}/006/"\n            try:\n                candidate = deterministic_candidate(get_links(session, url), model)\n            except Exception:\n                continue\n            if candidate:\n                run_time = parse_run_time(candidate)\n                if run_time:\n                    found.append((run_time, run_base, hour, candidate))\n    if not found:\n        return None\n    run_time, run_base, hour, candidate = max(found, key=lambda item: item[0])\n    return run_base, hour, candidate, run_time\n'''
text = replace_once(text, old, new, "latest deterministic")
text = replace_once(
    text,
    '    run_hour, probe_file, run_time = discovered\n',
    '    run_base, run_hour, probe_file, run_time = discovered\n',
    "deterministic unpack",
)
text = replace_once(
    text,
    '        directory = f"{config[\'base\']}/{run_hour}/{forecast_hour:03d}/"\n',
    '        directory = f"{run_base}/{run_hour}/{forecast_hour:03d}/"\n',
    "deterministic directory",
)
text = replace_once(
    text,
    '        "run_hour_utc": run_hour,\n        "probe_file": probe_file,\n',
    '        "run_hour_utc": run_hour,\n        "source_base": run_base,\n        "probe_file": probe_file,\n',
    "deterministic metadata",
)
old = '''def latest_reps_run(\n    session: requests.Session,\n) -> tuple[str, datetime] | None:\n    found: list[tuple[datetime, str]] = []\n    for hour in ["00", "06", "12", "18"]:\n        directory = f"{REPS_BASE}/{hour}/024/"\n        try:\n            candidate = reps_candidate(get_links(session, directory), 24, False)\n        except Exception:\n            continue\n        if candidate:\n            run_time = parse_run_time(candidate)\n            if run_time:\n                found.append((run_time, hour))\n    if not found:\n        return None\n    return max(found, key=lambda item: item[0])[1], max(\n        found, key=lambda item: item[0]\n    )[0]\n'''
new = '''def latest_reps_run(\n    session: requests.Session,\n) -> tuple[str, str, datetime] | None:\n    found: list[tuple[datetime, str, str]] = []\n    for run_base in source_bases(REPS_BASE):\n        for hour in ["00", "06", "12", "18"]:\n            directory = f"{run_base}/{hour}/024/"\n            try:\n                candidate = reps_candidate(get_links(session, directory), 24, False)\n            except Exception:\n                continue\n            if candidate:\n                run_time = parse_run_time(candidate)\n                if run_time:\n                    found.append((run_time, run_base, hour))\n    if not found:\n        return None\n    run_time, run_base, hour = max(found, key=lambda item: item[0])\n    return run_base, hour, run_time\n'''
text = replace_once(text, old, new, "latest reps")
text = replace_once(
    text,
    '    run_hour, run_time = discovered\n',
    '    run_base, run_hour, run_time = discovered\n',
    "reps unpack",
)
text = replace_once(
    text,
    '        directory = f"{REPS_BASE}/{run_hour}/{horizon:03d}/"\n',
    '        directory = f"{run_base}/{run_hour}/{horizon:03d}/"\n',
    "reps directory",
)
text = replace_once(
    text,
    '        "run_hour_utc": run_hour,\n        "files": file_metadata,\n',
    '        "run_hour_utc": run_hour,\n        "source_base": run_base,\n        "files": file_metadata,\n',
    "reps metadata",
)
old = '''    write_csv(\n        deterministic_rows, SPATIAL / "deterministic_qpf_by_subarea.csv"\n    )\n    reps_rows, reps_metadata, reps_warnings = process_reps(session, subareas)\n    warnings.extend(reps_warnings)\n    write_csv(reps_rows, SPATIAL / "ensemble_qpf_by_subarea.csv")\n'''
new = '''    deterministic_path = SPATIAL / "deterministic_qpf_by_subarea.csv"\n    preserved_previous_deterministic = False\n    if deterministic_rows:\n        write_csv(deterministic_rows, deterministic_path)\n    elif deterministic_path.exists() and deterministic_path.stat().st_size > 0:\n        preserved_previous_deterministic = True\n        warnings.append(\n            "No new deterministic QPF was retrieved; preserved the seeded previous "\n            "QPF so the existing 12-hour HRDPS carry-forward policy can decide validity."\n        )\n    else:\n        write_csv([], deterministic_path)\n\n    reps_rows, reps_metadata, reps_warnings = process_reps(session, subareas)\n    warnings.extend(reps_warnings)\n    reps_path = SPATIAL / "ensemble_qpf_by_subarea.csv"\n    preserved_previous_reps = False\n    if reps_rows:\n        write_csv(reps_rows, reps_path)\n    elif reps_path.exists() and reps_path.stat().st_size > 0:\n        preserved_previous_reps = True\n        warnings.append("No new REPS file was retrieved; preserved the seeded previous file.")\n    else:\n        write_csv([], reps_path)\n'''
text = replace_once(text, old, new, "qpf preserve previous")
text = replace_once(
    text,
    '        "warning_count": len(warnings),\n        "validation": {\n',
    '        "warning_count": len(warnings),\n        "preserved_previous_deterministic": preserved_previous_deterministic,\n        "preserved_previous_reps": preserved_previous_reps,\n        "validation": {\n',
    "qpf fallback metadata",
)
path.write_text(text)


# ---------------------------------------------------------------------------
# medium_range_qpf.py: use the same today/yesterday discovery for GDPS/GEPS.
# ---------------------------------------------------------------------------
path = Path("medium_range_qpf.py")
text = path.read_text()
text = replace_once(
    text,
    'from qpf_forecast_v2 import download, get_links, http, parse_run_time, summarize\n',
    'from qpf_forecast_v2 import download, get_links, http, parse_run_time, source_bases, summarize\n',
    "medium import",
)
old = '''def latest_gdps_run(session) -> tuple[str, datetime, str] | None:\n    found: list[tuple[datetime, str, str]] = []\n    for hour in ["00", "12"]:\n        directory = f"{GDPS_BASE}/{hour}/006/"\n        try:\n            candidate = gdps_candidate(get_links(session, directory))\n        except Exception:\n            continue\n        if candidate:\n            run_time = parse_run_time(candidate)\n            if run_time:\n                found.append((run_time, hour, candidate))\n    if not found:\n        return None\n    run_time, hour, candidate = max(found, key=lambda item: item[0])\n    return hour, run_time, candidate\n'''
new = '''def latest_gdps_run(session) -> tuple[str, str, datetime, str] | None:\n    found: list[tuple[datetime, str, str, str]] = []\n    for run_base in source_bases(GDPS_BASE):\n        for hour in ["00", "12"]:\n            directory = f"{run_base}/{hour}/006/"\n            try:\n                candidate = gdps_candidate(get_links(session, directory))\n            except Exception:\n                continue\n            if candidate:\n                run_time = parse_run_time(candidate)\n                if run_time:\n                    found.append((run_time, run_base, hour, candidate))\n    if not found:\n        return None\n    run_time, run_base, hour, candidate = max(found, key=lambda item: item[0])\n    return run_base, hour, run_time, candidate\n'''
text = replace_once(text, old, new, "latest gdps")
old = '''def latest_geps_run(session) -> tuple[str, datetime, str] | None:\n    """Select the newest GEPS cycle that is complete at every required horizon.\n\n    A newer PT024 file may appear hours before PT384. During that publication\n    window, continue using the previous complete cycle rather than failing the\n    live forecast or delaying a gauge-driven update.\n    """\n    found: list[tuple[datetime, str, str]] = []\n    for hour in ["00", "12"]:\n        run_times: list[datetime] = []\n        probe: str | None = None\n        complete = True\n        for horizon in GEPS_HORIZONS:\n            directory = f"{GEPS_BASE}/{hour}/{horizon:03d}/"\n            try:\n                candidate = geps_candidate(get_links(session, directory))\n            except Exception:\n                complete = False\n                break\n            if not candidate:\n                complete = False\n                break\n            run_time = geps_run_time(candidate)\n            if not run_time:\n                complete = False\n                break\n            run_times.append(run_time)\n            if horizon == GEPS_HORIZONS[0]:\n                probe = candidate\n        if complete and probe and len(run_times) == len(GEPS_HORIZONS):\n            if all(value == run_times[0] for value in run_times):\n                found.append((run_times[0], hour, probe))\n    if not found:\n        return None\n    run_time, hour, candidate = max(found, key=lambda item: item[0])\n    return hour, run_time, candidate\n'''
new = '''def latest_geps_run(session) -> tuple[str, str, datetime, str] | None:\n    """Select the newest GEPS cycle complete at every required horizon.\n\n    Search both UTC-day buckets so the previous complete cycle remains visible\n    during the 00 UTC directory rollover and while a newer cycle is publishing.\n    """\n    found: list[tuple[datetime, str, str, str]] = []\n    for run_base in source_bases(GEPS_BASE):\n        for hour in ["00", "12"]:\n            run_times: list[datetime] = []\n            probe: str | None = None\n            complete = True\n            for horizon in GEPS_HORIZONS:\n                directory = f"{run_base}/{hour}/{horizon:03d}/"\n                try:\n                    candidate = geps_candidate(get_links(session, directory))\n                except Exception:\n                    complete = False\n                    break\n                if not candidate:\n                    complete = False\n                    break\n                run_time = geps_run_time(candidate)\n                if not run_time:\n                    complete = False\n                    break\n                run_times.append(run_time)\n                if horizon == GEPS_HORIZONS[0]:\n                    probe = candidate\n            if complete and probe and len(run_times) == len(GEPS_HORIZONS):\n                if all(value == run_times[0] for value in run_times):\n                    found.append((run_times[0], run_base, hour, probe))\n    if not found:\n        return None\n    run_time, run_base, hour, candidate = max(found, key=lambda item: item[0])\n    return run_base, hour, run_time, candidate\n'''
text = replace_once(text, old, new, "latest geps")
text = replace_once(
    text,
    '    run_hour, run_time, probe = discovered\n',
    '    run_base, run_hour, run_time, probe = discovered\n',
    "gdps unpack",
)
text = replace_once(
    text,
    '        directory = f"{GDPS_BASE}/{run_hour}/{forecast_hour:03d}/"\n',
    '        directory = f"{run_base}/{run_hour}/{forecast_hour:03d}/"\n',
    "gdps directory",
)
text = replace_once(
    text,
    '        "run_hour_utc": run_hour,\n        "probe_file": probe,\n',
    '        "run_hour_utc": run_hour,\n        "source_base": run_base,\n        "probe_file": probe,\n',
    "gdps metadata",
)
# The same unpack text occurs a second time for GEPS after the GDPS replacement.
text = replace_once(
    text,
    '    run_hour, run_time, probe = discovered\n',
    '    run_base, run_hour, run_time, probe = discovered\n',
    "geps unpack",
)
text = replace_once(
    text,
    '        directory = f"{GEPS_BASE}/{run_hour}/{horizon:03d}/"\n',
    '        directory = f"{run_base}/{run_hour}/{horizon:03d}/"\n',
    "geps directory",
)
# Add source_base to the GEPS metadata occurrence (the remaining matching block).
text = replace_once(
    text,
    '        "run_hour_utc": run_hour,\n        "probe_file": probe,\n        "horizons_requested_h": GEPS_HORIZONS,\n',
    '        "run_hour_utc": run_hour,\n        "source_base": run_base,\n        "probe_file": probe,\n        "horizons_requested_h": GEPS_HORIZONS,\n',
    "geps metadata",
)
path.write_text(text)


# ---------------------------------------------------------------------------
# operational_update_probe.py: detector must see complete cycles across rollover.
# ---------------------------------------------------------------------------
path = Path("operational_update_probe.py")
text = path.read_text()
text = replace_once(
    text,
    'GEPS_RUN = re.compile(r"_(\\d{10})_P\\d{3}_allmbrs\\.grib2$", re.I)\n',
    'GEPS_RUN = re.compile(r"_(\\d{10})_P\\d{3}_allmbrs\\.grib2$", re.I)\nDAY_BUCKETS = ("today", "yesterday")\n\n\ndef source_bases(base: str) -> list[str]:\n    return [base.replace("/today/", f"/{bucket}/", 1) for bucket in DAY_BUCKETS]\n',
    "probe day buckets",
)
old = '''def latest_complete_hrdps(http: requests.Session) -> datetime | None:\n    complete: list[datetime] = []\n    for hour in ["00", "06", "12", "18"]:\n        try:\n            first = hrdps_candidate(links(http, f"{HRDPS_BASE}/{hour}/006/"))\n            last = hrdps_candidate(links(http, f"{HRDPS_BASE}/{hour}/048/"))\n        except Exception:\n            continue\n        first_time = hrdps_time(first)\n        last_time = hrdps_time(last)\n        if first_time and first_time == last_time:\n            complete.append(first_time)\n    return max(complete) if complete else None\n'''
new = '''def latest_complete_hrdps(http: requests.Session) -> datetime | None:\n    complete: list[datetime] = []\n    for run_base in source_bases(HRDPS_BASE):\n        for hour in ["00", "06", "12", "18"]:\n            try:\n                first = hrdps_candidate(links(http, f"{run_base}/{hour}/006/"))\n                last = hrdps_candidate(links(http, f"{run_base}/{hour}/048/"))\n            except Exception:\n                continue\n            first_time = hrdps_time(first)\n            last_time = hrdps_time(last)\n            if first_time and first_time == last_time:\n                complete.append(first_time)\n    return max(complete) if complete else None\n'''
text = replace_once(text, old, new, "probe hrdps")
old = '''    advertised: list[datetime] = []\n    complete: list[datetime] = []\n    for hour in ["00", "12"]:\n        try:\n            first = geps_candidate(links(http, f"{GEPS_BASE}/{hour}/024/"))\n            last = geps_candidate(links(http, f"{GEPS_BASE}/{hour}/384/"))\n        except Exception:\n            continue\n        first_time = geps_time(first)\n        last_time = geps_time(last)\n        if first_time:\n            advertised.append(first_time)\n        if first_time and first_time == last_time:\n            complete.append(first_time)\n'''
new = '''    advertised: list[datetime] = []\n    complete: list[datetime] = []\n    for run_base in source_bases(GEPS_BASE):\n        for hour in ["00", "12"]:\n            try:\n                first = geps_candidate(links(http, f"{run_base}/{hour}/024/"))\n                last = geps_candidate(links(http, f"{run_base}/{hour}/384/"))\n            except Exception:\n                continue\n            first_time = geps_time(first)\n            last_time = geps_time(last)\n            if first_time:\n                advertised.append(first_time)\n            if first_time and first_time == last_time:\n                complete.append(first_time)\n'''
text = replace_once(text, old, new, "probe geps")
path.write_text(text)

print("ECCC UTC-day rollover patch applied.")
