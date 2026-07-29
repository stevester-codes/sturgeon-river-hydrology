#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
SPATIAL = ROOT / "spatial"
DET = SPATIAL / "deterministic_qpf_by_subarea.csv"
META = SPATIAL / "qpf_v2.json"
OUT = SPATIAL / "qpf_validation.json"
ACCUM_RE = re.compile(r"_APCP-Accum6h_", re.I)


def main() -> None:
    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not DET.exists() or not DET.stat().st_size:
        errors.append("deterministic_qpf_by_subarea.csv is missing or empty")
        payload = {"passed": False, "checks": checks, "errors": errors, "warnings": warnings}
        OUT.write_text(json.dumps(payload, indent=2))
        raise SystemExit(2)
    frame = pd.read_csv(DET)
    required = {
        "model", "forecast_hour_start", "forecast_hour_end", "interval_hours",
        "accumulation_semantics", "subarea", "source_file", "mean_mm"
    }
    missing = sorted(required - set(frame.columns))
    checks.append({"name": "required_columns", "passed": not missing, "missing": missing})
    if missing:
        errors.append(f"Missing required deterministic-QPF columns: {missing}")
    bad_names = frame.loc[~frame.source_file.astype(str).str.contains(ACCUM_RE), "source_file"].unique().tolist()
    checks.append({"name": "six_hour_source_files", "passed": not bad_names, "bad_files": bad_names[:20]})
    if bad_names:
        errors.append("One or more deterministic files are not APCP-Accum6h products")
    bad_intervals = frame.loc[pd.to_numeric(frame.interval_hours, errors="coerce") != 6]
    checks.append({"name": "six_hour_intervals", "passed": bad_intervals.empty, "bad_row_count": int(len(bad_intervals))})
    if not bad_intervals.empty:
        errors.append("One or more deterministic rows do not represent a six-hour interval")
    semantics = set(frame.accumulation_semantics.astype(str).unique())
    semantic_ok = semantics == {"independent_6h_interval"}
    checks.append({"name": "accumulation_semantics", "passed": semantic_ok, "values": sorted(semantics)})
    if not semantic_ok:
        errors.append("Unexpected deterministic accumulation semantics")
    values = pd.to_numeric(frame.mean_mm, errors="coerce")
    finite = values[np.isfinite(values)]
    physical_ok = bool(len(finite)) and bool((finite >= 0).all()) and bool((finite <= 500).all())
    checks.append({"name": "physical_range", "passed": physical_ok, "min_mm": None if not len(finite) else float(finite.min()), "max_mm": None if not len(finite) else float(finite.max())})
    if not physical_ok:
        errors.append("Deterministic QPF contains missing or nonphysical basin means")
    duplicate_keys = ["model", "forecast_hour_end", "subarea"]
    duplicate_count = int(frame.duplicated(duplicate_keys).sum())
    checks.append({"name": "unique_model_hour_subarea", "passed": duplicate_count == 0, "duplicate_count": duplicate_count})
    if duplicate_count:
        errors.append("Duplicate deterministic model/hour/subarea rows detected")
    coverage = {}
    for model, group in frame.groupby("model"):
        hours = sorted(pd.to_numeric(group.forecast_hour_end, errors="coerce").dropna().astype(int).unique().tolist())
        subareas = sorted(group.subarea.astype(str).unique().tolist())
        expected_rows = len(hours) * len(subareas)
        coverage[model] = {"hours": hours, "subarea_count": len(subareas), "rows": int(len(group)), "expected_rows": expected_rows, "complete": int(len(group)) == expected_rows}
        if int(len(group)) != expected_rows:
            warnings.append(f"{model} QPF coverage is incomplete")
    checks.append({"name": "coverage", "passed": all(v["complete"] for v in coverage.values()), "models": coverage})
    metadata = json.loads(META.read_text()) if META.exists() else {}
    payload = {
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "qpf_metadata_generated_utc": metadata.get("generated_utc"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
