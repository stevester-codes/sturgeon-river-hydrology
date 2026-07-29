#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("sturgeon_pipeline_output")
BASE = ROOT / "calibration" / "calibration.json"
SUMMARY = ROOT / "summary" / "summary.json"
TARGET_STAGE_M = 1.70
MAX_HOURS = 24 * 60


def finite(value, default=None):
    try:
        number = float(value)
        return number
    except (TypeError, ValueError):
        return default


def recession_rate(model: dict, stage_m: float) -> float:
    intercept = finite(model.get("intercept_m_per_day"), -0.03)
    coefficient = finite(model.get("stage_coefficient_per_day"), -0.007)
    return min(-0.001, intercept + coefficient * stage_m)


def project(stage_m: float, model: dict) -> dict:
    stage = stage_m
    path_daily = []
    hours = 0
    while stage > TARGET_STAGE_M and hours < MAX_HOURS:
        stage += recession_rate(model, stage) / 24.0
        hours += 1
        if hours % 24 == 0:
            path_daily.append(
                {
                    "day": hours // 24,
                    "stage_m": stage,
                    "rate_m_per_day": recession_rate(model, stage),
                }
            )
    return {
        "hours": hours,
        "days": hours / 24.0,
        "target_stage_m": TARGET_STAGE_M,
        "path_daily": path_daily,
    }


def main() -> None:
    if not BASE.exists() or not SUMMARY.exists():
        raise FileNotFoundError("Base calibration or current summary is missing")
    base = json.loads(BASE.read_text())
    summary = json.loads(SUMMARY.read_text())
    target = summary.get("target_05EA002", {})
    stage = finite(target.get("latest"))
    timestamp = target.get("latest_utc")
    if stage is None or timestamp is None:
        raise RuntimeError("Current 05EA002 stage/timestamp is unavailable")
    model = base.get("master_recession", {})
    base["generated_utc"] = datetime.now(timezone.utc).isoformat()
    base["latest_stage_utc"] = timestamp
    base["latest_stage_m"] = stage
    base["rain_free_projection_to_1_70"] = project(stage, model)
    BASE.write_text(json.dumps(base, indent=2))
    print(
        json.dumps(
            {
                "latest_stage_utc": timestamp,
                "latest_stage_m": stage,
                "rain_free_days_to_1_70": base["rain_free_projection_to_1_70"]["days"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
