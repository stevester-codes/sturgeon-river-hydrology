#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_impacts_v2 import (
    BASE_CAL,
    CAL_V2,
    FEATURES,
    analog_predict,
    classify,
    current_antecedent_rain,
    finite,
    rain_free_stage,
    routing_info,
    training_frame,
)

ROOT = Path("sturgeon_pipeline_output")
GEPS = ROOT / "spatial" / "geps_qpf_by_subarea.csv"
MEDIUM_META = ROOT / "spatial" / "medium_range_qpf.json"
SHORT_FORECAST = ROOT / "forecast_v2" / "forecast_impacts_v2.json"
OUT = ROOT / "forecast_v2"
QUANTILES = {"dry": "p10_mm", "central": "p50_mm", "wet": "p90_mm"}
TARGET_HORIZONS = [72, 120, 168, 240, 384]
SUBAREAS = {
    "basin": "basin_to_05EA002",
    "lower": "lower_incremental_05EA005_to_05EA002",
    "upper": "upper_lake_chain_isle_lac_ste_anne",
}


def row_at(frame: pd.DataFrame, horizon: int, subarea: str):
    rows = frame[(frame.horizon_h == horizon) & (frame.subarea == subarea)]
    return rows.iloc[0] if len(rows) else None


def qvalue(frame: pd.DataFrame, horizon: int, subarea: str, column: str) -> float:
    row = row_at(frame, horizon, subarea)
    return finite(row.get(column), 0.0) if row is not None else 0.0


def probability_snapshot(frame: pd.DataFrame, horizon: int, subarea: str) -> dict:
    row = row_at(frame, horizon, subarea)
    if row is None:
        return {}
    return {
        "horizon_h": horizon,
        "p10_mm": finite(row.get("p10_mm"), 0.0),
        "p50_mm": finite(row.get("p50_mm"), 0.0),
        "p90_mm": finite(row.get("p90_mm"), 0.0),
        "prob_ge_5mm": finite(row.get("prob_ge_5mm"), 0.0),
        "prob_ge_10mm": finite(row.get("prob_ge_10mm"), 0.0),
        "prob_ge_20mm": finite(row.get("prob_ge_20mm"), 0.0),
        "prob_ge_30mm": finite(row.get("prob_ge_30mm"), 0.0),
        "prob_ge_50mm": finite(row.get("prob_ge_50mm"), 0.0),
    }


def short_range_base(short: dict) -> dict:
    scenarios = [
        item for item in short.get("deterministic_scenarios", [])
        if item.get("model") == "HRDPS" and int(item.get("horizon_h", 0)) == 48
    ]
    if not scenarios:
        rain_free = finite(short.get("rain_free_days_to_1_70"))
        return {
            "central_days": rain_free,
            "range_days": [rain_free, rain_free],
            "stage_departure_range_m": [0.0, 0.0],
            "source": "rain-free fallback; HRDPS 48 h scenario unavailable",
        }
    scenario = scenarios[0]
    central = finite(scenario.get("projected_1_70_days_central"), finite(short.get("rain_free_days_to_1_70")))
    range_days = scenario.get("projected_1_70_days_range", [central, central])
    return {
        "central_days": central,
        "range_days": [finite(range_days[0], central), finite(range_days[1], central)],
        "stage_departure_range_m": scenario.get("stage_departure_range_m", [0.0, 0.0]),
        "source": "validated HRDPS 0-48 h impact scenario",
    }


def effective_duration(frame: pd.DataFrame, column: str, horizon: int) -> int:
    available = sorted(
        int(value) for value in pd.to_numeric(frame.horizon_h, errors="coerce").dropna().unique()
        if 48 < int(value) <= horizon
    )
    previous = qvalue(frame, 48, SUBAREAS["basin"], column)
    wet_intervals = 0
    for current_h in available:
        current = qvalue(frame, current_h, SUBAREAS["basin"], column)
        if current - previous >= 0.5:
            wet_intervals += 1
        previous = current
    return max(0, wet_intervals * 24)


def scenario_impact(
    name: str,
    column: str,
    horizon: int,
    geps: pd.DataFrame,
    training: pd.DataFrame,
    stage_now: float,
    recession_model: dict,
    antecedent: float,
    short_base: dict,
    route: dict,
) -> dict:
    basin_48 = qvalue(geps, 48, SUBAREAS["basin"], column)
    lower_48 = qvalue(geps, 48, SUBAREAS["lower"], column)
    upper_48 = qvalue(geps, 48, SUBAREAS["upper"], column)
    basin_total = qvalue(geps, horizon, SUBAREAS["basin"], column)
    lower_total = qvalue(geps, horizon, SUBAREAS["lower"], column)
    upper_total = qvalue(geps, horizon, SUBAREAS["upper"], column)
    basin_extra = max(0.0, basin_total - basin_48)
    lower_extra = max(0.0, lower_total - lower_48)
    upper_extra = max(0.0, upper_total - upper_48)
    duration_h = effective_duration(geps, column, horizon)
    stage_at_48 = rain_free_stage(stage_now, recession_model, 48)
    storm_type = classify(
        basin_extra,
        lower_extra,
        upper_extra,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    detail = {
        "scenario": name,
        "quantile": column,
        "horizon_h": horizon,
        "cumulative_basin_rain_mm": basin_total,
        "additional_basin_rain_after_48h_mm": basin_extra,
        "additional_lower_rain_after_48h_mm": lower_extra,
        "additional_upper_rain_after_48h_mm": upper_extra,
        "effective_wet_duration_after_48h_h": duration_h,
        "storm_type": storm_type,
        "short_range_base": short_base,
    }
    if basin_extra < 0.5 or duration_h == 0 or len(training) < 2:
        detail.update({
            "additional_days_lost_central": 0.0,
            "additional_days_lost_range": [0.0, 0.0 if basin_extra < 0.5 else 1.5],
            "projected_1_70_days_central": short_base["central_days"],
            "projected_1_70_days_range": short_base["range_days"],
            "confidence": "very low",
            "impact": "no material post-48 h ensemble rainfall" if basin_extra < 0.5 else "insufficient uncensored analogues",
        })
        return detail
    features = np.asarray([
        basin_extra,
        lower_extra / basin_extra if basin_extra > 0 else 1.0,
        upper_extra / basin_extra if basin_extra > 0 else 1.0,
        duration_h,
        antecedent + basin_48,
        0.0,
        stage_at_48,
    ], dtype=float)
    same_type = training[training.storm_type == storm_type]
    pool = same_type if len(same_type) >= 2 else training
    prediction = analog_predict(pool, features, k=min(3, len(pool)))
    spread = max(1.0, float(pool.days_lost.std(ddof=0)) if len(pool) > 1 else 1.5)
    days_low = max(0.0, min(prediction["days_lost_min"], prediction["days_lost"] - spread))
    days_high = max(prediction["days_lost_max"], prediction["days_lost"] + spread)
    projected_central = short_base["central_days"] + prediction["days_lost"]
    projected_range = [short_base["range_days"][0] + days_low, short_base["range_days"][1] + days_high]
    peak_effect_h = 48 + prediction["lag_h"]
    starkey_central_h = peak_effect_h + finite(route.get("central_h"), 0.0)
    route_range = route.get("range_h", [0.0, 30.0])
    detail.update({
        "impact": "post-48 h GEPS rainfall translated using uncensored event analogues",
        "feature_vector": dict(zip(FEATURES, [float(value) for value in features])),
        "analog_prediction": prediction,
        "additional_days_lost_central": prediction["days_lost"],
        "additional_days_lost_range": [days_low, days_high],
        "projected_1_70_days_central": projected_central,
        "projected_1_70_days_range": projected_range,
        "starkey_additional_max_effect_hours_from_now_central": starkey_central_h,
        "starkey_additional_max_effect_hours_from_now_range": [
            48 + max(0.0, prediction["lag_h_min"]) + finite(route_range[0], 0.0),
            48 + prediction["lag_h_max"] + finite(route_range[1], 30.0),
        ],
        "confidence": "low",
    })
    return detail


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not GEPS.exists() or not GEPS.stat().st_size:
        raise RuntimeError("Validated GEPS subarea output is unavailable")
    metadata = json.loads(MEDIUM_META.read_text()) if MEDIUM_META.exists() else {}
    validation = metadata.get("geps", {}).get("validation", {})
    if not validation.get("passed"):
        raise RuntimeError("GEPS validation did not pass")
    geps = pd.read_csv(GEPS)
    geps["horizon_h"] = pd.to_numeric(geps.horizon_h, errors="coerce")
    base = json.loads(BASE_CAL.read_text())
    short = json.loads(SHORT_FORECAST.read_text())
    events = pd.read_csv(CAL_V2)
    _, training = training_frame(events)
    stage_now = finite(base.get("latest_stage_m"))
    recession_model = base.get("master_recession", {})
    antecedent = current_antecedent_rain(168)
    route = routing_info()
    short_base = short_range_base(short)
    available = set(pd.to_numeric(geps.horizon_h, errors="coerce").dropna().astype(int))
    horizon = 240 if 240 in available else max(value for value in TARGET_HORIZONS if value in available)
    paths = {
        name: scenario_impact(name, column, horizon, geps, training, stage_now, recession_model, antecedent, short_base, route)
        for name, column in QUANTILES.items()
    }
    now = datetime.now(timezone.utc)
    for path in paths.values():
        central = finite(path.get("projected_1_70_days_central"))
        bounds = path.get("projected_1_70_days_range", [central, central])
        path["projected_1_70_date_central_utc"] = (now + timedelta(days=central)).date().isoformat() if np.isfinite(central) else None
        path["projected_1_70_date_range_utc"] = [
            (now + timedelta(days=finite(bounds[0], central))).date().isoformat(),
            (now + timedelta(days=finite(bounds[1], central))).date().isoformat(),
        ]
    snapshots = {
        str(h): {
            key: probability_snapshot(geps, h, subarea)
            for key, subarea in SUBAREAS.items()
        }
        for h in TARGET_HORIZONS if h in available
    }
    result = {
        "generated_utc": now.isoformat(),
        "geps_run_time_utc": metadata.get("geps", {}).get("run_time_utc"),
        "geps_validation": validation,
        "planning_horizon_h": horizon,
        "latest_stage_m": stage_now,
        "antecedent_168h_basin_rain_mm": antecedent,
        "short_range_base": short_base,
        "paths_to_1_70": paths,
        "ensemble_probability_snapshots": snapshots,
        "training": {
            "event_constraints": int(len(events)),
            "uncensored_peak_training_events": int(len(training)),
            "complete_recovery_events": 0,
        },
        "limitations": [
            "GEPS is coarse and is used for scenario ranges, not exact storm timing or centimetre-level stage prediction.",
            "Only two uncensored peak-training events are available; ensemble rainfall-to-delay translation is low confidence.",
            "Post-48 h rainfall is estimated by subtracting each GEPS quantile at 48 h from the same quantile at the planning horizon; this is a scenario approximation, not member-wise hydrologic routing.",
            "Exact Starkey stage remains unavailable because no common site gauge datum exists.",
        ],
    }
    (OUT / "ensemble_paths_to_1_70.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(paths.values()).to_json(OUT / "ensemble_paths_to_1_70_table.json", orient="records", indent=2)
    print(json.dumps({"planning_horizon_h": horizon, "paths": list(paths), "training_events": len(training)}, indent=2))


if __name__ == "__main__":
    main()
