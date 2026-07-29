#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
CAL = ROOT / "calibration" / "calibration.json"
OBS = ROOT / "spatial" / "observed_event_grid_coverage.csv"
QPF = ROOT / "spatial" / "deterministic_qpf_by_subarea.csv"
OUT = ROOT / "forecast"
FEATURES = ["basin_mm", "lower_mm", "duration_h", "lower_ratio", "spread_ratio"]
TARGETS = ["departure_m", "days_lost", "lag_h"]


def safe(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def expected_rate(model: dict, stage: float) -> float:
    a = safe(model.get("intercept_m_per_day"), -0.01)
    b = safe(model.get("stage_coefficient_per_day"), -0.01)
    return min(-0.001, a + b * stage)


def rain_free_stage(stage: float, model: dict, hours: float) -> float:
    h = float(stage)
    for _ in range(max(0, int(round(hours)))):
        h += expected_rate(model, h) / 24.0
    return h


def event_training(cal: dict) -> tuple[pd.DataFrame, dict]:
    events = pd.DataFrame(cal.get("events", []))
    if events.empty:
        raise RuntimeError("No calibration events")
    spatial_used = False
    spatial = pd.DataFrame()
    if OBS.exists() and OBS.stat().st_size:
        spatial = pd.read_csv(OBS)
        spatial_used = not spatial.empty
    rows = []
    for _, e in events.iterrows():
        eid = int(e.event_id)
        # Event 2 begins at the left edge of the July dataset and is affected by
        # antecedent rainfall that is not fully represented; retain for reporting,
        # exclude from quantitative fitting.
        censored = eid == 2
        if spatial_used:
            s = spatial[spatial.event_id == eid]
            def area_value(name, col="mean_mm"):
                z = s[s.subarea == name]
                return safe(z.iloc[0][col]) if len(z) else np.nan
            basin = area_value("basin_to_05EA002")
            lower = area_value("lower_incremental_05EA005_to_05EA002")
            vals = [safe(v) for v in s.mean_mm.tolist() if np.isfinite(safe(v))]
            spread = min(vals) / max(vals) if vals and max(vals) > 0 else 0.0
        else:
            basin = safe(e.get("rain_05EA002_mm"))
            lower = safe(e.get("rain_incremental_05EA005_to_05EA002_mm"))
            vals = [safe(e.get(c)) for c in ["rain_05EA005_mm", "rain_05EA010_mm", "rain_05EA011_mm", "rain_05EA012_mm"]]
            vals = [v for v in vals if np.isfinite(v)]
            spread = min(vals) / max(vals) if vals and max(vals) > 0 else 0.0
        lower_ratio = lower / basin if np.isfinite(lower) and np.isfinite(basin) and basin > 0 else 1.0
        row = {
            "event_id": eid,
            "censored": censored,
            "basin_mm": basin,
            "lower_mm": lower,
            "duration_h": safe(e.get("rain_duration_h")),
            "lower_ratio": lower_ratio,
            "spread_ratio": spread,
            "departure_m": safe(e.get("baseline_departure_peak_m")),
            "days_lost": safe(e.get("estimated_recession_days_lost")),
            "lag_h": safe(e.get("lag_to_departure_peak_h")),
            "actual_rise_m": safe(e.get("actual_stage_rise_m")),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    valid = (~df.censored) & np.isfinite(df[FEATURES + TARGETS]).all(axis=1)
    train = df[valid].copy()
    meta = {"spatial_grid_features_used": spatial_used, "events_total": len(df), "events_fitted": len(train), "excluded_event_ids": df.loc[~valid, "event_id"].astype(int).tolist()}
    return train, meta


def standardizer(train: pd.DataFrame):
    mu = train[FEATURES].mean().to_numpy(float)
    sd = train[FEATURES].std(ddof=0).replace(0, 1).fillna(1).to_numpy(float)
    return mu, sd


def analog_predict(train: pd.DataFrame, x: np.ndarray, exclude_event=None, k=3) -> dict:
    pool = train if exclude_event is None else train[train.event_id != exclude_event]
    mu, sd = standardizer(pool)
    X = pool[FEATURES].to_numpy(float)
    d = np.sqrt(np.mean(((X - x) / sd) ** 2, axis=1))
    order = np.argsort(d)[: min(k, len(pool))]
    nearest = pool.iloc[order].copy()
    dist = d[order]
    w = 1.0 / np.maximum(dist, 0.15) ** 2
    w = w / w.sum()
    pred = {}
    for target in TARGETS:
        y = nearest[target].to_numpy(float)
        pred[target] = float(np.sum(w * y))
        pred[target + "_analog_min"] = float(np.min(y))
        pred[target + "_analog_max"] = float(np.max(y))
    pred["analog_event_ids"] = nearest.event_id.astype(int).tolist()
    pred["analog_distances"] = [float(v) for v in dist]
    pred["nearest_distance"] = float(dist[0])
    return pred


def cross_validate(train: pd.DataFrame) -> dict:
    residuals = {t: [] for t in TARGETS}
    rows = []
    for _, r in train.iterrows():
        x = r[FEATURES].to_numpy(float)
        pred = analog_predict(train, x, exclude_event=int(r.event_id), k=3)
        row = {"event_id": int(r.event_id)}
        for t in TARGETS:
            err = pred[t] - float(r[t])
            residuals[t].append(err)
            row[t + "_observed"] = float(r[t])
            row[t + "_predicted"] = pred[t]
            row[t + "_error"] = err
        rows.append(row)
    diag = {"leave_one_event_out": rows}
    for t, vals in residuals.items():
        a = np.asarray(vals, float)
        diag[t] = {"mae": float(np.mean(np.abs(a))), "rmse": float(np.sqrt(np.mean(a * a))), "bias": float(np.mean(a)), "n": int(len(a))}
    return diag


def qpf_features(qpf: pd.DataFrame, model: str, horizon: int) -> tuple[np.ndarray, dict] | None:
    q = qpf[(qpf.model == model) & (qpf.forecast_hour_end <= horizon)].copy()
    if q.empty:
        return None
    piv = q.groupby("subarea").agg(total_mm=("mean_mm", "sum"), wet_intervals=("mean_mm", lambda x: int((x > 0.5).sum()))).reset_index()
    def val(name):
        z = piv[piv.subarea == name]
        return safe(z.total_mm.iloc[0]) if len(z) else np.nan
    basin = val("basin_to_05EA002")
    lower = val("lower_incremental_05EA005_to_05EA002")
    area_names = ["upper_lake_chain_isle_lac_ste_anne", "lac_ste_anne_to_villeneuve_mainstem", "atim_creek_big_lake_tributary", "direct_big_lake_and_local_to_05EA002"]
    vals = [val(n) for n in area_names]
    vals = [v for v in vals if np.isfinite(v)]
    spread = min(vals) / max(vals) if vals and max(vals) > 0 else 0.0
    wet = q[q.subarea == "basin_to_05EA002"].sort_values("forecast_hour_end")
    wet = wet[wet.mean_mm > 0.5]
    if len(wet):
        first_wet = int(wet.forecast_hour_end.min() - wet.interval_hours.loc[wet.forecast_hour_end.idxmin()])
        last_wet = int(wet.forecast_hour_end.max())
        duration = max(6, last_wet - first_wet)
    else:
        first_wet, last_wet, duration = horizon, horizon, 0
    lower_ratio = lower / basin if np.isfinite(lower) and np.isfinite(basin) and basin > 0 else 1.0
    x = np.array([basin, lower, duration, lower_ratio, spread], float)
    detail = {"model": model, "horizon_h": horizon, "basin_mm": basin, "lower_incremental_mm": lower, "duration_h": duration, "first_wet_forecast_hour": first_wet, "last_wet_forecast_hour": last_wet, "lower_ratio": lower_ratio, "spread_ratio": spread, "subarea_totals_mm": {r.subarea: float(r.total_mm) for _, r in piv.iterrows()}}
    return x, detail


def confidence_label(nearest_distance: float, cv: dict, qpf_model: str) -> str:
    if nearest_distance <= 0.6 and cv["departure_m"]["rmse"] <= 0.08:
        return "moderate"
    if nearest_distance <= 1.2:
        return "low-moderate"
    return "low"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CAL.read_text())
    train, training_meta = event_training(cal)
    if len(train) < 4:
        raise RuntimeError(f"Only {len(train)} usable events; at least four required")
    cv = cross_validate(train)
    train.to_csv(OUT / "training_events.csv", index=False)
    qpf = pd.read_csv(QPF) if QPF.exists() and QPF.stat().st_size else pd.DataFrame()
    stage_now = safe(cal.get("latest_stage_m"))
    model = cal.get("master_recession", {})
    rain_free_days = safe(cal.get("rain_free_projection_to_1_70", {}).get("days"))
    scenarios = []
    if not qpf.empty:
        for qpf_model in sorted(qpf.model.unique()):
            max_h = int(qpf[qpf.model == qpf_model].forecast_hour_end.max())
            horizons = [h for h in [24, 48, 72, 84] if h <= max_h]
            for horizon in horizons:
                got = qpf_features(qpf, qpf_model, horizon)
                if got is None:
                    continue
                x, detail = got
                if not np.isfinite(x).all() or detail["basin_mm"] < 0.5:
                    detail.update({"impact": "negligible or no material QPF", "projected_1_70_days": rain_free_days})
                    scenarios.append(detail); continue
                pred = analog_predict(train, x, k=3)
                dep_rmse = cv["departure_m"]["rmse"]
                days_rmse = cv["days_lost"]["rmse"]
                lag_rmse = cv["lag_h"]["rmse"]
                departure_low = max(0.0, min(pred["departure_m_analog_min"], pred["departure_m"] - dep_rmse))
                departure_high = max(pred["departure_m_analog_max"], pred["departure_m"] + dep_rmse)
                days_low = max(0.0, min(pred["days_lost_analog_min"], pred["days_lost"] - days_rmse))
                days_high = max(pred["days_lost_analog_max"], pred["days_lost"] + days_rmse)
                lag_low = max(0.0, pred["lag_h"] - lag_rmse)
                lag_high = pred["lag_h"] + lag_rmse
                event_start_h = detail["first_wet_forecast_hour"]
                peak_h = event_start_h + pred["lag_h"]
                base_at_peak = rain_free_stage(stage_now, model, peak_h)
                peak_stage_central = base_at_peak + pred["departure_m"]
                peak_stage_range = [max(base_at_peak, base_at_peak + departure_low), base_at_peak + departure_high]
                detail.update({
                    "analog_prediction": pred,
                    "stage_departure_range_m": [departure_low, departure_high],
                    "estimated_days_lost_range": [days_low, days_high],
                    "lag_to_max_effect_range_h_from_rain_start": [lag_low, lag_high],
                    "estimated_peak_stage_m": peak_stage_central,
                    "estimated_peak_stage_range_m": peak_stage_range,
                    "projected_1_70_days_central": rain_free_days + pred["days_lost"],
                    "projected_1_70_days_range": [rain_free_days + days_low, rain_free_days + days_high],
                    "confidence": confidence_label(pred["nearest_distance"], cv, qpf_model),
                    "threshold_flags": {"rise_or_departure_ge_0_05": departure_high >= 0.05, "delay_ge_2_days": days_high >= 2.0, "credible_2_5_path": peak_stage_range[1] >= 2.5, "credible_3_0_path": peak_stage_range[1] >= 3.0},
                })
                scenarios.append(detail)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_stage_m": stage_now,
        "rain_free_days_to_1_70": rain_free_days,
        "training": training_meta,
        "features": FEATURES,
        "targets": TARGETS,
        "cross_validation": cv,
        "scenarios": scenarios,
        "interpretation": [
            "Predicted departure is stage above the fitted rain-free recession path, not necessarily raw rise from the pre-storm level.",
            "Ranges combine nearest observed-event outcomes with leave-one-event-out residual error.",
            "Only July 2026 events are available, so forecasts outside the observed rainfall/location envelope remain low confidence.",
            "Starkey Road timing remains operationally anchored to 05EA002 until a separate downstream reach calibration is available.",
        ],
    }
    (OUT / "forecast_impacts.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(scenarios).to_json(OUT / "forecast_scenarios.json", orient="records", indent=2)
    print(json.dumps({"training_events": len(train), "scenarios": len(scenarios), "cv_departure_rmse_m": cv["departure_m"]["rmse"], "cv_days_lost_rmse": cv["days_lost"]["rmse"]}, indent=2))


if __name__ == "__main__":
    main()
