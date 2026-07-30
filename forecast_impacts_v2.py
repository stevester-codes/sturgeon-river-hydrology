#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("sturgeon_pipeline_output")
BASE_CAL = ROOT / "calibration" / "calibration.json"
CAL_V2 = ROOT / "calibration_v2" / "event_response_v2.csv"
QPF_INTERVAL = ROOT / "spatial" / "deterministic_qpf_by_subarea.csv"
QPF_HORIZON = ROOT / "spatial" / "deterministic_qpf_horizon_by_subarea.csv"
QPF_META = ROOT / "spatial" / "qpf_v2.json"
REPS = ROOT / "spatial" / "ensemble_qpf_by_subarea.csv"
PRECIP = ROOT / "processed" / "watershed_precip_06h.csv"
ROUTING = ROOT / "routing" / "starkey_routing.json"
OUT = ROOT / "forecast_v2"
TIME_RE = re.compile(r"_(\d{10})_000_\d{2}\.dbf$")
FEATURES = [
    "basin_mm",
    "lower_ratio",
    "upper_ratio",
    "duration_h",
    "antecedent_168h_mm",
    "basin_pct_gt_10mm",
    "pre_stage_m",
]
TARGETS = ["departure_m", "days_lost", "lag_h"]


def finite(value, default=np.nan):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def json_default(value):
    """Convert NumPy/Pandas scalar types without hiding unsupported objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def model_rate(model: dict, stage: float) -> float:
    intercept = finite(model.get("intercept_m_per_day"), -0.01)
    coefficient = finite(model.get("stage_coefficient_per_day"), -0.01)
    return min(-0.001, intercept + coefficient * stage)


def rain_free_stage(stage: float, model: dict, hours: float) -> float:
    value = float(stage)
    for _ in range(max(0, int(round(hours)))):
        value += model_rate(model, value) / 24.0
    return value


def current_antecedent_rain(hours: int = 168) -> float:
    if not PRECIP.exists() or not PRECIP.stat().st_size:
        return 0.0
    frame = pd.read_csv(PRECIP)
    if frame.empty:
        return 0.0

    def valid_time(filename: str):
        match = TIME_RE.search(str(filename))
        if not match:
            return pd.NaT
        return pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)

    frame["valid_utc"] = frame["_source_file"].map(valid_time)
    frame["PR_mm"] = pd.to_numeric(frame["PR_mm"], errors="coerce")
    target = frame[(frame.Station == "05EA002") & frame.valid_utc.notna() & frame.PR_mm.notna()]
    if target.empty:
        return 0.0
    end = target.valid_utc.max()
    start = end - pd.Timedelta(hours=hours)
    return float(target.loc[(target.valid_utc > start) & (target.valid_utc <= end), "PR_mm"].sum())


def classify(
    basin: float,
    lower: float,
    upper: float,
    middle: float,
    atim: float,
    carrot: float,
    local: float,
    basin_pct10: float,
    basin_pct5: float,
) -> str:
    if basin_pct10 >= 70 or basin_pct5 >= 90:
        return "widespread_basin"
    if basin > 0 and lower >= 1.35 * basin and upper <= 0.85 * basin:
        return "lower_basin_concentrated"
    if lower > 0 and upper >= 1.35 * lower:
        return "upper_lake_chain_concentrated"
    if max(atim, carrot, local) >= max(2.0, 1.4 * basin):
        dominant = max(
            [(atim, "atim_big_lake"), (carrot, "carrot_creek"), (local, "direct_local")]
        )[1]
        return f"tributary_localized_{dominant}"
    if middle >= max(2.0, 1.25 * upper, 1.25 * lower):
        return "middle_mainstem_concentrated"
    return "mixed_or_weak"


def training_frame(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, event in events.iterrows():
        basin = finite(event.get("basin_mean_mm"), 0.0)
        lower = finite(event.get("lower_mean_mm"), 0.0)
        upper = finite(event.get("upper_mean_mm"), 0.0)
        row = {
            "event_id": int(event.event_id),
            "storm_type": str(event.get("storm_type", "mixed_or_weak")),
            "censored": bool(event.get("response_censored", False)),
            "eligible": bool(event.get("eligible_for_peak_training", False)),
            "basin_mm": basin,
            "lower_ratio": lower / basin if basin > 0 else 1.0,
            "upper_ratio": upper / basin if basin > 0 else 1.0,
            "duration_h": finite(event.get("rain_duration_h"), 0.0),
            "antecedent_168h_mm": finite(event.get("antecedent_168h_basin_rain_mm"), 0.0),
            "basin_pct_gt_10mm": finite(event.get("basin_pct_gt_10mm"), 0.0),
            "pre_stage_m": finite(event.get("pre_stage_m")),
            "departure_m": finite(event.get("departure_peak_m")),
            "days_lost": finite(event.get("estimated_recession_days_lost")),
            "lag_h": finite(event.get("lag_to_departure_peak_h")),
            "raw_stage_rise_m": finite(event.get("raw_stage_rise_m")),
            "response_quality": str(event.get("response_quality", "unknown")),
        }
        rows.append(row)
    all_events = pd.DataFrame(rows)
    valid = all_events.eligible & np.isfinite(all_events[FEATURES + TARGETS]).all(axis=1)
    return all_events, all_events[valid].copy()


def standardize(pool: pd.DataFrame):
    mean = pool[FEATURES].mean().to_numpy(float)
    scale = pool[FEATURES].std(ddof=0).replace(0, 1).fillna(1).to_numpy(float)
    return mean, scale


def analog_predict(pool: pd.DataFrame, features: np.ndarray, k: int = 3) -> dict:
    mean, scale = standardize(pool)
    matrix = pool[FEATURES].to_numpy(float)
    distance = np.sqrt(np.mean(((matrix - features) / scale) ** 2, axis=1))
    order = np.argsort(distance)[: min(k, len(pool))]
    nearest = pool.iloc[order]
    chosen_distance = distance[order]
    weights = 1.0 / np.maximum(chosen_distance, 0.2) ** 2
    weights /= weights.sum()
    result = {
        "analog_event_ids": nearest.event_id.astype(int).tolist(),
        "analog_storm_types": nearest.storm_type.astype(str).tolist(),
        "analog_distances": [float(value) for value in chosen_distance],
        "nearest_distance": float(chosen_distance[0]),
    }
    for target in TARGETS:
        values = nearest[target].to_numpy(float)
        result[target] = float(np.sum(weights * values))
        result[f"{target}_min"] = float(np.min(values))
        result[f"{target}_max"] = float(np.max(values))
    return result


def cross_validate(training: pd.DataFrame) -> dict:
    if len(training) < 3:
        return {target: {"n": len(training), "rmse": None, "mae": None, "bias": None} for target in TARGETS}
    residuals = {target: [] for target in TARGETS}
    rows = []
    for _, event in training.iterrows():
        pool = training[training.event_id != event.event_id]
        prediction = analog_predict(pool, event[FEATURES].to_numpy(float), k=min(3, len(pool)))
        row = {"event_id": int(event.event_id)}
        for target in TARGETS:
            error = prediction[target] - float(event[target])
            residuals[target].append(error)
            row[f"{target}_error"] = float(error)
        rows.append(row)
    result = {"leave_one_event_out": rows}
    for target, errors in residuals.items():
        values = np.asarray(errors, dtype=float)
        result[target] = {
            "n": int(len(values)),
            "rmse": float(np.sqrt(np.mean(values * values))),
            "mae": float(np.mean(np.abs(values))),
            "bias": float(np.mean(values)),
        }
    return result


def horizon_record(frame: pd.DataFrame, model: str, horizon: int, subarea: str):
    rows = frame[(frame.model == model) & (frame.horizon_h == horizon) & (frame.subarea == subarea)]
    return rows.iloc[0] if len(rows) else None


def interval_timing(intervals: pd.DataFrame, model: str, horizon: int) -> dict:
    rows = intervals[(intervals.model == model) & (intervals.forecast_hour_end <= horizon) & (intervals.subarea == "basin_to_05EA002")].copy()
    if rows.empty:
        return {"first_wet_h": horizon, "last_wet_h": horizon, "duration_h": 0}
    rows["mean_mm"] = pd.to_numeric(rows.mean_mm, errors="coerce").fillna(0.0)
    wet = rows[rows.mean_mm > 0.5].sort_values("forecast_hour_end")
    if wet.empty:
        return {"first_wet_h": horizon, "last_wet_h": horizon, "duration_h": 0}
    first = int(wet.forecast_hour_start.min())
    last = int(wet.forecast_hour_end.max())
    return {"first_wet_h": first, "last_wet_h": last, "duration_h": max(6, last - first)}


def uncertainty_value(cv: dict, target: str, fallback: float) -> float:
    value = cv.get(target, {}).get("rmse")
    return finite(value, fallback)


def routing_info() -> dict:
    if not ROUTING.exists():
        return {"central_h": 0.0, "range_h": [0.0, 30.0], "status": "routing unavailable"}
    data = json.loads(ROUTING.read_text())
    estimate = data.get("estimated_05EA002_to_starkey_wave_lag", {})
    return {
        "central_h": finite(estimate.get("central_h"), 0.0),
        "range_h": estimate.get("range_h", [0.0, 30.0]),
        "status": data.get("observed_05EA002_to_05EA001_lag", {}).get("status"),
        "correlation": data.get("observed_05EA002_to_05EA001_lag", {}).get("best_correlation"),
    }


def reps_validation() -> dict:
    if not QPF_META.exists():
        return {"validated": False, "reason": "QPF metadata missing"}
    metadata = json.loads(QPF_META.read_text()).get("reps", {})
    member_markers = 0
    band_count = 0
    examples = []
    for file_meta in metadata.get("files", []):
        band_count = max(band_count, int(file_meta.get("band_count", 0)))
        for band in file_meta.get("band_metadata_sample", []):
            text = json.dumps(band, sort_keys=True, default=json_default).lower()
            if any(token in text for token in ["perturbation", "ensemble member", "numberofensembleforecasts", "grib_perturbationnumber"]):
                member_markers += 1
                if len(examples) < 3:
                    examples.append(band)
    validated = band_count >= 15 and member_markers >= 5
    return {
        "validated": validated,
        "band_count": band_count,
        "member_metadata_markers": member_markers,
        "examples": examples,
        "reason": None if validated else "REPS bands have not yet been verified as individual ensemble members rather than mixed statistics.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    previous_path = OUT / "forecast_impacts_v2.json"
    previous_result = {}
    if previous_path.exists():
        try:
            previous_result = json.loads(previous_path.read_text())
        except Exception:
            previous_result = {}
    base = json.loads(BASE_CAL.read_text())
    events = pd.read_csv(CAL_V2)
    all_events, training = training_frame(events)
    training.to_csv(OUT / "training_events_v2.csv", index=False)
    all_events.to_csv(OUT / "all_event_constraints.csv", index=False)
    cv = cross_validate(training)
    horizons = pd.read_csv(QPF_HORIZON) if QPF_HORIZON.exists() and QPF_HORIZON.stat().st_size else pd.DataFrame()
    intervals = pd.read_csv(QPF_INTERVAL) if QPF_INTERVAL.exists() and QPF_INTERVAL.stat().st_size else pd.DataFrame()
    stage_now = finite(base.get("latest_stage_m"))
    recession_model = base.get("master_recession", {})
    rain_free_days = finite(base.get("rain_free_projection_to_1_70", {}).get("days"))
    antecedent = current_antecedent_rain(168)
    route = routing_info()
    scenarios = []
    if not horizons.empty:
        for model in sorted(horizons.model.unique()):
            for horizon in sorted(pd.to_numeric(horizons[horizons.model == model].horizon_h, errors="coerce").dropna().astype(int).unique()):
                basin_row = horizon_record(horizons, model, horizon, "basin_to_05EA002")
                lower_row = horizon_record(horizons, model, horizon, "lower_incremental_05EA005_to_05EA002")
                upper_row = horizon_record(horizons, model, horizon, "upper_lake_chain_isle_lac_ste_anne")
                middle_row = horizon_record(horizons, model, horizon, "lac_ste_anne_to_villeneuve_mainstem")
                atim_row = horizon_record(horizons, model, horizon, "atim_creek_big_lake_tributary")
                carrot_row = horizon_record(horizons, model, horizon, "carrot_creek")
                local_row = horizon_record(horizons, model, horizon, "direct_big_lake_and_local_to_05EA002")
                if basin_row is None:
                    continue
                basin = finite(basin_row.mean_mm, 0.0)
                lower = finite(lower_row.mean_mm, 0.0) if lower_row is not None else 0.0
                upper = finite(upper_row.mean_mm, 0.0) if upper_row is not None else 0.0
                middle = finite(middle_row.mean_mm, 0.0) if middle_row is not None else 0.0
                atim = finite(atim_row.mean_mm, 0.0) if atim_row is not None else 0.0
                carrot = finite(carrot_row.mean_mm, 0.0) if carrot_row is not None else 0.0
                local = finite(local_row.mean_mm, 0.0) if local_row is not None else 0.0
                pct10 = finite(basin_row.get("pct_gt_10mm"), 0.0)
                pct5 = finite(basin_row.get("pct_gt_5mm"), 0.0)
                timing = interval_timing(intervals, model, horizon)
                storm_type = classify(basin, lower, upper, middle, atim, carrot, local, pct10, pct5)
                detail = {
                    "model": model,
                    "horizon_h": int(horizon),
                    "run_time_utc": basin_row.get("run_time_utc"),
                    "complete_horizon": bool(basin_row.get("complete_horizon", False)),
                    "basin_mm": basin,
                    "lower_mm": lower,
                    "upper_mm": upper,
                    "middle_mm": middle,
                    "atim_mm": atim,
                    "carrot_mm": carrot,
                    "direct_local_mm": local,
                    "basin_pct_gt_5mm": pct5,
                    "basin_pct_gt_10mm": pct10,
                    "basin_pct_gt_20mm": finite(basin_row.get("pct_gt_20mm"), 0.0),
                    "basin_pct_gt_30mm": finite(basin_row.get("pct_gt_30mm"), 0.0),
                    "forecast_timing": timing,
                    "storm_type": storm_type,
                    "antecedent_168h_mm": antecedent,
                }
                if basin < 0.5 or timing["duration_h"] == 0:
                    detail.update({
                        "impact": "negligible or no material deterministic QPF",
                        "projected_1_70_days": rain_free_days,
                        "projected_1_70_date_utc": (datetime.now(timezone.utc) + timedelta(days=rain_free_days)).date().isoformat() if np.isfinite(rain_free_days) else None,
                    })
                    scenarios.append(detail)
                    continue
                features = np.asarray([
                    basin,
                    lower / basin if basin > 0 else 1.0,
                    upper / basin if basin > 0 else 1.0,
                    timing["duration_h"],
                    antecedent,
                    pct10,
                    stage_now,
                ], dtype=float)
                same_type = training[training.storm_type == storm_type]
                pool = same_type if len(same_type) >= 2 else training
                if len(pool) < 2:
                    detail.update({
                        "impact": "material QPF but insufficient uncensored analogues",
                        "confidence": "very low",
                        "training_event_count": int(len(training)),
                    })
                    scenarios.append(detail)
                    continue
                prediction = analog_predict(pool, features, k=min(3, len(pool)))
                departure_error = uncertainty_value(cv, "departure_m", max(0.05, float(pool.departure_m.std(ddof=0) if len(pool) > 1 else 0.08)))
                days_error = uncertainty_value(cv, "days_lost", max(1.0, float(pool.days_lost.std(ddof=0) if len(pool) > 1 else 1.5)))
                lag_error = uncertainty_value(cv, "lag_h", max(24.0, float(pool.lag_h.std(ddof=0) if len(pool) > 1 else 48.0)))
                departure_range = [
                    max(0.0, min(prediction["departure_m_min"], prediction["departure_m"] - departure_error)),
                    max(prediction["departure_m_max"], prediction["departure_m"] + departure_error),
                ]
                days_range = [
                    max(0.0, min(prediction["days_lost_min"], prediction["days_lost"] - days_error)),
                    max(prediction["days_lost_max"], prediction["days_lost"] + days_error),
                ]
                lag_range = [
                    max(0.0, prediction["lag_h"] - lag_error),
                    prediction["lag_h"] + lag_error,
                ]
                similar_censored = all_events[
                    all_events.censored
                    & (all_events.storm_type == storm_type)
                    & (abs(all_events.basin_mm - basin) <= max(5.0, 0.5 * basin))
                ]
                censored_lower_bounds = []
                if not similar_censored.empty:
                    censored_lower_bounds = similar_censored[["event_id", "departure_m", "days_lost", "response_quality"]].to_dict(orient="records")
                    departure_range[1] = max(departure_range[1], float(similar_censored.departure_m.max()) + departure_error)
                    days_range[1] = max(days_range[1], float(similar_censored.days_lost.max()) + days_error)
                event_start = timing["first_wet_h"]
                peak_effect_hour = event_start + prediction["lag_h"]
                baseline_peak = rain_free_stage(stage_now, recession_model, peak_effect_hour)
                peak_range = [baseline_peak + departure_range[0], baseline_peak + departure_range[1]]
                starkey_central = peak_effect_hour + route["central_h"]
                route_range = route.get("range_h", [0.0, 30.0])
                starkey_range = [event_start + lag_range[0] + finite(route_range[0], 0.0), event_start + lag_range[1] + finite(route_range[1], 30.0)]
                confidence = "low"
                if len(pool) >= 5 and prediction["nearest_distance"] <= 0.75 and not similar_censored.empty:
                    confidence = "low-moderate"
                elif len(pool) >= 5 and prediction["nearest_distance"] <= 0.75:
                    confidence = "moderate"
                detail.update({
                    "impact": "quantified from uncensored spatial analogues",
                    "feature_vector": dict(zip(FEATURES, [float(value) for value in features])),
                    "training_pool_event_count": int(len(pool)),
                    "training_pool_storm_type_matched": bool(len(same_type) >= 2),
                    "analog_prediction": prediction,
                    "similar_censored_event_lower_bounds": censored_lower_bounds,
                    "stage_departure_range_m": departure_range,
                    "estimated_days_lost_range": days_range,
                    "lag_to_max_effect_from_rain_start_h": lag_range,
                    "estimated_peak_stage_05EA002_m": baseline_peak + prediction["departure_m"],
                    "estimated_peak_stage_05EA002_range_m": peak_range,
                    "starkey_max_effect_hours_from_now_central": starkey_central,
                    "starkey_max_effect_hours_from_now_range": starkey_range,
                    "projected_1_70_days_central": rain_free_days + prediction["days_lost"],
                    "projected_1_70_days_range": [rain_free_days + days_range[0], rain_free_days + days_range[1]],
                    "confidence": confidence,
                    "threshold_flags": {
                        "departure_ge_0_05": departure_range[1] >= 0.05,
                        "delay_ge_2_days": days_range[1] >= 2.0,
                        "credible_2_5_path": peak_range[1] >= 2.5,
                        "credible_3_0_path": peak_range[1] >= 3.0,
                    },
                })
                scenarios.append(detail)
    current_complete_48 = any(
        str(row.get("model")) == "HRDPS"
        and int(row.get("horizon_h", 0) or 0) == 48
        and bool(row.get("complete_horizon"))
        for row in scenarios
    )
    short_range_provenance = {
        "status": "current_complete_hrdps" if current_complete_48 else "current_hrdps_incomplete",
        "maximum_carry_forward_age_hours": 12.0,
    }
    if not current_complete_48:
        previous_scenarios = previous_result.get("deterministic_scenarios", [])
        previous_hrdps = [
            dict(row)
            for row in previous_scenarios
            if str(row.get("model")) == "HRDPS"
            and int(row.get("horizon_h", 0) or 0) in {24, 48}
            and bool(row.get("complete_horizon"))
        ]
        previous_has_48 = any(
            int(row.get("horizon_h", 0) or 0) == 48 for row in previous_hrdps
        )
        previous_generated = previous_result.get("generated_utc")
        previous_age_hours = None
        if previous_generated:
            try:
                previous_time = datetime.fromisoformat(
                    str(previous_generated).replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                previous_age_hours = max(
                    0.0,
                    (datetime.now(timezone.utc) - previous_time).total_seconds() / 3600.0,
                )
            except Exception:
                previous_age_hours = None
        if (
            previous_has_48
            and previous_age_hours is not None
            and previous_age_hours <= 12.0
        ):
            scenarios = [
                row for row in scenarios if str(row.get("model")) != "HRDPS"
            ]
            for row in previous_hrdps:
                row["input_provenance"] = "carried_forward_last_valid_hrdps"
                row["carried_forward_from_generated_utc"] = previous_generated
                row["carried_forward_age_hours"] = previous_age_hours
                scenarios.append(row)
            short_range_provenance = {
                "status": "carried_forward_last_valid_hrdps",
                "source_generated_utc": previous_generated,
                "age_hours": previous_age_hours,
                "maximum_carry_forward_age_hours": 12.0,
                "interpretation": (
                    "The current HRDPS publication was incomplete. The previous complete "
                    "24/48-hour scenarios were retained for up to 12 hours rather than "
                    "silently treating missing short-range rainfall as zero."
                ),
            }
        else:
            short_range_provenance = {
                "status": "short_range_forecast_unavailable",
                "previous_complete_age_hours": previous_age_hours,
                "maximum_carry_forward_age_hours": 12.0,
                "interpretation": (
                    "No current or sufficiently recent complete HRDPS 48-hour scenario is available."
                ),
            }

    reps_status = reps_validation()
    storm_type_counts = {
        str(key): int(value)
        for key, value in training.storm_type.value_counts().to_dict().items()
    } if not training.empty else {}
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "latest_stage_m": stage_now,
        "rain_free_days_to_1_70": rain_free_days,
        "current_antecedent_168h_basin_rain_mm": antecedent,
        "training": {
            "all_event_constraints": int(len(all_events)),
            "uncensored_peak_training_events": int(len(training)),
            "storm_type_counts": storm_type_counts,
            "features": FEATURES,
            "targets": TARGETS,
        },
        "cross_validation": cv,
        "routing": route,
        "reps_validation": reps_status,
        "short_range_input_provenance": short_range_provenance,
        "deterministic_scenarios": scenarios,
        "limitations": [
            "Only uncensored event peaks are used for point prediction; censored events widen upper bounds as lower-bound constraints.",
            "Starkey timing is translated from 05EA002 using weakly correlated downstream timing and remains low confidence.",
            "Exact Starkey water level is not predicted because no common site gauge datum exists.",
            "REPS probabilities are withheld from operational use until GRIB band metadata verifies individual ensemble members.",
        ],
    }
    (OUT / "forecast_impacts_v2.json").write_text(
        json.dumps(result, indent=2, default=json_default)
    )
    pd.DataFrame(scenarios).to_json(OUT / "deterministic_scenarios_v2.json", orient="records", indent=2)
    print(
        json.dumps(
            {
                "training_events": len(training),
                "scenarios": len(scenarios),
                "reps_validated": reps_status["validated"],
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
