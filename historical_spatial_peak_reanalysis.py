#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from historical_rdpa_pairing import build_hourly
from historical_spatial_event_backfill import (
    FEATURES,
    dynamic_baseline,
    historical_recession_fits,
    safe_json,
)

EVENTS_DEFAULT = Path(
    "output/historical_event_backfill/historical_spatial_events.csv"
)
PAIRS_DEFAULT = Path("output/archive_probe/historical_rdpa_pairs.csv")
OUT_DEFAULT = Path(
    "output/historical_event_backfill/historical_spatial_peak_reanalysis.json"
)
AUGMENTED_DEFAULT = Path(
    "output/historical_event_backfill/historical_spatial_events_peak_reanalysis.csv"
)
MODEL_DEFAULT = Path(
    "output/historical_event_backfill/historical_spatial_peak_response_model.json"
)

AMOUNT_FEATURES = ["log_basin_mm", "log_antecedent_168h_mm", "pre_stage_m"]
SPATIAL_FEATURES = [
    "log_basin_mm",
    "lower_ratio",
    "upper_ratio",
    "log_duration_h",
    "log_antecedent_168h_mm",
    "basin_pct_gt_10mm_fraction",
    "pre_stage_m",
]


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "rmse_days": float(np.sqrt(np.mean(error**2))),
        "mae_days": float(np.mean(np.abs(error))),
        "bias_days": float(np.mean(error)),
        "median_absolute_error_days": float(np.median(np.abs(error))),
    }


def standardize(train_x: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-9] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    penalty: float,
) -> np.ndarray:
    x, test, _, _ = standardize(train_x, test_x)
    design = np.column_stack([np.ones(len(x)), x])
    test_design = np.column_stack([np.ones(len(test)), test])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ train_y
    )
    return test_design @ coefficients


def ridge_parameters(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    penalty: float,
) -> dict:
    standardized, _, mean, scale = standardize(x, x)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ y
    )
    return {
        "type": "standardized_log_target_ridge",
        "penalty": penalty,
        "features": names,
        "target_transform": "log1p_days_lost",
        "feature_means": {name: float(value) for name, value in zip(names, mean)},
        "feature_scales": {
            name: float(value) for name, value in zip(names, scale)
        },
        "intercept_log1p_days": float(coefficients[0]),
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(names, coefficients[1:])
        },
    }


def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k: int,
) -> np.ndarray:
    x, test, _, _ = standardize(train_x, test_x)
    predictions = []
    for row in test:
        distance = np.sqrt(np.sum((x - row) ** 2, axis=1))
        order = np.argsort(distance)[: max(1, min(k, len(distance)))]
        weights = 1.0 / np.maximum(distance[order], 0.35)
        predictions.append(float(np.average(train_y[order], weights=weights)))
    return np.asarray(predictions)


def transformed_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["log_basin_mm"] = np.log1p(
        pd.to_numeric(output.basin_mm, errors="coerce").clip(lower=0)
    )
    output["log_duration_h"] = np.log1p(
        pd.to_numeric(output.duration_h, errors="coerce").clip(lower=0)
    )
    output["log_antecedent_168h_mm"] = np.log1p(
        pd.to_numeric(output.antecedent_168h_mm, errors="coerce").clip(lower=0)
    )
    output["basin_pct_gt_10mm_fraction"] = (
        pd.to_numeric(output.basin_pct_gt_10mm, errors="coerce") / 100.0
    ).clip(lower=0, upper=1)
    output["lower_ratio"] = pd.to_numeric(
        output.lower_ratio, errors="coerce"
    ).clip(lower=0, upper=4)
    output["upper_ratio"] = pd.to_numeric(
        output.upper_ratio, errors="coerce"
    ).clip(lower=0, upper=4)
    output["pre_stage_m"] = pd.to_numeric(
        output.pre_stage_m, errors="coerce"
    )
    return output


def recompute_peak_support(
    row: pd.Series,
    hourly: pd.DataFrame,
    q_fit: dict,
    stage_fit: dict,
) -> dict:
    start = pd.Timestamp(row.rain_start_utc)
    analysis_end = pd.Timestamp(row.analysis_end_utc)
    pre_candidates = hourly.index[hourly.index <= start]
    if not len(pre_candidates):
        return {"peak_support_status": "no_pre_event_gauge_value"}
    t0 = pre_candidates[-1]
    observed = hourly.loc[t0:analysis_end].dropna(
        subset=["stage_m", "discharge_m3s"]
    )
    if len(observed) < 12:
        return {"peak_support_status": "insufficient_gauge_window"}

    q_base = dynamic_baseline(
        float(observed.discharge_m3s.iloc[0]), observed.index, q_fit
    )
    stage_base = dynamic_baseline(
        float(observed.stage_m.iloc[0]), observed.index, stage_fit
    )
    q_departure = observed.discharge_m3s - q_base
    stage_departure = observed.stage_m - stage_base
    q_post = q_departure.loc[q_departure.index >= start]
    stage_post = stage_departure.loc[stage_departure.index >= start]
    if q_post.empty:
        return {"peak_support_status": "no_post_event_gauge_values"}

    peak_time = q_post.idxmax()
    peak_q = float(q_post.loc[peak_time])
    stage_peak = float(stage_post.max()) if len(stage_post) else None
    end_q = float(q_post.iloc[-1])
    end_stage = float(stage_post.iloc[-1]) if len(stage_post) else None
    hours_after_peak = float(
        (analysis_end - peak_time).total_seconds() / 3600.0
    )
    decline_q = peak_q - end_q
    decline_fraction = decline_q / peak_q if peak_q > 0 else None

    post_peak = q_post.loc[q_post.index >= peak_time]
    last_6h_change = (
        float(post_peak.iloc[-1] - post_peak.iloc[-7])
        if len(post_peak) >= 7
        else None
    )
    last_12h_change = (
        float(post_peak.iloc[-1] - post_peak.iloc[-13])
        if len(post_peak) >= 13
        else None
    )
    minimum_required_decline = max(0.10, 0.10 * max(0.0, peak_q))
    peak_resolved = bool(
        peak_q >= 0.20
        and hours_after_peak >= 18.0
        and decline_q >= minimum_required_decline
        and (
            last_6h_change is None
            or last_6h_change <= max(0.05, 0.05 * peak_q)
        )
    )

    baseline_q_at_peak = float(q_base.loc[peak_time])
    recession_rate_at_peak = abs(
        min(
            -0.001,
            float(q_fit["intercept_per_day"])
            + float(q_fit["coefficient_per_day"]) * baseline_q_at_peak,
        )
    )
    peak_days_lost = (
        peak_q / recession_rate_at_peak
        if recession_rate_at_peak > 0.001
        else None
    )
    current_lower_bound_days = (
        max(0.0, end_q) / recession_rate_at_peak
        if recession_rate_at_peak > 0.001
        else None
    )

    recovery_q_threshold = max(0.15, 0.15 * max(0.0, peak_q))
    recovery_stage_threshold = max(0.015, 0.15 * max(0.0, stage_peak or 0.0))
    recovery_condition = (
        q_departure.loc[q_departure.index >= peak_time] <= recovery_q_threshold
    ) & (
        stage_departure.loc[stage_departure.index >= peak_time]
        <= recovery_stage_threshold
    )
    sustained = (
        recovery_condition.astype(int).rolling(12, min_periods=12).sum() >= 12
    )
    recovery_hits = sustained[sustained].index
    recovery = recovery_hits[0] if len(recovery_hits) else None

    return {
        "peak_support_status": "evaluated",
        "recomputed_peak_utc": peak_time.isoformat(),
        "recomputed_q_departure_peak_m3s": peak_q,
        "recomputed_stage_departure_peak_m": stage_peak,
        "q_departure_at_analysis_end_m3s": end_q,
        "stage_departure_at_analysis_end_m": end_stage,
        "hours_observed_after_peak": hours_after_peak,
        "post_peak_decline_m3s": decline_q,
        "post_peak_decline_fraction": decline_fraction,
        "last_6h_departure_change_m3s": last_6h_change,
        "last_12h_departure_change_m3s": last_12h_change,
        "minimum_required_peak_decline_m3s": minimum_required_decline,
        "peak_resolved": peak_resolved,
        "recomputed_peak_days_lost": peak_days_lost,
        "censored_remaining_days_lower_bound": current_lower_bound_days,
        "recomputed_recovery_utc": recovery.isoformat() if recovery is not None else None,
        "recovery_complete": recovery is not None,
        "recovery_duration_h": (
            float((recovery - peak_time).total_seconds() / 3600.0)
            if recovery is not None
            else None
        ),
    }


def fit_and_cross_validate(frame: pd.DataFrame) -> dict:
    work = transformed_features(frame)
    point = work[work.eligible_for_peak_training.astype(bool)].dropna(
        subset=SPATIAL_FEATURES + ["recomputed_peak_days_lost"]
    )
    lower_bounds = work[
        work.eligible_as_censored_lower_bound.astype(bool)
    ].dropna(subset=SPATIAL_FEATURES + ["recomputed_peak_days_lost"])
    result = {
        "resolved_peak_events": int(len(point)),
        "censored_lower_bound_events": int(len(lower_bounds)),
        "complete_recovery_events": int(
            work.eligible_for_recovery_training.astype(bool).sum()
        ),
        "storm_type_count": int(point.storm_type.nunique()) if len(point) else 0,
        "models": {},
        "preferred_candidate": None,
        "censored_constraint_evaluation": {},
        "promotion_screen": {},
    }
    if len(point) < 4:
        result["promotion_screen"] = {
            "candidate_passes_minimum_screen": False,
            "automatic_promotion_enabled": False,
            "reasons_not_to_promote": [
                "fewer_than_four_resolved_peak_spatial_events"
            ],
        }
        return result

    x_amount = point[AMOUNT_FEATURES].to_numpy(float)
    x_spatial = point[SPATIAL_FEATURES].to_numpy(float)
    y_days = point.recomputed_peak_days_lost.to_numpy(float)
    y_log = np.log1p(np.clip(y_days, 0, None))
    predictions: dict[str, list[float]] = {
        "median_baseline": [],
        "amount_only_log_ridge": [],
        "spatial_log_ridge": [],
        "spatial_log_knn": [],
    }
    fold_meta = []
    for index in range(len(point)):
        keep = np.arange(len(point)) != index
        predictions["median_baseline"].append(float(np.median(y_days[keep])))
        amount_log = ridge_predict(
            x_amount[keep], y_log[keep], x_amount[[index]], penalty=2.0
        )[0]
        spatial_log = ridge_predict(
            x_spatial[keep], y_log[keep], x_spatial[[index]], penalty=3.0
        )[0]
        knn_log = knn_predict(
            x_spatial[keep], y_log[keep], x_spatial[[index]], k=4
        )[0]
        predictions["amount_only_log_ridge"].append(
            float(max(0.0, np.expm1(amount_log)))
        )
        predictions["spatial_log_ridge"].append(
            float(max(0.0, np.expm1(spatial_log)))
        )
        predictions["spatial_log_knn"].append(
            float(max(0.0, np.expm1(knn_log)))
        )
        fold_meta.append(
            {
                "held_out_event_id": int(point.iloc[index].event_id),
                "storm_type": str(point.iloc[index].storm_type),
                "observed_days_lost": float(y_days[index]),
            }
        )

    for name, values in predictions.items():
        predicted = np.asarray(values, dtype=float)
        score = metrics(y_days, predicted)
        score["predictions"] = [
            {
                **fold_meta[index],
                "predicted_days_lost": float(predicted[index]),
            }
            for index in range(len(predicted))
        ]
        result["models"][name] = score

    candidate_names = (
        "amount_only_log_ridge",
        "spatial_log_ridge",
        "spatial_log_knn",
    )
    preferred = min(
        candidate_names,
        key=lambda name: result["models"][name]["rmse_days"],
    )
    result["preferred_candidate"] = preferred

    if preferred == "amount_only_log_ridge":
        result["fitted_shadow_model"] = ridge_parameters(
            x_amount, y_log, AMOUNT_FEATURES, penalty=2.0
        )
    elif preferred == "spatial_log_ridge":
        result["fitted_shadow_model"] = ridge_parameters(
            x_spatial, y_log, SPATIAL_FEATURES, penalty=3.0
        )
    else:
        result["fitted_shadow_model"] = {
            "type": "standardized_log_target_knn",
            "k": 4,
            "features": SPATIAL_FEATURES,
            "target_transform": "log1p_days_lost",
            "training_events": [
                {
                    "event_id": int(row.event_id),
                    "storm_type": str(row.storm_type),
                    "target_days_lost": float(row.recomputed_peak_days_lost),
                    "features": {
                        name: float(getattr(row, name))
                        for name in SPATIAL_FEATURES
                    },
                }
                for row in point.itertuples(index=False)
            ],
        }

    median_rmse = result["models"]["median_baseline"]["rmse_days"]
    amount_rmse = result["models"]["amount_only_log_ridge"]["rmse_days"]
    preferred_rmse = result["models"][preferred]["rmse_days"]
    gain_vs_median = (
        (median_rmse - preferred_rmse) / median_rmse * 100.0
        if median_rmse > 0
        else None
    )
    gain_vs_amount = (
        (amount_rmse - preferred_rmse) / amount_rmse * 100.0
        if amount_rmse > 0
        else None
    )

    # Evaluate unresolved peaks as one-sided constraints. Their observed maximum
    # is a lower bound on the eventual event response; underpredicting it is a
    # warning, while overpredicting it is not scored as an error.
    if len(lower_bounds):
        if preferred == "amount_only_log_ridge":
            train_x = x_amount
            test_x = lower_bounds[AMOUNT_FEATURES].to_numpy(float)
            predicted_log = ridge_predict(
                train_x, y_log, test_x, penalty=2.0
            )
        elif preferred == "spatial_log_ridge":
            train_x = x_spatial
            test_x = lower_bounds[SPATIAL_FEATURES].to_numpy(float)
            predicted_log = ridge_predict(
                train_x, y_log, test_x, penalty=3.0
            )
        else:
            train_x = x_spatial
            test_x = lower_bounds[SPATIAL_FEATURES].to_numpy(float)
            predicted_log = knn_predict(train_x, y_log, test_x, k=4)
        predicted_days = np.maximum(0.0, np.expm1(predicted_log))
        lower = lower_bounds.recomputed_peak_days_lost.to_numpy(float)
        shortfall = np.maximum(0.0, lower - predicted_days)
        result["censored_constraint_evaluation"] = {
            "events": int(len(lower)),
            "constraints_satisfied": int(np.sum(predicted_days >= lower)),
            "constraint_satisfaction_pct": float(
                np.mean(predicted_days >= lower) * 100.0
            ),
            "mean_underprediction_shortfall_days": float(np.mean(shortfall)),
            "maximum_underprediction_shortfall_days": float(np.max(shortfall)),
            "records": [
                {
                    "event_id": int(lower_bounds.iloc[index].event_id),
                    "observed_response_lower_bound_days": float(lower[index]),
                    "predicted_days_lost": float(predicted_days[index]),
                    "constraint_satisfied": bool(
                        predicted_days[index] >= lower[index]
                    ),
                }
                for index in range(len(lower))
            ],
        }

    reasons = []
    if len(point) < 10:
        reasons.append("fewer_than_ten_resolved_peak_spatial_events")
    if point.storm_type.nunique() < 3:
        reasons.append("fewer_than_three_resolved_peak_storm_types")
    if gain_vs_median is None or gain_vs_median < 15.0:
        reasons.append(
            "preferred_model_improves_median_baseline_by_less_than_15_percent"
        )
    if preferred.startswith("spatial") and (
        gain_vs_amount is None or gain_vs_amount < 10.0
    ):
        reasons.append(
            "spatial_features_improve_amount_only_model_by_less_than_10_percent"
        )
    if not preferred.startswith("spatial"):
        reasons.append("spatial_model_is_not_cross_validated_best")
    constraint_pct = finite(
        result.get("censored_constraint_evaluation", {}).get(
            "constraint_satisfaction_pct"
        )
    )
    if constraint_pct is not None and constraint_pct < 75.0:
        reasons.append("preferred_model_satisfies_fewer_than_75_percent_of_censored_bounds")

    result["promotion_screen"] = {
        "candidate_passes_minimum_screen": not reasons,
        "automatic_promotion_enabled": False,
        "reasons_not_to_promote": reasons,
        "skill_improvement_vs_median_pct": gain_vs_median,
        "skill_improvement_vs_amount_only_pct": gain_vs_amount,
        "requirements": [
            "at least ten isolated events with clearly resolved response peaks",
            "at least three resolved-peak storm-pattern classes",
            "leave-one-event-out validation available",
            "preferred model improves median baseline by at least 15 percent",
            "spatial model improves amount-only model by at least 10 percent",
            "at least 75 percent of unresolved censored lower bounds are satisfied",
            "manual engineering review and operational hindcast before promotion",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--events", default=str(EVENTS_DEFAULT))
    parser.add_argument("--pairs", default=str(PAIRS_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument("--augmented-output", default=str(AUGMENTED_DEFAULT))
    parser.add_argument("--model-output", default=str(MODEL_DEFAULT))
    args = parser.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    frame = pd.read_csv(events_path)
    if frame.empty:
        raise RuntimeError("Historical spatial event table is empty")
    hourly, retrieval = build_hourly(args.months)
    q_fit, stage_fit, recession_support = historical_recession_fits(
        Path(args.pairs)
    )

    support_rows = []
    for _, row in frame.iterrows():
        support_rows.append(
            recompute_peak_support(row, hourly, q_fit, stage_fit)
        )
    support = pd.DataFrame(support_rows, index=frame.index)
    augmented = pd.concat([frame, support], axis=1)

    isolated = augmented.isolated_event.map(bool_value)
    spatial_ok = (
        pd.to_numeric(
            augmented.spatial_coverage_fraction, errors="coerce"
        ).fillna(0.0)
        >= 0.80
    )
    peak_resolved = augmented.peak_resolved.map(bool_value)
    peak_days = pd.to_numeric(
        augmented.recomputed_peak_days_lost, errors="coerce"
    )
    response_detected = (
        pd.to_numeric(
            augmented.recomputed_q_departure_peak_m3s, errors="coerce"
        ).fillna(0.0)
        >= 0.20
    )
    plausible = peak_days.between(0.05, 45.0, inclusive="both")
    augmented["eligible_for_peak_training"] = (
        isolated & spatial_ok & peak_resolved & response_detected & plausible
    )
    augmented["eligible_for_recovery_training"] = (
        augmented.eligible_for_peak_training
        & augmented.recovery_complete.map(bool_value)
    )
    augmented["eligible_as_censored_lower_bound"] = (
        isolated
        & spatial_ok
        & ~peak_resolved
        & response_detected
        & plausible
    )
    augmented["peak_response_quality"] = np.select(
        [
            ~spatial_ok,
            ~isolated,
            ~response_detected,
            peak_resolved & plausible,
            ~peak_resolved & plausible,
            ~plausible,
        ],
        [
            "insufficient_spatial_coverage",
            "antecedent_rain_overlap",
            "weak_or_no_detectable_response",
            "resolved_peak",
            "unresolved_peak_lower_bound",
            "implausible_or_unsupported_peak_days_lost",
        ],
        default="unclassified",
    )

    augmented_path = Path(args.augmented_output)
    augmented_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(augmented_path, index=False)

    validation = fit_and_cross_validate(augmented)
    model_payload = {
        "generated_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "historical_spatial_peak_response_shadow_model_evaluated",
        "mode": "shadow_only_manual_promotion_required",
        "target": "peak-derived recession days lost",
        "amount_features": AMOUNT_FEATURES,
        "spatial_features": SPATIAL_FEATURES,
        "validation": validation,
        "peak_resolution_gate": {
            "minimum_hours_after_peak": 18.0,
            "minimum_peak_departure_m3s": 0.20,
            "minimum_post_peak_decline": "max(0.10 m3/s, 10 percent of peak departure)",
            "maximum_supported_point_target_days": 45.0,
            "interpretation": (
                "A clearly resolved response peak may train peak-derived delay even if "
                "the next storm arrives before full recovery. Full recovery is retained "
                "as a separate and stricter training target."
            ),
        },
        "limitations": [
            "Historical WCS RDPA is 10 km, while the live HRDPS and recent HRDPA products are 2.5 km.",
            "Peak-derived days lost is an empirical recession-equivalent metric, not observed construction delay.",
            "WaterOffice discharge is rating-derived and provisional.",
            "Unresolved peaks are one-sided lower-bound checks and are not point-training targets.",
            "No automatic operational promotion is permitted.",
        ],
    }
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(model_payload, indent=2, default=safe_json)
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "historical_spatial_peak_reanalysis_complete",
        "mode": "shadow_only_no_automatic_promotion",
        "station": "05EA002",
        "requested_months": args.months,
        "source_event_count": int(len(frame)),
        "wateroffice_retrieval": {
            "chunks": retrieval,
            "hourly_points": int(len(hourly)),
            "first_utc": hourly.index.min().isoformat(),
            "last_utc": hourly.index.max().isoformat(),
        },
        "recession_baseline": {
            "support": recession_support,
            "discharge_fit": q_fit,
            "stage_fit": stage_fit,
        },
        "event_quality_counts": {
            key: int(value)
            for key, value in augmented.peak_response_quality.value_counts().items()
        },
        "resolved_peak_training_events": int(
            augmented.eligible_for_peak_training.sum()
        ),
        "censored_lower_bound_events": int(
            augmented.eligible_as_censored_lower_bound.sum()
        ),
        "complete_recovery_training_events": int(
            augmented.eligible_for_recovery_training.sum()
        ),
        "resolved_peak_storm_type_counts": augmented.loc[
            augmented.eligible_for_peak_training.astype(bool), "storm_type"
        ].value_counts().to_dict(),
        "model_evaluation": validation,
        "outputs": {
            "augmented_events_csv": str(augmented_path),
            "model_json": str(model_path),
        },
        "interpretation": (
            "Historical events are now separated into resolved response peaks, full "
            "recoveries, unresolved censored lower bounds, antecedent-overlap events, "
            "and weak responses. Resolved peaks can train the spatial rainfall-response "
            "model without pretending that recovery was observed."
        ),
        "limitations": model_payload["limitations"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=safe_json))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "resolved_peak_events": summary[
                    "resolved_peak_training_events"
                ],
                "censored_lower_bounds": summary[
                    "censored_lower_bound_events"
                ],
                "complete_recoveries": summary[
                    "complete_recovery_training_events"
                ],
                "preferred_model": validation.get("preferred_candidate"),
                "promotion_screen": validation.get("promotion_screen"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
