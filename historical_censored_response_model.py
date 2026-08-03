#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from historical_spatial_event_backfill import safe_json
from historical_spatial_peak_reanalysis import (
    AMOUNT_FEATURES,
    SPATIAL_FEATURES,
    metrics,
    transformed_features,
)

INPUT_DEFAULT = Path(
    "output/historical_event_backfill/historical_response_target_diagnostics.csv"
)
OUT_DEFAULT = Path(
    "output/historical_event_backfill/historical_censored_response_model.json"
)
PREDICTIONS_DEFAULT = Path(
    "output/historical_event_backfill/historical_censored_response_predictions.csv"
)

TARGET = "stage_exact_recession_equivalent_days"


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def standardization(x: np.ndarray):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= 1e-9] = 1.0
    return mean, scale


def design_matrix(x: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    standardized = (x - mean) / scale
    return np.column_stack([np.ones(len(standardized)), standardized])


def weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge_penalty: float,
) -> np.ndarray:
    root_weight = np.sqrt(np.maximum(weights, 0.0))
    weighted_design = design * root_weight[:, None]
    weighted_target = target * root_weight
    regularizer = np.eye(design.shape[1]) * float(ridge_penalty)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_target,
    )


def fit_censored_active_set(
    exact_x: np.ndarray,
    exact_y_log: np.ndarray,
    censored_x: np.ndarray,
    lower_log: np.ndarray,
    ridge_penalty: float,
    censored_weight: float,
    maximum_iterations: int = 50,
) -> dict:
    combined = (
        np.vstack([exact_x, censored_x])
        if len(censored_x)
        else exact_x.copy()
    )
    mean, scale = standardization(combined)
    exact_design = design_matrix(exact_x, mean, scale)
    censored_design = (
        design_matrix(censored_x, mean, scale)
        if len(censored_x)
        else np.empty((0, exact_design.shape[1]))
    )
    coefficients = weighted_ridge(
        exact_design,
        exact_y_log,
        np.ones(len(exact_y_log)),
        ridge_penalty,
    )
    active = np.zeros(len(lower_log), dtype=bool)
    iterations = 0
    for iteration in range(maximum_iterations):
        iterations = iteration + 1
        predicted_censored = (
            censored_design @ coefficients
            if len(censored_design)
            else np.asarray([], dtype=float)
        )
        new_active = predicted_censored < lower_log
        if len(lower_log) and np.any(new_active):
            active_design = censored_design[new_active]
            active_target = lower_log[new_active]
            design = np.vstack([exact_design, active_design])
            target = np.concatenate([exact_y_log, active_target])
            weights = np.concatenate(
                [
                    np.ones(len(exact_y_log)),
                    np.full(len(active_target), float(censored_weight)),
                ]
            )
        else:
            design = exact_design
            target = exact_y_log
            weights = np.ones(len(exact_y_log))
        updated = weighted_ridge(
            design,
            target,
            weights,
            ridge_penalty,
        )
        stable = np.array_equal(new_active, active) and np.max(
            np.abs(updated - coefficients)
        ) < 1e-8
        coefficients = updated
        active = new_active
        if stable:
            break
    return {
        "coefficients": coefficients,
        "feature_mean": mean,
        "feature_scale": scale,
        "iterations": iterations,
        "active_constraints": int(active.sum()),
        "total_constraints": int(len(active)),
    }


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    design = design_matrix(
        x,
        model["feature_mean"],
        model["feature_scale"],
    )
    predicted_log = design @ model["coefficients"]
    return np.maximum(0.0, np.expm1(predicted_log))


def constraint_metrics(lower: np.ndarray, predicted: np.ndarray) -> dict:
    satisfied = predicted >= lower
    shortfall = np.maximum(0.0, lower - predicted)
    return {
        "events": int(len(lower)),
        "constraints_satisfied": int(satisfied.sum()),
        "constraint_satisfaction_pct": (
            float(np.mean(satisfied) * 100.0) if len(lower) else None
        ),
        "mean_shortfall_days": float(np.mean(shortfall)) if len(lower) else None,
        "maximum_shortfall_days": float(np.max(shortfall)) if len(lower) else None,
        "mean_predicted_days": float(np.mean(predicted)) if len(lower) else None,
        "mean_lower_bound_days": float(np.mean(lower)) if len(lower) else None,
    }


def model_parameters(model: dict, feature_names: list[str]) -> dict:
    coefficients = model["coefficients"]
    return {
        "type": "active_set_squared_hinge_log_target_ridge",
        "target": TARGET,
        "target_transform": "log1p_days",
        "features": feature_names,
        "ridge_penalty": model["ridge_penalty"],
        "censored_constraint_weight": model["censored_weight"],
        "iterations": model["iterations"],
        "active_constraints": model["active_constraints"],
        "total_constraints": model["total_constraints"],
        "feature_means": {
            name: float(value)
            for name, value in zip(feature_names, model["feature_mean"])
        },
        "feature_scales": {
            name: float(value)
            for name, value in zip(feature_names, model["feature_scale"])
        },
        "intercept_log1p_days": float(coefficients[0]),
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(feature_names, coefficients[1:])
        },
    }


def evaluate_candidate(
    exact: pd.DataFrame,
    censored: pd.DataFrame,
    feature_names: list[str],
    ridge_penalty: float,
    censored_weight: float,
) -> dict:
    exact_x = exact[feature_names].to_numpy(float)
    exact_y = exact[TARGET].to_numpy(float)
    exact_log = np.log1p(np.clip(exact_y, 0, None))
    censored_x = censored[feature_names].to_numpy(float)
    lower = censored[TARGET].to_numpy(float)
    lower_log = np.log1p(np.clip(lower, 0, None))
    fold_predictions = []
    fold_rows = []
    for index in range(len(exact)):
        keep = np.arange(len(exact)) != index
        model = fit_censored_active_set(
            exact_x[keep],
            exact_log[keep],
            censored_x,
            lower_log,
            ridge_penalty,
            censored_weight,
        )
        prediction = float(predict(model, exact_x[[index]])[0])
        fold_predictions.append(prediction)
        fold_rows.append(
            {
                "held_out_event_id": int(exact.iloc[index].event_id),
                "storm_type": str(exact.iloc[index].storm_type),
                "observed_days": float(exact_y[index]),
                "predicted_days": prediction,
            }
        )
    predicted_exact = np.asarray(fold_predictions, dtype=float)
    exact_score = metrics(exact_y, predicted_exact)
    exact_score["predictions"] = fold_rows

    full = fit_censored_active_set(
        exact_x,
        exact_log,
        censored_x,
        lower_log,
        ridge_penalty,
        censored_weight,
    )
    predicted_bounds = (
        predict(full, censored_x) if len(censored_x) else np.asarray([])
    )
    bounds_score = constraint_metrics(lower, predicted_bounds)
    full.update(
        {
            "ridge_penalty": float(ridge_penalty),
            "censored_weight": float(censored_weight),
        }
    )
    return {
        "feature_set": (
            "spatial" if feature_names == SPATIAL_FEATURES else "amount_only"
        ),
        "feature_names": feature_names,
        "ridge_penalty": float(ridge_penalty),
        "censored_weight": float(censored_weight),
        "exact_leave_one_event_out": exact_score,
        "censored_constraints": bounds_score,
        "full_model": model_parameters(full, feature_names),
        "censored_predictions": [
            {
                "event_id": int(censored.iloc[index].event_id),
                "storm_type": str(censored.iloc[index].storm_type),
                "lower_bound_days": float(lower[index]),
                "predicted_days": float(predicted_bounds[index]),
                "constraint_satisfied": bool(
                    predicted_bounds[index] >= lower[index]
                ),
            }
            for index in range(len(censored))
        ],
    }


def exact_only_baseline(
    exact: pd.DataFrame,
    feature_names: list[str],
    ridge_penalty: float,
) -> dict:
    empty = exact.iloc[0:0].copy()
    return evaluate_candidate(
        exact,
        empty,
        feature_names,
        ridge_penalty,
        censored_weight=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument(
        "--predictions-output", default=str(PREDICTIONS_DEFAULT)
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    if raw.empty:
        raise RuntimeError("Historical response-target table is empty")
    frame = transformed_features(raw)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    exact = frame[
        frame.eligible_for_peak_training.map(bool_value)
    ].dropna(subset=SPATIAL_FEATURES + [TARGET])
    censored = frame[
        frame.eligible_as_censored_lower_bound.map(bool_value)
    ].dropna(subset=SPATIAL_FEATURES + [TARGET])
    exact = exact[exact[TARGET].between(0.05, 60.0, inclusive="both")]
    censored = censored[
        censored[TARGET].between(0.05, 60.0, inclusive="both")
    ]
    if len(exact) < 4:
        raise RuntimeError("Fewer than four exact resolved-peak events")

    amount_baseline = exact_only_baseline(
        exact, AMOUNT_FEATURES, ridge_penalty=2.0
    )
    baseline_rmse = amount_baseline["exact_leave_one_event_out"]["rmse_days"]

    candidates = []
    for feature_names in (AMOUNT_FEATURES, SPATIAL_FEATURES):
        for ridge_penalty in (1.0, 3.0, 10.0):
            for censored_weight in (0.10, 0.25, 0.50, 1.0):
                candidates.append(
                    evaluate_candidate(
                        exact,
                        censored,
                        feature_names,
                        ridge_penalty,
                        censored_weight,
                    )
                )

    for candidate in candidates:
        rmse = candidate["exact_leave_one_event_out"]["rmse_days"]
        satisfaction = candidate["censored_constraints"].get(
            "constraint_satisfaction_pct"
        )
        shortfall = candidate["censored_constraints"].get(
            "mean_shortfall_days"
        )
        candidate["selection_metrics"] = {
            "rmse_ratio_vs_exact_only_amount": rmse / baseline_rmse,
            "exact_rmse_within_15pct_of_baseline": rmse
            <= 1.15 * baseline_rmse,
            "constraint_satisfaction_at_least_75pct": (
                satisfaction is not None and satisfaction >= 75.0
            ),
            "composite_score": float(
                rmse
                + 0.25 * (shortfall or 0.0)
                + 0.02 * max(0.0, 75.0 - (satisfaction or 0.0))
            ),
        }

    fully_eligible = [
        candidate
        for candidate in candidates
        if candidate["selection_metrics"][
            "exact_rmse_within_15pct_of_baseline"
        ]
        and candidate["selection_metrics"][
            "constraint_satisfaction_at_least_75pct"
        ]
    ]
    preferred = min(
        fully_eligible or candidates,
        key=lambda candidate: candidate["selection_metrics"]["composite_score"],
    )
    preferred_is_eligible = preferred in fully_eligible

    reasons = []
    if len(exact) < 10:
        reasons.append("fewer_than_ten_resolved_peak_events")
    if len(censored) < 15:
        reasons.append("fewer_than_fifteen_censored_lower_bound_events")
    if not preferred_is_eligible:
        reasons.append(
            "no_candidate_preserves_exact_event_skill_and_satisfies_75_percent_of_censored_bounds"
        )

    prediction_rows = []
    for record in preferred.get("censored_predictions", []):
        prediction_rows.append({"record_type": "censored_lower_bound", **record})
    for record in preferred["exact_leave_one_event_out"].get(
        "predictions", []
    ):
        prediction_rows.append({"record_type": "resolved_peak_holdout", **record})
    predictions_path = Path(args.predictions_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    output = {
        "generated_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "historical_censored_response_model_evaluated",
        "mode": "shadow_only_manual_promotion_required",
        "target": TARGET,
        "resolved_peak_events": int(len(exact)),
        "censored_lower_bound_events": int(len(censored)),
        "exact_only_amount_baseline": amount_baseline,
        "candidate_count": int(len(candidates)),
        "preferred_candidate": preferred,
        "preferred_candidate_meets_combined_screen": preferred_is_eligible,
        "promotion_screen": {
            "candidate_passes_minimum_screen": not reasons,
            "automatic_promotion_enabled": False,
            "reasons_not_to_promote": reasons,
            "requirements": [
                "at least ten resolved response peaks",
                "at least fifteen unresolved censored lower bounds",
                "leave-one-event-out exact-event RMSE no more than 15 percent worse than exact-only amount model",
                "at least 75 percent of censored lower bounds satisfied",
                "best-performing feature set selected by out-of-sample validation",
                "manual engineering review and operational hindcast",
            ],
        },
        "all_candidates": candidates,
        "output_predictions_csv": str(predictions_path),
        "interpretation": (
            "Resolved peaks are treated as exact response targets. Unresolved peaks are "
            "included through a one-sided active-set squared-hinge penalty: predictions "
            "below their observed lower bounds are penalized, while predictions above "
            "those bounds are not treated as errors."
        ),
        "limitations": [
            "Censored lower bounds may still be inflated when the empirical recession baseline is imperfect.",
            "Historical spatial rainfall is 10 km and may not resolve small tributary basins well.",
            "The target is recession-equivalent stage delay rather than directly observed construction delay.",
            "The model remains shadow-only and cannot replace the live forecast automatically.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, default=safe_json))
    print(
        json.dumps(
            {
                "status": output["status"],
                "resolved_events": len(exact),
                "censored_events": len(censored),
                "baseline_rmse_days": baseline_rmse,
                "preferred_feature_set": preferred["feature_set"],
                "preferred_ridge_penalty": preferred["ridge_penalty"],
                "preferred_censored_weight": preferred["censored_weight"],
                "preferred_exact_rmse": preferred[
                    "exact_leave_one_event_out"
                ]["rmse_days"],
                "preferred_constraint_satisfaction_pct": preferred[
                    "censored_constraints"
                ]["constraint_satisfaction_pct"],
                "combined_screen_passed": preferred_is_eligible,
                "promotion_screen": output["promotion_screen"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
