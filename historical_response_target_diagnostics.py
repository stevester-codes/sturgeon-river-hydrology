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
    dynamic_baseline,
    historical_recession_fits,
    safe_json,
)
from historical_spatial_peak_reanalysis import (
    AMOUNT_FEATURES,
    SPATIAL_FEATURES,
    knn_predict,
    metrics,
    ridge_predict,
    transformed_features,
)

EVENTS_DEFAULT = Path(
    "output/historical_event_backfill/historical_spatial_events_peak_reanalysis.csv"
)
PAIRS_DEFAULT = Path("output/archive_probe/historical_rdpa_pairs.csv")
OUT_DEFAULT = Path(
    "output/historical_event_backfill/historical_response_target_diagnostics.json"
)
CSV_DEFAULT = Path(
    "output/historical_event_backfill/historical_response_target_diagnostics.csv"
)


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


def recession_rate(fit: dict, value: float) -> float:
    return min(
        -0.001,
        float(fit["intercept_per_day"])
        + float(fit["coefficient_per_day"]) * float(value),
    )


def exact_recession_equivalent_days(
    high_value: float,
    low_value: float,
    fit: dict,
    maximum_days: float = 120.0,
) -> float | None:
    if not np.isfinite(high_value) or not np.isfinite(low_value):
        return None
    if high_value <= low_value:
        return 0.0
    value = float(high_value)
    hours = 0
    maximum_hours = int(maximum_days * 24)
    while value > low_value and hours < maximum_hours:
        value += recession_rate(fit, value) / 24.0
        hours += 1
    return hours / 24.0 if value <= low_value else None


def event_targets(
    row: pd.Series,
    hourly: pd.DataFrame,
    q_fit: dict,
    stage_fit: dict,
) -> dict:
    start = pd.Timestamp(row.rain_start_utc)
    analysis_end = pd.Timestamp(row.analysis_end_utc)
    peak_value = row.get("recomputed_peak_utc")
    if pd.isna(peak_value):
        return {"target_status": "peak_time_unavailable"}
    peak_time = pd.Timestamp(peak_value)
    pre = hourly.index[hourly.index <= start]
    if not len(pre):
        return {"target_status": "pre_event_gauge_unavailable"}
    t0 = pre[-1]
    observed = hourly.loc[t0:analysis_end].dropna(
        subset=["stage_m", "discharge_m3s"]
    )
    if peak_time not in observed.index:
        nearest_index = observed.index.get_indexer([peak_time], method="nearest")[0]
        if nearest_index < 0:
            return {"target_status": "peak_gauge_value_unavailable"}
        peak_time = observed.index[nearest_index]
    q_base = dynamic_baseline(
        float(observed.discharge_m3s.iloc[0]), observed.index, q_fit
    )
    stage_base = dynamic_baseline(
        float(observed.stage_m.iloc[0]), observed.index, stage_fit
    )
    observed_q = float(observed.loc[peak_time, "discharge_m3s"])
    observed_stage = float(observed.loc[peak_time, "stage_m"])
    baseline_q = float(q_base.loc[peak_time])
    baseline_stage = float(stage_base.loc[peak_time])
    q_departure = observed_q - baseline_q
    stage_departure = observed_stage - baseline_stage
    q_rate = abs(recession_rate(q_fit, baseline_q))
    stage_rate = abs(recession_rate(stage_fit, baseline_stage))
    q_linear = q_departure / q_rate if q_rate > 0.001 else None
    stage_linear = (
        stage_departure / stage_rate if stage_rate > 0.001 else None
    )
    q_exact = exact_recession_equivalent_days(
        observed_q, baseline_q, q_fit
    )
    stage_exact = exact_recession_equivalent_days(
        observed_stage, baseline_stage, stage_fit
    )
    difference = (
        stage_exact - q_exact
        if stage_exact is not None and q_exact is not None
        else None
    )
    ratio = (
        stage_exact / q_exact
        if stage_exact is not None and q_exact is not None and q_exact > 0
        else None
    )
    agreement = bool(
        stage_exact is not None
        and q_exact is not None
        and abs(stage_exact - q_exact)
        <= max(2.0, 0.50 * max(stage_exact, q_exact))
    )
    return {
        "target_status": "evaluated",
        "target_peak_utc": peak_time.isoformat(),
        "observed_q_at_peak_m3s": observed_q,
        "baseline_q_at_peak_m3s": baseline_q,
        "q_departure_at_peak_m3s": q_departure,
        "observed_stage_at_peak_m": observed_stage,
        "baseline_stage_at_peak_m": baseline_stage,
        "stage_departure_at_peak_m": stage_departure,
        "q_linear_days_lost": q_linear,
        "stage_linear_days_lost": stage_linear,
        "q_exact_recession_equivalent_days": q_exact,
        "stage_exact_recession_equivalent_days": stage_exact,
        "stage_minus_q_exact_days": difference,
        "stage_to_q_exact_ratio": ratio,
        "stage_q_target_agreement": agreement,
    }


def loo_models(frame: pd.DataFrame, target: str) -> dict:
    work = transformed_features(frame).dropna(
        subset=SPATIAL_FEATURES + [target]
    )
    work = work[
        pd.to_numeric(work[target], errors="coerce").between(
            0.05, 60.0, inclusive="both"
        )
    ]
    result = {
        "target": target,
        "events": int(len(work)),
        "storm_type_count": int(work.storm_type.nunique()) if len(work) else 0,
        "models": {},
        "preferred_candidate": None,
    }
    if len(work) < 4:
        result["status"] = "insufficient_events"
        return result
    x_amount = work[AMOUNT_FEATURES].to_numpy(float)
    x_spatial = work[SPATIAL_FEATURES].to_numpy(float)
    y = pd.to_numeric(work[target], errors="coerce").to_numpy(float)
    y_log = np.log1p(np.clip(y, 0, None))
    predictions = {
        "median_baseline": [],
        "amount_only_log_ridge": [],
        "spatial_log_ridge": [],
        "spatial_log_knn": [],
    }
    metadata = []
    for index in range(len(work)):
        keep = np.arange(len(work)) != index
        predictions["median_baseline"].append(float(np.median(y[keep])))
        amount = ridge_predict(
            x_amount[keep], y_log[keep], x_amount[[index]], penalty=2.0
        )[0]
        spatial = ridge_predict(
            x_spatial[keep], y_log[keep], x_spatial[[index]], penalty=3.0
        )[0]
        knn = knn_predict(
            x_spatial[keep], y_log[keep], x_spatial[[index]], k=4
        )[0]
        predictions["amount_only_log_ridge"].append(
            float(max(0.0, np.expm1(amount)))
        )
        predictions["spatial_log_ridge"].append(
            float(max(0.0, np.expm1(spatial)))
        )
        predictions["spatial_log_knn"].append(
            float(max(0.0, np.expm1(knn)))
        )
        metadata.append(
            {
                "event_id": int(work.iloc[index].event_id),
                "storm_type": str(work.iloc[index].storm_type),
                "observed_days": float(y[index]),
            }
        )
    for name, values in predictions.items():
        prediction = np.asarray(values, dtype=float)
        score = metrics(y, prediction)
        score["predictions"] = [
            {
                **metadata[index],
                "predicted_days": float(prediction[index]),
            }
            for index in range(len(prediction))
        ]
        result["models"][name] = score
    candidates = (
        "amount_only_log_ridge",
        "spatial_log_ridge",
        "spatial_log_knn",
    )
    preferred = min(
        candidates,
        key=lambda name: result["models"][name]["rmse_days"],
    )
    result["preferred_candidate"] = preferred
    baseline_rmse = result["models"]["median_baseline"]["rmse_days"]
    amount_rmse = result["models"]["amount_only_log_ridge"]["rmse_days"]
    preferred_rmse = result["models"][preferred]["rmse_days"]
    result["skill_improvement_vs_median_pct"] = (
        (baseline_rmse - preferred_rmse) / baseline_rmse * 100.0
        if baseline_rmse > 0
        else None
    )
    result["skill_improvement_vs_amount_only_pct"] = (
        (amount_rmse - preferred_rmse) / amount_rmse * 100.0
        if amount_rmse > 0
        else None
    )
    result["status"] = "leave_one_event_out_complete"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--events", default=str(EVENTS_DEFAULT))
    parser.add_argument("--pairs", default=str(PAIRS_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument("--csv-output", default=str(CSV_DEFAULT))
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    if events.empty:
        raise RuntimeError("Peak-reanalysis event table is empty")
    hourly, retrieval = build_hourly(args.months)
    q_fit, stage_fit, recession_support = historical_recession_fits(
        Path(args.pairs)
    )
    records = []
    for _, row in events.iterrows():
        records.append(event_targets(row, hourly, q_fit, stage_fit))
    targets = pd.DataFrame(records, index=events.index)
    combined = pd.concat([events, targets], axis=1)
    resolved = combined[
        combined.eligible_for_peak_training.map(bool_value)
    ].copy()
    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)

    paired = resolved.dropna(
        subset=[
            "q_exact_recession_equivalent_days",
            "stage_exact_recession_equivalent_days",
        ]
    )
    q_values = pd.to_numeric(
        paired.q_exact_recession_equivalent_days, errors="coerce"
    ).to_numpy(float)
    stage_values = pd.to_numeric(
        paired.stage_exact_recession_equivalent_days, errors="coerce"
    ).to_numpy(float)
    correlation = (
        float(np.corrcoef(q_values, stage_values)[0, 1])
        if len(paired) >= 3 and np.std(q_values) > 0 and np.std(stage_values) > 0
        else None
    )
    agreement_count = int(
        paired.stage_q_target_agreement.map(bool_value).sum()
    ) if len(paired) else 0

    q_validation = loo_models(
        resolved, "q_exact_recession_equivalent_days"
    )
    stage_validation = loo_models(
        resolved, "stage_exact_recession_equivalent_days"
    )

    recommendation = {
        "status": "retain_shadow_only",
        "preferred_target_for_operational_compatibility": (
            "stage_exact_recession_equivalent_days"
        ),
        "reason": (
            "The live rainfall-response model is stage-based, so stage-space delay is "
            "the compatible target. Discharge-space delay remains an independent diagnostic."
        ),
        "automatic_promotion_enabled": False,
        "requirements_before_use": [
            "stage and discharge targets show acceptable event-level agreement",
            "stage-target model materially outperforms a median baseline",
            "spatial features materially outperform amount-only features",
            "censored lower-bound behaviour is acceptable",
            "manual engineering review and hindcast",
        ],
    }
    stage_preferred = stage_validation.get("preferred_candidate")
    stage_skill = finite(
        stage_validation.get("skill_improvement_vs_median_pct")
    )
    spatial_skill = finite(
        stage_validation.get("skill_improvement_vs_amount_only_pct")
    )
    reasons = []
    if len(paired) < 10:
        reasons.append("fewer_than_ten_paired_resolved_peak_events")
    if correlation is None or correlation < 0.70:
        reasons.append("stage_and_discharge_delay_targets_correlate_below_0_70")
    if agreement_count < max(1, math.ceil(0.70 * len(paired))):
        reasons.append("fewer_than_70_percent_of_events_have_stage_q_target_agreement")
    if stage_skill is None or stage_skill < 15.0:
        reasons.append("stage_target_model_improves_median_by_less_than_15_percent")
    if not str(stage_preferred).startswith("spatial"):
        reasons.append("spatial_model_is_not_best_for_stage_target")
    elif spatial_skill is None or spatial_skill < 10.0:
        reasons.append("spatial_stage_model_improves_amount_only_by_less_than_10_percent")
    recommendation["reasons_not_to_promote"] = reasons
    recommendation["candidate_passes_target_screen"] = not reasons

    output = {
        "generated_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "historical_response_target_diagnostics_complete",
        "mode": "shadow_only_no_automatic_promotion",
        "station": "05EA002",
        "resolved_peak_events": int(len(resolved)),
        "paired_target_events": int(len(paired)),
        "recession_baseline": {
            "support": recession_support,
            "discharge_fit": q_fit,
            "stage_fit": stage_fit,
        },
        "stage_discharge_target_comparison": {
            "correlation": correlation,
            "agreement_events": agreement_count,
            "agreement_fraction": (
                agreement_count / len(paired) if len(paired) else None
            ),
            "mean_stage_minus_q_days": (
                float(np.mean(stage_values - q_values)) if len(paired) else None
            ),
            "median_stage_minus_q_days": (
                float(np.median(stage_values - q_values)) if len(paired) else None
            ),
            "mean_absolute_difference_days": (
                float(np.mean(np.abs(stage_values - q_values)))
                if len(paired)
                else None
            ),
            "records": [
                {
                    "event_id": int(row.event_id),
                    "storm_type": str(row.storm_type),
                    "q_exact_days": float(row.q_exact_recession_equivalent_days),
                    "stage_exact_days": float(row.stage_exact_recession_equivalent_days),
                    "difference_days": float(row.stage_minus_q_exact_days),
                    "agreement": bool_value(row.stage_q_target_agreement),
                }
                for row in paired.itertuples(index=False)
            ],
        },
        "q_target_model_validation": q_validation,
        "stage_target_model_validation": stage_validation,
        "target_recommendation": recommendation,
        "wateroffice_retrieval_chunks": retrieval,
        "output_csv": str(csv_path),
        "limitations": [
            "Both targets depend on empirical precipitation-screened recession fits.",
            "The current WaterOffice discharge is rating-derived and provisional.",
            "Stage and discharge delay targets can diverge when the rating relationship changes.",
            "The historical spatial rainfall grids are 10 km.",
            "No automatic operational promotion is permitted.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, default=safe_json))
    print(
        json.dumps(
            {
                "status": output["status"],
                "paired_events": len(paired),
                "target_correlation": correlation,
                "agreement_fraction": output[
                    "stage_discharge_target_comparison"
                ]["agreement_fraction"],
                "q_preferred_model": q_validation.get("preferred_candidate"),
                "stage_preferred_model": stage_validation.get("preferred_candidate"),
                "recommendation": recommendation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
