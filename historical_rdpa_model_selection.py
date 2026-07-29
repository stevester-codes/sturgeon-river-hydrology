#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PAIRING_DEFAULT = Path("output/archive_probe/historical_rdpa_pairing.json")
OUT_DEFAULT = Path("output/archive_probe/historical_rdpa_model_selection.json")
MIN_COVERAGE = 0.90
MIN_POINTS = 200
MIN_EVENTS = 3
MATERIAL_SKILL_GAIN_PCT = 5.0


def finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def candidate_record(name: str, model: dict, baseline_rmse: float | None) -> dict:
    projection = model.get("projection") or {}
    cv = model.get("event_block_cross_validation", {})
    aggregate = cv.get("aggregate") or {}
    rmse = finite(aggregate.get("rmse_per_day"))
    points = int(model.get("points", 0) or 0)
    events = int(model.get("events", 0) or 0)
    days = finite(projection.get("days")) if projection.get("reached") else None
    improvement = None
    if rmse is not None and baseline_rmse not in (None, 0):
        improvement = (baseline_rmse - rmse) / baseline_rmse * 100.0
    requirements = {
        "projection_reached": days is not None,
        "minimum_points": points >= MIN_POINTS,
        "minimum_events": events >= MIN_EVENTS,
        "event_block_cross_validation": rmse is not None,
        "does_not_worsen_baseline_rmse": (
            rmse is not None and baseline_rmse is not None and rmse <= baseline_rmse
        ),
    }
    return {
        "name": name,
        "points": points,
        "events": events,
        "rain_free_days_to_6_77_m3s": days,
        "fit": model.get("fit"),
        "event_block_cross_validation": cv,
        "event_block_rmse_per_day": rmse,
        "skill_improvement_vs_gauge_only_pct": improvement,
        "requirements": requirements,
        "passes_model_screen": all(requirements.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", default=str(PAIRING_DEFAULT))
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    pairing_path = Path(args.pairing)
    output_path = Path(args.output)
    if not pairing_path.exists():
        raise FileNotFoundError(pairing_path)
    pairing = json.loads(pairing_path.read_text())
    if pairing.get("status") != "historical_rdpa_pairing_complete":
        raise RuntimeError("Historical RDPA pairing is incomplete")

    coverage = finite(pairing.get("rdpa_retrieval", {}).get("coverage_fraction"), 0.0)
    models = pairing.get("models", {})
    gauge = models.get("gauge_only", {})
    gauge_rmse = finite(
        gauge.get("event_block_cross_validation", {})
        .get("aggregate", {})
        .get("rmse_per_day")
    )
    candidates = {
        name: candidate_record(name, models.get(name, {}), gauge_rmse)
        for name in ("rdpa_strict", "rdpa_moderate")
    }
    eligible = [
        candidate
        for candidate in candidates.values()
        if candidate["passes_model_screen"] and coverage >= MIN_COVERAGE
    ]
    preferred = min(
        eligible,
        key=lambda item: (
            float(item["event_block_rmse_per_day"]),
            -int(item["events"]),
            -int(item["points"]),
        ),
        default=None,
    )

    status = (
        "screened_candidate_ready_for_operational_sensitivity"
        if preferred is not None
        else "no_rdpa_screened_candidate_passed"
    )
    promotion_reasons: list[str] = []
    if coverage < MIN_COVERAGE:
        promotion_reasons.append("archived_rdpa_coverage_below_90_percent")
    if preferred is None:
        promotion_reasons.append("no_screened_candidate_met_minimum_model_requirements")
    else:
        improvement = finite(preferred.get("skill_improvement_vs_gauge_only_pct"), 0.0)
        if improvement < MATERIAL_SKILL_GAIN_PCT:
            promotion_reasons.append("event_block_skill_gain_is_less_than_5_percent")
        if int(preferred.get("events", 0)) < 10:
            promotion_reasons.append("fewer_than_10_independent_dry_event_blocks")
        promotion_reasons.extend(
            [
                "10_km_rdpa_screen_may_miss_sub_grid_convective_rain",
                "wsc_discharge_is_provisional_and_rating_derived",
                "manual_engineering_review_is_required",
            ]
        )

    generated = datetime.now(timezone.utc).replace(microsecond=0)
    output = {
        "generated_utc": generated.isoformat(),
        "status": status,
        "mode": "shadow_only_manual_promotion_required",
        "station": pairing.get("station"),
        "target_discharge_m3s": pairing.get("target_discharge_m3s"),
        "rdpa_coverage_fraction": coverage,
        "gauge_only_reference": {
            "points": gauge.get("points"),
            "events": gauge.get("events"),
            "rain_free_days_to_6_77_m3s": finite((gauge.get("projection") or {}).get("days")),
            "event_block_cross_validation": gauge.get("event_block_cross_validation", {}),
            "event_block_rmse_per_day": gauge_rmse,
        },
        "screened_candidates": candidates,
        "preferred_screened_candidate": preferred,
        "selection_rule": {
            "minimum_rdpa_coverage": MIN_COVERAGE,
            "minimum_points": MIN_POINTS,
            "minimum_events": MIN_EVENTS,
            "candidate_must_not_worsen_gauge_only_event_block_rmse": True,
            "ranking": "lowest event-block RMSE, then more events, then more points",
        },
        "operational_use": {
            "replace_live_central_forecast": False,
            "include_as_independent_schedule_sensitivity": preferred is not None,
            "reason": (
                "The selected RDPA-screened direct-discharge model is a more defensible independent sensitivity than the unscreened gauge-only fit, but its skill gain and event diversity are not sufficient for automatic operational promotion."
                if preferred is not None
                else "No precipitation-screened direct-discharge model passed the minimum evidence screen."
            ),
        },
        "promotion_recommendation": {
            "promote_now": False,
            "reasons_not_to_promote": promotion_reasons,
            "next_evidence_needed": [
                "additional independent summer falling-limb recession blocks",
                "surveyed RS18883 WSE observations near 650.20 m",
                "material out-of-sample skill improvement over the live recession model",
            ],
        },
        "interpretation": (
            "Use the preferred precipitation-screened model as a shadow direct-discharge schedule sensitivity. Keep the existing stage-recession model as the official central forecast and require manual review before any promotion."
            if preferred is not None
            else "Retain the gauge-only direct-discharge result as an unscreened sensitivity and do not promote it."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": status,
                "coverage": coverage,
                "gauge_rmse": gauge_rmse,
                "preferred_candidate": (
                    preferred.get("name") if preferred is not None else None
                ),
                "preferred_days": (
                    preferred.get("rain_free_days_to_6_77_m3s")
                    if preferred is not None
                    else None
                ),
                "preferred_skill_improvement_pct": (
                    preferred.get("skill_improvement_vs_gauge_only_pct")
                    if preferred is not None
                    else None
                ),
                "promote_now": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
