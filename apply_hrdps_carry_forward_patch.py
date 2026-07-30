#!/usr/bin/env python3
from pathlib import Path

path = Path("forecast_impacts_v2.py")
text = path.read_text()

needle_start = '''def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CAL.read_text())
'''
replacement_start = '''def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    previous_path = OUT / "forecast_impacts_v2.json"
    previous_result = {}
    if previous_path.exists():
        try:
            previous_result = json.loads(previous_path.read_text())
        except Exception:
            previous_result = {}
    base = json.loads(BASE_CAL.read_text())
'''
if needle_start not in text:
    raise SystemExit("main start marker not found")
text = text.replace(needle_start, replacement_start, 1)

needle_fallback = '''    reps_status = reps_validation()
    storm_type_counts = {
'''
replacement_fallback = '''    current_complete_48 = any(
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
'''
if needle_fallback not in text:
    raise SystemExit("fallback insertion marker not found")
text = text.replace(needle_fallback, replacement_fallback, 1)

needle_result = '''        "reps_validation": reps_status,
        "deterministic_scenarios": scenarios,
'''
replacement_result = '''        "reps_validation": reps_status,
        "short_range_input_provenance": short_range_provenance,
        "deterministic_scenarios": scenarios,
'''
if needle_result not in text:
    raise SystemExit("result marker not found")
text = text.replace(needle_result, replacement_result, 1)

path.write_text(text)
print("Patched forecast_impacts_v2.py with bounded last-valid HRDPS carry-forward")
