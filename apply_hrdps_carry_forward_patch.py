#!/usr/bin/env python3
from pathlib import Path

path = Path("forecast_impacts_v2.py")
text = path.read_text()

needle_state = '''    previous_path = OUT / "forecast_impacts_v2.json"
    previous_result = {}
    if previous_path.exists():
        try:
            previous_result = json.loads(previous_path.read_text())
        except Exception:
            previous_result = {}
    base = json.loads(BASE_CAL.read_text())
'''
replacement_state = '''    previous_path = OUT / "forecast_impacts_v2.json"
    state_path = OUT / "last_valid_hrdps.json"
    previous_result = {}
    state_result = {}
    if previous_path.exists():
        try:
            previous_result = json.loads(previous_path.read_text())
        except Exception:
            previous_result = {}
    if state_path.exists():
        try:
            state_result = json.loads(state_path.read_text())
        except Exception:
            state_result = {}
    base = json.loads(BASE_CAL.read_text())
'''
if needle_state not in text:
    raise SystemExit("state insertion marker not found")
text = text.replace(needle_state, replacement_state, 1)

needle_previous = '''    if not current_complete_48:
        previous_scenarios = previous_result.get("deterministic_scenarios", [])
        previous_hrdps = [
'''
replacement_previous = '''    if not current_complete_48:
        fallback_result = previous_result
        previous_candidates = fallback_result.get("deterministic_scenarios", [])
        if not any(
            str(row.get("model")) == "HRDPS"
            and int(row.get("horizon_h", 0) or 0) == 48
            and bool(row.get("complete_horizon"))
            for row in previous_candidates
        ):
            fallback_result = state_result
        previous_scenarios = fallback_result.get("deterministic_scenarios", [])
        previous_hrdps = [
'''
if needle_previous not in text:
    raise SystemExit("fallback selection marker not found")
text = text.replace(needle_previous, replacement_previous, 1)

needle_generated = '''        previous_generated = previous_result.get("generated_utc")
'''
replacement_generated = '''        previous_generated = fallback_result.get("generated_utc")
'''
if needle_generated not in text:
    raise SystemExit("fallback timestamp marker not found")
text = text.replace(needle_generated, replacement_generated, 1)

needle_before_reps = '''
    reps_status = reps_validation()
'''
replacement_before_reps = '''
    if current_complete_48:
        state_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "deterministic_scenarios": [
                row
                for row in scenarios
                if str(row.get("model")) == "HRDPS"
                and int(row.get("horizon_h", 0) or 0) in {24, 48}
                and bool(row.get("complete_horizon"))
            ],
        }
        state_path.write_text(json.dumps(state_payload, indent=2, default=json_default))

    reps_status = reps_validation()
'''
if needle_before_reps not in text:
    raise SystemExit("state-write marker not found")
text = text.replace(needle_before_reps, replacement_before_reps, 1)

path.write_text(text)
print("Extended forecast_impacts_v2.py with persistent last-valid HRDPS state")
