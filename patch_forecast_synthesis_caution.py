#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


path = Path('forecast_synthesis.py')
text = path.read_text()

text = replace_once(
    text,
    'from datetime import date, datetime, timezone\n',
    'from datetime import date, datetime, timedelta, timezone\n',
    'timedelta import',
)

text = replace_once(
    text,
    '            date_item("contractor_site_recession", field_crossing, "field_check"),\n',
    '            date_item("contractor_site_rain_free_recession", field_crossing, "field_check"),\n',
    'field consensus label',
)

old = '''    else:
        shadow_status = "historical_response_shadow_unavailable"
        shadow_difference = None
        shadow_historical_days = None

    event_blocks = int(
'''
new = '''    else:
        shadow_status = "historical_response_shadow_unavailable"
        shadow_difference = None
        shadow_historical_days = None

    material_shadow_disagreement = bool(
        shadow_aligned
        and shadow_difference is not None
        and abs(shadow_difference) >= 2.0
    )
    official_p50_days = finite(
        nested(readiness, "probabilistic_exposure", "quantiles", "p50", "days")
    )
    shadow_sensitivity_days = (
        official_p50_days + shadow_difference
        if material_shadow_disagreement
        and official_p50_days is not None
        and shadow_difference is not None
        else None
    )
    shadow_sensitivity_date = (
        (generated + timedelta(days=shadow_sensitivity_days)).date().isoformat()
        if shadow_sensitivity_days is not None
        else None
    )
    if material_shadow_disagreement and overall_confidence == "moderate":
        overall_confidence = "low_to_moderate"

    event_blocks = int(
'''
text = replace_once(text, old, new, 'shadow disagreement handling')

text = replace_once(
    text,
    '        "The current-cycle historical response comparison becomes available and materially exceeds the official rain-response delay.",\n',
    '        "The current-cycle historical response disagreement persists, grows, or is later validated for operational use.",\n',
    'invalidation wording',
)

old = '''            "headline": (
                "Not ready; use the consensus inspection window and retain a separate schedule contingency date."
                if decision_status == "not_ready"
                else "Inspection window is approaching; field verification remains mandatory."
            ),
'''
new = '''            "headline": (
                "Not ready; timing methods agree on an inspection window, but the historical rain-response shadow is materially later. Retain the schedule contingency date."
                if decision_status == "not_ready" and material_shadow_disagreement
                else "Not ready; use the consensus inspection window and retain a separate schedule contingency date."
                if decision_status == "not_ready"
                else "Inspection window is approaching; field verification remains mandatory."
            ),
'''
text = replace_once(text, old, new, 'decision headline')

text = replace_once(
    text,
    '            "engineering_schedule_contingency_date": contingency_date,\n',
    '            "engineering_schedule_contingency_date": contingency_date,\n            "historical_response_shadow_sensitivity_date": shadow_sensitivity_date,\n',
    'working forecast shadow date',
)

text = replace_once(
    text,
    '                "historical_minus_official_days": shadow_difference,\n                "operational_effect": "none_shadow_only",\n',
    '                "historical_minus_official_days": shadow_difference,\n                "shadow_adjusted_threshold_sensitivity_days": shadow_sensitivity_days,\n                "shadow_adjusted_threshold_sensitivity_date": shadow_sensitivity_date,\n                "material_disagreement": material_shadow_disagreement,\n                "operational_effect": "none_shadow_only",\n',
    'shadow evidence fields',
)

text = replace_once(
    text,
    '                f"Screened direct-discharge validation contains {event_blocks} independent event blocks.",\n',
    '                f"Screened direct-discharge validation contains {event_blocks} independent event blocks.",\n                (\n                    f"The current-cycle historical rain-response shadow is {abs(shadow_difference):.2f} days later than the official response."\n                    if material_shadow_disagreement and shadow_difference is not None\n                    else "No material current-cycle historical rain-response disagreement is available."\n                ),\n',
    'confidence reason',
)

text = replace_once(
    text,
    '- Contractor-site recession projection: **{field_crossing or \'unavailable\'}**\n',
    '- Contractor-site rain-free recession projection: **{field_crossing or \'unavailable\'}**\n',
    'brief field label',
)

text = replace_once(
    text,
    '- Engineering schedule contingency: **{contingency_date or \'unavailable\'}** — sensitivity envelope, not a formal p90 probability.\n',
    '- Engineering schedule contingency: **{contingency_date or \'unavailable\'}** — sensitivity envelope, not a formal p90 probability.\n- Historical response shadow sensitivity: **{shadow_sensitivity_date or \'not applicable\'}** — shadow only and excluded from the consensus window.\n',
    'brief shadow date',
)

path.write_text(text)
print('Forecast synthesis now surfaces material current-cycle rain-response disagreement.')
