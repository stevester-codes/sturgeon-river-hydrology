#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one match, found {text.count(old)}")
    return text.replace(old, new, 1)


# One-time integration patch. The workflow removes no production evidence and
# fails before committing if any expected source block has changed unexpectedly.

# ---------------------------------------------------------------------------
# operational_readiness.py
# ---------------------------------------------------------------------------
path = Path("operational_readiness.py")
text = path.read_text()
text = replace_once(
    text,
    "import hysteresis_diagnostics\nimport uncertainty_sensitivity\n",
    "import hysteresis_diagnostics\nimport project_site_recession_shadow\nimport uncertainty_sensitivity\n",
    "operational import",
)
text = replace_once(
    text,
    'DISCHARGE_CANDIDATE = ROOT / "diagnostics" / "discharge_recession_candidate.json"\nOUT = ROOT / "forecast_v2" / "construction_readiness.json"\n',
    'DISCHARGE_CANDIDATE = ROOT / "diagnostics" / "discharge_recession_candidate.json"\nPROJECT_SITE_SHADOW = ROOT / "diagnostics" / "project_site_recession_shadow.json"\nOUT = ROOT / "forecast_v2" / "construction_readiness.json"\n',
    "operational constant",
)
text = replace_once(
    text,
    "    discharge_recession_candidate.main()\n    for path in (CALIBRATION_HEALTH, HYSTERESIS, UNCERTAINTY, DISCHARGE_CANDIDATE):\n        if not path.exists():\n            raise RuntimeError(f\"Required diagnostic output was not generated: {path}\")\n    health = json.loads(CALIBRATION_HEALTH.read_text())\n    hysteresis = json.loads(HYSTERESIS.read_text())\n    uncertainty = json.loads(UNCERTAINTY.read_text())\n    discharge_candidate = json.loads(DISCHARGE_CANDIDATE.read_text())\n",
    "    discharge_recession_candidate.main()\n    project_site_recession_shadow.main()\n    for path in (\n        CALIBRATION_HEALTH,\n        HYSTERESIS,\n        UNCERTAINTY,\n        DISCHARGE_CANDIDATE,\n        PROJECT_SITE_SHADOW,\n    ):\n        if not path.exists():\n            raise RuntimeError(f\"Required diagnostic output was not generated: {path}\")\n    health = json.loads(CALIBRATION_HEALTH.read_text())\n    hysteresis = json.loads(HYSTERESIS.read_text())\n    uncertainty = json.loads(UNCERTAINTY.read_text())\n    discharge_candidate = json.loads(DISCHARGE_CANDIDATE.read_text())\n    project_site_shadow = json.loads(PROJECT_SITE_SHADOW.read_text())\n",
    "operational diagnostics",
)
text = replace_once(
    text,
    "    current_wse = finite(current.get(\"estimated_starkey_wse_m\"))\n    depth_main = finite(current.get(\"depth_over_main_floodplain_m\"))\n",
    "    design_profile_wse = finite(current.get(\"estimated_starkey_wse_m\"))\n    design_profile_depth_main = finite(current.get(\"depth_over_main_floodplain_m\"))\n    site_state = project_site_shadow.get(\"current_site_state\", {})\n    provisional_date_wse = finite(site_state.get(\"date_recession_estimated_project_wse_m\"))\n    provisional_date_depth = finite(site_state.get(\"date_recession_estimated_depth_over_650_20_m\"))\n    provisional_q_wse = finite(site_state.get(\"discharge_relation_estimated_project_wse_m\"))\n    provisional_q_depth = finite(site_state.get(\"discharge_relation_estimated_depth_over_650_20_m\"))\n",
    "operational current vars",
)
text = replace_once(
    text,
    '            "estimated_starkey_wse_m": current_wse,\n            "estimated_project_wse_m": current.get(\n                "estimated_project_wse_m", current_wse\n            ),\n            "estimated_starkey_wse_range_m": current.get(\n                "estimated_starkey_wse_range_m"\n            ),\n            "estimated_depth_over_main_floodplain_m": depth_main,\n            "estimated_depth_over_low_pocket_m": current.get(\n                "depth_over_low_pocket_m"\n            ),\n',
    '            "estimated_starkey_wse_m": None,\n            "estimated_project_wse_m": None,\n            "estimated_starkey_wse_range_m": None,\n            "estimated_depth_over_main_floodplain_m": None,\n            "estimated_depth_over_low_pocket_m": None,\n            "actual_2026_site_wse_status": "not_directly_observed_current_value; steady_state_design_profile_conflicted_by_field_evidence",\n            "steady_state_design_profile_equivalent_wse_m": design_profile_wse,\n            "steady_state_design_profile_equivalent_wse_range_m": current.get(\n                "estimated_starkey_wse_range_m"\n            ),\n            "steady_state_design_profile_equivalent_depth_over_650_20_m": design_profile_depth_main,\n            "provisional_field_recession_estimated_project_wse_m": provisional_date_wse,\n            "provisional_field_recession_estimated_depth_over_650_20_m": provisional_date_depth,\n            "provisional_field_discharge_relation_estimated_project_wse_m": provisional_q_wse,\n            "provisional_field_discharge_relation_estimated_depth_over_650_20_m": provisional_q_depth,\n            "site_wse_interpretation": (\n                "The steady-state flood-study transfer is retained as design-event context and threshold translation, "\n                "not as the actual 2026 construction-site water surface. Contractor observations support separate "\n                "provisional date-recession and discharge-relation estimates until datum, exact location and time are verified."\n            ),\n',
    "operational current conditions",
)
text = replace_once(
    text,
    '        "secondary_field_observations": {\n',
    '        "project_site_field_evidence": {\n            "status": project_site_shadow.get("status"),\n            "mode": project_site_shadow.get("mode"),\n            "observations": project_site_shadow.get("observations", []),\n            "date_recession_fit": project_site_shadow.get("date_recession_fit"),\n            "discharge_wse_fit": project_site_shadow.get("discharge_wse_fit"),\n            "current_site_state": project_site_shadow.get("current_site_state", {}),\n            "comparison_with_design_profile": project_site_shadow.get(\n                "comparison_with_design_profile", {}\n            ),\n            "operational_interpretation": project_site_shadow.get(\n                "operational_interpretation"\n            ),\n            "limitations": project_site_shadow.get("limitations", []),\n        },\n        "secondary_field_observations": {\n',
    "field evidence block",
)
text = replace_once(
    text,
    '            "concurrent_surveyed_triple_available": False,\n',
    '            "concurrent_surveyed_triple_available": False,\n            "contractor_date_level_site_wse_observations_m": {\n                "2026-07-16": 651.748,\n                "2026-07-23": 651.336,\n            },\n',
    "secondary observations",
)
text = replace_once(
    text,
    '            "status": "not_ready" if depth_main is None or depth_main > 0 else "inspect_now",\n',
    '            "status": (\n                "not_ready"\n                if provisional_date_depth is None or provisional_date_depth > 0\n                else "inspect_now"\n            ),\n',
    "decision status",
)
text = replace_once(
    text,
    '            "release_rule": "Release floodplain work only after the estimated WSE is at or below 650.20 m and a direct project-site inspection confirms drainage, access bearing and no renewed rise.",\n',
    '            "release_rule": "Release floodplain work only after a verified current project-site survey or direct drainage inspection confirms the work area is at or below 650.20 m, access has adequate bearing, and no renewed rise is forecast. Do not release from the steady-state design-profile equivalent WSE alone.",\n',
    "release rule",
)
text = replace_once(
    text,
    '            "Construction release remains a field decision, not an automatic model output.",\n',
    '            "Construction release remains a field decision, not an automatic model output.",\n            "Contractor-reported July 16 and July 23 site elevations are provisional because exact survey time, datum, method and shot location remain undocumented.",\n            "The steady-state RS18883 flood-study curve materially underpredicts the two provisional 2026 site observations and is no longer presented as actual current site depth.",\n',
    "operational limitations",
)
path.write_text(text)


# ---------------------------------------------------------------------------
# calibration_health.py
# ---------------------------------------------------------------------------
path = Path("calibration_health.py")
text = path.read_text()
text = replace_once(
    text,
    'HISTORICAL_SELECTION = Path("output/archive_probe/historical_rdpa_model_selection.json")\nOUT = ROOT / "diagnostics" / "calibration_health.json"\n',
    'HISTORICAL_SELECTION = Path("output/archive_probe/historical_rdpa_model_selection.json")\nPROVISIONAL_SITE_OBSERVATIONS = Path("project_site_date_observations.csv")\nOUT = ROOT / "diagnostics" / "calibration_health.json"\n',
    "health constant",
)
text = replace_once(
    text,
    '    if len(design_points) >= 10 and target_q is not None:\n        status, score = "approximate_low_flow_anchor_plus_complete_design_profile", 8.0\n    elif len(design_points) >= 3:\n        status, score = "approximate_low_flow_anchor_plus_partial_design_profile", 6.0\n    else:\n        status, score = "sparse_project_transfer_support", 3.0\n',
    '    provisional_site_count = 0\n    if PROVISIONAL_SITE_OBSERVATIONS.exists():\n        provisional_site_count = max(\n            0,\n            len([line for line in PROVISIONAL_SITE_OBSERVATIONS.read_text().splitlines() if line.strip()]) - 1,\n        )\n    if provisional_site_count >= 2:\n        status, score = "complete_design_profile_conflicted_by_provisional_2026_site_observations", 4.0\n    elif len(design_points) >= 10 and target_q is not None:\n        status, score = "approximate_low_flow_anchor_plus_complete_design_profile", 8.0\n    elif len(design_points) >= 3:\n        status, score = "approximate_low_flow_anchor_plus_partial_design_profile", 6.0\n    else:\n        status, score = "sparse_project_transfer_support", 3.0\n',
    "health transfer score",
)
text = replace_once(
    text,
    '        "design_profile_point_count": len(design_points),\n',
    '        "design_profile_point_count": len(design_points),\n        "provisional_2026_site_observation_count": provisional_site_count,\n',
    "health observation count",
)
text = replace_once(
    text,
    '        "interpretation": (\n            "The high-flow RS18883 transfer is constrained by the complete 2- to "\n            "1,000-year design profile. Remaining transfer uncertainty is concentrated "\n            "in the reconstructed 6.77 m3/s low-flow anchor and the segment to 14 m3/s."\n        ),\n',
    '        "interpretation": (\n            "The complete flood-study profile remains useful for design-event context and threshold translation. "\n            "However, two provisional contractor site elevations are materially above the steady-state profile at "\n            "comparable 2026 discharge, so the curve is not considered reliable for actual current site depth until "\n            "datum, location and timing are verified and the hydraulic anomaly is resolved."\n        ),\n',
    "health interpretation",
)
path.write_text(text)


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------
path = Path(".github/workflows/sturgeon-operational.yml")
text = path.read_text()
text = replace_once(
    text,
    "      - 'operational_readiness.py'\n",
    "      - 'operational_readiness.py'\n      - 'project_site_recession_shadow.py'\n      - 'project_site_date_observations.csv'\n",
    "workflow triggers",
)
text = replace_once(
    text,
    "            operational_readiness.py \\\n            calibration_health.py \\\n",
    "            operational_readiness.py \\\n            project_site_recession_shadow.py \\\n            calibration_health.py \\\n",
    "workflow compile",
)
path.write_text(text)

print("Project-site field evidence patch applied successfully.")
