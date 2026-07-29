#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd

_original_dumps = json.dumps


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _safe_dumps(obj, *args, **kwargs):
    kwargs.setdefault("default", _json_default)
    return _original_dumps(obj, *args, **kwargs)


json.dumps = _safe_dumps

import geps_ensemble_impacts  # noqa: E402

# Re-export the implementation helpers used by downstream operational scripts.
# This wrapper exists to install safe JSON serialization before the GEPS module
# is imported, so consumers should be able to import from the wrapper without
# depending on private implementation details.
REQUIRED_SUBAREAS = geps_ensemble_impacts.REQUIRED_SUBAREAS
WINDOW_ENDPOINTS = geps_ensemble_impacts.WINDOW_ENDPOINTS
load_geps_members = geps_ensemble_impacts.load_geps_members
member_value = geps_ensemble_impacts.member_value
short_range_delay = geps_ensemble_impacts.short_range_delay
window_prediction = geps_ensemble_impacts.window_prediction


if __name__ == "__main__":
    geps_ensemble_impacts.main()
