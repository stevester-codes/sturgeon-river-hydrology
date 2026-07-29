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


if __name__ == "__main__":
    geps_ensemble_impacts.main()
