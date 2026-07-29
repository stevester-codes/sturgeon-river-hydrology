#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

PATH = Path("sturgeon_pipeline_output/calibration_v2/event_response_v2.csv")
BOOLEAN_COLUMNS = [
    "response_censored",
    "eligible_for_peak_training",
    "eligible_for_recovery_training",
]


def parse_bool(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) != 0)
    return int(str(value).strip().lower() in {"true", "1", "yes", "y"})


def main() -> None:
    frame = pd.read_csv(PATH)
    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(parse_bool).astype(int)
    frame.to_csv(PATH, index=False)
    print({column: frame[column].value_counts().to_dict() for column in BOOLEAN_COLUMNS if column in frame.columns})


if __name__ == "__main__":
    main()
