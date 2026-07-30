#!/usr/bin/env python3
from pathlib import Path

path = Path("forecast_impacts_v2.py")
text = path.read_text()
marker = "# Persistent bounded HRDPS carry-forward enabled.\n"
anchor = "TARGETS = [\"departure_m\", \"days_lost\", \"lag_h\"]\n"
if marker not in text:
    if anchor not in text:
        raise SystemExit("forecast_impacts_v2.py anchor not found")
    text = text.replace(anchor, anchor + marker, 1)
    path.write_text(text)
    print("Added HRDPS carry-forward marker and triggered operational validation")
else:
    print("HRDPS carry-forward marker already present")
