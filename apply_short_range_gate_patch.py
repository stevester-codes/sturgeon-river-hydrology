#!/usr/bin/env python3
from pathlib import Path

path = Path("calibration_health.py")
text = path.read_text()
needle = '\n\nif __name__ == "__main__":\n    main()\n'
insertion = '''
    if context.get("status") in {
        "short_range_forecast_unavailable",
        "short_range_feature_vector_incomplete",
    }:
        raise RuntimeError(
            "Complete HRDPS 48-hour input is unavailable or incomplete; "
            "the operational package must be degraded rather than treating the short-range delay as zero."
        )


if __name__ == "__main__":
    main()
'''
if needle not in text:
    raise SystemExit("Expected calibration_health.py tail not found")
updated = text.replace(needle, "\n" + insertion, 1)
path.write_text(updated)
print("Patched calibration_health.py with a hard short-range input gate")
