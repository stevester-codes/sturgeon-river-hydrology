#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("output/historical_event_backfill")
ARCHIVE = Path("output/archive_probe")


def run(name: str, command: list[str], log_name: str, exit_name: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / log_name
    print(f"\n=== {name} ===")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    (ROOT / exit_name).write_text(f"{code}\n")
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def validate() -> dict:
    checks = {
        "backfill": (
            ROOT / "historical_spatial_event_backfill.json",
            "status",
            "historical_spatial_event_backfill_complete",
        ),
        "peak": (
            ROOT / "historical_spatial_peak_reanalysis.json",
            "status",
            "historical_spatial_peak_reanalysis_complete",
        ),
        "target": (
            ROOT / "historical_response_target_diagnostics.json",
            "status",
            "historical_response_target_diagnostics_complete",
        ),
        "censored": (
            ROOT / "historical_censored_response_model.json",
            "status",
            "historical_censored_response_model_evaluated",
        ),
    }
    result = {}
    for name, (path, key, expected) in checks.items():
        if not path.exists():
            raise RuntimeError(f"missing required historical output: {path}")
        data = load(path)
        if data.get(key) != expected:
            raise RuntimeError(
                f"{name} status is {data.get(key)!r}; expected {expected!r}"
            )
        result[name] = {"path": str(path), "status": data.get(key)}

    censored = load(ROOT / "historical_censored_response_model.json")
    if censored.get("mode") != "shadow_only_manual_promotion_required":
        raise RuntimeError("historical response model is not shadow-only")
    if censored.get("promotion_screen", {}).get("automatic_promotion_enabled") is not False:
        raise RuntimeError("automatic promotion is not explicitly disabled")
    return result


def main() -> None:
    required = [
        ARCHIVE / "historical_rdpa_pairing.json",
        ARCHIVE / "historical_rdpa_pairs.csv",
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"missing archive prerequisite: {path}")

    # Rebuild the directory once, then preserve every downstream product in it.
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in ROOT.iterdir():
        if path.is_file():
            path.unlink()

    run(
        "historical spatial event backfill",
        [
            "python",
            "historical_spatial_event_backfill.py",
            "--months",
            "18",
            "--workers",
            "4",
            "--pairing",
            str(ARCHIVE / "historical_rdpa_pairing.json"),
            "--pairs",
            str(ARCHIVE / "historical_rdpa_pairs.csv"),
            "--basin-cache",
            "archive_cache/historical_rdpa_05EA002.csv",
            "--grid-cache",
            "archive_cache/historical_rdpa_spatial_grids",
            "--output",
            str(ROOT / "historical_spatial_event_backfill.json"),
            "--events-output",
            str(ROOT / "historical_spatial_events.csv"),
            "--model-output",
            str(ROOT / "historical_spatial_response_model.json"),
        ],
        "workflow.log",
        "workflow.exitcode",
    )

    run(
        "historical peak-response reanalysis",
        [
            "python",
            "historical_spatial_peak_reanalysis.py",
            "--months",
            "18",
            "--events",
            str(ROOT / "historical_spatial_events.csv"),
            "--pairs",
            str(ARCHIVE / "historical_rdpa_pairs.csv"),
            "--output",
            str(ROOT / "historical_spatial_peak_reanalysis.json"),
            "--augmented-output",
            str(ROOT / "historical_spatial_events_peak_reanalysis.csv"),
            "--model-output",
            str(ROOT / "historical_spatial_peak_response_model.json"),
        ],
        "peak_reanalysis.log",
        "peak_reanalysis.exitcode",
    )

    run(
        "historical response-target diagnostics",
        [
            "python",
            "historical_response_target_diagnostics.py",
            "--months",
            "18",
            "--events",
            str(ROOT / "historical_spatial_events_peak_reanalysis.csv"),
            "--pairs",
            str(ARCHIVE / "historical_rdpa_pairs.csv"),
            "--output",
            str(ROOT / "historical_response_target_diagnostics.json"),
            "--csv-output",
            str(ROOT / "historical_response_target_diagnostics.csv"),
        ],
        "target_diagnostics.log",
        "target_diagnostics.exitcode",
    )

    run(
        "historical censored-response model",
        [
            "python",
            "historical_censored_response_model.py",
            "--input",
            str(ROOT / "historical_response_target_diagnostics.csv"),
            "--output",
            str(ROOT / "historical_censored_response_model.json"),
            "--predictions-output",
            str(ROOT / "historical_censored_response_predictions.csv"),
        ],
        "censored_response.log",
        "censored_response.exitcode",
    )

    checks = validate()
    summary = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "historical_response_refresh_complete",
        "mode": "shadow_only_no_effect_on_operational_forecast",
        "checks": checks,
        "sequence": [
            "spatial_event_backfill",
            "peak_response_reanalysis",
            "response_target_diagnostics",
            "censored_response_model",
        ],
    }
    (ROOT / "historical_response_refresh_status.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
