"""Runtime safeguards for source-ingestion scripts.

Python imports sitecustomize automatically during interpreter startup when this
repository is the working directory. These guards preserve the seeded last-valid
weather files if an ECCC directory listing is temporarily empty, including the
00 UTC /today/ directory rollover. Scientific age/provenance checks remain in
the downstream forecast code; this module only prevents good files from being
replaced by empty placeholders.
"""
from __future__ import annotations

import atexit
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("sturgeon_pipeline_output")
SPATIAL = ROOT / "spatial"
SCRIPT = Path(sys.argv[0]).name if sys.argv else ""
_BACKUP_DIR: Path | None = None
_BACKUPS: dict[Path, Path] = {}


def _backup(paths: list[Path]) -> None:
    global _BACKUP_DIR
    _BACKUP_DIR = Path(tempfile.mkdtemp(prefix="sturgeon_weather_guard_"))
    for source in paths:
        if source.exists() and source.stat().st_size > 0:
            destination = _BACKUP_DIR / source.name
            shutil.copy2(source, destination)
            _BACKUPS[source] = destination


def _empty(path: Path) -> bool:
    return not path.exists() or path.stat().st_size == 0


def _restore_empty_files(label: str, metadata_path: Path | None = None) -> None:
    restored: list[str] = []
    for destination, backup in _BACKUPS.items():
        if _empty(destination) and backup.exists() and backup.stat().st_size > 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, destination)
            restored.append(str(destination))
    if restored and metadata_path and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}
        metadata["last_valid_file_guard"] = {
            "status": "preserved_seeded_previous_files_after_empty_source_result",
            "script": label,
            "restored_files": restored,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation": (
                "The current source retrieval returned no rows, so seeded prior files "
                "were preserved. Downstream run-time and age checks determine whether "
                "they remain acceptable; missing data are not interpreted as zero rain."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
    if _BACKUP_DIR:
        shutil.rmtree(_BACKUP_DIR, ignore_errors=True)


if SCRIPT == "qpf_forecast_v2.py":
    deterministic = SPATIAL / "deterministic_qpf_by_subarea.csv"
    ensemble = SPATIAL / "ensemble_qpf_by_subarea.csv"
    metadata = SPATIAL / "qpf_v2.json"
    _backup([deterministic, ensemble])
    atexit.register(_restore_empty_files, SCRIPT, metadata)
elif SCRIPT == "medium_range_qpf.py":
    gdps_intervals = SPATIAL / "gdps_interval_qpf_by_subarea.csv"
    gdps_horizons = SPATIAL / "gdps_horizon_qpf_by_subarea.csv"
    geps = SPATIAL / "geps_qpf_by_subarea.csv"
    metadata = SPATIAL / "medium_range_qpf.json"
    _backup([gdps_intervals, gdps_horizons, geps])
    atexit.register(_restore_empty_files, SCRIPT, metadata)
