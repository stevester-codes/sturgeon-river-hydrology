#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

ROOT = Path("sturgeon_pipeline_output")
SPATIAL = ROOT / "spatial"
CACHE = Path("grid_cache") / "qpf_v2"
SUBAREAS = SPATIAL / "derived_subareas.geojson"
META = SPATIAL / "qpf_v2.json"
OUT = SPATIAL / "deterministic_qpf_horizon_by_subarea.csv"
THRESHOLDS = [5, 10, 20, 30, 50]


def clip(path: Path, geometry, geometry_crs) -> np.ma.MaskedArray:
    with rasterio.open(path) as dataset:
        projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(dataset.crs).iloc[0]
        values, _ = mask(dataset, [mapping(projected)], crop=True, filled=False, indexes=1)
        array = np.ma.asarray(values, dtype=float)
        return np.ma.masked_where(~np.isfinite(array) | (array < -0.01) | (array > 1000), array)


def summary(values: np.ma.MaskedArray) -> dict:
    compressed = values.compressed()
    if not len(compressed):
        return {"mean_mm": None, "max_mm": None, "valid_cells": 0, **{f"pct_gt_{threshold}mm": None for threshold in THRESHOLDS}}
    result = {"mean_mm": float(np.mean(compressed)), "max_mm": float(np.max(compressed)), "valid_cells": int(len(compressed))}
    for threshold in THRESHOLDS:
        result[f"pct_gt_{threshold}mm"] = float(np.mean(compressed > threshold) * 100)
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    metadata = json.loads(META.read_text())
    subareas = gpd.read_file(SUBAREAS)
    rows: list[dict] = []
    warnings: list[str] = []
    for model_meta in metadata.get("deterministic_models", []):
        model = model_meta.get("model")
        if model_meta.get("status") != "processed":
            continue
        candidates = model_meta.get("candidates", {})
        available = []
        for hour_text, filename in candidates.items():
            if not filename:
                continue
            hour = int(hour_text)
            path = CACHE / str(model).lower() / filename
            if path.exists():
                available.append((hour, path, filename))
            else:
                warnings.append(f"Missing cached {model} file: {filename}")
        available.sort()
        max_hour = max((hour for hour, _, _ in available), default=0)
        for horizon in [24, 48, 72, 84]:
            if horizon > max_hour:
                continue
            selected = [(hour, path, name) for hour, path, name in available if hour <= horizon]
            expected = horizon // 6
            if len(selected) != expected:
                warnings.append(f"{model} horizon {horizon}: expected {expected} intervals, found {len(selected)}")
            for _, area in subareas.iterrows():
                total = None
                used = []
                for hour, path, name in selected:
                    try:
                        array = clip(path, area.geometry, subareas.crs)
                        if total is None:
                            total = array.copy()
                        elif total.shape == array.shape:
                            total = np.ma.add(total, array)
                        else:
                            warnings.append(f"{model} {name} {area.subarea}: grid shape mismatch")
                            continue
                        used.append(name)
                    except Exception as exc:
                        warnings.append(f"{model} {name} {area.subarea}: {exc}")
                if total is None:
                    continue
                row = {
                    "model": model,
                    "run_time_utc": model_meta.get("run_time_utc"),
                    "horizon_h": horizon,
                    "subarea": area.subarea,
                    "intervals_expected": expected,
                    "intervals_used": len(used),
                    "complete_horizon": len(used) == expected,
                    "source_files": "|".join(used),
                }
                row.update(summary(total))
                rows.append(row)
    write_csv(rows, OUT)
    result = {"rows": len(rows), "warnings": len(warnings), "warning_messages": warnings}
    (SPATIAL / "qpf_horizon_aggregation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
