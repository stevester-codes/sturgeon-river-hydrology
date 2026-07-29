#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.mask import mask
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import mapping
from shapely.ops import unary_union
from urllib3.util.retry import Retry

ROOT = Path("sturgeon_pipeline_output")
OUT = ROOT / "spatial"
CACHE = Path("grid_cache")
BASIN_ZIP = CACHE / "basins" / "MDA_ADP_05.zip"
BASIN_URL = "https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/shp/MDA_ADP_05.zip"
TARGET_STATIONS = ["05EA002", "05EA005", "05EA006", "05EA004", "05EA010", "05EA011", "05EA012"]
THRESHOLDS = [5, 10, 20, 30, 50]
HREF = re.compile(r'href="([^"?]+)"', re.I)
HRDPA_RE = re.compile(r"(\d{8})T(\d{2})Z_MSC_HRDPA_APCP-Accum6h_Sfc_RLatLon0\.0225_PT0H\.grib2$")
FORECAST_H_RE = re.compile(r"_PT(\d{3})H\.grib2$")


def http() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "sturgeon-river-hydrology/2.0 (stevester-codes@users.noreply.github.com)"
    retry = Retry(total=4, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504], allowed_methods=["GET"], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def get_text(s: requests.Session, url: str) -> str:
    r = s.get(url, timeout=90)
    r.raise_for_status()
    return r.text


def download(s: requests.Session, url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    with s.get(url, timeout=300, stream=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)
    return path


def station_column(gdf: gpd.GeoDataFrame) -> str:
    exact = [c for c in gdf.columns if c.lower() in {"station", "station_number", "station_num", "station_id", "stn_num", "stn_id"}]
    if exact:
        return exact[0]
    for c in gdf.columns:
        sample = gdf[c].astype(str).head(200)
        if sample.str.match(r"^\d{2}[A-Z]{2}\d{3}").any():
            return c
    raise RuntimeError(f"Could not identify station field; columns={list(gdf.columns)}")


def load_basin_polygons(s: requests.Session) -> tuple[gpd.GeoDataFrame, dict]:
    download(s, BASIN_URL, BASIN_ZIP)
    extract = CACHE / "basins" / "mda05"
    marker = extract / ".extracted"
    if not marker.exists():
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BASIN_ZIP) as z:
            z.extractall(extract)
        marker.write_text(datetime.now(timezone.utc).isoformat())
    shp_files = sorted(extract.rglob("*.shp"), key=lambda p: p.stat().st_size, reverse=True)
    if not shp_files:
        raise RuntimeError("MDA_ADP_05.zip contained no shapefile")
    errors = []
    chosen = None
    gdf = None
    for shp in shp_files:
        try:
            candidate = gpd.read_file(shp)
            col = station_column(candidate)
            ids = candidate[col].astype(str).str.strip().str.upper()
            if ids.isin(TARGET_STATIONS).any():
                chosen, gdf = shp, candidate
                break
        except Exception as exc:
            errors.append(f"{shp.name}: {exc}")
    if gdf is None or chosen is None:
        raise RuntimeError("No basin shapefile containing target stations. " + "; ".join(errors[:5]))
    col = station_column(gdf)
    gdf["station"] = gdf[col].astype(str).str.strip().str.upper()
    gdf = gdf[gdf.station.isin(TARGET_STATIONS)].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    if gdf.crs is None:
        raise RuntimeError("Official WSC basin polygons have no CRS")
    meta = {"source": BASIN_URL, "source_shapefile": str(chosen), "station_field": col, "crs": str(gdf.crs), "stations_found": sorted(gdf.station.unique().tolist())}
    OUT.mkdir(parents=True, exist_ok=True)
    gdf[["station", "geometry"]].to_file(OUT / "official_target_basins.geojson", driver="GeoJSON")
    return gdf, meta


def union_station(gdf: gpd.GeoDataFrame, station: str):
    sub = gdf[gdf.station == station]
    if sub.empty:
        return None
    return unary_union(sub.geometry.tolist()).buffer(0)


def build_subareas(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    p = {st: union_station(gdf, st) for st in TARGET_STATIONS}
    if p["05EA002"] is None or p["05EA005"] is None:
        raise RuntimeError("05EA002/05EA005 polygons are required")
    upper_station = next((st for st in ["05EA006", "05EA004"] if p.get(st) is not None), None)
    upper = p.get(upper_station) if upper_station else None
    lower = p["05EA002"].difference(p["05EA005"]).buffer(0)
    atim = p.get("05EA012")
    carrot = p.get("05EA011")
    local = lower
    for tributary in [atim, carrot]:
        if tributary is not None:
            local = local.difference(tributary).buffer(0)
    rows = [
        {"subarea": "basin_to_05EA002", "source_station": "05EA002", "geometry": p["05EA002"]},
        {"subarea": "basin_to_05EA005_villeneuve", "source_station": "05EA005", "geometry": p["05EA005"]},
        {"subarea": "lower_incremental_05EA005_to_05EA002", "source_station": "difference", "geometry": lower},
    ]
    if upper is not None:
        rows.append({"subarea": "upper_lake_chain_isle_lac_ste_anne", "source_station": upper_station, "geometry": upper})
        rows.append({"subarea": "lac_ste_anne_to_villeneuve_mainstem", "source_station": f"05EA005-{upper_station}", "geometry": p["05EA005"].difference(upper).buffer(0)})
    if atim is not None:
        rows.append({"subarea": "atim_creek_big_lake_tributary", "source_station": "05EA012", "geometry": atim})
    if carrot is not None:
        rows.append({"subarea": "carrot_creek", "source_station": "05EA011", "geometry": carrot})
    if not local.is_empty:
        rows.append({"subarea": "direct_big_lake_and_local_to_05EA002", "source_station": "lower-minus-tributaries", "geometry": local})
    sub = gpd.GeoDataFrame(rows, crs=gdf.crs)
    metric = sub.to_crs("EPSG:3347")
    sub["area_km2"] = metric.area / 1e6
    sub.to_file(OUT / "derived_subareas.geojson", driver="GeoJSON")
    containment = {}
    for a, b in [("05EA005", "05EA002"), ("05EA012", "05EA002"), ("05EA011", "05EA002")]:
        if p.get(a) is not None and p.get(b) is not None:
            containment[f"{a}_within_{b}"] = bool(p[b].buffer(1e-8).contains(p[a]))
    if upper_station:
        containment[f"{upper_station}_within_05EA005"] = bool(p["05EA005"].buffer(1e-8).contains(upper))
    meta = {"upper_lake_polygon_station": upper_station, "containment_checks": containment, "subareas": [{"subarea": r.subarea, "source_station": r.source_station, "area_km2": float(r.area_km2)} for _, r in sub.iterrows()]}
    return sub, meta


def archive_roots(valid: datetime) -> list[str]:
    roots = []
    for d in [valid.date(), (valid + timedelta(days=1)).date(), (valid + timedelta(days=2)).date()]:
        roots.append(f"https://dd.weather.gc.ca/{d:%Y%m%d}/WXO-DD/")
        roots.append(f"https://dd.meteo.gc.ca/{d:%Y%m%d}/WXO-DD/")
    if valid.date() >= datetime.now(timezone.utc).date() - timedelta(days=1):
        roots += ["https://dd.weather.gc.ca/today/", "https://dd.meteo.gc.ca/today/"]
    return roots


def find_hrdpa_grid(s: requests.Session, valid: datetime) -> tuple[str, str] | None:
    wanted = f"{valid:%Y%m%dT%H}Z_MSC_HRDPA_APCP-Accum6h_Sfc_RLatLon0.0225_PT0H.grib2"
    for root in archive_roots(valid):
        url = f"{root}model_hrdpa/2.5km/{valid:%H}/"
        try:
            links = HREF.findall(get_text(s, url))
        except Exception:
            continue
        if wanted in links:
            return url + wanted, wanted
    return None


def choose_band(ds: rasterio.DatasetReader, kind: str = "precip") -> int:
    for i in range(1, ds.count + 1):
        tags = " ".join([str(ds.descriptions[i - 1] or ""), *[str(v) for v in ds.tags(i).values()]]).lower()
        if kind == "precip" and ("apcp" in tags or "total precipitation" in tags or "precipitation" in tags):
            if "confidence" not in tags and "cfia" not in tags:
                return i
        if kind == "cfia" and ("cfia" in tags or "confidence" in tags):
            return i
    return 1 if kind == "precip" else min(2, ds.count)


def raster_for_geometry(path: Path, geom, geom_crs, band: int | None = None) -> np.ma.MaskedArray:
    with rasterio.open(path) as ds:
        g = gpd.GeoSeries([geom], crs=geom_crs).to_crs(ds.crs).iloc[0]
        b = band or choose_band(ds, "precip")
        arr, _ = mask(ds, [mapping(g)], crop=True, filled=False, indexes=b)
        a = np.ma.asarray(arr, dtype=float)
        # GRIB missing/fill values can survive as very large values.
        a = np.ma.masked_where(~np.isfinite(a) | (a < -0.01) | (a > 1000), a)
        return a


def summarize_values(values: np.ma.MaskedArray) -> dict:
    v = values.compressed()
    if not len(v):
        return {"mean_mm": None, "max_mm": None, "valid_cells": 0, **{f"pct_gt_{t}mm": None for t in THRESHOLDS}}
    out = {"mean_mm": float(np.mean(v)), "max_mm": float(np.max(v)), "valid_cells": int(len(v))}
    for t in THRESHOLDS:
        out[f"pct_gt_{t}mm"] = float(np.mean(v > t) * 100)
    return out


def process_observed_events(s: requests.Session, subareas: gpd.GeoDataFrame) -> tuple[list[dict], list[str]]:
    calibration_path = ROOT / "calibration" / "calibration.json"
    if not calibration_path.exists():
        return [], ["Calibration output missing; observed event grids not processed"]
    events = json.loads(calibration_path.read_text()).get("events", [])
    rows, warnings = [], []
    for event in events:
        start = pd.Timestamp(event["rain_start_utc"]).to_pydatetime()
        end = pd.Timestamp(event["rain_end_utc"]).to_pydatetime()
        valid = start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=6)
        grids = []
        while valid <= end:
            found = find_hrdpa_grid(s, valid)
            if found:
                url, name = found
                path = CACHE / "hrdpa" / name
                try:
                    grids.append(download(s, url, path))
                except Exception as exc:
                    warnings.append(f"event {event['event_id']} {valid.isoformat()}: {exc}")
            else:
                warnings.append(f"event {event['event_id']} {valid.isoformat()}: final HRDPA grid not found")
            valid += timedelta(hours=6)
        if not grids:
            continue
        for _, area in subareas.iterrows():
            total = None
            for path in grids:
                try:
                    arr = raster_for_geometry(path, area.geometry, subareas.crs)
                    if total is None:
                        total = arr.copy()
                    elif total.shape == arr.shape:
                        total = np.ma.add(total, arr)
                    else:
                        warnings.append(f"grid shape mismatch {path.name} for {area.subarea}")
                except Exception as exc:
                    warnings.append(f"{path.name} {area.subarea}: {exc}")
            if total is not None:
                row = {"event_id": event["event_id"], "rain_start_utc": event["rain_start_utc"], "rain_end_utc": event["rain_end_utc"], "subarea": area.subarea, "area_km2": float(area.area_km2), "n_grids": len(grids)}
                row.update(summarize_values(total))
                rows.append(row)
    return rows, warnings


def forecast_candidates(links: Iterable[str]) -> list[str]:
    good = []
    for x in links:
        low = x.lower()
        if not low.endswith(".grib2"):
            continue
        if any(k in low for k in ["totalprecip-accum", "apcp-accum", "tprate-accum"]):
            if not any(bad in low for bad in ["convective", "freezing", "snow", "solid", "liquid"]):
                good.append(x)
    return sorted(good)


def latest_run(s: requests.Session, model: str) -> tuple[str, str] | None:
    if model == "HRDPS":
        base = "https://dd.weather.gc.ca/today/model_hrdps/continental/2.5km"
        hours = ["18", "12", "06", "00"]
        probe = "006"
    elif model == "RDPS":
        base = "https://dd.weather.gc.ca/today/model_rdps/10km"
        hours = ["18", "12", "06", "00"]
        probe = "006"
    else:
        base = "https://dd.weather.gc.ca/today/ensemble/reps/10km/grib2"
        hours = ["18", "12", "06", "00"]
        probe = "006"
    for h in hours:
        url = f"{base}/{h}/{probe}/"
        try:
            links = HREF.findall(get_text(s, url))
        except Exception:
            continue
        candidates = forecast_candidates(links)
        if candidates:
            return base, h
    return None


def process_deterministic_qpf(s: requests.Session, model: str, subareas: gpd.GeoDataFrame) -> tuple[list[dict], dict, list[str]]:
    run = latest_run(s, model)
    if not run:
        return [], {"model": model, "status": "no run discovered"}, [f"{model}: no precipitation run discovered"]
    base, hh = run
    max_h = 48 if model == "HRDPS" else 84
    forecast_hours = list(range(6, max_h + 1, 6))
    files, warnings = [], []
    candidate_log = {}
    for fh in forecast_hours:
        url = f"{base}/{hh}/{fh:03d}/"
        try:
            links = HREF.findall(get_text(s, url))
            candidates = forecast_candidates(links)
            candidate_log[str(fh)] = candidates
            if not candidates:
                continue
            # Prefer cumulative total precipitation rather than interval-only derivatives.
            name = sorted(candidates, key=lambda x: ("totalprecip-accum" not in x.lower(), len(x)))[0]
            path = CACHE / "qpf" / model.lower() / name
            files.append((fh, download(s, url + name, path), name))
        except Exception as exc:
            warnings.append(f"{model} PT{fh:03d}: {exc}")
    if not files:
        return [], {"model": model, "run_hour_utc": hh, "candidates": candidate_log, "status": "no files"}, warnings
    rows = []
    for _, area in subareas.iterrows():
        previous = None
        previous_h = 0
        for fh, path, name in files:
            try:
                cumulative = raster_for_geometry(path, area.geometry, subareas.crs)
                if previous is None or previous.shape != cumulative.shape:
                    increment = cumulative
                    interval_h = fh
                else:
                    increment = np.ma.maximum(cumulative - previous, 0)
                    interval_h = fh - previous_h
                row = {"model": model, "run_hour_utc": hh, "forecast_hour_end": fh, "interval_hours": interval_h, "subarea": area.subarea, "source_file": name}
                row.update(summarize_values(increment))
                rows.append(row)
                previous, previous_h = cumulative, fh
            except Exception as exc:
                warnings.append(f"{model} {name} {area.subarea}: {exc}")
    meta = {"model": model, "run_hour_utc": hh, "files": [name for _, _, name in files], "candidates": candidate_log, "status": "processed"}
    return rows, meta, warnings


def discover_reps(s: requests.Session) -> tuple[dict, list[str]]:
    run = latest_run(s, "REPS")
    if not run:
        return {"model": "REPS", "status": "no run discovered"}, ["REPS: no precipitation run discovered"]
    base, hh = run
    out = {"model": "REPS", "run_hour_utc": hh, "forecast_hours": {}, "status": "discovered"}
    warnings = []
    for fh in [6, 12, 24, 36, 48]:
        url = f"{base}/{hh}/{fh:03d}/"
        try:
            links = HREF.findall(get_text(s, url))
            out["forecast_hours"][str(fh)] = forecast_candidates(links)
        except Exception as exc:
            warnings.append(f"REPS PT{fh:03d}: {exc}")
    return out, warnings


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = http()
    warnings = []
    basin_gdf, basin_meta = load_basin_polygons(s)
    subareas, subarea_meta = build_subareas(basin_gdf)
    observed_rows, w = process_observed_events(s, subareas); warnings += w
    write_csv(observed_rows, OUT / "observed_event_grid_coverage.csv")
    qpf_rows = []
    qpf_meta = []
    for model in ["HRDPS", "RDPS"]:
        rows, meta, w = process_deterministic_qpf(s, model, subareas)
        qpf_rows += rows; qpf_meta.append(meta); warnings += w
    write_csv(qpf_rows, OUT / "deterministic_qpf_by_subarea.csv")
    reps_meta, w = discover_reps(s); warnings += w
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "basin_source": basin_meta,
        "subarea_derivation": subarea_meta,
        "observed_event_rows": len(observed_rows),
        "deterministic_qpf_rows": len(qpf_rows),
        "qpf_models": qpf_meta,
        "reps": reps_meta,
        "limitations": [
            "Grid-cell coverage percentages are based on valid HRDPA/forecast cell centres within official WSC basin polygons.",
            "The 05EA002-to-Starkey local reach is not yet a separately delineated drainage polygon.",
            "REPS is discovered here; ensemble-band extraction and probabilistic basin statistics are handled in the next model stage.",
        ],
        "warning_count": len(warnings),
    }
    (OUT / "spatial_qpf.json").write_text(json.dumps(result, indent=2))
    (OUT / "warnings.log").write_text("\n".join(warnings) if warnings else "No warnings.\n")
    print(json.dumps({"observed_rows": len(observed_rows), "qpf_rows": len(qpf_rows), "warnings": len(warnings)}, indent=2))


if __name__ == "__main__":
    main()
