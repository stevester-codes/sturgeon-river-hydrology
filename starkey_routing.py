#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

ROOT = Path("sturgeon_pipeline_output")
WO = ROOT / "raw" / "wateroffice"
OUT = ROOT / "routing"
COORDS = {
    "05EA002": (-113.62722, 53.63583),
    "STARKKEY_NORTH": (-113.57123328125776, 53.68453354693282),
    "STARKKEY_SOUTH": (-113.57107463558906, 53.68447117697513),
    "05EA001": (-113.2822222222, 53.8325),
}


def parse(station: str) -> pd.DataFrame:
    p = WO / f"{station}.csv"
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    dc = next(c for c in df if c.lower() == "date")
    pc = next(c for c in df if "parameter" in c.lower() or "paramètre" in c.lower())
    vc = next(c for c in df if "value" in c.lower() or "valeur" in c.lower())
    df[dc] = pd.to_datetime(df[dc], utc=True, errors="coerce")
    df[pc] = pd.to_numeric(df[pc], errors="coerce")
    df[vc] = pd.to_numeric(df[vc], errors="coerce")
    piv = df.dropna(subset=[dc, pc, vc]).pivot_table(index=dc, columns=pc, values=vc, aggfunc="median")
    return piv.rename(columns={46: "stage_m", 47: "flow_m3s"}).sort_index().resample("1h").median().interpolate(limit=2)


def corr_lag(up: pd.Series, down: pd.Series, max_lag=96) -> dict:
    common = pd.concat([up.rename("up"), down.rename("down")], axis=1).dropna()
    if len(common) < 72:
        return {"status": "insufficient overlapping data", "n": len(common)}
    # Six-hour smoothed changes isolate hydrograph movement while reducing
    # datum and basin-size differences between gauges.
    u = common.up.rolling(6, center=True).mean().diff(6)
    d = common.down.rolling(6, center=True).mean().diff(6)
    rows = []
    for lag in range(max_lag + 1):
        pair = pd.concat([u.rename("u"), d.shift(-lag).rename("d")], axis=1).dropna()
        if len(pair) < 48 or pair.u.std() == 0 or pair.d.std() == 0:
            continue
        rows.append((lag, float(pair.u.corr(pair.d)), len(pair)))
    if not rows:
        return {"status": "no valid lag correlation"}
    rows.sort(key=lambda x: x[1], reverse=True)
    best = rows[0]
    near = [r[0] for r in rows if r[1] >= best[1] - 0.05]
    return {
        "status": "calculated",
        "best_lag_h": int(best[0]),
        "best_correlation": best[1],
        "n_pairs": int(best[2]),
        "near_optimal_lag_range_h": [int(min(near)), int(max(near))],
        "top_lags": [{"lag_h": int(a), "correlation": b, "n": int(n)} for a, b, n in rows[:10]],
    }


def distance_km(a, b):
    g = Geod(ellps="WGS84")
    _, _, m = g.inv(a[0], a[1], b[0], b[1])
    return m / 1000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    up = parse("05EA002")
    down = parse("05EA001")
    metric = "flow_m3s" if "flow_m3s" in up and "flow_m3s" in down else "stage_m"
    lag = corr_lag(up[metric], down[metric])
    starkey = ((COORDS["STARKKEY_NORTH"][0] + COORDS["STARKKEY_SOUTH"][0]) / 2, (COORDS["STARKKEY_NORTH"][1] + COORDS["STARKKEY_SOUTH"][1]) / 2)
    d_starkey = distance_km(COORDS["05EA002"], starkey)
    d_down = distance_km(COORDS["05EA002"], COORDS["05EA001"])
    ratio = d_starkey / d_down if d_down else None
    estimate = None
    if lag.get("status") == "calculated" and ratio:
        central = lag["best_lag_h"] * ratio
        lo_src, hi_src = lag["near_optimal_lag_range_h"]
        lo = max(0.5, lo_src * ratio * 0.65)
        hi = max(lo, hi_src * ratio * 1.5)
        estimate = {"central_h": central, "range_h": [lo, hi], "method": "05EA002-to-05EA001 observed hydrograph lag scaled by geodesic distance ratio with routing uncertainty allowance"}
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metric_used": metric,
        "coordinates": {"05EA002": COORDS["05EA002"], "starkey_bridge_midpoint": starkey, "05EA001": COORDS["05EA001"]},
        "geodesic_distances_km": {"05EA002_to_starkey": d_starkey, "05EA002_to_05EA001": d_down, "distance_ratio": ratio},
        "observed_05EA002_to_05EA001_lag": lag,
        "estimated_05EA002_to_starkey_wave_lag": estimate,
        "stage_translation": {"status": "not calibrated", "reason": "No water-level gauge or common vertical datum exists at the Starkey construction crossing. The operational 1.70 m threshold remains referenced to 05EA002."},
        "limitations": [
            "Geodesic distance is shorter than channel distance, but using the same ratio for both reaches reduces first-order bias.",
            "05EA001 receives additional downstream drainage, so only coherent hydrograph timing—not absolute stage—is used.",
            "The wave lag is relevant to river response timing; direct local runoff at Starkey may respond faster during a localized storm.",
        ],
    }
    (OUT / "starkey_routing.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
