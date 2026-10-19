#!/usr/bin/env python
"""
Stage 3: static fields -> meta/static_fields.npz.

    python src/build_static.py --config config/default.yaml [--ref-file <glorys month .nc>]

From one GLORYS12 day (the first day of --ref-file, default the first monthly
file found) on the model grid:
  n_valid_levels  (h,w) int16: leading levels where every variables.mask_vars
                  field is valid. Level k is ocean iff k < n_valid (as in
                  Oceryn3D Global). The intersection matters: u/v can be
                  missing on a coastal ring where temp is valid.
  land_mask       1 = ocean (n_valid > 0)
  bathymetry      log1p(depth) / log1p(6000 m), clipped to [0, 1]. depth is
                  GLORYS12's deptho (paths.glorys_static_nc) when present,
                  else the depth of the deepest valid level (capped at the
                  last configured level, which is all the model sees anyway).
  sin_lat cos_lat sin_lon cos_lon, cos_lat_weights (area weight)
  edge_weight     0 within dataset.edge_margin_px of the domain edge
  lat lon depth_m
The ocean geometry never changes between days, so one day suffices; the
script checks a second day if the file has one and reports any difference.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg                                         # noqa: E402
from glorys_io import GlorysMonth, list_month_files            # noqa: E402
from grid import coarsen_nan, edge_weight, invalid, match_axis  # noqa: E402

BATHY_MAX_M = 6000.0


def valid_levels(fields: dict, mask_vars, n_levels: int) -> np.ndarray:
    """(h,w) number of leading levels valid in every mask var."""
    ok = np.ones_like(fields[mask_vars[0]], dtype=bool)
    for v in mask_vars:
        ok &= ~invalid(fields[v])
    return np.cumprod(ok, axis=0).sum(axis=0).astype(np.int16)


def read_deptho(path: str, cfg: Cfg):
    """GLORYS12 deptho on the model grid (NaN on land), or None."""
    if not path or not os.path.exists(path):
        return None
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        la = np.asarray(ds.variables["latitude"][:], np.float64)
        lo = np.asarray(ds.variables["longitude"][:], np.float64)
        a = np.ma.filled(np.ma.asarray(ds.variables["deptho"][:], np.float32), np.nan)
    a = a.reshape(len(la), len(lo))
    lat_n, lon_n = cfg.native_axes()
    tol = 0.25 * float(cfg.grid["dlat"])
    a = a[match_axis(la, lat_n, tol, "deptho latitude")][:, match_axis(lo, lon_n, tol, "deptho longitude")]
    return coarsen_nan(a, cfg.coarsen)


def build(cfg: Cfg, ref_file: str) -> dict:
    with GlorysMonth(ref_file, cfg) as gm:
        f0 = gm.read_day(gm.dates[0], cfg.mask_vars)
        nv = valid_levels(f0, cfg.mask_vars, cfg.n_levels)
        if len(gm.dates) > 1:
            nv1 = valid_levels(gm.read_day(gm.dates[-1], cfg.mask_vars),
                               cfg.mask_vars, cfg.n_levels)
            diff = int((nv1 != nv).sum())
            if diff:
                print(f"[static] note: {diff} cells change valid-level count between "
                      f"{gm.dates[0]} and {gm.dates[-1]}; using the first day")
    lat, lon = cfg.axes()
    deptho = read_deptho(cfg.paths.get("glorys_static_nc", ""), cfg)
    if deptho is None:
        print("[static] no deptho file: bathymetry from the deepest valid level")
        dz = np.asarray(cfg.depth_m, np.float32)
        depth = np.where(nv > 0, dz[np.clip(nv - 1, 0, None)], 0.0)
    else:
        depth = np.where(nv > 0, np.nan_to_num(deptho, nan=0.0), 0.0)
    bathy = np.clip(np.log1p(np.maximum(depth, 0.0)) / np.log1p(BATHY_MAX_M), 0, 1)
    la2 = np.broadcast_to(np.deg2rad(lat)[:, None], nv.shape)
    lo2 = np.broadcast_to(np.deg2rad(lon)[None, :], nv.shape)
    ocean = (nv > 0).astype(np.float32)
    return {
        "n_valid_levels": nv,
        "land_mask": ocean,
        "bathymetry": (bathy * ocean).astype(np.float32),
        "sin_lat": np.sin(la2).astype(np.float32), "cos_lat": np.cos(la2).astype(np.float32),
        "sin_lon": np.sin(lo2).astype(np.float32), "cos_lon": np.cos(lo2).astype(np.float32),
        "cos_lat_weights": np.cos(la2).astype(np.float32),
        "edge_weight": edge_weight(cfg.h, cfg.w, cfg.edge_margin),
        "lat": lat.astype(np.float64), "lon": lon.astype(np.float64),
        "depth_m": np.asarray(cfg.depth_m, np.float64),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref-file")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config)
    ref = args.ref_file
    if not ref:
        files = list_month_files(cfg.paths["glorys_dir"])
        if not files:
            sys.exit(f"no GLORYS files under {cfg.paths['glorys_dir']}")
        ref = files[0][2]
    z = build(cfg, ref)
    out = cfg.paths["static_npz"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, out)
    nv = z["n_valid_levels"]
    print(f"[static] {cfg.h} x {cfg.w}, ocean {100 * (nv > 0).mean():.1f}% of cells, "
          f"full-depth ({cfg.n_levels} levels) {100 * (nv == cfg.n_levels).mean():.1f}% "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
