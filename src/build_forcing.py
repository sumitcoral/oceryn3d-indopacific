#!/usr/bin/env python
"""
Stage 1: ERA5 daily stress -> the model grid, one .npy per month.

    python src/build_forcing.py --config config/default.yaml [--year Y] [--overwrite]

Reads <era5_dir>/YYYY/era5_stress_YYYY_MM.nc (download_era5.py; a netCDF or
a zip of one netCDF per variable), interpolates bilinearly from ERA5's
0.25 deg grid to the model grid (the coarse-cell centres for the 1/4 deg
prototype) and writes
    <forcing_dir>/YYYY/forcing_YYYY_MM.npy   (ndays, n_forcing, h, w) float32, N/m2
    <forcing_dir>/YYYY/forcing_YYYY_MM.json  {"dates": [...], "vars": [...]}
in the order of the config's forcing fields (taux, tauy). Values over land
are kept here; the cache sets them to 0 with the ocean mask, like every
other channel.

Idempotent (existing months are skipped unless --overwrite) and atomic.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg                       # noqa: E402
from grid import bilinear                    # noqa: E402

ERA5_RE = re.compile(r"era5_stress_(\d{4})_(\d{2})\.nc$")


def forcing_paths(forcing_dir: str, y: int, m: int):
    base = os.path.join(forcing_dir, f"{y:04d}", f"forcing_{y:04d}_{m:02d}")
    return base + ".npy", base + ".json"


def list_era5(era5_dir: str):
    out = []
    if not os.path.isdir(era5_dir):
        return out
    for sub in sorted(os.listdir(era5_dir)):
        d = os.path.join(era5_dir, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                mt = ERA5_RE.match(fn)
                if mt:
                    out.append((int(mt.group(1)), int(mt.group(2)), os.path.join(d, fn)))
    return out


def _open_all(path: str, tmpdir: str):
    """netCDF4 Datasets of a plain netCDF or of each member of a zip."""
    import netCDF4
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".nc")]
            z.extractall(tmpdir, members=names)
        return [netCDF4.Dataset(os.path.join(tmpdir, n)) for n in names]
    return [netCDF4.Dataset(path)]


def _dates(ds):
    import netCDF4
    tn = "valid_time" if "valid_time" in ds.variables else "time"
    tv = ds.variables[tn]
    ts = netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                          only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return [date(t.year, t.month, t.day) for t in ts]


def read_era5_month(path: str, cfg: Cfg):
    """(dates, {name: (n, ny, nx) float32}, lat asc, lon asc) for the forcing fields."""
    want = cfg.forcing["era5_netcdf_names"]
    with tempfile.TemporaryDirectory() as td:
        dss = _open_all(path, td)
        try:
            fields, dates, lat, lon = {}, None, None, None
            for name in cfg.forcing_vars:
                for ds in dss:
                    hit = next((n for n in want[name] if n in ds.variables), None)
                    if hit is None:
                        continue
                    d = _dates(ds)
                    la = np.asarray(ds.variables["latitude" if "latitude" in ds.variables else "lat"][:], np.float64)
                    lo = np.asarray(ds.variables["longitude" if "longitude" in ds.variables else "lon"][:], np.float64)
                    a = np.ma.filled(np.ma.asarray(ds.variables[hit][:], np.float32), np.nan)
                    a = a.reshape(len(d), len(la), len(lo))
                    if la[0] > la[-1]:
                        la, a = la[::-1], a[:, ::-1]
                    lo = np.where(lo > 180.0 + 1e-6, lo - 360.0, lo)
                    o = np.argsort(lo)
                    lo, a = lo[o], a[:, :, o]
                    if dates is None:
                        dates, lat, lon = d, la, lo
                    elif d != dates or len(la) != len(lat) or len(lo) != len(lon):
                        raise ValueError(f"{path}: {name} has other times or grid than "
                                         f"{cfg.forcing_vars[0]}")
                    fields[name] = np.ascontiguousarray(a)
                    break
                else:
                    raise KeyError(f"{path}: none of {want[name]} found for {name}; "
                                   "add the name to forcing.era5_netcdf_names")
        finally:
            for ds in dss:
                ds.close()
    return dates, fields, lat, lon


def build_month(cfg: Cfg, y: int, m: int, src: str, overwrite=False) -> str:
    npy, js = forcing_paths(cfg.paths["forcing_dir"], y, m)
    if os.path.exists(npy) and os.path.exists(js) and not overwrite:
        return "exists"
    dates, fields, lat, lon = read_era5_month(src, cfg)
    lat_d, lon_d = cfg.axes()
    out = np.stack([bilinear(fields[n], lat, lon, lat_d, lon_d) for n in cfg.forcing_vars],
                   axis=1).astype(np.float32)                  # (n, F, h, w)
    if not np.isfinite(out).all():
        raise ValueError(f"{src}: interpolated stress has missing values")
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    tmp = npy + ".part.npy"
    np.save(tmp, out)
    os.replace(tmp, npy)
    with open(js + ".part", "w") as f:
        json.dump({"dates": [d.isoformat() for d in dates], "vars": cfg.forcing_vars,
                   "units": "N m-2", "source": os.path.basename(src)}, f)
    os.replace(js + ".part", js)
    return f"{len(dates)} days"


def load_month_dates(js_path: str):
    with open(js_path) as f:
        meta = json.load(f)
    return [date.fromisoformat(s) for s in meta["dates"]], meta["vars"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config)
    files = [f for f in list_era5(cfg.paths["era5_dir"])
             if args.year is None or f[0] == args.year]
    if not files:
        sys.exit(f"no ERA5 files under {cfg.paths['era5_dir']}"
                 f"{f' for {args.year}' if args.year else ''}: run scripts/download_era5.py")
    print(f"[forcing] {len(files)} months -> {cfg.paths['forcing_dir']} "
          f"({cfg.h} x {cfg.w}, {cfg.forcing_vars})", flush=True)
    for y, m, p in files:
        print(f"  {y}-{m:02d}: {build_month(cfg, y, m, p, args.overwrite)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
