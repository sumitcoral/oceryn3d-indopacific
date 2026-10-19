#!/usr/bin/env python
"""
Download GLORYS12V1 daily means for the Indo-Pacific domain, one file per month.

    python scripts/download_glorys.py --config config/default.yaml [--start 1993-01 --end 2023-12]
           [--workers 2] [--dry-run] [--static]

Needs internet and a Copernicus Marine account: run `copernicusmarine login`
once (it stores the credentials), on the NSCC login node or the Ubuntu
workstation (then write to the sshfs mount with --remap OLD=NEW).

What it writes, per month: <paths.glorys_dir>/YYYY/glorys12_YYYY_MM.nc with
uo, vo, thetao, so (time, depth, latitude, longitude) and zos, cut to the
domain and to the 24 configured depths only, packed as int16 with the scale
factors in download.pack (zlib). The data are read lazily through
copernicusmarine.open_dataset, so only the selected levels are transferred.

Resumable and atomic: an existing month is skipped, a month is written to
<name>.part and renamed when complete, and a failed month is reported and
retried on the next run. A month must have every day: a short month raises
instead of leaving a silent gap.

Size: ~24 levels x 4 variables x 720 x 1800 cells x 2 B ~ 250 MB/day before
compression (land and the seafloor compress well), so plan on ~1-2 TB for
1993-2023. --dry-run prints the months to fetch.
"""
from __future__ import annotations

import argparse
import calendar
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from config import Cfg                                   # noqa: E402
from glorys_io import month_path                         # noqa: E402
from grid import match_axis, match_depths                # noqa: E402

GLORYS_VARS3D = ("uo", "vo", "thetao", "so")


def remap(path: str, pairs) -> str:
    for old, new in pairs:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def months(start: str, end: str):
    y, m = (int(x) for x in start.split("-")[:2])
    y1, m1 = (int(x) for x in end.split("-")[:2])
    while (y, m) <= (y1, m1):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_days(y: int, m: int):
    return [date(y, m, d) for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def dataset_for(cfg: Cfg, day: date) -> str:
    for e in cfg.download["glorys_datasets"]:
        if date.fromisoformat(e["start"]) <= day <= date.fromisoformat(e["end"]):
            return e["id"]
    raise ValueError(f"no download.glorys_datasets entry covers {day}")


def bounds(cfg: Cfg):
    """Request box: the native axes plus half a cell, so edge rows are inside."""
    lat, lon = cfg.native_axes()
    h = 0.5 * float(cfg.grid["dlat"])
    return (float(lat[0]) - h, float(lat[-1]) + h, float(lon[0]) - h, float(lon[-1]) + h)


def open_glorys(dataset_id: str, start: date, end: date, cfg: Cfg):
    """Lazy xarray Dataset (tests replace this)."""
    import copernicusmarine
    s, n, w, e = bounds(cfg)
    names = cfg.variables["glorys_names"]
    var = [names[v] for v in cfg.vars3d] + [names[v] for v in cfg.targets_2d]
    return copernicusmarine.open_dataset(
        dataset_id=dataset_id, variables=var,
        minimum_latitude=s, maximum_latitude=n, minimum_longitude=w, maximum_longitude=e,
        minimum_depth=0.0, maximum_depth=1.02 * max(cfg.depth_m),
        start_datetime=f"{start:%Y-%m-%d}T00:00:00",
        end_datetime=f"{end:%Y-%m-%d}T23:59:59")


def subset(ds, cfg: Cfg):
    """Cut a lazy dataset to the configured levels, rows and columns."""
    lat_n, lon_n = cfg.native_axes()
    tol = 0.25 * float(cfg.grid["dlat"])
    iz = match_depths(ds["depth"].values, cfg.depth_m)
    iy = match_axis(ds["latitude"].values, lat_n, tol, "GLORYS latitude")
    ix = match_axis(ds["longitude"].values, lon_n, tol, "GLORYS longitude")
    return ds.isel(depth=iz, latitude=iy, longitude=ix)


def encoding(cfg: Cfg, ds) -> dict:
    names = cfg.variables["glorys_names"]
    pack = cfg.download["pack"]
    enc = {}
    for model_name, (scale, offset) in pack.items():
        g = names[model_name]
        if g in ds.variables:
            enc[g] = {"dtype": "int16", "scale_factor": float(scale),
                      "add_offset": float(offset), "_FillValue": np.int16(-32767),
                      "zlib": True, "complevel": 4}
    return enc


def fetch_month(cfg: Cfg, y: int, m: int, out: str, opener=open_glorys) -> str:
    import xarray as xr
    days = month_days(y, m)
    groups = {}
    for d in days:
        groups.setdefault(dataset_for(cfg, d), []).append(d)
    parts = []
    for did, ds_days in groups.items():
        ds = subset(opener(did, ds_days[0], ds_days[-1], cfg), cfg)
        parts.append(ds)
    ds = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")
    have = sorted({np.datetime64(t, "D") for t in ds["time"].values})
    want = [np.datetime64(d, "D") for d in days]
    miss = [str(t) for t in want if t not in have]
    if miss:
        raise ValueError(f"{y}-{m:02d}: no GLORYS data for {miss[:5]}"
                         f"{' ...' if len(miss) > 5 else ''} ({len(miss)} days). Check "
                         "the dataset end dates in download.glorys_datasets.")
    ds = ds.load()
    tmp = out + ".part"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.to_netcdf(tmp, encoding=encoding(cfg, ds))
    os.replace(tmp, out)
    return out


def fetch_static(cfg: Cfg, out: str):
    """GLORYS12 sea-floor depth (deptho) on the domain, for the bathymetry channel."""
    import copernicusmarine
    s, n, w, e = bounds(cfg)
    ds = copernicusmarine.open_dataset(
        dataset_id=cfg.download["glorys_static_dataset"], variables=["deptho"],
        minimum_latitude=s, maximum_latitude=n, minimum_longitude=w, maximum_longitude=e)
    tmp = out + ".part"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.load().to_netcdf(tmp)
    os.replace(tmp, out)


def main(argv=None, opener=open_glorys) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", help="YYYY-MM (default: data.year_start-01)")
    ap.add_argument("--end", help="YYYY-MM (default: data.year_end-12)")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel months (the service throttles large parallel pulls)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--static", action="store_true",
                    help="also fetch the static file (deptho) to paths.glorys_static_nc")
    ap.add_argument("--remap", action="append", default=[],
                    help="OLD=NEW path prefix, e.g. writing to NSCC over sshfs")
    args = ap.parse_args(argv)

    cfg = Cfg.load(args.config)
    maps = [tuple(x.split("=", 1)) for x in args.remap]
    gdir = remap(cfg.paths["glorys_dir"], maps)
    start = args.start or f"{cfg.data['year_start']}-01"
    end = args.end or f"{cfg.data['year_end']}-12"
    todo = [(y, m) for y, m in months(start, end)
            if not os.path.exists(month_path(gdir, y, m))]
    print(f"[glorys] {start}..{end}: {len(todo)} months to fetch -> {gdir} "
          f"({cfg.n_levels} levels {cfg.depth_m[0]:g}-{cfg.depth_m[-1]:g} m, "
          f"{cfg.native_h} x {cfg.native_w} cells)", flush=True)
    if args.static:
        sp = remap(cfg.paths["glorys_static_nc"], maps)
        if os.path.exists(sp):
            print(f"[glorys] static file present: {sp}")
        elif not args.dry_run:
            fetch_static(cfg, sp)
            print(f"[glorys] static -> {sp}")
    if args.dry_run or not todo:
        if todo:
            print(f"  first: {todo[:3]}  last: {todo[-3:]}")
        return 0

    def get(ym):
        y, m = ym
        out = month_path(gdir, y, m)
        t0 = time.time()
        try:
            fetch_month(cfg, y, m, out, opener)
        except ValueError:
            raise                                   # a real gap: stop, do not skip
        except Exception as e:                      # network etc.: report, rerun retries
            return f"{y}-{m:02d}: FAILED {type(e).__name__}: {str(e)[:200]}"
        return f"{y}-{m:02d}: {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f} s"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for k, msg in enumerate(ex.map(get, todo), 1):
            print(f"  [{k}/{len(todo)}] {msg}", flush=True)
    left = [ym for ym in todo if not os.path.exists(month_path(gdir, *ym))]
    if left:
        print(f"[glorys] {len(left)} months missing; rerun the same command", flush=True)
        return 1
    print("[glorys] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
