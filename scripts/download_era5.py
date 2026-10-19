#!/usr/bin/env python
"""
Download ERA5 daily-mean surface wind stress over the domain, one file per month.

    python scripts/download_era5.py --config config/default.yaml [--start 1993-01 --end 2023-12]
           [--workers 4] [--dry-run] [--remap OLD=NEW]

Needs internet and a CDS account (~/.cdsapirc). Uses the CDS dataset
download.era5_dataset ("derived-era5-single-levels-daily-statistics"):
daily means of the hourly ERA5 fields forcing.era5_variables (the eastward and
northward "mean turbulent surface stress", N/m2), computed by the CDS over
00-23 UTC of each day. For these mean-rate fields the hour stamped T is the
mean over (T-1 h, T], so a daily value covers 23:00 the day before to 23:00:
a 1 h offset from GLORYS12's 00-24 UTC day, negligible for a daily mean.

Writes <paths.era5_dir>/YYYY/era5_stress_YYYY_MM.nc on ERA5's 0.25 deg grid,
domain plus forcing.margin_deg. The CDS may return a zip with one netCDF per
variable; build_forcing.py reads either. Resumable and atomic like
download_glorys.py. ~40 MB per month.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from config import Cfg                                  # noqa: E402
from download_glorys import months, remap               # noqa: E402


def era5_path(era5_dir: str, y: int, m: int) -> str:
    return os.path.join(era5_dir, f"{y:04d}", f"era5_stress_{y:04d}_{m:02d}.nc")


def area(cfg: Cfg):
    """CDS area [north, west, south, east] = domain + margin. ERA5 longitudes
    are requested in 0..360 order; the east edge is capped at 180 so the
    request never crosses the date line (the domain stops at 179.92 E)."""
    lat, lon = cfg.native_axes()
    mg = float(cfg.forcing.get("margin_deg", 1.0))
    return [round(float(lat[-1]) + mg, 2), round(float(lon[0]) - mg, 2),
            round(float(lat[0]) - mg, 2), min(180.0, round(float(lon[-1]) + mg, 2))]


def cds_retrieve(cfg: Cfg, y: int, m: int, target: str) -> None:
    """One month of daily means (tests replace this)."""
    import cdsapi
    cdsapi.Client(quiet=True).retrieve(cfg.download["era5_dataset"], {
        "product_type": "reanalysis",
        "variable": list(cfg.forcing["era5_variables"].values()),
        "year": f"{y:04d}", "month": f"{m:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],          # the CDS drops invalid days
        "daily_statistic": "daily_mean", "time_zone": "utc+00:00",
        "frequency": "1_hourly", "area": area(cfg),
    }, target)


def main(argv=None, retrieve=cds_retrieve) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remap", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = Cfg.load(args.config)
    maps = [tuple(x.split("=", 1)) for x in args.remap]
    edir = remap(cfg.paths["era5_dir"], maps)
    start = args.start or f"{cfg.data['year_start']}-01"
    end = args.end or f"{cfg.data['year_end']}-12"
    todo = [(y, m) for y, m in months(start, end) if not os.path.exists(era5_path(edir, y, m))]
    print(f"[era5] {start}..{end}: {len(todo)} months to fetch -> {edir}, area "
          f"{area(cfg)} (N, W, S, E)", flush=True)
    if args.dry_run or not todo:
        return 0

    def get(ym):
        y, m = ym
        out = era5_path(edir, y, m)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".part"
        t0 = time.time()
        try:
            retrieve(cfg, y, m, tmp)
            os.replace(tmp, out)
        except Exception as e:
            return f"{y}-{m:02d}: FAILED {type(e).__name__}: {str(e)[:200]}"
        return f"{y}-{m:02d}: {os.path.getsize(out) / 1e6:.1f} MB in {time.time() - t0:.0f} s"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for k, msg in enumerate(ex.map(get, todo), 1):
            print(f"  [{k}/{len(todo)}] {msg}", flush=True)
    left = [ym for ym in todo if not os.path.exists(era5_path(edir, *ym))]
    if left:
        print(f"[era5] {len(left)} months missing; rerun the same command", flush=True)
        return 1
    print("[era5] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
