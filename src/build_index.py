#!/usr/bin/env python
"""
Stage 2: day index -> meta/days.csv.

    python src/build_index.py --config config/default.yaml [--dry-run]

One row per day that has BOTH a GLORYS12 day (in a monthly file) and an ERA5
forcing day (build_forcing.py output), with its split by year. Days in
data.skip_dates and days outside every split are left out. Seconds: it reads
only time axes and the forcing .json files.

The loader builds samples from this list: a sample is horizon+1 consecutive
days of one split, so a missing day breaks only the windows that contain it.
Gaps are printed here so they are noticed before training.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_forcing import forcing_paths, load_month_dates     # noqa: E402
from config import Cfg                                        # noqa: E402
from fields import DAYS_FIELDS                                # noqa: E402
from glorys_io import _dates, list_month_files                # noqa: E402


def glorys_days(cfg: Cfg) -> dict:
    """{date: monthly file path} from the files' time axes."""
    import netCDF4
    out = {}
    for _y, _m, p in list_month_files(cfg.paths["glorys_dir"]):
        with netCDF4.Dataset(p) as ds:
            for d in _dates(ds):
                out[d] = p
    return out


def forcing_days(cfg: Cfg, months) -> dict:
    """{date: (npy path, index)} for the given (year, month) set."""
    out = {}
    for y, m in sorted(months):
        npy, js = forcing_paths(cfg.paths["forcing_dir"], y, m)
        if not (os.path.exists(npy) and os.path.exists(js)):
            continue
        dates, names = load_month_dates(js)
        if names != cfg.forcing_vars:
            raise ValueError(f"{js} holds {names}, config forcing is {cfg.forcing_vars}: "
                             "rebuild with build_forcing.py --overwrite")
        for i, d in enumerate(dates):
            out[d] = (npy, i)
    return out


def gaps(days):
    """[(first missing, last missing)] between consecutive days of a sorted list."""
    out = []
    for a, b in zip(days, days[1:]):
        if (b - a).days > 1:
            out.append((a + timedelta(days=1), b - timedelta(days=1)))
    return out


def build(cfg: Cfg):
    g = glorys_days(cfg)
    f = forcing_days(cfg, {(d.year, d.month) for d in g})
    skip = cfg.skip_dates
    rows, no_forcing = [], []
    for d in sorted(g):
        sp = cfg.split_of(d)
        if sp is None or d in skip:
            continue
        if d not in f:
            no_forcing.append(d)
            continue
        npy, i = f[d]
        rows.append({"date": d.isoformat(), "split": sp, "glorys_path": g[d],
                     "forcing_path": npy, "forcing_index": i})
    return rows, no_forcing, len(g)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config)
    rows, no_forcing, n_g = build(cfg)
    if not rows:
        sys.exit(f"no usable days: {n_g} GLORYS days found under "
                 f"{cfg.paths['glorys_dir']}; run the downloads and build_forcing.py")
    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r)
    print(f"[index] {len(rows)} days from {n_g} GLORYS days "
          f"({len(no_forcing)} without ERA5 forcing)")
    from datetime import date
    for sp, rs in by.items():
        ds = [date.fromisoformat(r["date"]) for r in rs]
        gp = gaps(ds)
        print(f"  {sp:5s} {len(rs):6d} days {ds[0]}..{ds[-1]}, {len(gp)} gaps"
              + (f", first {gp[0][0]}..{gp[0][1]}" if gp else ""))
    if no_forcing:
        print(f"  without forcing: {no_forcing[0]}..{no_forcing[-1]} "
              f"(run download_era5.py and build_forcing.py)")
    if args.dry_run:
        return 0
    out = cfg.paths["days_csv"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DAYS_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".part", out)
    print(f"[index] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
