#!/usr/bin/env python
"""
Stage 5: normalized fp16 day stacks -> <cache_dir>/YYYY/day_YYYY_MM_DD.npy

    python src/build_cache.py --config config/default.yaml --year Y [--overwrite] [--verify]

One (99, h, w) float16 file per day of days.csv in year Y: the z-scored
state and forcing in the C1 channel order (fields.assemble_day). The loader
memory-maps these; reading GLORYS12 netCDF per sample would be far too
slow. fp16 is ample for z-scored O(1) values (Oceryn3D Global does the same).
Size: 99 x 720 x 1800 x 2 B = 257 MB per day, ~2.9 TB for 1993-2023
(1/4 deg prototype: 29 MB per day).

Idempotent (existing files are skipped unless --overwrite), atomic (written
as .part.npy and renamed). --verify checks shape, dtype and finiteness of
the year's existing files instead of building.
Needs days.csv, static_fields.npz and stats.json. A change of stats.json
means rebuilding everything with --overwrite.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg                                                     # noqa: E402
from fields import Static, assemble_day, cache_path, load_days, load_forcing_day, load_stats  # noqa: E402
from glorys_io import GlorysMonth                                          # noqa: E402


def write_atomic(path: str, a: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part.npy"
    np.save(tmp, a)
    os.replace(tmp, path)


def verify(cfg: Cfg, days) -> int:
    bad = 0
    for r in days:
        p = cache_path(cfg.paths["cache_dir"], r["date"])
        if not os.path.exists(p):
            print(f"  missing {p}")
            bad += 1
            continue
        a = np.load(p, mmap_mode="r")
        if a.shape != (cfg.n_dyn, cfg.h, cfg.w) or a.dtype != np.float16:
            print(f"  bad shape/dtype {a.shape} {a.dtype}: {p}")
            bad += 1
        elif not np.isfinite(a).all():
            print(f"  non-finite values: {p}")
            bad += 1
    print(f"[cache] verify: {len(days) - bad}/{len(days)} ok")
    return 1 if bad else 0


def build(cfg: Cfg, days, overwrite=False) -> int:
    stats = load_stats(cfg.paths["stats_json"])
    static = Static(cfg.paths["static_npz"], cfg)
    by_file = defaultdict(list)
    for r in days:
        if overwrite or not os.path.exists(cache_path(cfg.paths["cache_dir"], r["date"])):
            by_file[r["glorys_path"]].append(r)
    todo = sum(len(v) for v in by_file.values())
    print(f"[cache] {todo} of {len(days)} days to build -> {cfg.paths['cache_dir']}", flush=True)
    t0, k = time.time(), 0
    for path, rs in sorted(by_file.items()):
        with GlorysMonth(path, cfg) as gm:
            for r in rs:
                f = gm.read_day(r["date"])
                frc = load_forcing_day(r["forcing_path"], r["forcing_index"])
                a = assemble_day(cfg, stats, f, frc, static.n_valid)
                if not np.isfinite(a).all():
                    raise ValueError(f"{r['date']}: non-finite values after normalization")
                write_atomic(cache_path(cfg.paths["cache_dir"], r["date"]), a.astype(np.float16))
                k += 1
        el = time.time() - t0
        print(f"  {os.path.basename(path)}: {k}/{todo} days, {el / max(k, 1):.1f} s/day",
              flush=True)
    return k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--year", type=int, help="one year (PBS array task); default all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config)
    days = [r for r in load_days(cfg.paths["days_csv"])
            if args.year is None or r["date"].year == args.year]
    if not days:
        sys.exit(f"no days{f' in {args.year}' if args.year else ''} in {cfg.paths['days_csv']}")
    if args.verify:
        return verify(cfg, days)
    build(cfg, days, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
