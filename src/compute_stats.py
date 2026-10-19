#!/usr/bin/env python
"""
Stage 4: normalization statistics -> meta/stats.json.

    python src/compute_stats.py --config config/default.yaml [--stride 5]

Train-split days only, every --stride-th day. 3D fields get a mean and std
per level over that level's ocean cells (k < n_valid_levels); 2D fields
(ssh and the forcing) a scalar over surface ocean. Population std, float64
sums. Same file layout as Oceryn3D Global:
    {"_meta": {...}, "vars3d": {v: {"mean": [L], "std": [L], "count": [L]}},
     "vars2d": {v: {"mean": m, "std": s, "count": n}}}
Needs days.csv (Stage 2) and static_fields.npz (Stage 3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg                                    # noqa: E402
from fields import Static, load_days, load_forcing_day    # noqa: E402
from glorys_io import GlorysMonth                         # noqa: E402
from grid import invalid, level_mask                      # noqa: E402


class Acc:
    def __init__(self, shape):
        self.n = np.zeros(shape, np.float64)
        self.s = np.zeros(shape, np.float64)
        self.ss = np.zeros(shape, np.float64)

    def add(self, a, m, axes):
        ok = m & ~invalid(a)
        x = np.where(ok, a, 0.0).astype(np.float64)
        self.n += ok.sum(axis=axes)
        self.s += x.sum(axis=axes)
        self.ss += (x * x).sum(axis=axes)

    def result(self):
        n = np.maximum(self.n, 1)
        mean = self.s / n
        std = np.sqrt(np.maximum(self.ss / n - mean ** 2, 0.0))
        return mean, std, self.n


def compute(cfg: Cfg, stride: int = 5, limit: int = 0) -> dict:
    static = Static(cfg.paths["static_npz"], cfg)
    days = load_days(cfg.paths["days_csv"], "train")[::max(1, stride)]
    if limit:
        days = days[:limit]
    if not days:
        raise ValueError("no train days in days.csv")
    lm = level_mask(static.n_valid, cfg.n_levels).astype(bool)        # (L,h,w)
    ocean = static.ocean
    acc3 = {v: Acc((cfg.n_levels,)) for v in cfg.vars3d}
    acc2 = {v: Acc(()) for v in cfg.vars2d}
    by_file = defaultdict(list)
    for r in days:
        by_file[r["glorys_path"]].append(r)
    t0, k = time.time(), 0
    for path, rs in by_file.items():
        with GlorysMonth(path, cfg) as gm:
            for r in rs:
                f = gm.read_day(r["date"])
                frc = load_forcing_day(r["forcing_path"], r["forcing_index"])
                for v in cfg.vars3d:
                    acc3[v].add(f[v], lm, (1, 2))
                for v in cfg.vars2d:
                    a = frc[cfg.forcing_vars.index(v)] if v in cfg.forcing_vars else f[v]
                    acc2[v].add(a, ocean, (0, 1))
                k += 1
        print(f"  {k}/{len(days)} days, {time.time() - t0:.0f} s", flush=True)
    out = {"_meta": {"split": "train", "stride_days": stride, "n_days": len(days),
                     "first": days[0]["date"].isoformat(), "last": days[-1]["date"].isoformat(),
                     "grid": [cfg.h, cfg.w], "coarsen": cfg.coarsen,
                     "depth_m": cfg.depth_m, "config": os.path.abspath(cfg.path)},
           "vars3d": {}, "vars2d": {}}
    for v, a in acc3.items():
        m, s, n = a.result()
        if (n == 0).any():
            raise ValueError(f"{v}: some levels have no ocean cells: {n.tolist()}")
        out["vars3d"][v] = {"mean": m.tolist(), "std": s.tolist(), "count": n.tolist()}
    for v, a in acc2.items():
        m, s, n = a.result()
        out["vars2d"][v] = {"mean": float(m), "std": float(s), "count": float(n)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N sampled days (tests)")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config)
    st = compute(cfg, args.stride, args.limit)
    out = cfg.paths["stats_json"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(out + ".part", out)
    for v in cfg.vars3d:
        s = st["vars3d"][v]
        print(f"  {v:5s} level 0 mean {s['mean'][0]:9.4f} std {s['std'][0]:8.4f}   "
              f"level {cfg.n_levels - 1} mean {s['mean'][-1]:9.4f} std {s['std'][-1]:8.4f}")
    for v in cfg.vars2d:
        s = st["vars2d"][v]
        print(f"  {v:5s} mean {s['mean']:9.4f} std {s['std']:8.4f}")
    print(f"[stats] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
