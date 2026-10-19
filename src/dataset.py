#!/usr/bin/env python
"""
Training/validation stream of (patch, window) samples from the fp16 day cache.

A sample is a WINDOW of horizon+1 consecutive days t..t+H of one split, cut
to one patch. It yields, as float32 tensors:
    x0   (n_dyn, p, p)    day t's cached stack (state + forcing)
    frc  (H+1, F, p, p)   forcing of days t..t+H (vars2d order: taux, tauy)
    fns  (H+1, Fn, p, p)  forcing_next fields of days t..t+H, box-smoothed
                          over +-forcing_next_smooth_deg if configured
    y    (H, out, p, p)   true state of days t+1..t+H (the targets)
    st   (6, p, p)        spatial static channels
    doy  (H,)             day-of-year angle of input days t..t+H-1
    m    (L, p, p)        per-level ocean mask
    e    (1, p, p)        edge weight (0 near the open boundary)
H = dataset.horizon: 1 for one-day training, 5 for rollout fine-tuning.

Iteration is window-major, like Oceryn3D Global's pair-major stream: a
worker gets runs of consecutive start days (train.day_block), so consecutive
windows share their memory-mapped day stacks, and a shuffle buffer of
dataset.cache.shuffle_windows windows mixes patches across days. Each epoch
draws a new train.days_per_epoch_frac share of start days, the same for all
workers (shards are disjoint); val/test replay the same draw every epoch.

Self-check (config only, no data): python src/dataset.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg                                      # noqa: E402
from fields import Static, cache_path, load_days            # noqa: E402
from grid import crop, doy_angle, level_mask, patch_origins, smooth_ocean  # noqa: E402

try:
    from torch.utils.data import IterableDataset as _Base
except Exception:  # pragma: no cover - numpy-only use
    _Base = object


def window_starts(days, horizon: int):
    """Indices i such that days[i..i+horizon] are consecutive calendar days."""
    ds = [r["date"] for r in days]
    out = []
    for i in range(len(ds) - horizon):
        if (ds[i + horizon] - ds[i]).days == horizon:
            out.append(i)
    return out


def epoch_starts(starts, frac: float, seed: int, epoch: int, wid: int = 0,
                 nworkers: int = 1, block: int = 1):
    """This worker's start indices for one epoch: runs of `block`
    consecutive starts, a `frac` share drawn with an RNG seeded by
    (seed, epoch) only (same draw in every worker), dealt round-robin."""
    starts = list(starts)
    block = max(1, int(block))
    runs = [starts[i:i + block] for i in range(0, len(starts), block)]
    if frac < 1.0 and runs:
        k = max(1, round(len(runs) * frac))
        pick = np.random.default_rng([seed, epoch]).choice(len(runs), k, replace=False)
        runs = [runs[i] for i in sorted(pick)]
    return [s for run in runs[wid::nworkers] for s in run]


def projected_ram_gb(cfg: Cfg, num_workers: int, windows: int | None = None) -> float:
    """Worst-case resident bytes of the day stacks a run of workers touches
    (memory-mapped, so it is reclaimable page cache rather than heap)."""
    k = windows or int(cfg.dataset.get("cache", {}).get("shuffle_windows", 4))
    per_day = cfg.n_dyn * cfg.h * cfg.w * 2
    return per_day * (k + cfg.horizon) * max(1, num_workers) / 1e9


class DayStacks:
    """LRU of memory-mapped day stacks (and their smoothed forcing_next)."""

    def __init__(self, cfg: Cfg, static: Static, size: int):
        self.cfg, self.static, self.size = cfg, static, max(2, size)
        self.d = OrderedDict()
        self.fn_idx = cfg.forcing_index(cfg.forcing_next)
        self.k_smooth = int(round(cfg.forcing_next_smooth_deg / cfg.dlat))

    def get(self, day):
        if day in self.d:
            self.d.move_to_end(day)
            return self.d[day]
        p = cache_path(self.cfg.paths["cache_dir"], day)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: build the cache for {day.year} with "
                                    f"src/build_cache.py --year {day.year}")
        a = np.load(p, mmap_mode="r")
        if a.shape != (self.cfg.n_dyn, self.cfg.h, self.cfg.w):
            raise ValueError(f"{p} is {a.shape}, config needs "
                             f"({self.cfg.n_dyn}, {self.cfg.h}, {self.cfg.w}): another "
                             "variable set or grid? Point paths.cache_dir at this config's cache.")
        fn = None
        if self.cfg.forcing_next and self.k_smooth > 0:
            fn = smooth_ocean(np.asarray(a[self.fn_idx], np.float32), self.static.ocean,
                              self.k_smooth).astype(np.float16)
        self.d[day] = (a, fn)
        while len(self.d) > self.size:
            self.d.popitem(last=False)
        return self.d[day]


class WindowStream(_Base):
    def __init__(self, config_path: str, split: str = "train", frac: float = 1.0,
                 seed: int = 0, block: int = 1, horizon: int | None = None,
                 max_patches: int | None = None):
        self.cfg = Cfg.load(config_path)
        self.H = int(horizon or self.cfg.horizon)
        self.split, self.frac, self.seed, self.block = split, frac, seed, block
        self.epoch = 0
        self.static = Static(self.cfg.paths["static_npz"], self.cfg)
        self.days = load_days(self.cfg.paths["days_csv"], split)
        self.starts = window_starts(self.days, self.H)
        if not self.starts:
            raise ValueError(f"no {self.H + 1}-day windows in split {split!r} of "
                             f"{self.cfg.paths['days_csv']}")
        c = self.cfg
        self.origins = patch_origins(c.h, c.w, c.patch, c.stride)
        # Mostly-land patches carry almost no loss: drop them once, here.
        ok = [(r, cc) for r, cc in self.origins
              if crop(self.static.ocean, r, cc, c.patch).mean() >= 1.0 - float(
                  c.dataset.get("land_frac_max", 0.9))]
        self.origins = ok[:max_patches] if max_patches else ok
        self.f_idx = c.forcing_index(c.forcing_vars)
        self.fn_idx = c.forcing_index(c.forcing_next)
        self.lmask = level_mask(self.static.n_valid, c.n_levels)            # (L,h,w)

    def __len__(self):
        return int(round(len(self.starts) * self.frac)) * len(self.origins)

    def _sample(self, win, oi):
        c, p = self.cfg, self.cfg.patch
        r, cc = self.origins[oi]
        stacks, days = win
        H = self.H

        def cr(a):
            return np.asarray(crop(a, r, cc, p), np.float32)

        x0 = cr(stacks[0][0])
        frc = np.stack([cr(s[0][self.f_idx]) for s in stacks])
        if c.forcing_next:
            fns = np.stack([cr(s[1]) if s[1] is not None else cr(s[0][self.fn_idx])
                            for s in stacks])
        else:
            fns = np.zeros((H + 1, 0, p, p), np.float32)
        y = np.stack([cr(s[0][:c.out_ch]) for s in stacks[1:]])
        st = cr(self.static.spatial)
        doy = np.array([doy_angle(d) for d in days[:H]], np.float32)
        m = cr(self.lmask)
        e = cr(self.static.edge)[None]
        return x0, frc, fns, y, st, doy, m, e

    def __iter__(self):
        import torch
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        epoch = self.epoch if self.split == "train" else 0
        self.epoch += 1
        mine = epoch_starts(self.starts, self.frac, self.seed, epoch, wid, nw, self.block)
        rng = np.random.default_rng([self.seed, epoch, 1 + wid])
        k = max(1, int(self.cfg.dataset.get("cache", {}).get("shuffle_windows", 4)))
        cache = DayStacks(self.cfg, self.static, k + self.H + 2)
        todo = iter(mine)
        live = []                                   # [window, pending origins]

        def refill():
            while len(live) < k:
                i = next(todo, None)
                if i is None:
                    return
                days = [self.days[i + j]["date"] for j in range(self.H + 1)]
                live.append([([cache.get(d) for d in days], days),
                             list(rng.permutation(len(self.origins)))])

        refill()
        while live:
            left = np.array([len(e[1]) for e in live], np.float64)
            j = int(rng.choice(len(live), p=left / left.sum()))
            win, pending = live[j]
            s = self._sample(win, pending.pop())
            if not pending:
                live.pop(j)
                refill()
            yield tuple(torch.from_numpy(np.ascontiguousarray(t)) for t in s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="loader self-check from the config alone")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--num-workers", type=int, default=12)
    a = ap.parse_args(argv)
    c = Cfg.load(a.config)
    org = patch_origins(c.h, c.w, c.patch, c.stride)
    print(f"model grid      : {c.h} x {c.w} (coarsen {c.coarsen}), {c.dlat:.4f} deg")
    print(f"input channels  : {c.in_ch}   output channels: {c.out_ch}")
    print(f"patch origins   : {len(org)} (before dropping mostly-land patches)")
    print(f"horizon         : {c.horizon} day(s); cached day {c.n_dyn * c.h * c.w * 2 / 1e6:.0f} MB")
    print(f"projected RAM   : {projected_ram_gb(c, a.num_workers):.0f} GB of mapped stacks "
          f"at {a.num_workers} workers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
