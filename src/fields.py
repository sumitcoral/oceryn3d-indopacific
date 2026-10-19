"""
Day stacks: physical fields -> the normalized (n_dyn, h, w) channel stack the
cache stores and the loader reads, plus the readers for the shared meta files
(days.csv, stats.json, static_fields.npz).

Channel order of a day stack (contract C1), var-major:
    u(L) v(L) temp(L) salt(L) ssh taux tauy      -> 4*24 + 3 = 99
Channels 0..out_ch-1 (0..96) are the targets, so one stack is both the
input at t and, sliced, the target at t-1's forecast.
Invalid values, land, and levels below the sea floor are 0 after z-scoring,
exactly as in Oceryn3D Global.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import date

import numpy as np

from config import Cfg, SPATIAL_STATIC
from grid import level_mask, normalize_2d, normalize_3d


# ---------------------------------------------------------------------------
# Meta files
# ---------------------------------------------------------------------------
DAYS_FIELDS = ["date", "split", "glorys_path", "forcing_path", "forcing_index"]


def load_days(days_csv: str, split: str | None = None):
    """[dict] rows of days.csv (date parsed), optionally one split, date order."""
    if not os.path.exists(days_csv):
        raise FileNotFoundError(f"{days_csv} not found: run src/build_index.py first")
    rows = []
    with open(days_csv) as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            r["date"] = date.fromisoformat(r["date"])
            r["forcing_index"] = int(r["forcing_index"])
            rows.append(r)
    rows.sort(key=lambda r: r["date"])
    return rows


def load_stats(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run src/compute_stats.py first")
    with open(path) as f:
        return json.load(f)


class Static:
    """static_fields.npz: per-level ocean mask, input channels, grid."""

    def __init__(self, path: str, cfg: Cfg):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: run src/build_static.py first")
        z = np.load(path, allow_pickle=False)
        self.n_valid = z["n_valid_levels"].astype(np.int16)
        if self.n_valid.shape != (cfg.h, cfg.w):
            raise ValueError(f"{path} is {self.n_valid.shape}, the config's grid is "
                             f"({cfg.h}, {cfg.w}): rebuild it with build_static.py")
        self.lat, self.lon = z["lat"], z["lon"]
        self.ocean = self.n_valid > 0
        self.cos_lat = z["cos_lat_weights"].astype(np.float32)
        self.edge = z["edge_weight"].astype(np.float32)
        # (6,h,w) in STATIC_ORDER without the calendar channels.
        self.spatial = np.stack([z[k].astype(np.float32) for k in SPATIAL_STATIC])


def cache_path(cache_dir: str, d: date) -> str:
    return os.path.join(cache_dir, f"{d.year:04d}", f"day_{d:%Y_%m_%d}.npy")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_day(cfg: Cfg, stats: dict, fields: dict, forcing: np.ndarray,
                 n_valid: np.ndarray) -> np.ndarray:
    """(n_dyn,h,w) float32. fields: {var: physical (L,h,w) or (h,w)} from
    glorys_io; forcing: (len(forcing_vars),h,w) physical from build_forcing."""
    lm = level_mask(n_valid, cfg.n_levels)                  # (L,h,w)
    ocean = (n_valid > 0).astype(np.float32)
    chans = []
    for v in cfg.vars3d:
        s = stats["vars3d"][v]
        chans.append(normalize_3d(fields[v], s["mean"], s["std"]) * lm)
    for v in cfg.vars2d:
        s = stats["vars2d"][v]
        if v in cfg.forcing_vars:
            a = forcing[cfg.forcing_vars.index(v)]
        else:
            a = fields[v]
        chans.append((normalize_2d(a, s["mean"], s["std"]) * ocean)[None])
    out = np.concatenate(chans, axis=0).astype(np.float32)
    if out.shape != (cfg.n_dyn, cfg.h, cfg.w):
        raise ValueError(f"day stack {out.shape} != ({cfg.n_dyn}, {cfg.h}, {cfg.w})")
    return out


def load_forcing_day(path: str, index: int) -> np.ndarray:
    """(F,h,w) float32 physical stress of one day from a monthly .npy."""
    return np.asarray(np.load(path, mmap_mode="r")[index], np.float32)
