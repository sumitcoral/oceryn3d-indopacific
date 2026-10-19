"""
Configuration: one YAML, read by every stage.

load_config(path) reads a YAML file; a file with `base: other.yaml` starts
from that file (path relative to its own directory) and overrides it. Dicts
merge recursively and everything else, lists included, replaces. Cfg wraps the
merged dict with the derived quantities every stage needs (model grid after
coarsening, axes, channel counts) and asserts the channel contract C1-C4 of
config/default.yaml before anything expensive runs.

Pure python + numpy, so the tests import it without torch or xarray.
"""
from __future__ import annotations

import copy
import os
from datetime import date

import numpy as np
import yaml

# Static channels, in the order they are appended to the input. The same order
# as Oceryn3D Global (ocean-emulator dataset.STATIC_ORDER).
STATIC_ORDER = ["land_mask", "bathymetry", "sin_lat", "cos_lat",
                "sin_lon", "cos_lon", "sin_doy", "cos_doy"]
CALENDAR = ("sin_doy", "cos_doy")          # per day; the rest come from static_fields.npz
SPATIAL_STATIC = [k for k in STATIC_ORDER if k not in CALENDAR]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, _seen=None) -> dict:
    """Merged config dict. Raises on a missing file or a base cycle."""
    path = os.path.abspath(path)
    seen = set(_seen or ())
    if path in seen:
        raise ValueError(f"config base cycle at {path}")
    seen.add(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base), seen)
        raw = _merge(parent, raw)
    return raw


def parse_date(s) -> date:
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s).replace("_", "-"))


class Cfg:
    """Merged config plus derived values. Attribute access to the blocks:
    cfg.paths, cfg.grid, cfg.variables, ... are the raw dicts."""

    def __init__(self, raw: dict, path: str = ""):
        for blk in ("paths", "grid", "variables", "data", "dataset", "model"):
            if blk not in raw:
                raise KeyError(f"config is missing the {blk!r} block: {path}")
        self.raw, self.path = raw, path
        for k, v in raw.items():
            setattr(self, k, v)
        g, v, d = raw["grid"], raw["variables"], raw["dataset"]
        for k in ("lon0", "lat0", "dlon", "dlat", "width", "height", "n_levels", "depth_m"):
            if k not in g:
                raise KeyError(f"config grid.{k} is required (see config/default.yaml)")
        self.coarsen = int(g.get("coarsen", 1))
        if self.coarsen < 1 or g["width"] % self.coarsen or g["height"] % self.coarsen:
            raise ValueError(f"grid.coarsen={self.coarsen} must divide width "
                             f"{g['width']} and height {g['height']}")
        self.native_h, self.native_w = int(g["height"]), int(g["width"])
        self.h, self.w = self.native_h // self.coarsen, self.native_w // self.coarsen
        self.n_levels = int(g["n_levels"])
        self.depth_m = [float(x) for x in g["depth_m"]]
        self.vars3d, self.vars2d = list(v["vars3d"]), list(v["vars2d"])
        self.target_vars = list(v["target_vars"])
        self.forcing_vars = [x for x in self.vars2d if x not in self.target_vars]
        self.forcing_next = list(v.get("forcing_next") or [])
        self.forcing_next_smooth_deg = float(v.get("forcing_next_smooth_deg") or 0.0)
        self.mask_vars = list(v.get("mask_vars", ["temp", "u", "v"]))
        self.patch, self.stride = int(d["patch"]), int(d["stride"])
        self.horizon = int(d.get("horizon", 1))
        self.edge_margin = int(d.get("edge_margin_px", 0))
        self.depth_encoding = bool(d.get("depth_encoding", True))
        self.static_channels = list(d.get("static_channels", STATIC_ORDER))
        self.validate()

    @classmethod
    def load(cls, path: str) -> "Cfg":
        return cls(load_config(path), path)

    # -- grid -----------------------------------------------------------------
    @property
    def dlon(self) -> float:
        """Model-grid spacing (deg) after coarsening."""
        return float(self.grid["dlon"]) * self.coarsen

    @property
    def dlat(self) -> float:
        return float(self.grid["dlat"]) * self.coarsen

    def axes(self):
        """(lat (h,), lon (w,)) of the MODEL grid, ascending, deg. A coarse
        cell's centre is the mean of its block's native centres."""
        c = self.coarsen
        off = (c - 1) / 2.0
        lat = float(self.grid["lat0"]) + float(self.grid["dlat"]) * (c * np.arange(self.h) + off)
        lon = float(self.grid["lon0"]) + float(self.grid["dlon"]) * (c * np.arange(self.w) + off)
        return lat, lon

    def native_axes(self):
        lat = float(self.grid["lat0"]) + float(self.grid["dlat"]) * np.arange(self.native_h)
        lon = float(self.grid["lon0"]) + float(self.grid["dlon"]) * np.arange(self.native_w)
        return lat, lon

    # -- channels -------------------------------------------------------------
    @property
    def targets_3d(self):
        return [x for x in self.target_vars if x in self.vars3d]

    @property
    def targets_2d(self):
        return [x for x in self.target_vars if x in self.vars2d]

    @property
    def n_dyn(self) -> int:
        """Channels of a cached day: vars3d x levels + vars2d (99)."""
        return len(self.vars3d) * self.n_levels + len(self.vars2d)

    @property
    def out_ch(self) -> int:
        return len(self.targets_3d) * self.n_levels + len(self.targets_2d)

    @property
    def in_ch(self) -> int:
        return (self.n_dyn + (self.n_levels if self.depth_encoding else 0)
                + len(self.static_channels) + len(self.forcing_next))

    def forcing_index(self, names) -> list:
        """Channels of 2D fields `names` inside a cached (n_dyn,H,W) day."""
        off = len(self.vars3d) * self.n_levels
        return [off + self.vars2d.index(n) for n in names]

    def target_slices(self):
        """[(name, start, width)] in the (out_ch,H,W) state."""
        out, off = [], 0
        for x in self.targets_3d:
            out.append((x, off, self.n_levels))
            off += self.n_levels
        for x in self.targets_2d:
            out.append((x, off, 1))
            off += 1
        return out

    # -- data -----------------------------------------------------------------
    def split_of(self, day) -> str | None:
        y = parse_date(day).year
        for name, (a, b) in self.data["splits"].items():
            if int(a) <= y <= int(b):
                return name
        return None

    @property
    def skip_dates(self) -> set:
        return {parse_date(s) for s in (self.data.get("skip_dates") or [])}

    # -- contracts ------------------------------------------------------------
    def validate(self) -> None:
        t3, t2 = self.targets_3d, self.targets_2d
        unknown = [x for x in self.target_vars if x not in self.vars3d + self.vars2d]
        if unknown:
            raise ValueError(f"target_vars {unknown} are in neither vars3d nor vars2d: "
                             "every target must also be an input (residual model)")
        if self.target_vars != t3 + t2:
            raise ValueError(f"C1: 3D targets must precede 2D ones in target_vars, "
                             f"got {self.target_vars}")
        if self.vars3d[:len(t3)] != t3:
            raise ValueError(f"C1: vars3d must lead with the 3D targets {t3}, got "
                             f"{self.vars3d}. The model adds x[:, :out_ch] as its "
                             "baseline, so the leading input channels must be the "
                             "targets in target order.")
        if self.vars2d[:len(t2)] != t2:
            raise ValueError(f"C1: vars2d must lead with the 2D targets {t2}, got "
                             f"{self.vars2d}: the 2D block follows the 3D block, so a "
                             "2D target is only inside the residual prefix if it is first.")
        bad = [f for f in self.forcing_next if f not in self.forcing_vars]
        if bad:
            raise ValueError(f"forcing_next {bad}: each entry must be a vars2d field "
                             "that is not a target (a target at t+1 is the answer)")
        if len(set(self.forcing_next)) != len(self.forcing_next):
            raise ValueError(f"forcing_next has duplicates: {self.forcing_next}")
        if self.forcing_next_smooth_deg < 0 or (self.forcing_next_smooth_deg
                                                and not self.forcing_next):
            raise ValueError("forcing_next_smooth_deg must be >= 0 and needs forcing_next")
        if len(self.depth_m) != self.n_levels:
            raise ValueError(f"C4: grid.depth_m has {len(self.depth_m)} values, "
                             f"n_levels is {self.n_levels}")
        if list(np.sort(self.depth_m)) != self.depth_m:
            raise ValueError("grid.depth_m must be increasing")
        if self.static_channels != STATIC_ORDER:
            raise ValueError(f"dataset.static_channels must be {STATIC_ORDER}")
        if self.horizon < 1:
            raise ValueError("dataset.horizon must be >= 1")
        m = self.model
        if int(m["in_ch"]) != self.in_ch:
            raise ValueError(f"C2: model.in_ch={m['in_ch']} but the config implies "
                             f"{self.in_ch} (= {len(self.vars3d)}*{self.n_levels} + "
                             f"{len(self.vars2d)} + depth {self.n_levels if self.depth_encoding else 0}"
                             f" + static {len(self.static_channels)} + forcing_next "
                             f"{len(self.forcing_next)})")
        if int(m["out_ch"]) != self.out_ch:
            raise ValueError(f"C3: model.out_ch={m['out_ch']} but the config implies "
                             f"{self.out_ch}")
        for x in self.vars3d + self.vars2d:
            if x in self.forcing_vars:
                continue
            if x not in self.variables.get("glorys_names", {}):
                raise ValueError(f"variables.glorys_names has no entry for {x!r}")
        if self.forcing_vars and "forcing" not in self.raw:
            raise KeyError("config has forcing fields in vars2d but no forcing block")
        if self.patch % 16:
            raise ValueError(f"dataset.patch={self.patch} must be a multiple of 16 "
                             "(four 2x poolings in the U-Net)")
        if self.patch > min(self.h, self.w):
            raise ValueError(f"dataset.patch={self.patch} exceeds the model grid "
                             f"{self.h} x {self.w}")
