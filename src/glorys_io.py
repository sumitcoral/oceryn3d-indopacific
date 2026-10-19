"""
GLORYS12 monthly files: naming, reading one day on the model grid.

Raw files come from scripts/download_glorys.py, one per month:
    <glorys_dir>/YYYY/glorys12_YYYY_MM.nc
with uo, vo, thetao, so on (time, depth, latitude, longitude) and zos on
(time, latitude, longitude), already cut to the domain and the configured
depths. They are matched here BY COORDINATE VALUE, never by position: rows,
columns and levels are looked up against the config's axes and depths, and a
file that does not cover them raises. Then the fields are coarsened if the
config says so (1/4 deg prototype).
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime

import numpy as np

from config import Cfg
from grid import coarsen_nan, match_axis, match_depths

MONTH_RE = re.compile(r"glorys12_(\d{4})_(\d{2})\.nc$")


def month_path(glorys_dir: str, year: int, month: int) -> str:
    return os.path.join(glorys_dir, f"{year:04d}", f"glorys12_{year:04d}_{month:02d}.nc")


def list_month_files(glorys_dir: str):
    """[(year, month, path)] sorted, for every monthly file present."""
    out = []
    if not os.path.isdir(glorys_dir):
        return out
    for sub in sorted(os.listdir(glorys_dir)):
        d = os.path.join(glorys_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            m = MONTH_RE.match(fn)
            if m:
                out.append((int(m.group(1)), int(m.group(2)), os.path.join(d, fn)))
    return out


def _dates(ds) -> list:
    """Calendar date of each time step (GLORYS daily means; the stamp's hour
    is dropped, whatever convention the file uses)."""
    import netCDF4
    tv = ds.variables["time"]
    ts = netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"),
                          only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return [date(t.year, t.month, t.day) for t in ts]


def _coord(ds, *names):
    for n in names:
        if n in ds.variables:
            return np.asarray(ds.variables[n][:], np.float64)
    raise KeyError(f"none of {names} in {ds.filepath()}")


class GlorysMonth:
    """One open monthly file with its row/column/level index into the config grid."""

    def __init__(self, path: str, cfg: Cfg):
        import netCDF4
        self.path, self.cfg = path, cfg
        self.ds = netCDF4.Dataset(path)
        self.ds.set_auto_maskandscale(True)
        self.dates = _dates(self.ds)
        lat_n, lon_n = cfg.native_axes()
        tol = 0.25 * float(cfg.grid["dlat"])
        self.iy = match_axis(_coord(self.ds, "latitude", "lat"), lat_n, tol,
                             f"{os.path.basename(path)} latitude")
        self.ix = match_axis(_coord(self.ds, "longitude", "lon"), lon_n, tol,
                             f"{os.path.basename(path)} longitude")
        self.iz = match_depths(_coord(self.ds, "depth", "lev"), cfg.depth_m)
        self.names = cfg.variables["glorys_names"]

    def close(self):
        self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _read(self, var: str, t: int, three_d: bool, levels=None) -> np.ndarray:
        v = self.ds.variables[var]
        # Contiguous hyperslab first (fast), then fancy-index the subset.
        if three_d:
            iz = self.iz if levels is None else self.iz[np.asarray(levels)]
            z0, z1 = int(iz.min()), int(iz.max()) + 1
            y0, y1 = int(self.iy.min()), int(self.iy.max()) + 1
            x0, x1 = int(self.ix.min()), int(self.ix.max()) + 1
            a = v[t, z0:z1, y0:y1, x0:x1]
            a = np.ma.filled(np.ma.asarray(a, np.float32), np.nan)
            a = a[iz - z0][:, self.iy - y0][:, :, self.ix - x0]
        else:
            y0, y1 = int(self.iy.min()), int(self.iy.max()) + 1
            x0, x1 = int(self.ix.min()), int(self.ix.max()) + 1
            a = v[t, y0:y1, x0:x1]
            a = np.ma.filled(np.ma.asarray(a, np.float32), np.nan)
            a = a[self.iy - y0][:, self.ix - x0]
        return a

    def read_day(self, day: date, names=None, levels=None) -> dict:
        """{model var: physical field on the MODEL grid}, NaN where invalid.
        3D (L,h,w), 2D (h,w). names defaults to every GLORYS variable of the
        config (vars3d and the non-forcing vars2d); levels (indices into
        grid.depth_m) reads only those levels of the 3D fields."""
        if day not in self.dates:
            raise KeyError(f"{day} is not in {self.path}")
        t = self.dates.index(day)
        cfg = self.cfg
        want = names or [v for v in cfg.vars3d + cfg.vars2d if v not in cfg.forcing_vars]
        out = {}
        for v in want:
            a = self._read(self.names[v], t, v in cfg.vars3d, levels)
            out[v] = coarsen_nan(a, cfg.coarsen)
        return out


def read_day(path: str, cfg: Cfg, day: date, names=None) -> dict:
    with GlorysMonth(path, cfg) as m:
        return m.read_day(day, names)


def as_date(s) -> date:
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s))
