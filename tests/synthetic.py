"""
Synthetic GLORYS12 and ERA5 files on a tiny piece of the real grid, so the
whole pipeline runs on CPU in seconds.

Grid: the real config's 1/12 deg axes from 30E/30S, 64 x 48 cells, with one
extra row and column in the files (matching must pick by coordinate). Levels:
all 36 GLORYS12 levels to 1062 m in the files (matching must pick the 24
configured ones). An island and a shelf give land and partial columns; u/v
are also missing on the island's coastal ring where temp is valid (the mask
must intersect). Fields drift in time so persistence is imperfect.
"""
from __future__ import annotations

import os
import zipfile
from datetime import date, timedelta

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GLORYS_DEPTHS = [0.494025, 1.541375, 2.645669, 3.819495, 5.078224, 6.440614, 7.92956,
                 9.572997, 11.405, 13.46714, 15.81007, 18.49556, 21.59882, 25.21141,
                 29.44473, 34.43415, 40.34405, 47.37369, 55.76429, 65.80727, 77.85385,
                 92.32607, 109.7293, 130.666, 155.8507, 186.1256, 222.4752, 266.0403,
                 318.1274, 380.213, 453.9377, 541.0889, 643.5668, 763.3331, 902.3393,
                 1062.44]
NX, NY = 64, 48
D = 1.0 / 12.0

# Days per split year: train 2000, val 2001, test 2002 (January).
MONTH_DAYS = {(2000, 1): 14, (2001, 1): 9, (2002, 1): 9}


def write_config(root: str, **over) -> str:
    cfg = {
        "base": os.path.join(REPO, "config", "default.yaml"),
        "paths": {
            "root": root,
            "glorys_dir": f"{root}/GLORYS12", "era5_dir": f"{root}/ERA5",
            "forcing_dir": f"{root}/forcing", "cache_dir": f"{root}/cache",
            "meta_dir": f"{root}/meta", "days_csv": f"{root}/meta/days.csv",
            "stats_json": f"{root}/meta/stats.json",
            "static_npz": f"{root}/meta/static_fields.npz",
            "glorys_static_nc": f"{root}/GLORYS12/glorys12_static.nc",
            "out_dir": f"{root}/runs/daily",
        },
        "grid": {"width": NX, "height": NY},
        "data": {"year_start": 2000, "year_end": 2002,
                 "splits": {"train": [2000, 2000], "val": [2001, 2001], "test": [2002, 2002]}},
        "dataset": {"patch": 32, "stride": 24, "edge_margin_px": 2,
                    "cache": {"shuffle_windows": 2}},
        "model": {"base": 8},
        "train": {"batch_size": 2, "num_workers": 0, "epochs": 2, "warmup_steps": 2,
                  "days_per_epoch_frac": 1.0, "val_days_frac": 1.0, "log_every": 0,
                  "require_cuda": False, "amp": False, "day_block": 2},
        "evaluate": {"tile": 32, "tile_stride": 24,
                     "regions": {"domain": [30.0, 36.0, -30.0, -26.0],
                                 "east": [33.0, 36.0, -30.0, -26.0]}},
    }
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    path = os.path.join(root, "test.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def axes(extra=1):
    lat = -30.0 + D * np.arange(NY + extra)
    lon = 30.0 + D * np.arange(NX + extra)
    return lat, lon


def geometry(lat, lon, depths):
    """(nz, ny, nx) valid masks for temp and for u/v."""
    yy, xx = np.meshgrid(np.arange(len(lat)), np.arange(len(lon)), indexing="ij")
    island = ((yy - 24) ** 2 + (xx - 20) ** 2) < 25
    ring = (((yy - 24) ** 2 + (xx - 20) ** 2) < 36) & ~island
    shelf_depth = np.where(xx > 50, 60.0, 5000.0)           # shallow strip in the east
    d = np.asarray(depths)[:, None, None]
    t_ok = (~island)[None] & (d < shelf_depth[None])
    uv_ok = t_ok & ~ring[None]
    return t_ok, uv_ok


def fields(day: date, lat, lon, depths):
    n = (day - date(2000, 1, 1)).days
    la, lo = np.meshgrid(np.deg2rad(lat) * 12, np.deg2rad(lon) * 12, indexing="ij")
    z = np.asarray(depths)[:, None, None]
    decay = np.exp(-z / 300.0)
    ph = 0.25 * n
    u = 0.4 * np.sin(lo + ph)[None] * decay
    v = 0.3 * np.cos(la - 0.7 * ph)[None] * decay
    t = 28.0 - 0.02 * z + 0.5 * np.sin(la + lo + 0.1 * n)[None]
    s = 34.5 + 0.0005 * z + 0.1 * np.cos(lo - 0.05 * n)[None]
    ssh = 0.3 + 0.1 * np.sin(la - lo + 0.2 * n)
    return u, v, t, s, ssh


def write_glorys_month(gdir: str, y: int, m: int, ndays: int):
    import xarray as xr
    lat, lon = axes()
    days = [date(y, m, 1) + timedelta(days=k) for k in range(ndays)]
    t_ok, uv_ok = geometry(lat, lon, GLORYS_DEPTHS)
    U, V, T, S, Z = [], [], [], [], []
    for d in days:
        u, v, t, s, ssh = fields(d, lat, lon, GLORYS_DEPTHS)
        U.append(np.where(uv_ok, u, np.nan))
        V.append(np.where(uv_ok, v, np.nan))
        T.append(np.where(t_ok, t, np.nan))
        S.append(np.where(t_ok, s, np.nan))
        Z.append(np.where(t_ok[0], ssh, np.nan))
    dims = ("time", "depth", "latitude", "longitude")
    ds = xr.Dataset(
        {"uo": (dims, np.array(U, np.float32)), "vo": (dims, np.array(V, np.float32)),
         "thetao": (dims, np.array(T, np.float32)), "so": (dims, np.array(S, np.float32)),
         "zos": (("time", "latitude", "longitude"), np.array(Z, np.float32))},
        coords={"time": np.array([np.datetime64(d, "ns") for d in days]),
                "depth": np.array(GLORYS_DEPTHS, np.float32),
                "latitude": lat.astype(np.float32), "longitude": lon.astype(np.float32)})
    enc = {k: {"dtype": "int16", "scale_factor": 0.001, "add_offset": off,
               "_FillValue": np.int16(-32767), "zlib": True}
           for k, off in (("uo", 0.0), ("vo", 0.0), ("thetao", 15.0), ("so", 30.0), ("zos", 0.0))}
    out = os.path.join(gdir, f"{y:04d}", f"glorys12_{y:04d}_{m:02d}.nc")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.to_netcdf(out, encoding=enc)
    return out


def write_era5_month(edir: str, y: int, m: int, ndays: int, as_zip=False):
    import xarray as xr
    lat = np.arange(31.0, -31.25, -0.25)                     # ERA5: descending
    lon = np.arange(29.0, 37.25, 0.25)
    days = [date(y, m, 1) + timedelta(days=k) for k in range(ndays)]
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    tx = np.array([0.1 * np.sin(np.deg2rad(lo) * 20 + 0.3 * k) for k in range(ndays)], np.float32)
    ty = np.array([0.05 * np.cos(np.deg2rad(la) * 20 - 0.2 * k) for k in range(ndays)], np.float32)
    coords = {"valid_time": np.array([np.datetime64(d, "ns") for d in days]),
              "latitude": lat, "longitude": lon}
    dims = ("valid_time", "latitude", "longitude")
    out = os.path.join(edir, f"{y:04d}", f"era5_stress_{y:04d}_{m:02d}.nc")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not as_zip:
        xr.Dataset({"avg_iews": (dims, tx), "avg_inss": (dims, ty)}, coords=coords).to_netcdf(out)
    else:
        tmp = out + ".d"
        os.makedirs(tmp, exist_ok=True)
        xr.Dataset({"avg_iews": (dims, tx)}, coords=coords).to_netcdf(f"{tmp}/x.nc")
        xr.Dataset({"avg_inss": (dims, ty)}, coords=coords).to_netcdf(f"{tmp}/y.nc")
        with zipfile.ZipFile(out, "w") as z:
            z.write(f"{tmp}/x.nc", "eastward.nc")
            z.write(f"{tmp}/y.nc", "northward.nc")
    return out


def write_world(root: str):
    for (y, m), n in MONTH_DAYS.items():
        write_glorys_month(f"{root}/GLORYS12", y, m, n)
        write_era5_month(f"{root}/ERA5", y, m, n, as_zip=(y == 2001))
