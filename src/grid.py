"""
Grid helpers, pure numpy: coarsening, coordinate matching, bilinear
regridding, normalization, patch origins and cropping, region masks.

Kept free of xarray/netCDF/torch so the tests exercise them without data.
"""
from __future__ import annotations

import math

import numpy as np

FILL_THRESHOLD = 1e19          # |x| > this is fill, as in Oceryn3D Global


def invalid(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return ~np.isfinite(a) | (np.abs(np.nan_to_num(a, nan=0.0)) > FILL_THRESHOLD)


# ---------------------------------------------------------------------------
# Coarsening (1/12 -> 1/4 deg prototype)
# ---------------------------------------------------------------------------
def coarsen_nan(a: np.ndarray, c: int) -> np.ndarray:
    """Block-mean the last two axes by c, NaN-aware.

    A coarse cell is the mean of its valid fine cells and is valid when at
    least half of them are (5 of 9 for c=3), so a coast does not flip
    between land and ocean depending on one fine cell. Invalid -> NaN.
    """
    a = np.asarray(a, np.float32)
    if c == 1:
        return np.where(invalid(a), np.nan, a).astype(np.float32)
    *lead, h, w = a.shape
    if h % c or w % c:
        raise ValueError(f"shape {a.shape} not divisible by coarsen={c}")
    b = a.reshape(*lead, h // c, c, w // c, c)
    ok = ~invalid(b)
    n = ok.sum(axis=(-3, -1))
    s = np.where(ok, b, 0.0).sum(axis=(-3, -1), dtype=np.float64)
    out = np.where(n * 2 >= c * c, s / np.maximum(n, 1), np.nan)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def match_axis(have: np.ndarray, want: np.ndarray, tol: float, what: str) -> np.ndarray:
    """Index into `have` of the nearest value to each of `want`; raises when
    any is farther than tol (wrong domain, wrong grid, missing rows)."""
    have = np.asarray(have, np.float64)
    want = np.asarray(want, np.float64)
    order = np.argsort(have)
    hs = have[order]
    j = np.clip(np.searchsorted(hs, want), 1, len(hs) - 1)
    pick = np.where(np.abs(hs[j - 1] - want) <= np.abs(hs[j] - want), j - 1, j)
    err = np.abs(hs[pick] - want)
    if (err > tol).any():
        k = int(np.argmax(err))
        raise ValueError(
            f"{what}: wanted {want[k]:.5f}, nearest in file is {hs[pick[k]]:.5f} "
            f"(off by {err[k]:.5f} > {tol}). The file does not cover the configured "
            f"grid; file range {hs[0]:.4f}..{hs[-1]:.4f}, {len(hs)} values.")
    return order[pick]


def match_depths(have, want, rel_tol: float = 0.01) -> np.ndarray:
    """Level index in `have` for each configured depth, within rel_tol."""
    have = np.asarray(have, np.float64)
    idx = []
    for d in want:
        k = int(np.argmin(np.abs(have - d)))
        if abs(have[k] - d) > rel_tol * max(d, 1.0):
            raise ValueError(
                f"configured depth {d:g} m has no GLORYS level within {rel_tol:.0%}: "
                f"nearest is {have[k]:g} m. File levels: "
                + ", ".join(f"{x:g}" for x in have[:40]))
        idx.append(k)
    if len(set(idx)) != len(idx):
        raise ValueError(f"two configured depths map to the same level: {idx}")
    return np.asarray(idx)


def lin_weights(src: np.ndarray, dst: np.ndarray):
    """(i0, w) for linear interpolation of ascending src at dst points:
    value = (1 - w) * f[i0] + w * f[i0 + 1]. Raises on extrapolation."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if (dst < src[0] - 1e-9).any() or (dst > src[-1] + 1e-9).any():
        raise ValueError(f"interpolation outside the source grid: source "
                         f"{src[0]:.3f}..{src[-1]:.3f}, wanted "
                         f"{dst.min():.3f}..{dst.max():.3f}. Download a wider area.")
    i0 = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, len(src) - 2)
    w = (dst - src[i0]) / (src[i0 + 1] - src[i0])
    return i0, np.clip(w, 0.0, 1.0)


def bilinear(field: np.ndarray, lat_s, lon_s, lat_d, lon_d) -> np.ndarray:
    """(..., ny_s, nx_s) on ascending lat_s/lon_s -> (..., ny_d, nx_d)."""
    iy, wy = lin_weights(lat_s, lat_d)
    ix, wx = lin_weights(lon_s, lon_d)
    f = np.asarray(field, np.float64)
    a = f[..., iy, :] * (1 - wy)[:, None] + f[..., iy + 1, :] * wy[:, None]
    out = a[..., ix] * (1 - wx) + a[..., ix + 1] * wx
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization (same formulas as Oceryn3D Global)
# ---------------------------------------------------------------------------
def normalize_3d(arr, mean, std) -> np.ndarray:
    """(L,H,W), per-level mean/std; invalid -> 0 (neutral after z-scoring)."""
    out = np.asarray(arr, np.float32).copy()
    bad = invalid(out)
    m = np.asarray(mean, np.float32).reshape(-1, 1, 1)
    s = np.asarray(std, np.float32).reshape(-1, 1, 1)
    out = (out - m) / np.where(s <= 0, 1.0, s)
    out[bad] = 0.0
    return out


def normalize_2d(arr, mean: float, std: float) -> np.ndarray:
    out = np.asarray(arr, np.float32).copy()
    bad = invalid(out)
    out = (out - float(mean)) / (float(std) if std > 0 else 1.0)
    out[bad] = 0.0
    return out


def depth_encoding(depth_m, h: int, w: int) -> np.ndarray:
    """(L,H,W) normalized log-depth, as in Oceryn3D Global."""
    d = np.asarray(depth_m, np.float32)
    logd = np.log1p(d)
    logd = (logd - logd.min()) / (logd.max() - logd.min() + 1e-9)
    return np.broadcast_to(logd.reshape(-1, 1, 1), (d.size, h, w)).astype(np.float32)


def doy_angle(day) -> float:
    """2 pi (doy - 1) / 365.25, Oceryn3D Global's calendar convention."""
    return 2.0 * math.pi * (day.timetuple().tm_yday - 1) / 365.25


def level_mask(n_valid: np.ndarray, n_levels: int) -> np.ndarray:
    """(..,H,W) valid-level counts -> (L,..,H,W) float32, 1 = ocean."""
    k = np.arange(n_levels, dtype=np.int16).reshape((-1,) + (1,) * n_valid.ndim)
    return (k < n_valid[None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Patches (no wrap: the domain is regional)
# ---------------------------------------------------------------------------
def _starts(n: int, patch: int, stride: int):
    s = list(range(0, max(1, n - patch + 1), stride))
    if n > patch and s[-1] != n - patch:
        s.append(n - patch)
    return s or [0]


def patch_origins(h: int, w: int, patch: int, stride: int):
    """Top-left (row, col) origins covering the grid; the last row/col is
    pinned to the edge so every cell is inside some patch."""
    return [(r, c) for r in _starts(h, patch, stride) for c in _starts(w, patch, stride)]


def crop(a, r: int, c: int, p: int):
    return a[..., r:r + p, c:c + p]


def edge_weight(h: int, w: int, margin: int) -> np.ndarray:
    """(H,W) 1 inside, 0 within `margin` cells of the domain edge."""
    e = np.ones((h, w), np.float32)
    if margin > 0:
        e[:margin] = e[-margin:] = 0.0
        e[:, :margin] = e[:, -margin:] = 0.0
    return e


def region_mask(lat, lon, box) -> np.ndarray:
    """(H,W) bool for [lon_min, lon_max, lat_min, lat_max]."""
    lo0, lo1, la0, la1 = (float(x) for x in box)
    return ((lat[:, None] >= la0) & (lat[:, None] <= la1)
            & (lon[None, :] >= lo0) & (lon[None, :] <= lo1))


# ---------------------------------------------------------------------------
# Box smoothing of forcing (as Oceryn3D Global's smooth_forcing, no wrap)
# ---------------------------------------------------------------------------
def box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over a (2k+1)^2 box, zero beyond the edges (regional: no wrap)."""
    a = np.asarray(a, np.float64)
    if k == 0:
        return a
    c = np.cumsum(np.pad(a, ((0, 0), (k + 1, k))), axis=1)
    a = c[:, 2 * k + 1:] - c[:, :-2 * k - 1]
    c = np.cumsum(np.pad(a, ((k + 1, k), (0, 0))), axis=0)
    return c[2 * k + 1:] - c[:-2 * k - 1]


def smooth_ocean(f: np.ndarray, ocean: np.ndarray, k: int) -> np.ndarray:
    """(C,H,W) box mean over +-k cells on ocean points only; land stays 0.
    Smoothing z-scored values equals smoothing physical ones (affine)."""
    if k <= 0:
        return f
    out = np.empty_like(f)
    m = ocean.astype(np.float64)
    den = np.maximum(box_sum(m, k), 1.0)
    for ch in range(f.shape[0]):
        s = box_sum(np.where(ocean, f[ch].astype(np.float64), 0.0), k) / den
        out[ch] = np.where(ocean, s, 0.0).astype(f.dtype)
    return out
