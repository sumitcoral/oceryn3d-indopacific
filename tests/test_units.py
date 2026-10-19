"""Unit tests of the pure helpers and the config contracts (no data)."""
import os
from datetime import date, timedelta

import numpy as np
import pytest
import yaml

from config import Cfg, load_config
from grid import (bilinear, coarsen_nan, match_axis, match_depths, patch_origins,
                  smooth_ocean, normalize_3d)
from dataset import epoch_starts, window_starts
import synthetic

REPO = synthetic.REPO
DEFAULT = os.path.join(REPO, "config", "default.yaml")


def test_default_config_contracts():
    c = Cfg.load(DEFAULT)
    assert (c.h, c.w) == (720, 1800)
    assert (c.in_ch, c.out_ch, c.n_dyn) == (133, 97, 99)
    lat, lon = c.axes()
    assert lat[0] == pytest.approx(-30.0) and lon[-1] == pytest.approx(30 + 1799 / 12)
    assert c.forcing_vars == ["taux", "tauy"]


def test_derived_configs_load():
    q = Cfg.load(os.path.join(REPO, "config", "proto_quarter.yaml"))
    assert (q.h, q.w, q.coarsen) == (240, 600, 3)
    lat, lon = q.axes()
    assert lon[0] == pytest.approx(30.0 + 1 / 12)          # centre of the first 3x3 block
    r = Cfg.load(os.path.join(REPO, "config", "rollout5.yaml"))
    assert r.horizon == 5 and r.train["init_from"].endswith("ip_daily_v1/best.pt")
    assert r.train["num_workers"] == 12                     # inherited


def _variant(tmp_path, **over):
    raw = load_config(DEFAULT)
    for blk, kv in over.items():
        raw[blk].update(kv)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


def test_contract_violations_raise(tmp_path):
    with pytest.raises(ValueError, match="C2"):
        Cfg.load(_variant(tmp_path, model={"in_ch": 131}))
    with pytest.raises(ValueError, match="C1"):
        Cfg.load(_variant(tmp_path, variables={"vars2d": ["taux", "ssh", "tauy"]}))
    with pytest.raises(ValueError, match="forcing_next"):
        Cfg.load(_variant(tmp_path, variables={"forcing_next": ["ssh"]}, model={"in_ch": 132}))
    with pytest.raises(ValueError, match="C4"):
        Cfg.load(_variant(tmp_path, grid={"n_levels": 23}))


def test_coarsen_nan_majority_rule():
    a = np.arange(36, dtype=np.float32).reshape(6, 6)
    a[0:3, 0:3][0, :2] = np.nan          # 2 of 9 missing -> still valid
    a[0:3, 3:6][:2, :] = np.nan          # 6 of 9 missing -> invalid
    c = coarsen_nan(a, 3)
    blk = np.array([2, 6, 7, 8, 12, 13, 14], np.float32)
    assert c[0, 0] == pytest.approx(blk.mean())
    assert np.isnan(c[0, 1])
    assert c[1, 1] == pytest.approx(a[3:6, 3:6].mean())


def test_match_axis_and_depths():
    have = -30 + np.arange(50) / 12
    assert list(match_axis(have[::-1], have[3:6], 0.01, "x")) == [46, 45, 44]
    with pytest.raises(ValueError, match="does not cover"):
        match_axis(have, np.array([10.0]), 0.01, "lat")
    idx = match_depths(synthetic.GLORYS_DEPTHS, Cfg.load(DEFAULT).depth_m)
    assert len(idx) == 24 and idx[0] == 0 and idx[-1] == 35
    with pytest.raises(ValueError, match="no GLORYS level"):
        match_depths(synthetic.GLORYS_DEPTHS, [100.0])


def test_bilinear_is_exact_on_linear_fields():
    lat_s, lon_s = np.arange(-5, 5.01, 0.25), np.arange(28, 36.01, 0.25)
    f = 2 * lat_s[:, None] - 3 * lon_s[None, :]
    lat_d, lon_d = np.linspace(-4, 4, 7), np.linspace(29.1, 35.3, 9)
    out = bilinear(f, lat_s, lon_s, lat_d, lon_d)
    np.testing.assert_allclose(out, 2 * lat_d[:, None] - 3 * lon_d[None, :], atol=1e-4)
    with pytest.raises(ValueError, match="outside"):
        bilinear(f, lat_s, lon_s, lat_d, np.array([40.0]))


def test_patch_origins_cover_grid():
    org = patch_origins(720, 1800, 256, 224)
    cover = np.zeros((720, 1800), bool)
    for r, c in org:
        assert r + 256 <= 720 and c + 256 <= 1800
        cover[r:r + 256, c:c + 256] = True
    assert cover.all() and len(org) == 4 * 8


def test_window_starts_respects_gaps():
    d0 = date(2000, 1, 1)
    ds = [d0 + timedelta(days=k) for k in (0, 1, 2, 3, 5, 6, 7)]
    rows = [{"date": d} for d in ds]
    assert window_starts(rows, 1) == [0, 1, 2, 4, 5]
    assert window_starts(rows, 2) == [0, 1, 4]


def test_epoch_starts_disjoint_and_change():
    st = list(range(100))
    a = [epoch_starts(st, 0.5, 0, 3, w, 3, 4) for w in range(3)]
    flat = [x for s in a for x in s]
    assert len(flat) == len(set(flat)) == 48 and all(s for s in a)   # 25 runs of 4, round(12.5) = 12
    b = [x for w in range(3) for x in epoch_starts(st, 0.5, 0, 4, w, 3, 4)]
    assert set(flat) != set(b)


def test_smooth_ocean_keeps_land_zero():
    f = np.random.default_rng(0).normal(size=(1, 20, 30)).astype(np.float32)
    ocean = np.ones((20, 30), bool)
    ocean[5:10, 5:10] = False
    s = smooth_ocean(f, ocean, 2)
    assert (s[0][~ocean] == 0).all()
    assert abs(s[0][ocean].std()) < abs(f[0][ocean].std())


def test_normalize_invalid_to_zero():
    a = np.array([[[1.0, np.nan], [1e20, 3.0]]], np.float32)
    n = normalize_3d(a, [1.0], [2.0])
    assert n[0, 0, 1] == 0 and n[0, 1, 0] == 0 and n[0, 1, 1] == pytest.approx(1.0)
