"""End-to-end on synthetic data: Stages 1-5 (conftest.world), then the loader,
training (one-day and rollout fine-tuning) and evaluation, on CPU."""
import csv
import json
import os
from datetime import date

import numpy as np
import pytest
import torch
import yaml

import synthetic
from config import Cfg
from fields import Static, cache_path, load_days, load_stats
from glorys_io import GlorysMonth


def test_index_splits_and_forcing(world):
    cfg = Cfg.load(world["cfg"])
    days = load_days(cfg.paths["days_csv"])
    by = {}
    for r in days:
        by.setdefault(r["split"], []).append(r)
    assert {k: len(v) for k, v in by.items()} == {"train": 14, "val": 9, "test": 9}
    # The 2001 ERA5 month was a zip of one netCDF per variable.
    f = np.load(by["val"][0]["forcing_path"])
    assert f.shape == (9, 2, cfg.h, cfg.w) and np.isfinite(f).all()


def test_forcing_regridded_correctly(world):
    """ERA5 taux = 0.1 sin(20 lon_rad + 0.3 k) on its 0.25 deg grid; the
    bilinear value at a model cell is close to the analytic one."""
    cfg = Cfg.load(world["cfg"])
    r = load_days(cfg.paths["days_csv"], "train")[3]
    f = np.load(r["forcing_path"])[r["forcing_index"]]
    _lat, lon = cfg.axes()
    exact = 0.1 * np.sin(np.deg2rad(lon) * 20 + 0.3 * 3)
    assert np.abs(f[0][10] - exact).max() < 2e-3


def test_glorys_reading_selects_levels_and_rows(world):
    cfg = Cfg.load(world["cfg"])
    p = os.path.join(world["root"], "GLORYS12", "2000", "glorys12_2000_01.nc")
    with GlorysMonth(p, cfg) as gm:
        f = gm.read_day(date(2000, 1, 5))
    assert f["temp"].shape == (24, cfg.h, cfg.w) and f["ssh"].shape == (cfg.h, cfg.w)
    lat, lon = cfg.axes()
    _u, _v, t, _s, _z = synthetic.fields(date(2000, 1, 5), lat, lon, cfg.depth_m)
    ok = np.isfinite(f["temp"])
    assert ok.mean() > 0.5
    np.testing.assert_allclose(f["temp"][ok], t[ok], atol=1.5e-3)    # int16 packing: 0.001


def test_static_mask_intersects_and_shelf(world):
    cfg = Cfg.load(world["cfg"])
    st = Static(cfg.paths["static_npz"], cfg)
    nv = st.n_valid
    assert nv[24, 20] == 0                      # island
    assert nv[24, 25] == 0                      # coastal ring: temp valid, u/v missing
    assert nv[5, 5] == cfg.n_levels             # deep ocean, all 24 levels
    assert 0 < nv[5, 60] < cfg.n_levels         # shelf in the east
    assert st.edge[0].sum() == 0 and st.edge[10, 10] == 1


def test_stats_and_cache_normalized(world):
    cfg = Cfg.load(world["cfg"])
    stats = load_stats(cfg.paths["stats_json"])
    assert stats["_meta"]["split"] == "train" and len(stats["vars3d"]["u"]["std"]) == 24
    st = Static(cfg.paths["static_npz"], cfg)
    a = np.load(cache_path(cfg.paths["cache_dir"], date(2000, 1, 2)))
    assert a.shape == (99, cfg.h, cfg.w) and a.dtype == np.float16
    ocean = st.ocean
    assert (a[:, ~ocean] == 0).all()                         # land is 0 in every channel
    assert abs(float(a[0][ocean].mean())) < 1.5              # z-scored
    k = cfg.n_levels - 1                                     # below the shelf floor -> 0
    assert (a[k][st.n_valid <= k] == 0).all()


def test_build_cache_is_idempotent_and_verifies(world):
    import build_cache
    assert build_cache.main(["--config", world["cfg"], "--year", "2000"]) == 0
    assert build_cache.main(["--config", world["cfg"], "--verify"]) == 0


def test_loader_windows(world):
    from dataset import WindowStream
    ds = WindowStream(world["cfg"], "train", horizon=3)
    cfg = ds.cfg
    x0, frc, fns, y, st, doy, m, e = next(iter(ds))
    p = cfg.patch
    assert x0.shape == (99, p, p) and frc.shape == (4, 2, p, p) and fns.shape == (4, 2, p, p)
    assert y.shape == (3, 97, p, p) and st.shape == (6, p, p) and doy.shape == (3,)
    assert m.shape == (24, p, p) and e.shape == (1, p, p)
    # Targets are the next days' stacks: y[0] of one window == x0 of the window
    # one day later at the same origin.
    assert torch.equal(frc[0], x0[97:99])
    n = sum(1 for _ in WindowStream(world["cfg"], "train", horizon=3))
    assert n == len(ds.starts) * len(ds.origins)


def _train(cfg_path, extra=()):
    import train
    return train.main(["--config", cfg_path, *extra])


def test_train_one_day_then_rollout_and_evaluate(world):
    import evaluate
    cfg = Cfg.load(world["cfg"])
    assert _train(world["cfg"], ["--smoke", "1"]) == 0
    assert _train(world["cfg"]) == 0
    out = cfg.paths["out_dir"]
    rows = list(csv.DictReader(open(os.path.join(out, "metrics.csv"))))
    assert len(rows) == 2 and np.isfinite(float(rows[-1]["val_skill_l1_u"]))
    assert _train(world["cfg"]) == 0                         # resume: nothing left to do
    assert len(list(csv.DictReader(open(os.path.join(out, "metrics.csv"))))) == 2

    # Rollout fine-tuning from the one-day best.pt, horizon 3.
    raw = yaml.safe_load(open(world["cfg"]))
    raw["base"] = world["cfg"]
    for k in ("paths", "grid", "data", "evaluate", "model"):
        raw.pop(k, None)
    raw["paths"] = {"out_dir": os.path.join(world["root"], "runs", "roll3")}
    raw["dataset"] = {"horizon": 3}
    raw["train"] = {"init_from": os.path.join(out, "best.pt"), "epochs": 1, "lr": 1e-4}
    p = os.path.join(world["root"], "roll3.yaml")
    yaml.safe_dump(raw, open(p, "w"))
    assert _train(p) == 0
    rows = list(csv.DictReader(open(os.path.join(world["root"], "runs", "roll3", "metrics.csv"))))
    assert "val_skill_l3_salt" in rows[0]

    ck = os.path.join(world["root"], "runs", "roll3", "best.pt")
    for spec in ("true", "persist"):
        assert evaluate.main(["--checkpoint", ck, "--lead", "3", "--next-forcing", spec]) == 0
    res = json.load(open(ck.replace("best.pt", "eval_test_best.json")))
    assert res["n_start_days"] == 6 and res["lead_days"] == 3
    s = res["scores"]["lead2/domain/u"]
    assert s["skill"] is not None and np.isfinite(s["rmse_surface"])
    assert os.path.exists(ck.replace("best.pt", "eval_test_best_persist.json"))


class ZeroDelta(torch.nn.Module):
    """A residual model whose change is zero: it IS persistence."""
    def __init__(self, out_ch):
        super().__init__()
        self.out_ch = out_ch
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x[:, :self.out_ch] + 0 * self.w


def test_zero_model_scores_as_persistence(world, monkeypatch):
    import evaluate
    import model as model_mod
    cfg = Cfg.load(world["cfg"])
    monkeypatch.setattr(evaluate, "build_model_from_cfg", lambda c: ZeroDelta(c.out_ch))
    ck = os.path.join(world["root"], "zero.pt")
    torch.save({"model": ZeroDelta(97).state_dict(), "cfg": cfg.raw, "epoch": 0}, ck)
    assert evaluate.main(["--checkpoint", ck, "--lead", "2"]) == 0
    res = json.load(open(os.path.join(world["root"], "eval_test_zero.json")))
    for key, s in res["scores"].items():
        assert s["skill"] == pytest.approx(1.0, rel=1e-5), key
        assert s["rmse_surface"] == pytest.approx(s["rmse_surface_persistence"], rel=1e-5)


def test_tiled_forward_equals_direct_for_pointwise_model():
    from rollout import tiled_forward
    torch.manual_seed(0)
    m = torch.nn.Conv2d(5, 3, 1)
    x = torch.randn(1, 5, 70, 90)
    with torch.no_grad():
        direct = m(x)
    tiled = tiled_forward(m, x, 32, 24, batch=4, amp=False)
    torch.testing.assert_close(tiled, direct, atol=1e-5, rtol=1e-5)


def test_blend_future():
    from rollout import blend_future, parse_forcing_spec
    f = torch.arange(4.0).view(1, 4, 1, 1, 1)
    assert torch.equal(blend_future(f, 1.0), f)
    assert blend_future(f, 0.0).flatten().tolist() == [0, 0, 0, 0]
    assert blend_future(f, 0.5).flatten().tolist() == [0, 0.5, 1.0, 1.5]
    with pytest.raises(ValueError):
        parse_forcing_spec("gfs")


def test_download_glorys_with_fake_service(tmp_path):
    """fetch_month on a fake lazy dataset: right levels/rows, int16 packed, and
    a short month raises instead of leaving a gap."""
    import xarray as xr
    import download_glorys as dg
    root = str(tmp_path)
    src = synthetic.write_glorys_month(root + "/src", 2000, 2, 29)
    cfg_path = synthetic.write_config(root)
    cfg = Cfg.load(cfg_path)
    full = xr.open_dataset(src)

    def opener(did, start, end, c):
        return full.sel(time=slice(str(start), str(end)))

    out = str(tmp_path / "out.nc")
    dg.fetch_month(cfg, 2000, 2, out, opener)
    with GlorysMonth(out, cfg) as gm:
        assert len(gm.dates) == 29 and len(gm.iz) == 24
        assert gm.ds.variables["so"].dtype == np.int16
    with pytest.raises(ValueError, match="no GLORYS data"):
        dg.fetch_month(cfg, 2000, 3, str(tmp_path / "x.nc"),
                       lambda *a: full.sel(time=slice("2000-02-01", "2000-02-10")))
    assert dg.dataset_for(Cfg.load(os.path.join(synthetic.REPO, "config", "default.yaml")),
                          date(2021, 7, 1)).startswith("cmems_mod_glo_phy_myint")


def test_download_era5_with_fake_cds(tmp_path):
    import download_era5 as de
    cfg_path = synthetic.write_config(str(tmp_path))
    calls = []

    def fake(cfg, y, m, target):
        calls.append((y, m, de.area(cfg)))
        synthetic.write_era5_month(str(tmp_path / "tmp"), y, m, 3)
        os.replace(str(tmp_path / "tmp" / f"{y:04d}" / f"era5_stress_{y:04d}_{m:02d}.nc"), target)

    assert de.main(["--config", cfg_path, "--start", "2000-01", "--end", "2000-02", "--workers", "1"], fake) == 0
    assert len(calls) == 2
    n, w, s, e = calls[0][2]
    assert n > -30 + 47 / 12 and s < -30 and w < 30 and e > 30 + 63 / 12
    assert de.main(["--config", cfg_path, "--start", "2000-01", "--end", "2000-02", "--workers", "1"], fake) == 0
    assert len(calls) == 2                                   # resumable: nothing refetched


def test_coarsened_prototype_with_smoothed_forcing(world, tmp_path):
    """The 1/4 deg path (coarsen 3) on the same raw files, with the t+1
    forcing box-smoothed: static, stats, cache, loader and a training smoke."""
    import build_cache
    import build_forcing
    import build_index
    import build_static
    import compute_stats
    import train
    from dataset import WindowStream
    r = str(tmp_path)
    p = synthetic.write_config(
        r, grid={"width": 63, "height": 48, "coarsen": 3},
        dataset={"patch": 16, "stride": 8, "edge_margin_px": 1, "cache": {"shuffle_windows": 2}},
        variables={"forcing_next_smooth_deg": 0.5},
        evaluate={"tile": 16, "tile_stride": 8,
                  "regions": {"domain": [30.0, 36.0, -30.0, -26.0]}})
    raw = yaml.safe_load(open(p))
    raw["paths"]["glorys_dir"] = os.path.join(world["root"], "GLORYS12")
    raw["paths"]["era5_dir"] = os.path.join(world["root"], "ERA5")
    yaml.safe_dump(raw, open(p, "w"))
    cfg = Cfg.load(p)
    assert (cfg.h, cfg.w) == (16, 21)
    for mod, extra in ((build_forcing, []), (build_index, []), (build_static, []),
                       (compute_stats, ["--stride", "1"]), (build_cache, [])):
        assert mod.main(["--config", p, *extra]) == 0
    st = Static(cfg.paths["static_npz"], cfg)
    assert st.n_valid.shape == (16, 21) and (st.n_valid == 0).any() and (st.n_valid == 24).any()
    ds = WindowStream(p, "train", horizon=2)
    x0, frc, fns, *_ = next(iter(ds))
    # Smoothed t+1 forcing differs from the raw forcing but keeps land at 0.
    assert not torch.allclose(fns[1], frc[1]) and torch.equal(fns[1] == 0, frc[1] == 0)
    assert train.main(["--config", p, "--smoke", "1"]) == 0


def test_self_checks_run():
    import dataset
    import model
    assert dataset.main(["--config", os.path.join(synthetic.REPO, "config", "default.yaml")]) == 0
    m = model.OceanCurrentUNet(in_ch=133, out_ch=97, base=64)
    assert 13.0 < sum(p.numel() for p in m.parameters()) / 1e6 < 14.0
