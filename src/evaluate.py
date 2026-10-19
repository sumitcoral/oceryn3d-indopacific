#!/usr/bin/env python
"""
Stage 7: evaluation by full-domain rollouts.

    python src/evaluate.py --checkpoint <run>/best.pt [--split test] [--lead 5]
           [--days-frac 0.25] [--next-forcing true|persist|blend:A] [--config <yaml>]

For each sampled start day t of the split (with t..t+lead all present), the
model forecasts days t+1..t+lead autoregressively over the WHOLE domain:
each step is run tile by tile (training-sized tiles, overlapping, blended),
so information crosses tile borders between steps, unlike the patch-based
validation in train.py. Persistence (day t's state held) is scored on the
same cells.

Scores, pooled over all sampled days and divided once, per lead day, region
(evaluate.regions, each minus the edge margin) and variable:
  mse_norm / skill   normalized MSE pooled over levels; skill = model/persistence
                     MSE (< 1 beats persistence)
  rmse_surface       physical RMSE at the top level (m/s, degC, psu, m)
and per level (physical RMSE) in the _levels.csv. Output next to the
checkpoint: eval_<split>_<ckpt><tag>.json and _levels.csv, where <tag>
names a non-true forcing spec.

--next-forcing: the forcing of days after t (both the day's own and the
next day's), as in rollout.py: true (ERA5, a perfect forecast), persist
(day t's wind kept: no forecast) or blend:A. A model is only as good in
operation as its score with a REAL forecast, which sits between the two.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Cfg, load_config                              # noqa: E402
from dataset import window_starts                                # noqa: E402
from fields import Static, cache_path, load_days, load_stats     # noqa: E402
from grid import depth_encoding, doy_angle, level_mask, region_mask  # noqa: E402
from model import build_model_from_cfg                           # noqa: E402
from rollout import (StepAssembler, blend_future, parse_forcing_spec,  # noqa: E402
                     smooth_ocean_torch, tiled_forward)


def spec_tag(spec: str) -> str:
    return "" if spec == "true" else "_" + spec.replace(":", "")


class Sums:
    """Pooled per (lead, region, var, level) sums for model and persistence."""

    def __init__(self, leads, regions, slices):
        self.leads, self.regions, self.slices = leads, regions, slices
        self.d = {}

    def add(self, lead, region, who, name, sse_lev, w_lev):
        k = (lead, region, who, name)
        if k not in self.d:
            self.d[k] = [np.zeros_like(sse_lev), np.zeros_like(w_lev)]
        self.d[k][0] += sse_lev
        self.d[k][1] += w_lev


def per_level(se: torch.Tensor, w: torch.Tensor, slices):
    """se, w (out,H,W) -> {name: (sse per level, wsum per level)} as float64."""
    out = {}
    for name, s, wd in slices:
        a = (se[s:s + wd] * w[s:s + wd]).sum(dim=(1, 2)).double().cpu().numpy()
        b = w[s:s + wd].sum(dim=(1, 2)).double().cpu().numpy()
        out[name] = (a, b)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", help="override paths (default: the checkpoint's own config)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--lead", type=int, default=5)
    ap.add_argument("--days-frac", type=float, default=1.0)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--next-forcing", default="true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8, help="tiles per forward pass")
    args = ap.parse_args(argv)
    a_blend = parse_forcing_spec(args.next_forcing)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    raw = ck["cfg"]
    if args.config:                  # paths from a config file, model from the checkpoint
        raw = dict(raw)
        raw["paths"] = load_config(args.config)["paths"]
    cfg = Cfg(raw, args.checkpoint)
    model = build_model_from_cfg(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    stats = load_stats(cfg.paths["stats_json"])
    static = Static(cfg.paths["static_npz"], cfg)
    ev = raw.get("evaluate", {}) or {}
    tile, stride = int(ev.get("tile", cfg.patch)), int(ev.get("tile_stride", cfg.patch * 3 // 4))
    if tile != cfg.patch:
        print(f"[warn] evaluate.tile {tile} != training patch {cfg.patch}: GroupNorm "
              "statistics will differ from training", file=sys.stderr)

    days = load_days(cfg.paths["days_csv"], args.split)
    starts = window_starts(days, args.lead)
    if args.days_frac < 1.0:
        k = max(1, round(len(starts) * args.days_frac))
        starts = sorted(np.random.default_rng(args.seed).choice(starts, k, replace=False).tolist())
    if args.max_days:
        starts = starts[:args.max_days]
    if not starts:
        sys.exit(f"no {args.lead + 1}-day windows in split {args.split}")

    # Constant fields on the device.
    lm_np = level_mask(static.n_valid, cfg.n_levels)
    lm = torch.from_numpy(lm_np).to(device)[None]                       # (1,L,H,W)
    st = torch.from_numpy(static.spatial).to(device)[None]
    de = torch.from_numpy(depth_encoding(cfg.depth_m, cfg.h, cfg.w)).to(device)
    asm = StepAssembler(cfg, de, st, lm)
    lat, lon = static.lat, static.lon
    edge = torch.from_numpy(static.edge).to(device)
    cos = torch.from_numpy(static.cos_lat).to(device)
    regions = {}
    for name, box in (ev.get("regions") or {"domain": [-1e9, 1e9, -90, 90]}).items():
        rm = torch.from_numpy(region_mask(lat, lon, box).astype(np.float32)).to(device)
        if float((rm * edge * torch.from_numpy(static.ocean.astype(np.float32)).to(device)).sum()) == 0:
            print(f"[warn] region {name} has no scored ocean cells", file=sys.stderr)
        regions[name] = rm * edge * cos
    state_mask = asm.state_mask[0]                                     # (out,H,W)
    slices = cfg.target_slices()
    f_idx = cfg.forcing_index(cfg.forcing_vars)
    fn_idx = cfg.forcing_index(cfg.forcing_next)
    ksm = int(round(cfg.forcing_next_smooth_deg / cfg.dlat))
    ocean_t = torch.from_numpy(static.ocean.astype(np.float32)).to(device)[None, None]
    sums = Sums(range(1, args.lead + 1), list(regions), slices)

    print(f"[eval] {args.checkpoint} epoch {ck.get('epoch')}: {len(starts)} start days "
          f"({args.split}), lead {args.lead}, forcing {args.next_forcing}, tiles "
          f"{tile}/{stride}, {device}", flush=True)
    t0 = time.time()
    for n, i in enumerate(starts, 1):
        ds = [days[i + j]["date"] for j in range(args.lead + 1)]
        stacks = [torch.from_numpy(np.asarray(np.load(cache_path(cfg.paths["cache_dir"], d),
                                                      mmap_mode="r"), np.float32)).to(device)
                  for d in ds]
        truth = [s[:cfg.out_ch] for s in stacks]
        frc = blend_future(torch.stack([s[f_idx] for s in stacks])[None], a_blend)[0]
        fns = None
        if cfg.forcing_next:
            fns = torch.stack([s[fn_idx] for s in stacks])
            fns = smooth_ocean_torch(fns, ocean_t[0], ksm) if ksm > 0 else fns
            fns = blend_future(fns[None], a_blend)[0]
        state = truth[0][None] * state_mask
        for k in range(args.lead):
            doy = torch.tensor([doy_angle(ds[k])], device=device)
            x = asm(state, frc[k][None], fns[k + 1][None] if fns is not None else None, doy)
            state = tiled_forward(model, x, tile, stride, args.batch) * state_mask
            y = truth[k + 1]
            se_m = (state[0] - y) ** 2
            se_p = (truth[0] - y) ** 2
            for rname, rw in regions.items():
                w = state_mask * rw
                for who, se in (("model", se_m), ("pers", se_p)):
                    for name, (a, b) in per_level(se, w, slices).items():
                        sums.add(k + 1, rname, who, name, a, b)
        if n % 10 == 0 or n == len(starts):
            print(f"  {n}/{len(starts)} days, {(time.time() - t0) / n:.1f} s/day", flush=True)

    # ---- report ----------------------------------------------------------
    res = {"checkpoint": os.path.abspath(args.checkpoint), "epoch": ck.get("epoch"),
           "split": args.split, "n_start_days": len(starts), "lead_days": args.lead,
           "next_forcing": args.next_forcing, "tile": tile, "tile_stride": stride,
           "edge_margin_px": cfg.edge_margin, "scores": {}}
    rows = []
    for k in range(1, args.lead + 1):
        for rname in regions:
            for name, s, wd in slices:
                sm, wm = sums.d[(k, rname, "model", name)]
                sp, _ = sums.d[(k, rname, "pers", name)]
                if wm.sum() == 0:
                    continue
                std = (np.asarray(stats["vars3d"][name]["std"], np.float64) if wd > 1
                       else np.array([stats["vars2d"][name]["std"]], np.float64))
                with np.errstate(invalid="ignore", divide="ignore"):
                    rm = np.sqrt(sm / wm) * std
                    rp = np.sqrt(sp / wm) * std
                res["scores"][f"lead{k}/{rname}/{name}"] = {
                    "mse_norm": float(sm.sum() / wm.sum()),
                    "skill": float(sm.sum() / sp.sum()) if sp.sum() > 0 else None,
                    "rmse_surface": float(rm[0]), "rmse_surface_persistence": float(rp[0])}
                for lev in range(wd):
                    if wm[lev] > 0:
                        rows.append([k, rname, name, lev,
                                     cfg.depth_m[lev] if wd > 1 else 0.0,
                                     rm[lev], rp[lev], sm[lev] / sp[lev] if sp[lev] > 0 else ""])
    base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    stem = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                        f"eval_{args.split}_{base}{spec_tag(args.next_forcing)}")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=1)
    with open(stem + "_levels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lead_day", "region", "var", "level", "depth_m", "rmse_model",
                    "rmse_persistence", "skill_mse"])
        w.writerows(rows)
    print("lead  region              " + "  ".join(f"{n:>6s}" for n, _s, _w in slices))
    for k in range(1, args.lead + 1):
        for rname in regions:
            sk = [res["scores"].get(f"lead{k}/{rname}/{n}", {}).get("skill") for n, _s, _w in slices]
            print(f"{k:4d}  {rname:18s}  " + "  ".join(
                f"{x:6.3f}" if x is not None else "   nan" for x in sk))
    print(f"[eval] skill = model MSE / persistence MSE (< 1 beats persistence) -> {stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
