"""
Loss: per-variable-weighted, per-level-weighted, masked, area-weighted MSE on
normalized fields, the data term of Oceryn3D Global's OceanLoss. The
continuity and gradient terms are left out on purpose: without vertical
velocity, 2D continuity is physically false (see ocean-emulator losses.py).

Channel layout of pred/target: (B, out_ch, H, W), var-major in target_vars
order: each 3D target takes n_levels channels, each 2D target one.
Weights per cell: ocean mask (per level; 2D targets use the surface level)
x cos(lat) x edge weight (0 near the open boundary).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateLoss(nn.Module):
    def __init__(self, cfg, var_weights=None):
        super().__init__()
        L = cfg.n_levels
        self.L = L
        self.slices = cfg.target_slices()                # [(name, start, width)]
        self.target_vars = [s[0] for s in self.slices]
        self.out_ch = cfg.out_ch
        # Surface-heavy level weights, as in Oceryn3D Global: exp(-k/8),
        # mean 1. The product is near-surface currents.
        lw = torch.exp(-torch.arange(L, dtype=torch.float32) / 8.0)
        lw = lw / lw.mean()
        vw = dict(var_weights or {})
        blocks = [float(vw.get(n, 1.0)) * (lw if w == L else torch.ones(w))
                  for n, _s, w in self.slices]
        self.register_buffer("chan_w", torch.cat(blocks).view(1, -1, 1, 1))

    def weights(self, mask, cos_lat, edge):
        """(B,L,H,W) level mask, (B,1,H,W) cos_lat and edge -> (B,out,H,W)."""
        blocks = [mask if w == self.L else mask[:, :1] for _n, _s, w in self.slices]
        return torch.cat(blocks, dim=1) * (cos_lat.clamp_min(0.0) * edge)

    def forward(self, pred, target, w):
        """w from weights(). Returns (loss, parts); parts holds per-variable
        raw sums sse_<v>/wsum_<v> (floats) so callers pool and divide once."""
        se = (pred.float() - target.float()) ** 2
        loss = (se * self.chan_w * w).sum() / w.sum().clamp_min(1.0)
        parts = {}
        with torch.no_grad():
            for n, s, wd in self.slices:
                parts[f"sse_{n}"] = float((se[:, s:s + wd] * w[:, s:s + wd]).sum())
                parts[f"wsum_{n}"] = float(w[:, s:s + wd].sum())
        return loss, parts


def lead_weights(cfg, horizon: int) -> list:
    """Normalized weights of the rollout steps (loss.lead_weights or equal)."""
    lw = list((cfg.raw.get("loss", {}) or {}).get("lead_weights") or [])
    if len(lw) < horizon:
        lw = [1.0] * horizon
    lw = [float(x) for x in lw[:horizon]]
    s = sum(lw)
    return [x / s for x in lw]
