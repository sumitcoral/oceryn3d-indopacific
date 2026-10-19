"""
Model input assembly and autoregressive rollout, on the GPU.

Input of one daily step (contract C2, the same order as Oceryn3D Global):
    state(out_ch=97)                      u v temp salt (x24), ssh   at day t+k
    forcing(F=2)                          taux tauy                  at day t+k
    depth encoding(L=24)
    static(8)                             land bathy sin/cos lat sin/cos lon sin/cos doy(t+k)
    forcing_next(Fn=2)                    taux tauy                  at day t+k+1
-> 133 channels. The first 97 are the state, so the residual model predicts
the change over one day.

Rollout: the step-k forecast, re-masked (land and below-sea-floor cells set
to 0, exactly as the cache stores them), is the step-(k+1) state. The
forcing of days after the issue day t is what a forecast would supply; the
`spec` argument chooses it:
    true      the ERA5 values (perfect forecast; training and the upper bound)
    persist   day t's forcing for every later day (no forecast: lower bound)
    blend:A   t + A * (true - t), a forecast with (1 - A) x persistence's error
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def parse_forcing_spec(spec: str) -> float:
    """'true' -> 1, 'persist' -> 0, 'blend:A' -> A in [0, 1]."""
    if spec == "true":
        return 1.0
    if spec == "persist":
        return 0.0
    if spec.startswith("blend:"):
        try:
            a = float(spec[6:])
        except ValueError:
            a = -1.0
        if 0.0 <= a <= 1.0:
            return a
    raise ValueError(f"forcing spec must be true, persist or blend:A (0<=A<=1), got {spec!r}")


def blend_future(f: torch.Tensor, a: float) -> torch.Tensor:
    """(B, K+1, C, H, W) forcing of days t..t+K -> days after t replaced by
    t + a * (f - t). a = 1 returns f unchanged."""
    if a == 1.0:
        return f
    f0 = f[:, :1]
    return torch.cat([f0, f0 + a * (f[:, 1:] - f0)], dim=1)


class StepAssembler:
    """Holds the constant channels for one batch of patches (or the domain)."""

    def __init__(self, cfg, depth_enc: torch.Tensor, static_sp: torch.Tensor,
                 level_mask: torch.Tensor):
        """depth_enc (L,H,W) or (B,L,H,W); static_sp (B,6,H,W); level_mask (B,L,H,W)."""
        self.cfg = cfg
        B, _, H, W = static_sp.shape
        if depth_enc.dim() == 3:
            depth_enc = depth_enc.unsqueeze(0).expand(B, -1, -1, -1)
        self.const = torch.cat([depth_enc, static_sp], dim=1)        # (B, L+6, H, W)
        self.mask = level_mask
        n3 = len(cfg.targets_3d) * cfg.n_levels
        # (B, out_ch, H, W): per-level mask for 3D targets, surface for 2D.
        self.state_mask = torch.cat(
            [level_mask] * len(cfg.targets_3d) + [level_mask[:, :1]] * len(cfg.targets_2d), dim=1)
        assert self.state_mask.shape[1] == cfg.out_ch and n3 <= cfg.out_ch

    def __call__(self, state, f_cur, f_next, doy):
        """state (B,out,H,W); f_cur (B,F,H,W); f_next (B,Fn,H,W) or None;
        doy (B,) angle of the step's input day -> (B, in_ch, H, W)."""
        B, _, H, W = state.shape
        s = torch.sin(doy).view(B, 1, 1, 1).expand(B, 1, H, W).to(state.dtype)
        c = torch.cos(doy).view(B, 1, 1, 1).expand(B, 1, H, W).to(state.dtype)
        parts = [state, f_cur.to(state.dtype), self.const.to(state.dtype), s, c]
        if self.cfg.forcing_next:
            parts.append(f_next.to(state.dtype))
        return torch.cat(parts, dim=1)

    def remask(self, pred):
        return pred * self.state_mask.to(pred.dtype)


def rollout(model, asm: StepAssembler, state0, frc, fns, doy, steps: int):
    """Autoregressive forecast. state0 (B,out,H,W); frc (B,K+1,F,H,W) and
    fns (B,K+1,Fn,H,W) forcing of days t..t+K (already blended per spec);
    doy (B,K) input-day angles. Returns [pred_1..pred_steps], each (B,out,H,W),
    re-masked."""
    preds, s = [], state0
    for k in range(steps):
        x = asm(s, frc[:, k], fns[:, k + 1] if fns is not None else None, doy[:, k])
        s = asm.remask(model(x))
        preds.append(s)
    return preds


# ---------------------------------------------------------------------------
# Full-domain inference by tiles (evaluation and operation)
# ---------------------------------------------------------------------------
def tile_origins(n: int, tile: int, stride: int):
    s = list(range(0, max(1, n - tile + 1), stride))
    if n > tile and s[-1] != n - tile:
        s.append(n - tile)
    return s or [0]


def blend_window(tile: int, device) -> torch.Tensor:
    """(tile,tile) weights that fall to near 0 at the tile edges, where a
    patch lacks context: distance to the nearest edge + 1, normalized."""
    i = torch.arange(tile, device=device, dtype=torch.float32)
    d = torch.minimum(i, tile - 1 - i) + 1.0
    w = torch.outer(d, d)
    return w / w.max()


@torch.no_grad()
def tiled_forward(model, x: torch.Tensor, tile: int, stride: int, batch: int = 8,
                  amp: bool = True) -> torch.Tensor:
    """(1, in_ch, H, W) -> (1, out_ch, H, W) by overlapping tiles, each the
    size of a training patch (GroupNorm sees what it saw in training),
    blended with blend_window."""
    _, _, H, W = x.shape
    if H < tile or W < tile:
        raise ValueError(f"domain {H}x{W} smaller than the tile {tile}")
    out = None
    wsum = torch.zeros(1, 1, H, W, device=x.device)
    win = blend_window(tile, x.device)
    org = [(r, c) for r in tile_origins(H, tile, stride) for c in tile_origins(W, tile, stride)]
    for i in range(0, len(org), batch):
        chunk = org[i:i + batch]
        xb = torch.cat([x[:, :, r:r + tile, c:c + tile] for r, c in chunk], dim=0)
        with torch.autocast(device_type=x.device.type, enabled=amp and x.device.type == "cuda"):
            yb = model(xb).float()
        if out is None:
            out = torch.zeros(1, yb.shape[1], H, W, device=x.device)
        for j, (r, c) in enumerate(chunk):
            out[:, :, r:r + tile, c:c + tile] += yb[j] * win
            wsum[:, :, r:r + tile, c:c + tile] += win
    return out / wsum.clamp_min(1e-6)


def smooth_ocean_torch(f: torch.Tensor, ocean: torch.Tensor, k: int) -> torch.Tensor:
    """(B,C,H,W) box mean over +-k cells on ocean only (normalized
    convolution), land 0. Matches grid.smooth_ocean away from patch edges."""
    if k <= 0:
        return f
    o = ocean.to(f.dtype)
    num = F.avg_pool2d(f * o, 2 * k + 1, stride=1, padding=k, count_include_pad=True)
    den = F.avg_pool2d(o, 2 * k + 1, stride=1, padding=k, count_include_pad=True)
    return torch.where(o > 0, num / den.clamp_min(1e-6), torch.zeros_like(f))
