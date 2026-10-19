"""
U-Net for the 3D ocean state, one daily step. Copied from Oceryn3D Global
(ocean-emulator src/model.py, commit 84672b2) unchanged in architecture:
residual (out = delta + x[:, :out_ch]), replicate padding, GroupNorm, base 64
-> 13.4 M parameters. Replicate padding is the right choice on a regional
domain too: the domain edge is an open boundary, not a wall or a wrap.

Input : (B, 133, H, W)  state(97) + forcing at t(2) + depth(24) + static(8)
                        + forcing at t+1(2), assembled by rollout.step_input
Output: (B,  97, H, W)  u, v, temp, salt x 24 levels + ssh, the next day
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Replicate padding in both H and W.

    Replicate rather than zeros: a zero-padded border reads as "ocean at rest"
    to the network, which is a stronger and more misleading claim than "same as
    the edge". At the poles it also avoids fabricating values across the
    latitude singularity.
    """
    if p == 0:
        return x
    return F.pad(x, (p, p, p, p), mode="replicate")


class ConvBlock(nn.Module):
    """(Conv -> GroupNorm -> SiLU) x2 with geometry-aware padding.

    GroupNorm, not BatchNorm: batch is 8 and normalization statistics must not
    depend on which patches happen to share a batch. GroupNorm is per-sample, so
    train and inference behave identically -- which matters because inference
    tiles the globe patch by patch.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=0)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(_pad(x, 1))))
        x = self.act(self.norm2(self.conv2(_pad(x, 1))))
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return self.block(torch.cat([skip, x], dim=1))


class OceanCurrentUNet(nn.Module):
    """(B,in_ch,H,W) -> (B,out_ch,H,W).

    Receptive field at these settings is 140 px at the bottleneck and ~200 px at
    the output; effective RF is roughly a third of that, so ~700-1100 km at
    0.1 deg. A 2 m/s western boundary current advects ~860 km in five days, so
    one 5-day step fits with modest margin. For a 1-day step the margin is ~5x.
    """

    def __init__(self, in_ch: int = 131, out_ch: int = 97, base: int = 64,
                 depth_levels: int = 24, residual: bool = True):
        super().__init__()
        if residual and in_ch < out_ch:
            raise ValueError(
                f"residual mode needs in_ch >= out_ch, got {in_ch} < {out_ch}. "
                "The baseline is x[:, :out_ch], which must be the target "
                "variables at time t (contract C1).")
        self.in_ch, self.out_ch = in_ch, out_ch
        self.depth_levels, self.residual = depth_levels, residual

        self.inc = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 8)
        self.u1 = Up(base * 8, base * 8, base * 4)
        self.u2 = Up(base * 4, base * 4, base * 2)
        self.u3 = Up(base * 2, base * 2, base)
        self.u4 = Up(base, base, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        if x.shape[1] != self.in_ch:
            raise ValueError(
                f"model built for in_ch={self.in_ch} but got {x.shape[1]}. "
                "Config and dataset disagree -- check grid.st_ocean and "
                "dataset.static_channels (contract C2).")
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        y = self.u1(x4, x3)
        y = self.u2(y, x2)
        y = self.u3(y, x1)
        y = self.u4(y, x0)
        out = self.outc(y)
        if self.residual:
            # Channels 0..out_ch-1 are the target variables at time t, in target
            # order, under the same per-level normalization. The network learns
            # only the change from persistence over the 5-day step.
            out = out + x[:, :self.out_ch]
        return out


def build_model_from_cfg(cfg) -> "OceanCurrentUNet":
    """From a config.Cfg (contracts C1-C3 are already asserted by Cfg)."""
    m = cfg.model
    return OceanCurrentUNet(in_ch=int(m["in_ch"]), out_ch=int(m["out_ch"]),
                            base=int(m.get("base", 64)), depth_levels=cfg.n_levels,
                            residual=bool(m.get("residual", True)))


if __name__ == "__main__":
    m = OceanCurrentUNet(in_ch=133, out_ch=97, residual=True)
    print(f"Parameters: {sum(p.numel() for p in m.parameters())/1e6:.1f} M")
    xb = torch.randn(1, 133, 256, 256)
    with torch.no_grad():
        yb = m(xb)
    assert yb.shape == (1, 97, 256, 256), yb.shape
    print("shape self-check OK")
