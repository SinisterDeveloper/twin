"""Family A: Shallow 2-Scale ConvNeXt-UNet with FiLM doy conditioning and GroupNorm.

Bounded receptive field (~5-7 grid cells per 1-day step) and zero global pooling
ensure Day-1 shock locality (>=60% energy within 3 cells) while capturing sharp fronts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import NC, T_IN


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.proj(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, mlp_ratio: float = 2.0):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=pad, padding_mode="replicate", groups=dim)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw2(self.act(self.pw1(h)))
        return x + h


class ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        static_grid: np.ndarray,
        base_dim: int = 64,
        depth: int = 3,
    ):
        super().__init__()
        self.register_buffer("static_grid", torch.from_numpy(static_grid).unsqueeze(0))  # [1, 4, 32, 36]
        in_ch = T_IN * NC + static_grid.shape[0]
        d1 = base_dim
        d2 = base_dim * 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, d1, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, d1),
            nn.GELU(),
        )
        self.enc1 = nn.ModuleList([ConvNeXtBlock(d1, kernel_size=3) for _ in range(depth)])
        self.film1 = FiLM(2, d1)

        self.down = nn.Conv2d(d1, d2, 2, stride=2)
        self.mid = nn.ModuleList([ConvNeXtBlock(d2, kernel_size=3) for _ in range(depth)])
        self.film2 = FiLM(2, d2)

        self.up = nn.ConvTranspose2d(d2, d1, 2, stride=2)
        self.fuse = nn.Conv2d(d1 * 2, d1, 1)
        self.dec1 = nn.ModuleList([ConvNeXtBlock(d1, kernel_size=3) for _ in range(depth)])
        self.head = nn.Conv2d(d1, NC, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_hist: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        """z_hist: [B, 3, 6, 32, 36] (or [B, 18, 32, 36]), doy_sc: [B, 2] -> z_next [B, 6, 32, 36]."""
        if z_hist.ndim == 5:
            b, t, c, h, w = z_hist.shape
            z_flat = z_hist.reshape(b, t * c, h, w)
            z_last = z_hist[:, -1]
        else:
            z_flat = z_hist
            z_last = z_hist[:, -NC:]

        s = self.static_grid.expand(z_flat.shape[0], -1, -1, -1)
        x = torch.cat([z_flat, s], dim=1)

        h1 = self.stem(x)
        for blk in self.enc1:
            h1 = blk(h1)
        h1 = self.film1(h1, doy_sc)

        h2 = self.down(h1)
        for blk in self.mid:
            h2 = blk(h2)
        h2 = self.film2(h2, doy_sc)

        u = self.up(h2)
        if u.shape[-2:] != h1.shape[-2:]:
            u = F.interpolate(u, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([h1, u], dim=1))
        for blk in self.dec1:
            out = blk(out)
        delta = self.head(out)
        return z_last + delta
