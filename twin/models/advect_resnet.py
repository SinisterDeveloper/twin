"""Family B: Physics-Informed Advection-Diffusion ResNet.

Combines differentiable semi-Lagrangian wind advection (grid_sample driven by u10/v10)
with local residual reaction-diffusion blocks and FiLM doy conditioning.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import NC, NLAT, NLON, T_IN


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="replicate")
        self.norm1 = nn.GroupNorm(min(8, dim), dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(min(8, dim), dim)
        self.film = nn.Linear(2, dim * 2)

    def forward(self, x: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        gb = self.film(doy_sc).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        h = self.norm2(self.conv2(h)) * (1.0 + gamma) + beta
        return x + F.gelu(h)


class AdvectResNet(nn.Module):
    def __init__(
        self,
        static_grid: np.ndarray,
        dim: int = 80,
        depth: int = 5,
    ):
        super().__init__()
        self.register_buffer("static_grid", torch.from_numpy(static_grid).unsqueeze(0))

        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, NLAT),
            torch.linspace(-1.0, 1.0, NLON),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        self.register_buffer("base_grid", base_grid)

        # Learnable wind-to-grid-displacement scale (initialized conservatively)
        self.advect_scale = nn.Parameter(torch.tensor([0.04, 0.04], dtype=torch.float32))

        in_ch = T_IN * NC + NC + static_grid.shape[0]  # z_hist + advected_z_last + static
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([FiLMResBlock(dim) for _ in range(depth)])
        self.head = nn.Conv2d(dim, NC, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _advect(self, z_last: torch.Tensor) -> torch.Tensor:
        # u10 is channel 2, v10 is channel 3; backward semi-Lagrangian trace
        u = z_last[:, 2]
        v = z_last[:, 3]
        dx = torch.clamp(u * self.advect_scale[0], -0.25, 0.25)
        dy = torch.clamp(v * self.advect_scale[1], -0.25, 0.25)
        grid = self.base_grid.expand(z_last.shape[0], -1, -1, -1)
        sample_grid = torch.stack([grid[..., 0] - dx, grid[..., 1] - dy], dim=-1)
        return F.grid_sample(z_last, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, z_hist: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        if z_hist.ndim == 5:
            b, t, c, h, w = z_hist.shape
            z_flat = z_hist.reshape(b, t * c, h, w)
            z_last = z_hist[:, -1]
        else:
            z_flat = z_hist
            z_last = z_hist[:, -NC:]

        z_adv = self._advect(z_last)
        s = self.static_grid.expand(z_flat.shape[0], -1, -1, -1)
        h = self.stem(torch.cat([z_flat, z_adv, s], dim=1))
        for blk in self.blocks:
            h = blk(h, doy_sc)
        delta = self.head(h)
        return z_last + delta
