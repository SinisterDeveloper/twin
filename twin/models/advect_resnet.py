"""Family B: Strictly Local Physics-Informed Advection-Diffusion ResNet v3."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import NC, NLAT, NLON, T_IN
from .physics import ChannelLayerNorm2d, PhysicsFeatureBlock, eval_harmonic_4


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="replicate")
        self.norm1 = ChannelLayerNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="replicate")
        self.norm2 = ChannelLayerNorm2d(dim)
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
        clim_uv_coef: np.ndarray | None = None,
        clim_phys_coef: np.ndarray | None = None,
        anom_std: np.ndarray | None = None,
        dim: int = 96,
        depth: int = 3,
    ):
        super().__init__()
        self.register_buffer("static_grid", torch.from_numpy(static_grid).unsqueeze(0))

        if clim_uv_coef is None and clim_phys_coef is not None:
            clim_uv_coef = clim_phys_coef[:, 2:4]
        if clim_uv_coef is None:
            clim_uv_coef = np.zeros((9, 2, NLAT, NLON), dtype=np.float32)
        self.register_buffer("clim_uv_coef", torch.from_numpy(clim_uv_coef.astype(np.float32)))

        if anom_std is None:
            uv_std = np.array([3.5, 2.5], dtype=np.float32)
        elif len(anom_std) >= 4:
            uv_std = np.asarray(anom_std[2:4], dtype=np.float32)
        else:
            uv_std = np.asarray(anom_std[:2], dtype=np.float32)
        self.register_buffer("anom_std_uv", torch.from_numpy(uv_std).view(1, 2, 1, 1))

        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, NLAT),
            torch.linspace(-1.0, 1.0, NLON),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        self.register_buffer("base_grid", base_grid)

        self.advect_scale = nn.Parameter(torch.tensor([0.025, 0.025], dtype=torch.float32))
        self.phys = PhysicsFeatureBlock(clim_phys_coef=clim_phys_coef, anom_std=anom_std)

        in_ch = T_IN * NC + NC + PhysicsFeatureBlock.OUT_CHANNELS + static_grid.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1, padding_mode="replicate"),
            ChannelLayerNorm2d(dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([FiLMResBlock(dim) for _ in range(depth)])
        self.head = nn.Conv2d(dim, NC, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _advect(self, z_last: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        clim_uv = eval_harmonic_4(doy_sc, self.clim_uv_coef, shift_minus_one=True)
        uv_phys = z_last[:, 2:4] * self.anom_std_uv + clim_uv

        grid = self.base_grid.expand(z_last.shape[0], -1, -1, -1)
        dx1 = torch.clamp(uv_phys[:, 0] * self.advect_scale[0], -0.16, 0.16)
        dy1 = torch.clamp(uv_phys[:, 1] * self.advect_scale[1], -0.16, 0.16)
        mid_grid = torch.stack([grid[..., 0] - 0.5 * dx1, grid[..., 1] - 0.5 * dy1], dim=-1)

        uv_mid = F.grid_sample(uv_phys, mid_grid, mode="bilinear", padding_mode="border", align_corners=True)
        dx = torch.clamp(uv_mid[:, 0] * self.advect_scale[0], -0.16, 0.16)
        dy = torch.clamp(uv_mid[:, 1] * self.advect_scale[1], -0.16, 0.16)
        sample_grid = torch.stack([grid[..., 0] - dx, grid[..., 1] - dy], dim=-1)
        return F.grid_sample(z_last, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, z_hist: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        if z_hist.ndim == 5:
            b, t, c, h, w = z_hist.shape
            z_seq = z_hist
            z_flat = z_hist.reshape(b, t * c, h, w)
            z_last = z_hist[:, -1]
        else:
            b, _, h, w = z_hist.shape
            z_seq = z_hist.reshape(b, T_IN, NC, h, w)
            z_flat = z_hist
            z_last = z_seq[:, -1]

        z_adv = self._advect(z_last, doy_sc)
        pf = self.phys(z_seq, doy_sc)
        s = self.static_grid.expand(b, -1, -1, -1)
        h = self.stem(torch.cat([z_flat, z_adv, pf, s], dim=1))
        for blk in self.blocks:
            h = blk(h, doy_sc)
        delta = self.head(h)
        return z_last + delta
