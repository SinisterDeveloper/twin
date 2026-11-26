"""Family D: Strictly Local Explicit Physics-Operator Residual Net (LocalPDENet v3)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..contract import NC, NLAT, NLON, T_IN
from .convnext_unet import ConvNeXtBlock, FiLM
from .physics import ChannelLayerNorm2d, PhysicsFeatureBlock, eval_harmonic_4, spatial_grad_central, spatial_laplacian


class LocalPDENet(nn.Module):
    def __init__(
        self,
        static_grid: np.ndarray,
        clim_uv_coef: np.ndarray | None = None,
        clim_phys_coef: np.ndarray | None = None,
        anom_std: np.ndarray | None = None,
        dim: int = 80,
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

        self.phys = PhysicsFeatureBlock(clim_phys_coef=clim_phys_coef, anom_std=anom_std)
        in_ch = T_IN * NC + 4 * NC + PhysicsFeatureBlock.OUT_CHANNELS + static_grid.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1, padding_mode="replicate"),
            ChannelLayerNorm2d(dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, kernel_size=3, mlp_ratio=2.0) for _ in range(depth)])
        self.films = nn.ModuleList([FiLM(2, dim) for _ in range(depth)])
        self.head = nn.Conv2d(dim, NC, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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

        clim_uv = eval_harmonic_4(doy_sc, self.clim_uv_coef, shift_minus_one=True)
        uv_phys = z_last[:, 2:4] * self.anom_std_uv + clim_uv
        dz_dx, dz_dy = spatial_grad_central(z_last)
        lap = spatial_laplacian(z_last)
        u_eff = torch.clamp(uv_phys[:, 0:1], -20.0, 20.0) * 0.05
        v_eff = torch.clamp(uv_phys[:, 1:2], -20.0, 20.0) * 0.05
        adv_term = -(u_eff * dz_dx + v_eff * dz_dy)

        pf = self.phys(z_seq, doy_sc)
        s = self.static_grid.expand(b, -1, -1, -1)
        feat = self.stem(torch.cat([z_flat, dz_dx, dz_dy, lap, adv_term, pf, s], dim=1))
        for blk, fm in zip(self.blocks, self.films):
            feat = fm(blk(feat), doy_sc)
        delta = self.head(feat)
        return z_last + delta
