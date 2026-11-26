"""Strictly pointwise ChannelLayerNorm2d and exact spherical physical feature extractors."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import LAT, NC, NLAT, NLON

ONE_DAY_RAD = 2.0 * math.pi / 365.0


class ChannelLayerNorm2d(nn.Module):
    """Strictly per-pixel LayerNorm along channel axis (dim=1), with zero spatial coupling."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean = xf.mean(dim=1, keepdim=True)
        var = (xf - mean).pow(2).mean(dim=1, keepdim=True)
        normed = ((xf - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)
        return normed * self.weight + self.bias


def eval_harmonic_4(doy_sc: torch.Tensor, coef: torch.Tensor, shift_minus_one: bool = True) -> torch.Tensor:
    """Evaluate 4-harmonic climatology from [sin theta, cos theta] and coef [9, C, H, W]."""
    s1 = doy_sc[:, 0]
    c1 = doy_sc[:, 1]
    if shift_minus_one:
        ca = math.cos(ONE_DAY_RAD)
        sa = math.sin(ONE_DAY_RAD)
        s1_prev = s1 * ca - c1 * sa
        c1_prev = c1 * ca + s1 * sa
        s1, c1 = s1_prev, c1_prev

    s2 = 2.0 * s1 * c1
    c2 = c1 * c1 - s1 * s1
    s3 = s2 * c1 + c2 * s1
    c3 = c2 * c1 - s2 * s1
    s4 = 2.0 * s2 * c2
    c4 = c2 * c2 - s2 * s2

    ones = torch.ones_like(s1)
    basis = torch.stack([ones, s1, c1, s2, c2, s3, c3, s4, c4], dim=1)  # [B, 9]
    b = basis.shape[0]
    c, h, w = coef.shape[1], coef.shape[2], coef.shape[3]
    with torch.amp.autocast(doy_sc.device.type, enabled=False):
        out = torch.matmul(basis.float(), coef.float().reshape(9, c * h * w)).reshape(b, c, h, w)
    return out.to(doy_sc.dtype)


def spatial_grad_central(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Strictly 3x3 replicate-padded central differences (dx, dy) along lon/lat."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2])
    dy = 0.5 * (xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1])
    return dx, dy


def spatial_laplacian(x: torch.Tensor) -> torch.Tensor:
    """Strictly 3x3 replicate-padded 5-point discrete Laplacian."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return (
        xp[:, :, 1:-1, 2:]
        + xp[:, :, 1:-1, :-2]
        + xp[:, :, 2:, 1:-1]
        + xp[:, :, :-2, 1:-1]
        - 4.0 * x
    )


class PhysicsFeatureBlock(nn.Module):
    """Computes 17 strictly local temporal tendency and spherical physical derivative channels."""

    OUT_CHANNELS = 17

    def __init__(
        self,
        clim_phys_coef: np.ndarray | None = None,
        anom_std: np.ndarray | None = None,
    ):
        super().__init__()
        if clim_phys_coef is None:
            clim_phys_coef = np.zeros((9, NC, NLAT, NLON), dtype=np.float32)
        else:
            clim_phys_coef = np.array(clim_phys_coef, dtype=np.float32, copy=True)
            # Subtract standard MSL (101325 Pa) from constant harmonic so values stay < 1500 (safe in fp16)
            clim_phys_coef[0, 4] -= 101325.0
        self.register_buffer("clim_phys_coef", torch.from_numpy(clim_phys_coef))

        if anom_std is None:
            std_full = np.array([1.7, 0.7, 1.5, 1.4, 280.0, 6.1], dtype=np.float32)
        else:
            std_full = np.asarray(anom_std, dtype=np.float32)
        self.register_buffer("anom_std_full", torch.from_numpy(std_full).view(1, NC, 1, 1))

        cos_lat = np.cos(np.deg2rad(LAT.astype(np.float64))).astype(np.float32)
        inv_cos_lat = (1.0 / np.maximum(cos_lat, 1e-3)).astype(np.float32)
        self.register_buffer("cos_lat", torch.from_numpy(cos_lat).view(1, 1, NLAT, 1))
        self.register_buffer("inv_cos_lat", torch.from_numpy(inv_cos_lat).view(1, 1, NLAT, 1))

    def forward(self, z_hist: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        """z_hist: [B, 3, 6, H, W], doy_sc: [B, 2] -> [B, 17, H, W]."""
        with torch.amp.autocast(z_hist.device.type, enabled=False):
            zf = z_hist.float()
            z_t = zf[:, -1]
            z_m1 = zf[:, -2]
            z_m2 = zf[:, -3]

            dz1 = z_t - z_m1
            dz0 = z_m1 - z_m2

            # Reconstruct true physical u10, v10, msl (shifted by 101325 Pa), tcwv
            clim_phys = eval_harmonic_4(doy_sc.float(), self.clim_phys_coef.float(), shift_minus_one=True)
            phys_t = z_t * self.anom_std_full.float() + clim_phys
            u = phys_t[:, 2:3]
            v = phys_t[:, 3:4]
            msl = phys_t[:, 4:5]
            tcwv = torch.clamp(phys_t[:, 5:6], min=0.0)

            dmsl_dx, dmsl_dy = spatial_grad_central(msl)
            dmsl_dx_s = (dmsl_dx * self.inv_cos_lat.float()) * 0.005
            dmsl_dy_s = dmsl_dy * 0.005

            du_dx, _ = spatial_grad_central(u)
            _, duc_dy = spatial_grad_central(u * self.cos_lat.float())
            dv_dx, _ = spatial_grad_central(v)
            _, dvc_dy = spatial_grad_central(v * self.cos_lat.float())

            div = (du_dx * self.inv_cos_lat.float() + dvc_dy * self.inv_cos_lat.float()) * 0.2
            vort = (dv_dx * self.inv_cos_lat.float() - duc_dy * self.inv_cos_lat.float()) * 0.2

            dqu_dx, _ = spatial_grad_central(tcwv * u)
            _, dqvc_dy = spatial_grad_central(tcwv * v * self.cos_lat.float())
            mfc = -(dqu_dx * self.inv_cos_lat.float() + dqvc_dy * self.inv_cos_lat.float()) * 0.01

            return torch.cat([dz1, dz0, dmsl_dx_s, dmsl_dy_s, div, vort, mfc], dim=1).to(z_hist.dtype)
