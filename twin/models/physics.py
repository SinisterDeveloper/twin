"""Shared strictly local physics feature extractors and harmonic trig helpers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ONE_DAY_RAD = 2.0 * math.pi / 365.0


def eval_harmonic_4(doy_sc: torch.Tensor, coef: torch.Tensor, shift_minus_one: bool = True) -> torch.Tensor:
    """Evaluate 4-harmonic climatology from [sin theta, cos theta] and coef [9, C, H, W].

    Args:
        doy_sc: [B, 2] containing [sin(2*pi*doy1/365), cos(2*pi*doy1/365)] for target step.
        coef: [9, C, H, W] fitted with 0-based index (day 0 -> angle 0).
        shift_minus_one: If True, shifts doy_sc back by 1 day so it matches z_last's calendar day.
    """
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
    # Use explicit reshape + matmul instead of einsum for rock-solid ONNX export
    b = basis.shape[0]
    c, h, w = coef.shape[1], coef.shape[2], coef.shape[3]
    return torch.matmul(basis, coef.reshape(9, c * h * w)).reshape(b, c, h, w)


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
    """Computes 17 strictly local physical tendency and finite-difference channels from z_hist."""

    OUT_CHANNELS = 17

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        """z_hist: [B, 3, 6, H, W] -> [B, 17, H, W]."""
        z_t = z_hist[:, -1]
        z_m1 = z_hist[:, -2]
        z_m2 = z_hist[:, -3]

        dz1 = z_t - z_m1
        dz0 = z_m1 - z_m2

        dx, dy = spatial_grad_central(z_t)
        dmsl_dx = dx[:, 4:5]
        dmsl_dy = dy[:, 4:5]
        div = dx[:, 2:3] + dy[:, 3:4]
        vort = dx[:, 3:4] - dy[:, 2:3]

        qu = z_t[:, 5:6] * z_t[:, 2:3]
        qv = z_t[:, 5:6] * z_t[:, 3:4]
        dqu_dx, _ = spatial_grad_central(qu)
        _, dqv_dy = spatial_grad_central(qv)
        mfc = -(dqu_dx + dqv_dy)

        return torch.cat([dz1, dz0, dmsl_dx, dmsl_dy, div, vort, mfc], dim=1)
