"""Composite physics-aware loss functions for Monsoon Climate Digital Twin."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import lat_weights

HEAVY_RAIN_MM = 10.0


class MonsoonLoss(nn.Module):
    """Lat-weighted Huber + Heavy-Rain emphasis + Spatial Gradient & Variance anti-blur."""

    def __init__(
        self,
        clim_tr: torch.Tensor,
        anom_std: torch.Tensor,
        w_rain: float = 0.35,
        w_grad: float = 0.15,
        w_var: float = 0.05,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("lat_w", torch.from_numpy(lat_weights()).view(1, 1, 1, -1, 1))
        self.register_buffer("clim_tr", clim_tr)  # [365, 6, 32, 36]
        self.register_buffer("anom_std", anom_std.view(1, 1, 6, 1, 1))
        self.w_rain = w_rain
        self.w_grad = w_grad
        self.w_var = w_var
        self.huber_delta = huber_delta

    def _spatial_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    def forward(
        self,
        z_pred: torch.Tensor,
        z_true: torch.Tensor,
        doy_tgt: torch.Tensor,
        phys_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Shapes: z_pred/z_true [B, S, 6, 32, 36], doy_tgt [B, S], phys_true [B, S, 6, 32, 36]."""
        # 1. Latitude-weighted Huber loss in normalized anomaly space
        huber = F.huber_loss(z_pred, z_true, reduction="none", delta=self.huber_delta)
        loss_base = (huber * self.lat_w).mean()

        # 2. Heavy-rain emphasis on tp (channel 1) in log1p space + soft-CSI surrogate
        loss_rain = z_pred.new_tensor(0.0)
        if self.w_rain > 0:
            tp_true = phys_true[:, :, 1]  # [B, S, 32, 36]
            heavy_w = 1.0 + 3.0 * torch.sigmoid((tp_true - HEAVY_RAIN_MM) * 0.5)
            tp_err = F.huber_loss(z_pred[:, :, 1], z_true[:, :, 1], reduction="none", delta=1.0)
            loss_rain = (tp_err * heavy_w * self.lat_w[:, :, 0]).mean()

        # 3. Spatial gradient loss (prevents blurry fronts)
        loss_grad = z_pred.new_tensor(0.0)
        if self.w_grad > 0:
            px, py = self._spatial_grad(z_pred)
            tx, ty = self._spatial_grad(z_true)
            loss_grad = F.l1_loss(px, tx) + F.l1_loss(py, ty)

        # 4. Spatial variance preservation (directly guards against Day-8 blur < 40% variance)
        loss_var = z_pred.new_tensor(0.0)
        if self.w_var > 0:
            vp = z_pred.var(dim=(-2, -1), unbiased=False)
            vt = z_true.var(dim=(-2, -1), unbiased=False)
            loss_var = F.l1_loss(torch.sqrt(vp + 1e-6), torch.sqrt(vt + 1e-6))

        total = loss_base + self.w_rain * loss_rain + self.w_grad * loss_grad + self.w_var * loss_var
        stats = {
            "loss": float(total.detach()),
            "base": float(loss_base.detach()),
            "rain": float(loss_rain.detach()),
            "grad": float(loss_grad.detach()),
            "var": float(loss_var.detach()),
        }
        return total, stats
