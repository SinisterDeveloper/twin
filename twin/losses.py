"""Composite physics-aware loss functions aligned with Round-1 scoring metrics."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import LEAD_DAYS, lat_weights

HEAVY_RAIN_MM = 10.0


class MonsoonLoss(nn.Module):
    """Lat/Lead-weighted Charbonnier+MSE + Soft-CSI & Physical tp + Gradient + Physical Anti-Blur."""

    def __init__(
        self,
        clim_tr: torch.Tensor,
        anom_std: torch.Tensor,
        w_rain: float = 0.30,
        w_csi: float = 0.25,
        w_grad: float = 0.15,
        w_var: float = 0.15,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("lat_w", torch.from_numpy(lat_weights()).view(1, 1, 1, -1, 1))
        self.register_buffer("clim_tr", clim_tr)  # [365, 6, 32, 36]
        self.register_buffer("anom_std", anom_std.view(1, 1, 6, 1, 1))
        self.w_rain = w_rain
        self.w_csi = w_csi
        self.w_grad = w_grad
        self.w_var = w_var
        self.huber_delta = huber_delta

    def _spatial_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    def _lead_weights(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = torch.ones(steps, device=device, dtype=dtype)
        for d in LEAD_DAYS:
            if d - 1 < steps:
                w[d - 1] = 1.5
        w = w / w.mean()
        return w.view(1, steps, 1, 1, 1)

    def forward(
        self,
        z_pred: torch.Tensor,
        z_true: torch.Tensor,
        doy_tgt: torch.Tensor,
        phys_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Shapes: z_pred/z_true [B, S, 6, 32, 36], doy_tgt [B, S], phys_true [B, S, 6, 32, 36]."""
        zp = z_pred.float()
        zt = z_true.float()
        pt = phys_true.float()
        steps = zp.shape[1]
        lead_w = self._lead_weights(steps, zp.device, zp.dtype)
        spatial_w = self.lat_w * lead_w

        # 1. Lead & latitude weighted Huber + MSE blend in normalized anomaly space
        huber = F.huber_loss(zp, zt, reduction="none", delta=self.huber_delta)
        mse = (zp - zt) ** 2
        loss_base = ((huber + 0.25 * mse) * spatial_w).mean()

        # Reconstruct physical predictions carefully in float32 for rain & Day 5-10 anti-blur
        clim_sel = self.clim_tr[doy_tgt]  # [B, S, 6, 32, 36]
        pred_tr = zp * self.anom_std + clim_sel
        tp_pred = torch.expm1(torch.clamp(pred_tr[:, :, 1], min=0.0, max=8.0))
        tp_true = pt[:, :, 1]

        # 2. Heavy-rain emphasis in z-space + physical-space Jensen-bias correction
        loss_rain = zp.new_tensor(0.0)
        if self.w_rain > 0:
            heavy_w = 1.0 + 3.0 * torch.sigmoid((tp_true - HEAVY_RAIN_MM) * 0.5)
            tp_err_z = F.huber_loss(zp[:, :, 1], zt[:, :, 1], reduction="none", delta=1.0)
            tp_err_phys = F.huber_loss(tp_pred * 0.1, tp_true * 0.1, reduction="none", delta=1.0)
            loss_rain = ((tp_err_z * heavy_w + 0.15 * tp_err_phys) * spatial_w[:, :, 0]).mean()

        # 3. Differentiable Soft-CSI surrogate at 10 mm/day
        loss_csi = zp.new_tensor(0.0)
        if self.w_csi > 0:
            p_soft = torch.sigmoid((tp_pred - HEAVY_RAIN_MM) / 2.0)
            t_bin = (tp_true > HEAVY_RAIN_MM).float()
            inter = (p_soft * t_bin).sum()
            union = p_soft.sum() + t_bin.sum() - inter + 1e-4
            loss_csi = 1.0 - inter / union

        # 4. Spatial gradient loss (prevents blurry fronts)
        loss_grad = zp.new_tensor(0.0)
        if self.w_grad > 0:
            px, py = self._spatial_grad(zp)
            tx, ty = self._spatial_grad(zt)
            loss_grad = F.l1_loss(px, tx) + F.l1_loss(py, ty)

        # 5. Physical-space spatial variance preservation (targets Day 5-10 anti-blur >= 0.40)
        loss_var = zp.new_tensor(0.0)
        if self.w_var > 0:
            s0 = 4 if steps >= 5 else 0
            late_tr = pred_tr[:, s0:]
            tp_late = torch.expm1(torch.clamp(late_tr[:, :, 1:2], min=0.0, max=8.0))
            pred_phys_late = torch.cat([late_tr[:, :, 0:1], tp_late, late_tr[:, :, 2:]], dim=2)
            true_phys_late = pt[:, s0:]
            vp = pred_phys_late.var(dim=(-2, -1), unbiased=False)
            vt = true_phys_late.var(dim=(-2, -1), unbiased=False)
            ratio_std = torch.sqrt(vp / (vt + 1e-4) + 1e-6)
            loss_var = F.l1_loss(ratio_std, torch.ones_like(ratio_std))

        total = (
            loss_base
            + self.w_rain * loss_rain
            + self.w_csi * loss_csi
            + self.w_grad * loss_grad
            + self.w_var * loss_var
        )
        stats = {
            "loss": float(total.detach()),
            "base": float(loss_base.detach()),
            "rain": float(loss_rain.detach()),
            "csi": float(loss_csi.detach()),
            "grad": float(loss_grad.detach()),
            "var": float(loss_var.detach()),
        }
        return total, stats


def shock_stability_loss(
    z_base: torch.Tensor,
    z_pert: torch.Tensor,
    doy_tgt: torch.Tensor,
    clim_tr: torch.Tensor,
    anom_std: torch.Tensor,
    phys_std: torch.Tensor,
) -> torch.Tensor:
    """Bound convective tp runaway under random thermodynamic perturbations while keeping baseline detached."""
    zb = z_base.float()
    zp = z_pert.float()
    clim_sel = clim_tr[doy_tgt].float()
    a_std = anom_std.view(1, 1, 6, 1, 1).float()
    p_std = phys_std.view(1, 1, 6, 1, 1).float()

    tp_base = torch.expm1(torch.clamp(zb[:, :, 1] * a_std[:, :, 1] + clim_sel[:, :, 1], 0.0, 8.0))
    tp_pert = torch.expm1(torch.clamp(zp[:, :, 1] * a_std[:, :, 1] + clim_sel[:, :, 1], 0.0, 8.0))
    tp_z = torch.abs(tp_pert - tp_base) / phys_std[1].float()
    all_z = torch.abs(zp - zb) * a_std / p_std
    return F.relu(tp_z - 2.8).pow(2).mean() + 0.25 * F.relu(all_z - 3.5).pow(2).mean()
