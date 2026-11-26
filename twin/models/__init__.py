"""Model registry for Monsoon Climate Digital Twin."""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from .advect_resnet import AdvectResNet
from .convnext_unet import ConvNeXtUNet
from .local_swin import LocalWeatherFormer
from .pde_net import LocalPDENet

MODEL_NAMES = ("convnext_unet", "advect_resnet", "local_swin", "pde_net")


def build_model(
    name: str,
    static_grid: np.ndarray,
    clim_uv_coef: np.ndarray | None = None,
    anom_std: np.ndarray | None = None,
    **kwargs,
) -> nn.Module:
    if name == "convnext_unet":
        return ConvNeXtUNet(static_grid=static_grid, clim_uv_coef=clim_uv_coef, anom_std=anom_std, **kwargs)
    if name == "advect_resnet":
        return AdvectResNet(static_grid=static_grid, clim_uv_coef=clim_uv_coef, anom_std=anom_std, **kwargs)
    if name == "local_swin":
        return LocalWeatherFormer(static_grid=static_grid, clim_uv_coef=clim_uv_coef, anom_std=anom_std, **kwargs)
    if name == "pde_net":
        return LocalPDENet(static_grid=static_grid, clim_uv_coef=clim_uv_coef, anom_std=anom_std, **kwargs)
    raise ValueError(f"Unknown model family: {name}")
