"""Model registry for Monsoon Climate Digital Twin."""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from .advect_resnet import AdvectResNet
from .convnext_unet import ConvNeXtUNet


def build_model(name: str, static_grid: np.ndarray, **kwargs) -> nn.Module:
    if name == "convnext_unet":
        return ConvNeXtUNet(static_grid=static_grid, **kwargs)
    if name == "advect_resnet":
        return AdvectResNet(static_grid=static_grid, **kwargs)
    raise ValueError(f"Unknown model family: {name}")
