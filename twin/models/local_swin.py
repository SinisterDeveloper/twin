"""Family C: Strictly Local Shifted-Window WeatherFormer (LocalSwin v3)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..contract import NC, T_IN
from .convnext_unet import FiLM
from .physics import ChannelLayerNorm2d, PhysicsFeatureBlock


class LocalWindowBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4, shifted: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ws = window_size
        self.shifted = shifted
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = ChannelLayerNorm2d(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        n_tok = window_size * window_size
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, n_tok, n_tok))

        self.norm2 = ChannelLayerNorm2d(dim)
        self.film = FiLM(2, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )

    def _window_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.ws
        nh, nw = h // ws, w // ws

        wins = x.view(b, c, nh, ws, nw, ws).permute(0, 2, 4, 3, 5, 1).reshape(b * nh * nw, ws * ws, c)
        qkv = self.qkv(wins).reshape(b * nh * nw, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + self.rel_pos_bias.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b * nh * nw, ws * ws, c)
        out = self.proj(out)

        return out.reshape(b, nh, nw, ws, ws, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor, doy_sc: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.shifted:
            s = self.ws // 2
            hp = F.pad(h, (s, s, s, s), mode="replicate")
            ap = self._window_attn(hp)
            a = ap[:, :, s:-s, s:-s]
        else:
            a = self._window_attn(h)
        x = x + a

        m = self.film(self.norm2(x), doy_sc)
        x = x + self.mlp(m)
        return x


class LocalWeatherFormer(nn.Module):
    def __init__(
        self,
        static_grid: np.ndarray,
        clim_uv_coef: np.ndarray | None = None,
        clim_phys_coef: np.ndarray | None = None,
        anom_std: np.ndarray | None = None,
        dim: int = 96,
        num_heads: int = 4,
        window_size: int = 4,
    ):
        super().__init__()
        self.register_buffer("static_grid", torch.from_numpy(static_grid).unsqueeze(0))
        self.phys = PhysicsFeatureBlock(clim_phys_coef=clim_phys_coef, anom_std=anom_std)
        in_ch = T_IN * NC + PhysicsFeatureBlock.OUT_CHANNELS + static_grid.shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1, padding_mode="replicate"),
            ChannelLayerNorm2d(dim),
            nn.GELU(),
        )
        self.block1 = LocalWindowBlock(dim=dim, num_heads=num_heads, window_size=window_size, shifted=False)
        self.block2 = LocalWindowBlock(dim=dim, num_heads=num_heads, window_size=window_size, shifted=True)
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

        pf = self.phys(z_seq, doy_sc)
        s = self.static_grid.expand(b, -1, -1, -1)
        feat = self.stem(torch.cat([z_flat, pf, s], dim=1))
        feat = self.block1(feat, doy_sc)
        feat = self.block2(feat, doy_sc)
        delta = self.head(feat)
        return z_last + delta
