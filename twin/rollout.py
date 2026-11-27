"""Differentiable PyTorch ensemble rollout reproducing serve.simulate end-to-end."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .contract import NC, T_IN
from .data import inv_transform_physical_torch, transform_physical_torch
from .models import build_model


def load_ensemble_torch(
    art_dir: Path = Path("artifacts"),
    pt_dir: Path | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Load frozen PyTorch models, active ensemble weights, climatology, and norm stats."""
    dev = torch.device(device)
    pt_root = pt_dir or (art_dir / "pt")

    static_grid = np.load(art_dir / "static.npy")
    uv_p = art_dir / "clim_uv_coef.npy"
    clim_uv_coef = np.load(uv_p) if uv_p.exists() else None
    phys_p = art_dir / "clim_phys_coef.npy"
    clim_phys_coef = np.load(phys_p) if phys_p.exists() else None
    stats = json.loads((art_dir / "norm_stats.json").read_text())
    anom_std_np = np.array(stats["anom_std"], dtype=np.float32)

    raw_w = json.loads((art_dir / "ensemble_weights.json").read_text())
    onnx_stems = sorted(p.stem for p in art_dir.glob("*.onnx"))
    if not onnx_stems:
        onnx_stems = sorted(raw_w.keys())

    models: dict[str, torch.nn.Module] = {}
    weight_rows: list[np.ndarray] = []
    for stem in onnx_stems:
        v = raw_w.get(stem, 0.0)
        w_vec = np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple)) else np.full(NC, float(v), dtype=np.float32)
        w_vec = np.maximum(w_vec, 0.0)
        if float(np.max(w_vec)) <= 1e-8:
            continue
        ckpt_p = pt_root / f"{stem}_best.pt"
        ckpt = torch.load(ckpt_p, map_location=dev, weights_only=False)
        m = build_model(
            ckpt["model_name"],
            static_grid=static_grid,
            clim_uv_coef=clim_uv_coef,
            clim_phys_coef=clim_phys_coef,
            anom_std=anom_std_np,
        ).to(dev)
        m.load_state_dict(ckpt["model"])
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[stem] = m
        weight_rows.append(w_vec)

    w_mat = np.stack(weight_rows, axis=0)
    w_mat = w_mat / np.maximum(w_mat.sum(axis=0, keepdims=True), 1e-8)

    clim_t = torch.from_numpy(np.load(art_dir / "clim.npy").astype(np.float32)).to(dev)
    anom_std_t = torch.from_numpy(anom_std_np).to(dev)
    weights_t = torch.from_numpy(w_mat.astype(np.float32)).to(dev)

    return {
        "models": models,
        "weights": weights_t,
        "clim": clim_t,
        "anom_std": anom_std_t,
        "device": dev,
    }


def _doy_last_from_sincos(doy_sincos: list[float] | tuple[float, float] | torch.Tensor) -> int:
    s_v = float(doy_sincos[0])
    c_v = float(doy_sincos[1])
    ang = math.atan2(s_v, c_v)
    if ang < 0.0:
        ang += 2.0 * math.pi
    return (int(round(ang * 365.0 / (2.0 * math.pi))) - 1) % 365


def rollout_ensemble_torch(
    bundle: dict,
    init_states: torch.Tensor,
    doy_sincos: list[float] | tuple[float, float],
    steps: int,
    perturbation: torch.Tensor | None = None,
    return_families: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiably unroll the active ensemble in physical space matching serve.simulate."""
    dev = bundle["device"]
    clim: torch.Tensor = bundle["clim"]
    anom_std: torch.Tensor = bundle["anom_std"]
    models: dict[str, torch.nn.Module] = bundle["models"]
    weights: torch.Tensor = bundle["weights"]

    x = init_states.to(device=dev, dtype=torch.float32)
    if perturbation is not None:
        pert = perturbation.to(device=dev, dtype=torch.float32)
        x = torch.cat([x[:-1], (x[-1] + pert).unsqueeze(0)], dim=0)

    s_v = float(doy_sincos[0])
    c_v = float(doy_sincos[1])
    doy_last = _doy_last_from_sincos((s_v, c_v))
    doy_hist = torch.tensor([(doy_last - 2) % 365, (doy_last - 1) % 365, doy_last], dtype=torch.long, device=dev)

    z_hist = (transform_physical_torch(x) - clim[doy_hist]) / anom_std.view(1, NC, 1, 1)

    ang_step = 2.0 * math.pi / 365.0
    ca, sa = math.cos(ang_step), math.sin(ang_step)

    family_rollouts: dict[str, torch.Tensor] = {}
    stacked_list: list[torch.Tensor] = []
    for name, model in models.items():
        zh = z_hist.unsqueeze(0)  # [1, T_IN, NC, 32, 36]
        sv, cv = s_v, c_v
        preds: list[torch.Tensor] = []
        for k in range(steps):
            sc_t = torch.tensor([[sv, cv]], dtype=torch.float32, device=dev)
            z_next = model(zh, sc_t)[0]
            doy_k = (doy_last + k + 1) % 365
            t_phys = z_next * anom_std.view(NC, 1, 1) + clim[doy_k]
            phys_k = inv_transform_physical_torch(t_phys)
            t2m_clamped = torch.clamp(phys_k[0:1], min=233.15, max=333.15)
            phys_k = torch.cat([t2m_clamped, phys_k[1:]], dim=0)
            preds.append(phys_k)
            zh = torch.cat([zh[:, 1:], z_next.unsqueeze(0).unsqueeze(1)], dim=1)
            sv, cv = sv * ca + cv * sa, cv * ca - sv * sa
        fam_roll = torch.stack(preds, dim=0)
        family_rollouts[name] = fam_roll
        stacked_list.append(fam_roll)

    stacked = torch.stack(stacked_list, dim=0)  # [M, steps, NC, 32, 36]
    ens = torch.sum(stacked * weights.view(-1, 1, NC, 1, 1), dim=0)
    ens = torch.cat(
        [
            torch.clamp(ens[:, 0:1], min=233.15, max=333.15),
            torch.clamp(ens[:, 1:2], min=0.0),
            ens[:, 2:],
        ],
        dim=1,
    )
    if return_families:
        return ens, family_rollouts
    return ens
