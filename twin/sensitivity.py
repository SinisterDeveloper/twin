"""Adjoint sensitivity (Butterfly Map), 1-step reverse shock solver, and physics self-checks."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .contract import CHANNELS, LAT, LON, MAX_STEPS, NC, NLAT, NLON
from .scoring import STD_DEFAULT

DEFAULT_SHOCK_CAPS: dict[str, float] = {
    "t2m": 3.0,
    "tp": 10.0,
    "u10": 5.0,
    "v10": 5.0,
    "msl": 500.0,
    "tcwv": 10.0,
}


def resolve_channel_index(channel: int | str) -> int:
    """Resolve channel name or index to [0, NC)."""
    if isinstance(channel, (int, np.integer)):
        idx = int(channel)
        if not 0 <= idx < NC:
            raise ValueError(f"Channel index out of range: {idx}")
        return idx
    if isinstance(channel, str):
        if channel in CHANNELS:
            return CHANNELS.index(channel)
        raise ValueError(f"Unknown channel name: {channel!r}")
    raise TypeError(f"Invalid channel type: {type(channel).__name__}")


def build_target_mask(region: dict) -> tuple[np.ndarray, tuple[float, float]]:
    """Build normalized [32, 36] target weights and (center_lat, center_lon) for point or box regions."""
    rtype = str(region.get("type", "")).lower()
    if not rtype:
        if "lat" in region and "lon" in region:
            rtype = "point"
        elif all(k in region for k in ("lat0", "lat1", "lon0", "lon1")):
            rtype = "box"

    mask = np.zeros((NLAT, NLON), dtype=np.float32)
    if rtype == "point":
        lat = float(region["lat"])
        lon = float(region["lon"])
        radius = int(region.get("radius_cells", 0))
        if radius < 0:
            raise ValueError(f"radius_cells must be >= 0, got {radius}")
        i0 = int(np.argmin(np.abs(LAT - lat)))
        j0 = int(np.argmin(np.abs(LON - lon)))
        i_lo, i_hi = max(0, i0 - radius), min(NLAT, i0 + radius + 1)
        j_lo, j_hi = max(0, j0 - radius), min(NLON, j0 + radius + 1)
        mask[i_lo:i_hi, j_lo:j_hi] = 1.0
        mask_w = (mask / float(mask.sum())).astype(np.float32)
        return mask_w, (float(LAT[i0]), float(LON[j0]))

    if rtype == "box":
        lat0, lat1 = float(region["lat0"]), float(region["lat1"])
        lon0, lon1 = float(region["lon0"]), float(region["lon1"])
        if lat0 > lat1:
            lat0, lat1 = lat1, lat0
        if lon0 > lon1:
            lon0, lon1 = lon1, lon0
        sel = ((LAT >= lat0) & (LAT <= lat1))[:, None] & ((LON >= lon0) & (LON <= lon1))[None, :]
        if not np.any(sel):
            i0 = int(np.argmin(np.abs(LAT - 0.5 * (lat0 + lat1))))
            j0 = int(np.argmin(np.abs(LON - 0.5 * (lon0 + lon1))))
            mask[i0, j0] = 1.0
        else:
            mask[sel] = 1.0
        mask_w = (mask / float(mask.sum())).astype(np.float32)
        center_lat = float((LAT[:, None] * mask_w).sum())
        center_lon = float((LON[None, :] * mask_w).sum())
        return mask_w, (center_lat, center_lon)

    raise ValueError(f"Unsupported region specification: {region!r}")


def _filter_bundle_models(bundle: dict, models: list[str] | None) -> dict:
    """Return a shallow bundle view restricted to the requested model names with renormalized weights."""
    if models is None:
        return bundle
    import torch

    all_models: dict = bundle["models"]
    all_names = list(all_models.keys())
    keep_idx = [i for i, name in enumerate(all_names) if name in models]
    if not keep_idx:
        raise ValueError(f"None of requested models {models} found in bundle {all_names}")
    sub_models = {all_names[i]: all_models[all_names[i]] for i in keep_idx}
    w = bundle["weights"][keep_idx]
    w = w / torch.clamp(w.sum(dim=0, keepdim=True), min=1e-8)
    return {**bundle, "models": sub_models, "weights": w}


def compute_adjoint_sensitivity(
    bundle: dict,
    init_states: np.ndarray,
    doy_sincos: list[float] | tuple[float, float],
    target_channel: int | str,
    target_day: int,
    mask_w: np.ndarray,
    models: list[str] | None = None,
) -> dict:
    """Unroll active ensemble for target_day steps and backprop scalar target J to input perturbation."""
    import torch
    from .rollout import rollout_ensemble_torch

    day = int(target_day)
    if not 1 <= day <= MAX_STEPS:
        raise ValueError(f"target_day must be in [1, {MAX_STEPS}], got {day}")
    ch_idx = resolve_channel_index(target_channel)
    sub_bundle = _filter_bundle_models(bundle, models)
    dev = sub_bundle["device"]

    init_t = torch.as_tensor(np.asarray(init_states, dtype=np.float32), dtype=torch.float32, device=dev)
    pert = torch.zeros((NC, NLAT, NLON), dtype=torch.float32, device=dev, requires_grad=True)
    w_t = torch.as_tensor(np.asarray(mask_w, dtype=np.float32), dtype=torch.float32, device=dev)

    ens = rollout_ensemble_torch(sub_bundle, init_t, doy_sincos, steps=day, perturbation=pert)
    j_val = torch.sum(w_t * ens[day - 1, ch_idx])
    baseline_value = float(j_val.detach().cpu().item())
    j_val.backward()

    grad = pert.grad.detach().cpu().numpy().astype(np.float32)
    clamped_zero_grad = bool(np.max(np.abs(grad)) < 1e-12)
    grad_std = (grad * STD_DEFAULT[:, None, None]).astype(np.float32)
    butterfly_map = np.sqrt(np.sum(grad_std ** 2, axis=0)).astype(np.float32)

    ch_energy = np.sum(grad_std ** 2, axis=(1, 2))
    tot_energy = float(ch_energy.sum())
    if tot_energy > 0.0:
        channel_importance = {ch: float(ch_energy[i] / tot_energy) for i, ch in enumerate(CHANNELS)}
    else:
        channel_importance = {ch: 0.0 for ch in CHANNELS}

    return {
        "grad": grad,
        "grad_std": grad_std,
        "butterfly_map": butterfly_map,
        "channel_importance": channel_importance,
        "clamped_zero_grad": clamped_zero_grad,
        "baseline_value": baseline_value,
    }


def compute_reverse_shock(
    grad: np.ndarray,
    init_states: np.ndarray,
    doy_sincos: list[float] | tuple[float, float],
    target_day: int,
    target_channel: int | str,
    mask_w: np.ndarray,
    forward_fn: Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray],
    direction: str = "increase",
    threshold: float | None = None,
    caps: dict[str, float] | None = None,
    allowed_channels: list[int | str] | None = None,
) -> dict:
    """Compute 1-step projected gradient reverse shock clip(eps * d, -cap, cap) and evaluate via forward_fn."""
    day = int(target_day)
    ch_idx = resolve_channel_index(target_channel)
    if direction not in ("increase", "decrease"):
        raise ValueError(f"direction must be 'increase' or 'decrease', got {direction!r}")
    sign = 1.0 if direction == "increase" else -1.0

    caps_merged = dict(DEFAULT_SHOCK_CAPS)
    if caps:
        for k, v in caps.items():
            if k in caps_merged:
                caps_merged[k] = float(v)
    cap_arr = np.array([caps_merged[ch] for ch in CHANNELS], dtype=np.float32)[:, None, None]

    grad_arr = np.asarray(grad, dtype=np.float32)
    d = (sign * (STD_DEFAULT[:, None, None] ** 2) * grad_arr).astype(np.float32)

    if allowed_channels is not None:
        allowed_idx = {resolve_channel_index(c) for c in allowed_channels}
        chan_mask = np.array([1.0 if i in allowed_idx else 0.0 for i in range(NC)], dtype=np.float32)[:, None, None]
        d = d * chan_mask

    base_out = np.asarray(forward_fn(init_states, list(doy_sincos), day, None), dtype=np.float32)
    baseline_value = float(np.sum(mask_w * base_out[day - 1, ch_idx]))

    scale_max = float(np.max(np.abs(d) / np.maximum(cap_arr, 1e-8)))
    if scale_max < 1e-12:
        pert = np.zeros((NC, NLAT, NLON), dtype=np.float32)
        eps = 0.0
        achieved_value = baseline_value
    else:
        eps_unit = 1.0 / scale_max
        if threshold is None:
            eps = eps_unit
        else:
            # First-order linear estimate of step size needed to reach threshold, capped at 5x peak saturation.
            lin_slope = float(np.sum(grad_arr * d))
            if abs(lin_slope) > 1e-12:
                eps_est = abs(float(threshold) - baseline_value) / abs(lin_slope)
                eps = min(max(eps_est, 0.0), 5.0 * eps_unit)
            else:
                eps = eps_unit
        pert = np.clip(eps * d, -cap_arr, cap_arr).astype(np.float32)
        pert_out = np.asarray(forward_fn(init_states, list(doy_sincos), day, pert), dtype=np.float32)
        achieved_value = float(np.sum(mask_w * pert_out[day - 1, ch_idx]))

    if threshold is None:
        target_met = True
    elif direction == "increase":
        target_met = achieved_value >= float(threshold)
    else:
        target_met = achieved_value <= float(threshold)

    linf_by_channel = {ch: float(np.max(np.abs(pert[i]))) for i, ch in enumerate(CHANNELS)}
    l2_norm_std = float(np.sqrt(np.sum((pert / STD_DEFAULT[:, None, None]) ** 2)))

    return {
        "perturbation": pert,
        "direction": direction,
        "threshold": float(threshold) if threshold is not None else None,
        "baseline_value": baseline_value,
        "achieved_value": achieved_value,
        "delta": achieved_value - baseline_value,
        "target_met": bool(target_met),
        "eps": float(eps),
        "linf_by_channel": linf_by_channel,
        "l2_norm_std": l2_norm_std,
    }


def verify_sensitivity_physics(
    grad: np.ndarray,
    butterfly_map: np.ndarray,
    init_states: np.ndarray,
    doy_sincos: list[float] | tuple[float, float],
    target_day: int,
    target_channel: int | str,
    mask_w: np.ndarray,
    forward_fn: Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray],
    eta: float = 0.05,
) -> dict:
    """Verify adjoint sensitivity via central finite difference and causal light-cone checks."""
    day = int(target_day)
    ch_idx = resolve_channel_index(target_channel)
    grad_arr = np.asarray(grad, dtype=np.float32)
    bmap = np.asarray(butterfly_map, dtype=np.float32)
    w_arr = np.asarray(mask_w, dtype=np.float32)

    grad_std = grad_arr * STD_DEFAULT[:, None, None]
    norm_std = float(np.sqrt(np.sum(grad_std ** 2)))

    if norm_std < 1e-12:
        fd_check = {
            "eta": float(eta),
            "adj_deriv": 0.0,
            "fd_deriv": 0.0,
            "same_sign": False,
            "rel_error": 1.0,
            "passed": False,
        }
    else:
        delta_phys = (float(eta) * (STD_DEFAULT[:, None, None] ** 2) * grad_arr / norm_std).astype(np.float32)
        out_plus = np.asarray(forward_fn(init_states, list(doy_sincos), day, delta_phys), dtype=np.float32)
        out_minus = np.asarray(forward_fn(init_states, list(doy_sincos), day, -delta_phys), dtype=np.float32)
        j_plus = float(np.sum(w_arr * out_plus[day - 1, ch_idx]))
        j_minus = float(np.sum(w_arr * out_minus[day - 1, ch_idx]))
        fd_deriv = (j_plus - j_minus) / (2.0 * float(eta))
        same_sign = bool(fd_deriv > 0.0)
        rel_error = float(abs(fd_deriv - norm_std) / max(abs(norm_std), 1e-12))
        fd_check = {
            "eta": float(eta),
            "adj_deriv": norm_std,
            "fd_deriv": float(fd_deriv),
            "same_sign": same_sign,
            "rel_error": rel_error,
            "passed": bool(same_sign and rel_error <= 0.20),
        }

    # Causal cone check
    ii, jj = np.indices((NLAT, NLON))
    bi, bj = np.where(w_arr > 0.0)
    dist = np.sqrt((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2).min(axis=-1)

    energy = bmap.astype(np.float64) ** 2
    tot_energy = float(energy.sum())
    far_bound = 3.0 + 8.0 * float(day)

    if tot_energy <= 1e-24:
        r90_cells = 0.0
        cone_speed = 0.0
        far_energy_frac = 0.0
        centroid_lat = float((LAT[:, None] * w_arr).sum())
        centroid_lon = float((LON[None, :] * w_arr).sum())
        cone_passed = False
    else:
        order = np.argsort(dist.ravel())
        dist_sorted = dist.ravel()[order]
        cum_e = np.cumsum(energy.ravel()[order])
        idx_90 = int(np.searchsorted(cum_e, 0.90 * tot_energy, side="left"))
        idx_90 = min(idx_90, len(dist_sorted) - 1)
        r90_cells = float(dist_sorted[idx_90])
        cone_speed = float(r90_cells / max(float(day), 1.0))
        far_energy_frac = float(energy[dist > far_bound].sum() / tot_energy)
        centroid_lat = float((LAT[:, None] * energy).sum() / tot_energy)
        centroid_lon = float((LON[None, :] * energy).sum() / tot_energy)
        cone_passed = bool(cone_speed <= 8.0 and far_energy_frac < 0.05)

    cone_check = {
        "r90_cells": r90_cells,
        "cone_speed_cells_per_day": cone_speed,
        "far_bound_cells": far_bound,
        "far_energy_frac": far_energy_frac,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "passed": cone_passed,
    }

    return {
        "fd_check": fd_check,
        "cone_check": cone_check,
        "physics_consistent": bool(fd_check["passed"] and cone_check["passed"]),
    }
