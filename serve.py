"""Offline FastAPI inference server for Multiverse of Monsoon (Round 1)."""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()
ART_DIR = Path(__file__).resolve().parent / "artifacts"

_SESSIONS = []
_SESSION_WEIGHTS = None
_CLIM = None
_ANOM_STD = None
MODEL_NAME = "monsoon-twin-persistence-fallback"


_SESSION_STEMS: list[str] = []


def _load_artifacts() -> None:
    global _SESSIONS, _SESSION_WEIGHTS, _SESSION_STEMS, _CLIM, _ANOM_STD, MODEL_NAME
    clim_p = ART_DIR / "clim.npy"
    stats_p = ART_DIR / "norm_stats.json"
    onnx_files = sorted(ART_DIR.glob("*.onnx"))

    if clim_p.exists() and stats_p.exists() and onnx_files:
        import onnxruntime as ort

        _CLIM = np.load(clim_p).astype(np.float32)
        meta = json.loads(stats_p.read_text())
        _ANOM_STD = np.array(meta["anom_std"], dtype=np.float32)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
        _SESSIONS = [ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"]) for p in onnx_files]
        _SESSION_STEMS = [p.stem for p in onnx_files]

        weights_p = ART_DIR / "ensemble_weights.json"
        if weights_p.exists():
            raw_w = json.loads(weights_p.read_text())
            w = np.zeros((len(onnx_files), 6), dtype=np.float32)
            matched = False
            for i, p in enumerate(onnx_files):
                if p.stem in raw_w:
                    matched = True
                    v = raw_w[p.stem]
                    w[i] = np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple)) else float(v)
            if matched:
                w = np.maximum(w, 0.0)
                col_sum = w.sum(axis=0, keepdims=True)
                _SESSION_WEIGHTS = np.where(col_sum > 1e-8, w / np.maximum(col_sum, 1e-8), 1.0 / len(onnx_files)).astype(
                    np.float32
                )

        # Warm up all ONNX sessions before reporting ready so first /simulate has zero JIT spike
        zh_warm = np.zeros((1, 3, 6, 32, 36), dtype=np.float32)
        sc_warm = np.array([[0.0, 1.0]], dtype=np.float32)
        for sess in _SESSIONS:
            sess.run(None, {"z_hist": zh_warm, "doy_sc": sc_warm})

        MODEL_NAME = "monsoon-twin-ensemble(" + "+".join(p.stem for p in onnx_files) + ")"


_load_artifacts()


def _arr(x) -> np.ndarray:
    if isinstance(x, dict):
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def _transform_phys(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    out[..., 1, :, :] = np.log1p(np.maximum(out[..., 1, :, :], 0.0))
    return out


def _inv_transform_phys(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    out[..., 1, :, :] = np.expm1(np.clip(out[..., 1, :, :], 0.0, 4.15))
    out[..., 0, :, :] = np.clip(out[..., 0, :, :], 233.15, 333.15)
    return out


def _advance_sc(sin_v: float, cos_v: float) -> tuple[float, float]:
    ang = 2.0 * np.pi / 365.0
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    return sin_v * ca + cos_v * sa, cos_v * ca - sin_v * sa


def simulate_with_families(
    init_states: np.ndarray,
    doy_sincos: list[float],
    steps: int,
    perturbation: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run autoregressive ensemble rollout and return (ensemble, per_family_rollouts)."""
    x = init_states.astype(np.float32).copy()
    if perturbation is not None:
        x[-1] = x[-1] + perturbation.astype(np.float32)

    if not _SESSIONS or _CLIM is None or _ANOM_STD is None:
        out = np.repeat(x[-1][None], steps, axis=0)
        out[:, 1] = np.maximum(out[:, 1], 0.0)
        out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
        return out, {}

    s_v, c_v = float(doy_sincos[0]), float(doy_sincos[1])
    ang = float(np.arctan2(s_v, c_v))
    if ang < 0.0:
        ang += 2.0 * np.pi
    doy_last = (int(round(ang * 365.0 / (2.0 * np.pi))) - 1) % 365
    doy_hist = np.array([(doy_last - 2) % 365, (doy_last - 1) % 365, doy_last], dtype=np.int64)

    z_hist = ((_transform_phys(x) - _CLIM[doy_hist]) / _ANOM_STD[None, :, None, None]).astype(np.float32)

    use_weights = _SESSION_WEIGHTS is not None and len(_SESSIONS) == _SESSION_WEIGHTS.shape[0]
    active_idx = [
        i
        for i in range(len(_SESSIONS))
        if not use_weights or float(np.max(_SESSION_WEIGHTS[i])) > 1e-8
    ]
    if not active_idx:
        active_idx = list(range(len(_SESSIONS)))

    all_rollouts = []
    family_rollouts: dict[str, np.ndarray] = {}
    for i in active_idx:
        sess = _SESSIONS[i]
        stem = _SESSION_STEMS[i] if i < len(_SESSION_STEMS) else f"model_{i}"
        zh = z_hist.copy()
        sv, cv = s_v, c_v
        preds = []
        for k in range(steps):
            sc_arr = np.array([[sv, cv]], dtype=np.float32)
            z_next = sess.run(None, {"z_hist": zh[None], "doy_sc": sc_arr})[0][0]
            doy_k = (doy_last + k + 1) % 365
            t_phys = z_next * _ANOM_STD[:, None, None] + _CLIM[doy_k]
            preds.append(_inv_transform_phys(t_phys))
            zh = np.concatenate([zh[1:], z_next[None]], axis=0)
            sv, cv = _advance_sc(sv, cv)
        fam_arr = np.stack(preds, axis=0).astype(np.float32)
        all_rollouts.append(fam_arr)
        family_rollouts[stem] = fam_arr

    stacked = np.stack(all_rollouts, axis=0).astype(np.float32)
    if use_weights:
        w_act = _SESSION_WEIGHTS[active_idx]
        w_act = w_act / np.maximum(w_act.sum(axis=0, keepdims=True), 1e-8)
        ens = np.sum(stacked * w_act[:, None, :, None, None], axis=0).astype(np.float32)
    else:
        ens = np.mean(stacked, axis=0).astype(np.float32)
    ens[:, 1] = np.maximum(ens[:, 1], 0.0)
    ens[:, 0] = np.clip(ens[:, 0], 233.15, 333.15)
    return ens, family_rollouts


def simulate(init_states: np.ndarray, doy_sincos: list[float], steps: int, perturbation: np.ndarray | None) -> np.ndarray:
    ens, _ = simulate_with_families(init_states, doy_sincos, steps, perturbation)
    return ens


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    body = await request.json()
    init = _arr(body["init_states"])
    pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


_TORCH_BUNDLE = None
_TORCH_LOCK = threading.Lock()


def _get_torch_bundle() -> dict:
    """Lazily load PyTorch ensemble bundle in a thread-safe manner."""
    global _TORCH_BUNDLE
    if _TORCH_BUNDLE is not None:
        return _TORCH_BUNDLE
    with _TORCH_LOCK:
        if _TORCH_BUNDLE is not None:
            return _TORCH_BUNDLE
        import torch  # noqa: F401
        from twin.rollout import load_ensemble_torch

        _TORCH_BUNDLE = load_ensemble_torch(ART_DIR, device="cpu")
        return _TORCH_BUNDLE


@app.post("/sensitivity")
async def sensitivity_endpoint(request: Request):
    from twin.contract import CHANNELS
    from twin.sensitivity import (
        build_target_mask,
        compute_adjoint_sensitivity,
        compute_reverse_shock,
        resolve_channel_index,
        verify_sensitivity_physics,
    )

    t0 = time.perf_counter()
    body = await request.json()
    try:
        init = _arr(body["init_states"])
        doy_sc = [float(body["doy_sincos"][0]), float(body["doy_sincos"][1])]
        target_day = int(body["target_day"])
        target_channel = body["target_channel"]
        ch_idx = resolve_channel_index(target_channel)
        ch_name = CHANNELS[ch_idx]
        region = body["region"]
        mask_w, (center_lat, center_lon) = build_target_mask(region)
        direction = str(body.get("direction", "increase"))
        threshold = body.get("threshold")
        threshold_val = None if threshold is None else float(threshold)
        caps = body.get("caps")
        allowed_channels = body.get("allowed_channels")
        models = body.get("models")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        bundle = _get_torch_bundle()
    except (ImportError, FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Adjoint PyTorch backend unavailable: {exc}") from exc

    adj = compute_adjoint_sensitivity(
        bundle=bundle,
        init_states=init,
        doy_sincos=doy_sc,
        target_channel=ch_idx,
        target_day=target_day,
        mask_w=mask_w,
        models=models,
    )
    rev = compute_reverse_shock(
        grad=adj["grad"],
        init_states=init,
        doy_sincos=doy_sc,
        target_day=target_day,
        target_channel=ch_idx,
        mask_w=mask_w,
        forward_fn=simulate,
        direction=direction,
        threshold=threshold_val,
        caps=caps,
        allowed_channels=allowed_channels,
    )
    self_check = verify_sensitivity_physics(
        grad=adj["grad"],
        butterfly_map=adj["butterfly_map"],
        init_states=init,
        doy_sincos=doy_sc,
        target_day=target_day,
        target_channel=ch_idx,
        mask_w=mask_w,
        forward_fn=simulate,
    )

    rev_out = {
        **rev,
        "perturbation": np.asarray(rev["perturbation"], dtype="f4").tolist(),
    }
    return {
        "target": {
            "lat": center_lat,
            "lon": center_lon,
            "channel": ch_name,
            "day": target_day,
            "mask": np.asarray(mask_w, dtype="f4").tolist(),
        },
        "butterfly_map": np.asarray(adj["butterfly_map"], dtype="f4").tolist(),
        "sensitivity_signed": np.asarray(adj["grad_std"], dtype="f4").tolist(),
        "channel_importance": adj["channel_importance"],
        "reverse_shock": rev_out,
        "self_check": self_check,
        "clamped_zero_grad": bool(adj["clamped_zero_grad"]),
        "compute_ms": (time.perf_counter() - t0) * 1000.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
