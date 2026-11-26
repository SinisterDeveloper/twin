"""Offline FastAPI inference server for Multiverse of Monsoon (Round 1)."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
ART_DIR = Path(__file__).resolve().parent / "artifacts"

_SESSIONS = []
_SESSION_WEIGHTS = None
_CLIM = None
_ANOM_STD = None
MODEL_NAME = "monsoon-twin-persistence-fallback"


def _load_artifacts() -> None:
    global _SESSIONS, _SESSION_WEIGHTS, _CLIM, _ANOM_STD, MODEL_NAME
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
    out[..., 1, :, :] = np.expm1(np.clip(out[..., 1, :, :], 0.0, 8.0))
    out[..., 0, :, :] = np.clip(out[..., 0, :, :], 233.15, 333.15)
    return out


def _advance_sc(sin_v: float, cos_v: float) -> tuple[float, float]:
    ang = 2.0 * np.pi / 365.0
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    return sin_v * ca + cos_v * sa, cos_v * ca - sin_v * sa


def simulate(init_states: np.ndarray, doy_sincos: list[float], steps: int, perturbation: np.ndarray | None) -> np.ndarray:
    x = init_states.astype(np.float32).copy()
    if perturbation is not None:
        x[-1] = x[-1] + perturbation.astype(np.float32)

    if not _SESSIONS or _CLIM is None or _ANOM_STD is None:
        out = np.repeat(x[-1][None], steps, axis=0)
        out[:, 1] = np.maximum(out[:, 1], 0.0)
        out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
        return out

    s_v, c_v = float(doy_sincos[0]), float(doy_sincos[1])
    ang = float(np.arctan2(s_v, c_v))
    if ang < 0.0:
        ang += 2.0 * np.pi
    doy_last = (int(round(ang * 365.0 / (2.0 * np.pi))) - 1) % 365
    doy_hist = np.array([(doy_last - 2) % 365, (doy_last - 1) % 365, doy_last], dtype=np.int64)

    z_hist = ((_transform_phys(x) - _CLIM[doy_hist]) / _ANOM_STD[None, :, None, None]).astype(np.float32)

    all_rollouts = []
    for sess in _SESSIONS:
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
        all_rollouts.append(np.stack(preds, axis=0))

    stacked = np.stack(all_rollouts, axis=0).astype(np.float32)
    if _SESSION_WEIGHTS is not None and stacked.shape[0] == _SESSION_WEIGHTS.shape[0]:
        ens = np.sum(stacked * _SESSION_WEIGHTS[:, None, :, None, None], axis=0).astype(np.float32)
    else:
        ens = np.mean(stacked, axis=0).astype(np.float32)
    ens[:, 1] = np.maximum(ens[:, 1], 0.0)
    ens[:, 0] = np.clip(ens[:, 0], 233.15, 333.15)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
