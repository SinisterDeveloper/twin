"""Export trained PyTorch checkpoints to ONNX and benchmark CPU rollout speed."""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .contract import NC, NLAT, NLON, SCORE_STEPS, T_IN
from .models import build_model


def export_checkpoint_to_onnx(
    ckpt_path: Path,
    static_path: Path,
    onnx_out: Path,
    opset: int = 17,
) -> Path:
    static_grid = np.load(static_path)
    art_dir = static_path.parent
    uv_p = art_dir / "clim_uv_coef.npy"
    clim_uv_coef = np.load(uv_p) if uv_p.exists() else None
    phys_p = art_dir / "clim_phys_coef.npy"
    clim_phys_coef = np.load(phys_p) if phys_p.exists() else None
    stats_p = art_dir / "norm_stats.json"
    anom_std = np.array(json.loads(stats_p.read_text())["anom_std"], dtype=np.float32) if stats_p.exists() else None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    model = build_model(
        model_name,
        static_grid=static_grid,
        clim_uv_coef=clim_uv_coef,
        clim_phys_coef=clim_phys_coef,
        anom_std=anom_std,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    z_dummy = torch.randn(1, T_IN, NC, NLAT, NLON, dtype=torch.float32)
    sc_dummy = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=["z_hist", "doy_sc"],
        output_names=["z_next"],
        opset_version=opset,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, (z_dummy, sc_dummy), str(onnx_out), **export_kwargs)

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
    sess = ort.InferenceSession(str(onnx_out), sess_options=opts, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(z_dummy, sc_dummy).numpy()
    ort_out = sess.run(None, {"z_hist": z_dummy.numpy(), "doy_sc": sc_dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))

    # 10-step CPU latency benchmark
    t0 = time.perf_counter()
    zh = z_dummy.numpy()
    sc = sc_dummy.numpy()
    for _ in range(SCORE_STEPS):
        zn = sess.run(None, {"z_hist": zh, "doy_sc": sc})[0]
        zh = np.concatenate([zh[:, 1:], zn[:, None]], axis=1)
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"Exported {onnx_out.name} ({model_name}) | max_diff={max_diff:.2e} | 10-step CPU={ms:.1f} ms")
    return onnx_out
