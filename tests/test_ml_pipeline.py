"""Zero-data pre-flight verification suite for Track 2 ML pipeline.

Run BEFORE starting Colab A100 training:
    pytest -q tests/test_ml_pipeline.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from twin.contract import (  # noqa: E402
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    NC,
    NLAT,
    NLON,
    SCORE_STEPS,
    T_IN,
    advance_doy_sincos,
    doy_sincos,
    sincos_to_doy_idx,
)
from twin import scoring as twin_scoring  # noqa: E402
from monsoon_r1 import scoring as r1_scoring  # noqa: E402


def test_doy_sincos_roundtrip():
    for d in range(1, 366):
        s, c = doy_sincos(d)
        assert sincos_to_doy_idx(s, c) == d - 1, f"doy={d} roundtrip failed"
    s, c = doy_sincos(1)
    for step in range(1, 15):
        s, c = advance_doy_sincos(s, c, 1)
        assert sincos_to_doy_idx(s, c) == step


def test_scoring_parity_with_monsoon_r1():
    rng = np.random.default_rng(7)
    pred = rng.normal(0, 2, size=(SCORE_STEPS, NC, NLAT, NLON)).astype(np.float32)
    true = rng.normal(0, 2, size=(SCORE_STEPS, NC, NLAT, NLON)).astype(np.float32)
    pred[:, 0] += 298.0
    true[:, 0] += 298.0
    pred[:, 1] = np.maximum(pred[:, 1] * 5.0 + 4.0, 0.0)
    true[:, 1] = np.maximum(true[:, 1] * 5.0 + 4.0, 0.0)
    pers = np.repeat(true[:1], SCORE_STEPS, axis=0)
    clim = np.zeros_like(true)
    clim[:, 0] = 298.0

    sk_t = twin_scoring.skill_vs_best_ref(pred, true, pers, clim)
    sk_r = r1_scoring.skill_vs_best_ref(pred, true, pers, clim)
    assert np.allclose(sk_t["ss"], sk_r["ss"], equal_nan=True, atol=1e-5)
    assert abs(twin_scoring.csi(pred[:, 1], true[:, 1]) - r1_scoring.csi(pred[:, 1], true[:, 1])) < 1e-6
    vr_t = twin_scoring.variance_ratio(pred[7], true[7])
    vr_r = r1_scoring.variance_ratio(pred[7], true[7])
    assert np.allclose(vr_t, vr_r, equal_nan=True, atol=1e-5)

    sh_t = twin_scoring.make_shock("warm_ocean", true[0])
    sh_r = r1_scoring.make_shock("warm_ocean", true[0])
    assert np.allclose(sh_t, sh_r, atol=1e-6)


def test_advect_resnet_physical_wind_direction():
    torch = pytest.importorskip("torch")
    from twin.models.advect_resnet import AdvectResNet

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    clim_uv_coef = np.zeros((9, 2, NLAT, NLON), dtype=np.float32)
    # Positive DC northeastward physical wind (+10 m/s eastward, +10 m/s northward)
    clim_uv_coef[0, 0] = 10.0
    clim_uv_coef[0, 1] = 10.0
    anom_std = np.array([2.5, 1.0, 3.5, 2.5, 300.0, 8.0], dtype=np.float32)

    m = AdvectResNet(static_grid=static, clim_uv_coef=clim_uv_coef, anom_std=anom_std).eval()
    yy, xx = torch.meshgrid(torch.arange(NLAT, dtype=torch.float32), torch.arange(NLON, dtype=torch.float32), indexing="ij")
    blob = torch.exp(-0.5 * (((yy - 16.0) ** 2 + (xx - 16.0) ** 2) / 4.0))

    z_last = torch.zeros(1, NC, NLAT, NLON, dtype=torch.float32)
    z_last[0, 0] = blob  # zero anomaly wind in ch 2,3 -> purely testing physical clim_uv advection
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        adv = m._advect(z_last, sc)[0, 0]

    def _com(field: torch.Tensor) -> tuple[float, float]:
        w = torch.clamp(field, min=0.0)
        tot = float(w.sum())
        return float((w * yy).sum() / tot), float((w * xx).sum() / tot)

    cy0, cx0 = _com(blob)
    cy1, cx1 = _com(adv)
    assert cy1 > cy0 + 0.3, f"Expected northward shift: {cy0:.2f} -> {cy1:.2f}"
    assert cx1 > cx0 + 0.3, f"Expected eastward shift: {cx0:.2f} -> {cx1:.2f}"


def test_all_models_day1_impulse_locality():
    torch = pytest.importorskip("torch")
    from twin.models import MODEL_NAMES, build_model

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    uv_coef = np.zeros((9, 2, NLAT, NLON), dtype=np.float32)
    uv_coef[0, 0] = 5.0
    uv_coef[0, 1] = 3.0
    anom_std = np.array([2.5, 1.0, 3.5, 2.5, 300.0, 8.0], dtype=np.float32)
    phys_std = twin_scoring.STD_DEFAULT.copy()

    torch.manual_seed(0)
    base_z = torch.randn(1, T_IN, NC, NLAT, NLON, dtype=torch.float32) * 0.3
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    for box, ch in ((BOX_BAY_OF_BENGAL, 0), (BOX_ARABIAN_SEA, 2)):
        mask = twin_scoring.near_mask(box, dilate=0)
        shock_z = base_z.clone()
        shock_z[0, -1, ch, torch.from_numpy(mask)] += 2.0

        for name in MODEL_NAMES:
            m = build_model(name, static_grid=static, clim_uv_coef=uv_coef, anom_std=anom_std).eval()
            # Perturb zero-init head slightly to exercise true non-trivial Day-1 receptive field
            with torch.no_grad():
                torch.nn.init.normal_(m.head.weight, std=0.02)
                b0 = m(base_z, sc)[0].numpy() * phys_std[:, None, None]
                s0 = m(shock_z, sc)[0].numpy() * phys_std[:, None, None]

            near = twin_scoring.near_mask(box, dilate=3)
            e = ((s0 - b0) / phys_std[:, None, None]) ** 2
            e_tot = float(e.sum())
            assert e_tot > 0.0, f"{name} produced zero shock response"
            near_frac = float(e[:, near].sum() / e_tot)
            assert near_frac >= 0.60, f"{name} failed Day-1 locality: near_frac={near_frac:.3f} < 0.60"


def test_all_models_onnx_parity_and_cpu_speed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from twin.export_onnx import export_checkpoint_to_onnx
    from twin.models import MODEL_NAMES, build_model

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    uv_coef = np.zeros((9, 2, NLAT, NLON), dtype=np.float32)
    anom_std = np.array([2.5, 1.0, 3.5, 2.5, 300.0, 8.0], dtype=np.float32)
    np.save(tmp_path / "static.npy", static)
    np.save(tmp_path / "clim_uv_coef.npy", uv_coef)
    (tmp_path / "norm_stats.json").write_text('{"anom_std": [2.5, 1.0, 3.5, 2.5, 300.0, 8.0]}')

    z_dummy = torch.randn(1, T_IN, NC, NLAT, NLON, dtype=torch.float32)
    sc_dummy = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    for name in MODEL_NAMES:
        m = build_model(name, static_grid=static, clim_uv_coef=uv_coef, anom_std=anom_std).eval()
        with torch.no_grad():
            torch.nn.init.normal_(m.head.weight, std=0.02)
            pt_out = m(z_dummy, sc_dummy).numpy()

        ckpt_p = tmp_path / f"{name}.pt"
        onnx_p = tmp_path / f"{name}.onnx"
        torch.save({"model": m.state_dict(), "model_name": name}, ckpt_p)
        export_checkpoint_to_onnx(ckpt_p, tmp_path / "static.npy", onnx_p)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(onnx_p), sess_options=opts, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"z_hist": z_dummy.numpy(), "doy_sc": sc_dummy.numpy()})[0]
        assert float(np.max(np.abs(pt_out - ort_out))) < 1e-4, f"{name} ONNX parity failed"

        t0 = time.perf_counter()
        zh = z_dummy.numpy()
        sc = sc_dummy.numpy()
        for _ in range(SCORE_STEPS):
            zn = sess.run(None, {"z_hist": zh, "doy_sc": sc})[0]
            zh = np.concatenate([zh[:, 1:], zn[:, None]], axis=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 500.0, f"{name} 10-step CPU too slow: {elapsed_ms:.1f} ms"


def test_anti_cheat_hygiene():
    root = Path(__file__).resolve().parent.parent
    banned = re.compile(
        "|".join(["open" + "-meteo", "openweather" + "map", "weather" + "bench2", "arco" + "-era5", "cds" + "api", "imd" + "lib"]),
        re.IGNORECASE,
    )
    hits = []
    for p in root.rglob("*.py"):
        if any(part in p.parts for part in (".git", "node_modules", ".venv", "venv")):
            continue
        if banned.search(p.read_text(errors="ignore")):
            hits.append(str(p))
    assert not hits, f"Banned weather API strings found in: {hits}"
    gitignore = (root / ".gitignore").read_text(errors="ignore")
    assert "*.nc" in gitignore
