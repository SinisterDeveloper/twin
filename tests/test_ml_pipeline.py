"""Zero-data pre-flight verification suite for Track 2 ML pipeline.

Run BEFORE starting Colab A100 training:
    pytest -q tests/test_ml_pipeline.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

    sh_t = twin_scoring.make_shock("warm_ocean", true[0], twin_scoring.STD_DEFAULT)
    sh_r = r1_scoring.make_shock("warm_ocean", true[0], r1_scoring.STD_DEFAULT)
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
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
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
        assert elapsed_ms < 1500.0, f"{name} 10-step CPU too slow: {elapsed_ms:.1f} ms"


def test_monsoon_loss_backward_no_inplace():
    torch = pytest.importorskip("torch")
    from twin.losses import MonsoonLoss

    clim_tr = torch.zeros(365, NC, NLAT, NLON, dtype=torch.float32)
    anom_std = torch.tensor([1.7, 0.7, 1.5, 1.4, 280.0, 6.0], dtype=torch.float32)
    crit = MonsoonLoss(clim_tr=clim_tr, anom_std=anom_std)

    for steps in (1, 6):
        z_pred = torch.randn(4, steps, NC, NLAT, NLON, dtype=torch.float32, requires_grad=True)
        z_true = torch.randn(4, steps, NC, NLAT, NLON, dtype=torch.float32)
        doy_tgt = torch.randint(0, 365, (4, steps))
        phys_true = torch.rand(4, steps, NC, NLAT, NLON, dtype=torch.float32) * 15.0
        loss, _ = crit(z_pred, z_true, doy_tgt, phys_true)
        loss.backward()
        assert z_pred.grad is not None and torch.isfinite(z_pred.grad).all()


def test_shock_stability_loss_backward():
    torch = pytest.importorskip("torch")
    from twin.losses import shock_stability_loss

    clim_tr = torch.zeros(365, NC, NLAT, NLON, dtype=torch.float32)
    anom_std = torch.tensor([1.7, 0.7, 1.5, 1.4, 280.0, 6.0], dtype=torch.float32)
    phys_std = torch.tensor([13.4, 7.5, 2.9, 2.4, 618.0, 20.4], dtype=torch.float32)
    z_base = torch.zeros(4, SCORE_STEPS, NC, NLAT, NLON, dtype=torch.float32)
    # Below-threshold perturbation gives exact zero penalty
    z_small = z_base + 0.1
    doy_tgt = torch.randint(0, 365, (4, SCORE_STEPS))
    loss_small = shock_stability_loss(z_base, z_small, doy_tgt, clim_tr, anom_std, phys_std)
    assert float(loss_small) == 0.0

    # Large perturbation over 10-step horizon yields positive loss and finite gradients
    z_pert = torch.randn(4, SCORE_STEPS, NC, NLAT, NLON, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        z_pert[:, -1, 1] += 5.5
    loss = shock_stability_loss(z_base, z_pert, doy_tgt, clim_tr, anom_std, phys_std)
    assert float(loss) > 0.0
    loss.backward()
    assert torch.isfinite(loss) and z_pert.grad is not None and torch.isfinite(z_pert.grad).all()


def test_models_soft_bounded_under_extreme_shock():
    torch = pytest.importorskip("torch")
    from twin.models import MODEL_NAMES, build_model

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    uv_coef = np.zeros((9, 2, NLAT, NLON), dtype=np.float32)
    anom_std = np.array([1.7, 0.7, 1.5, 1.4, 280.0, 6.1], dtype=np.float32)

    torch.manual_seed(3)
    z_hist = torch.randn(1, T_IN, NC, NLAT, NLON, dtype=torch.float32)
    z_hist[0, -1, 0, 8:16, 8:16] += 20.0
    z_hist[0, -1, 5, 8:16, 8:16] += 20.0
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    for name in MODEL_NAMES:
        m = build_model(name, static_grid=static, clim_uv_coef=uv_coef, anom_std=anom_std).eval()
        with torch.no_grad():
            torch.nn.init.normal_(m.head.weight, std=1.0)
            z_next = m(z_hist, sc)
        assert torch.isfinite(z_next).all(), f"{name} non-finite under extreme shock"
        max_delta = float(torch.abs(z_next - z_hist[:, -1]).max())
        assert max_delta <= 3.5 + 1e-5, f"{name} exceeded residual bound: {max_delta}"


def test_causality_checks_reports_max_z():
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 1.0, size=(SCORE_STEPS, NC, NLAT, NLON)).astype(np.float32)
    shock = base.copy()
    near = twin_scoring.near_mask(BOX_BAY_OF_BENGAL, dilate=0)
    shock[0, 1, near] += 30.0
    shock[5, 1, near] += 40.0
    std = twin_scoring.STD_DEFAULT.copy()

    res_t = twin_scoring.causality_checks(base, shock, BOX_BAY_OF_BENGAL, std)
    res_r = r1_scoring.causality_checks(base, shock, BOX_BAY_OF_BENGAL, std)
    assert res_t["max_z"] == pytest.approx(5.0, abs=1e-4)
    assert res_t["max_z_channel"] == "tp"
    for k in ("finite", "nonzero", "stable", "localized", "no_teleport", "propagates"):
        assert res_t[k] == res_r[k]


def test_compute_shock_stab_loss_runs_full_horizon():
    torch = pytest.importorskip("torch")
    from twin.models import build_model
    from twin.train import _compute_shock_stab_loss

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    anom_std = np.array([1.7, 0.7, 1.5, 1.4, 280.0, 6.1], dtype=np.float32)
    phys_std = twin_scoring.STD_DEFAULT.copy()
    clim_tr = torch.zeros(365, NC, NLAT, NLON, dtype=torch.float32)
    clim_tr[:, 5] = 45.0
    anom_std_t = torch.from_numpy(anom_std)
    phys_std_t = torch.from_numpy(phys_std)

    m = build_model("convnext_unet", static_grid=static, anom_std=anom_std).train()
    with torch.no_grad():
        torch.nn.init.normal_(m.head.weight, std=0.05)

    z_in = torch.randn(6, T_IN, NC, NLAT, NLON, dtype=torch.float32) * 0.5
    sc_seq = torch.tensor([[[0.0, 1.0]] * 4] * 6, dtype=torch.float32)
    doy_tgt = torch.full((6, 4), 180, dtype=torch.long)

    for b_idx in (0, 2):
        m.zero_grad()
        loss = _compute_shock_stab_loss(m, z_in, sc_seq, doy_tgt, clim_tr, anom_std_t, phys_std_t, b_idx, max_samples=2)
        assert torch.isfinite(loss)
        loss.backward()
        assert m.head.weight.grad is not None and torch.isfinite(m.head.weight.grad).all()


def test_strict_zero_far_field_leakage():
    torch = pytest.importorskip("torch")
    from twin.models import MODEL_NAMES, build_model

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    phys_coef = np.zeros((9, NC, NLAT, NLON), dtype=np.float32)
    phys_coef[0, 2] = 6.0  # physical eastward wind
    phys_coef[0, 3] = 4.0  # physical northward wind
    phys_coef[0, 5] = 40.0
    anom_std = np.array([1.7, 0.7, 1.5, 1.4, 280.0, 6.1], dtype=np.float32)

    torch.manual_seed(1)
    base_z = torch.randn(1, T_IN, NC, NLAT, NLON, dtype=torch.float32) * 0.4
    shock_z = base_z.clone()
    shock_z[0, -1, 0, 16, 18] += 4.0  # single-pixel point impulse at center
    sc = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    yy, xx = np.meshgrid(np.arange(NLAT), np.arange(NLON), indexing="ij")
    far_mask = (np.abs(yy - 16) >= 14) | (np.abs(xx - 18) >= 14)

    for name in MODEL_NAMES:
        m = build_model(name, static_grid=static, clim_phys_coef=phys_coef, anom_std=anom_std).eval()
        with torch.no_grad():
            torch.nn.init.normal_(m.head.weight, std=0.05)
            out_b = m(base_z, sc)[0].numpy()
            out_s = m(shock_z, sc)[0].numpy()
        far_diff = float(np.max(np.abs(out_s - out_b)[:, far_mask]))
        assert far_diff < 1e-6, f"{name} leaked globally to far field: max_diff={far_diff:.2e}"


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


def test_serve_ensemble_weights_skip_disabled(tmp_path, monkeypatch):
    import json as _json
    import serve

    class _CountingSess:
        def __init__(self, val: float):
            self.val = val
            self.calls = 0

        def run(self, *_args, **_kwargs):
            self.calls += 1
            return [np.full((1, NC, NLAT, NLON), self.val, dtype=np.float32)]

    s_keep = _CountingSess(290.0)
    s_zero = _CountingSess(330.0)
    monkeypatch.setattr(serve, "_SESSIONS", [s_keep, s_zero])
    monkeypatch.setattr(serve, "_SESSION_WEIGHTS", np.array([[1.0] * NC, [0.0] * NC], dtype=np.float32))
    monkeypatch.setattr(serve, "_CLIM", np.zeros((365, NC, NLAT, NLON), dtype=np.float32))
    monkeypatch.setattr(serve, "_ANOM_STD", np.ones(NC, dtype=np.float32))

    init = np.zeros((T_IN, NC, NLAT, NLON), dtype=np.float32)
    init[:, 0] = 300.0
    out = serve.simulate(init, [0.0, 1.0], 2, None)
    assert s_keep.calls == 2
    assert s_zero.calls == 0
    assert np.allclose(out[:, 0], 290.0)

    w_file = tmp_path / "ensemble_weights.json"
    w_file.write_text(_json.dumps({"convnext_unet": [0.7] * NC, "local_swin": [0.0] * NC}))
    from twin.evaluate import _load_ensemble_weights

    w = _load_ensemble_weights(w_file, ["convnext_unet", "local_swin"])
    assert np.allclose(w[0], 1.0)
    assert np.allclose(w[1], 0.0)


def test_cli_eval_live_parser():
    from pathlib import Path
    from twin.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0

    from scripts.live_client import main as live_main

    with pytest.raises(SystemExit) as exc2:
        live_main(["eval-live", "--help"])
    assert exc2.value.code == 0

    # Ensure twin/ stays strictly training/offline-eval with zero HTTP endpoint strings
    endpoint = "/" + "simulate"
    for p in Path("twin").rglob("*.py"):
        assert endpoint not in p.read_text(encoding="utf-8"), f"{endpoint} found in {p}"


def test_monsoon_loss_asymmetric_tp_variance():
    torch = pytest.importorskip("torch")
    from twin.losses import MonsoonLoss

    clim_tr = torch.zeros(365, NC, NLAT, NLON, dtype=torch.float32)
    anom_std = torch.tensor([1.7, 0.7, 1.5, 1.4, 280.0, 6.0], dtype=torch.float32)
    crit = MonsoonLoss(clim_tr=clim_tr, anom_std=anom_std, w_rain=0.0, w_csi=0.0, w_grad=0.0, w_var=1.0)

    # Construct a target field with spatial variance mostly in tp, keeping other channels exact
    z_true = torch.zeros(2, 6, NC, NLAT, NLON, dtype=torch.float32)
    pat = torch.linspace(-1.0, 1.0, NLAT * NLON, dtype=torch.float32).view(1, 1, NLAT, NLON)
    z_true[:, :, 1] = pat
    doy_tgt = torch.zeros((2, 6), dtype=torch.long)
    phys_true = z_true * anom_std.view(1, 1, NC, 1, 1)
    phys_true[:, :, 1] = torch.expm1(torch.clamp(phys_true[:, :, 1], min=0.0, max=5.8))

    # Isolate loss_var magnitude comparison by matching tp mean while scaling variance down vs up
    z_under = z_true.clone()
    z_under[:, :, 1] = 0.5 * pat
    z_over = z_true.clone()
    z_over[:, :, 1] = 1.5 * pat
    _, s_under = crit(z_under, z_true, doy_tgt, phys_true)
    _, s_over = crit(z_over, z_true, doy_tgt, phys_true)
    assert s_under["var"] > s_over["var"] > 0.0


def test_shock_stability_loss_propagation_hinge():
    torch = pytest.importorskip("torch")
    from twin.losses import shock_stability_loss

    clim_tr = torch.zeros(365, NC, NLAT, NLON, dtype=torch.float32)
    anom_std = torch.tensor([1.7, 0.7, 1.5, 1.4, 280.0, 6.0], dtype=torch.float32)
    phys_std = torch.tensor([13.4, 7.5, 2.9, 2.4, 618.0, 20.4], dtype=torch.float32)
    z_base = torch.zeros(2, SCORE_STEPS, NC, NLAT, NLON, dtype=torch.float32)
    doy_tgt = torch.zeros((2, SCORE_STEPS), dtype=torch.long)

    near = torch.zeros((NLAT, NLON), dtype=torch.bool)
    near[8:16, 8:16] = True

    # Trapped inside near_mask (never propagates outward in steps 3..7)
    z_trapped = z_base.clone().requires_grad_(True)
    with torch.no_grad():
        z_trapped[:, 0, 2, 10:14, 10:14] = 1.0
    loss_trapped = shock_stability_loss(z_base, z_trapped, doy_tgt, clim_tr, anom_std, phys_std, near_mask=near)
    loss_trapped.backward()
    assert torch.isfinite(loss_trapped) and torch.isfinite(z_trapped.grad).all()

    # Properly localized day-1 + outward wave propagation in steps 3..7
    z_prop = z_base.clone()
    z_prop[:, 0, 2, 10:14, 10:14] = 1.0
    z_prop[:, 3:8, 2, 20:26, 20:26] = 0.8
    loss_prop = shock_stability_loss(z_base, z_prop, doy_tgt, clim_tr, anom_std, phys_std, near_mask=near)
    assert float(loss_trapped) > float(loss_prop)
