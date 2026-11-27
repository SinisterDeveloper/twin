"""Unit and FastAPI integration tests for USP 1 adjoint sensitivity & reverse shock solver."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np = pytest.importorskip("numpy")

from twin.contract import CHANNELS, LAT, LON, NC, NLAT, NLON, T_IN, doy_sincos  # noqa: E402
from twin.scoring import STD_DEFAULT  # noqa: E402
from twin.sensitivity import (  # noqa: E402
    DEFAULT_SHOCK_CAPS,
    build_target_mask,
    compute_reverse_shock,
    verify_sensitivity_physics,
)


def _linear_forward_fn(weights_6hw: np.ndarray, target_day: int, target_ch: int, baseline: float = 12.0):
    """Build a deterministic linear forward_fn whose exact gradient at target_day, target_ch is weights_6hw."""

    def _fn(init_states: np.ndarray, doy_sc: list[float], steps: int, perturbation: np.ndarray | None) -> np.ndarray:
        _ = (init_states, doy_sc)
        out = np.zeros((steps, NC, NLAT, NLON), dtype=np.float32)
        val = baseline
        if perturbation is not None:
            val += float(np.sum(weights_6hw * perturbation))
        out[target_day - 1, target_ch] = val
        return out

    return _fn


def test_no_top_level_torch_import():
    root = Path(__file__).resolve().parent.parent
    for rel in ("serve.py", "twin/sensitivity.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name != "torch" for alias in node.names), f"Top-level import torch in {rel}"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "torch", f"Top-level from torch import in {rel}"


def test_build_target_mask_point_and_box():
    for radius, expected_cells in ((0, 1), (1, 9), (2, 25)):
        w, (clat, clon) = build_target_mask({"type": "point", "lat": 10.5, "lon": 76.5, "radius_cells": radius})
        assert w.shape == (NLAT, NLON)
        assert w.dtype == np.float32
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert int((w > 0).sum()) == expected_cells
        assert clat == pytest.approx(10.5)
        assert clon == pytest.approx(76.5)

    w_box, (blat, blon) = build_target_mask({"type": "box", "lat0": 10.0, "lat1": 12.0, "lon0": 75.0, "lon1": 78.0})
    assert w_box.shape == (NLAT, NLON)
    assert np.isclose(w_box.sum(), 1.0, atol=1e-6)
    assert 10.0 <= blat <= 12.0
    assert 75.0 <= blon <= 78.0
    assert int((w_box > 0).sum()) == 6  # lats [10.5, 11.5] x lons [75.5, 76.5, 77.5]


def test_compute_reverse_shock_linear_mock():
    mask_w, _ = build_target_mask({"type": "point", "lat": 10.5, "lon": 76.5, "radius_cells": 0})
    i0 = int(np.argmin(np.abs(LAT - 10.5)))
    j0 = int(np.argmin(np.abs(LON - 76.5)))

    grad = np.zeros((NC, NLAT, NLON), dtype=np.float32)
    grad[0, i0, j0] = 0.5   # t2m
    grad[5, i0, j0] = 0.25  # tcwv

    init = np.zeros((T_IN, NC, NLAT, NLON), dtype=np.float32)
    sc = doy_sincos(180)
    fwd = _linear_forward_fn(grad, target_day=4, target_ch=1, baseline=10.0)

    res_inc = compute_reverse_shock(
        grad=grad,
        init_states=init,
        doy_sincos=sc,
        target_day=4,
        target_channel="tp",
        mask_w=mask_w,
        forward_fn=fwd,
        direction="increase",
    )
    assert res_inc["baseline_value"] == pytest.approx(10.0)
    assert res_inc["achieved_value"] > 10.0
    assert res_inc["delta"] > 0.0
    assert res_inc["target_met"] is True
    for idx, ch in enumerate(CHANNELS):
        assert float(np.max(np.abs(res_inc["perturbation"][idx]))) <= DEFAULT_SHOCK_CAPS[ch] + 1e-6

    res_dec = compute_reverse_shock(
        grad=grad,
        init_states=init,
        doy_sincos=sc,
        target_day=4,
        target_channel="tp",
        mask_w=mask_w,
        forward_fn=fwd,
        direction="decrease",
        allowed_channels=["t2m"],
    )
    assert res_dec["achieved_value"] < 10.0
    assert float(np.max(np.abs(res_dec["perturbation"][5]))) == 0.0

    # Explicit reachable threshold targeting
    res_thr = compute_reverse_shock(
        grad=grad,
        init_states=init,
        doy_sincos=sc,
        target_day=4,
        target_channel="tp",
        mask_w=mask_w,
        forward_fn=fwd,
        direction="increase",
        threshold=12.5,
    )
    assert res_thr["achieved_value"] == pytest.approx(12.5, abs=1e-3)
    assert res_thr["target_met"] is True

    # Unreachable threshold under caps reports target_met=False
    res_unreach = compute_reverse_shock(
        grad=grad,
        init_states=init,
        doy_sincos=sc,
        target_day=4,
        target_channel="tp",
        mask_w=mask_w,
        forward_fn=fwd,
        direction="increase",
        threshold=20.0,
    )
    assert res_unreach["target_met"] is False


def test_verify_sensitivity_physics_fd_and_cone():
    mask_w, (clat, clon) = build_target_mask({"type": "point", "lat": 15.5, "lon": 78.5, "radius_cells": 1})
    i0 = int(np.argmin(np.abs(LAT - clat)))
    j0 = int(np.argmin(np.abs(LON - clon)))

    ii, jj = np.indices((NLAT, NLON))
    blob = np.exp(-0.5 * (((ii - i0) ** 2 + (jj - j0) ** 2) / 3.0)).astype(np.float32)

    grad = np.zeros((NC, NLAT, NLON), dtype=np.float32)
    grad[5] = 0.05 * blob  # localized moisture precursor
    grad_std = grad * STD_DEFAULT[:, None, None]
    bmap = np.sqrt(np.sum(grad_std ** 2, axis=0)).astype(np.float32)

    init = np.zeros((T_IN, NC, NLAT, NLON), dtype=np.float32)
    sc = doy_sincos(180)
    fwd = _linear_forward_fn(grad, target_day=3, target_ch=1, baseline=5.0)

    chk = verify_sensitivity_physics(grad, bmap, init, sc, 3, "tp", mask_w, fwd)
    assert chk["fd_check"]["same_sign"] is True
    assert chk["fd_check"]["rel_error"] < 1e-4
    assert chk["fd_check"]["passed"] is True
    assert chk["cone_check"]["cone_speed_cells_per_day"] <= 8.0
    assert chk["cone_check"]["far_energy_frac"] < 0.05
    assert chk["cone_check"]["passed"] is True
    assert chk["physics_consistent"] is True

    # Teleportation / far-field leakage should fail cone_check
    bmap_bad = bmap.copy()
    bmap_bad[-1, -1] = 500.0
    chk_bad = verify_sensitivity_physics(grad, bmap_bad, init, sc, 3, "tp", mask_w, fwd)
    assert chk_bad["cone_check"]["far_energy_frac"] > 0.05
    assert chk_bad["cone_check"]["passed"] is False
    assert chk_bad["physics_consistent"] is False


def test_serve_sensitivity_and_existing_endpoints_unchanged(monkeypatch):
    from fastapi.testclient import TestClient
    import serve
    import twin.sensitivity as sens_mod

    client = TestClient(serve.app)

    # 1. /health and /simulate remain intact
    h = client.post("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ready"

    init = np.zeros((T_IN, NC, NLAT, NLON), dtype=np.float32)
    init[:, 0] = 300.0
    sc = doy_sincos(180)
    sim_res = client.post(
        "/simulate",
        json={"init_states": init.tolist(), "doy_sincos": sc, "steps": 3, "perturbation": None},
    )
    assert sim_res.status_code == 200
    assert np.asarray(sim_res.json()["rollout"]).shape == (3, NC, NLAT, NLON)

    # 2. /sensitivity with mocked bundle & adjoint sensitivity
    monkeypatch.setattr(serve, "_get_torch_bundle", lambda: {"mock": True})

    def _mock_adj(bundle, init_states, doy_sincos, target_channel, target_day, mask_w, models=None):
        _ = (bundle, init_states, doy_sincos, target_channel, target_day, models)
        grad = np.zeros((NC, NLAT, NLON), dtype=np.float32)
        grad[5] = mask_w * 0.2
        grad_std = (grad * STD_DEFAULT[:, None, None]).astype(np.float32)
        bmap = np.sqrt(np.sum(grad_std ** 2, axis=0)).astype(np.float32)
        return {
            "grad": grad,
            "grad_std": grad_std,
            "butterfly_map": bmap,
            "channel_importance": {ch: (1.0 if ch == "tcwv" else 0.0) for ch in CHANNELS},
            "clamped_zero_grad": False,
            "baseline_value": 12.0,
        }

    monkeypatch.setattr(sens_mod, "compute_adjoint_sensitivity", _mock_adj)
    monkeypatch.setattr(serve, "simulate", _linear_forward_fn(np.eye(1)[0] * 0 + (lambda g: g)(np.where(
        np.arange(NC)[:, None, None] == 5, 0.2, 0.0
    ).astype(np.float32) * build_target_mask({"type": "point", "lat": 10.5, "lon": 76.5, "radius_cells": 0})[0]), 4, 1, 12.0))

    resp = client.post(
        "/sensitivity",
        json={
            "init_states": init.tolist(),
            "doy_sincos": sc,
            "target_day": 4,
            "target_channel": "tp",
            "region": {"type": "point", "lat": 10.5, "lon": 76.5, "radius_cells": 0},
            "direction": "increase",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "target",
        "butterfly_map",
        "sensitivity_signed",
        "channel_importance",
        "reverse_shock",
        "self_check",
        "clamped_zero_grad",
        "compute_ms",
    ):
        assert key in data, f"Missing key {key} in /sensitivity response"
    assert data["target"]["channel"] == "tp"
    assert data["target"]["day"] == 4
    assert np.asarray(data["butterfly_map"]).shape == (NLAT, NLON)
    assert np.asarray(data["sensitivity_signed"]).shape == (NC, NLAT, NLON)
    assert data["self_check"]["physics_consistent"] is True


def test_adjoint_sensitivity_torch_end_to_end():
    pytest.importorskip("torch")
    from twin.rollout import load_ensemble_torch
    from twin.sensitivity import compute_adjoint_sensitivity

    art_dir = Path("artifacts")
    if not (art_dir / "pt").exists():
        pytest.skip("artifacts/pt not present")

    import serve

    bundle = load_ensemble_torch(art_dir, device="cpu")
    clim = np.load(art_dir / "clim.npy").astype(np.float32)
    init = clim[177:180].copy()
    init[:, 1] = np.expm1(np.maximum(init[:, 1], 0.0))
    sc = doy_sincos(180)
    mask_w, _ = build_target_mask({"type": "point", "lat": 10.5, "lon": 76.5, "radius_cells": 1})

    adj = compute_adjoint_sensitivity(bundle, init, sc, "tp", 3, mask_w)
    assert adj["butterfly_map"].shape == (NLAT, NLON)
    assert np.isclose(sum(adj["channel_importance"].values()), 1.0, atol=1e-5)

    chk = verify_sensitivity_physics(adj["grad"], adj["butterfly_map"], init, sc, 3, "tp", mask_w, serve.simulate)
    assert chk["fd_check"]["same_sign"] is True
    assert chk["cone_check"]["passed"] is True
