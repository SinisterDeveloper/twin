"""Diagnostic script for adjoint sensitivity smoothness, latency, and FD/cone checks."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.contract import doy_sincos  # noqa: E402
from twin.sensitivity import (  # noqa: E402
    build_target_mask,
    compute_adjoint_sensitivity,
    compute_reverse_shock,
    verify_sensitivity_physics,
)


def _laplacian_smoothness(field: np.ndarray) -> float:
    """Return ratio of discrete Laplacian RMS to field RMS (lower = smoother)."""
    rms = float(np.sqrt(np.mean(field ** 2)))
    if rms < 1e-12:
        return 0.0
    lap = (
        field[2:, 1:-1]
        + field[:-2, 1:-1]
        + field[1:-1, 2:]
        + field[1:-1, :-2]
        - 4.0 * field[1:-1, 1:-1]
    )
    return float(np.sqrt(np.mean(lap ** 2)) / rms)


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not installed; skipping adjoint smoothness diagnostic.")
        return 0

    import serve
    from twin.rollout import load_ensemble_torch

    art_dir = Path("artifacts")
    bundle = load_ensemble_torch(art_dir, device="cpu")
    clim = np.load(art_dir / "clim.npy").astype(np.float32)
    init = clim[177:180].copy()
    init[:, 1] = np.expm1(np.maximum(init[:, 1], 0.0))
    sc = doy_sincos(180)

    cases = [
        ("Kerala Rain Day+6", {"type": "point", "lat": 10.5, "lon": 76.5, "radius_cells": 1}, "tp", 6),
        ("Delhi Heat Day+8", {"type": "point", "lat": 28.5, "lon": 77.5, "radius_cells": 1}, "t2m", 8),
    ]

    for label, region, ch, day in cases:
        mask_w, (clat, clon) = build_target_mask(region)
        t0 = time.perf_counter()
        adj = compute_adjoint_sensitivity(bundle, init, sc, ch, day, mask_w)
        adj_ms = (time.perf_counter() - t0) * 1000.0

        rev = compute_reverse_shock(adj["grad"], init, sc, day, ch, mask_w, serve.simulate)
        chk = verify_sensitivity_physics(adj["grad"], adj["butterfly_map"], init, sc, day, ch, mask_w, serve.simulate)
        smooth = _laplacian_smoothness(adj["butterfly_map"])

        print(f"=== {label} ({clat:.1f}N, {clon:.1f}E, ch={ch}, day={day}) ===")
        print(f"  adjoint_ms={adj_ms:.1f}  clamped_zero_grad={adj['clamped_zero_grad']}  lap_ratio={smooth:.3f}")
        print(f"  channel_importance={ {k: round(v, 3) for k, v in adj['channel_importance'].items()} }")
        print(f"  reverse_shock: base={rev['baseline_value']:.4f} -> achieved={rev['achieved_value']:.4f} (delta={rev['delta']:+.4f})")
        print(f"  fd_check={chk['fd_check']}")
        print(f"  cone_check={chk['cone_check']}")
        print(f"  physics_consistent={chk['physics_consistent']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
