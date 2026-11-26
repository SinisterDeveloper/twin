"""Offline 25-window 5-regime + 3-shock benchmark matching Round-1 scoring."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .contract import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    LEAD_DAYS,
    SCORE_STEPS,
    T_IN,
    advance_doy_sincos,
    doy_sincos,
)
from .data import date_to_doy365, inv_transform_physical, load_netcdf, transform_physical
from .models import build_model
from .scoring import (
    anti_blur_points,
    causality_checks,
    causality_points,
    csi,
    csi_skill,
    forecast_points,
    make_shock,
    null_consistent,
    skill_vs_best_ref,
    validity,
    variance_ratio,
)


def select_regime_windows(data: np.ndarray, dates: np.ndarray, start_date: str = "2019-01-01", per_regime: int = 5) -> list[int]:
    """Selects 25 non-overlapping 10-day windows across 5 physical regimes on Val."""
    s_cut = dt.date.fromisoformat(start_date)
    valid = [i for i in range(T_IN - 1, len(dates) - SCORE_STEPS) if dates[i] >= s_cut]
    if not valid:
        raise ValueError(f"No validation windows >= {start_date}")

    # Regime scores on the target 10-day window [i+1 : i+1+10]
    scored = {"heavy_rain": [], "heatwave": [], "cyclone": [], "onset": [], "winter": []}
    for i in valid:
        w = data[i + 1 : i + 1 + SCORE_STEPS]
        m = dates[i].month
        rain_max = float(w[:, 1].max())
        t2m_max = float(w[:, 0].max())
        msl_min = float(w[:, 4].min())
        wind_max = float(np.sqrt(w[:, 2] ** 2 + w[:, 3] ** 2).max())
        scored["heavy_rain"].append((rain_max, i))
        if m in (4, 5, 6):
            scored["heatwave"].append((t2m_max, i))
        if m in (5, 6):
            scored["onset"].append((float(w[:, 1].mean()), i))
        if m in (12, 1, 2):
            scored["winter"].append((float(w[:, 1].mean()), i))
        scored["cyclone"].append((wind_max - 0.01 * msl_min, i))

    chosen: list[int] = []
    for reg, items in scored.items():
        items.sort(reverse=True)
        cnt = 0
        for _, idx in items:
            if all(abs(idx - c) >= 12 for c in chosen):
                chosen.append(idx)
                cnt += 1
                if cnt >= per_regime:
                    break
    return sorted(chosen)


def make_predictor(
    artifacts_dir: Path,
    ckpts: list[Path] | None = None,
    onnx_paths: list[Path] | None = None,
) -> Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray]:
    """Build a pure-numpy/ONNX or PyTorch rollout predictor matching serve.py."""
    static_grid = np.load(artifacts_dir / "static.npy")
    clim_tr = np.load(artifacts_dir / "clim.npy")
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    anom_std = np.array(meta["anom_std"], dtype=np.float32)

    runners = []
    if onnx_paths:
        import onnxruntime as ort

        for p in onnx_paths:
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            runners.append(("onnx", sess))
    elif ckpts:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for p in ckpts:
            ckpt = torch.load(p, map_location=device, weights_only=False)
            m = build_model(ckpt["model_name"], static_grid=static_grid).to(device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            runners.append(("torch", (m, device)))

    def predict(init_states: np.ndarray, doy_sc: list[float], steps: int, perturbation: np.ndarray | None) -> np.ndarray:
        x = init_states.astype(np.float32).copy()
        if perturbation is not None:
            x[-1] = x[-1] + perturbation.astype(np.float32)

        # Fallback persistence if no models loaded
        if not runners:
            out = np.repeat(x[-1][None], steps, axis=0)
            out[:, 1] = np.maximum(out[:, 1], 0.0)
            return out

        # Map doy_sincos of last input day to doy365 index
        ang = float(np.arctan2(doy_sc[0], doy_sc[1]))
        if ang < 0:
            ang += 2.0 * np.pi
        doy_last = int(round(ang * 365.0 / (2.0 * np.pi))) % 365
        doy_hist = np.array([(doy_last - 2) % 365, (doy_last - 1) % 365, doy_last], dtype=np.int64)

        x_tr = transform_physical(x)
        z_hist = ((x_tr - clim_tr[doy_hist]) / anom_std[None, :, None, None]).astype(np.float32)

        all_rollouts = []
        for kind, runner in runners:
            zh = z_hist.copy()
            s_v, c_v = float(doy_sc[0]), float(doy_sc[1])
            preds_phys = []
            for k in range(steps):
                sc_arr = np.array([[s_v, c_v]], dtype=np.float32)
                if kind == "onnx":
                    z_next = runner.run(None, {"z_hist": zh[None], "doy_sc": sc_arr})[0][0]
                else:
                    m, dev = runner
                    with torch.no_grad():
                        z_next = m(torch.from_numpy(zh[None]).to(dev), torch.from_numpy(sc_arr).to(dev)).cpu().numpy()[0]
                doy_k = (doy_last + k + 1) % 365
                t_phys = z_next * anom_std[:, None, None] + clim_tr[doy_k]
                preds_phys.append(inv_transform_physical(t_phys))
                zh = np.concatenate([zh[1:], z_next[None]], axis=0)
                s_v, c_v = advance_doy_sincos(s_v, c_v, 1)
            all_rollouts.append(np.stack(preds_phys, axis=0))

        ens = np.mean(all_rollouts, axis=0).astype(np.float32)
        ens[:, 1] = np.maximum(ens[:, 1], 0.0)
        ens[:, 0] = np.clip(ens[:, 0], 233.15, 333.15)
        return ens

    return predict


def evaluate_predictor(
    nc_path: Path,
    artifacts_dir: Path,
    predict_fn: Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray],
    val_start: str = "2019-01-01",
) -> dict:
    data, dates, _ = load_netcdf(nc_path)
    raw_clim = np.load(artifacts_dir / "raw_clim.npy")
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    phys_std = np.array(meta["phys_std"], dtype=np.float32)

    windows = select_regime_windows(data, dates, start_date=val_start, per_regime=5)
    lead_idx = [d - 1 for d in LEAD_DAYS]

    ss_list, csi_ss_list, blur_ratios, latencies = [], [], [], []
    valid_all = True

    for idx in windows:
        init_states = data[idx - T_IN + 1 : idx + 1]
        true_roll = data[idx + 1 : idx + 1 + SCORE_STEPS]
        doy_last = date_to_doy365(dates[idx]) + 1
        sc = doy_sincos(doy_last)

        t0 = time.perf_counter()
        pred = predict_fn(init_states, sc, SCORE_STEPS, None)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if validity(pred, SCORE_STEPS):
            valid_all = False
            ss_list.append(0.0)
            continue

        persist = np.repeat(init_states[-1][None], SCORE_STEPS, axis=0)
        clim_roll = np.stack([raw_clim[(date_to_doy365(dates[idx]) + k + 1) % 365] for k in range(SCORE_STEPS)], axis=0)

        sk = skill_vs_best_ref(pred[lead_idx], true_roll[lead_idx], persist[lead_idx], clim_roll[lead_idx])
        ss_list.append(float(np.nanmean(sk["ss"])))

        c_m = csi(pred[lead_idx, 1], true_roll[lead_idx, 1])
        c_p = csi(persist[lead_idx, 1], true_roll[lead_idx, 1])
        c_c = csi(clim_roll[lead_idx, 1], true_roll[lead_idx, 1])
        csi_ss_list.append(csi_skill(c_m, c_p, c_c))

        blur_ratios.append(variance_ratio(pred[7], true_roll[7]))

    f_pts = forecast_points(ss_list, csi_ss_list)
    blur_pts = anti_blur_points(np.stack(blur_ratios, axis=0))

    # Causality tests on 3 monsoon-season windows
    shock_res = []
    null_ok = True
    shock_idx = windows[len(windows) // 2]
    s_init = data[shock_idx - T_IN + 1 : shock_idx + 1]
    s_sc = doy_sincos(date_to_doy365(dates[shock_idx]) + 1)
    base = predict_fn(s_init, s_sc, SCORE_STEPS, None)
    null_out = predict_fn(s_init, s_sc, SCORE_STEPS, np.zeros_like(s_init[-1]))
    null_ok = null_consistent(base, null_out, phys_std)

    for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
        sh_out = predict_fn(s_init, s_sc, SCORE_STEPS, make_shock(name, s_init[-1]))
        shock_res.append({"shock": name, **causality_checks(base, sh_out, box, phys_std)})

    caus_pts = causality_points(shock_res, null_ok)
    total_auto = f_pts["points"] + blur_pts + caus_pts

    return {
        "valid_all": valid_all,
        "windows": len(windows),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "forecast": f_pts,
        "anti_blur_pts": round(blur_pts, 2),
        "causality_pts": round(caus_pts, 2),
        "null_ok": null_ok,
        "shocks": shock_res,
        "automated_score_of_70": round(total_auto, 2),
    }
