"""Offline 25-window 5-regime + multi-season shock benchmark matching Round-1 scoring."""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .contract import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    CHANNELS,
    LEAD_DAYS,
    NC,
    SCORE_STEPS,
    T_IN,
    advance_doy_sincos,
    doy_sincos,
    sincos_to_doy_idx,
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


def select_regime_windows(
    data: np.ndarray,
    dates: np.ndarray,
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    per_regime: int = 5,
) -> list[int]:
    """Selects non-overlapping 10-day windows across 5 physical regimes in [start_date, end_date]."""
    s_cut = dt.date.fromisoformat(start_date)
    e_cut = dt.date.fromisoformat(end_date) if end_date else dt.date.max
    valid = [i for i in range(T_IN - 1, len(dates) - SCORE_STEPS) if s_cut <= dates[i] <= e_cut and dates[i + SCORE_STEPS] <= e_cut]
    if not valid:
        raise ValueError(f"No validation windows in [{start_date}, {e_cut}]")

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
    for _, items in scored.items():
        items.sort(reverse=True)
        cnt = 0
        for _, idx in items:
            if all(abs(idx - c) >= 12 for c in chosen):
                chosen.append(idx)
                cnt += 1
                if cnt >= per_regime:
                    break
    return sorted(chosen)


def _load_ensemble_weights(weights_json: Path | None, names: list[str]) -> np.ndarray | None:
    if weights_json is None or not weights_json.exists() or not names:
        return None
    raw = json.loads(weights_json.read_text())
    w = np.zeros((len(names), NC), dtype=np.float32)
    matched = False
    for i, name in enumerate(names):
        if name in raw:
            matched = True
            v = raw[name]
            w[i] = np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple)) else float(v)
    if not matched:
        return None
    w = np.maximum(w, 0.0)
    col_sum = w.sum(axis=0, keepdims=True)
    w = np.where(col_sum > 1e-8, w / np.maximum(col_sum, 1e-8), 1.0 / len(names))
    return w.astype(np.float32)


def _rollout_with_runners(
    runners: list[tuple[str, object]],
    clim_tr: np.ndarray,
    anom_std: np.ndarray,
    init_states: np.ndarray,
    doy_sc: list[float],
    steps: int,
    perturbation: np.ndarray | None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    x = init_states.astype(np.float32).copy()
    if perturbation is not None:
        x[-1] = x[-1] + perturbation.astype(np.float32)

    if not runners:
        out = np.repeat(x[-1][None], steps, axis=0)
        out[:, 1] = np.maximum(out[:, 1], 0.0)
        out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
        return out

    doy_last = sincos_to_doy_idx(float(doy_sc[0]), float(doy_sc[1]))
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

    stacked = np.stack(all_rollouts, axis=0).astype(np.float32)  # [M, S, 6, 32, 36]
    if weights is not None and stacked.shape[0] == weights.shape[0]:
        ens = np.sum(stacked * weights[:, None, :, None, None], axis=0).astype(np.float32)
    else:
        ens = np.mean(stacked, axis=0).astype(np.float32)
    ens[:, 1] = np.maximum(ens[:, 1], 0.0)
    ens[:, 0] = np.clip(ens[:, 0], 233.15, 333.15)
    return ens


def make_predictor_from_model(
    artifacts_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray]:
    clim_tr = np.load(artifacts_dir / "clim.npy")
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    anom_std = np.array(meta["anom_std"], dtype=np.float32)
    runners = [("torch", (model, device))]

    def predict(init_states: np.ndarray, doy_sc: list[float], steps: int, perturbation: np.ndarray | None) -> np.ndarray:
        return _rollout_with_runners(runners, clim_tr, anom_std, init_states, doy_sc, steps, perturbation)

    return predict


def make_predictor(
    artifacts_dir: Path,
    ckpts: list[Path] | None = None,
    onnx_paths: list[Path] | None = None,
    weights_json: Path | None = None,
) -> Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray]:
    """Build a pure-numpy/ONNX or PyTorch rollout predictor matching serve.py."""
    static_grid = np.load(artifacts_dir / "static.npy")
    clim_tr = np.load(artifacts_dir / "clim.npy")
    uv_p = artifacts_dir / "clim_uv_coef.npy"
    clim_uv_coef = np.load(uv_p) if uv_p.exists() else None
    phys_p = artifacts_dir / "clim_phys_coef.npy"
    clim_phys_coef = np.load(phys_p) if phys_p.exists() else None
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    anom_std = np.array(meta["anom_std"], dtype=np.float32)

    runners = []
    names = []
    if onnx_paths:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
        for p in onnx_paths:
            sess = ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
            runners.append(("onnx", sess))
            names.append(p.stem)
    elif ckpts:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for p in ckpts:
            ckpt = torch.load(p, map_location=device, weights_only=False)
            m = build_model(
                ckpt["model_name"],
                static_grid=static_grid,
                clim_uv_coef=clim_uv_coef,
                clim_phys_coef=clim_phys_coef,
                anom_std=anom_std,
            ).to(device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            runners.append(("torch", (m, device)))
            names.append(ckpt["model_name"] if p.stem.endswith("_best") else p.stem)

    weights = _load_ensemble_weights(weights_json, names)

    def predict(init_states: np.ndarray, doy_sc: list[float], steps: int, perturbation: np.ndarray | None) -> np.ndarray:
        return _rollout_with_runners(runners, clim_tr, anom_std, init_states, doy_sc, steps, perturbation, weights=weights)

    return predict


def evaluate_predictor(
    nc_path: Path,
    artifacts_dir: Path,
    predict_fn: Callable[[np.ndarray, list[float], int, np.ndarray | None], np.ndarray],
    val_start: str = "2019-01-01",
    val_end: str | None = None,
    n_causality_windows: int = 10,
) -> dict:
    data, dates, _ = load_netcdf(nc_path)
    raw_clim = np.load(artifacts_dir / "raw_clim.npy")
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    phys_std = np.array(meta["phys_std"], dtype=np.float32)

    windows = select_regime_windows(data, dates, start_date=val_start, end_date=val_end, per_regime=5)
    lead_idx = [d - 1 for d in LEAD_DAYS]

    ss_list, csi_ss_list, ch_ss_rows, blur_ratios, latencies = [], [], [], [], []
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
        ch_ss_rows.append(sk["ss"])
        ss_list.append(float(np.nanmean(sk["ss"])))

        c_m = csi(pred[lead_idx, 1], true_roll[lead_idx, 1])
        c_p = csi(persist[lead_idx, 1], true_roll[lead_idx, 1])
        c_c = csi(clim_roll[lead_idx, 1], true_roll[lead_idx, 1])
        csi_ss_list.append(csi_skill(c_m, c_p, c_c))

        blur_ratios.append(variance_ratio(pred[7], true_roll[7]))

    f_pts = forecast_points(ss_list, csi_ss_list)
    blur_mat = np.stack(blur_ratios, axis=0) if blur_ratios else np.zeros((1, NC), dtype=np.float32)
    blur_pts = anti_blur_points(blur_mat)

    ch_ss_mean = np.nanmean(np.stack(ch_ss_rows, axis=0), axis=0) if ch_ss_rows else np.zeros(NC)
    ch_blur_mean = np.nanmean(blur_mat, axis=0)

    # Multi-season causality tests across up to n_causality_windows windows
    sel_pos = np.linspace(0, len(windows) - 1, min(len(windows), n_causality_windows), dtype=int)
    shock_res = []
    null_ok = True
    for p_idx in sel_pos:
        shock_idx = windows[int(p_idx)]
        s_init = data[shock_idx - T_IN + 1 : shock_idx + 1]
        s_sc = doy_sincos(date_to_doy365(dates[shock_idx]) + 1)
        base = predict_fn(s_init, s_sc, SCORE_STEPS, None)
        null_out = predict_fn(s_init, s_sc, SCORE_STEPS, np.zeros_like(s_init[-1]))
        null_ok = null_ok and null_consistent(base, null_out, phys_std)

        for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
            sh_out = predict_fn(s_init, s_sc, SCORE_STEPS, make_shock(name, s_init[-1]))
            shock_res.append({"window_date": dates[shock_idx].isoformat(), "shock": name, **causality_checks(base, sh_out, box, phys_std)})

    caus_pts = causality_points(shock_res, null_ok)
    total_auto = f_pts["points"] + blur_pts + caus_pts

    monsoon_res = [r for r in shock_res if int(r["window_date"][5:7]) in (5, 6, 7, 8, 9)]
    monsoon_caus_pts = causality_points(monsoon_res, null_ok) if monsoon_res else caus_pts
    monsoon_shocks_pass = bool(
        null_ok and all(r["nonzero"] and r["localized"] and r["propagates"] and r["stable"] for r in (monsoon_res or shock_res))
    )

    worst_near = float(min((r.get("near_frac_day1", 0.0) for r in shock_res), default=0.0))
    worst_far = float(max((r.get("far_frac_max", 1.0) for r in shock_res), default=1.0))
    worst_max_z = float(max((r.get("max_z", 0.0) for r in shock_res), default=0.0))
    all_shocks_pass = bool(null_ok and all(r["nonzero"] and r["localized"] and r["propagates"] and r["stable"] for r in shock_res))

    return {
        "valid_all": valid_all,
        "windows": len(windows),
        "causality_windows": int(len(sel_pos)),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "forecast": f_pts,
        "channel_ss": {CHANNELS[i]: round(float(ch_ss_mean[i]), 4) for i in range(NC)},
        "channel_blur_ratio": {CHANNELS[i]: round(float(ch_blur_mean[i]), 4) for i in range(NC)},
        "anti_blur_pts": round(blur_pts, 2),
        "causality_pts": round(caus_pts, 2),
        "monsoon_causality_pts": round(monsoon_caus_pts, 2),
        "monsoon_shocks_pass": monsoon_shocks_pass,
        "null_ok": null_ok,
        "all_shocks_pass": all_shocks_pass,
        "worst_near_frac_day1": round(worst_near, 4),
        "worst_far_frac_max": round(worst_far, 4),
        "worst_max_z": round(worst_max_z, 4),
        "shocks": shock_res,
        "automated_score_of_70": round(total_auto, 2),
    }


def fit_ensemble_weights(
    nc_path: Path,
    artifacts_dir: Path,
    ckpts: list[Path],
    val_start: str = "2019-01-01",
    val_end: str | None = None,
    out_path: Path | None = None,
    n_samples: int = 400,
    seed: int = 42,
) -> dict:
    """Search simplex per-model x per-channel weights strictly on Val maximizing Round-1 score."""
    data, dates, _ = load_netcdf(nc_path)
    raw_clim = np.load(artifacts_dir / "raw_clim.npy")
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    phys_std = np.array(meta["phys_std"], dtype=np.float32)

    windows = select_regime_windows(data, dates, start_date=val_start, end_date=val_end, per_regime=5)
    lead_idx = [d - 1 for d in LEAD_DAYS]
    sel_pos = np.linspace(0, len(windows) - 1, min(len(windows), 6), dtype=int)

    model_preds = []
    model_shocks = []
    model_names = []
    for p in ckpts:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model_names.append(ckpt["model_name"] if p.stem.endswith("_best") else p.stem)
        pred_fn = make_predictor(artifacts_dir, ckpts=[p])
        w_preds = [
            pred_fn(data[idx - T_IN + 1 : idx + 1], doy_sincos(date_to_doy365(dates[idx]) + 1), SCORE_STEPS, None)
            for idx in windows
        ]
        s_preds = []
        for p_idx in sel_pos:
            idx = windows[int(p_idx)]
            init = data[idx - T_IN + 1 : idx + 1]
            sc = doy_sincos(date_to_doy365(dates[idx]) + 1)
            s_preds.append(
                {
                    "base": pred_fn(init, sc, SCORE_STEPS, None),
                    "null": pred_fn(init, sc, SCORE_STEPS, np.zeros_like(init[-1])),
                    "warm_ocean": pred_fn(init, sc, SCORE_STEPS, make_shock("warm_ocean", init[-1])),
                    "monsoon_stall": pred_fn(init, sc, SCORE_STEPS, make_shock("monsoon_stall", init[-1])),
                }
            )
        model_preds.append(np.stack(w_preds, axis=0))  # [W, 10, 6, 32, 36]
        model_shocks.append(s_preds)

    # [M, W, 10, 6, 32, 36]
    stack_w = np.stack(model_preds, axis=0)
    m_cnt = stack_w.shape[0]
    trues = np.stack([data[idx + 1 : idx + 1 + SCORE_STEPS] for idx in windows], axis=0)
    persists = np.stack([np.repeat(data[idx][None], SCORE_STEPS, axis=0) for idx in windows], axis=0)
    clims = np.stack(
        [np.stack([raw_clim[(date_to_doy365(dates[idx]) + k + 1) % 365] for k in range(SCORE_STEPS)], axis=0) for idx in windows],
        axis=0,
    )

    def _score_weights(w: np.ndarray) -> tuple[float, float, float, float]:
        ens_w = np.sum(stack_w * w[:, None, None, :, None, None], axis=0)
        ens_w[:, :, 1] = np.maximum(ens_w[:, :, 1], 0.0)
        ens_w[:, :, 0] = np.clip(ens_w[:, :, 0], 233.15, 333.15)

        ss_l, csi_l, blur_l = [], [], []
        for wi in range(len(windows)):
            sk = skill_vs_best_ref(ens_w[wi, lead_idx], trues[wi, lead_idx], persists[wi, lead_idx], clims[wi, lead_idx])
            ss_l.append(float(np.nanmean(sk["ss"])))
            csi_l.append(
                csi_skill(
                    csi(ens_w[wi, lead_idx, 1], trues[wi, lead_idx, 1]),
                    csi(persists[wi, lead_idx, 1], trues[wi, lead_idx, 1]),
                    csi(clims[wi, lead_idx, 1], trues[wi, lead_idx, 1]),
                )
            )
            blur_l.append(variance_ratio(ens_w[wi, 7], trues[wi, 7]))
        f_pts = forecast_points(ss_l, csi_l)
        b_pts = anti_blur_points(np.stack(blur_l, axis=0))

        shock_res = []
        null_ok = True
        for si in range(len(sel_pos)):
            b = np.sum(np.stack([model_shocks[m][si]["base"] for m in range(m_cnt)], axis=0) * w[:, None, :, None, None], axis=0)
            nl = np.sum(np.stack([model_shocks[m][si]["null"] for m in range(m_cnt)], axis=0) * w[:, None, :, None, None], axis=0)
            null_ok = null_ok and null_consistent(b, nl, phys_std)
            for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
                sh = np.sum(np.stack([model_shocks[m][si][name] for m in range(m_cnt)], axis=0) * w[:, None, :, None, None], axis=0)
                shock_res.append(causality_checks(b, sh, box, phys_std))
        c_pts = causality_points(shock_res, null_ok)
        tot = f_pts["points"] + b_pts + c_pts
        return tot, f_pts["points"], b_pts, c_pts

    rng = np.random.default_rng(seed)
    best_w = np.full((m_cnt, NC), 1.0 / m_cnt, dtype=np.float32)
    best_metrics = _score_weights(best_w)

    # Global model-level Dirichlet search + per-channel coordinate perturbation
    candidates = [np.eye(m_cnt, dtype=np.float32)[i][:, None].repeat(NC, axis=1) for i in range(m_cnt)]
    for _ in range(n_samples):
        g = rng.dirichlet(np.ones(m_cnt)).astype(np.float32)[:, None].repeat(NC, axis=1)
        candidates.append(g)
        c = np.stack([rng.dirichlet(np.ones(m_cnt) * 2.0) for _ in range(NC)], axis=1).astype(np.float32)
        candidates.append(c)

    for w_cand in candidates:
        m_val = _score_weights(w_cand)
        # Prefer full anti-blur + causality before raw points
        feasible = m_val[2] >= 9.95 and m_val[3] >= 14.95
        best_feasible = best_metrics[2] >= 9.95 and best_metrics[3] >= 14.95
        if (feasible and not best_feasible) or ((feasible == best_feasible) and m_val[0] > best_metrics[0]):
            best_w = w_cand
            best_metrics = m_val

    mapping = {model_names[i]: [round(float(x), 4) for x in best_w[i]] for i in range(m_cnt)}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(mapping, indent=2))
    return {
        "weights": mapping,
        "automated_score_of_70": round(best_metrics[0], 2),
        "forecast_pts": round(best_metrics[1], 2),
        "anti_blur_pts": round(best_metrics[2], 2),
        "causality_pts": round(best_metrics[3], 2),
    }
