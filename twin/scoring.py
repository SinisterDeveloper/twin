"""Offline Round-1 scoring metrics (aligned with monsoon_r1.scoring)."""
from __future__ import annotations

import numpy as np
from .contract import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    CHANNELS,
    LAT,
    NC,
    NLAT,
    NLON,
    box_mask,
    lat_weights,
)

SS_REF = 0.12
CSI_SS_REF = 0.10
RAIN_THRESH = 10.0
BLUR_MIN_RATIO = 0.40
BLUR_MAX_RATIO = 2.5
STD_DEFAULT = np.array([8.0, 8.0, 3.5, 2.5, 450.0, 20.0], "f4")
NULL_TOL = 1e-3
NONZERO_MIN = 1e-3
LOCAL_MIN_FRAC = 0.60
DILATE = 3
MAX_SPEED_CELLS_PER_DAY = 8.0
STABLE_MAX_STD = 5.0


def rmse_lat(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    w = lat_weights()[:, None]
    se = (pred - true) ** 2
    return np.sqrt((se * w).mean(axis=(-2, -1)).reshape(-1, NC).mean(axis=0))


def skill_vs_best_ref(pred: np.ndarray, true: np.ndarray, persist: np.ndarray, clim: np.ndarray) -> dict:
    r, rp, rc = rmse_lat(pred, true), rmse_lat(persist, true), rmse_lat(clim, true)
    best = np.minimum(rp, rc)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = np.where(best > 0, 1.0 - r / best, np.nan)
        sp = np.where(rp > 0, 1.0 - r / rp, np.nan)
        sc = np.where(rc > 0, 1.0 - r / rc, np.nan)
    return {"rmse": r, "ss": ss, "ss_persist": sp, "ss_clim": sc}


def csi(pred_tp: np.ndarray, true_tp: np.ndarray, thr: float = RAIN_THRESH) -> float:
    p, t = pred_tp > thr, true_tp > thr
    h = int((p & t).sum())
    m = int((~p & t).sum())
    f = int((p & ~t).sum())
    return float("nan") if h + m + f == 0 else h / (h + m + f)


def validity(out: np.ndarray, steps: int) -> list[str]:
    bad = []
    if out.shape != (steps, NC, NLAT, NLON):
        return [f"shape {out.shape}"]
    if not np.isfinite(out).all():
        bad.append("NaN/Inf")
    else:
        if out[:, 1].min() < -1e-6:
            bad.append("negative rain")
        t = out[:, 0]
        if t.min() < 233.15 or t.max() > 333.15:
            bad.append("t2m outside [-40, +60] C")
    return bad


def variance_ratio(pred_day: np.ndarray, true_day: np.ndarray) -> np.ndarray:
    vp = pred_day.reshape(NC, -1).var(axis=1)
    vt = true_day.reshape(NC, -1).var(axis=1)
    return np.where(vt > 0, vp / np.maximum(vt, 1e-30), np.nan)


def anti_blur_points(ratios: np.ndarray, max_pts: float = 10.0) -> float:
    ok_cnt = np.isfinite(ratios).sum(axis=0)
    r = np.where(
        ok_cnt > 0,
        np.nansum(np.where(np.isfinite(ratios), ratios, 0.0), axis=0) / np.maximum(ok_cnt, 1),
        np.nan,
    )
    ok = np.where(np.isfinite(r), np.clip(r / BLUR_MIN_RATIO, 0, 1) * (r <= BLUR_MAX_RATIO), 0.0)
    return float(max_pts * ok.mean())


def _energy(delta: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((delta / std[None, :, None, None]) ** 2).sum(axis=1)


def near_mask(box, dilate: int = DILATE) -> np.ndarray:
    m = box_mask(box)
    out = m.copy()
    for _ in range(dilate):
        o = out.copy()
        o[1:] |= out[:-1]
        o[:-1] |= out[1:]
        o[:, 1:] |= out[:, :-1]
        o[:, :-1] |= out[:, 1:]
        out = o
    return out


def causality_checks(base: np.ndarray, shock: np.ndarray, box, std: np.ndarray) -> dict:
    d = shock - base
    res = {"finite": bool(np.isfinite(d).all())}
    if not res["finite"]:
        return {**res, "nonzero": False, "localized": False, "propagates": False, "stable": False}
    z = np.abs(d / std[None, :, None, None])
    max_z = float(z.max())
    max_ch = CHANNELS[int(np.unravel_index(int(np.argmax(z)), z.shape)[1])]
    res["max_z"] = round(max_z, 4)
    res["max_z_channel"] = max_ch
    res["nonzero"] = bool(max_z >= NONZERO_MIN)
    res["stable"] = bool(max_z <= STABLE_MAX_STD)
    e = _energy(d, std)
    steps = e.shape[0]
    near = near_mask(box)
    tot1 = e[0].sum()
    res["near_frac_day1"] = float(e[0][near].sum() / tot1) if tot1 > 0 else 0.0
    res["localized"] = bool(tot1 > 0 and res["near_frac_day1"] >= LOCAL_MIN_FRAC)

    out_e = np.array([e[k][~near].sum() for k in range(steps)])
    late = out_e[3:8].mean() if steps >= 8 else out_e[-1]
    grows = bool(late > 1.5 * out_e[0] + 1e-12 and late > 0.02 * max(e.sum(axis=(1, 2)).max(), 1e-12))

    ii, jj = np.indices((NLAT, NLON))
    boxm = box_mask(box)
    bi, bj = np.where(boxm)
    dist = np.sqrt(((ii[..., None] - bi) ** 2 + (jj[..., None] - bj) ** 2)).min(axis=-1)
    tele = 0.0
    for k in range(steps):
        far = dist > (DILATE + MAX_SPEED_CELLS_PER_DAY * (k + 1))
        tele = max(tele, float(e[k][far].sum() / max(e[k].sum(), 1e-12)))
    res["far_frac_max"] = tele
    res["no_teleport"] = bool(tele < 0.05)
    res["propagates"] = bool(grows and res["no_teleport"])
    return res


def causality_points(shock_results: list[dict], null_ok: bool, max_pts: float = 15.0) -> float:
    per = max_pts / max(len(shock_results), 1)
    got = 0.0
    for r in shock_results:
        c = 0
        if r["nonzero"]:
            c += 1
            if r["localized"]:
                c += 1 + int(bool(r["propagates"])) + int(bool(r["stable"]))
        got += per * c / 4.0
    return float(got * (1.0 if null_ok else 0.5))


def null_consistent(a: np.ndarray, b: np.ndarray, std: np.ndarray) -> bool:
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.shape != b.shape:
        return False
    return bool((np.abs(a - b) / std[None, :, None, None]).max() <= NULL_TOL)


def make_shock(name: str, last_state: np.ndarray, std: np.ndarray | None = None) -> np.ndarray:
    _ = std
    d = np.zeros((NC, NLAT, NLON), "f4")
    if name == "warm_ocean":
        m = box_mask(BOX_BAY_OF_BENGAL)
        d[0][m] = 3.0
        d[5][m] = 0.15 * last_state[5][m]
    elif name == "monsoon_stall":
        m = box_mask(BOX_ARABIAN_SEA)
        d[2][m] = -0.4 * last_state[2][m]
        d[3][m] = -0.4 * last_state[3][m]
    elif name != "null":
        raise ValueError(name)
    return d


def csi_skill(csi_model: float, csi_persist: float, csi_clim: float) -> float:
    ref = np.nanmax([csi_persist, csi_clim]) if np.isfinite([csi_persist, csi_clim]).any() else float("nan")
    if not np.isfinite(csi_model) or not np.isfinite(ref) or ref >= 1:
        return float("nan")
    return float((csi_model - ref) / (1.0 - ref))


def forecast_points(ss_by_window: list[float], csi_ss_by_window: list[float]) -> dict:
    ss = np.array([x for x in ss_by_window if np.isfinite(x)])
    cs = np.array([x for x in csi_ss_by_window if np.isfinite(x)])
    ss_m = float(ss.mean()) if len(ss) else float("nan")
    cs_m = float(cs.mean()) if len(cs) else float("nan")
    rmse_pts = 30.0 * float(np.clip(ss_m / SS_REF, 0, 1)) if np.isfinite(ss_m) else 0.0
    csi_pts = 15.0 * float(np.clip(cs_m / CSI_SS_REF, 0, 1)) if np.isfinite(cs_m) else 0.0
    return {
        "mean_ss": ss_m,
        "mean_csi_skill": cs_m,
        "rmse_pts": rmse_pts,
        "csi_pts": csi_pts,
        "points": rmse_pts + csi_pts,
    }
