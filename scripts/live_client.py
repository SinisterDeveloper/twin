"""Standalone HTTP client CLI for testing a live Monsoon Digital Twin server."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._http_sim import decode_arr, post_health, post_simulate, wire_encode  # noqa: E402
from twin.contract import CHANNELS, LAT, LON, NC, NLAT, NLON, SCORE_STEPS, T_IN, doy_sincos  # noqa: E402


def _base_init_states(doy: int, artifacts_dir: Path = Path("artifacts")) -> np.ndarray:
    clim_p = artifacts_dir / "clim.npy"
    if clim_p.exists():
        clim = np.load(clim_p).astype(np.float32)
        idx = [(doy - 2) % 365, (doy - 1) % 365, doy % 365]
        x = clim[idx].copy()
        x[:, 1] = np.expm1(np.maximum(x[:, 1], 0.0))
        return x
    x = np.zeros((T_IN, NC, NLAT, NLON), dtype=np.float32)
    defaults = [298.0, 2.5, 3.0, 2.0, 101000.0, 40.0]
    for c, v in enumerate(defaults):
        x[:, c] = v
    return x


def _build_perturbation(args, last_state: np.ndarray) -> np.ndarray | None:
    if args.shock in ("none", "null"):
        return np.zeros((NC, NLAT, NLON), dtype=np.float32) if args.shock == "null" else None
    if args.shock in ("warm_ocean", "monsoon_stall"):
        from monsoon_r1.scoring import STD_DEFAULT, make_shock

        return make_shock(args.shock, last_state, STD_DEFAULT).astype(np.float32)
    ch = CHANNELS.index(args.shock_channel) if args.shock_channel in CHANNELS else int(args.shock_channel)
    lat_i = np.clip(args.shock_lat - float(LAT[0]), 0.0, float(NLAT - 1))
    lon_i = np.clip(args.shock_lon - float(LON[0]), 0.0, float(NLON - 1))
    yy, xx = np.meshgrid(np.arange(NLAT, dtype=np.float32), np.arange(NLON, dtype=np.float32), indexing="ij")
    r2 = ((yy - lat_i) ** 2 + (xx - lon_i) ** 2) / max(args.shock_radius ** 2, 1e-3)
    pert = np.zeros((NC, NLAT, NLON), dtype=np.float32)
    pert[ch] = float(args.shock_intensity) * np.exp(-0.5 * r2)
    return pert


def _run_simulate_cli(args) -> int:
    url = args.url.rstrip("/")
    init_states = _base_init_states(args.doy, args.artifacts_dir)
    sc = doy_sincos(int(args.doy))
    pert = _build_perturbation(args, init_states[-1])
    payload = {
        "init_states": wire_encode(init_states, args.wire),
        "doy_sincos": sc,
        "steps": args.days,
        "perturbation": wire_encode(pert, args.wire) if pert is not None else None,
    }
    resp, ms = post_simulate(url, payload)
    rollout = decode_arr(resp["rollout"] if "rollout" in resp else resp["states"])

    delta = None
    if args.compare_null:
        base_payload = {
            "init_states": wire_encode(init_states, args.wire),
            "doy_sincos": sc,
            "steps": args.days,
            "perturbation": wire_encode(np.zeros((NC, NLAT, NLON), dtype=np.float32), args.wire),
        }
        base_resp, _ = post_simulate(url, base_payload)
        base_rollout = decode_arr(base_resp["rollout"] if "rollout" in base_resp else base_resp["states"])
        delta = rollout - base_rollout

    ch_indices = range(NC) if args.var == "all" else [CHANNELS.index(args.var)]
    summary = {
        "url": url,
        "wire": args.wire,
        "shock": args.shock,
        "days": args.days,
        "doy": args.doy,
        "latency_ms": round(ms, 2),
        "shape": list(rollout.shape),
        "channels": {
            CHANNELS[c]: {
                "min": round(float(rollout[:, c].min()), 4),
                "mean": round(float(rollout[:, c].mean()), 4),
                "max": round(float(rollout[:, c].max()), 4),
                **(
                    {
                        "delta_max_abs": round(float(np.abs(delta[:, c]).max()), 6),
                        "delta_mean_abs": round(float(np.abs(delta[:, c]).mean()), 6),
                    }
                    if delta is not None
                    else {}
                ),
            }
            for c in ch_indices
        },
    }
    txt = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    if args.json:
        print(txt)
    else:
        print(f"SIMULATE OK | shock={args.shock} | days={args.days} | latency={ms:.1f}ms | shape={tuple(rollout.shape)}")
        for name, stats in summary["channels"].items():
            extra = f" | d_max={stats['delta_max_abs']}" if "delta_max_abs" in stats else ""
            print(f"  {name:5s} | min={stats['min']:9.3f} | mean={stats['mean']:9.3f} | max={stats['max']:9.3f}{extra}")
    return 0


def _run_verify_cli(args) -> int:
    from monsoon_r1.scoring import STD_DEFAULT, make_shock, null_consistent, validity

    url = args.url.rstrip("/")
    health, h_ms = post_health(url)
    init_states = _base_init_states(180, args.artifacts_dir)
    sc = doy_sincos(180)

    base_payload = {
        "init_states": wire_encode(init_states, args.wire),
        "doy_sincos": sc,
        "steps": args.days,
        "perturbation": None,
    }
    base_resp, _ = post_simulate(url, base_payload)
    base_arr = decode_arr(base_resp["rollout"] if "rollout" in base_resp else base_resp["states"])
    base_bad = validity(base_arr, args.days)

    latencies = []
    validity_errors = list(base_bad)
    null_checks = []
    for shock_name in ("null", "warm_ocean", "monsoon_stall") * ((args.calls + 2) // 3):
        if len(latencies) >= args.calls:
            break
        pert = make_shock(shock_name, init_states[-1], STD_DEFAULT).astype(np.float32)
        payload = {
            "init_states": wire_encode(init_states, args.wire),
            "doy_sincos": sc,
            "steps": args.days,
            "perturbation": wire_encode(pert, args.wire),
        }
        resp, ms = post_simulate(url, payload)
        arr = decode_arr(resp["rollout"] if "rollout" in resp else resp["states"])
        errs = validity(arr, args.days)
        if errs:
            validity_errors.extend(f"{shock_name}: {e}" for e in errs)
        if shock_name == "null":
            null_checks.append(null_consistent(base_arr, arr, STD_DEFAULT))
        latencies.append(ms)

    banned = re.compile(
        "|".join(["open" + "-meteo", "openweather" + "map", "weather" + "bench2", "arco" + "-era5", "cds" + "api", "imd" + "lib"]),
        re.IGNORECASE,
    )
    banned_hits = []
    for p in Path(".").rglob("*.py"):
        if any(part in p.parts for part in (".git", "node_modules", ".venv", "venv")):
            continue
        if banned.search(p.read_text(errors="ignore")):
            banned_hits.append(str(p))

    valid_ok = len(validity_errors) == 0
    null_ok = bool(null_checks) and all(null_checks)
    latency_ok = max(latencies) / 1000.0 <= args.max_latency
    anti_cheat_ok = len(banned_hits) == 0

    report = {
        "health": health,
        "health_ms": round(h_ms, 2),
        "wire": args.wire,
        "calls": len(latencies),
        "max_latency_s": round(max(latencies) / 1000.0, 3),
        "mean_latency_s": round((sum(latencies) / len(latencies)) / 1000.0, 3),
        "latency_ok": latency_ok,
        "valid_ok": valid_ok,
        "validity_errors": validity_errors,
        "null_ok": null_ok,
        "banned_hits": banned_hits,
        "anti_cheat_ok": anti_cheat_ok,
    }
    print(json.dumps(report, indent=2))
    return 0 if (latency_ok and valid_ok and null_ok and anti_cheat_ok) else 1


def _event_metrics(arr: np.ndarray) -> dict:
    wind = np.sqrt(arr[:, 2] ** 2 + arr[:, 3] ** 2)
    return {
        "t2m_mean_K": round(float(arr[:, 0].mean()), 2),
        "tp_mean_mm": round(float(arr[:, 1].mean()), 3),
        "tp_max_mm": round(float(arr[:, 1].max()), 2),
        "wind_max_ms": round(float(wind.max()), 2),
        "msl_min_hpa": round(float(arr[:, 4].min() / 100.0), 2),
        "tcwv_mean_kgm2": round(float(arr[:, 5].mean()), 2),
    }


def _run_eval_live_cli(args) -> int:
    from twin.contract import LEAD_DAYS
    from twin.data import date_to_doy365, load_netcdf
    from twin.evaluate import evaluate_predictor, select_regime_windows
    from twin.scoring import csi, csi_skill, skill_vs_best_ref

    url = args.url.rstrip("/")
    health, h_ms = post_health(url)
    http_lats: list[float] = []

    def http_predict(init_states: np.ndarray, sc: list[float], steps: int, pert: np.ndarray | None) -> np.ndarray:
        payload = {
            "init_states": wire_encode(init_states, args.wire),
            "doy_sincos": [float(sc[0]), float(sc[1])],
            "steps": int(steps),
            "perturbation": wire_encode(pert, args.wire) if pert is not None else None,
        }
        resp, ms = post_simulate(url, payload, timeout=60.0)
        http_lats.append(ms)
        return decode_arr(resp["rollout"] if "rollout" in resp else resp["states"])

    res = evaluate_predictor(
        args.nc,
        args.artifacts_dir,
        http_predict,
        val_start=args.start_date,
        val_end=args.end_date,
        n_causality_windows=args.causality_windows,
    )

    data, dates, _ = load_netcdf(args.nc)
    raw_clim = np.load(args.artifacts_dir / "raw_clim.npy")
    windows = select_regime_windows(data, dates, start_date=args.start_date, end_date=args.end_date, per_regime=args.per_regime)
    lead_idx = [d - 1 for d in LEAD_DAYS]
    sample_pos = np.linspace(0, len(windows) - 1, min(len(windows), 8), dtype=int)
    events = []
    for p_i in sample_pos:
        idx = windows[int(p_i)]
        init_states = data[idx - T_IN + 1 : idx + 1]
        true_roll = data[idx + 1 : idx + 1 + SCORE_STEPS]
        sc = doy_sincos(date_to_doy365(dates[idx]) + 1)
        pred_roll = http_predict(init_states, sc, SCORE_STEPS, None)
        persist = np.repeat(init_states[-1][None], SCORE_STEPS, axis=0)
        clim_roll = np.stack([raw_clim[(date_to_doy365(dates[idx]) + k + 1) % 365] for k in range(SCORE_STEPS)], axis=0)
        sk = skill_vs_best_ref(pred_roll[lead_idx], true_roll[lead_idx], persist[lead_idx], clim_roll[lead_idx])
        c_ss = csi_skill(
            csi(pred_roll[lead_idx, 1], true_roll[lead_idx, 1]),
            csi(persist[lead_idx, 1], true_roll[lead_idx, 1]),
            csi(clim_roll[lead_idx, 1], true_roll[lead_idx, 1]),
        )
        events.append(
            {
                "init_date": dates[idx].isoformat(),
                "mean_ss": round(float(np.nanmean(sk["ss"])), 4),
                "tp_ss": round(float(sk["ss"][1]), 4),
                "csi_skill": round(float(c_ss), 4),
                "pred": _event_metrics(pred_roll),
                "actual": _event_metrics(true_roll),
            }
        )

    max_http_s = max(http_lats) / 1000.0 if http_lats else 0.0
    mean_http_s = (sum(http_lats) / len(http_lats)) / 1000.0 if http_lats else 0.0
    out_obj = {
        "health": health,
        "health_ms": round(h_ms, 2),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "wire": args.wire,
        "http_calls": len(http_lats),
        "http_max_latency_s": round(max_http_s, 3),
        "http_mean_latency_s": round(mean_http_s, 3),
        **res,
        "events": events,
    }
    txt = json.dumps(out_obj, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
    print(
        f"EVAL-LIVE | model={health.get('model_name')} | windows={res['windows']} | "
        f"score={res['automated_score_of_70']}/70 | mean_ss={res['forecast']['mean_ss']:.4f} | "
        f"csi_ss={res['forecast']['mean_csi_skill']:.4f} | blur={res['anti_blur_pts']} | "
        f"caus={res['causality_pts']} | worst_max_z={res['worst_max_z']} | http_max={max_http_s:.3f}s"
    )
    print("Channel SS:", json.dumps(res["channel_ss"]))
    for ev in events:
        p, a = ev["pred"], ev["actual"]
        print(
            f"  {ev['init_date']} | ss={ev['mean_ss']:+.3f} tp_ss={ev['tp_ss']:+.3f} csi={ev['csi_skill']:+.3f} | "
            f"t2m {p['t2m_mean_K']:.1f}/{a['t2m_mean_K']:.1f}K | tp_mean {p['tp_mean_mm']:.2f}/{a['tp_mean_mm']:.2f} | "
            f"tp_max {p['tp_max_mm']:.1f}/{a['tp_max_mm']:.1f}mm | wind_max {p['wind_max_ms']:.1f}/{a['wind_max_ms']:.1f}m/s | "
            f"msl_min {p['msl_min_hpa']:.1f}/{a['msl_min_hpa']:.1f}hPa"
        )
    ok = bool(res["valid_all"] and res["null_ok"] and max_http_s <= args.max_latency)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live HTTP client tools for Monsoon Digital Twin")
    sp = ap.add_subparsers(dest="cmd", required=True)

    h = sp.add_parser("health", help="Check live health endpoint on inference server")
    h.add_argument("--url", type=str, default="http://localhost:8000")
    h.add_argument("--json", action="store_true")

    s = sp.add_parser("simulate", help="Fast CLI pathway to run live simulation with custom shocks & flags")
    s.add_argument("--url", type=str, default="http://localhost:8000")
    s.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    s.add_argument("--shock", choices=["none", "null", "warm_ocean", "monsoon_stall", "custom"], default="null")
    s.add_argument("--shock-channel", type=str, default="t2m")
    s.add_argument("--shock-lat", type=float, default=15.0)
    s.add_argument("--shock-lon", type=float, default=72.0)
    s.add_argument("--shock-radius", type=float, default=2.5)
    s.add_argument("--shock-intensity", type=float, default=2.0)
    s.add_argument("--days", type=int, default=SCORE_STEPS)
    s.add_argument("--doy", type=int, default=180)
    s.add_argument("--var", choices=["all", *CHANNELS], default="all")
    s.add_argument("--wire", choices=["lists", "b64"], default="lists")
    s.add_argument("--compare-null", action="store_true", help="Also run null shock and report delta magnitudes")
    s.add_argument("--json", action="store_true", help="Emit machine-readable JSON for agents")
    s.add_argument("--out", type=Path, default=None)

    v = sp.add_parser("verify", help="Run automated Round-1 contract, latency, and anti-cheat checks")
    v.add_argument("--url", type=str, default="http://localhost:8000")
    v.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    v.add_argument("--wire", choices=["lists", "b64"], default="lists")
    v.add_argument("--calls", type=int, default=8)
    v.add_argument("--days", type=int, default=SCORE_STEPS)
    v.add_argument("--max-latency", type=float, default=5.0)

    el = sp.add_parser("eval-live", help="Evaluate live inference server over HTTP on NetCDF regime windows")
    el.add_argument("--url", type=str, default="http://localhost:8000")
    el.add_argument("--nc", type=Path, required=True)
    el.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    el.add_argument("--start-date", type=str, default=None)
    el.add_argument("--end-date", type=str, default=None)
    el.add_argument("--per-regime", type=int, default=5)
    el.add_argument("--causality-windows", type=int, default=6)
    el.add_argument("--wire", choices=["lists", "b64"], default="lists")
    el.add_argument("--max-latency", type=float, default=5.0)
    el.add_argument("--out", type=Path, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "health":
        resp, ms = post_health(args.url)
        if args.json:
            print(json.dumps({"latency_ms": round(ms, 2), **resp}, indent=2))
        else:
            print(f"HEALTH OK ({ms:.1f}ms) | {resp}")
    elif args.cmd == "simulate":
        return _run_simulate_cli(args)
    elif args.cmd == "verify":
        return _run_verify_cli(args)
    elif args.cmd == "eval-live":
        return _run_eval_live_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
