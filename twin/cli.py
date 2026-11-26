"""CLI workflow entrypoint with feature flags for agents and Colab A100 runs."""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from .contract import CHANNELS, LAT, LON, NC, NLAT, NLON, SCORE_STEPS, T_IN, doy_sincos, encode_list
from .models import MODEL_NAMES


def _http_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 30.0,
    method: str | None = None,
) -> tuple[dict, float]:
    if method is None:
        method = "POST" if payload is not None else "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
    elif method == "POST":
        data = b""
        headers = {}
    else:
        data = None
        headers = {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, (time.perf_counter() - t0) * 1000.0


def _decode_arr(obj) -> np.ndarray:
    if isinstance(obj, dict) and "b64" in obj:
        return np.frombuffer(base64.b64decode(obj["b64"]), dtype="<f4").reshape(obj["shape"]).astype(np.float32)
    return np.asarray(obj, dtype=np.float32)


def _encode_arr(arr: np.ndarray) -> dict:
    a = np.ascontiguousarray(arr, dtype="<f4")
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape)}


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


def _wire_encode(arr: np.ndarray, wire: str):
    return encode_list(arr) if wire == "lists" else _encode_arr(arr)



def _build_perturbation(args, last_state: np.ndarray) -> np.ndarray | None:
    if args.shock in ("none", "null"):
        return np.zeros((NC, NLAT, NLON), dtype=np.float32) if args.shock == "null" else None
    if args.shock in ("warm_ocean", "monsoon_stall"):
        from monsoon_r1.scoring import STD_DEFAULT, make_shock
        return make_shock(args.shock, last_state, STD_DEFAULT).astype(np.float32)
    # custom localized Gaussian shock on LAT[0]=6.5..37.5, LON[0]=65.5..100.5
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
        "init_states": _wire_encode(init_states, args.wire),
        "doy_sincos": sc,
        "steps": args.days,
        "perturbation": _wire_encode(pert, args.wire) if pert is not None else None,
    }
    resp, ms = _http_json(f"{url}/simulate", payload)
    rollout = _decode_arr(resp["rollout"] if "rollout" in resp else resp["states"])

    delta = None
    if args.compare_null:
        base_payload = {
            "init_states": _wire_encode(init_states, args.wire),
            "doy_sincos": sc,
            "steps": args.days,
            "perturbation": _wire_encode(np.zeros((NC, NLAT, NLON), dtype=np.float32), args.wire),
        }
        base_resp, _ = _http_json(f"{url}/simulate", base_payload)
        base_rollout = _decode_arr(base_resp["rollout"] if "rollout" in base_resp else base_resp["states"])
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
    health, h_ms = _http_json(f"{url}/health", method="POST")
    init_states = _base_init_states(180, args.artifacts_dir)
    sc = doy_sincos(180)

    # Unperturbed baseline for null-shock consistency check
    base_payload = {
        "init_states": _wire_encode(init_states, args.wire),
        "doy_sincos": sc,
        "steps": args.days,
        "perturbation": None,
    }
    base_resp, _ = _http_json(f"{url}/simulate", base_payload)
    base_arr = _decode_arr(base_resp["rollout"] if "rollout" in base_resp else base_resp["states"])
    base_bad = validity(base_arr, args.days)

    latencies = []
    validity_errors = list(base_bad)
    null_checks = []
    for shock_name in ("null", "warm_ocean", "monsoon_stall") * ((args.calls + 2) // 3):
        if len(latencies) >= args.calls:
            break
        pert = make_shock(shock_name, init_states[-1], STD_DEFAULT).astype(np.float32)
        payload = {
            "init_states": _wire_encode(init_states, args.wire),
            "doy_sincos": sc,
            "steps": args.days,
            "perturbation": _wire_encode(pert, args.wire),
        }
        resp, ms = _http_json(f"{url}/simulate", payload)
        arr = _decode_arr(resp["rollout"] if "rollout" in resp else resp["states"])
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


def _smoke_test() -> None:
    import torch
    from .models import build_model

    static = np.zeros((6, NLAT, NLON), dtype=np.float32)
    uv_coef = np.zeros((9, 2, NLAT, NLON), dtype=np.float32)
    phys_coef = np.zeros((9, NC, NLAT, NLON), dtype=np.float32)
    anom_std = np.array([2.5, 1.0, 3.5, 2.5, 300.0, 8.0], dtype=np.float32)
    z = torch.randn(2, T_IN, NC, NLAT, NLON)
    sc = torch.tensor([[0.0, 1.0], [0.5, 0.86]], dtype=torch.float32)
    for name in MODEL_NAMES:
        m = build_model(name, static_grid=static, clim_uv_coef=uv_coef, clim_phys_coef=phys_coef, anom_std=anom_std)
        out = m(z, sc)
        assert out.shape == (2, NC, NLAT, NLON), f"{name} bad shape {out.shape}"
        assert torch.isfinite(out).all(), f"{name} non-finite"
        n_params = sum(p.numel() for p in m.parameters())
        print(f"SMOKE OK | {name:15s} | params={n_params:,} | out={tuple(out.shape)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Multiverse of Monsoon Digital Twin CLI")
    sp = ap.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("download-data", help="Download india_climate_train.nc and stats.json")
    d.add_argument("--data-dir", type=Path, default=Path("data"))
    d.add_argument("--force", action="store_true")

    p = sp.add_parser("prepare-artifacts", help="Compute harmonic climatology, wind coefs, norm stats, and static.npy")
    p.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--train-end", type=str, default="2018-12-31")

    t = sp.add_parser("train", help="Run 3-stage GPU-resident A100 training curriculum with EMA")
    t.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    t.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    t.add_argument("--out-dir", type=Path, default=Path("runs"))
    t.add_argument("--model", choices=list(MODEL_NAMES), required=True)
    t.add_argument("--train-end", type=str, default="2018-12-31")
    t.add_argument("--val-end", type=str, default="2021-12-31")
    t.add_argument("--epochs-s1", type=int, default=40)
    t.add_argument("--epochs-s2", type=int, default=50)
    t.add_argument("--epochs-s3", type=int, default=20)
    t.add_argument("--rollout-schedule", type=str, default="2,4,6,8,10")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--w-rain", type=float, default=0.30)
    t.add_argument("--w-csi", type=float, default=0.25)
    t.add_argument("--w-grad", type=float, default=0.15)
    t.add_argument("--w-var", type=float, default=0.15)
    t.add_argument("--w-stab", type=float, default=0.40)
    t.add_argument("--pushforward-noise", type=float, default=0.03)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--resume-ckpt", type=Path, default=None)

    e = sp.add_parser("eval", help="Evaluate checkpoints or ONNX ensemble on 25-window + multi-season shock benchmark")
    e.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    e.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    e.add_argument("--ckpts", type=Path, nargs="*", default=None)
    e.add_argument("--onnx", type=Path, nargs="*", default=None)
    e.add_argument("--weights-json", type=Path, default=None)
    e.add_argument("--val-start", type=str, default="2019-01-01")
    e.add_argument("--val-end", type=str, default=None)
    e.add_argument("--out-json", type=Path, default=None)

    fe = sp.add_parser("fit-ensemble", help="Optimize simplex ensemble weights strictly on Val")
    fe.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    fe.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    fe.add_argument("--ckpts", type=Path, nargs="+", required=True)
    fe.add_argument("--val-start", type=str, default="2019-01-01")
    fe.add_argument("--val-end", type=str, default=None)
    fe.add_argument("--out", type=Path, default=Path("artifacts/ensemble_weights.json"))

    x = sp.add_parser("export-onnx", help="Export PyTorch checkpoint to ONNX and benchmark CPU latency")
    x.add_argument("--ckpt", type=Path, required=True)
    x.add_argument("--static", type=Path, default=Path("artifacts/static.npy"))
    x.add_argument("--out", type=Path, required=True)

    sp.add_parser("smoke-test", help="Verify all 4 model architectures on synthetic tensors")

    h = sp.add_parser("health", help="Check live /health endpoint on inference server")
    h.add_argument("--url", type=str, default="http://localhost:8000")
    h.add_argument("--json", action="store_true")

    s = sp.add_parser("simulate", help="Fast CLI pathway to run /simulate with custom shocks & flags")
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

    args = ap.parse_args(argv)
    if args.cmd == "download-data":
        from .data import download_dataset

        nc, st = download_dataset(args.data_dir, force=args.force)
        print(f"Downloaded: {nc} ({nc.stat().st_size / 1e6:.1f} MB), {st}")
    elif args.cmd == "prepare-artifacts":
        from .data import prepare_artifacts

        meta = prepare_artifacts(args.nc, args.artifacts_dir, train_end=args.train_end)
        print("Prepared artifacts:", json.dumps(meta, indent=2))
    elif args.cmd == "train":
        from .train import run_training

        sched = tuple(int(v) for v in args.rollout_schedule.split(","))
        ckpt = run_training(
            nc_path=args.nc,
            artifacts_dir=args.artifacts_dir,
            out_dir=args.out_dir,
            model_name=args.model,
            train_end=args.train_end,
            val_end=args.val_end,
            epochs_stage1=args.epochs_s1,
            epochs_stage2=args.epochs_s2,
            epochs_stage3=args.epochs_s3,
            rollout_schedule=sched,
            batch_size=args.batch_size,
            lr=args.lr,
            w_rain=args.w_rain,
            w_csi=args.w_csi,
            w_grad=args.w_grad,
            w_var=args.w_var,
            w_stab=args.w_stab,
            pushforward_noise=args.pushforward_noise,
            seed=args.seed,
            amp=not args.no_amp,
            resume_ckpt=args.resume_ckpt,
        )
        print(f"Best checkpoint saved: {ckpt}")
    elif args.cmd == "eval":
        from .evaluate import evaluate_predictor, make_predictor

        pred_fn = make_predictor(
            args.artifacts_dir,
            ckpts=args.ckpts,
            onnx_paths=args.onnx,
            weights_json=args.weights_json,
        )
        res = evaluate_predictor(args.nc, args.artifacts_dir, pred_fn, val_start=args.val_start, val_end=args.val_end)
        txt = json.dumps(res, indent=2)
        print(txt)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(txt)
    elif args.cmd == "fit-ensemble":
        from .evaluate import fit_ensemble_weights

        res = fit_ensemble_weights(
            nc_path=args.nc,
            artifacts_dir=args.artifacts_dir,
            ckpts=args.ckpts,
            val_start=args.val_start,
            val_end=args.val_end,
            out_path=args.out,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "export-onnx":
        from .export_onnx import export_checkpoint_to_onnx

        export_checkpoint_to_onnx(args.ckpt, args.static, args.out)
    elif args.cmd == "smoke-test":
        _smoke_test()
    elif args.cmd == "health":
        resp, ms = _http_json(f"{args.url.rstrip('/')}/health", method="POST")
        if args.json:
            print(json.dumps({"latency_ms": round(ms, 2), **resp}, indent=2))
        else:
            print(f"HEALTH OK ({ms:.1f}ms) | {resp}")
    elif args.cmd == "simulate":
        return _run_simulate_cli(args)
    elif args.cmd == "verify":
        return _run_verify_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
