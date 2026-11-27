"""CLI workflow entrypoint for training, offline evaluation, and ONNX export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import NC, NLAT, NLON, T_IN
from .models import MODEL_NAMES


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
    t.add_argument("--w-var", type=float, default=0.22)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
