"""CLI workflow entrypoint with feature flags for agents and Colab A100 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import NC, NLAT, NLON, SCORE_STEPS, T_IN, doy_sincos
from .data import download_dataset, prepare_artifacts
from .evaluate import evaluate_predictor, make_predictor
from .export_onnx import export_checkpoint_to_onnx
from .models import build_model
from .train import run_training


def _smoke_test() -> None:
    import torch

    static = np.zeros((4, NLAT, NLON), dtype=np.float32)
    z = torch.randn(2, T_IN, NC, NLAT, NLON)
    sc = torch.tensor([[0.0, 1.0], [0.5, 0.86]], dtype=torch.float32)
    for name in ("convnext_unet", "advect_resnet"):
        m = build_model(name, static_grid=static)
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

    p = sp.add_parser("prepare-artifacts", help="Compute harmonic climatology, norm stats, and static.npy")
    p.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--train-end", type=str, default="2018-12-31")

    t = sp.add_parser("train", help="Run 2-stage A100 training curriculum")
    t.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    t.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    t.add_argument("--out-dir", type=Path, default=Path("runs"))
    t.add_argument("--model", choices=["convnext_unet", "advect_resnet"], required=True)
    t.add_argument("--train-end", type=str, default="2018-12-31")
    t.add_argument("--val-end", type=str, default="2021-12-31")
    t.add_argument("--epochs-s1", type=int, default=25)
    t.add_argument("--epochs-s2", type=int, default=25)
    t.add_argument("--rollout-schedule", type=str, default="4,8,10")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--w-rain", type=float, default=0.35)
    t.add_argument("--w-grad", type=float, default=0.15)
    t.add_argument("--w-var", type=float, default=0.05)
    t.add_argument("--pushforward-noise", type=float, default=0.03)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-amp", action="store_true")

    e = sp.add_parser("eval", help="Evaluate checkpoints or ONNX ensemble on 25-window + shock benchmark")
    e.add_argument("--nc", type=Path, default=Path("data/india_climate_train.nc"))
    e.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    e.add_argument("--ckpts", type=Path, nargs="*", default=None)
    e.add_argument("--onnx", type=Path, nargs="*", default=None)
    e.add_argument("--val-start", type=str, default="2019-01-01")
    e.add_argument("--out-json", type=Path, default=None)

    x = sp.add_parser("export-onnx", help="Export PyTorch checkpoint to ONNX and benchmark CPU latency")
    x.add_argument("--ckpt", type=Path, required=True)
    x.add_argument("--static", type=Path, default=Path("artifacts/static.npy"))
    x.add_argument("--out", type=Path, required=True)

    sp.add_parser("smoke-test", help="Verify both model architectures on synthetic tensors")

    args = ap.parse_args(argv)
    if args.cmd == "download-data":
        nc, st = download_dataset(args.data_dir, force=args.force)
        print(f"Downloaded: {nc} ({nc.stat().st_size / 1e6:.1f} MB), {st}")
    elif args.cmd == "prepare-artifacts":
        meta = prepare_artifacts(args.nc, args.artifacts_dir, train_end=args.train_end)
        print("Prepared artifacts:", json.dumps(meta, indent=2))
    elif args.cmd == "train":
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
            rollout_schedule=sched,
            batch_size=args.batch_size,
            lr=args.lr,
            w_rain=args.w_rain,
            w_grad=args.w_grad,
            w_var=args.w_var,
            pushforward_noise=args.pushforward_noise,
            seed=args.seed,
            amp=not args.no_amp,
        )
        print(f"Best checkpoint saved: {ckpt}")
    elif args.cmd == "eval":
        pred_fn = make_predictor(args.artifacts_dir, ckpts=args.ckpts, onnx_paths=args.onnx)
        res = evaluate_predictor(args.nc, args.artifacts_dir, pred_fn, val_start=args.val_start)
        txt = json.dumps(res, indent=2)
        print(txt)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(txt)
    elif args.cmd == "export-onnx":
        export_checkpoint_to_onnx(args.ckpt, args.static, args.out)
    elif args.cmd == "smoke-test":
        _smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
