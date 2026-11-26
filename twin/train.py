"""3-stage GPU-resident A100 training curriculum with EMA and Round-1 Val checkpoint selection."""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from .data import GPURolloutBatcher, load_netcdf
from .evaluate import evaluate_predictor, make_predictor_from_model
from .losses import MonsoonLoss, shock_stability_loss
from .models import build_model


class EMA:
    """Exponential Moving Average of model parameters for smoother rollouts."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def rollout_step_teacher(
    model: torch.nn.Module,
    z_in: torch.Tensor,
    sc_seq: torch.Tensor,
    pushforward_noise: float = 0.0,
) -> torch.Tensor:
    """Autgressive rollout for S = sc_seq.shape[1] steps."""
    steps = sc_seq.shape[1]
    hist = z_in
    preds = []
    for k in range(steps):
        if pushforward_noise > 0.0 and k > 0 and model.training:
            hist_in = hist + torch.randn_like(hist) * pushforward_noise
        else:
            hist_in = hist
        z_next = model(hist_in, sc_seq[:, k])
        preds.append(z_next)
        hist = torch.cat([hist[:, 1:], z_next.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


def run_training(
    nc_path: Path,
    artifacts_dir: Path,
    out_dir: Path,
    model_name: str = "convnext_unet",
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
    epochs_stage1: int = 40,
    epochs_stage2: int = 50,
    epochs_stage3: int = 20,
    rollout_schedule: tuple[int, ...] = (2, 4, 6, 8, 10),
    batch_size: int = 64,
    lr: float = 1e-3,
    w_rain: float = 0.30,
    w_csi: float = 0.25,
    w_grad: float = 0.15,
    w_var: float = 0.15,
    w_stab: float = 0.15,
    pushforward_noise: float = 0.03,
    ema_decay: float = 0.995,
    seed: int = 42,
    amp: bool = True,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    static_grid = np.load(artifacts_dir / "static.npy")
    clim_tr = np.load(artifacts_dir / "clim.npy")
    uv_p = artifacts_dir / "clim_uv_coef.npy"
    clim_uv_coef = np.load(uv_p) if uv_p.exists() else None
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    anom_std = np.array(meta["anom_std"], dtype=np.float32)
    phys_std_t = torch.tensor(meta["phys_std"], dtype=torch.float32, device=device)
    anom_std_t = torch.from_numpy(anom_std).to(device)
    clim_tr_t = torch.from_numpy(clim_tr).to(device)

    data, dates, _ = load_netcdf(nc_path)
    t_cut = dt.date.fromisoformat(train_end)
    v_cut = dt.date.fromisoformat(val_end)
    tr_mask = np.array([d <= t_cut for d in dates])
    va_mask = np.array([(d > t_cut) and (d <= v_cut) for d in dates])

    model = build_model(model_name, static_grid=static_grid, clim_uv_coef=clim_uv_coef, anom_std=anom_std).to(device)
    ema = EMA(model, decay=ema_decay)

    criterion = MonsoonLoss(
        clim_tr=torch.from_numpy(clim_tr),
        anom_std=torch.from_numpy(anom_std),
        w_rain=w_rain,
        w_csi=w_csi,
        w_grad=w_grad,
        w_var=w_var,
    ).to(device)

    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tr_batcher = GPURolloutBatcher(
        data[tr_mask], dates[tr_mask], clim_tr, anom_std, device=device, batch_size=batch_size, rollout_steps=1, shuffle=True, drop_last=True
    )
    va_batcher = (
        GPURolloutBatcher(
            data[va_mask], dates[va_mask], clim_tr, anom_std, device=device, batch_size=batch_size, rollout_steps=1, shuffle=False, drop_last=False
        )
        if va_mask.any()
        else None
    )

    best_ckpt = out_dir / f"{model_name}_best.pt"
    best_val_loss = float("inf")
    best_r1_score = -1.0

    # Reusable eval model shell for EMA evaluation
    eval_model = copy.deepcopy(model).eval()

    def _eval_val_loss(steps: int) -> float:
        if va_batcher is None:
            return 0.0
        va_batcher.set_rollout_steps(steps)
        eval_model.load_state_dict(ema.shadow)
        eval_model.eval()
        losses = []
        with torch.no_grad():
            for z_in, sc_seq, doy_tgt, z_tgt, phys_tgt in va_batcher:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = rollout_step_teacher(eval_model, z_in, sc_seq, pushforward_noise=0.0)
                    loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                losses.append(float(loss))
        return float(np.mean(losses)) if losses else 0.0

    def _eval_r1_score() -> float:
        if va_batcher is None:
            return 0.0
        eval_model.load_state_dict(ema.shadow)
        eval_model.eval()
        pred_fn = make_predictor_from_model(artifacts_dir, eval_model, device)
        val_start = dates[va_mask][0].isoformat()
        res = evaluate_predictor(nc_path, artifacts_dir, pred_fn, val_start=val_start, val_end=val_end, n_causality_windows=3)
        return float(res["automated_score_of_70"])

    def _save_ckpt(stage: int, val_loss: float, val_score: float) -> None:
        torch.save(
            {
                "model": ema.state_dict(),
                "model_name": model_name,
                "stage": stage,
                "val_loss": val_loss,
                "val_score": val_score,
            },
            best_ckpt,
        )

    # Stage 1: 1-step pretraining on GPU-resident batches
    if epochs_stage1 > 0:
        tr_batcher.set_rollout_steps(1)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage1, eta_min=lr * 0.05)
        for ep in range(1, epochs_stage1 + 1):
            model.train()
            tr_losses = []
            for z_in, sc_seq, doy_tgt, z_tgt, phys_tgt in tr_batcher:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = rollout_step_teacher(model, z_in, sc_seq, pushforward_noise=0.0)
                    loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                tr_losses.append(float(loss.detach()))
            sched.step()
            val_loss = _eval_val_loss(1)
            metric = val_loss if va_batcher is not None else float(np.mean(tr_losses))
            if metric <= best_val_loss or not best_ckpt.exists():
                best_val_loss = metric
                _save_ckpt(1, val_loss, 0.0)
            if ep == 1 or ep % 10 == 0 or ep == epochs_stage1:
                print(f"[S1 {ep:02d}/{epochs_stage1}] train={np.mean(tr_losses):.4f} val={val_loss:.4f}", flush=True)

    # Stage 2: Progressive unrolled autoregressive BPTT curriculum
    if epochs_stage2 > 0:
        best_val_s2 = float("inf")
        opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.35, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage2, eta_min=lr * 0.02)
        for ep in range(1, epochs_stage2 + 1):
            stage_idx = min(int((ep - 1) * len(rollout_schedule) / max(epochs_stage2, 1)), len(rollout_schedule) - 1)
            steps = rollout_schedule[stage_idx]
            tr_batcher.set_rollout_steps(steps)
            model.train()
            tr_losses = []
            for b_idx, (z_in, sc_seq, doy_tgt, z_tgt, phys_tgt) in enumerate(tr_batcher):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = rollout_step_teacher(model, z_in, sc_seq, pushforward_noise=pushforward_noise)
                    loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                    if w_stab > 0.0 and b_idx % 2 == 0:
                        n_s = min(8, z_in.shape[0])
                        s_stab = min(2, steps)
                        z_in_p = z_in[:n_s].clone()
                        rh = int(torch.randint(6, 13, (1,)))
                        cw = int(torch.randint(6, 13, (1,)))
                        r0 = int(torch.randint(0, 32 - rh + 1, (1,)))
                        c0 = int(torch.randint(0, 36 - cw + 1, (1,)))
                        d_t2m = float(1.5 + 1.7 * torch.rand(1))
                        d_tcwv = float(3.0 + 6.0 * torch.rand(1))
                        z_in_p[:, -1, 0, r0 : r0 + rh, c0 : c0 + cw] += d_t2m / float(anom_std[0])
                        z_in_p[:, -1, 5, r0 : r0 + rh, c0 : c0 + cw] += d_tcwv / float(anom_std[5])
                        z_pert = rollout_step_teacher(model, z_in_p, sc_seq[:n_s, :s_stab], pushforward_noise=0.0)
                        loss = loss + w_stab * shock_stability_loss(
                            pred[:n_s, :s_stab].detach(), z_pert, doy_tgt[:n_s, :s_stab], clim_tr_t, anom_std_t, phys_std_t
                        )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                tr_losses.append(float(loss.detach()))
            sched.step()

            if ep % 5 == 0 or ep == epochs_stage2:
                val_loss = _eval_val_loss(rollout_schedule[-1])
                if va_batcher is not None:
                    r1_score = _eval_r1_score()
                    if r1_score >= best_r1_score:
                        best_r1_score = r1_score
                        _save_ckpt(2, val_loss, r1_score)
                    print(f"[S2 {ep:02d}/{epochs_stage2} steps={steps}] train={np.mean(tr_losses):.4f} val_10={val_loss:.4f} R1={r1_score:.2f}/70", flush=True)
                else:
                    tr_m = float(np.mean(tr_losses))
                    if tr_m <= best_val_s2:
                        best_val_s2 = tr_m
                        _save_ckpt(2, tr_m, 0.0)
                    print(f"[S2 {ep:02d}/{epochs_stage2} steps={steps}] train={tr_m:.4f}", flush=True)

    # Stage 3: Low-LR 10-step rollout polish
    if epochs_stage3 > 0:
        best_val_s3 = float("inf")
        final_steps = rollout_schedule[-1]
        tr_batcher.set_rollout_steps(final_steps)
        opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.10, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage3, eta_min=lr * 0.005)
        for ep in range(1, epochs_stage3 + 1):
            model.train()
            tr_losses = []
            for b_idx, (z_in, sc_seq, doy_tgt, z_tgt, phys_tgt) in enumerate(tr_batcher):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = rollout_step_teacher(model, z_in, sc_seq, pushforward_noise=pushforward_noise * 0.5)
                    loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                    if w_stab > 0.0 and b_idx % 2 == 0:
                        n_s = min(8, z_in.shape[0])
                        s_stab = min(2, final_steps)
                        z_in_p = z_in[:n_s].clone()
                        rh = int(torch.randint(6, 13, (1,)))
                        cw = int(torch.randint(6, 13, (1,)))
                        r0 = int(torch.randint(0, 32 - rh + 1, (1,)))
                        c0 = int(torch.randint(0, 36 - cw + 1, (1,)))
                        d_t2m = float(1.5 + 1.7 * torch.rand(1))
                        d_tcwv = float(3.0 + 6.0 * torch.rand(1))
                        z_in_p[:, -1, 0, r0 : r0 + rh, c0 : c0 + cw] += d_t2m / float(anom_std[0])
                        z_in_p[:, -1, 5, r0 : r0 + rh, c0 : c0 + cw] += d_tcwv / float(anom_std[5])
                        z_pert = rollout_step_teacher(model, z_in_p, sc_seq[:n_s, :s_stab], pushforward_noise=0.0)
                        loss = loss + w_stab * shock_stability_loss(
                            pred[:n_s, :s_stab].detach(), z_pert, doy_tgt[:n_s, :s_stab], clim_tr_t, anom_std_t, phys_std_t
                        )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                tr_losses.append(float(loss.detach()))
            sched.step()

            if ep % 5 == 0 or ep == epochs_stage3:
                val_loss = _eval_val_loss(final_steps)
                if va_batcher is not None:
                    r1_score = _eval_r1_score()
                    if r1_score >= best_r1_score:
                        best_r1_score = r1_score
                        _save_ckpt(3, val_loss, r1_score)
                    print(f"[S3 {ep:02d}/{epochs_stage3} steps={final_steps}] train={np.mean(tr_losses):.4f} val_10={val_loss:.4f} R1={r1_score:.2f}/70", flush=True)
                else:
                    tr_m = float(np.mean(tr_losses))
                    if tr_m <= best_val_s3:
                        best_val_s3 = tr_m
                        _save_ckpt(3, tr_m, 0.0)
                    print(f"[S3 {ep:02d}/{epochs_stage3} steps={final_steps}] train={tr_m:.4f}", flush=True)

    return best_ckpt
