"""2-stage A100 training curriculum: 1-step pretrain -> unrolled autoregressive BPTT."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contract import T_IN
from .data import MonsoonRolloutDataset, load_netcdf
from .losses import MonsoonLoss
from .models import build_model


def rollout_step_teacher(
    model: torch.nn.Module,
    z_in: torch.Tensor,
    sc_seq: torch.Tensor,
    pushforward_noise: float = 0.0,
) -> torch.Tensor:
    """Autoregressive rollout for S = sc_seq.shape[1] steps."""
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
    epochs_stage1: int = 25,
    epochs_stage2: int = 25,
    rollout_schedule: tuple[int, ...] = (4, 8, 10),
    batch_size: int = 64,
    lr: float = 1e-3,
    w_rain: float = 0.35,
    w_grad: float = 0.15,
    w_var: float = 0.05,
    pushforward_noise: float = 0.03,
    seed: int = 42,
    amp: bool = True,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    static_grid = np.load(artifacts_dir / "static.npy")
    clim_tr = np.load(artifacts_dir / "clim.npy")
    meta = json.loads((artifacts_dir / "norm_stats.json").read_text())
    anom_std = np.array(meta["anom_std"], dtype=np.float32)

    data, dates, _ = load_netcdf(nc_path)
    t_cut = dt.date.fromisoformat(train_end)
    v_cut = dt.date.fromisoformat(val_end)
    tr_mask = np.array([d <= t_cut for d in dates])
    va_mask = np.array([(d > t_cut) and (d <= v_cut) for d in dates])

    model = build_model(model_name, static_grid=static_grid).to(device)
    criterion = MonsoonLoss(
        clim_tr=torch.from_numpy(clim_tr),
        anom_std=torch.from_numpy(anom_std),
        w_rain=w_rain,
        w_grad=w_grad,
        w_var=w_var,
    ).to(device)

    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_ckpt = out_dir / f"{model_name}_best.pt"

    def _eval_val(steps: int) -> float:
        if not va_mask.any():
            return 0.0
        ds_va = MonsoonRolloutDataset(data[va_mask], dates[va_mask], clim_tr, anom_std, rollout_steps=steps)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
        model.eval()
        losses = []
        with torch.no_grad():
            for z_in, sc_seq, doy_tgt, z_tgt, phys_tgt in dl_va:
                z_in = z_in.to(device)
                sc_seq = sc_seq.to(device)
                doy_tgt = doy_tgt.to(device)
                z_tgt = z_tgt.to(device)
                phys_tgt = phys_tgt.to(device)
                pred = rollout_step_teacher(model, z_in, sc_seq, pushforward_noise=0.0)
                loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                losses.append(float(loss))
        return float(np.mean(losses)) if losses else 0.0

    # Stage 1: 1-step pretraining
    if epochs_stage1 > 0:
        ds_tr1 = MonsoonRolloutDataset(data[tr_mask], dates[tr_mask], clim_tr, anom_std, rollout_steps=1)
        dl_tr1 = DataLoader(ds_tr1, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage1, eta_min=lr * 0.05)
        for ep in range(1, epochs_stage1 + 1):
            model.train()
            tr_losses = []
            for z_in, sc_seq, doy_tgt, z_tgt, phys_tgt in dl_tr1:
                z_in, sc_seq = z_in.to(device), sc_seq.to(device)
                doy_tgt, z_tgt, phys_tgt = doy_tgt.to(device), z_tgt.to(device), phys_tgt.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = rollout_step_teacher(model, z_in, sc_seq, pushforward_noise=0.0)
                    loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tr_losses.append(float(loss.detach()))
            sched.step()
            val_loss = _eval_val(1)
            if val_loss <= best_val or not best_ckpt.exists():
                best_val = val_loss
                torch.save({"model": model.state_dict(), "model_name": model_name, "stage": 1, "val_loss": val_loss}, best_ckpt)
            if ep == 1 or ep % 5 == 0 or ep == epochs_stage1:
                print(f"[S1 {ep:02d}/{epochs_stage1}] train={np.mean(tr_losses):.4f} val={val_loss:.4f}", flush=True)

    # Stage 2: Unrolled autoregressive BPTT curriculum
    if epochs_stage2 > 0:
        best_val_s2 = float("inf")
        opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.35, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage2, eta_min=lr * 0.01)
        for ep in range(1, epochs_stage2 + 1):
            stage_idx = min(int((ep - 1) * len(rollout_schedule) / max(epochs_stage2, 1)), len(rollout_schedule) - 1)
            steps = rollout_schedule[stage_idx]
            ds_tr = MonsoonRolloutDataset(data[tr_mask], dates[tr_mask], clim_tr, anom_std, rollout_steps=steps)
            dl_tr = DataLoader(ds_tr, batch_size=max(16, batch_size // 2), shuffle=True, num_workers=0, drop_last=True)
            model.train()
            tr_losses = []
            for z_in, sc_seq, doy_tgt, z_tgt, phys_tgt in dl_tr:
                z_in, sc_seq = z_in.to(device), sc_seq.to(device)
                doy_tgt, z_tgt, phys_tgt = doy_tgt.to(device), z_tgt.to(device), phys_tgt.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = rollout_step_teacher(model, z_in, sc_seq, pushforward_noise=pushforward_noise)
                    loss, _ = criterion(pred, z_tgt, doy_tgt, phys_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tr_losses.append(float(loss.detach()))
            sched.step()
            val_loss = _eval_val(rollout_schedule[-1])
            if val_loss <= best_val_s2:
                best_val_s2 = val_loss
                torch.save({"model": model.state_dict(), "model_name": model_name, "stage": 2, "val_loss": val_loss}, best_ckpt)
            if ep == 1 or ep % 5 == 0 or ep == epochs_stage2:
                print(f"[S2 {ep:02d}/{epochs_stage2} steps={steps}] train={np.mean(tr_losses):.4f} val_10={val_loss:.4f}", flush=True)

    return best_ckpt
