"""Dataset download, temporal splits, harmonic climatology, and GPU rollout batcher."""
from __future__ import annotations

import datetime as dt
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contract import CHANNELS, LAT, LON, NC, NLAT, NLON, T_IN, doy_sincos

RELEASE_BASE = "https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/download/v1.0.0"
NC_URL = f"{RELEASE_BASE}/india_climate_train.nc"
STATS_URL = f"{RELEASE_BASE}/stats.json"


def download_dataset(data_dir: Path, force: bool = False) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    nc_path = data_dir / "india_climate_train.nc"
    stats_path = data_dir / "stats.json"
    for url, dst in ((NC_URL, nc_path), (STATS_URL, stats_path)):
        if force or not dst.exists() or dst.stat().st_size < 100:
            print(f"Downloading {url} -> {dst} ...", flush=True)
            urllib.request.urlretrieve(url, dst)
    return nc_path, stats_path


def load_netcdf(nc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns data [N,6,32,36] float32, dates [N] datetime.date, static [2,32,36] float32."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    data = ds["data"].values.astype(np.float32)
    data[:, 1] = np.maximum(data[:, 1], 0.0)
    raw_times = ds["time"].values
    dates = np.array([dt.date.fromisoformat(str(t)[:10]) for t in raw_times])
    if "static" in ds:
        static = ds["static"].values.astype(np.float32)
    else:
        static = np.zeros((2, NLAT, NLON), dtype=np.float32)
    ds.close()
    return data, dates, static


TP_LOG1P_MAX = 4.15


def transform_physical(x: np.ndarray) -> np.ndarray:
    """Physical [..., 6, 32, 36] -> transformed space with log1p(tp)."""
    out = np.array(x, dtype=np.float32, copy=True)
    out[..., 1, :, :] = np.log1p(np.maximum(out[..., 1, :, :], 0.0))
    return out


def inv_transform_physical(x: np.ndarray) -> np.ndarray:
    """Transformed space -> physical [..., 6, 32, 36] with non-negative rain."""
    out = np.array(x, dtype=np.float32, copy=True)
    out[..., 1, :, :] = np.expm1(np.clip(out[..., 1, :, :], 0.0, TP_LOG1P_MAX))
    out[..., 0, :, :] = np.clip(out[..., 0, :, :], 233.15, 333.15)
    return out


def transform_physical_torch(x: torch.Tensor) -> torch.Tensor:
    tp_log = torch.log1p(torch.clamp(x[..., 1:2, :, :], min=0.0))
    return torch.cat([x[..., :1, :, :], tp_log, x[..., 2:, :, :]], dim=-3)


def inv_transform_physical_torch(x: torch.Tensor) -> torch.Tensor:
    tp_mm = torch.expm1(torch.clamp(x[..., 1:2, :, :], min=0.0, max=TP_LOG1P_MAX))
    return torch.cat([x[..., :1, :, :], tp_mm, x[..., 2:, :, :]], dim=-3)


def date_to_doy365(d: dt.date) -> int:
    """0..364 day-of-year index (clamps Dec 31 of leap years to 364)."""
    return min(d.timetuple().tm_yday - 1, 364)


def fit_harmonic_coefs(
    data_arr: np.ndarray,
    dates: np.ndarray,
    harmonics: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 365-day harmonic regression, returning (coef [1+2H, C, 32, 36], smooth [365, C, 32, 36])."""
    c_dim = data_arr.shape[1]
    doys = np.array([date_to_doy365(d) for d in dates], dtype=np.int64)
    daily_mean = np.zeros((365, c_dim, NLAT, NLON), dtype=np.float64)
    counts = np.zeros(365, dtype=np.float64)
    for i, k in enumerate(doys):
        daily_mean[k] += data_arr[i]
        counts[k] += 1.0
    daily_mean /= np.maximum(counts, 1.0)[:, None, None, None]

    t = np.arange(365, dtype=np.float64) * (2.0 * np.pi / 365.0)
    cols = [np.ones(365, dtype=np.float64)]
    for h in range(1, harmonics + 1):
        cols.append(np.sin(h * t))
        cols.append(np.cos(h * t))
    X = np.stack(cols, axis=1)  # [365, 1 + 2*harmonics]
    pinv = np.linalg.pinv(X)
    coef = (pinv @ daily_mean.reshape(365, -1)).reshape(1 + 2 * harmonics, c_dim, NLAT, NLON).astype(np.float32)
    smooth = (X @ coef.reshape(1 + 2 * harmonics, -1)).reshape(365, c_dim, NLAT, NLON).astype(np.float32)
    return coef, smooth


def fit_harmonic_climatology(data_trans: np.ndarray, dates: np.ndarray, harmonics: int = 4) -> np.ndarray:
    """Fit smooth 365-day harmonic climatology [365, C, 32, 36]."""
    _, smooth = fit_harmonic_coefs(data_trans, dates, harmonics=harmonics)
    return smooth


def normalize_static(static: np.ndarray) -> np.ndarray:
    """Build [6, 32, 36] static tensor: lsm, norm_orography, norm_lat, norm_lon, doro_dx, doro_dy."""
    lsm = np.clip(static[0], 0.0, 1.0)
    oro = static[1]
    oro_n = (oro - oro.mean()) / (oro.std() + 1e-6)
    doro_dy, doro_dx = np.gradient(oro_n)
    doro_dx = doro_dx / (doro_dx.std() + 1e-6)
    doro_dy = doro_dy / (doro_dy.std() + 1e-6)
    lat_grid = np.broadcast_to(((LAT - LAT.mean()) / (LAT.std() + 1e-6))[:, None], (NLAT, NLON))
    lon_grid = np.broadcast_to(((LON - LON.mean()) / (LON.std() + 1e-6))[None, :], (NLAT, NLON))
    return np.stack([lsm, oro_n, lat_grid, lon_grid, doro_dx, doro_dy], axis=0).astype(np.float32)


def prepare_artifacts(
    nc_path: Path,
    out_dir: Path,
    train_end: str = "2018-12-31",
) -> dict:
    """Compute climatology, wind harmonic coefficients, anomaly stats, and static grids on train split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data, dates, static = load_netcdf(nc_path)
    cutoff = dt.date.fromisoformat(train_end)
    train_mask = np.array([d <= cutoff for d in dates])
    if not train_mask.any():
        raise ValueError(f"No dates <= {train_end}")

    data_tr = transform_physical(data[train_mask])
    clim_tr = fit_harmonic_climatology(data_tr, dates[train_mask], harmonics=4)
    clim_phys_coef, raw_clim = fit_harmonic_coefs(data[train_mask], dates[train_mask], harmonics=4)
    raw_clim[:, 1] = np.maximum(raw_clim[:, 1], 0.0)
    clim_uv_coef = clim_phys_coef[:, 2:4].copy()

    doys = np.array([date_to_doy365(d) for d in dates[train_mask]], dtype=np.int64)
    anom = data_tr - clim_tr[doys]
    anom_std = anom.std(axis=(0, 2, 3)).astype(np.float32)
    anom_std = np.maximum(anom_std, 1e-4)

    static_norm = normalize_static(static)
    np.save(out_dir / "static.npy", static_norm)
    np.save(out_dir / "clim.npy", clim_tr)
    np.save(out_dir / "raw_clim.npy", raw_clim)
    np.save(out_dir / "clim_uv_coef.npy", clim_uv_coef)
    np.save(out_dir / "clim_phys_coef.npy", clim_phys_coef)

    meta = {
        "channels": list(CHANNELS),
        "train_end": train_end,
        "anom_std": [float(x) for x in anom_std],
        "phys_std": [float(x) for x in data[train_mask].std(axis=(0, 2, 3))],
        "phys_mean": [float(x) for x in data[train_mask].mean(axis=(0, 2, 3))],
    }
    (out_dir / "norm_stats.json").write_text(json.dumps(meta, indent=2))
    return meta


class MonsoonRolloutDataset(Dataset):
    """Yields multi-step rollout windows in normalized anomaly space."""

    def __init__(
        self,
        data_phys: np.ndarray,
        dates: np.ndarray,
        clim_tr: np.ndarray,
        anom_std: np.ndarray,
        rollout_steps: int = 1,
    ):
        self.rollout_steps = rollout_steps
        self.data_phys = data_phys.astype(np.float32)
        self.doys = np.array([date_to_doy365(d) for d in dates], dtype=np.int64)
        self.doy_sc = np.array([doy_sincos(int(d) + 1) for d in self.doys], dtype=np.float32)
        data_tr = transform_physical(self.data_phys)
        self.z = ((data_tr - clim_tr[self.doys]) / anom_std[None, :, None, None]).astype(np.float32)
        self.valid_idx = np.arange(T_IN - 1, len(dates) - rollout_steps)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int):
        t = int(self.valid_idx[idx])
        z_in = self.z[t - T_IN + 1 : t + 1]  # [3, 6, 32, 36]
        z_tgt = self.z[t + 1 : t + 1 + self.rollout_steps]  # [steps, 6, 32, 36]
        sc_seq = self.doy_sc[t : t + self.rollout_steps]  # [steps, 2]
        doy_tgt = self.doys[t + 1 : t + 1 + self.rollout_steps]  # [steps]
        phys_tgt = self.data_phys[t + 1 : t + 1 + self.rollout_steps]  # [steps, 6, 32, 36]
        return (
            torch.from_numpy(z_in),
            torch.from_numpy(sc_seq),
            torch.from_numpy(doy_tgt),
            torch.from_numpy(z_tgt),
            torch.from_numpy(phys_tgt),
        )


class GPURolloutBatcher:
    """Holds the full ~171 MB climate tensor directly in GPU/device memory for zero-bottleneck batching."""

    def __init__(
        self,
        data_phys: np.ndarray,
        dates: np.ndarray,
        clim_tr: np.ndarray,
        anom_std: np.ndarray,
        device: torch.device,
        batch_size: int = 64,
        rollout_steps: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.device = device
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.shuffle = shuffle
        self.drop_last = drop_last

        doys = np.array([date_to_doy365(d) for d in dates], dtype=np.int64)
        doy_sc = np.array([doy_sincos(int(d) + 1) for d in doys], dtype=np.float32)
        data_tr = transform_physical(data_phys)
        z = ((data_tr - clim_tr[doys]) / anom_std[None, :, None, None]).astype(np.float32)

        self.z = torch.from_numpy(z).to(device)
        self.doy_sc = torch.from_numpy(doy_sc).to(device)
        self.doys = torch.from_numpy(doys).to(device)
        self.data_phys = torch.from_numpy(data_phys.astype(np.float32)).to(device)
        self.valid_idx = torch.arange(T_IN - 1, len(dates) - rollout_steps, device=device, dtype=torch.long)

        self.in_offsets = torch.arange(-T_IN + 1, 1, device=device, dtype=torch.long)
        self.set_rollout_steps(rollout_steps)

    def set_rollout_steps(self, rollout_steps: int) -> None:
        self.rollout_steps = rollout_steps
        n_total = self.z.shape[0]
        self.valid_idx = torch.arange(T_IN - 1, n_total - rollout_steps, device=self.device, dtype=torch.long)
        self.sc_offsets = torch.arange(0, rollout_steps, device=self.device, dtype=torch.long)
        self.tgt_offsets = torch.arange(1, rollout_steps + 1, device=self.device, dtype=torch.long)

    def __len__(self) -> int:
        n = len(self.valid_idx)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.valid_idx)
        perm = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            t = self.valid_idx[perm[start : start + self.batch_size]]
            z_in = self.z[t[:, None] + self.in_offsets[None, :]]
            sc_seq = self.doy_sc[t[:, None] + self.sc_offsets[None, :]]
            tgt_idx = t[:, None] + self.tgt_offsets[None, :]
            doy_tgt = self.doys[tgt_idx]
            z_tgt = self.z[tgt_idx]
            phys_tgt = self.data_phys[tgt_idx]
            yield z_in, sc_seq, doy_tgt, z_tgt, phys_tgt
