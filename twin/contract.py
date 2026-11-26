"""API contract constants and physical grid helpers (aligned with monsoon_r1.contract)."""
from __future__ import annotations

import base64
import numpy as np

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
NC, NLAT, NLON = 6, 32, 36
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0, dtype=np.float32)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0, dtype=np.float32)
T_IN = 3
MAX_STEPS = 14
TIME_LIMIT_S = 5.0
LEAD_DAYS = (1, 4, 8)
SCORE_STEPS = 10

BOX_BAY_OF_BENGAL = (8.0, 20.0, 82.0, 94.0)
BOX_ARABIAN_SEA = (8.0, 20.0, 65.0, 75.0)


def box_mask(box: tuple[float, float, float, float]) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    return ((LAT >= la0) & (LAT <= la1))[:, None] & ((LON >= lo0) & (LON <= lo1))[None, :]


def lat_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(LAT.astype(np.float64)))
    return (w / w.mean()).astype(np.float32)


def encode_list(a: np.ndarray) -> list:
    return np.asarray(a, dtype="f4").tolist()


def decode_array(x, expect_shape=None) -> np.ndarray:
    if isinstance(x, dict):
        if "b64" not in x or "shape" not in x:
            raise ValueError("array dict needs 'b64' and 'shape'")
        shape = tuple(int(s) for s in x["shape"])
        if any(s < 0 or s > 100000 for s in shape) or int(np.prod(shape)) > 20_000_000:
            raise ValueError("array too large")
        raw = base64.b64decode(x["b64"], validate=True)
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError("b64 length does not match shape")
        a = np.frombuffer(raw, dtype="<f4").reshape(shape).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if expect_shape is not None and a.shape != tuple(expect_shape):
        raise ValueError(f"shape {a.shape} != expected {tuple(expect_shape)}")
    return a


def doy_sincos(doy: int) -> list[float]:
    return [float(np.sin(2.0 * np.pi * doy / 365.0)), float(np.cos(2.0 * np.pi * doy / 365.0))]


def advance_doy_sincos(sin_v: float, cos_v: float, delta_days: int) -> tuple[float, float]:
    ang = 2.0 * np.pi * float(delta_days) / 365.0
    ca, sa = np.cos(ang), np.sin(ang)
    return float(sin_v * ca + cos_v * sa), float(cos_v * ca - sin_v * sa)


def sincos_to_doy_idx(sin_v: float, cos_v: float) -> int:
    ang = float(np.arctan2(sin_v, cos_v))
    if ang < 0:
        ang += 2.0 * np.pi
    return (int(round(ang * 365.0 / (2.0 * np.pi))) - 1) % 365
