"""Export the latest full 365-day year from a local ERA5 NetCDF4/HDF5 dataset for the offline dashboard."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import h5py
import numpy as np

CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
DAYS = 365
NC = 6
NLAT = 32
NLON = 36


def _decode_dates(time_ds: h5py.Dataset) -> list[dt.date]:
    raw_units = time_ds.attrs.get("units", b"days since 2005-01-01 00:00:00")
    if isinstance(raw_units, bytes):
        raw_units = raw_units.decode("utf-8")
    elif isinstance(raw_units, np.ndarray):
        raw_units = raw_units.item().decode("utf-8") if isinstance(raw_units.item(), bytes) else str(raw_units.item())
    m = re.search(r"days\s+since\s+(\d{4}-\d{2}-\d{2})", str(raw_units), re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported time units attribute: {raw_units!r}")
    origin = dt.date.fromisoformat(m.group(1))
    vals = np.asarray(time_ds[:], dtype=np.int64)
    return [origin + dt.timedelta(days=int(v)) for v in vals]


def export_dataset(nc_path: Path, out_dir: Path) -> None:
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    with h5py.File(nc_path, "r") as f:
        dates = _decode_dates(f["time"])
        # Pick the latest calendar year with >= 365 days; otherwise fall back to last 365 days.
        year_indices: dict[int, list[int]] = {}
        for idx, d in enumerate(dates):
            year_indices.setdefault(d.year, []).append(idx)

        selected_year = None
        selected_indices: list[int] | None = None
        for yr in sorted(year_indices.keys(), reverse=True):
            if len(year_indices[yr]) >= DAYS:
                selected_year = yr
                selected_indices = year_indices[yr][:DAYS]
                break

        if selected_indices is None:
            if len(dates) < DAYS:
                raise ValueError(f"Dataset has only {len(dates)} days; need at least {DAYS}")
            selected_indices = list(range(len(dates) - DAYS, len(dates)))
            selected_year = dates[selected_indices[-1]].year

        slice_arr = np.asarray(f["data"][selected_indices], dtype=np.float32)

    if slice_arr.shape != (DAYS, NC, NLAT, NLON):
        raise ValueError(f"Expected shape {(DAYS, NC, NLAT, NLON)}, got {slice_arr.shape}")

    slice_arr[:, 1] = np.maximum(slice_arr[:, 1], 0.0)
    slice_arr = np.nan_to_num(slice_arr, nan=0.0, posinf=0.0, neginf=0.0)

    c_min: list[float] = []
    c_max: list[float] = []
    max_err: list[float] = []
    q = np.zeros_like(slice_arr, dtype="<u2")

    for c in range(NC):
        vmin = float(slice_arr[:, c].min())
        vmax = float(slice_arr[:, c].max())
        span = max(vmax - vmin, 1e-6)
        norm = np.clip((slice_arr[:, c] - vmin) / span, 0.0, 1.0)
        q[:, c] = np.round(norm * 65535.0).astype("<u2")
        recon = vmin + (q[:, c].astype(np.float32) / 65535.0) * span
        c_min.append(round(vmin, 6))
        c_max.append(round(vmin + span, 6))
        max_err.append(float(np.abs(recon - slice_arr[:, c]).max()))

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "era5_year.bin"
    json_path = out_dir / "era5_year.json"

    q.tofile(bin_path)
    selected_dates = [dates[i].isoformat() for i in selected_indices]
    meta = {
        "source": nc_path.name,
        "year": int(selected_year),
        "dates": selected_dates,
        "shape": [DAYS, NC, NLAT, NLON],
        "channels": CHANNELS,
        "channelMin": c_min,
        "channelMax": c_max,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"Exported ERA5 year {selected_year} ({selected_dates[0]} .. {selected_dates[-1]}) -> "
        f"{bin_path} ({bin_path.stat().st_size:,} bytes), {json_path} ({json_path.stat().st_size:,} bytes)"
    )
    for c, ch in enumerate(CHANNELS):
        print(f"  {ch:4s}: min={c_min[c]:.3f} max={c_max[c]:.3f} max_quant_err={max_err[c]:.5f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc", type=Path, default=Path("temp/india_climate_train.nc"))
    ap.add_argument("--out", type=Path, default=Path("dashboard/public"))
    args = ap.parse_args()
    export_dataset(args.nc, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
