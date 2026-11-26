"""Automated Round-1 submission verification harness for Multiverse of Monsoon."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon_r1 import contract as C  # noqa: E402
from monsoon_r1 import schema_validator, scoring as S  # noqa: E402


def _post_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> tuple[dict, float]:
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
    else:
        data = b""
        headers = {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, (time.perf_counter() - t0) * 1000.0


def _base_init_states(doy: int, artifacts_dir: Path) -> np.ndarray:
    clim_p = artifacts_dir / "clim.npy"
    if clim_p.exists():
        clim = np.load(clim_p).astype(np.float32)
        idx = [(doy - 2) % 365, (doy - 1) % 365, doy % 365]
        x = clim[idx].copy()
        x[:, 1] = np.expm1(np.maximum(x[:, 1], 0.0))
        return x
    x = np.zeros((C.T_IN, C.NC, C.NLAT, C.NLON), dtype=np.float32)
    for c, v in enumerate((298.0, 2.5, 3.0, 2.0, 101000.0, 40.0)):
        x[:, c] = v
    return x


def _git_lines(args: list[str]) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _check_rss_gb() -> float | None:
    try:
        import psutil  # type: ignore

        rss = psutil.Process().memory_info().rss
        for p in psutil.process_iter(["cmdline", "memory_info"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if "serve.py" in cmd or "uvicorn" in cmd:
                rss = max(rss, p.info["memory_info"].rss)
        return round(rss / (1024**3), 4)
    except Exception:
        return None


def _run_hygiene_and_anticheat(max_repo_gb: float, require_weights: bool) -> dict:
    banned = re.compile(
        "|".join(["open" + "-meteo", "openweather" + "map", "weather" + "bench2", "arco" + "-era5", "cds" + "api", "imd" + "lib"]),
        re.IGNORECASE,
    )
    banned_hits: list[str] = []
    for p in ROOT.rglob("*.py"):
        if any(part in p.parts for part in (".git", "node_modules", ".venv", "venv")):
            continue
        if banned.search(p.read_text(errors="ignore")):
            banned_hits.append(str(p.relative_to(ROOT)))

    tracked = _git_lines(["ls-files"])
    bad_ext = (".nc", ".nc4", ".grib", ".grib2", ".zarr")
    tracked_data_files = [f for f in tracked if f.lower().endswith(bad_ext)]

    lfs_files = _git_lines(["lfs", "ls-files", "-n"])
    onnx_files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "artifacts").glob("*.onnx")) if (ROOT / "artifacts").exists() else []
    onnx_in_lfs = [f for f in onnx_files if f in lfs_files]
    weights_present = len(onnx_files) > 0
    weights_ok = (not require_weights) or (weights_present and len(onnx_in_lfs) == len(onnx_files))

    total_bytes = 0
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in p.parts for part in (".git", "node_modules", ".venv", "venv", "data")):
            continue
        total_bytes += p.stat().st_size
    repo_gb = total_bytes / (1024**3)

    return {
        "anti_cheat_ok": len(banned_hits) == 0,
        "banned_hits": banned_hits,
        "tracked_data_files": tracked_data_files,
        "no_tracked_data_ok": len(tracked_data_files) == 0,
        "repo_size_gb": round(repo_gb, 4),
        "repo_size_ok": repo_gb <= max_repo_gb,
        "weights_present": weights_present,
        "onnx_files": onnx_files,
        "onnx_in_lfs": onnx_in_lfs,
        "weights_ok": weights_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify Round-1 submission contract, latency, causality, and repo hygiene")
    ap.add_argument("--url", type=str, default="http://localhost:8000")
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--days", type=int, default=C.SCORE_STEPS)
    ap.add_argument("--max-latency", type=float, default=5.0)
    ap.add_argument("--max-repo-gb", type=float, default=1.0)
    ap.add_argument("--require-weights", action="store_true")
    args = ap.parse_args(argv)

    url = args.url.rstrip("/")
    health_resp, health_ms = _post_json(f"{url}/health", None)
    health_ok = health_resp.get("status") == "ready"

    schema_rc = schema_validator.main([url])
    schema_ok = schema_rc == 0

    init_states = _base_init_states(180, ROOT / "artifacts")
    sc = C.doy_sincos(180)
    base_payload = {
        "init_states": C.encode_list(init_states),
        "doy_sincos": sc,
        "steps": args.days,
        "perturbation": None,
    }
    base_resp, _ = _post_json(f"{url}/simulate", base_payload)
    base_arr = C.decode_array(base_resp["rollout"], (args.days, C.NC, C.NLAT, C.NLON))
    validity_errors = [f"baseline: {e}" for e in S.validity(base_arr, args.days)]

    latencies_ms: list[float] = []
    null_checks: list[bool] = []
    causality_summary: dict[str, dict] = {}
    cycle = ("null", "warm_ocean", "monsoon_stall")
    for idx in range(args.calls):
        shock_name = cycle[idx % len(cycle)]
        pert = S.make_shock(shock_name, init_states[-1], S.STD_DEFAULT)
        payload = {
            "init_states": C.encode_list(init_states),
            "doy_sincos": sc,
            "steps": args.days,
            "perturbation": C.encode_list(pert),
        }
        resp, ms = _post_json(f"{url}/simulate", payload)
        arr = C.decode_array(resp["rollout"], (args.days, C.NC, C.NLAT, C.NLON))
        errs = S.validity(arr, args.days)
        if errs:
            validity_errors.extend(f"call[{idx}]({shock_name}): {e}" for e in errs)
        if shock_name == "null":
            null_checks.append(S.null_consistent(base_arr, arr, S.STD_DEFAULT))
        elif shock_name not in causality_summary:
            box = C.BOX_BAY_OF_BENGAL if shock_name == "warm_ocean" else C.BOX_ARABIAN_SEA
            causality_summary[shock_name] = S.causality_checks(base_arr, arr, box, S.STD_DEFAULT)
        latencies_ms.append(ms)

    max_lat_s = max(latencies_ms) / 1000.0 if latencies_ms else 0.0
    mean_lat_s = (sum(latencies_ms) / len(latencies_ms)) / 1000.0 if latencies_ms else 0.0
    rss_gb = _check_rss_gb()
    rss_ok = True if rss_gb is None else rss_gb <= 6.0

    hygiene = _run_hygiene_and_anticheat(args.max_repo_gb, args.require_weights)
    null_ok = bool(null_checks) and all(null_checks)
    validity_ok = len(validity_errors) == 0
    latency_ok = max_lat_s <= args.max_latency

    all_ok = all(
        [
            health_ok,
            schema_ok,
            validity_ok,
            null_ok,
            latency_ok,
            rss_ok,
            hygiene["anti_cheat_ok"],
            hygiene["no_tracked_data_ok"],
            hygiene["repo_size_ok"],
            hygiene["weights_ok"],
        ]
    )

    report = {
        "ok": all_ok,
        "health": health_resp,
        "health_ms": round(health_ms, 2),
        "schema_validator_ok": schema_ok,
        "calls": len(latencies_ms),
        "max_latency_s": round(max_lat_s, 3),
        "mean_latency_s": round(mean_lat_s, 3),
        "latency_ok": latency_ok,
        "validity_ok": validity_ok,
        "validity_errors": validity_errors,
        "null_ok": null_ok,
        "causality_summary": causality_summary,
        "rss_gb": rss_gb,
        "rss_ok": rss_ok,
        "hygiene": hygiene,
    }
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
