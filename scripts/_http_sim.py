"""Minimal stdlib+numpy HTTP client helpers for /health and /simulate."""
from __future__ import annotations

import base64
import json
import time
import urllib.request

import numpy as np


def http_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 30.0,
    method: str | None = None,
) -> tuple[dict, float]:
    """Send an HTTP JSON request and return (parsed_json, latency_ms)."""
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


def decode_arr(obj) -> np.ndarray:
    """Decode base64-packed or nested-list float32 array from server JSON."""
    if isinstance(obj, dict) and "b64" in obj:
        return np.frombuffer(base64.b64decode(obj["b64"]), dtype="<f4").reshape(obj["shape"]).astype(np.float32)
    return np.asarray(obj, dtype=np.float32)


def encode_arr(arr: np.ndarray) -> dict:
    """Encode a numpy array as base64 little-endian float32."""
    a = np.ascontiguousarray(arr, dtype="<f4")
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape)}


def wire_encode(arr: np.ndarray, wire: str):
    """Encode array for either 'lists' or 'b64' wire format."""
    if wire == "lists":
        return np.asarray(arr, dtype=np.float32).tolist()
    return encode_arr(arr)


def post_health(url: str, timeout: float = 30.0) -> tuple[dict, float]:
    """Call POST /health on the inference server."""
    return http_json(f"{url.rstrip('/')}/health", method="POST", timeout=timeout)


def post_simulate(url: str, payload: dict, timeout: float = 60.0) -> tuple[dict, float]:
    """Call POST /simulate on the inference server."""
    return http_json(f"{url.rstrip('/')}/simulate", payload, timeout=timeout)
