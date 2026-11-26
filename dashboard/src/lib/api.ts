import { NC, NLAT, NLON } from "./contract";

export interface HealthResponse {
  status: string;
  model_name: string;
  latency_ms: number;
}

export interface SimulateRequest {
  init_states: number[][][][];
  doy_sincos: [number, number];
  steps: number;
  perturbation: number[][][] | null;
}

export interface SimulateResponse {
  rollout: number[][][][];
  inference_ms: number;
  wall_ms: number;
  source: "live" | "mock";
}

export async function fetchHealth(baseUrl = "/api"): Promise<HealthResponse> {
  const t0 = performance.now();
  const res = await fetch(`${baseUrl}/health`, { method: "POST" });
  if (!res.ok) {
    throw new Error(`Health HTTP ${res.status}`);
  }
  const data = (await res.json()) as { status: string; model_name: string };
  return {
    ...data,
    latency_ms: performance.now() - t0,
  };
}

export async function postSimulate(
  req: SimulateRequest,
  baseUrl = "/api",
): Promise<SimulateResponse> {
  const t0 = performance.now();
  const res = await fetch(`${baseUrl}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`Simulate HTTP ${res.status}`);
  }
  const data = (await res.json()) as { rollout: number[][][][]; inference_ms: number };
  return {
    rollout: data.rollout,
    inference_ms: data.inference_ms,
    wall_ms: performance.now() - t0,
    source: "live",
  };
}

/**
 * Local fallback used only when serve.py is unreachable or user selects Offline Demo mode.
 * Propagates the initial perturbation outward gently with eastward advection + damping
 * so UI timeline and maps remain inspectable without a running backend.
 */
export function mockSimulate(req: SimulateRequest): SimulateResponse {
  const t0 = performance.now();
  const last = req.init_states[req.init_states.length - 1];
  const base = last.map((ch) => ch.map((row) => [...row]));

  let state = last.map((ch, c) =>
    ch.map((row, i) =>
      row.map((v, j) => v + (req.perturbation ? req.perturbation[c][i][j] : 0)),
    ),
  );

  const rollout: number[][][][] = [];
  for (let s = 0; s < req.steps; s++) {
    const next: number[][][] = Array.from({ length: NC }, () =>
      Array.from({ length: NLAT }, () => new Array<number>(NLON).fill(0)),
    );
    for (let c = 0; c < NC; c++) {
      for (let i = 0; i < NLAT; i++) {
        for (let j = 0; j < NLON; j++) {
          const im = Math.max(0, i - 1);
          const ip = Math.min(NLAT - 1, i + 1);
          const jm = Math.max(0, j - 1);
          const jp = Math.min(NLON - 1, j + 1);
          // 5-point diffusion + slight eastward drift of anomaly relative to base
          const anomC = state[c][i][j] - base[c][i][j];
          const anomLap =
            0.25 *
            ((state[c][im][j] - base[c][im][j]) +
              (state[c][ip][j] - base[c][ip][j]) +
              (state[c][i][jm] - base[c][i][jm]) +
              (state[c][i][jp] - base[c][i][jp]));
          const drifted = base[c][i][j] + 0.55 * anomC + 0.42 * anomLap;
          const val = c === 1 ? Math.max(0, drifted) : c === 0 ? Math.min(333.15, Math.max(233.15, drifted)) : drifted;
          next[c][i][j] = val;
        }
      }
    }
    state = next;
    rollout.push(next);
  }
  const ms = performance.now() - t0;
  return {
    rollout,
    inference_ms: ms,
    wall_ms: ms,
    source: "mock",
  };
}
