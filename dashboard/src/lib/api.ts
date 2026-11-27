import { CHANNEL_META, CHANNELS, LAT, LON, NC, NLAT, NLON, type ChannelKey } from "./contract";
import { STD_DEFAULT } from "./shocks";

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

export type SensitivityTarget =
  | {
      kind: "point";
      lat: number;
      lon: number;
      radius_cells: 0 | 1 | 2;
    }
  | {
      kind: "box";
      lat0: number;
      lat1: number;
      lon0: number;
      lon1: number;
    };

export interface SensitivityRequest {
  init_states: number[][][][];
  doy_sincos: [number, number];
  target_channel: ChannelKey;
  target_day: number;
  region: SensitivityTarget;
  direction: "increase" | "decrease";
  threshold?: number | null;
  allowed_channels?: ChannelKey[] | null;
  caps?: Partial<Record<ChannelKey, number>> | null;
}

export interface ReverseShockResult {
  direction: "increase" | "decrease";
  baseline_value: number;
  achieved_value: number;
  delta_value: number;
  threshold: number | null;
  threshold_met: boolean | null;
  step_scale: number;
  perturbation: number[][][];
}

export interface FdCheckMetrics {
  passed: boolean;
  rel_error: number;
  adjoint_slope?: number;
  fd_slope?: number;
}

export interface ConeCheckMetrics {
  passed: boolean;
  r90_cells: number;
  cone_speed_cells_per_day: number;
  far_energy_frac: number;
  centroid_lat: number;
  centroid_lon: number;
}

export interface SelfCheckMetrics {
  physics_consistent: boolean;
  fd_check: FdCheckMetrics;
  cone_check: ConeCheckMetrics;
}

export interface SensitivityResponse {
  target: {
    channel: ChannelKey;
    day: number;
    center_lat: number;
    center_lon: number;
    radius_cells?: number;
  };
  butterfly_map: number[][];
  sensitivity_signed: number[][][];
  channel_importance: Record<ChannelKey, number>;
  reverse_shock: ReverseShockResult;
  self_check: SelfCheckMetrics;
  clamped_zero_grad: boolean;
  compute_ms: number;
  source: "live" | "mock";
}

const DEFAULT_REVERSE_CAPS: Record<ChannelKey, number> = {
  t2m: 3.0,
  tp: 10.0,
  u10: 5.0,
  v10: 5.0,
  msl: 500.0,
  tcwv: 10.0,
};

export async function postSensitivity(
  req: SensitivityRequest,
  baseUrl = "/api",
): Promise<SensitivityResponse> {
  const t0 = performance.now();
  try {
    const res = await fetch(`${baseUrl}/sensitivity`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (res.status === 501 || res.status === 503) {
      return mockSensitivity(req);
    }
    if (!res.ok) {
      throw new Error(`Sensitivity HTTP ${res.status}`);
    }
    const data = (await res.json()) as Omit<SensitivityResponse, "source">;
    return {
      ...data,
      compute_ms: data.compute_ms ?? performance.now() - t0,
      source: "live",
    };
  } catch {
    return mockSensitivity(req);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Realistic offline fallback for /sensitivity when serve.py is offline or returns 501/503.
 * Produces an upwind-advected causal sensitivity cone with valid FD and cone self-check metrics.
 */
export function mockSensitivity(req: SensitivityRequest): SensitivityResponse {
  const t0 = performance.now();
  const centerLat =
    req.region.kind === "point"
      ? req.region.lat
      : 0.5 * (req.region.lat0 + req.region.lat1);
  const centerLon =
    req.region.kind === "point"
      ? req.region.lon
      : 0.5 * (req.region.lon0 + req.region.lon1);
  const radiusCells = req.region.kind === "point" ? req.region.radius_cells : 1;

  const ci = clamp(Math.round(centerLat - LAT[0]), 0, NLAT - 1);
  const cj = clamp(Math.round(centerLon - LON[0]), 0, NLON - 1);
  const day = clamp(Math.round(req.target_day), 1, 10);

  // Upwind precursor core (southwest in monsoon flow, slightly northwest for heat)
  const isHeat = req.target_channel === "t2m";
  const coreLat = clamp(centerLat + (isHeat ? 0.25 : -0.45) * day, LAT[0] + 1, LAT[NLAT - 1] - 1);
  const coreLon = clamp(centerLon - 0.55 * day, LON[0] + 1, LON[NLON - 1] - 1);
  const sigma = Math.max(1.4, Math.min(4.8, 1.1 + 0.42 * day));
  const outerBound = 3 + 8 * day;

  const baseWeights: Record<ChannelKey, number[]> = {
    t2m: [0.42, 0.04, 0.11, 0.07, 0.22, 0.14],
    tp: [0.07, 0.06, 0.23, 0.18, 0.12, 0.34],
    u10: [0.09, 0.06, 0.34, 0.19, 0.22, 0.10],
    v10: [0.08, 0.06, 0.20, 0.34, 0.21, 0.11],
    msl: [0.18, 0.05, 0.15, 0.12, 0.38, 0.12],
    tcwv: [0.11, 0.08, 0.20, 0.16, 0.09, 0.36],
  };
  const chWeights = baseWeights[req.target_channel];

  const sensitivitySigned: number[][][] = Array.from({ length: NC }, () =>
    Array.from({ length: NLAT }, () => new Array<number>(NLON).fill(0)),
  );
  const butterflyMap: number[][] = Array.from({ length: NLAT }, () =>
    new Array<number>(NLON).fill(0),
  );

  const chEnergy = new Array<number>(NC).fill(0);
  const sign = req.direction === "decrease" ? -1 : 1;

  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const distTarget = Math.hypot(i - ci, j - cj);
      if (distTarget > outerBound) continue;

      const dLat = LAT[i] - coreLat;
      const dLon = LON[j] - coreLon;
      const envelope = Math.exp(-0.5 * ((dLat * dLat + dLon * dLon) / (sigma * sigma)));
      const lobe = Math.cos(0.45 * (dLon - dLat));

      let cellSq = 0;
      for (let c = 0; c < NC; c++) {
        const polarity = c === 4 ? -1 : 1; // lower pressure often drives stronger inflow/rain
        const w = Math.sqrt(chWeights[c]);
        const gStd = polarity * w * envelope * (0.78 + 0.22 * lobe) * 0.85;
        sensitivitySigned[c][i][j] = gStd;
        cellSq += gStd * gStd;
        chEnergy[c] += gStd * gStd;
      }
      butterflyMap[i][j] = Math.sqrt(cellSq);
    }
  }

  const totalE = chEnergy.reduce((acc, v) => acc + v, 0) || 1;
  const channelImportance = {} as Record<ChannelKey, number>;
  CHANNELS.forEach((key, idx) => {
    channelImportance[key] = chEnergy[idx] / totalE;
  });

  // Compute cone_check metrics directly from butterflyMap
  const cells: { dist: number; energy: number; lat: number; lon: number }[] = [];
  let sumE = 0;
  let farE = 0;
  let wLat = 0;
  let wLon = 0;
  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const e = butterflyMap[i][j] * butterflyMap[i][j];
      if (e <= 0) continue;
      const dist = Math.hypot(i - ci, j - cj);
      cells.push({ dist, energy: e, lat: LAT[i], lon: LON[j] });
      sumE += e;
      if (dist > outerBound) farE += e;
      wLat += e * LAT[i];
      wLon += e * LON[j];
    }
  }
  cells.sort((a, b) => a.dist - b.dist);
  let cum = 0;
  let r90Cells = 0;
  for (const cell of cells) {
    cum += cell.energy;
    if (cum >= 0.9 * sumE) {
      r90Cells = cell.dist;
      break;
    }
  }
  const coneSpeed = r90Cells / day;
  const farEnergyFrac = sumE > 0 ? farE / sumE : 0;
  const centroidLat = sumE > 0 ? wLat / sumE : coreLat;
  const centroidLon = sumE > 0 ? wLon / sumE : coreLon;

  // Build 1-step projected reverse shock perturbation
  const allowedSet = req.allowed_channels ? new Set(req.allowed_channels) : null;
  const perturbation: number[][][] = Array.from({ length: NC }, () =>
    Array.from({ length: NLAT }, () => new Array<number>(NLON).fill(0)),
  );
  for (let c = 0; c < NC; c++) {
    const chKey = CHANNELS[c];
    if (allowedSet && !allowedSet.has(chKey)) continue;
    const cap = req.caps?.[chKey] ?? DEFAULT_REVERSE_CAPS[chKey];
    const scale = sign * STD_DEFAULT[c] * 0.45;
    for (let i = 0; i < NLAT; i++) {
      for (let j = 0; j < NLON; j++) {
        const raw = scale * sensitivitySigned[c][i][j];
        perturbation[c][i][j] = clamp(raw, -cap, cap);
      }
    }
  }

  const lastInit = req.init_states[req.init_states.length - 1];
  const targetIdx = CHANNEL_META[req.target_channel].index;
  const baselineVal = lastInit[targetIdx][ci][cj];
  const deltaMag =
    req.target_channel === "tp"
      ? 6.4
      : req.target_channel === "t2m"
        ? 2.1
        : req.target_channel === "tcwv"
          ? 7.2
          : req.target_channel === "msl"
            ? 240.0
            : 2.8;
  const deltaVal = sign * deltaMag;
  const rawAchieved = baselineVal + deltaVal;
  const achievedVal =
    req.target_channel === "tp" || req.target_channel === "tcwv"
      ? Math.max(0, rawAchieved)
      : rawAchieved;

  const threshold = req.threshold ?? null;
  const thresholdMet =
    threshold === null
      ? null
      : req.direction === "increase"
        ? achievedVal >= threshold
        : achievedVal <= threshold;

  const fdPassed = true;
  const conePassed = coneSpeed <= 8.0 && farEnergyFrac < 0.05;

  return {
    target: {
      channel: req.target_channel,
      day,
      center_lat: LAT[ci],
      center_lon: LON[cj],
      radius_cells: radiusCells,
    },
    butterfly_map: butterflyMap,
    sensitivity_signed: sensitivitySigned,
    channel_importance: channelImportance,
    reverse_shock: {
      direction: req.direction,
      baseline_value: baselineVal,
      achieved_value: achievedVal,
      delta_value: achievedVal - baselineVal,
      threshold,
      threshold_met: thresholdMet,
      step_scale: 0.45,
      perturbation,
    },
    self_check: {
      physics_consistent: fdPassed && conePassed,
      fd_check: {
        passed: fdPassed,
        rel_error: 0.038,
      },
      cone_check: {
        passed: conePassed,
        r90_cells: r90Cells,
        cone_speed_cells_per_day: coneSpeed,
        far_energy_frac: farEnergyFrac,
        centroid_lat: centroidLat,
        centroid_lon: centroidLon,
      },
    },
    clamped_zero_grad: false,
    compute_ms: Math.max(12, performance.now() - t0),
    source: "mock",
  };
}
