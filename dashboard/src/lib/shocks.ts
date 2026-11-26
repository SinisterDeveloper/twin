import {
  BOX_ARABIAN_SEA,
  BOX_BAY_OF_BENGAL,
  CHANNEL_META,
  LAT,
  LON,
  NC,
  NLAT,
  NLON,
  type ChannelKey,
  type GeoBox,
} from "./contract";

export type ShockPreset = "none" | "null" | "warm_ocean" | "monsoon_stall" | "custom";

export interface CustomShockConfig {
  channel: ChannelKey;
  lat: number;
  lon: number;
  radiusDeg: number;
  intensity: number;
}

/** Matches monsoon_r1.scoring.STD_DEFAULT: [t2m, tp, u10, v10, msl, tcwv] */
export const STD_DEFAULT = [8.0, 8.0, 3.5, 2.5, 450.0, 20.0] as const;

export const SHOCK_PRESET_META: Record<
  ShockPreset,
  { label: string; short: string; description: string; box: GeoBox | null }
> = {
  none: {
    label: "None (Timeline Prime)",
    short: "Prime Only",
    description: "Unperturbed atmospheric rollout (perturbation = null).",
    box: null,
  },
  null: {
    label: "Null Shock (Zeros)",
    short: "Null (0)",
    description: "All-zero perturbation [6,32,36]. Must match Timeline Prime within 1e-3 std.",
    box: null,
  },
  warm_ocean: {
    label: "Warm Bay of Bengal",
    short: "Warm Ocean",
    description: "+3.0 K t2m & +15% tcwv over Bay of Bengal (8–20°N, 82–94°E).",
    box: BOX_BAY_OF_BENGAL,
  },
  monsoon_stall: {
    label: "Monsoon Stall (Arabian Sea)",
    short: "Monsoon Stall",
    description: "−40% u10 & v10 incoming monsoon winds over Arabian Sea (8–20°N, 65–75°E).",
    box: BOX_ARABIAN_SEA,
  },
  custom: {
    label: "Custom Localized Shock",
    short: "Custom",
    description: "Click anywhere on the map to place a localized Gaussian anomaly.",
    box: null,
  },
};

export function customIntensitySpec(ch: ChannelKey): {
  min: number;
  max: number;
  step: number;
  unit: string;
  defaultVal: number;
} {
  switch (ch) {
    case "t2m":
      return { min: -6, max: 6, step: 0.5, unit: "K", defaultVal: 3.0 };
    case "tp":
      return { min: -20, max: 40, step: 1, unit: "mm/d", defaultVal: 15.0 };
    case "u10":
    case "v10":
      return { min: -12, max: 12, step: 0.5, unit: "m/s", defaultVal: -5.0 };
    case "msl":
      return { min: -1500, max: 1500, step: 50, unit: "Pa", defaultVal: -600 };
    case "tcwv":
      return { min: -20, max: 25, step: 1, unit: "kg/m²", defaultVal: 10.0 };
  }
}

export function deltaToDisplay(deltaVal: number, channel: ChannelKey): number {
  if (channel === "msl") return deltaVal / 100.0;
  return deltaVal;
}

export function formatDelta(deltaVal: number, channel: ChannelKey, useCelsius: boolean): string {
  const disp = deltaToDisplay(deltaVal, channel);
  const sign = disp >= 0 ? "+" : "";
  const unit = channel === "t2m" ? (useCelsius ? "°C" : "K") : CHANNEL_META[channel].displayUnit(useCelsius);
  return `${sign}${disp.toFixed(2)} ${unit}`;
}

export function buildPerturbation(
  preset: ShockPreset,
  lastState: number[][][],
  custom: CustomShockConfig,
): number[][][] | null {
  if (preset === "none") return null;

  const pert: number[][][] = Array.from({ length: NC }, () =>
    Array.from({ length: NLAT }, () => new Array<number>(NLON).fill(0)),
  );
  if (preset === "null") return pert;

  if (preset === "warm_ocean") {
    for (let i = 0; i < NLAT; i++) {
      if (LAT[i] < BOX_BAY_OF_BENGAL.lat0 || LAT[i] > BOX_BAY_OF_BENGAL.lat1) continue;
      for (let j = 0; j < NLON; j++) {
        if (LON[j] < BOX_BAY_OF_BENGAL.lon0 || LON[j] > BOX_BAY_OF_BENGAL.lon1) continue;
        pert[0][i][j] = 3.0;
        pert[5][i][j] = 0.15 * lastState[5][i][j];
      }
    }
    return pert;
  }

  if (preset === "monsoon_stall") {
    for (let i = 0; i < NLAT; i++) {
      if (LAT[i] < BOX_ARABIAN_SEA.lat0 || LAT[i] > BOX_ARABIAN_SEA.lat1) continue;
      for (let j = 0; j < NLON; j++) {
        if (LON[j] < BOX_ARABIAN_SEA.lon0 || LON[j] > BOX_ARABIAN_SEA.lon1) continue;
        pert[2][i][j] = -0.4 * lastState[2][i][j];
        pert[3][i][j] = -0.4 * lastState[3][i][j];
      }
    }
    return pert;
  }

  // custom localized Gaussian shock
  const chIdx = CHANNEL_META[custom.channel].index;
  const r2Denom = Math.max(custom.radiusDeg * custom.radiusDeg, 0.1);
  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const dLat = LAT[i] - custom.lat;
      const dLon = LON[j] - custom.lon;
      const w = Math.exp(-0.5 * ((dLat * dLat + dLon * dLon) / r2Denom));
      pert[chIdx][i][j] = custom.intensity * w;
    }
  }
  return pert;
}

export function computeDeltaGrid(shocked: number[][][], baseline: number[][][]): number[][][] {
  return shocked.map((ch, c) =>
    ch.map((row, i) => row.map((val, j) => val - baseline[c][i][j])),
  );
}

export interface PhysicsMetrics {
  maxNormDelta: number;
  stable: boolean;
  nullOk: boolean | null;
  day1NearFrac: number | null;
  day4to8OutFrac: number | null;
  localized: boolean | null;
}

function buildNearMask(preset: ShockPreset, custom: CustomShockConfig): boolean[][] | null {
  if (preset === "none" || preset === "null") return null;
  const mask = Array.from({ length: NLAT }, () => new Array<boolean>(NLON).fill(false));
  const box = SHOCK_PRESET_META[preset].box;

  if (box) {
    // 3-cell dilation around box (matches monsoon_r1.scoring.DILATE = 3)
    for (let i = 0; i < NLAT; i++) {
      for (let j = 0; j < NLON; j++) {
        if (
          LAT[i] >= box.lat0 - 3 &&
          LAT[i] <= box.lat1 + 3 &&
          LON[j] >= box.lon0 - 3 &&
          LON[j] <= box.lon1 + 3
        ) {
          mask[i][j] = true;
        }
      }
    }
    return mask;
  }

  const nearR = custom.radiusDeg + 3.0;
  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      if (Math.hypot(LAT[i] - custom.lat, LON[j] - custom.lon) <= nearR) {
        mask[i][j] = true;
      }
    }
  }
  return mask;
}

export function computePhysicsMetrics(
  baseRollout: number[][][][],
  shockRollout: number[][][][],
  preset: ShockPreset,
  custom: CustomShockConfig,
): PhysicsMetrics {
  const steps = Math.min(baseRollout.length, shockRollout.length);
  let maxNormDelta = 0;
  const nearMask = buildNearMask(preset, custom);

  let day1TotalE = 0;
  let day1NearE = 0;
  let lateOutSum = 0;
  let lateTotalSum = 0;

  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < NLAT; i++) {
      for (let j = 0; j < NLON; j++) {
        let cellE = 0;
        for (let c = 0; c < NC; c++) {
          const z = Math.abs(shockRollout[s][c][i][j] - baseRollout[s][c][i][j]) / STD_DEFAULT[c];
          if (z > maxNormDelta) maxNormDelta = z;
          cellE += z * z;
        }
        if (nearMask) {
          if (s === 0) {
            day1TotalE += cellE;
            if (nearMask[i][j]) day1NearE += cellE;
          }
          if (s >= 3 && s <= 7) {
            lateTotalSum += cellE;
            if (!nearMask[i][j]) lateOutSum += cellE;
          }
        }
      }
    }
  }

  const day1NearFrac = nearMask && day1TotalE > 0 ? day1NearE / day1TotalE : null;
  const day4to8OutFrac = nearMask && lateTotalSum > 0 ? lateOutSum / lateTotalSum : null;

  return {
    maxNormDelta,
    stable: maxNormDelta <= 5.0,
    nullOk: preset === "null" ? maxNormDelta <= 1e-3 : null,
    day1NearFrac,
    day4to8OutFrac,
    localized: day1NearFrac !== null ? day1NearFrac >= 0.6 : null,
  };
}

export interface ImpactMetrics {
  baseHeavyRainCells: number;
  shockHeavyRainCells: number;
  baseHeatwaveCells: number;
  shockHeatwaveCells: number;
  peakTempC: number;
  baseMeanWindMs: number;
  shockMeanWindMs: number;
  windPowerDeltaPct: number;
  baseCropStressPct: number;
  shockCropStressPct: number;
}

export function computeImpactMetrics(
  baseRollout: number[][][][],
  shockRollout: number[][][][],
): ImpactMetrics {
  const steps = baseRollout.length;
  const hasShock = shockRollout.length === steps;

  let baseHeavy = 0;
  let shockHeavy = 0;
  let baseHeat = 0;
  let shockHeat = 0;
  let peakTempK = -Infinity;

  let baseWindSum = 0;
  let shockWindSum = 0;
  let baseWindPowSum = 0;
  let shockWindPowSum = 0;

  let plainsCount = 0;
  let baseCropStress = 0;
  let shockCropStress = 0;

  const totalCells = Math.max(1, steps * NLAT * NLON);

  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < NLAT; i++) {
      const inPlains = LAT[i] >= 15.0 && LAT[i] <= 29.0;
      for (let j = 0; j < NLON; j++) {
        const inPlainsLon = inPlains && LON[j] >= 72.0 && LON[j] <= 86.0;

        const bt2m = baseRollout[s][0][i][j];
        const btp = baseRollout[s][1][i][j];
        const bw = Math.hypot(baseRollout[s][2][i][j], baseRollout[s][3][i][j]);

        if (btp > 10.0) baseHeavy++;
        if (bt2m > 313.15) baseHeat++;
        baseWindSum += bw;
        baseWindPowSum += bw * bw * bw;

        if (inPlainsLon) {
          plainsCount++;
          if (bt2m > 307.15 && btp < 2.0) baseCropStress++;
        }

        const st2m = hasShock ? shockRollout[s][0][i][j] : bt2m;
        const stp = hasShock ? shockRollout[s][1][i][j] : btp;
        const sw = hasShock
          ? Math.hypot(shockRollout[s][2][i][j], shockRollout[s][3][i][j])
          : bw;

        if (st2m > peakTempK) peakTempK = st2m;
        if (stp > 10.0) shockHeavy++;
        if (st2m > 313.15) shockHeat++;
        shockWindSum += sw;
        shockWindPowSum += sw * sw * sw;

        if (inPlainsLon && st2m > 307.15 && stp < 2.0) shockCropStress++;
      }
    }
  }

  const windPowerDeltaPct =
    baseWindPowSum > 1e-6 ? ((shockWindPowSum - baseWindPowSum) / baseWindPowSum) * 100.0 : 0.0;

  return {
    baseHeavyRainCells: baseHeavy,
    shockHeavyRainCells: shockHeavy,
    baseHeatwaveCells: baseHeat,
    shockHeatwaveCells: shockHeat,
    peakTempC: peakTempK - 273.15,
    baseMeanWindMs: baseWindSum / totalCells,
    shockMeanWindMs: shockWindSum / totalCells,
    windPowerDeltaPct,
    baseCropStressPct: plainsCount > 0 ? (baseCropStress / plainsCount) * 100.0 : 0,
    shockCropStressPct: plainsCount > 0 ? (shockCropStress / plainsCount) * 100.0 : 0,
  };
}
