export const CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"] as const;
export type ChannelKey = (typeof CHANNELS)[number];

export const NC = 6;
export const NLAT = 32;
export const NLON = 36;
export const T_IN = 3;
export const SCORE_STEPS = 10;

export const LAT: number[] = Array.from({ length: NLAT }, (_, i) => 6.5 + i);
export const LON: number[] = Array.from({ length: NLON }, (_, j) => 65.5 + j);

export const LAT_MIN = 6.0;
export const LAT_MAX = 38.0;
export const LON_MIN = 65.0;
export const LON_MAX = 101.0;

export interface GeoBox {
  id: string;
  name: string;
  lat0: number;
  lat1: number;
  lon0: number;
  lon1: number;
}

export const BOX_BAY_OF_BENGAL: GeoBox = {
  id: "bob",
  name: "Bay of Bengal",
  lat0: 8.0,
  lat1: 20.0,
  lon0: 82.0,
  lon1: 94.0,
};

export const BOX_ARABIAN_SEA: GeoBox = {
  id: "arabian",
  name: "Arabian Sea",
  lat0: 8.0,
  lat1: 20.0,
  lon0: 65.0,
  lon1: 75.0,
};

export interface ChannelMeta {
  index: number;
  key: ChannelKey;
  label: string;
  shortLabel: string;
  unit: string;
  displayUnit: (useCelsius: boolean) => string;
  toDisplay: (v: number, useCelsius: boolean) => number;
  format: (v: number, useCelsius: boolean) => string;
  range: (useCelsius: boolean) => [number, number];
  deltaMax: number;
}

export const CHANNEL_META: Record<ChannelKey, ChannelMeta> = {
  t2m: {
    index: 0,
    key: "t2m",
    label: "2m Air Temperature",
    shortLabel: "Temp (t2m)",
    unit: "K",
    displayUnit: (c) => (c ? "°C" : "K"),
    toDisplay: (v, c) => (c ? v - 273.15 : v),
    format: (v, c) => (c ? `${(v - 273.15).toFixed(1)} °C` : `${v.toFixed(1)} K`),
    range: (c) => (c ? [5, 42] : [278.15, 315.15]),
    deltaMax: 4.0,
  },
  tp: {
    index: 1,
    key: "tp",
    label: "Daily Precipitation",
    shortLabel: "Rain (tp)",
    unit: "mm/day",
    displayUnit: () => "mm/day",
    toDisplay: (v) => v,
    format: (v) => `${v.toFixed(1)} mm/d`,
    range: () => [0, 40],
    deltaMax: 12.0,
  },
  u10: {
    index: 2,
    key: "u10",
    label: "10m Eastward Wind",
    shortLabel: "U-Wind (u10)",
    unit: "m/s",
    displayUnit: () => "m/s",
    toDisplay: (v) => v,
    format: (v) => `${v.toFixed(1)} m/s`,
    range: () => [-12, 12],
    deltaMax: 5.0,
  },
  v10: {
    index: 3,
    key: "v10",
    label: "10m Northward Wind",
    shortLabel: "V-Wind (v10)",
    unit: "m/s",
    displayUnit: () => "m/s",
    toDisplay: (v) => v,
    format: (v) => `${v.toFixed(1)} m/s`,
    range: () => [-12, 12],
    deltaMax: 5.0,
  },
  msl: {
    index: 4,
    key: "msl",
    label: "Mean Sea-Level Pressure",
    shortLabel: "Pressure (msl)",
    unit: "Pa",
    displayUnit: () => "hPa",
    toDisplay: (v) => v / 100.0,
    format: (v) => `${(v / 100.0).toFixed(1)} hPa`,
    range: () => [998, 1020],
    deltaMax: 5.0,
  },
  tcwv: {
    index: 5,
    key: "tcwv",
    label: "Total Column Water Vapour",
    shortLabel: "Moisture (tcwv)",
    unit: "kg/m²",
    displayUnit: () => "kg/m²",
    toDisplay: (v) => v,
    format: (v) => `${v.toFixed(1)} kg/m²`,
    range: () => [10, 68],
    deltaMax: 10.0,
  },
};

export interface SeasonPreset {
  id: string;
  name: string;
  doy: number;
  note: string;
}

export const SEASON_PRESETS: SeasonPreset[] = [
  { id: "onset", name: "Monsoon Onset", doy: 160, note: "Jun 9 — SW flow builds over Arabian Sea" },
  { id: "peak", name: "Peak Monsoon", doy: 205, note: "Jul 24 — active trough & Western Ghats rain" },
  { id: "heatwave", name: "Pre-Monsoon Heatwave", doy: 135, note: "May 15 — hot northwest plains" },
  { id: "cyclone", name: "Post-Monsoon Cyclone", doy: 295, note: "Oct 22 — Bay of Bengal moisture surge" },
  { id: "winter", name: "Winter Disturbance", doy: 20, note: "Jan 20 — cool northern plains & dry trades" },
];

export function doySinCos(doy: number): [number, number] {
  const rad = (2.0 * Math.PI * doy) / 365.0;
  return [Math.sin(rad), Math.cos(rad)];
}

/**
 * Generates a physically plausible [3, 6, 32, 36] initial state window
 * in raw physical units for interactive testing and offline visualization.
 */
export function buildDefaultInitStates(doy: number): number[][][][] {
  const phase = (2.0 * Math.PI * (doy - 150)) / 365.0;
  const monsoonStrength = 0.5 + 0.5 * Math.cos((2.0 * Math.PI * (doy - 200)) / 365.0);
  const summerWarmth = Math.cos(phase);

  const states: number[][][][] = [];
  for (let t = 0; t < T_IN; t++) {
    const dayShift = (t - 2) * 0.35;
    const chGrid: number[][][] = Array.from({ length: NC }, () =>
      Array.from({ length: NLAT }, () => new Array<number>(NLON).fill(0)),
    );

    for (let i = 0; i < NLAT; i++) {
      const lat = LAT[i];
      const latNorm = (lat - 20.0) / 15.0;
      for (let j = 0; j < NLON; j++) {
        const lon = LON[j];
        const lonNorm = (lon - 80.0) / 18.0;

        // Rough topography proxy for cooler Himalayas / Tibetan plateau
        const himalaya = Math.exp(-((lat - 33.5) ** 2) / 18.0) * (lon > 73 ? 1.0 : 0.4);
        const wGhatsRain =
          Math.exp(-((lon - 74.0) ** 2) / 5.0) * (lat >= 9 && lat <= 20 ? 1.0 : 0.15);
        const bobRain =
          Math.exp(-((lat - 18.0) ** 2) / 28.0 - ((lon - 89.0) ** 2) / 36.0);

        // t2m (K)
        const baseTemp =
          299.5 +
          4.5 * summerWarmth * (0.4 + 0.6 * Math.max(0, latNorm)) -
          16.0 * himalaya -
          2.0 * Math.max(0, latNorm) * (1 - summerWarmth) +
          0.3 * Math.sin(lonNorm * 2.2 + dayShift);
        chGrid[0][i][j] = Math.min(326, Math.max(255, baseTemp));

        // tp (mm/day)
        const rainCore =
          monsoonStrength * (18.0 * wGhatsRain + 14.0 * bobRain + 4.5 * Math.exp(-((lat - 22) ** 2) / 35)) +
          1.2 * Math.max(0, Math.sin(latNorm * 2.5 + lonNorm * 1.8 + dayShift));
        chGrid[1][i][j] = Math.max(0.0, rainCore);

        // u10 (m/s) - SW monsoon vs winter NE trades
        const uSw = (monsoonStrength * 11.5 - 3.5) * Math.exp(-((lat - 14.0) ** 2) / 90.0);
        chGrid[2][i][j] = uSw + 1.2 * Math.cos(latNorm * 2.0 + dayShift * 0.5);

        // v10 (m/s)
        const vSw = (monsoonStrength * 6.5 - 2.5) * Math.exp(-((lon - 72.0) ** 2) / 120.0);
        chGrid[3][i][j] = vSw + 0.9 * Math.sin(lonNorm * 2.1 + dayShift * 0.5);

        // msl (Pa) - monsoon heat low over NW India
        const monsoonLow =
          -850.0 * monsoonStrength * Math.exp(-((lat - 25.5) ** 2) / 40.0 - ((lon - 74.0) ** 2) / 55.0);
        chGrid[4][i][j] = 101150.0 + monsoonLow + 180.0 * latNorm * (1 - monsoonStrength);

        // tcwv (kg/m2)
        const moisture =
          22.0 +
          32.0 * monsoonStrength * Math.exp(-((lat - 16.0) ** 2) / 140.0) -
          18.0 * himalaya +
          5.0 * bobRain;
        chGrid[5][i][j] = Math.max(5.0, moisture);
      }
    }
    states.push(chGrid);
  }
  return states;
}

/**
 * Simplified coastline polylines ([lon, lat]) for mainland India, Sri Lanka,
 * and neighboring coast context within 65..101E, 6..38N.
 */
export const INDIA_COAST_POLYLINES: [number, number][][] = [
  // Arabian Sea coast -> Kanyakumari -> Bay of Bengal -> Myanmar coast
  [
    [65.2, 25.2], [67.4, 24.8], [68.8, 23.6], [70.2, 22.8], [69.0, 22.2],
    [70.3, 20.9], [72.1, 21.7], [72.8, 21.1], [72.8, 19.0], [73.3, 17.0],
    [74.1, 15.0], [74.8, 13.0], [76.2, 10.0], [77.5, 8.08], [78.9, 9.2],
    [79.8, 10.3], [80.3, 13.1], [80.1, 15.8], [82.2, 16.9], [83.3, 17.7],
    [85.0, 19.3], [86.9, 20.8], [88.2, 21.6], [89.8, 21.9], [91.5, 22.4],
    [92.2, 21.2], [93.2, 19.5], [94.4, 17.5], [95.8, 16.2], [97.6, 16.4],
    [98.4, 13.5], [98.6, 10.0],
  ],
  // Sri Lanka
  [
    [80.0, 9.6], [81.2, 8.6], [81.8, 7.2], [81.2, 6.1], [80.2, 6.0],
    [79.7, 7.1], [79.8, 8.6], [80.0, 9.6],
  ],
  // Northern / Himalayan & NE border context (dashed reference)
  [
    [68.8, 23.8], [70.5, 26.8], [73.5, 29.8], [74.6, 32.5], [73.8, 34.4],
    [76.2, 35.8], [78.8, 34.8], [79.5, 31.2], [81.2, 30.0], [84.5, 28.6],
    [88.1, 27.9], [89.8, 27.6], [92.1, 27.8], [95.2, 28.9], [97.1, 28.1],
    [95.4, 26.2], [94.3, 24.1], [92.8, 23.0], [92.2, 24.8], [89.9, 25.5],
    [88.6, 24.4], [88.2, 21.6],
  ],
];

export function computeFieldStats(
  grid: number[][],
  channel: ChannelKey,
  useCelsius: boolean,
): { min: number; mean: number; max: number } {
  const meta = CHANNEL_META[channel];
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const v = meta.toDisplay(grid[i][j], useCelsius);
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      count++;
    }
  }
  return { min, mean: sum / count, max };
}
