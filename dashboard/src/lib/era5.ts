import { NC, NLAT, NLON, T_IN } from "./contract";

export interface Era5Meta {
  source: string;
  year: number;
  dates: string[];
  shape: [number, number, number, number];
  channels: string[];
  channelMin: number[];
  channelMax: number[];
}

export interface Era5YearDataset {
  meta: Era5Meta;
  data: Uint16Array;
}

const EXPECTED_DAYS = 365;
const EXPECTED_LEN = EXPECTED_DAYS * NC * NLAT * NLON;

export async function loadEra5YearDataset(): Promise<Era5YearDataset | null> {
  try {
    const [metaRes, binRes] = await Promise.all([
      fetch("/era5_year.json"),
      fetch("/era5_year.bin"),
    ]);
    if (!metaRes.ok || !binRes.ok) return null;

    const meta = (await metaRes.json()) as Era5Meta;
    const buf = await binRes.arrayBuffer();
    const data = new Uint16Array(buf);
    if (data.length !== EXPECTED_LEN || meta.dates.length !== EXPECTED_DAYS) {
      return null;
    }
    return { meta, data };
  } catch {
    return null;
  }
}

export function getEra5InitStates(
  ds: Era5YearDataset,
  doy: number,
): { initStates: number[][][][]; date: string } {
  const idx = Math.min(EXPECTED_DAYS - 1, Math.max(0, doy - 1));
  const dayIndices = [Math.max(0, idx - 2), Math.max(0, idx - 1), idx];
  const planeStride = NLAT * NLON;
  const chStride = NC * planeStride;

  const initStates: number[][][][] = [];
  for (let t = 0; t < T_IN; t++) {
    const dayOffset = dayIndices[t] * chStride;
    const chGrid: number[][][] = [];
    for (let c = 0; c < NC; c++) {
      const cMin = ds.meta.channelMin[c];
      const span = ds.meta.channelMax[c] - cMin;
      const chOffset = dayOffset + c * planeStride;
      const rows: number[][] = [];
      for (let i = 0; i < NLAT; i++) {
        const rowOffset = chOffset + i * NLON;
        const row = new Array<number>(NLON);
        for (let j = 0; j < NLON; j++) {
          row[j] = cMin + (ds.data[rowOffset + j] / 65535.0) * span;
        }
        rows.push(row);
      }
      chGrid.push(rows);
    }
    initStates.push(chGrid);
  }

  return { initStates, date: ds.meta.dates[idx] };
}
