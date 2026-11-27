import { LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, NLAT, NLON, sampleBilinear, type ChannelKey } from "./contract";

export type ColormapName = "thermal" | "rain" | "wind" | "pressure" | "moisture" | "delta" | "butterfly";

type RGB = [number, number, number];

/**
 * Rich meteorological colormaps inspired by ECMWF / Windy / Doppler radar.
 */
const STOPS: Record<ColormapName, RGB[]> = {
  thermal: [
    [68, 28, 128],
    [34, 88, 214],
    [18, 172, 224],
    [44, 206, 132],
    [244, 222, 54],
    [246, 122, 28],
    [214, 24, 46],
  ],
  rain: [
    [11, 18, 32],
    [17, 46, 88],
    [14, 108, 178],
    [18, 188, 212],
    [42, 212, 92],
    [248, 226, 44],
    [246, 118, 24],
    [224, 34, 162],
  ],
  wind: [
    [22, 68, 188],
    [42, 148, 232],
    [82, 214, 198],
    [234, 236, 218],
    [248, 192, 72],
    [238, 96, 36],
    [178, 22, 54],
  ],
  pressure: [
    [52, 18, 114],
    [28, 86, 194],
    [22, 168, 182],
    [74, 208, 118],
    [248, 224, 62],
    [244, 128, 42],
  ],
  moisture: [
    [18, 18, 38],
    [28, 58, 138],
    [18, 136, 188],
    [34, 206, 158],
    [182, 236, 94],
    [250, 236, 146],
  ],
  delta: [
    [24, 72, 204],
    [78, 164, 246],
    [28, 34, 48],
    [248, 134, 62],
    [214, 28, 44],
  ],
  butterfly: [
    [9, 9, 14],
    [76, 29, 149],
    [192, 38, 158],
    [245, 158, 11],
    [255, 255, 255],
  ],
};

export function channelColormap(ch: ChannelKey): ColormapName {
  switch (ch) {
    case "t2m":
      return "thermal";
    case "tp":
      return "rain";
    case "u10":
    case "v10":
      return "wind";
    case "msl":
      return "pressure";
    case "tcwv":
      return "moisture";
  }
}

export function sampleColormap(name: ColormapName, tRaw: number): RGB {
  const stops = STOPS[name];
  const t = Math.min(1, Math.max(0, tRaw));
  const scaled = t * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const a = stops[idx];
  const b = stops[idx + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

export function cssLinearGradient(name: ColormapName): string {
  const stops = STOPS[name]
    .map((rgb, idx) => {
      const pct = Math.round((idx / (STOPS[name].length - 1)) * 100);
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]}) ${pct}%`;
    })
    .join(", ");
  return `linear-gradient(to right, ${stops})`;
}

export function renderFieldToCanvas(
  ctx: CanvasRenderingContext2D,
  scalarGrid: number[][],
  width: number,
  height: number,
  toDisplay: (v: number) => number,
  rMin: number,
  rMax: number,
  cmap: ColormapName,
  smooth: boolean,
): void {
  const img = ctx.createImageData(width, height);
  const data = img.data;
  const invSpan = 1.0 / Math.max(rMax - rMin, 1e-6);

  for (let py = 0; py < height; py++) {
    const lat = LAT_MAX - ((py + 0.5) / height) * (LAT_MAX - LAT_MIN);
    const nearestI = Math.min(NLAT - 1, Math.max(0, Math.floor((1 - (py + 0.5) / height) * NLAT)));

    for (let px = 0; px < width; px++) {
      const lon = LON_MIN + ((px + 0.5) / width) * (LON_MAX - LON_MIN);
      const nearestJ = Math.min(NLON - 1, Math.max(0, Math.floor(((px + 0.5) / width) * NLON)));

      const raw = smooth ? sampleBilinear(scalarGrid, lat, lon) : scalarGrid[nearestI][nearestJ];
      const norm = (toDisplay(raw) - rMin) * invSpan;
      const [r, g, b] = sampleColormap(cmap, norm);

      const off = (py * width + px) * 4;
      data[off] = r;
      data[off + 1] = g;
      data[off + 2] = b;
      data[off + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
