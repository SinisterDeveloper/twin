import type { ChannelKey } from "./contract";

export type ColormapName = "thermal" | "rain" | "wind" | "pressure" | "moisture" | "delta";

type RGB = [number, number, number];

const STOPS: Record<ColormapName, RGB[]> = {
  thermal: [
    [22, 34, 82],
    [38, 102, 168],
    [56, 178, 164],
    [232, 208, 72],
    [239, 114, 42],
    [186, 24, 38],
  ],
  rain: [
    [13, 21, 38],
    [23, 56, 98],
    [28, 118, 166],
    [45, 184, 178],
    [136, 224, 134],
    [248, 226, 74],
    [230, 82, 54],
  ],
  wind: [
    [35, 86, 198],
    [84, 152, 232],
    [186, 214, 235],
    [224, 228, 234],
    [242, 182, 128],
    [226, 102, 58],
    [164, 24, 38],
  ],
  pressure: [
    [48, 24, 96],
    [46, 98, 168],
    [52, 176, 158],
    [162, 212, 104],
    [248, 222, 68],
  ],
  moisture: [
    [24, 22, 46],
    [44, 68, 134],
    [32, 138, 168],
    [54, 196, 154],
    [186, 230, 122],
  ],
  delta: [
    [26, 78, 196],
    [88, 162, 238],
    [30, 41, 59],
    [244, 134, 72],
    [206, 36, 48],
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
