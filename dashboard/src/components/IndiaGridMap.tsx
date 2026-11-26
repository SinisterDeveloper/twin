import { useEffect, useMemo, useRef, useState } from "react";
import {
  BOX_ARABIAN_SEA,
  BOX_BAY_OF_BENGAL,
  CHANNEL_META,
  CHANNELS,
  INDIA_COAST_POLYLINES,
  LAT,
  LAT_MAX,
  LAT_MIN,
  LON,
  LON_MAX,
  LON_MIN,
  NLAT,
  NLON,
  type ChannelKey,
} from "../lib/contract";
import { channelColormap, cssLinearGradient, sampleColormap } from "../lib/colormaps";

interface IndiaGridMapProps {
  /** [6, 32, 36] state grid for the active day */
  stateGrid: number[][][];
  channel: ChannelKey;
  useCelsius: boolean;
  showVectors: boolean;
  smoothGrid: boolean;
  showBoxes: boolean;
  title?: string;
  subtitle?: string;
  onCellClick?: (lat: number, lon: number) => void;
}

const SVG_W = 360;
const SVG_H = 320;

function lonToX(lon: number): number {
  return ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * SVG_W;
}

function latToY(lat: number): number {
  return ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * SVG_H;
}

export function IndiaGridMap({
  stateGrid,
  channel,
  useCelsius,
  showVectors,
  smoothGrid,
  showBoxes,
  title,
  subtitle,
  onCellClick,
}: IndiaGridMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  const meta = CHANNEL_META[channel];
  const cmap = channelColormap(channel);
  const [rMin, rMax] = meta.range(useCelsius);
  const scalarGrid = stateGrid[meta.index];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(NLON, NLAT);
    const data = img.data;
    for (let i = 0; i < NLAT; i++) {
      // North-up: row 0 on canvas corresponds to NLAT - 1 (37.5°N)
      const latIdx = NLAT - 1 - i;
      for (let j = 0; j < NLON; j++) {
        const disp = meta.toDisplay(scalarGrid[latIdx][j], useCelsius);
        const norm = (disp - rMin) / Math.max(rMax - rMin, 1e-6);
        const [r, g, b] = sampleColormap(cmap, norm);
        const off = (i * NLON + j) * 4;
        data[off] = r;
        data[off + 1] = g;
        data[off + 2] = b;
        data[off + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [scalarGrid, meta, useCelsius, rMin, rMax, cmap]);

  const windArrows = useMemo(() => {
    if (!showVectors) return [];
    const uGrid = stateGrid[2];
    const vGrid = stateGrid[3];
    const arrows: { x1: number; y1: number; x2: number; y2: number; key: string }[] = [];
    for (let i = 1; i < NLAT; i += 2) {
      for (let j = 1; j < NLON; j += 2) {
        const u = uGrid[i][j];
        const v = vGrid[i][j];
        const mag = Math.hypot(u, v);
        if (mag < 0.2) continue;
        const cx = lonToX(LON[j]);
        const cy = latToY(LAT[i]);
        const scale = Math.min(14, Math.max(3, mag * 0.95)) / mag;
        arrows.push({
          key: `${i}-${j}`,
          x1: cx,
          y1: cy,
          x2: cx + u * scale,
          y2: cy - v * scale,
        });
      }
    }
    return arrows;
  }, [showVectors, stateGrid]);

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const xFrac = Math.min(0.9999, Math.max(0, (e.clientX - rect.left) / rect.width));
    const yFrac = Math.min(0.9999, Math.max(0, (e.clientY - rect.top) / rect.height));
    const j = Math.floor(xFrac * NLON);
    const i = NLAT - 1 - Math.floor(yFrac * NLAT);
    setHover({ i, j });
  };

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onCellClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const xFrac = Math.min(0.9999, Math.max(0, (e.clientX - rect.left) / rect.width));
    const yFrac = Math.min(0.9999, Math.max(0, (e.clientY - rect.top) / rect.height));
    const j = Math.floor(xFrac * NLON);
    const i = NLAT - 1 - Math.floor(yFrac * NLAT);
    onCellClick(LAT[i], LON[j]);
  };

  const boxes = showBoxes ? [BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL] : [];

  return (
    <div className="flex flex-col rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 shadow-lg">
      {(title || subtitle) && (
        <div className="mb-2.5 flex items-baseline justify-between gap-2">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-100">{title}</h3>}
            {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
          </div>
          <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-cyan-300">
            32 × 36 · 1°
          </span>
        </div>
      )}

      <div className="relative mx-auto w-full max-w-[660px] aspect-[36/32] overflow-hidden rounded-lg border border-slate-700/80 bg-slate-950">
        <canvas
          ref={canvasRef}
          width={NLON}
          height={NLAT}
          className="h-full w-full block"
          style={{ imageRendering: smoothGrid ? "auto" : "pixelated" }}
        />

        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          className="absolute inset-0 h-full w-full cursor-crosshair select-none"
          onPointerMove={handlePointerMove}
          onPointerLeave={() => setHover(null)}
          onClick={handleClick}
        >
          <defs>
            <marker
              id="wind-arrow"
              markerWidth="4"
              markerHeight="4"
              refX="2.5"
              refY="2"
              orient="auto"
            >
              <path d="M0,0 L4,2 L0,4 Z" fill="rgba(248, 250, 252, 0.85)" />
            </marker>
          </defs>

          {/* Lat/Lon Graticule */}
          {[10, 20, 30].map((lat) => {
            const y = latToY(lat);
            return (
              <g key={`lat-${lat}`}>
                <line
                  x1={0}
                  y1={y}
                  x2={SVG_W}
                  y2={y}
                  stroke="rgba(148, 163, 184, 0.22)"
                  strokeWidth={0.6}
                  strokeDasharray="2 3"
                />
                <text x={4} y={y - 2} fill="rgba(226, 232, 240, 0.65)" fontSize="8" fontFamily="monospace">
                  {lat}°N
                </text>
              </g>
            );
          })}
          {[70, 80, 90, 100].map((lon) => {
            const x = lonToX(lon);
            return (
              <g key={`lon-${lon}`}>
                <line
                  x1={x}
                  y1={0}
                  x2={x}
                  y2={SVG_H}
                  stroke="rgba(148, 163, 184, 0.22)"
                  strokeWidth={0.6}
                  strokeDasharray="2 3"
                />
                <text x={x + 2} y={SVG_H - 4} fill="rgba(226, 232, 240, 0.65)" fontSize="8" fontFamily="monospace">
                  {lon}°E
                </text>
              </g>
            );
          })}

          {/* Approximate Coastline & Himalayan Context */}
          {INDIA_COAST_POLYLINES.map((poly, idx) => {
            const pts = poly.map(([lo, la]) => `${lonToX(lo).toFixed(1)},${latToY(la).toFixed(1)}`).join(" ");
            return (
              <polyline
                key={`coast-${idx}`}
                points={pts}
                fill="none"
                stroke={idx === 2 ? "rgba(248, 250, 252, 0.45)" : "rgba(248, 250, 252, 0.88)"}
                strokeWidth={idx === 2 ? 0.9 : 1.3}
                strokeDasharray={idx === 2 ? "3 2" : undefined}
                strokeLinejoin="round"
              />
            );
          })}

          {/* Round-1 Shock Regions */}
          {boxes.map((box) => {
            const x = lonToX(box.lon0 - 0.5);
            const y = latToY(box.lat1 + 0.5);
            const w = lonToX(box.lon1 + 0.5) - x;
            const h = latToY(box.lat0 - 0.5) - y;
            return (
              <g key={box.id}>
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  fill="rgba(34, 211, 238, 0.06)"
                  stroke="rgba(34, 211, 238, 0.75)"
                  strokeWidth={1.1}
                  strokeDasharray="4 2"
                />
                <text
                  x={x + 4}
                  y={y + 10}
                  fill="rgba(165, 243, 252, 0.92)"
                  fontSize="8"
                  fontWeight="600"
                >
                  {box.name}
                </text>
              </g>
            );
          })}

          {/* 10m Wind Vectors */}
          {windArrows.map((a) => (
            <line
              key={a.key}
              x1={a.x1}
              y1={a.y1}
              x2={a.x2}
              y2={a.y2}
              stroke="rgba(248, 250, 252, 0.75)"
              strokeWidth={0.95}
              markerEnd="url(#wind-arrow)"
            />
          ))}

          {/* Hovered Cell Highlight */}
          {hover && (
            <rect
              x={hover.j * (SVG_W / NLON)}
              y={(NLAT - 1 - hover.i) * (SVG_H / NLAT)}
              width={SVG_W / NLON}
              height={SVG_H / NLAT}
              fill="none"
              stroke="#ffffff"
              strokeWidth={1.5}
            />
          )}
        </svg>

        {hover && (
          <div className="pointer-events-none absolute top-2 right-2 rounded-lg border border-slate-700 bg-slate-950/90 px-2.5 py-2 text-[11px] shadow-xl backdrop-blur">
            <div className="mb-1 font-mono font-semibold text-cyan-300">
              {LAT[hover.i].toFixed(1)}°N, {LON[hover.j].toFixed(1)}°E
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-slate-200">
              {CHANNELS.map((c) => {
                const m = CHANNEL_META[c];
                return (
                  <div key={c} className={c === channel ? "font-semibold text-amber-300" : ""}>
                    <span className="text-slate-400">{c}: </span>
                    {m.format(stateGrid[m.index][hover.i][hover.j], useCelsius)}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Colorbar Legend */}
      <div className="mx-auto mt-3 w-full max-w-[660px]">
        <div
          className="h-2.5 w-full rounded-full border border-slate-700"
          style={{ background: cssLinearGradient(cmap) }}
        />
        <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
          <span>
            {rMin.toFixed(1)} {meta.displayUnit(useCelsius)}
          </span>
          <span>
            {((rMin + rMax) / 2).toFixed(1)} {meta.displayUnit(useCelsius)}
          </span>
          <span>
            {rMax.toFixed(1)} {meta.displayUnit(useCelsius)}
          </span>
        </div>
      </div>
    </div>
  );
}
