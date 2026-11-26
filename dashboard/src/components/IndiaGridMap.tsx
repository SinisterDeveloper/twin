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
        const scale = Math.min(13, Math.max(3, mag * 0.9)) / mag;
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
    <div className="flex flex-col rounded-xl border border-zinc-800/80 bg-zinc-950 p-4">
      {(title || subtitle) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          <div>
            {title && <h3 className="text-sm font-medium tracking-tight text-zinc-100">{title}</h3>}
            {subtitle && <p className="text-xs text-zinc-500">{subtitle}</p>}
          </div>
          <span className="rounded-md border border-zinc-800 bg-black px-2 py-0.5 font-mono text-[11px] tabular-nums text-zinc-400">
            32×36 · 1°
          </span>
        </div>
      )}

      <div className="relative mx-auto w-full max-w-[660px] aspect-[36/32] overflow-hidden rounded-lg border border-zinc-800/90 bg-black">
        <canvas
          ref={canvasRef}
          width={NLON}
          height={NLAT}
          className="block h-full w-full"
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
              markerWidth="3.5"
              markerHeight="3.5"
              refX="2.2"
              refY="1.75"
              orient="auto"
            >
              <path d="M0,0 L3.5,1.75 L0,3.5 Z" fill="rgba(255, 255, 255, 0.78)" />
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
                  stroke="rgba(255, 255, 255, 0.12)"
                  strokeWidth={0.55}
                  strokeDasharray="2 3"
                />
                <text x={5} y={y - 3} fill="rgba(244, 244, 245, 0.55)" fontSize="7.5" fontFamily="monospace">
                  {lat}°N
                </text>
              </g>
            );
          })}
          {[70, 80, 90, 100].map((lon) => {
            const x = lonToX(lon);
            const rightEdge = lon >= 98;
            return (
              <g key={`lon-${lon}`}>
                <line
                  x1={x}
                  y1={0}
                  x2={x}
                  y2={SVG_H}
                  stroke="rgba(255, 255, 255, 0.12)"
                  strokeWidth={0.55}
                  strokeDasharray="2 3"
                />
                <text
                  x={rightEdge ? x - 3 : x + 3}
                  y={SVG_H - 5}
                  textAnchor={rightEdge ? "end" : "start"}
                  fill="rgba(244, 244, 245, 0.55)"
                  fontSize="7.5"
                  fontFamily="monospace"
                >
                  {lon}°E
                </text>
              </g>
            );
          })}

          {/* Coastline & Border Context */}
          {INDIA_COAST_POLYLINES.map((poly, idx) => {
            const pts = poly.map(([lo, la]) => `${lonToX(lo).toFixed(1)},${latToY(la).toFixed(1)}`).join(" ");
            return (
              <polyline
                key={`coast-${idx}`}
                points={pts}
                fill="none"
                stroke={idx === 2 ? "rgba(255, 255, 255, 0.36)" : "rgba(255, 255, 255, 0.85)"}
                strokeWidth={idx === 2 ? 0.85 : 1.15}
                strokeDasharray={idx === 2 ? "2.5 2" : undefined}
                strokeLinejoin="round"
              />
            );
          })}

          {/* Reference Shock Boxes */}
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
                  fill="rgba(255, 255, 255, 0.04)"
                  stroke="rgba(255, 255, 255, 0.55)"
                  strokeWidth={0.9}
                  strokeDasharray="3.5 2.5"
                />
                <text
                  x={x + 4}
                  y={y + 10}
                  fill="rgba(250, 250, 250, 0.82)"
                  fontSize="7.5"
                  fontWeight="500"
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
              stroke="rgba(255, 255, 255, 0.72)"
              strokeWidth={0.85}
              markerEnd="url(#wind-arrow)"
            />
          ))}

          {/* Hovered Cell */}
          {hover && (
            <rect
              x={hover.j * (SVG_W / NLON)}
              y={(NLAT - 1 - hover.i) * (SVG_H / NLAT)}
              width={SVG_W / NLON}
              height={SVG_H / NLAT}
              fill="none"
              stroke="#ffffff"
              strokeWidth={1.3}
            />
          )}
        </svg>

        {hover && (
          <div className="pointer-events-none absolute top-2.5 right-2.5 rounded-lg border border-zinc-800 bg-black/90 px-3 py-2 text-[11px] shadow-2xl backdrop-blur-md">
            <div className="mb-1.5 font-mono font-medium tabular-nums text-zinc-100">
              {LAT[hover.i].toFixed(1)}°N, {LON[hover.j].toFixed(1)}°E
            </div>
            <div className="grid grid-cols-2 gap-x-3.5 gap-y-0.5 font-mono tabular-nums text-zinc-400">
              {CHANNELS.map((c) => {
                const m = CHANNEL_META[c];
                return (
                  <div key={c} className={c === channel ? "font-medium text-zinc-50" : ""}>
                    <span className="text-zinc-500">{c} </span>
                    {m.format(stateGrid[m.index][hover.i][hover.j], useCelsius)}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Colorbar */}
      <div className="mx-auto mt-3.5 w-full max-w-[660px]">
        <div
          className="h-1.5 w-full rounded-full border border-zinc-800"
          style={{ background: cssLinearGradient(cmap) }}
        />
        <div className="mt-1.5 flex justify-between font-mono text-[11px] tabular-nums text-zinc-500">
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
