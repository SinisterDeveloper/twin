import { useEffect, useRef, useState } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";
import {
  BOX_ARABIAN_SEA,
  BOX_BAY_OF_BENGAL,
  CHANNEL_META,
  CHANNELS,
  LAT,
  LAT_MAX,
  LAT_MIN,
  LON,
  LON_MAX,
  LON_MIN,
  MAJOR_CITIES,
  NLAT,
  NLON,
  sampleBilinear,
  type ChannelKey,
  type GeoBox,
} from "../lib/contract";
import { INDIA_STATE_PATHS, NEIGHBOR_COUNTRY_PATHS } from "../lib/geo";
import {
  channelColormap,
  cssLinearGradient,
  renderFieldToCanvas,
} from "../lib/colormaps";
import { deltaToDisplay, formatDelta } from "../lib/shocks";

export interface MapViewState {
  zoom: number;
  panX: number;
  panY: number;
}

export const DEFAULT_MAP_VIEW: MapViewState = { zoom: 1, panX: 0, panY: 0 };

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.5;

function clampView(zoom: number, panX: number, panY: number): MapViewState {
  const z = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(zoom * 100) / 100));
  const maxPan = (z - 1) / 2;
  return {
    zoom: z,
    panX: Math.min(maxPan, Math.max(-maxPan, panX)),
    panY: Math.min(maxPan, Math.max(-maxPan, panY)),
  };
}

interface IndiaGridMapProps {
  /** [6, 32, 36] state or delta grid for the active day */
  stateGrid: number[][][];
  channel: ChannelKey;
  useCelsius: boolean;
  showVectors: boolean;
  smoothGrid: boolean;
  showBoxes: boolean;
  showCities?: boolean;
  showStates?: boolean;
  mode?: "field" | "delta";
  highlightBox?: GeoBox | null;
  customCenter?: { lat: number; lon: number; radiusDeg: number } | null;
  compact?: boolean;
  title?: string;
  subtitle?: string;
  view?: MapViewState;
  onViewChange?: (next: MapViewState) => void;
  onCellClick?: (lat: number, lon: number) => void;
}

const SVG_W = 360;
const SVG_H = 320;
const FIELD_W = 360;
const FIELD_H = 320;
const STREAM_W = 720;
const STREAM_H = 640;
const PARTICLE_COUNT = 260;

function lonToX(lon: number): number {
  return ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * SVG_W;
}

function latToY(lat: number): number {
  return ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * SVG_H;
}

function compassDir(u: number, v: number): string {
  const deg = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.round(deg / 45) % 8];
}

export function IndiaGridMap({
  stateGrid,
  channel,
  useCelsius,
  showVectors,
  smoothGrid,
  showBoxes,
  showCities = true,
  showStates = true,
  mode = "field",
  highlightBox = null,
  customCenter = null,
  compact = false,
  title,
  subtitle,
  view,
  onViewChange,
  onCellClick,
}: IndiaGridMapProps) {
  const fieldCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startPanX: number;
    startPanY: number;
    moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef(false);

  const [internalView, setInternalView] = useState<MapViewState>(DEFAULT_MAP_VIEW);
  const [isDragging, setIsDragging] = useState(false);
  const [hover, setHover] = useState<{ i: number; j: number; lat: number; lon: number } | null>(null);

  const activeView = view ?? internalView;
  const setView = (next: MapViewState) => {
    if (onViewChange) {
      onViewChange(next);
    } else {
      setInternalView(next);
    }
  };

  const zoomBy = (delta: number) => {
    const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, activeView.zoom + delta));
    const ratio = nextZoom / activeView.zoom;
    setView(clampView(nextZoom, activeView.panX * ratio, activeView.panY * ratio));
  };

  const resetZoom = () => {
    setView(DEFAULT_MAP_VIEW);
  };

  const isDelta = mode === "delta";
  const meta = CHANNEL_META[channel];
  const cmap = isDelta ? "delta" : channelColormap(channel);
  const [rMin, rMax] = isDelta ? [-meta.deltaMax, meta.deltaMax] : meta.range(useCelsius);
  const scalarGrid = stateGrid[meta.index];

  // Render smooth bilinear or crisp 1° meteorological heatmap / diverging delta
  useEffect(() => {
    const canvas = fieldCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    renderFieldToCanvas(
      ctx,
      scalarGrid,
      FIELD_W,
      FIELD_H,
      (v) => (isDelta ? deltaToDisplay(v, channel) : meta.toDisplay(v, useCelsius)),
      rMin,
      rMax,
      cmap,
      smoothGrid,
    );
  }, [scalarGrid, meta, channel, isDelta, useCelsius, rMin, rMax, cmap, smoothGrid]);

  // Animated Windy-style wind particle streamlines (suppressed in delta mode)
  useEffect(() => {
    const canvas = streamCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, STREAM_W, STREAM_H);
    if (!showVectors || isDelta) return;

    const uGrid = stateGrid[2];
    const vGrid = stateGrid[3];

    interface Particle {
      x: number;
      y: number;
      age: number;
      maxAge: number;
    }

    const spawn = (): Particle => ({
      x: Math.random() * STREAM_W,
      y: Math.random() * STREAM_H,
      age: Math.floor(Math.random() * 45),
      maxAge: 45 + Math.floor(Math.random() * 40),
    });

    const particles: Particle[] = Array.from({ length: PARTICLE_COUNT }, spawn);
    let rafId = 0;

    const tick = () => {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0, 0, 0, 0.085)";
      ctx.fillRect(0, 0, STREAM_W, STREAM_H);
      ctx.globalCompositeOperation = "source-over";

      ctx.lineCap = "round";
      for (let p = 0; p < particles.length; p++) {
        const pt = particles[p];
        if (pt.age >= pt.maxAge || pt.x < 0 || pt.x >= STREAM_W || pt.y < 0 || pt.y >= STREAM_H) {
          particles[p] = spawn();
          particles[p].age = 0;
          continue;
        }

        const lon = LON_MIN + (pt.x / STREAM_W) * (LON_MAX - LON_MIN);
        const lat = LAT_MAX - (pt.y / STREAM_H) * (LAT_MAX - LAT_MIN);
        const u = sampleBilinear(uGrid, lat, lon);
        const v = sampleBilinear(vGrid, lat, lon);
        const speed = Math.hypot(u, v);

        const dx = u * 0.34;
        const dy = -v * 0.34;
        const nx = pt.x + dx;
        const ny = pt.y + dy;

        const alpha = Math.min(0.88, 0.25 + speed * 0.055);
        ctx.strokeStyle = `rgba(255, 255, 255, ${alpha.toFixed(2)})`;
        ctx.lineWidth = speed > 8 ? 1.55 : 1.15;
        ctx.beginPath();
        ctx.moveTo(pt.x, pt.y);
        ctx.lineTo(nx, ny);
        ctx.stroke();

        pt.x = nx;
        pt.y = ny;
        pt.age += 1;
      }

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [showVectors, isDelta, stateGrid]);

  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button === 0 && activeView.zoom > 1) {
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = {
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        startPanX: activeView.panX,
        startPanY: activeView.panY,
        moved: false,
      };
      setIsDragging(true);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    if (drag && drag.pointerId === e.pointerId && viewportRef.current) {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      if (Math.hypot(dx, dy) > 4) {
        drag.moved = true;
      }
      const rect = viewportRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setView(
          clampView(
            activeView.zoom,
            drag.startPanX + dx / rect.width,
            drag.startPanY + dy / rect.height,
          ),
        );
      }
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const xFrac = Math.min(0.9999, Math.max(0, (e.clientX - rect.left) / rect.width));
    const yFrac = Math.min(0.9999, Math.max(0, (e.clientY - rect.top) / rect.height));
    const j = Math.floor(xFrac * NLON);
    const i = NLAT - 1 - Math.floor(yFrac * NLAT);
    const lon = LON_MIN + xFrac * (LON_MAX - LON_MIN);
    const lat = LAT_MAX - yFrac * (LAT_MAX - LAT_MIN);
    setHover({ i, j, lat, lon });
  };

  const handlePointerUpOrCancel = (e: React.PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    if (drag && drag.pointerId === e.pointerId) {
      suppressClickRef.current = drag.moved;
      dragRef.current = null;
      setIsDragging(false);
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    }
  };

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (!onCellClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const xFrac = Math.min(0.9999, Math.max(0, (e.clientX - rect.left) / rect.width));
    const yFrac = Math.min(0.9999, Math.max(0, (e.clientY - rect.top) / rect.height));
    const j = Math.floor(xFrac * NLON);
    const i = NLAT - 1 - Math.floor(yFrac * NLAT);
    onCellClick(LAT[i], LON[j]);
  };

  const boxes = showBoxes ? [BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL] : [];
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => rMin + f * (rMax - rMin));
  const unitLabel = isDelta && channel === "t2m" ? (useCelsius ? "°C" : "K") : meta.displayUnit(useCelsius);

  return (
    <div className={`flex flex-col rounded-xl border border-zinc-800/80 bg-zinc-950 ${compact ? "p-2.5" : "p-3.5"}`}>
      {(title || subtitle) && (
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="min-w-0">
            {title && (
              <h3 className={`truncate font-medium tracking-tight text-zinc-100 ${compact ? "text-xs" : "text-sm"}`}>
                {title}
              </h3>
            )}
            {subtitle && <p className="truncate text-[11px] text-zinc-500">{subtitle}</p>}
          </div>
          {!compact && (
            <span className="shrink-0 rounded-md border border-zinc-800 bg-black px-2 py-0.5 font-mono text-[11px] tabular-nums text-zinc-400">
              6.5°–37.5°N · 65.5°–100.5°E
            </span>
          )}
        </div>
      )}

      <div
        ref={viewportRef}
        className={`relative mx-auto w-full ${compact ? "" : "max-w-[680px]"} aspect-[36/32] overflow-hidden rounded-lg border border-zinc-800 bg-slate-950 shadow-2xl`}
      >
        <div
          className={`relative h-full w-full ${isDragging ? "" : "transition-transform duration-150 ease-out"}`}
          style={{
            transform: `translate(${activeView.panX * 100}%, ${activeView.panY * 100}%) scale(${activeView.zoom})`,
            transformOrigin: "center center",
          }}
        >
          <canvas
            ref={fieldCanvasRef}
            width={FIELD_W}
            height={FIELD_H}
            className="block h-full w-full"
            style={{ imageRendering: smoothGrid ? "auto" : "pixelated" }}
          />

          <canvas
            ref={streamCanvasRef}
            width={STREAM_W}
            height={STREAM_H}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />

          <svg
            viewBox={`0 0 ${SVG_W} ${SVG_H}`}
            className={`absolute inset-0 h-full w-full select-none ${
              activeView.zoom > 1
                ? `touch-none ${isDragging ? "cursor-grabbing" : "cursor-grab"}`
                : onCellClick
                  ? "cursor-pointer"
                  : "cursor-crosshair"
            }`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUpOrCancel}
            onPointerCancel={handlePointerUpOrCancel}
            onPointerLeave={() => setHover(null)}
            onClick={handleClick}
          >
          {[10, 20, 30].map((lat) => {
            const y = latToY(lat);
            return (
              <g key={`lat-${lat}`}>
                <line
                  x1={0}
                  y1={y}
                  x2={SVG_W}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.16)"
                  strokeWidth={0.5}
                  strokeDasharray="2 3"
                />
                <text
                  x={5}
                  y={y - 3}
                  fill="rgba(255, 255, 255, 0.72)"
                  stroke="rgba(0, 0, 0, 0.75)"
                  strokeWidth={1.6}
                  paintOrder="stroke"
                  fontSize="7.5"
                  fontFamily="monospace"
                >
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
                  stroke="rgba(255, 255, 255, 0.16)"
                  strokeWidth={0.5}
                  strokeDasharray="2 3"
                />
                <text
                  x={rightEdge ? x - 3 : x + 3}
                  y={SVG_H - 5}
                  textAnchor={rightEdge ? "end" : "start"}
                  fill="rgba(255, 255, 255, 0.72)"
                  stroke="rgba(0, 0, 0, 0.75)"
                  strokeWidth={1.6}
                  paintOrder="stroke"
                  fontSize="7.5"
                  fontFamily="monospace"
                >
                  {lon}°E
                </text>
              </g>
            );
          })}

          <text
            x={lonToX(69.5)}
            y={latToY(14.5)}
            fill="rgba(255, 255, 255, 0.25)"
            fontSize="7.5"
            fontWeight="700"
            letterSpacing="1.6"
            textAnchor="middle"
          >
            ARABIAN SEA
          </text>
          <text
            x={lonToX(88.2)}
            y={latToY(14.5)}
            fill="rgba(255, 255, 255, 0.25)"
            fontSize="7.5"
            fontWeight="700"
            letterSpacing="1.6"
            textAnchor="middle"
          >
            BAY OF BENGAL
          </text>

          {/* Neighboring Countries & Sri Lanka Coastlines (50m Natural Earth) */}
          {NEIGHBOR_COUNTRY_PATHS.map((country, idx) => (
            <g key={`nbr-${idx}`}>
              <path
                d={country.d}
                fill="none"
                stroke="rgba(0, 0, 0, 0.55)"
                strokeWidth={1.4}
                strokeLinejoin="round"
              />
              <path
                d={country.d}
                fill="none"
                stroke="rgba(255, 255, 255, 0.36)"
                strokeWidth={0.7}
                strokeLinejoin="round"
              />
            </g>
          ))}

          {/* Real India Coastline & State Boundaries */}
          {INDIA_STATE_PATHS.map((st, idx) => (
            <path
              key={`state-${idx}`}
              d={st.d}
              fill="none"
              stroke="rgba(0, 0, 0, 0.65)"
              strokeWidth={showStates ? 1.7 : 1.0}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {showStates &&
            INDIA_STATE_PATHS.map((st, idx) => (
              <path
                key={`state-fg-${idx}`}
                d={st.d}
                fill="none"
                stroke="rgba(255, 255, 255, 0.86)"
                strokeWidth={0.85}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}

          {boxes.map((box) => {
            const x = lonToX(box.lon0 - 0.5);
            const y = latToY(box.lat1 + 0.5);
            const w = lonToX(box.lon1 + 0.5) - x;
            const h = latToY(box.lat0 - 0.5) - y;
            const isHot = highlightBox?.id === box.id;
            return (
              <g key={box.id}>
                <rect
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  fill={isHot ? "rgba(251, 191, 36, 0.12)" : "rgba(56, 189, 248, 0.06)"}
                  stroke={isHot ? "#fbbf24" : "rgba(103, 232, 249, 0.78)"}
                  strokeWidth={isHot ? 1.7 : 1.0}
                  strokeDasharray={isHot ? undefined : "4 2.5"}
                />
                <text
                  x={x + 4}
                  y={y + 10}
                  fill={isHot ? "#fde68a" : "#cffafe"}
                  stroke="rgba(0,0,0,0.8)"
                  strokeWidth={1.8}
                  paintOrder="stroke"
                  fontSize="7.5"
                  fontWeight="600"
                >
                  {box.name}
                </text>
              </g>
            );
          })}

          {customCenter && (
            <g>
              <circle
                cx={lonToX(customCenter.lon)}
                cy={latToY(customCenter.lat)}
                r={(customCenter.radiusDeg / (LON_MAX - LON_MIN)) * SVG_W}
                fill="rgba(251, 191, 36, 0.1)"
                stroke="#fbbf24"
                strokeWidth={1.3}
                strokeDasharray="3 2"
              />
              <circle cx={lonToX(customCenter.lon)} cy={latToY(customCenter.lat)} r={3} fill="#fbbf24" stroke="#000" strokeWidth={0.9} />
            </g>
          )}

          {showCities &&
            MAJOR_CITIES.map((city) => {
              const cx = lonToX(city.lon);
              const cy = latToY(city.lat);
              const leftLabel = city.name === "Bengaluru";
              return (
                <g key={city.name}>
                  <circle cx={cx} cy={cy} r={2.2} fill="#ffffff" stroke="#000000" strokeWidth={0.9} />
                  <text
                    x={leftLabel ? cx - 4 : cx + 4}
                    y={cy + 2.5}
                    textAnchor={leftLabel ? "end" : "start"}
                    fill="#ffffff"
                    stroke="rgba(0, 0, 0, 0.85)"
                    strokeWidth={2.2}
                    paintOrder="stroke"
                    fontSize="7.2"
                    fontWeight="600"
                  >
                    {city.name}
                  </text>
                </g>
              );
            })}

          {hover && (
            <rect
              x={hover.j * (SVG_W / NLON)}
              y={(NLAT - 1 - hover.i) * (SVG_H / NLAT)}
              width={SVG_W / NLON}
              height={SVG_H / NLAT}
              fill="rgba(255,255,255,0.08)"
              stroke="#ffffff"
              strokeWidth={1.2}
            />
          )}
            </svg>
        </div>

        {/* Bottom-right Zoom Controls */}
        <div
          className="absolute bottom-2.5 right-2.5 z-10 flex flex-col items-end gap-1"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          {activeView.zoom > 1 && (
            <button
              type="button"
              aria-label="Reset zoom"
              onClick={resetZoom}
              className="flex items-center gap-1 rounded-md border border-zinc-700/80 bg-black/85 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-zinc-200 shadow-lg backdrop-blur-md transition hover:bg-zinc-900"
            >
              <span>{activeView.zoom.toFixed(1)}×</span>
              <RotateCcw className="h-2.5 w-2.5" />
            </button>
          )}
          <div className="flex flex-col overflow-hidden rounded-md border border-zinc-700/80 bg-black/85 shadow-lg backdrop-blur-md">
            <button
              type="button"
              aria-label="Zoom in"
              disabled={activeView.zoom >= MAX_ZOOM}
              onClick={() => zoomBy(ZOOM_STEP)}
              className="flex h-7 w-7 items-center justify-center text-zinc-200 transition hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-35"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
            <div className="h-px w-full bg-zinc-800" />
            <button
              type="button"
              aria-label="Zoom out"
              disabled={activeView.zoom <= MIN_ZOOM}
              onClick={() => zoomBy(-ZOOM_STEP)}
              className="flex h-7 w-7 items-center justify-center text-zinc-200 transition hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-35"
            >
              <Minus className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {hover && (
          <div className="pointer-events-none absolute top-2 right-2 rounded-lg border border-zinc-700/80 bg-black/88 px-2.5 py-1.5 text-[10px] shadow-2xl backdrop-blur-md">
            <div className="mb-1 flex items-center justify-between gap-2 font-mono font-medium tabular-nums text-zinc-100">
              <span>
                {hover.lat.toFixed(1)}°N, {hover.lon.toFixed(1)}°E
              </span>
              {!isDelta && (
                <span className="text-cyan-300">
                  {Math.hypot(stateGrid[2][hover.i][hover.j], stateGrid[3][hover.i][hover.j]).toFixed(1)} m/s{" "}
                  {compassDir(stateGrid[2][hover.i][hover.j], stateGrid[3][hover.i][hover.j])}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono tabular-nums text-zinc-300">
              {CHANNELS.map((c) => {
                const m = CHANNEL_META[c];
                const rawVal = stateGrid[m.index][hover.i][hover.j];
                return (
                  <div key={c} className={c === channel ? "font-semibold text-amber-300" : ""}>
                    <span className="text-zinc-500">{c} </span>
                    {isDelta ? formatDelta(rawVal, c, useCelsius) : m.format(rawVal, useCelsius)}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Meteorological 5-Tick Colorbar */}
      <div className={`mx-auto mt-2.5 w-full ${compact ? "" : "max-w-[680px]"}`}>
        <div
          className="h-2 w-full rounded-full border border-zinc-800"
          style={{ background: cssLinearGradient(cmap) }}
        />
        <div className="mt-1 flex justify-between font-mono text-[10px] tabular-nums text-zinc-400">
          {ticks.map((t, idx) => {
            const prefix = isDelta && t > 0 ? "+" : "";
            return (
              <span key={idx}>
                {prefix}
                {t.toFixed(idx === 0 || idx === ticks.length - 1 ? 0 : 1)}
                {idx === ticks.length - 1 ? ` ${unitLabel}` : ""}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
