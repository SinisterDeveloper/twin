import type { CSSProperties } from "react";
import type { SensitivityResponse } from "../lib/api";
import { CHANNEL_META, CHANNELS, type ChannelKey } from "../lib/contract";

export interface AdjointConfig {
  lat: number;
  lon: number;
  targetChannel: ChannelKey;
  targetDay: number;
  radiusCells: 0 | 1 | 2;
  direction: "increase" | "decrease";
  threshold: string;
}

export type AdjointOverlayChannel = "all" | ChannelKey;

export interface AdjointPreset {
  id: string;
  label: string;
  lat: number;
  lon: number;
  targetChannel: ChannelKey;
  targetDay: number;
  direction: "increase" | "decrease";
}

export const ADJOINT_PRESETS: AdjointPreset[] = [
  {
    id: "kerala_rain",
    label: "Kerala Rain",
    lat: 10.5,
    lon: 76.5,
    targetChannel: "tp",
    targetDay: 6,
    direction: "increase",
  },
  {
    id: "delhi_heat",
    label: "Delhi Heat",
    lat: 28.5,
    lon: 77.5,
    targetChannel: "t2m",
    targetDay: 8,
    direction: "increase",
  },
  {
    id: "mumbai_rain",
    label: "Mumbai Rain",
    lat: 19.5,
    lon: 72.5,
    targetChannel: "tp",
    targetDay: 4,
    direction: "increase",
  },
  {
    id: "bob_moisture",
    label: "BoB Moisture",
    lat: 14.5,
    lon: 88.5,
    targetChannel: "tcwv",
    targetDay: 5,
    direction: "increase",
  },
];

interface AdjointPanelProps {
  config: AdjointConfig;
  overlayChannel: AdjointOverlayChannel;
  loading: boolean;
  result: SensitivityResponse | null;
  error: string | null;
  useCelsius: boolean;
  onChangeConfig: (next: AdjointConfig) => void;
  onChangeOverlayChannel: (next: AdjointOverlayChannel) => void;
  onCompute: () => void;
  onApplyReverseShock: () => void;
}

const RADIUS_OPTIONS: { value: 0 | 1 | 2; label: string }[] = [
  { value: 0, label: "1×1" },
  { value: 1, label: "3×3" },
  { value: 2, label: "5×5" },
];

export function AdjointPanel({
  config,
  overlayChannel,
  loading,
  result,
  error,
  useCelsius,
  onChangeConfig,
  onChangeOverlayChannel,
  onCompute,
  onApplyReverseShock,
}: AdjointPanelProps) {
  const dayPct = `${((config.targetDay - 1) / 9) * 100}%`;
  const targetMeta = CHANNEL_META[config.targetChannel];

  const sortedDrivers = result
    ? [...CHANNELS].sort(
        (a, b) => (result.channel_importance[b] ?? 0) - (result.channel_importance[a] ?? 0),
      )
    : [];

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          Adjoint Target
        </span>
        {result && (
          <span className="font-mono text-[10px] tabular-nums text-zinc-400">
            {result.compute_ms.toFixed(0)} ms{result.source === "mock" ? " · mock" : ""}
          </span>
        )}
      </div>

      {/* Presets */}
      <div className="grid grid-cols-2 gap-1.5">
        {ADJOINT_PRESETS.map((p) => {
          const active =
            Math.abs(config.lat - p.lat) < 0.01 &&
            Math.abs(config.lon - p.lon) < 0.01 &&
            config.targetChannel === p.targetChannel &&
            config.targetDay === p.targetDay;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() =>
                onChangeConfig({
                  ...config,
                  lat: p.lat,
                  lon: p.lon,
                  targetChannel: p.targetChannel,
                  targetDay: p.targetDay,
                  direction: p.direction,
                })
              }
              className={`rounded-md border px-2 py-1.5 text-left transition ${
                active
                  ? "border-white/20 bg-white/[0.10] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]"
                  : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
              }`}
            >
              <div className="truncate text-[11px] font-medium">{p.label}</div>
              <div className="font-mono text-[10px] text-zinc-500">
                Day +{p.targetDay} · {p.targetChannel}
              </div>
            </button>
          );
        })}
      </div>

      {/* Click-to-pick coordinates */}
      <div className="flex items-center justify-between rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-1.5 font-mono text-[11px] tabular-nums">
        <span className="text-zinc-500">Target (click map)</span>
        <span className="text-zinc-200">
          {config.lat.toFixed(1)}°N, {config.lon.toFixed(1)}°E
        </span>
      </div>

      {/* Target Variable */}
      <div>
        <div className="mb-1.5 text-[11px] text-zinc-500">Target Variable</div>
        <div className="grid grid-cols-3 gap-1">
          {CHANNELS.map((c) => {
            const active = config.targetChannel === c;
            return (
              <button
                key={c}
                type="button"
                onClick={() => onChangeConfig({ ...config, targetChannel: c })}
                className={`rounded border py-1 font-mono text-[11px] transition ${
                  active
                    ? "border-white/20 bg-white/[0.10] font-semibold text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]"
                    : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
                }`}
              >
                {c}
              </button>
            );
          })}
        </div>
      </div>

      {/* Lead Day Slider */}
      <div>
        <div className="mb-1 flex justify-between text-[11px]">
          <span className="text-zinc-500">Lead Day</span>
          <span className="font-mono tabular-nums text-zinc-200">Day +{config.targetDay}</span>
        </div>
        <input
          type="range"
          min={1}
          max={10}
          step={1}
          value={config.targetDay}
          style={{ "--slider-pct": dayPct } as CSSProperties}
          onChange={(e) => onChangeConfig({ ...config, targetDay: Number(e.target.value) })}
          className="shadcn-slider w-full"
        />
      </div>

      {/* Patch Radius & Direction */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <div className="mb-1 text-[11px] text-zinc-500">Patch</div>
          <div className="grid grid-cols-3 gap-1">
            {RADIUS_OPTIONS.map((opt) => {
              const active = config.radiusCells === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => onChangeConfig({ ...config, radiusCells: opt.value })}
                  className={`rounded border py-1 font-mono text-[10px] transition ${
                    active
                      ? "border-white/20 bg-white/[0.10] font-semibold text-white"
                      : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <div className="mb-1 text-[11px] text-zinc-500">Direction</div>
          <div className="grid grid-cols-2 gap-1">
            {(["increase", "decrease"] as const).map((dir) => {
              const active = config.direction === dir;
              return (
                <button
                  key={dir}
                  type="button"
                  onClick={() => onChangeConfig({ ...config, direction: dir })}
                  className={`rounded border py-1 text-[10px] font-medium capitalize transition ${
                    active
                      ? "border-white/20 bg-white/[0.10] text-white"
                      : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
                  }`}
                >
                  {dir === "increase" ? "Inc" : "Dec"}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Optional Threshold */}
      <div>
        <div className="mb-1 flex items-center justify-between text-[11px]">
          <span className="text-zinc-500">Target Threshold (opt.)</span>
          <span className="font-mono text-[10px] text-zinc-600">{targetMeta.unit}</span>
        </div>
        <input
          type="number"
          step="any"
          placeholder={`e.g. ${config.targetChannel === "tp" ? "25.0" : config.targetChannel === "t2m" ? "312.0" : "auto"}`}
          value={config.threshold}
          onChange={(e) => onChangeConfig({ ...config, threshold: e.target.value })}
          className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 py-1 font-mono text-[11px] text-zinc-200 placeholder:text-zinc-600 focus:border-white/25 focus:outline-none"
        />
      </div>

      {/* Overlay Channel Selector */}
      <div>
        <div className="mb-1.5 text-[11px] text-zinc-500">Overlay Channel</div>
        <div className="grid grid-cols-4 gap-1">
          <button
            type="button"
            onClick={() => onChangeOverlayChannel("all")}
            className={`col-span-2 rounded border py-1 font-mono text-[10px] transition ${
              overlayChannel === "all"
                ? "border-zinc-100 bg-zinc-100 font-semibold text-zinc-950"
                : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
            }`}
          >
            All (||grad||)
          </button>
          {CHANNELS.map((c) => {
            const active = overlayChannel === c;
            return (
              <button
                key={c}
                type="button"
                onClick={() => onChangeOverlayChannel(c)}
                className={`rounded border py-1 font-mono text-[10px] transition ${
                  active
                    ? "border-zinc-100 bg-zinc-100 font-semibold text-zinc-950"
                    : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
                }`}
              >
                {c}
              </button>
            );
          })}
        </div>
      </div>

      {/* Compute Button */}
      <button
        type="button"
        disabled={loading}
        onClick={onCompute}
        className="w-full rounded-md border border-zinc-100 bg-zinc-100 px-3 py-1.5 text-xs font-semibold text-zinc-950 shadow-sm transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Computing Adjoint..." : "Compute Butterfly Map"}
      </button>

      {error && <p role="alert" className="text-xs text-amber-300">Sensitivity unavailable: {error}. Check the backend and try again.</p>}

      {/* Results Card */}
      {result && (
        <div className="flex flex-col gap-2.5 rounded-lg border border-white/[0.08] bg-white/[0.025] p-2.5">
          {/* Physics-Consistent Badge */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between gap-2">
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                  result.self_check.physics_consistent
                    ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-300"
                    : "border-amber-500/30 bg-amber-500/15 text-amber-300"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    result.self_check.physics_consistent ? "bg-emerald-400" : "bg-amber-400"
                  }`}
                />
                {result.self_check.physics_consistent ? "Physics-Consistent" : "Check Warning"}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-zinc-500">
                {result.compute_ms.toFixed(0)} ms
              </span>
            </div>
            <div className="flex justify-between font-mono text-[10px] tabular-nums text-zinc-400">
              <span>FD err {(result.self_check.fd_check.rel_error * 100).toFixed(1)}%</span>
              <span>
                Cone {result.self_check.cone_check.cone_speed_cells_per_day.toFixed(1)} c/d
              </span>
            </div>
          </div>

          {/* Top Precursor Drivers */}
          <div className="border-t border-white/[0.07] pt-2">
            <div className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
              Top Precursor Drivers
            </div>
            <div className="flex flex-col gap-1">
              {sortedDrivers.slice(0, 4).map((ch) => {
                const pct = Math.max(0, Math.min(100, (result.channel_importance[ch] ?? 0) * 100));
                return (
                  <div key={ch} className="flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-8 text-zinc-300">{ch}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-violet-500 via-fuchsia-500 to-amber-400"
                        style={{ width: `${pct.toFixed(1)}%` }}
                      />
                    </div>
                    <span className="w-9 text-right tabular-nums text-zinc-400">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Reverse Shock Summary */}
          <div className="border-t border-white/[0.07] pt-2">
            <div className="mb-1 flex items-center justify-between text-[10px]">
              <span className="font-medium uppercase tracking-wider text-zinc-500">
                Reverse Shock
              </span>
              <span className="font-mono capitalize text-zinc-400">
                {result.reverse_shock.direction}
              </span>
            </div>
            <div className="flex items-center justify-between font-mono text-[11px] tabular-nums text-zinc-200">
              <span>
                {CHANNEL_META[result.target.channel].format(
                  result.reverse_shock.baseline_value,
                  useCelsius,
                )}
              </span>
              <span className="text-zinc-500">→</span>
              <span className="font-semibold text-amber-300">
                {CHANNEL_META[result.target.channel].format(
                  result.reverse_shock.achieved_value,
                  useCelsius,
                )}
              </span>
            </div>
          </div>

          {/* Apply as Forward Shock */}
          <button
            type="button"
            onClick={onApplyReverseShock}
            className="w-full rounded-md border border-white/15 bg-white/[0.08] px-2.5 py-1.5 text-[11px] font-medium text-zinc-100 transition hover:bg-white/[0.14]"
          >
            Apply as Forward Shock →
          </button>
        </div>
      )}
    </section>
  );
}
