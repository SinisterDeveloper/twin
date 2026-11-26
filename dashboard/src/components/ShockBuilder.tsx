import type { CSSProperties } from "react";
import { CHANNELS, CHANNEL_META, type ChannelKey } from "../lib/contract";
import {
  customIntensitySpec,
  SHOCK_PRESET_META,
  type CustomShockConfig,
  type ShockPreset,
} from "../lib/shocks";

interface ShockBuilderProps {
  preset: ShockPreset;
  custom: CustomShockConfig;
  loading: boolean;
  onChangePreset: (p: ShockPreset) => void;
  onChangeCustom: (c: CustomShockConfig) => void;
}

const PRESET_ORDER: ShockPreset[] = ["none", "null", "warm_ocean", "monsoon_stall", "custom"];

export function ShockBuilder({
  preset,
  custom,
  loading,
  onChangePreset,
  onChangeCustom,
}: ShockBuilderProps) {
  const meta = SHOCK_PRESET_META[preset];
  const spec = customIntensitySpec(custom.channel);
  const radiusPct = `${((custom.radiusDeg - 1.0) / (6.0 - 1.0)) * 100}%`;
  const intensityPct = `${((custom.intensity - spec.min) / (spec.max - spec.min)) * 100}%`;

  return (
    <section className="flex flex-col gap-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          Shock
        </span>
        {loading && <span className="font-mono text-[10px] text-zinc-400">Running...</span>}
      </div>

      <div className="grid grid-cols-2 gap-1.5">
        {PRESET_ORDER.map((key) => {
          const active = key === preset;
          const m = SHOCK_PRESET_META[key];
          return (
            <button
              key={key}
              type="button"
              onClick={() => onChangePreset(key)}
              className={`rounded-md border px-2.5 py-1.5 text-left text-[11px] font-medium transition ${
                key === "custom" ? "col-span-2" : ""
              } ${
                active
                  ? "border-white/20 bg-white/[0.10] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]"
                  : "border-white/[0.06] bg-white/[0.02] text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
              }`}
            >
              {m.short}
            </button>
          );
        })}
      </div>

      <p className="text-[11px] leading-relaxed text-zinc-500">{meta.description}</p>

      {preset === "custom" && (
        <div className="flex flex-col gap-2.5 border-t border-white/[0.07] pt-2.5 text-xs">
          <div>
            <div className="mb-1.5 text-[11px] text-zinc-500">Target Channel</div>
            <div className="grid grid-cols-3 gap-1">
              {CHANNELS.map((c: ChannelKey) => {
                const active = custom.channel === c;
                return (
                  <button
                    key={c}
                    type="button"
                    onClick={() => {
                      const nextSpec = customIntensitySpec(c);
                      onChangeCustom({
                        ...custom,
                        channel: c,
                        intensity: nextSpec.defaultVal,
                      });
                    }}
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

          <div className="flex items-center justify-between rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-1.5 font-mono text-[11px] tabular-nums">
            <span className="text-zinc-500">Center (click map)</span>
            <span className="text-zinc-200">
              {custom.lat.toFixed(1)}°N, {custom.lon.toFixed(1)}°E
            </span>
          </div>

          <div>
            <div className="mb-1 flex justify-between text-[11px]">
              <span className="text-zinc-500">Radius</span>
              <span className="font-mono tabular-nums text-zinc-200">
                {custom.radiusDeg.toFixed(1)}°
              </span>
            </div>
            <input
              type="range"
              min={1.0}
              max={6.0}
              step={0.5}
              value={custom.radiusDeg}
              style={{ "--slider-pct": radiusPct } as CSSProperties}
              onChange={(e) => onChangeCustom({ ...custom, radiusDeg: Number(e.target.value) })}
              className="shadcn-slider w-full"
            />
          </div>

          <div>
            <div className="mb-1 flex justify-between text-[11px]">
              <span className="text-zinc-500">
                Intensity ({CHANNEL_META[custom.channel].key})
              </span>
              <span className="font-mono tabular-nums text-zinc-200">
                {custom.intensity >= 0 ? "+" : ""}
                {custom.intensity} {spec.unit}
              </span>
            </div>
            <input
              type="range"
              min={spec.min}
              max={spec.max}
              step={spec.step}
              value={custom.intensity}
              style={{ "--slider-pct": intensityPct } as CSSProperties}
              onChange={(e) => onChangeCustom({ ...custom, intensity: Number(e.target.value) })}
              className="shadcn-slider w-full"
            />
          </div>
        </div>
      )}
    </section>
  );
}
