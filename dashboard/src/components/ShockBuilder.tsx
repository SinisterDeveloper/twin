import { Flame, Play } from "lucide-react";
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
  onRunBranch: () => void;
}

const PRESET_ORDER: ShockPreset[] = ["none", "null", "warm_ocean", "monsoon_stall", "custom"];

export function ShockBuilder({
  preset,
  custom,
  loading,
  onChangePreset,
  onChangeCustom,
  onRunBranch,
}: ShockBuilderProps) {
  const meta = SHOCK_PRESET_META[preset];
  const spec = customIntensitySpec(custom.channel);

  return (
    <section className="rounded-xl border border-zinc-800/80 bg-zinc-950 p-3.5">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          <Flame className="h-3.5 w-3.5 text-amber-400" />
          Multiverse Shock
        </span>
        {preset !== "none" && (
          <span className="rounded border border-zinc-800 bg-black px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
            {meta.short}
          </span>
        )}
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
              className={`rounded-md border px-2 py-1.5 text-left text-[11px] font-medium transition ${
                key === "custom" ? "col-span-2" : ""
              } ${
                active
                  ? "border-zinc-200 bg-zinc-100 text-zinc-950"
                  : "border-zinc-800/80 bg-black text-zinc-300 hover:border-zinc-700 hover:bg-zinc-900/60"
              }`}
            >
              {m.short}
            </button>
          );
        })}
      </div>

      <p className="mt-2.5 rounded-md border border-zinc-900 bg-black/70 px-2.5 py-2 text-[11px] leading-relaxed text-zinc-400">
        {meta.description}
      </p>

      {preset === "custom" && (
        <div className="mt-3 flex flex-col gap-2.5 border-t border-zinc-900 pt-3 text-xs">
          <div>
            <div className="mb-1 text-[11px] text-zinc-500">Target Channel</div>
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
                        ? "border-zinc-200 bg-zinc-100 font-semibold text-zinc-950"
                        : "border-zinc-800 bg-black text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    {c}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex items-center justify-between rounded-md border border-zinc-800 bg-black px-2.5 py-1.5 font-mono text-[11px] tabular-nums">
            <span className="text-zinc-500">Center (click map)</span>
            <span className="text-amber-300">
              {custom.lat.toFixed(1)}°N, {custom.lon.toFixed(1)}°E
            </span>
          </div>

          <div>
            <div className="mb-1 flex justify-between text-[11px]">
              <span className="text-zinc-500">Radius</span>
              <span className="font-mono tabular-nums text-zinc-200">{custom.radiusDeg.toFixed(1)}°</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={6.0}
              step={0.5}
              value={custom.radiusDeg}
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
              onChange={(e) => onChangeCustom({ ...custom, intensity: Number(e.target.value) })}
              className="shadcn-slider w-full"
            />
          </div>
        </div>
      )}

      <button
        type="button"
        disabled={loading}
        onClick={onRunBranch}
        className="mt-3 flex h-8 w-full items-center justify-center gap-1.5 rounded-md bg-zinc-100 px-3 text-xs font-medium text-zinc-950 transition hover:bg-white disabled:opacity-50"
      >
        <Play className="h-3.5 w-3.5" />
        {loading ? "Simulating..." : preset === "none" ? "Run Timeline Prime" : "Run Branch Comparison"}
      </button>
    </section>
  );
}
