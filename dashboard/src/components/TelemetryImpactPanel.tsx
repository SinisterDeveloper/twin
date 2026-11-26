import type { ImpactMetrics, PhysicsMetrics, ShockPreset } from "../lib/shocks";

interface TelemetryImpactPanelProps {
  wallMs: number | null;
  physics: PhysicsMetrics | null;
  impact: ImpactMetrics | null;
  preset: ShockPreset;
}

export function TelemetryImpactPanel({
  wallMs,
  physics,
  impact,
  preset,
}: TelemetryImpactPanelProps) {
  const hasBranch = preset !== "none";

  const validationItems: { label: string; value: string; ok: boolean | null }[] = [
    {
      label: "Latency",
      value: wallMs !== null ? `${wallMs.toFixed(0)} ms` : "—",
      ok: wallMs !== null ? wallMs <= 5000 : null,
    },
    {
      label: "Stability",
      value: physics ? `${physics.maxNormDelta.toFixed(3)} σ` : "—",
      ok: physics ? physics.stable : null,
    },
    {
      label: "Null Shock",
      value:
        physics && physics.nullOk !== null ? physics.maxNormDelta.toExponential(1) : "N/A",
      ok: physics ? physics.nullOk : null,
    },
    {
      label: "Day-1 Locality",
      value:
        physics && physics.day1NearFrac !== null
          ? `${(physics.day1NearFrac * 100).toFixed(0)}%`
          : "N/A",
      ok: physics ? physics.localized : null,
    },
  ];

  return (
    <div className="flex flex-col gap-2.5">
      {impact && (
        <section className="surface-panel overflow-hidden rounded-xl">
          <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-2">
            <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
              10-Day Impact
            </h4>
            {hasBranch && (
              <span className="text-[11px] text-zinc-500">Shocked vs. Baseline</span>
            )}
          </div>

          <div className="grid grid-cols-1 divide-y divide-white/[0.06] sm:grid-cols-2 lg:grid-cols-4 lg:divide-x lg:divide-y-0">
            <div className="px-4 py-3">
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Heavy Rain (&gt;10 mm/d)
              </div>
              <div className="mt-1 font-mono text-xl font-semibold tracking-tight tabular-nums text-zinc-50">
                {hasBranch ? impact.shockHeavyRainCells : impact.baseHeavyRainCells}
                <span className="ml-1.5 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              {hasBranch && (
                <div className="mt-0.5 font-mono text-[11px] tabular-nums text-zinc-500">
                  Δ {impact.shockHeavyRainCells - impact.baseHeavyRainCells >= 0 ? "+" : ""}
                  {impact.shockHeavyRainCells - impact.baseHeavyRainCells} vs {impact.baseHeavyRainCells}
                </div>
              )}
            </div>

            <div className="px-4 py-3">
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Heat Stress (&gt;40 °C)
              </div>
              <div className="mt-1 font-mono text-xl font-semibold tracking-tight tabular-nums text-zinc-50">
                {hasBranch ? impact.shockHeatwaveCells : impact.baseHeatwaveCells}
                <span className="ml-1.5 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              <div className="mt-0.5 font-mono text-[11px] tabular-nums text-zinc-500">
                Peak {impact.peakTempC.toFixed(1)} °C
              </div>
            </div>

            <div className="px-4 py-3">
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Wind Power (v³)
              </div>
              <div className="mt-1 font-mono text-xl font-semibold tracking-tight tabular-nums text-zinc-50">
                {hasBranch
                  ? `${impact.windPowerDeltaPct >= 0 ? "+" : ""}${impact.windPowerDeltaPct.toFixed(1)}%`
                  : `${impact.baseMeanWindMs.toFixed(1)} m/s`}
              </div>
              <div className="mt-0.5 font-mono text-[11px] tabular-nums text-zinc-500">
                Mean {(hasBranch ? impact.shockMeanWindMs : impact.baseMeanWindMs).toFixed(2)} m/s
              </div>
            </div>

            <div className="px-4 py-3">
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Crop Stress
              </div>
              <div className="mt-1 font-mono text-xl font-semibold tracking-tight tabular-nums text-zinc-50">
                {(hasBranch ? impact.shockCropStressPct : impact.baseCropStressPct).toFixed(1)}%
              </div>
              {hasBranch && (
                <div className="mt-0.5 font-mono text-[11px] tabular-nums text-zinc-500">
                  Baseline {impact.baseCropStressPct.toFixed(1)}%
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      <section className="surface-panel flex flex-wrap items-center justify-between gap-x-6 gap-y-2 rounded-lg px-4 py-2">
        <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
          Validation
        </span>
        <div className="flex flex-1 flex-wrap items-center justify-end gap-x-6 gap-y-1.5">
          {validationItems.map((item) => {
            const tone =
              item.ok === false
                ? "text-rose-400"
                : item.ok === null
                  ? "text-zinc-500"
                  : "text-zinc-200";
            return (
              <div key={item.label} className="flex items-baseline gap-2">
                <span className="text-[11px] text-zinc-500">{item.label}</span>
                <span className={`font-mono text-xs font-medium tabular-nums ${tone}`}>
                  {item.value}
                </span>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
