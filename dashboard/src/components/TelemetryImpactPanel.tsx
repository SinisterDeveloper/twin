import type { ImpactMetrics, PhysicsMetrics, ShockPreset } from "../lib/shocks";

interface TelemetryImpactPanelProps {
  wallMs: number | null;
  physics: PhysicsMetrics | null;
  impact: ImpactMetrics | null;
  preset: ShockPreset;
}

function ValidationCell({
  label,
  value,
  sub,
  ok,
}: {
  label: string;
  value: string;
  sub: string;
  ok: boolean | null;
}) {
  const valueTone = ok === false ? "text-rose-400" : ok === null ? "text-zinc-400" : "text-zinc-100";

  return (
    <div className="flex flex-col gap-1 px-4 py-3">
      <span className="text-[11px] text-zinc-500">{label}</span>
      <div className={`font-mono text-base font-semibold tabular-nums ${valueTone}`}>{value}</div>
      <div className="font-mono text-[10px] text-zinc-600">{sub}</div>
    </div>
  );
}

export function TelemetryImpactPanel({
  wallMs,
  physics,
  impact,
  preset,
}: TelemetryImpactPanelProps) {
  const hasBranch = preset !== "none";

  return (
    <div className="flex flex-col gap-3">
      <section className="overflow-hidden rounded-lg border border-zinc-900 bg-black">
        <div className="border-b border-zinc-900 px-4 py-2">
          <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            Validation
          </h4>
        </div>
        <div className="grid grid-cols-2 divide-y divide-zinc-900 sm:grid-cols-4 sm:divide-x sm:divide-y-0">
          <ValidationCell
            label="Latency"
            value={wallMs !== null ? `${wallMs.toFixed(0)} ms` : "—"}
            sub="≤ 5000 ms"
            ok={wallMs !== null ? wallMs <= 5000 : null}
          />
          <ValidationCell
            label="Stability"
            value={physics ? `${physics.maxNormDelta.toFixed(3)} σ` : "—"}
            sub="|Δ|/σ ≤ 5"
            ok={physics ? physics.stable : null}
          />
          <ValidationCell
            label="Null Shock"
            value={
              physics && physics.nullOk !== null
                ? `${physics.maxNormDelta.toExponential(1)}`
                : "N/A"
            }
            sub="≤ 1e-3"
            ok={physics ? physics.nullOk : null}
          />
          <ValidationCell
            label="Day-1 Locality"
            value={
              physics && physics.day1NearFrac !== null
                ? `${(physics.day1NearFrac * 100).toFixed(0)}%`
                : "N/A"
            }
            sub="≥ 60%"
            ok={physics ? physics.localized : null}
          />
        </div>
      </section>

      {impact && (
        <section className="overflow-hidden rounded-lg border border-zinc-900 bg-black">
          <div className="flex items-center justify-between border-b border-zinc-900 px-4 py-2">
            <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              10-Day Impact
            </h4>
            {hasBranch && <span className="text-[11px] text-zinc-500">Shocked vs. Baseline</span>}
          </div>

          <div className="grid grid-cols-1 divide-y divide-zinc-900 sm:grid-cols-2 lg:grid-cols-4 lg:divide-x lg:divide-y-0">
            <div className="px-4 py-3">
              <div className="text-xs text-zinc-500">Heavy Rain (&gt;10 mm/d)</div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch ? impact.shockHeavyRainCells : impact.baseHeavyRainCells}
                <span className="ml-1.5 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              {hasBranch && (
                <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                  Δ {impact.shockHeavyRainCells - impact.baseHeavyRainCells >= 0 ? "+" : ""}
                  {impact.shockHeavyRainCells - impact.baseHeavyRainCells} vs {impact.baseHeavyRainCells}
                </div>
              )}
            </div>

            <div className="px-4 py-3">
              <div className="text-xs text-zinc-500">Heat Stress (&gt;40 °C)</div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch ? impact.shockHeatwaveCells : impact.baseHeatwaveCells}
                <span className="ml-1.5 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                Peak {impact.peakTempC.toFixed(1)} °C
              </div>
            </div>

            <div className="px-4 py-3">
              <div className="text-xs text-zinc-500">Wind Power (v³)</div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch
                  ? `${impact.windPowerDeltaPct >= 0 ? "+" : ""}${impact.windPowerDeltaPct.toFixed(1)}%`
                  : `${impact.baseMeanWindMs.toFixed(1)} m/s`}
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                Mean {(hasBranch ? impact.shockMeanWindMs : impact.baseMeanWindMs).toFixed(2)} m/s
              </div>
            </div>

            <div className="px-4 py-3">
              <div className="text-xs text-zinc-500">Crop Stress</div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {(hasBranch ? impact.shockCropStressPct : impact.baseCropStressPct).toFixed(1)}%
              </div>
              {hasBranch && (
                <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                  Baseline {impact.baseCropStressPct.toFixed(1)}%
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
