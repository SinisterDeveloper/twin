import { CheckCircle2, CloudRain, ShieldAlert, Sun, Wheat, Wind } from "lucide-react";
import type { ImpactMetrics, PhysicsMetrics, ShockPreset } from "../lib/shocks";

interface TelemetryImpactPanelProps {
  wallMs: number | null;
  serverMs: number | null;
  physics: PhysicsMetrics | null;
  impact: ImpactMetrics | null;
  preset: ShockPreset;
}

function GuardrailCell({
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
  const valueTone =
    ok === null ? "text-zinc-400" : ok ? "text-emerald-400" : "text-rose-400";

  return (
    <div className="flex flex-col gap-1 px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-zinc-500">{label}</span>
        {ok !== null &&
          (ok ? (
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
          ) : (
            <ShieldAlert className="h-3.5 w-3.5 shrink-0 text-rose-400" />
          ))}
      </div>
      <div className={`font-mono text-base font-semibold tabular-nums ${valueTone}`}>{value}</div>
      <div className="font-mono text-[10px] text-zinc-600">{sub}</div>
    </div>
  );
}

export function TelemetryImpactPanel({
  wallMs,
  serverMs,
  physics,
  impact,
  preset,
}: TelemetryImpactPanelProps) {
  const hasBranch = preset !== "none";

  return (
    <div className="flex flex-col gap-3">
      {/* Physics & Contract Guardrails */}
      <section className="overflow-hidden rounded-xl border border-zinc-900 bg-zinc-950/60">
        <div className="flex items-center justify-between border-b border-zinc-900 px-4 py-2">
          <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            Physics & Contract Guardrails
          </h4>
          {serverMs !== null && (
            <span className="font-mono text-[11px] tabular-nums text-zinc-500">
              Model compute · {serverMs.toFixed(1)} ms
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 divide-y divide-zinc-900 sm:grid-cols-4 sm:divide-x sm:divide-y-0">
          <GuardrailCell
            label="Latency"
            value={wallMs !== null ? `${wallMs.toFixed(0)} ms` : "—"}
            sub="Limit ≤ 5000 ms"
            ok={wallMs !== null ? wallMs <= 5000 : null}
          />
          <GuardrailCell
            label="Stability"
            value={physics ? `${physics.maxNormDelta.toFixed(3)} σ` : "—"}
            sub="Bound |Δ|/σ ≤ 5"
            ok={physics ? physics.stable : null}
          />
          <GuardrailCell
            label="Null Shock"
            value={
              physics && physics.nullOk !== null
                ? `${physics.maxNormDelta.toExponential(1)}`
                : "N/A"
            }
            sub="Tolerance ≤ 1e-3"
            ok={physics ? physics.nullOk : null}
          />
          <GuardrailCell
            label="Day-1 Locality"
            value={
              physics && physics.day1NearFrac !== null
                ? `${(physics.day1NearFrac * 100).toFixed(0)}%`
                : "N/A"
            }
            sub="Near-box ≥ 60%"
            ok={physics ? physics.localized : null}
          />
        </div>
      </section>

      {/* Live Climate Impact Module */}
      {impact && (
        <section className="overflow-hidden rounded-xl border border-zinc-900 bg-zinc-950/60">
          <div className="flex items-center justify-between border-b border-zinc-900 px-4 py-2">
            <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              10-Day Climate Impact
            </h4>
            <span className="text-[11px] text-zinc-500">
              {hasBranch ? "Shocked vs. Timeline Prime" : "Timeline Prime Baseline"}
            </span>
          </div>

          <div className="grid grid-cols-1 divide-y divide-zinc-900 sm:grid-cols-2 lg:grid-cols-4 lg:divide-x lg:divide-y-0">
            {/* Heavy Rain / Flood Risk */}
            <div className="px-4 py-3">
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span>Heavy Rain (&gt;10 mm/d)</span>
                <CloudRain className="h-3.5 w-3.5 text-cyan-400/80" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch ? impact.shockHeavyRainCells : impact.baseHeavyRainCells}
                <span className="ml-1.5 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-600">
                {hasBranch
                  ? `Δ ${impact.shockHeavyRainCells - impact.baseHeavyRainCells >= 0 ? "+" : ""}${
                      impact.shockHeavyRainCells - impact.baseHeavyRainCells
                    } vs prime (${impact.baseHeavyRainCells})`
                  : "Flood exposure threshold"}
              </div>
            </div>

            {/* Heatwave Stress */}
            <div className="px-4 py-3">
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span>Heat Stress (&gt;40 °C)</span>
                <Sun className="h-3.5 w-3.5 text-amber-400/80" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch ? impact.shockHeatwaveCells : impact.baseHeatwaveCells}
                <span className="ml-1.5 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-600">
                Peak domain temp · {impact.peakTempC.toFixed(1)} °C
              </div>
            </div>

            {/* Wind Energy Potential */}
            <div className="px-4 py-3">
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span>Wind Power Index (v³)</span>
                <Wind className="h-3.5 w-3.5 text-emerald-400/80" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch
                  ? `${impact.windPowerDeltaPct >= 0 ? "+" : ""}${impact.windPowerDeltaPct.toFixed(1)}%`
                  : `${impact.baseMeanWindMs.toFixed(1)} m/s`}
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-600">
                Mean speed · {(hasBranch ? impact.shockMeanWindMs : impact.baseMeanWindMs).toFixed(2)} m/s
              </div>
            </div>

            {/* Kharif Crop Stress */}
            <div className="px-4 py-3">
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span>Kharif Plains Stress</span>
                <Wheat className="h-3.5 w-3.5 text-yellow-400/80" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {(hasBranch ? impact.shockCropStressPct : impact.baseCropStressPct).toFixed(1)}%
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-600">
                {hasBranch
                  ? `Prime ${impact.baseCropStressPct.toFixed(1)}% (hot & <2 mm/d)`
                  : "15–29°N, 72–86°E dry-heat share"}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
