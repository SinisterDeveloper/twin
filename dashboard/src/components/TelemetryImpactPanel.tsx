import { CheckCircle2, CloudRain, ShieldAlert, Sun, Wheat, Wind } from "lucide-react";
import type { ImpactMetrics, PhysicsMetrics, ShockPreset } from "../lib/shocks";

interface TelemetryImpactPanelProps {
  wallMs: number | null;
  serverMs: number | null;
  physics: PhysicsMetrics | null;
  impact: ImpactMetrics | null;
  preset: ShockPreset;
}

function StatusPill({
  label,
  value,
  ok,
}: {
  label: string;
  value: string;
  ok: boolean | null;
}) {
  const tone =
    ok === null
      ? "border-zinc-800 text-zinc-400"
      : ok
        ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-300"
        : "border-rose-500/40 bg-rose-500/10 text-rose-300";

  return (
    <div className={`flex items-center justify-between rounded-lg border bg-black px-3 py-2 text-xs ${tone}`}>
      <span className="text-zinc-500">{label}</span>
      <span className="flex items-center gap-1.5 font-mono font-medium tabular-nums">
        {ok !== null &&
          (ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <ShieldAlert className="h-3.5 w-3.5" />)}
        {value}
      </span>
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
      {/* Physics & Latency Guardrails */}
      <section className="rounded-xl border border-zinc-800/80 bg-zinc-950 p-3.5">
        <div className="mb-2.5 flex items-center justify-between">
          <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            Physics & Contract Guardrails
          </h4>
          {serverMs !== null && (
            <span className="font-mono text-[11px] tabular-nums text-zinc-500">
              Model compute: {serverMs.toFixed(1)} ms
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <StatusPill
            label="Latency ≤ 5000ms"
            value={wallMs !== null ? `${wallMs.toFixed(0)} ms` : "—"}
            ok={wallMs !== null ? wallMs <= 5000 : null}
          />
          <StatusPill
            label="Stability |Δ|/σ ≤ 5"
            value={physics ? `${physics.maxNormDelta.toFixed(3)} σ` : "—"}
            ok={physics ? physics.stable : null}
          />
          <StatusPill
            label="Null Shock ≤ 1e-3"
            value={
              physics && physics.nullOk !== null
                ? `${physics.maxNormDelta.toExponential(1)}`
                : "N/A"
            }
            ok={physics ? physics.nullOk : null}
          />
          <StatusPill
            label="Day-1 Locality ≥ 60%"
            value={
              physics && physics.day1NearFrac !== null
                ? `${(physics.day1NearFrac * 100).toFixed(0)}%`
                : "N/A"
            }
            ok={physics ? physics.localized : null}
          />
        </div>
      </section>

      {/* Live Climate Impact Module */}
      {impact && (
        <section className="rounded-xl border border-zinc-800/80 bg-zinc-950 p-3.5">
          <div className="mb-2.5 flex items-center justify-between">
            <h4 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Live 10-Day Climate Impact Module
            </h4>
            <span className="text-[11px] text-zinc-500">
              {hasBranch ? "Shocked vs. Timeline Prime" : "Timeline Prime Baseline"}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
            {/* Heavy Rain / Flood Risk */}
            <div className="rounded-lg border border-zinc-800/80 bg-black p-3">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Heavy Rain (&gt;10 mm/d)</span>
                <CloudRain className="h-3.5 w-3.5 text-cyan-400" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch ? impact.shockHeavyRainCells : impact.baseHeavyRainCells}
                <span className="ml-1 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                {hasBranch
                  ? `Δ ${impact.shockHeavyRainCells - impact.baseHeavyRainCells >= 0 ? "+" : ""}${
                      impact.shockHeavyRainCells - impact.baseHeavyRainCells
                    } vs prime (${impact.baseHeavyRainCells})`
                  : "Flood exposure threshold"}
              </div>
            </div>

            {/* Heatwave Stress */}
            <div className="rounded-lg border border-zinc-800/80 bg-black p-3">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Heat Stress (&gt;40 °C)</span>
                <Sun className="h-3.5 w-3.5 text-amber-400" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch ? impact.shockHeatwaveCells : impact.baseHeatwaveCells}
                <span className="ml-1 text-[11px] font-normal text-zinc-500">cell-days</span>
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                Peak domain temp: {impact.peakTempC.toFixed(1)} °C
              </div>
            </div>

            {/* Wind Energy Potential */}
            <div className="rounded-lg border border-zinc-800/80 bg-black p-3">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Wind Power Index (v³)</span>
                <Wind className="h-3.5 w-3.5 text-emerald-400" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {hasBranch
                  ? `${impact.windPowerDeltaPct >= 0 ? "+" : ""}${impact.windPowerDeltaPct.toFixed(1)}%`
                  : `${impact.baseMeanWindMs.toFixed(1)} m/s`}
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                Mean speed: {(hasBranch ? impact.shockMeanWindMs : impact.baseMeanWindMs).toFixed(2)} m/s
              </div>
            </div>

            {/* Kharif Crop Stress */}
            <div className="rounded-lg border border-zinc-800/80 bg-black p-3">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Kharif Plains Stress</span>
                <Wheat className="h-3.5 w-3.5 text-yellow-400" />
              </div>
              <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-100">
                {(hasBranch ? impact.shockCropStressPct : impact.baseCropStressPct).toFixed(1)}%
              </div>
              <div className="mt-1 font-mono text-[11px] tabular-nums text-zinc-500">
                {hasBranch
                  ? `Prime: ${impact.baseCropStressPct.toFixed(1)}% (hot & <2 mm/d)`
                  : "15–29°N, 72–86°E dry-heat share"}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
