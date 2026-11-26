import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, CloudRain, Compass, Layers, RefreshCw, Thermometer, Wind } from "lucide-react";
import { DayScrubber } from "./components/DayScrubber";
import { IndiaGridMap } from "./components/IndiaGridMap";
import {
  fetchHealth,
  mockSimulate,
  postSimulate,
  type HealthResponse,
  type SimulateResponse,
} from "./lib/api";
import {
  buildDefaultInitStates,
  CHANNEL_META,
  CHANNELS,
  computeFieldStats,
  doySinCos,
  SCORE_STEPS,
  SEASON_PRESETS,
  type ChannelKey,
} from "./lib/contract";

export default function App() {
  const [doy, setDoy] = useState<number>(160);
  const [channel, setChannel] = useState<ChannelKey>("tp");
  const [useCelsius, setUseCelsius] = useState<boolean>(true);
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [smoothGrid, setSmoothGrid] = useState<boolean>(false);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  const [day, setDay] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<SimulateResponse | null>(null);
  const [loadingSim, setLoadingSim] = useState<boolean>(false);
  const [preferMock, setPreferMock] = useState<boolean>(false);

  const initStates = useMemo(() => buildDefaultInitStates(doy), [doy]);

  const checkServerHealth = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
      setHealthError(null);
      return true;
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : "Server unreachable");
      return false;
    }
  }, []);

  const runSimulation = useCallback(async () => {
    setLoadingSim(true);
    const req = {
      init_states: initStates,
      doy_sincos: doySinCos(doy),
      steps: SCORE_STEPS,
      perturbation: null,
    };
    try {
      if (preferMock) {
        setSimResult(mockSimulate(req));
      } else {
        const res = await postSimulate(req);
        setSimResult(res);
        setHealthError(null);
      }
    } catch {
      // Fall back to local mock so map & timeline remain responsive offline
      setSimResult(mockSimulate(req));
    } finally {
      setLoadingSim(false);
    }
  }, [initStates, doy, preferMock]);

  useEffect(() => {
    void checkServerHealth();
  }, [checkServerHealth]);

  useEffect(() => {
    void runSimulation();
  }, [runSimulation]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setDay((prev) => (prev >= SCORE_STEPS ? 0 : prev + 1));
    }, Math.round(750 / speed));
    return () => window.clearInterval(id);
  }, [playing, speed]);

  const activeGrid = useMemo(() => {
    if (day === 0 || !simResult) return initStates[initStates.length - 1];
    return simResult.rollout[day - 1] ?? initStates[initStates.length - 1];
  }, [day, simResult, initStates]);

  const stats = useMemo(
    () => computeFieldStats(activeGrid[CHANNEL_META[channel].index], channel, useCelsius),
    [activeGrid, channel, useCelsius],
  );

  const meta = CHANNEL_META[channel];
  const unit = meta.displayUnit(useCelsius);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Mission-Control Bar */}
      <header className="border-b border-slate-800/90 bg-slate-900/80 px-5 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-400 ring-1 ring-cyan-500/30">
              <CloudRain className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-slate-100">
                Multiverse of Monsoon
              </h1>
              <p className="text-xs text-slate-400">
                India Climate Digital Twin · 32×36 1° Physical Grid
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 text-xs">
            <div
              className={`flex items-center gap-2 rounded-full border px-3 py-1 ${
                health
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : "border-amber-500/40 bg-amber-500/10 text-amber-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${health ? "bg-emerald-400" : "bg-amber-400"}`}
              />
              {health ? (
                <span>
                  {health.model_name} · {health.latency_ms.toFixed(0)} ms
                </span>
              ) : (
                <span>{healthError ?? "Offline"} (using local demo fallback)</span>
              )}
            </div>

            {simResult && (
              <span className="rounded-full border border-slate-700 bg-slate-800/90 px-3 py-1 font-mono text-slate-300">
                {simResult.source.toUpperCase()} · {simResult.inference_ms.toFixed(1)} ms server
              </span>
            )}

            <button
              type="button"
              onClick={() => void checkServerHealth()}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-slate-200 hover:bg-slate-700"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Health
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto grid max-w-[1440px] grid-cols-1 gap-4 p-4 lg:grid-cols-[310px_1fr]">
        {/* Left Controls */}
        <aside className="flex flex-col gap-4">
          <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 shadow-lg">
            <div className="mb-2.5 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <Layers className="h-3.5 w-3.5 text-cyan-400" />
              Channel Selection
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {CHANNELS.map((key) => {
                const m = CHANNEL_META[key];
                const active = key === channel;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setChannel(key)}
                    className={`rounded-lg border px-2.5 py-2 text-left text-xs transition ${
                      active
                        ? "border-cyan-400/70 bg-cyan-500/15 text-cyan-100"
                        : "border-slate-800 bg-slate-950/60 text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="font-semibold">{m.key}</div>
                    <div className="truncate text-[10px] text-slate-400">{m.shortLabel}</div>
                  </button>
                );
              })}
            </div>

            <div className="mt-3 flex flex-col gap-2 border-t border-slate-800 pt-3 text-xs">
              <label className="flex cursor-pointer items-center justify-between text-slate-300">
                <span className="flex items-center gap-1.5">
                  <Thermometer className="h-3.5 w-3.5 text-slate-400" />
                  Temperature in °C
                </span>
                <input
                  type="checkbox"
                  checked={useCelsius}
                  onChange={(e) => setUseCelsius(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>

              <label className="flex cursor-pointer items-center justify-between text-slate-300">
                <span className="flex items-center gap-1.5">
                  <Wind className="h-3.5 w-3.5 text-slate-400" />
                  10m Wind Vectors
                </span>
                <input
                  type="checkbox"
                  checked={showVectors}
                  onChange={(e) => setShowVectors(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>

              <label className="flex cursor-pointer items-center justify-between text-slate-300">
                <span>Smooth Interpolation</span>
                <input
                  type="checkbox"
                  checked={smoothGrid}
                  onChange={(e) => setSmoothGrid(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>

              <label className="flex cursor-pointer items-center justify-between text-slate-300">
                <span>Show BoB / Arabian Boxes</span>
                <input
                  type="checkbox"
                  checked={showBoxes}
                  onChange={(e) => setShowBoxes(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>
            </div>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 shadow-lg">
            <div className="mb-2.5 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <Compass className="h-3.5 w-3.5 text-cyan-400" />
              Seasonal Regime (DOY)
            </div>
            <div className="flex flex-col gap-1.5">
              {SEASON_PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    setDoy(p.doy);
                    setDay(0);
                  }}
                  className={`rounded-lg border px-2.5 py-2 text-left text-xs transition ${
                    doy === p.doy
                      ? "border-cyan-400/70 bg-cyan-500/15 text-cyan-100"
                      : "border-slate-800 bg-slate-950/60 text-slate-300 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between font-semibold">
                    <span>{p.name}</span>
                    <span className="font-mono text-[10px] text-cyan-300">DOY {p.doy}</span>
                  </div>
                  <div className="mt-0.5 text-[10px] text-slate-400">{p.note}</div>
                </button>
              ))}
            </div>

            <div className="mt-3 border-t border-slate-800 pt-3">
              <div className="mb-1 flex justify-between text-xs text-slate-300">
                <span>Custom DOY</span>
                <span className="font-mono text-cyan-300">{doy} / 365</span>
              </div>
              <input
                type="range"
                min={1}
                max={365}
                value={doy}
                onChange={(e) => setDoy(Number(e.target.value))}
                className="h-2 w-full cursor-pointer accent-cyan-400"
              />
            </div>

            <div className="mt-3 flex flex-col gap-2 border-t border-slate-800 pt-3">
              <label className="flex cursor-pointer items-center justify-between text-xs text-slate-300">
                <span>Force Offline Demo Mode</span>
                <input
                  type="checkbox"
                  checked={preferMock}
                  onChange={(e) => setPreferMock(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>

              <button
                type="button"
                disabled={loadingSim}
                onClick={() => void runSimulation()}
                className="mt-1 flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-500 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
              >
                <Activity className="h-4 w-4" />
                {loadingSim ? "Running /simulate..." : "Run 10-Day Simulation"}
              </button>
            </div>
          </section>
        </aside>

        {/* Map & Timeline Workspace */}
        <section className="flex flex-col gap-4">
          <IndiaGridMap
            stateGrid={activeGrid}
            channel={channel}
            useCelsius={useCelsius}
            showVectors={showVectors}
            smoothGrid={smoothGrid}
            showBoxes={showBoxes}
            title={`Timeline Prime — ${meta.label}`}
            subtitle={day === 0 ? "Initial State (t = 0)" : `Autoregressive Rollout · Day +${day}`}
          />

          <DayScrubber
            day={day}
            maxDay={SCORE_STEPS}
            playing={playing}
            speed={speed}
            doy={doy}
            onChangeDay={(d) => {
              setPlaying(false);
              setDay(d);
            }}
            onTogglePlay={() => setPlaying((p) => !p)}
            onChangeSpeed={setSpeed}
          />

          {/* Field Summary Strip */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
              <div className="text-[11px] text-slate-400">Domain Min</div>
              <div className="mt-0.5 font-mono text-base font-semibold text-slate-100">
                {stats.min.toFixed(2)} {unit}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
              <div className="text-[11px] text-slate-400">Domain Mean</div>
              <div className="mt-0.5 font-mono text-base font-semibold text-cyan-300">
                {stats.mean.toFixed(2)} {unit}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
              <div className="text-[11px] text-slate-400">Domain Max</div>
              <div className="mt-0.5 font-mono text-base font-semibold text-amber-300">
                {stats.max.toFixed(2)} {unit}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
              <div className="text-[11px] text-slate-400">Rollout Latency</div>
              <div className="mt-0.5 font-mono text-base font-semibold text-emerald-300">
                {simResult ? `${simResult.wall_ms.toFixed(0)} ms` : "—"}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
