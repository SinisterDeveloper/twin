import { useCallback, useEffect, useMemo, useState } from "react";
import { Play, RefreshCw } from "lucide-react";
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

interface SwitchRowProps {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}

function SwitchRow({ label, checked, onChange }: SwitchRowProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between py-1 text-xs text-zinc-400 transition hover:text-zinc-200"
    >
      <span>{label}</span>
      <span
        className={`relative inline-flex h-4 w-7 shrink-0 items-center rounded-full border transition ${
          checked
            ? "border-zinc-200 bg-zinc-100"
            : "border-zinc-700 bg-zinc-900"
        }`}
      >
        <span
          className={`inline-block h-2.5 w-2.5 rounded-full transition-transform ${
            checked ? "translate-x-3.5 bg-black" : "translate-x-0.5 bg-zinc-400"
          }`}
        />
      </span>
    </button>
  );
}

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
      setHealthError(err instanceof Error ? err.message : "Offline");
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
    <div className="min-h-screen bg-black text-zinc-50">
      {/* Slim Header */}
      <header className="sticky top-0 z-20 h-14 border-b border-zinc-900 bg-black/80 px-5 backdrop-blur-md">
        <div className="mx-auto flex h-full max-w-[1360px] items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="h-2 w-2 rounded-full bg-zinc-100" />
            <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
              Multiverse of Monsoon
            </h1>
            <span className="hidden text-xs text-zinc-600 sm:inline">/</span>
            <span className="hidden text-xs text-zinc-500 sm:inline">Climate Digital Twin</span>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950 px-2.5 py-1 text-zinc-300">
              <span
                className={`h-1.5 w-1.5 rounded-full ${health ? "bg-emerald-400" : "bg-amber-400"}`}
              />
              {health ? (
                <span className="font-mono text-[11px] tabular-nums">
                  {health.model_name} · {health.latency_ms.toFixed(0)}ms
                </span>
              ) : (
                <span className="font-mono text-[11px] text-zinc-400">
                  {healthError ?? "Offline"} · fallback
                </span>
              )}
            </div>

            {simResult && (
              <span className="hidden rounded-full border border-zinc-800 bg-zinc-950 px-2.5 py-1 font-mono text-[11px] tabular-nums text-zinc-400 sm:inline-block">
                {simResult.source.toUpperCase()} · {simResult.inference_ms.toFixed(1)}ms
              </span>
            )}

            <button
              type="button"
              onClick={() => void checkServerHealth()}
              className="flex h-7 items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-950 px-2.5 text-[11px] font-medium text-zinc-300 transition hover:bg-zinc-900 hover:text-zinc-100"
            >
              <RefreshCw className="h-3 w-3" />
              Health
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="mx-auto grid max-w-[1360px] grid-cols-1 gap-4 p-4 md:grid-cols-[272px_1fr]">
        {/* Minimalist Control Sidebar */}
        <aside className="flex flex-col gap-3">
          <section className="rounded-xl border border-zinc-800/80 bg-zinc-950 p-3.5">
            <div className="mb-2.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Variables
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
                    className={`rounded-md border px-2.5 py-2 text-left transition ${
                      active
                        ? "border-zinc-200 bg-zinc-100 text-zinc-950"
                        : "border-zinc-800/80 bg-black text-zinc-300 hover:border-zinc-700 hover:bg-zinc-900/60"
                    }`}
                  >
                    <div className="font-mono text-xs font-semibold">{m.key}</div>
                    <div className={`truncate text-[10px] ${active ? "text-zinc-600" : "text-zinc-500"}`}>
                      {m.shortLabel.replace(/\s*\(.*\)/, "")}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-3 flex flex-col gap-1.5 border-t border-zinc-900 pt-3">
              <SwitchRow label="Temperature in °C" checked={useCelsius} onChange={setUseCelsius} />
              <SwitchRow label="10m Wind Vectors" checked={showVectors} onChange={setShowVectors} />
              <SwitchRow label="Smooth Grid" checked={smoothGrid} onChange={setSmoothGrid} />
              <SwitchRow label="Ocean Boxes" checked={showBoxes} onChange={setShowBoxes} />
            </div>
          </section>

          <section className="rounded-xl border border-zinc-800/80 bg-zinc-950 p-3.5">
            <div className="mb-2.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Season & Initial State
            </div>
            <div className="flex flex-col gap-1">
              {SEASON_PRESETS.map((p) => {
                const active = doy === p.doy;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => {
                      setDoy(p.doy);
                      setDay(0);
                    }}
                    className={`flex items-center justify-between rounded-md border px-2.5 py-1.5 text-left text-xs transition ${
                      active
                        ? "border-zinc-200 bg-zinc-100 text-zinc-950"
                        : "border-transparent bg-transparent text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
                    }`}
                  >
                    <span className="font-medium">{p.name}</span>
                    <span className={`font-mono text-[10px] tabular-nums ${active ? "text-zinc-600" : "text-zinc-600"}`}>
                      {p.doy}
                    </span>
                  </button>
                );
              })}
            </div>

            <div className="mt-3 border-t border-zinc-900 pt-3">
              <div className="mb-1.5 flex justify-between text-xs">
                <span className="text-zinc-500">Day of Year</span>
                <span className="font-mono tabular-nums text-zinc-200">{doy} / 365</span>
              </div>
              <input
                type="range"
                min={1}
                max={365}
                value={doy}
                onChange={(e) => setDoy(Number(e.target.value))}
                className="shadcn-slider w-full"
              />
            </div>

            <div className="mt-3 flex flex-col gap-2.5 border-t border-zinc-900 pt-3">
              <SwitchRow label="Offline Demo Mode" checked={preferMock} onChange={setPreferMock} />

              <button
                type="button"
                disabled={loadingSim}
                onClick={() => void runSimulation()}
                className="flex h-8 w-full items-center justify-center gap-1.5 rounded-md bg-zinc-100 px-3 text-xs font-medium text-zinc-950 transition hover:bg-white disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                {loadingSim ? "Simulating..." : "Run Rollout"}
              </button>
            </div>
          </section>
        </aside>

        {/* Map, Scrubber & Metrics */}
        <section className="flex flex-col gap-3">
          <IndiaGridMap
            stateGrid={activeGrid}
            channel={channel}
            useCelsius={useCelsius}
            showVectors={showVectors}
            smoothGrid={smoothGrid}
            showBoxes={showBoxes}
            title={`Timeline Prime · ${meta.label}`}
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

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-zinc-800/80 bg-zinc-950 px-3.5 py-2.5">
              <div className="text-[11px] text-zinc-500">Min</div>
              <div className="mt-0.5 font-mono text-sm font-medium tabular-nums text-zinc-100">
                {stats.min.toFixed(2)} <span className="text-xs text-zinc-500">{unit}</span>
              </div>
            </div>
            <div className="rounded-xl border border-zinc-800/80 bg-zinc-950 px-3.5 py-2.5">
              <div className="text-[11px] text-zinc-500">Mean</div>
              <div className="mt-0.5 font-mono text-sm font-medium tabular-nums text-zinc-100">
                {stats.mean.toFixed(2)} <span className="text-xs text-zinc-500">{unit}</span>
              </div>
            </div>
            <div className="rounded-xl border border-zinc-800/80 bg-zinc-950 px-3.5 py-2.5">
              <div className="text-[11px] text-zinc-500">Max</div>
              <div className="mt-0.5 font-mono text-sm font-medium tabular-nums text-zinc-100">
                {stats.max.toFixed(2)} <span className="text-xs text-zinc-500">{unit}</span>
              </div>
            </div>
            <div className="rounded-xl border border-zinc-800/80 bg-zinc-950 px-3.5 py-2.5">
              <div className="text-[11px] text-zinc-500">Latency</div>
              <div className="mt-0.5 font-mono text-sm font-medium tabular-nums text-zinc-100">
                {simResult ? simResult.wall_ms.toFixed(0) : "—"} <span className="text-xs text-zinc-500">ms</span>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
