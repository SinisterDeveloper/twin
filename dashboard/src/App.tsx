import { useCallback, useEffect, useMemo, useState } from "react";
import { Columns3, Maximize2, RefreshCw } from "lucide-react";
import { DayScrubber } from "./components/DayScrubber";
import { IndiaGridMap } from "./components/IndiaGridMap";
import { ShockBuilder } from "./components/ShockBuilder";
import { TelemetryImpactPanel } from "./components/TelemetryImpactPanel";
import {
  fetchHealth,
  mockSimulate,
  postSimulate,
  type HealthResponse,
  type SimulateRequest,
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
import {
  buildPerturbation,
  computeDeltaGrid,
  computeImpactMetrics,
  computePhysicsMetrics,
  SHOCK_PRESET_META,
  type CustomShockConfig,
  type ShockPreset,
} from "./lib/shocks";

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
          checked ? "border-zinc-200 bg-zinc-100" : "border-zinc-700 bg-zinc-900"
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
  const [smoothGrid, setSmoothGrid] = useState<boolean>(true);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showCities, setShowCities] = useState<boolean>(true);

  const [shockPreset, setShockPreset] = useState<ShockPreset>("none");
  const [customShock, setCustomShock] = useState<CustomShockConfig>({
    channel: "t2m",
    lat: 15.5,
    lon: 73.5,
    radiusDeg: 2.5,
    intensity: 3.0,
  });
  const [viewMode, setViewMode] = useState<"single" | "compare">("single");

  const [day, setDay] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [baseResult, setBaseResult] = useState<SimulateResponse | null>(null);
  const [shockResult, setShockResult] = useState<SimulateResponse | null>(null);
  const [loadingSim, setLoadingSim] = useState<boolean>(false);
  const [preferMock, setPreferMock] = useState<boolean>(false);

  const initStates = useMemo(() => buildDefaultInitStates(doy), [doy]);
  const perturbation = useMemo(
    () => buildPerturbation(shockPreset, initStates[initStates.length - 1], customShock),
    [shockPreset, initStates, customShock],
  );

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

  const executeSim = useCallback(
    async (req: SimulateRequest): Promise<SimulateResponse> => {
      if (preferMock) return mockSimulate(req);
      try {
        return await postSimulate(req);
      } catch {
        return mockSimulate(req);
      }
    },
    [preferMock],
  );

  const runSimulation = useCallback(async () => {
    setLoadingSim(true);
    const sc = doySinCos(doy);
    const baseReq: SimulateRequest = {
      init_states: initStates,
      doy_sincos: sc,
      steps: SCORE_STEPS,
      perturbation: null,
    };
    try {
      if (perturbation) {
        const shockReq: SimulateRequest = {
          init_states: initStates,
          doy_sincos: sc,
          steps: SCORE_STEPS,
          perturbation,
        };
        const [bRes, sRes] = await Promise.all([executeSim(baseReq), executeSim(shockReq)]);
        setBaseResult(bRes);
        setShockResult(sRes);
      } else {
        const bRes = await executeSim(baseReq);
        setBaseResult(bRes);
        setShockResult(null);
      }
    } finally {
      setLoadingSim(false);
    }
  }, [initStates, doy, perturbation, executeSim]);

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

  const handlePresetChange = (next: ShockPreset) => {
    setShockPreset(next);
    if (next === "none") {
      setViewMode("single");
    } else {
      setViewMode("compare");
      if (day === 0) setDay(1);
    }
  };

  const baseActiveGrid = useMemo(() => {
    if (day === 0 || !baseResult) return initStates[initStates.length - 1];
    return baseResult.rollout[day - 1] ?? initStates[initStates.length - 1];
  }, [day, baseResult, initStates]);

  const shockActiveGrid = useMemo(() => {
    if (!perturbation) return baseActiveGrid;
    if (day === 0) {
      return initStates[initStates.length - 1].map((ch, c) =>
        ch.map((row, i) => row.map((v, j) => v + perturbation[c][i][j])),
      );
    }
    return shockResult?.rollout[day - 1] ?? baseActiveGrid;
  }, [day, perturbation, shockResult, initStates, baseActiveGrid]);

  const deltaActiveGrid = useMemo(
    () => computeDeltaGrid(shockActiveGrid, baseActiveGrid),
    [shockActiveGrid, baseActiveGrid],
  );

  const physicsMetrics = useMemo(() => {
    if (!baseResult || !shockResult || shockPreset === "none") return null;
    return computePhysicsMetrics(baseResult.rollout, shockResult.rollout, shockPreset, customShock);
  }, [baseResult, shockResult, shockPreset, customShock]);

  const impactMetrics = useMemo(() => {
    if (!baseResult) return null;
    return computeImpactMetrics(baseResult.rollout, shockResult ? shockResult.rollout : baseResult.rollout);
  }, [baseResult, shockResult]);

  const stats = useMemo(
    () => computeFieldStats(baseActiveGrid[CHANNEL_META[channel].index], channel, useCelsius),
    [baseActiveGrid, channel, useCelsius],
  );

  const meta = CHANNEL_META[channel];
  const unit = meta.displayUnit(useCelsius);
  const activeSim = shockResult ?? baseResult;
  const highlightBox = SHOCK_PRESET_META[shockPreset].box;
  const customRing =
    shockPreset === "custom"
      ? { lat: customShock.lat, lon: customShock.lon, radiusDeg: customShock.radiusDeg }
      : null;

  const handleMapClick =
    shockPreset === "custom"
      ? (lat: number, lon: number) => setCustomShock((prev) => ({ ...prev, lat, lon }))
      : undefined;

  return (
    <div className="min-h-screen bg-black text-zinc-50">
      {/* Slim Header */}
      <header className="sticky top-0 z-20 h-14 border-b border-zinc-900 bg-black/80 px-5 backdrop-blur-md">
        <div className="mx-auto flex h-full max-w-[1460px] items-center justify-between gap-3">
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

            {activeSim && (
              <span className="hidden rounded-full border border-zinc-800 bg-zinc-950 px-2.5 py-1 font-mono text-[11px] tabular-nums text-zinc-400 sm:inline-block">
                {activeSim.source.toUpperCase()} · {activeSim.inference_ms.toFixed(1)}ms
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
      <main className="mx-auto grid max-w-[1460px] grid-cols-1 gap-4 p-4 md:grid-cols-[276px_1fr]">
        {/* Left Sidebar */}
        <aside className="flex flex-col gap-3">
          <ShockBuilder
            preset={shockPreset}
            custom={customShock}
            loading={loadingSim}
            onChangePreset={handlePresetChange}
            onChangeCustom={setCustomShock}
            onRunBranch={() => void runSimulation()}
          />

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
              <SwitchRow label="Wind Streamlines" checked={showVectors} onChange={setShowVectors} />
              <SwitchRow label="Smooth Radar Grid" checked={smoothGrid} onChange={setSmoothGrid} />
              <SwitchRow label="City Labels" checked={showCities} onChange={setShowCities} />
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
                    <span className="font-mono text-[10px] tabular-nums text-zinc-600">{p.doy}</span>
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

            <div className="mt-3 border-t border-zinc-900 pt-2.5">
              <SwitchRow label="Offline Demo Mode" checked={preferMock} onChange={setPreferMock} />
            </div>
          </section>
        </aside>

        {/* Right Maps, Scrubber & Telemetry */}
        <section className="flex flex-col gap-3">
          {/* View Mode Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-zinc-800/80 bg-zinc-950 px-3.5 py-2">
            <div className="flex items-center gap-2 text-xs">
              <span className="font-medium text-zinc-200">{meta.label}</span>
              <span className="text-zinc-600">·</span>
              <span className="font-mono text-[11px] tabular-nums text-zinc-400">
                {day === 0 ? "Day 0 (t=0)" : `Day +${day}`} · Min {stats.min.toFixed(1)} / Mean{" "}
                {stats.mean.toFixed(1)} / Max {stats.max.toFixed(1)} {unit}
              </span>
            </div>

            <div className="flex items-center gap-1 rounded-md border border-zinc-800 bg-black p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => setViewMode("single")}
                className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition ${
                  viewMode === "single"
                    ? "bg-zinc-100 text-zinc-950"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Maximize2 className="h-3 w-3" />
                Single
              </button>
              <button
                type="button"
                onClick={() => setViewMode("compare")}
                className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition ${
                  viewMode === "compare"
                    ? "bg-zinc-100 text-zinc-950"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Columns3 className="h-3 w-3" />
                Compare (3-Up)
              </button>
            </div>
          </div>

          {viewMode === "single" ? (
            <IndiaGridMap
              stateGrid={shockPreset !== "none" ? shockActiveGrid : baseActiveGrid}
              channel={channel}
              useCelsius={useCelsius}
              showVectors={showVectors}
              smoothGrid={smoothGrid}
              showBoxes={showBoxes}
              showCities={showCities}
              highlightBox={highlightBox}
              customCenter={customRing}
              onCellClick={handleMapClick}
              title={
                shockPreset !== "none"
                  ? `Branched Timeline (${SHOCK_PRESET_META[shockPreset].short}) · ${meta.label}`
                  : `Timeline Prime · ${meta.label}`
              }
              subtitle={
                day === 0 ? "Initial State (t = 0)" : `Autoregressive Rollout · Day +${day}`
              }
            />
          ) : (
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
              <IndiaGridMap
                compact
                stateGrid={baseActiveGrid}
                channel={channel}
                useCelsius={useCelsius}
                showVectors={showVectors}
                smoothGrid={smoothGrid}
                showBoxes={showBoxes}
                showCities={showCities}
                onCellClick={handleMapClick}
                title="Timeline Prime"
                subtitle="Unperturbed baseline"
              />
              <IndiaGridMap
                compact
                stateGrid={shockActiveGrid}
                channel={channel}
                useCelsius={useCelsius}
                showVectors={showVectors}
                smoothGrid={smoothGrid}
                showBoxes={showBoxes}
                showCities={showCities}
                highlightBox={highlightBox}
                customCenter={customRing}
                onCellClick={handleMapClick}
                title={`Branched · ${SHOCK_PRESET_META[shockPreset].short}`}
                subtitle="Perturbed rollout"
              />
              <IndiaGridMap
                compact
                mode="delta"
                stateGrid={deltaActiveGrid}
                channel={channel}
                useCelsius={useCelsius}
                showVectors={false}
                smoothGrid={smoothGrid}
                showBoxes={showBoxes}
                showCities={showCities}
                highlightBox={highlightBox}
                customCenter={customRing}
                onCellClick={handleMapClick}
                title="Delta (Shocked − Prime)"
                subtitle="Causal plume propagation"
              />
            </div>
          )}

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

          <TelemetryImpactPanel
            wallMs={activeSim ? activeSim.wall_ms : null}
            serverMs={activeSim ? activeSim.inference_ms : null}
            physics={physicsMetrics}
            impact={impactMetrics}
            preset={shockPreset}
          />
        </section>
      </main>
    </div>
  );
}
