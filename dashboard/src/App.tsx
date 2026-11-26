import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { DayScrubber } from "./components/DayScrubber";
import { DEFAULT_MAP_VIEW, IndiaGridMap, type MapViewState } from "./components/IndiaGridMap";
import { ShockBuilder } from "./components/ShockBuilder";
import { TelemetryImpactPanel } from "./components/TelemetryImpactPanel";
import {
  fetchHealth,
  mockSimulate,
  postSimulate,
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
import { getEra5InitStates, loadEra5YearDataset, type Era5YearDataset } from "./lib/era5";
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
          checked ? "border-zinc-200 bg-zinc-200" : "border-white/[0.08] bg-zinc-800/90"
        }`}
      >
        <span
          className={`inline-block h-2.5 w-2.5 rounded-full transition-transform ${
            checked ? "translate-x-3.5 bg-zinc-950" : "translate-x-0.5 bg-zinc-400"
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
  const [showCities, setShowCities] = useState<boolean>(true);
  const [showStates, setShowStates] = useState<boolean>(true);
  const [mapView, setMapView] = useState<MapViewState>(DEFAULT_MAP_VIEW);

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

  const [serverOffline, setServerOffline] = useState<boolean>(false);
  const [baseResult, setBaseResult] = useState<SimulateResponse | null>(null);
  const [shockResult, setShockResult] = useState<SimulateResponse | null>(null);
  const [loadingSim, setLoadingSim] = useState<boolean>(false);
  const [era5, setEra5] = useState<Era5YearDataset | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadEra5YearDataset().then((ds) => {
      if (!cancelled && ds) setEra5(ds);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const { initStates, activeDate } = useMemo(() => {
    if (era5) {
      const { initStates: states, date } = getEra5InitStates(era5, doy);
      return { initStates: states, activeDate: date as string | null };
    }
    return { initStates: buildDefaultInitStates(doy), activeDate: null as string | null };
  }, [era5, doy]);

  const perturbation = useMemo(
    () => buildPerturbation(shockPreset, initStates[initStates.length - 1], customShock),
    [shockPreset, initStates, customShock],
  );

  const checkServerHealth = useCallback(async () => {
    try {
      await fetchHealth();
      setServerOffline(false);
    } catch {
      setServerOffline(true);
    }
  }, []);

  const executeSim = useCallback(async (req: SimulateRequest): Promise<SimulateResponse> => {
    try {
      return await postSimulate(req);
    } catch {
      return mockSimulate(req);
    }
  }, []);

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

  const doyPct = `${((doy - 1) / 364) * 100}%`;

  return (
    <div className="min-h-screen bg-black bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.035),transparent_55%)] text-zinc-50">
      {/* Slim Header */}
      <header className="sticky top-0 z-20 h-11 border-b border-white/[0.07] bg-black/80 px-5 backdrop-blur-md">
        <div className="mx-auto flex h-full max-w-[1440px] items-center justify-between gap-3">
          <h1 className="text-sm font-medium tracking-tight text-zinc-100">
            Multiverse of Monsoon
          </h1>

          {serverOffline && (
            <span className="font-mono text-[11px] text-zinc-500">Server offline · mock</span>
          )}
        </div>
      </header>

      {/* Main Workspace */}
      <main className="mx-auto grid max-w-[1440px] grid-cols-1 md:grid-cols-[268px_1fr]">
        {/* Unified Left Sidebar Rail */}
        <aside className="flex flex-col gap-5 border-b border-white/[0.07] bg-white/[0.012] px-4 py-4 md:border-b-0 md:border-r">
          <ShockBuilder
            preset={shockPreset}
            custom={customShock}
            loading={loadingSim}
            onChangePreset={handlePresetChange}
            onChangeCustom={setCustomShock}
          />

          <section className="border-t border-white/[0.07] pt-4">
            <div className="mb-2.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Variable
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              {CHANNELS.map((key) => {
                const m = CHANNEL_META[key];
                const active = key === channel;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setChannel(key)}
                    className={`rounded-md border px-2 py-1.5 text-left transition ${
                      active
                        ? "border-zinc-100 bg-zinc-100 text-zinc-950 shadow-sm"
                        : "border-white/[0.06] bg-white/[0.02] text-zinc-300 hover:bg-white/[0.05] hover:text-zinc-100"
                    }`}
                  >
                    <div className="font-mono text-[11px] font-semibold">{m.key}</div>
                    <div className={`truncate text-[10px] ${active ? "text-zinc-600" : "text-zinc-500"}`}>
                      {m.shortLabel.replace(/\s*\(.*\)/, "")}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-3.5 flex flex-col gap-1 border-t border-white/[0.07] pt-3">
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                Layers
              </div>
              <SwitchRow label="Temperature in °C" checked={useCelsius} onChange={setUseCelsius} />
              <SwitchRow label="Wind Streamlines" checked={showVectors} onChange={setShowVectors} />
              <SwitchRow label="Smooth Radar Grid" checked={smoothGrid} onChange={setSmoothGrid} />
              <SwitchRow label="City Labels" checked={showCities} onChange={setShowCities} />
              <SwitchRow label="State Borders" checked={showStates} onChange={setShowStates} />
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                Season
              </span>
              {activeDate && (
                <span className="shrink-0 whitespace-nowrap font-mono text-[11px] tabular-nums text-zinc-500">
                  {activeDate}
                </span>
              )}
            </div>
            <div className="flex flex-col gap-0.5">
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
                    className={`flex items-center justify-between rounded-md px-2.5 py-1.5 text-left text-xs transition ${
                      active
                        ? "bg-white/[0.09] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
                        : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-100"
                    }`}
                  >
                    <span className="font-medium">{p.name}</span>
                    <span className={`font-mono text-[10px] tabular-nums ${active ? "text-zinc-300" : "text-zinc-600"}`}>
                      {p.doy}
                    </span>
                  </button>
                );
              })}
            </div>

            <div className="mt-3 border-t border-white/[0.07] pt-3">
              <div className="mb-1.5 flex justify-between text-xs">
                <span className="text-zinc-500">Day of Year</span>
                <span className="font-mono tabular-nums text-zinc-200">{doy} / 365</span>
              </div>
              <input
                type="range"
                min={1}
                max={365}
                value={doy}
                style={{ "--slider-pct": doyPct } as CSSProperties}
                onChange={(e) => setDoy(Number(e.target.value))}
                className="shadcn-slider w-full"
              />
            </div>
          </section>
        </aside>

        {/* Right Maps, Scrubber & Telemetry */}
        <section className="flex flex-col gap-3 p-4 lg:p-5">
          {/* Borderless View Mode Bar */}
          <div className="flex flex-wrap items-center justify-between gap-2 px-0.5">
            <div className="flex flex-wrap items-baseline gap-2">
              <span className="text-xs font-medium text-zinc-200">{meta.label}</span>
              <span className="text-zinc-700">·</span>
              <span className="font-mono text-[11px] tabular-nums text-zinc-400">
                {day === 0 ? "Day 0" : `Day +${day}`} · Min {stats.min.toFixed(1)} / Mean{" "}
                {stats.mean.toFixed(1)} / Max {stats.max.toFixed(1)} {unit}
              </span>
            </div>

            <div className="flex items-center gap-0.5 rounded-md border border-white/[0.08] bg-white/[0.03] p-0.5 text-[11px]">
              <button
                type="button"
                onClick={() => setViewMode("single")}
                className={`rounded px-2.5 py-1 font-medium transition ${
                  viewMode === "single"
                    ? "bg-zinc-100 text-zinc-950 shadow-sm"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Single
              </button>
              <button
                type="button"
                onClick={() => setViewMode("compare")}
                className={`rounded px-2.5 py-1 font-medium transition ${
                  viewMode === "compare"
                    ? "bg-zinc-100 text-zinc-950 shadow-sm"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Compare
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
              showCities={showCities}
              showStates={showStates}
              highlightBox={highlightBox}
              customCenter={customRing}
              view={mapView}
              onViewChange={setMapView}
              onCellClick={handleMapClick}
              title={shockPreset !== "none" ? SHOCK_PRESET_META[shockPreset].short : "Baseline"}
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
                showCities={showCities}
                showStates={showStates}
                view={mapView}
                onViewChange={setMapView}
                onCellClick={handleMapClick}
                title="Baseline"
              />
              <IndiaGridMap
                compact
                stateGrid={shockActiveGrid}
                channel={channel}
                useCelsius={useCelsius}
                showVectors={showVectors}
                smoothGrid={smoothGrid}
                showCities={showCities}
                showStates={showStates}
                highlightBox={highlightBox}
                customCenter={customRing}
                view={mapView}
                onViewChange={setMapView}
                onCellClick={handleMapClick}
                title={SHOCK_PRESET_META[shockPreset].short}
              />
              <IndiaGridMap
                compact
                mode="delta"
                stateGrid={deltaActiveGrid}
                channel={channel}
                useCelsius={useCelsius}
                showVectors={false}
                smoothGrid={smoothGrid}
                showCities={showCities}
                showStates={showStates}
                highlightBox={highlightBox}
                customCenter={customRing}
                view={mapView}
                onViewChange={setMapView}
                onCellClick={handleMapClick}
                title="Delta"
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
            physics={physicsMetrics}
            impact={impactMetrics}
            preset={shockPreset}
          />
        </section>
      </main>
    </div>
  );
}
