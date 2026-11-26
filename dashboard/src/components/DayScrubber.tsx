import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";

interface DayScrubberProps {
  day: number;
  maxDay: number;
  playing: boolean;
  speed: number;
  doy: number;
  onChangeDay: (d: number) => void;
  onTogglePlay: () => void;
  onChangeSpeed: (s: number) => void;
}

export function DayScrubber({
  day,
  maxDay,
  playing,
  speed,
  doy,
  onChangeDay,
  onTogglePlay,
  onChangeSpeed,
}: DayScrubberProps) {
  const effectiveDoy = ((doy + day - 1) % 365) + 1;

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 shadow-lg">
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => onChangeDay(0)}
          title="Reset to Day 0"
          className="rounded-lg border border-slate-700 bg-slate-800 p-2 text-slate-200 hover:bg-slate-700"
        >
          <RotateCcw className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={() => onChangeDay(Math.max(0, day - 1))}
          disabled={day === 0}
          title="Previous Day"
          className="rounded-lg border border-slate-700 bg-slate-800 p-2 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
        >
          <SkipBack className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={onTogglePlay}
          className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400"
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          onClick={() => onChangeDay(Math.min(maxDay, day + 1))}
          disabled={day === maxDay}
          title="Next Day"
          className="rounded-lg border border-slate-700 bg-slate-800 p-2 text-slate-200 hover:bg-slate-700 disabled:opacity-40"
        >
          <SkipForward className="h-4 w-4" />
        </button>
      </div>

      <div className="flex min-w-[200px] flex-1 flex-col gap-1">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-200">
            {day === 0 ? "Day 0 (Initial State)" : `Forecast Day +${day}`}
          </span>
          <span className="font-mono text-slate-400">DOY {effectiveDoy}</span>
        </div>
        <input
          type="range"
          min={0}
          max={maxDay}
          step={1}
          value={day}
          onChange={(e) => onChangeDay(Number(e.target.value))}
          className="h-2 w-full cursor-pointer accent-cyan-400"
        />
        <div className="flex justify-between font-mono text-[10px] text-slate-500">
          {Array.from({ length: maxDay + 1 }, (_, d) => (
            <span key={d} className={d === day ? "font-bold text-cyan-300" : ""}>
              {d === 0 ? "T0" : `+${d}`}
            </span>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 p-1 text-xs">
        {[0.5, 1, 2].map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onChangeSpeed(s)}
            className={`rounded px-2 py-1 font-mono transition ${
              speed === s ? "bg-slate-700 text-cyan-300" : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
