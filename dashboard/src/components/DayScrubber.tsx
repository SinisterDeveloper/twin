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
    <div className="flex flex-wrap items-center gap-4 rounded-xl border border-zinc-800/80 bg-zinc-950 px-4 py-3">
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => onChangeDay(0)}
          title="Reset to Day 0"
          className="rounded-md border border-zinc-800 bg-black p-1.5 text-zinc-400 transition hover:bg-zinc-900 hover:text-zinc-100"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={() => onChangeDay(Math.max(0, day - 1))}
          disabled={day === 0}
          title="Previous Day"
          className="rounded-md border border-zinc-800 bg-black p-1.5 text-zinc-400 transition hover:bg-zinc-900 hover:text-zinc-100 disabled:opacity-35"
        >
          <SkipBack className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={onTogglePlay}
          className="flex h-7 items-center gap-1.5 rounded-md bg-zinc-50 px-3 text-xs font-medium text-zinc-950 transition hover:bg-zinc-200"
        >
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          onClick={() => onChangeDay(Math.min(maxDay, day + 1))}
          disabled={day === maxDay}
          title="Next Day"
          className="rounded-md border border-zinc-800 bg-black p-1.5 text-zinc-400 transition hover:bg-zinc-900 hover:text-zinc-100 disabled:opacity-35"
        >
          <SkipForward className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex min-w-[200px] flex-1 flex-col gap-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-zinc-200">
            {day === 0 ? "Day 0 · Initial State" : `Forecast Day +${day}`}
          </span>
          <span className="font-mono tabular-nums text-zinc-500">DOY {effectiveDoy}</span>
        </div>
        <input
          type="range"
          min={0}
          max={maxDay}
          step={1}
          value={day}
          onChange={(e) => onChangeDay(Number(e.target.value))}
          className="shadcn-slider w-full"
        />
        <div className="flex justify-between font-mono text-[10px] tabular-nums text-zinc-600">
          {Array.from({ length: maxDay + 1 }, (_, d) => (
            <span key={d} className={d === day ? "font-semibold text-zinc-100" : ""}>
              {d === 0 ? "T0" : `+${d}`}
            </span>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-0.5 rounded-md border border-zinc-800 bg-black p-0.5 text-[11px]">
        {[0.5, 1, 2].map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onChangeSpeed(s)}
            className={`rounded px-2 py-1 font-mono tabular-nums transition ${
              speed === s ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
