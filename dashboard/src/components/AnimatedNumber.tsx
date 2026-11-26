import { useEffect, useRef, useState } from "react";

interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  duration?: number;
  signed?: boolean;
  prefix?: string;
  suffix?: string;
  className?: string;
}

export function AnimatedNumber({
  value,
  decimals = 1,
  duration = 340,
  signed = false,
  prefix = "",
  suffix = "",
  className = "",
}: AnimatedNumberProps) {
  const [display, setDisplay] = useState<number>(value);
  const displayRef = useRef<number>(value);

  useEffect(() => {
    if (!Number.isFinite(value)) {
      displayRef.current = value;
      setDisplay(value);
      return;
    }

    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    if (prefersReducedMotion || !Number.isFinite(displayRef.current)) {
      displayRef.current = value;
      setDisplay(value);
      return;
    }

    const from = displayRef.current;
    if (Math.abs(value - from) < 1e-6) {
      displayRef.current = value;
      setDisplay(value);
      return;
    }

    let rafId = 0;
    const t0 = performance.now();

    const tick = (now: number) => {
      const elapsed = now - t0;
      const t = Math.min(1, elapsed / duration);
      const eased = 1 - (1 - t) ** 3;
      const next = t >= 1 ? value : from + (value - from) * eased;
      displayRef.current = next;
      setDisplay(next);
      if (t < 1) {
        rafId = window.requestAnimationFrame(tick);
      }
    };

    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [value, duration]);

  const rounded = Object.is(Number(display.toFixed(decimals)), -0) ? 0 : display;
  const formatted = Number.isFinite(rounded) ? rounded.toFixed(decimals) : "—";
  const sign = signed && Number.isFinite(rounded) && rounded >= 0 ? "+" : "";

  return (
    <span className={className ? `tabular-nums ${className}` : "tabular-nums"}>
      {prefix}
      {sign}
      {formatted}
      {suffix}
    </span>
  );
}
