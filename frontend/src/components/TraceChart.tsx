import { useEffect, useMemo, useRef, useState } from "react";
import type { TokenEvent } from "../api/types";
import { displayTokenText, formatNumber } from "../utils/format";
import { SegmentedControl } from "./ui";

export interface ChartMetric {
  key: string;
  label: string;
  unit?: string;
  /** A CSS colour, normally a `var(--viz-n)` data-visualisation token. */
  color: string;
  value: (token: TokenEvent) => number | undefined;
}

interface TraceChartProps {
  title: string;
  tokens: TokenEvent[];
  metrics: ChartMetric[];
  revealTokenText?: boolean;
  selectedToken: number | null;
  onSelectToken: (index: number) => void;
}

type WindowSize = "128" | "512" | "all";

const height = 176;
const plot = { left: 44, right: 12, top: 12, bottom: 28 };
const windowOptions: Array<{ value: WindowSize; label: string; title: string }> = [
  { value: "128", label: "128", title: "Show the last 128 tokens" },
  { value: "512", label: "512", title: "Show the last 512 tokens" },
  { value: "all", label: "All", title: "Show every token" },
];

function sampled<T>(items: T[], maximum: number): T[] {
  if (items.length <= maximum) return items;
  const stride = (items.length - 1) / (maximum - 1);
  return Array.from({ length: maximum }, (_, index) => items[Math.round(index * stride)]).filter((item): item is T => item !== undefined);
}

/** Tracks an element's width so the SVG is drawn in real pixels and its labels keep their CSS size. */
function useWidth(fallback: number): [React.RefObject<HTMLElement>, number] {
  const element = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const node = element.current;
    if (!node || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(([entry]) => {
      if (entry && entry.contentRect.width > 0) setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [element, width];
}

export function TraceChart({ title, tokens, metrics, revealTokenText = false, selectedToken, onSelectToken }: TraceChartProps): React.ReactNode {
  const [windowSize, setWindowSize] = useState<WindowSize>("128");
  const [endIndex, setEndIndex] = useState<number | null>(null);
  // Measured on the always-mounted card so the width is known before the first data arrives.
  const [container, width] = useWidth(360);
  const effectiveEnd = endIndex === null ? Math.max(0, tokens.length - 1) : Math.min(Math.max(endIndex, 0), Math.max(0, tokens.length - 1));
  const count = windowSize === "all" ? tokens.length : Number(windowSize);
  const start = Math.max(0, effectiveEnd - count + 1);
  const visible = tokens.slice(start, effectiveEnd + 1);

  const series = useMemo(() => metrics.map((metric) => ({
    metric,
    points: visible.map((token) => ({ token, value: metric.value(token) })).filter((point): point is { token: TokenEvent; value: number } => point.value !== undefined && Number.isFinite(point.value)),
  })), [metrics, visible]);
  const allValues = series.flatMap((item) => item.points.map((point) => point.value));
  const minimum = allValues.length ? Math.min(...allValues) : 0;
  const maximum = allValues.length ? Math.max(...allValues) : 1;
  const valueRange = maximum - minimum || 1;
  const plotWidth = Math.max(1, width - plot.left - plot.right);
  const plotHeight = height - plot.top - plot.bottom;
  const x = (index: number): number => plot.left + ((index - start) / Math.max(1, visible.length - 1)) * plotWidth;
  const y = (value: number): number => plot.top + (1 - (value - minimum) / valueRange) * plotHeight;
  const ticks = [0, 0.5, 1];
  const pointColor = metrics[0]?.color ?? "var(--viz-1)";

  return (
    <section className="trace-chart" ref={container}>
      <header className="trace-chart-head">
        <div><h3>{title}</h3><span>by generated token position</span></div>
        <SegmentedControl label={`${title} window`} onChange={setWindowSize} options={windowOptions} size="sm" value={windowSize} />
      </header>
      {tokens.length === 0 || allValues.length === 0 ? (
        <div className="chart-empty">No values were captured for this trace.</div>
      ) : (
        <>
          <div aria-label={`${title} token trace`} className="chart-canvas" role="img">
            <svg height={height} viewBox={`0 0 ${String(width)} ${String(height)}`} width={width}>
              {ticks.map((tick) => {
                const tickY = plot.top + tick * plotHeight;
                const value = maximum - tick * valueRange;
                return <g key={tick}><line className="chart-grid" x1={plot.left} x2={width - plot.right} y1={tickY} y2={tickY} /><text className="chart-label" x={plot.left - 8} y={tickY + 4}>{formatNumber(value, 2)}</text></g>;
              })}
              <text className="chart-label chart-axis-start" x={plot.left} y={height - 8}>{String(start)}</text>
              <text className="chart-label" x={width - plot.right} y={height - 8}>{String(effectiveEnd)}</text>
              {series.map(({ metric, points }) => {
                const visualPoints = sampled(points, 600);
                const path = visualPoints.map((point, index) => `${index === 0 ? "M" : "L"}${String(x(point.token.index))},${String(y(point.value))}`).join(" ");
                return <path d={path} fill="none" key={metric.key} strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" style={{ stroke: metric.color }} />;
              })}
              {series[0] && sampled(series[0].points, 220).map(({ token, value }) => (
                <circle
                  className={selectedToken === token.index ? "chart-point selected" : "chart-point"}
                  cx={x(token.index)}
                  cy={y(value)}
                  key={token.index}
                  onClick={() => onSelectToken(token.index)}
                  r={selectedToken === token.index ? 4.5 : 2.5}
                  style={{ fill: pointColor }}
                  tabIndex={0}
                ><title>#{String(token.index)}{revealTokenText ? ` “${displayTokenText(token.displayText || token.piece) || "∅"}”` : ""} · {formatNumber(value, 4)}</title></circle>
              ))}
            </svg>
          </div>
          {tokens.length > count && (
            <label className="chart-scrubber">
              <span>Window ends at token {String(effectiveEnd)}</span>
              <input className="slider" max={tokens.length - 1} min={Math.min(count - 1, tokens.length - 1)} onChange={(event) => setEndIndex(Number(event.target.value))} style={{ "--fill": `${String((effectiveEnd / Math.max(1, tokens.length - 1)) * 100)}%` } as React.CSSProperties} type="range" value={effectiveEnd} />
            </label>
          )}
          <div className="chart-legend">
            {metrics.map((metric) => <span key={metric.key}><i aria-hidden="true" style={{ background: metric.color }} />{metric.label}{metric.unit ? ` (${metric.unit})` : ""}</span>)}
            {tokens.length > 600 && <small>Display is downsampled; selection and table keep original positions.</small>}
          </div>
        </>
      )}
    </section>
  );
}
