import { Maximize2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { TokenEvent } from "../api/types";
import { escapeToken, formatNumber } from "../utils/format";

export interface ChartMetric {
  key: string;
  label: string;
  unit?: string;
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

const width = 720;
const height = 240;
const plot = { left: 48, right: 16, top: 20, bottom: 38 };

function sampled<T>(items: T[], maximum: number): T[] {
  if (items.length <= maximum) return items;
  const stride = (items.length - 1) / (maximum - 1);
  return Array.from({ length: maximum }, (_, index) => items[Math.round(index * stride)]).filter((item): item is T => item !== undefined);
}

export function TraceChart({ title, tokens, metrics, revealTokenText = false, selectedToken, onSelectToken }: TraceChartProps): React.ReactNode {
  const [windowSize, setWindowSize] = useState<"128" | "512" | "all">("128");
  const [endIndex, setEndIndex] = useState<number | null>(null);
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
  const x = (index: number): number => plot.left + ((index - start) / Math.max(1, visible.length - 1)) * (width - plot.left - plot.right);
  const y = (value: number): number => plot.top + (1 - (value - minimum) / valueRange) * (height - plot.top - plot.bottom);
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <section className="trace-chart-card">
      <div className="chart-heading">
        <div><span className="eyebrow">Token position on x-axis</span><h3>{title}</h3></div>
        <div className="chart-actions">
          <label>Window <select onChange={(event) => setWindowSize(event.target.value as "128" | "512" | "all")} value={windowSize}><option value="128">128</option><option value="512">512</option><option value="all">All</option></select></label>
          <Maximize2 size={14} />
        </div>
      </div>
      {tokens.length === 0 || allValues.length === 0 ? (
        <div className="chart-empty">No values were captured for this trace.</div>
      ) : (
        <>
          <div className="chart-canvas" role="img" aria-label={`${title} token trace`}>
            <svg preserveAspectRatio="none" viewBox={`0 0 ${String(width)} ${String(height)}`}>
              {ticks.map((tick) => {
                const tickY = plot.top + tick * (height - plot.top - plot.bottom);
                const value = maximum - tick * valueRange;
                return <g key={tick}><line className="chart-grid" x1={plot.left} x2={width - plot.right} y1={tickY} y2={tickY} /><text className="chart-label" x={plot.left - 8} y={tickY + 4}>{formatNumber(value, 2)}</text></g>;
              })}
              {series.map(({ metric, points }) => {
                const visualPoints = sampled(points, 600);
                const path = visualPoints.map((point, index) => `${index === 0 ? "M" : "L"}${String(x(point.token.index))},${String(y(point.value))}`).join(" ");
                return <path d={path} fill="none" key={metric.key} stroke={metric.color} strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" vectorEffect="non-scaling-stroke" />;
              })}
              {series[0] && sampled(series[0].points, 220).map(({ token, value }) => (
                <circle
                  className={selectedToken === token.index ? "chart-point selected" : "chart-point"}
                  cx={x(token.index)}
                  cy={y(value)}
                  key={token.index}
                  onClick={() => onSelectToken(token.index)}
                  r={selectedToken === token.index ? 5 : 2.5}
                  tabIndex={0}
                ><title>#{String(token.index)}{revealTokenText ? ` “${escapeToken(token.piece)}”` : ""} · {formatNumber(value, 4)}</title></circle>
              ))}
              <text className="chart-axis-title" x={(plot.left + width - plot.right) / 2} y={height - 7}>Generated token position</text>
            </svg>
          </div>
          {tokens.length > count && (
            <label className="chart-scrubber"><span>Visible end: token {String(effectiveEnd)}</span><input max={tokens.length - 1} min={Math.min(count - 1, tokens.length - 1)} onChange={(event) => setEndIndex(Number(event.target.value))} type="range" value={effectiveEnd} /></label>
          )}
          <div className="chart-legend">
            {metrics.map((metric) => <span key={metric.key}><i style={{ background: metric.color }} />{metric.label}{metric.unit ? ` (${metric.unit})` : ""}</span>)}
            {tokens.length > 600 && <small>Display is downsampled; selection and table keep original positions.</small>}
          </div>
        </>
      )}
    </section>
  );
}
