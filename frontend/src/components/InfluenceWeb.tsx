import { Film, Image } from "lucide-react";
import { type Ref, useMemo, useState } from "react";
import {
  edgeLabels,
  edgeStyle,
  formatInfluenceWeight,
  polar,
  ringLayout,
  separateLabels,
  sourceKindLabel,
  sourcePosition,
  sourceText,
  type InfluenceScale,
  type RingNode,
  type WebModel,
  type WebSource,
} from "../domain/influence";

export type InfluenceWebVariant = "compact" | "expanded";

interface InfluenceWebProps {
  web: WebModel;
  scale: InfluenceScale;
  target: { label: string; index: number };
  variant: InfluenceWebVariant;
  /** Edge labels are written on the strongest this-many edges (all of them when expanded). */
  labelLimit: number;
  title: string;
  onSelectGeneratedToken: (index: number) => void;
  svgRef?: Ref<SVGSVGElement>;
}

const GEOMETRY: Record<InfluenceWebVariant, { size: number; inset: number; nodeLabels: number; textLength: number }> = {
  compact: { size: 360, inset: 50, nodeLabels: 8, textLength: 9 },
  expanded: { size: 720, inset: 92, nodeLabels: 64, textLength: 14 },
};
const ARC_GAP = 0.2;

function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

function arcPath(cx: number, cy: number, radius: number, start: number, end: number): string {
  const from = polar(cx, cy, radius, start);
  const to = polar(cx, cy, radius, end);
  return `M ${String(from.x)} ${String(from.y)} A ${String(radius)} ${String(radius)} 0 ${end - start > Math.PI ? 1 : 0} 1 ${String(to.x)} ${String(to.y)}`;
}

function nodeDescription(source: WebSource, scale: InfluenceScale): string {
  const weight = scale === "relative"
    ? `${formatInfluenceWeight(source.display)} of the largest (share ${formatInfluenceWeight(source.share)})`
    : formatInfluenceWeight(source.share);
  return `${sourceText(source)}, ${sourceKindLabel(source)}, position ${sourcePosition(source)}, weight ${weight}, rank ${String(source.rank)}`;
}

/**
 * A radial "web": the predicted token sits in the centre and every shown earlier
 * source sits on a ring in sequence order, joined to it by an edge whose width,
 * opacity, colour, and written label carry the source's weight.
 */
export function InfluenceWeb({ web, scale, target, variant, labelLimit, title, onSelectGeneratedToken, svgRef }: InfluenceWebProps): React.ReactNode {
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const geometry = GEOMETRY[variant];
  const center = { x: geometry.size / 2, y: geometry.size / 2 };
  const radius = geometry.size / 2 - geometry.inset;
  const maxDisplay = Math.max(1e-12, ...web.shown.map((source) => source.display));

  const layout = useMemo(() => {
    const middle = { x: geometry.size / 2, y: geometry.size / 2 };
    const ring = ringLayout(web.shown, { cx: middle.x, cy: middle.y, radius: geometry.size / 2 - geometry.inset, gap: ARC_GAP });
    const byStrength = [...ring.nodes].sort((left, right) => left.source.rank - right.source.rank);
    const labelled = new Set(byStrength.slice(0, labelLimit).map((node) => node.source.contextIndex));
    const named = new Set(byStrength.slice(0, geometry.nodeLabels).map((node) => node.source.contextIndex));
    const labels = separateLabels(edgeLabels(ring.nodes.filter((node) => labelled.has(node.source.contextIndex)), middle, (node) => formatInfluenceWeight(node.source.display)));
    return { ...ring, labels, named };
  }, [web.shown, labelLimit, geometry]);

  const active = hovered ?? pinned;
  const activeNode = layout.nodes.find((node) => node.source.contextIndex === active);
  const dimmed = (key: number): boolean => active !== null && active !== key;
  const activate = (node: RingNode): void => {
    if (node.source.sourceKind === "generated" && node.source.generatedTokenIndex !== null) {
      onSelectGeneratedToken(node.source.generatedTokenIndex);
    } else {
      setPinned((value) => (value === node.source.contextIndex ? null : node.source.contextIndex));
    }
  };

  return (
    <div className={`influence-web-frame influence-web-${variant}`}>
      <svg
        aria-label={title}
        className="influence-web"
        fontFamily="var(--font-mono)"
        ref={svgRef}
        role="group"
        viewBox={`0 0 ${String(geometry.size)} ${String(geometry.size)}`}
        xmlns="http://www.w3.org/2000/svg"
      >
        <title>{title}</title>
        {layout.arcs.map((arc) => {
          const middle = polar(center.x, center.y, radius + geometry.inset * 0.62, (arc.start + arc.end) / 2);
          return (
            <g key={arc.kind}>
              <path d={arcPath(center.x, center.y, radius, arc.start, arc.end)} fill="none" stroke={arc.kind === "prompt" ? "var(--border-strong)" : "var(--accent-muted)"} strokeLinecap="round" strokeWidth={variant === "expanded" ? 3 : 2} />
              <text dominantBaseline="middle" fill="var(--text-tertiary)" fontSize={11} textAnchor="middle" x={middle.x} y={middle.y}>
                {arc.kind === "prompt" ? "prompt/history" : "generated"}
              </text>
            </g>
          );
        })}
        <g aria-hidden="true">
          {layout.nodes.map((node) => {
            const style = edgeStyle(node.source.display / maxDisplay);
            return <line data-edge={node.source.contextIndex} key={node.source.contextIndex} opacity={dimmed(node.source.contextIndex) ? 0.08 : style.opacity} stroke={style.color} strokeLinecap="round" strokeWidth={style.width} x1={node.x} x2={center.x} y1={node.y} y2={center.y} />;
          })}
        </g>
        <g aria-hidden="true" className="influence-edge-labels">
          {layout.labels.map((label) => {
            const node = layout.nodes.find((item) => item.source.contextIndex === label.key);
            const style = edgeStyle((node?.source.display ?? 0) / maxDisplay);
            return (
              <g data-edge-label={label.key} key={label.key} opacity={dimmed(label.key) ? 0.12 : 1}>
                <rect fill="var(--surface-1)" height={label.height} rx={label.height / 2} stroke={style.color} strokeWidth={1} width={label.width} x={label.x - label.width / 2} y={label.y - label.height / 2} />
                <text dominantBaseline="central" fill="var(--text-primary)" fontSize={11} textAnchor="middle" x={label.x} y={label.y}>{label.text}</text>
              </g>
            );
          })}
        </g>
        {layout.nodes.map((node) => {
          const { source } = node;
          const strength = source.display / maxDisplay;
          const style = edgeStyle(strength);
          const nodeRadius = (variant === "expanded" ? 6 : 4.5) + strength * (variant === "expanded" ? 8 : 6);
          const outside = polar(center.x, center.y, radius + nodeRadius + 7, node.angle);
          const side = Math.sin(node.angle);
          const media = source.sourceKind === "image" || source.sourceKind === "video";
          const Icon = source.sourceKind === "video" ? Film : Image;
          const icon = Math.max(10, nodeRadius * 1.3);
          return (
            <g
              aria-label={`${nodeDescription(source, scale)}${source.sourceKind === "generated" ? ". Press Enter to inspect this token." : ""}`}
              className={`influence-node ${source.sourceKind}`}
              data-node={source.contextIndex}
              key={source.contextIndex}
              onBlur={() => setHovered(null)}
              onClick={() => activate(node)}
              onFocus={() => setHovered(source.contextIndex)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  activate(node);
                }
              }}
              onMouseEnter={() => setHovered(source.contextIndex)}
              onMouseLeave={() => setHovered(null)}
              opacity={dimmed(source.contextIndex) ? 0.25 : 1}
              role="button"
              tabIndex={0}
            >
              <circle className="influence-node-hit" cx={node.x} cy={node.y} fill="transparent" r={nodeRadius + 6} />
              <circle cx={node.x} cy={node.y} fill={source.isSpecial ? "var(--surface-4)" : "var(--surface-3)"} r={nodeRadius} stroke={style.color} strokeDasharray={source.isSpecial ? "2 2" : undefined} strokeWidth={source.sourceKind === "generated" ? 2 : 1.5} />
              {media && <Icon aria-hidden="true" color="var(--text-primary)" height={icon} width={icon} x={node.x - icon / 2} y={node.y - icon / 2} />}
              {(layout.named.has(source.contextIndex) || active === source.contextIndex) && (
                <text dominantBaseline="middle" fill={active === source.contextIndex ? "var(--text-primary)" : "var(--text-secondary)"} fontSize={11} textAnchor={Math.abs(side) < 0.2 ? "middle" : side > 0 ? "start" : "end"} x={outside.x} y={outside.y}>
                  {truncate(sourceText(source), geometry.textLength)}
                </text>
              )}
            </g>
          );
        })}
        <g aria-hidden="true">
          <circle cx={center.x} cy={center.y} fill="var(--surface-3)" r={variant === "expanded" ? 34 : 25} stroke="var(--accent)" strokeWidth={3} />
          <text dominantBaseline="central" fill="var(--text-primary)" fontSize={variant === "expanded" ? 14 : 12} textAnchor="middle" x={center.x} y={center.y - 5}>{truncate(target.label, variant === "expanded" ? 9 : 7)}</text>
          <text dominantBaseline="central" fill="var(--accent-text)" fontSize={11} textAnchor="middle" x={center.x} y={center.y + 10}>#{target.index}</text>
        </g>
      </svg>
      {activeNode && (
        <div
          className="influence-tooltip"
          role="tooltip"
          style={{ left: `${String((activeNode.x / geometry.size) * 100)}%`, top: `${String((activeNode.y / geometry.size) * 100)}%` }}
        >
          <code>{sourceText(activeNode.source)}</code>
          <dl>
            <div><dt>Weight</dt><dd>{formatInfluenceWeight(activeNode.source.display)}{scale === "relative" && <small> of max · share {formatInfluenceWeight(activeNode.source.share)}</small>}</dd></div>
            <div><dt>Rank</dt><dd>#{activeNode.source.rank}</dd></div>
            <div><dt>Position</dt><dd>{sourcePosition(activeNode.source)}</dd></div>
            <div><dt>Kind</dt><dd>{sourceKindLabel(activeNode.source)}</dd></div>
          </dl>
          {activeNode.source.sourceKind === "generated" && <small>Click or press Enter to inspect this token.</small>}
        </div>
      )}
    </div>
  );
}
