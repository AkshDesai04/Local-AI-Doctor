import type { AttentionAttribution, InfluenceSource, TokenEvent, TokenInfluence } from "../api/types";
import { displayTokenText } from "../utils/format";

export type InfluenceScale = "share" | "relative";

/** Weights are written as 0–1 numbers with two decimals; anything smaller reads "<0.01". */
export function formatInfluenceWeight(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value > 0 && value < 0.01) return "<0.01";
  return value.toFixed(2);
}

/**
 * The attention captured live during generation, in the influence shape. It keeps
 * single positions (media placeholders are not grouped, because only the top
 * retained positions were stored) and has no special-token flags.
 */
export function liveInfluence(attribution: AttentionAttribution, token: TokenEvent, runId: string, promptTokenCount: number | null): TokenInfluence {
  return {
    method: "live_attention",
    runId,
    tokenIndex: token.index,
    cached: true,
    target: { tokenId: token.tokenId, piece: token.piece, displayText: token.displayText, alternativeTokenId: null, alternativePiece: null },
    contextTokenCount: attribution.totalSourceCount,
    promptTokenCount,
    sources: attribution.sourceTokens.map((source) => ({
      sourceKind: source.media ? source.media.kind : source.sourceKind,
      contextIndex: source.contextIndex,
      span: null,
      tokenCount: 1,
      tokenId: source.tokenId,
      piece: source.piece,
      displayText: source.displayText,
      generatedTokenIndex: source.generatedTokenIndex ?? null,
      mediaIndex: source.media?.index ?? null,
      isSpecial: null,
      weight: source.weight,
    })),
    retainedWeight: attribution.retainedWeight,
    omittedWeight: attribution.omittedWeight,
    layers: null,
    capturedLayers: attribution.capturedLayers,
    headsPerLayer: attribution.capturedHeads || null,
    objective: null,
    objectiveValue: null,
    semantics: "Post-softmax self-attention captured while generating, averaged over the captured layers and heads. Allocation, not causal attribution.",
    normalization: "sum_to_one",
    modelFingerprint: null,
    durationMs: null,
    placement: null,
    device: null,
    dtype: null,
  };
}

export interface WebSource extends InfluenceSource {
  /** Weight as a share of the visible mass (equals `weight` unless special tokens are hidden). */
  share: number;
  /** The value drawn and written: the share, or the share relative to the largest shown share. */
  display: number;
  /** 1 = heaviest visible source. */
  rank: number;
}

export interface WebModel {
  /** The top-N visible sources, in context order. */
  shown: WebSource[];
  /** Every visible retained source, heaviest first. */
  ranked: WebSource[];
  /** Visible retained sources beyond the top N. */
  otherCount: number;
  /** Their share plus the mass never retained by the source limit. */
  otherShare: number;
  /** Mass of the special tokens removed from view (0 when they are shown). */
  hiddenSpecialWeight: number;
  maxShare: number;
}

/**
 * Pick what the web draws. Hiding special tokens removes them and rescales every
 * remaining share by the mass left (positions beyond the retained limit count as
 * visible, because their special status is unknown).
 */
export function prepareWeb(influence: Pick<TokenInfluence, "sources" | "omittedWeight">, options: { topN: number; hideSpecial: boolean; scale: InfluenceScale }): WebModel {
  const hidden = options.hideSpecial ? influence.sources.filter((source) => source.isSpecial === true) : [];
  const hiddenSpecialWeight = hidden.reduce((sum, source) => sum + source.weight, 0);
  const visible = options.hideSpecial ? influence.sources.filter((source) => source.isSpecial !== true) : influence.sources;
  const denominator = 1 - hiddenSpecialWeight;
  const share = (weight: number): number => (denominator > 0 ? weight / denominator : 0);
  const ordered = [...visible].sort((left, right) => right.weight - left.weight || left.contextIndex - right.contextIndex);
  const topN = Math.max(0, Math.floor(options.topN));
  const maxShare = ordered.length ? share(ordered[0]?.weight ?? 0) : 0;
  const ranked = ordered.map((source, index): WebSource => {
    const value = share(source.weight);
    return { ...source, share: value, rank: index + 1, display: options.scale === "relative" ? (maxShare > 0 ? value / maxShare : 0) : value };
  });
  const top = ranked.slice(0, topN);
  const rest = ranked.slice(topN);
  return {
    shown: [...top].sort((left, right) => left.contextIndex - right.contextIndex),
    ranked,
    otherCount: rest.length,
    otherShare: rest.reduce((sum, source) => sum + source.share, 0) + share(influence.omittedWeight),
    hiddenSpecialWeight,
    maxShare,
  };
}

export type ArcKind = "prompt" | "generated";

export interface RingArc {
  kind: ArcKind;
  /** Radians clockwise from 12 o'clock. */
  start: number;
  end: number;
  count: number;
}

export interface RingNode {
  source: WebSource;
  arc: ArcKind;
  angle: number;
  x: number;
  y: number;
}

/** A point at `radius` and `angle` (radians clockwise from 12 o'clock). */
export function polar(cx: number, cy: number, radius: number, angle: number): { x: number; y: number } {
  return { x: cx + radius * Math.sin(angle), y: cy - radius * Math.cos(angle) };
}

/**
 * Place sources on a ring in sequence order, clockwise from 12 o'clock. The prompt
 * arc (history, template, and media) and the generated arc share the circle in
 * proportion to their counts, each followed by a gap so the two read apart.
 */
export function ringLayout(sources: WebSource[], geometry: { cx: number; cy: number; radius: number; gap: number }): { nodes: RingNode[]; arcs: RingArc[] } {
  const groups: Array<{ kind: ArcKind; items: WebSource[] }> = [
    { kind: "prompt" as const, items: sources.filter((source) => source.sourceKind !== "generated") },
    { kind: "generated" as const, items: sources.filter((source) => source.sourceKind === "generated") },
  ].filter((group) => group.items.length > 0);
  const total = groups.reduce((sum, group) => sum + group.items.length, 0);
  const available = Math.max(0, 2 * Math.PI - groups.length * geometry.gap);
  const arcs: RingArc[] = [];
  const nodes: RingNode[] = [];
  let cursor = geometry.gap / 2;
  for (const group of groups) {
    const span = total ? (available * group.items.length) / total : 0;
    arcs.push({ kind: group.kind, start: cursor, end: cursor + span, count: group.items.length });
    group.items.forEach((source, index) => {
      const angle = cursor + ((index + 0.5) * span) / group.items.length;
      nodes.push({ source, arc: group.kind, angle, ...polar(geometry.cx, geometry.cy, geometry.radius, angle) });
    });
    cursor += span + geometry.gap;
  }
  return { nodes, arcs };
}

export interface LabelBox {
  key: number;
  text: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Unit normal of the label's edge; collisions push labels along it. */
  nx: number;
  ny: number;
}

const CHARACTER_WIDTH = 6.8;
export const LABEL_HEIGHT = 16;

/**
 * One weight pill per edge, alternating 45% and 60% of the way from the source
 * node toward the centre so neighbouring labels start out staggered. `textScale`
 * grows the pills with the text when the web is drawn smaller than its viewBox.
 */
export function edgeLabels(nodes: RingNode[], center: { x: number; y: number }, text: (node: RingNode) => string, textScale = 1): LabelBox[] {
  return nodes.map((node, index) => {
    const t = index % 2 === 0 ? 0.45 : 0.6;
    const dx = center.x - node.x;
    const dy = center.y - node.y;
    const length = Math.hypot(dx, dy) || 1;
    const label = text(node);
    return {
      key: node.source.contextIndex,
      text: label,
      x: node.x + dx * t,
      y: node.y + dy * t,
      width: (label.length * CHARACTER_WIDTH + 10) * textScale,
      height: LABEL_HEIGHT * textScale,
      nx: -dy / length,
      ny: dx / length,
    };
  });
}

function overlap(left: LabelBox, right: LabelBox, padding: number): { x: number; y: number } | null {
  const x = (left.width + right.width) / 2 + padding - Math.abs(left.x - right.x);
  const y = (left.height + right.height) / 2 + padding - Math.abs(left.y - right.y);
  return x > 0 && y > 0 ? { x, y } : null;
}

/** True when any two pills overlap (with `padding` px of breathing room). */
export function labelsOverlap(labels: LabelBox[], padding = 1): boolean {
  return labels.some((left, index) => labels.slice(index + 1).some((right) => overlap(left, right, padding) !== null));
}

/**
 * Push overlapping pills apart along their edge normals, so a label stays on (or
 * beside) its own line.
 */
export function separateLabels(labels: LabelBox[], padding = 2, iterations = 64): LabelBox[] {
  const placed = labels.map((label) => ({ ...label }));
  // A pill keeps the direction it first moved in, so one wedged between two others
  // escapes past both instead of bouncing between them.
  const directions = new Map<number, number>();
  // ponytail: O(n²) pairwise pass over at most 64 pills; a spatial grid if far more labels are ever drawn
  for (let pass = 0; pass < iterations; pass += 1) {
    let moved = false;
    for (let i = 0; i < placed.length; i += 1) {
      for (let j = i + 1; j < placed.length; j += 1) {
        const left = placed[i];
        const right = placed[j];
        if (!left || !right) continue;
        const amount = overlap(left, right, padding);
        if (!amount) continue;
        // Move along the normal far enough to clear the overlap on its dominant axis.
        const dominant = Math.max(Math.abs(right.nx), Math.abs(right.ny), 1e-6);
        const distance = (Math.abs(right.nx) >= Math.abs(right.ny) ? amount.x : amount.y) / dominant + 0.5;
        const side = (right.x - left.x) * right.nx + (right.y - left.y) * right.ny;
        const direction = directions.get(j) ?? (side === 0 ? (j % 2 === 0 ? 1 : -1) : Math.sign(side));
        directions.set(j, direction);
        right.x += right.nx * distance * direction;
        right.y += right.ny * distance * direction;
        moved = true;
      }
    }
    if (!moved) break;
  }
  return placed;
}

/** Stroke width, opacity, and sequential-ramp colour for an edge carrying `strength` (0–1 of the max). */
export function edgeStyle(strength: number): { width: number; opacity: number; color: string } {
  const value = Math.min(1, Math.max(0, Number.isFinite(strength) ? strength : 0));
  return {
    width: 0.75 + value * 5.25,
    opacity: 0.28 + value * 0.72,
    color: `var(--seq-${String(2 + Math.round(value * 3))})`,
  };
}

/** Human description of where a source sits. */
export function sourceKindLabel(source: InfluenceSource): string {
  if (source.sourceKind === "generated") return `generated #${String(source.generatedTokenIndex ?? "?")}`;
  if (source.sourceKind === "image" || source.sourceKind === "video") {
    const item = source.mediaIndex === null ? source.sourceKind : `${source.sourceKind} ${String(source.mediaIndex + 1)}`;
    return source.tokenCount > 1 ? `${item} · ${String(source.tokenCount)} positions` : item;
  }
  return source.isSpecial ? "prompt/history · special token" : "prompt/history";
}

/** "12" or "12–75" for a grouped span. */
export function sourcePosition(source: InfluenceSource): string {
  return source.span ? `${String(source.span[0])}–${String(source.span[1] - 1)}` : String(source.contextIndex);
}

/** The visible text for a source: the token's display text, or the media item's label. */
export function sourceText(source: InfluenceSource): string {
  if (source.sourceKind === "image" || source.sourceKind === "video") return source.displayText || source.sourceKind;
  return displayTokenText(source.displayText || source.piece || "") || "∅";
}
