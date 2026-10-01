import { describe, expect, it } from "vitest";
import type { AttentionAttribution, InfluenceSource, TokenEvent } from "../api/types";
import {
  edgeLabels,
  edgeStyle,
  formatInfluenceWeight,
  labelsOverlap,
  liveInfluence,
  prepareWeb,
  ringLayout,
  separateLabels,
  sourceKindLabel,
  type LabelBox,
  type WebSource,
} from "./influence";

function source(contextIndex: number, weight: number, overrides: Partial<InfluenceSource> = {}): InfluenceSource {
  return {
    sourceKind: "prompt",
    contextIndex,
    span: null,
    tokenCount: 1,
    tokenId: 100 + contextIndex,
    piece: `p${String(contextIndex)}`,
    displayText: `t${String(contextIndex)}`,
    generatedTokenIndex: null,
    mediaIndex: null,
    isSpecial: false,
    weight,
    ...overrides,
  };
}

function webSource(contextIndex: number, overrides: Partial<InfluenceSource> = {}): WebSource {
  return { ...source(contextIndex, 0.1, overrides), share: 0.1, display: 0.1, rank: contextIndex + 1 };
}

describe("influence weights", () => {
  it("writes two decimals, and <0.01 for anything smaller", () => {
    expect(formatInfluenceWeight(0.256)).toBe("0.26");
    expect(formatInfluenceWeight(1)).toBe("1.00");
    expect(formatInfluenceWeight(0.0099)).toBe("<0.01");
    expect(formatInfluenceWeight(0.004)).toBe("<0.01");
    expect(formatInfluenceWeight(0)).toBe("0.00");
    expect(formatInfluenceWeight(Number.NaN)).toBe("—");
  });

  it("keeps the top N in context order and summarises the rest plus omitted mass as other", () => {
    const web = prepareWeb(
      { sources: [source(0, 0.1), source(1, 0.4), source(2, 0.2), source(3, 0.05)], omittedWeight: 0.25 },
      { topN: 2, hideSpecial: false, scale: "share" },
    );

    expect(web.shown.map((item) => item.contextIndex)).toEqual([1, 2]);
    expect(web.shown.map((item) => item.rank)).toEqual([1, 2]);
    expect(web.ranked.map((item) => item.contextIndex)).toEqual([1, 2, 0, 3]);
    expect(web.otherCount).toBe(2);
    expect(web.otherShare).toBeCloseTo(0.1 + 0.05 + 0.25);
    expect(web.shown.reduce((sum, item) => sum + item.share, 0) + web.otherShare).toBeCloseTo(1);
  });

  it("hides special tokens and rescales the rest by the mass that remains", () => {
    const web = prepareWeb(
      { sources: [source(0, 0.5, { isSpecial: true }), source(1, 0.3), source(2, 0.2)], omittedWeight: 0 },
      { topN: 8, hideSpecial: true, scale: "share" },
    );

    expect(web.hiddenSpecialWeight).toBeCloseTo(0.5);
    expect(web.shown.map((item) => item.contextIndex)).toEqual([1, 2]);
    expect(web.shown.map((item) => item.share)).toEqual([expect.closeTo(0.6), expect.closeTo(0.4)]);
    expect(web.shown.reduce((sum, item) => sum + item.share, 0) + web.otherShare).toBeCloseTo(1);
  });

  it("scales relative to the largest shown weight on request", () => {
    const sources = [source(0, 0.1), source(1, 0.4), source(2, 0.2)];
    const share = prepareWeb({ sources, omittedWeight: 0.3 }, { topN: 3, hideSpecial: false, scale: "share" });
    const relative = prepareWeb({ sources, omittedWeight: 0.3 }, { topN: 3, hideSpecial: false, scale: "relative" });

    expect(share.shown.map((item) => item.display)).toEqual([0.1, 0.4, 0.2]);
    expect(relative.shown.map((item) => item.display)).toEqual([expect.closeTo(0.25), 1, expect.closeTo(0.5)]);
    expect(relative.shown.map((item) => item.share)).toEqual([0.1, 0.4, 0.2]);
  });
});

describe("ring layout", () => {
  const geometry = { cx: 100, cy: 100, radius: 80, gap: 0.2 };

  it("places sources clockwise from 12 o'clock on the ring radius", () => {
    const { nodes } = ringLayout([webSource(0), webSource(1), webSource(2), webSource(3)], geometry);

    expect(nodes.map((node) => node.source.contextIndex)).toEqual([0, 1, 2, 3]);
    for (const node of nodes) expect(Math.hypot(node.x - 100, node.y - 100)).toBeCloseTo(80);
    // The first source sits just clockwise of the top; angles increase clockwise.
    expect(nodes[0]?.x).toBeGreaterThan(100);
    expect(nodes[0]?.y).toBeLessThan(100);
    const angles = nodes.map((node) => node.angle);
    expect([...angles].sort((left, right) => left - right)).toEqual(angles);
  });

  it("splits the prompt and generated arcs with a gap between them", () => {
    const sources = [webSource(0), webSource(1), webSource(2, { sourceKind: "image", span: [2, 6], tokenCount: 4 }), webSource(10, { sourceKind: "generated", generatedTokenIndex: 0 })];
    const { nodes, arcs } = ringLayout(sources, geometry);

    expect(arcs.map((arc) => [arc.kind, arc.count])).toEqual([["prompt", 3], ["generated", 1]]);
    const [prompt, generated] = arcs;
    expect((generated?.start ?? 0) - (prompt?.end ?? 0)).toBeCloseTo(0.2);
    // Arcs share the circle in proportion to their counts.
    expect((prompt?.end ?? 0) - (prompt?.start ?? 0)).toBeCloseTo(3 * ((generated?.end ?? 0) - (generated?.start ?? 0)));
    expect(nodes.find((node) => node.source.sourceKind === "generated")?.arc).toBe("generated");
    expect(nodes.find((node) => node.source.sourceKind === "image")?.arc).toBe("prompt");
  });

  it("keeps a gap at the top when only one arc exists", () => {
    const { arcs } = ringLayout([webSource(0), webSource(1)], geometry);
    expect(arcs).toHaveLength(1);
    expect(arcs[0]?.start).toBeCloseTo(0.1);
    expect(arcs[0]?.end).toBeCloseTo(2 * Math.PI - 0.1);
  });
});

describe("edge labels", () => {
  it("alternate between 45% and 60% of the way to the centre", () => {
    const nodes = ringLayout([webSource(0), webSource(1)], { cx: 0, cy: 0, radius: 100, gap: 0.2 }).nodes;
    const labels = edgeLabels(nodes, { x: 0, y: 0 }, () => "0.50");

    expect(Math.hypot(labels[0]?.x ?? 0, labels[0]?.y ?? 0)).toBeCloseTo(55);
    expect(Math.hypot(labels[1]?.x ?? 0, labels[1]?.y ?? 0)).toBeCloseTo(40);
    expect(labels[0]?.text).toBe("0.50");
  });

  it("are pushed apart along their edge normals until no pills overlap", () => {
    const pill = (key: number, x: number, y: number, nx: number, ny: number): LabelBox => ({ key, text: "0.12", x, y, width: 38, height: 16, nx, ny });
    const crowded = [pill(0, 50, 50, 0, 1), pill(1, 52, 51, 0, 1), pill(2, 49, 49, 1, 0), pill(3, 50, 52, 0.6, 0.8)];
    expect(labelsOverlap(crowded)).toBe(true);

    const separated = separateLabels(crowded);

    expect(labelsOverlap(separated)).toBe(false);
    expect(separated.map((item) => item.key)).toEqual([0, 1, 2, 3]);
    // The first label keeps its place; the others moved only along their normals.
    expect(separated[0]).toMatchObject({ x: 50, y: 50 });
    expect(separated[1]?.x).toBe(52);
    expect(separated[2]?.y).toBe(49);
  });

  it("use the sequential ramp, widening and brightening with weight", () => {
    expect(edgeStyle(0)).toMatchObject({ color: "var(--seq-2)" });
    expect(edgeStyle(1)).toMatchObject({ color: "var(--seq-5)" });
    expect(edgeStyle(1).width).toBeGreaterThan(edgeStyle(0.2).width);
    expect(edgeStyle(1).opacity).toBe(1);
  });
});

describe("live attention", () => {
  it("maps the captured sources without inventing special-token flags", () => {
    const attribution: AttentionAttribution = {
      method: "mean_causal_self_attention",
      aggregation: "arithmetic_mean_over_layers_and_heads",
      semantics: "attention_weights_not_causal_contributions",
      sourceTokens: [
        { contextIndex: 0, tokenId: 5, piece: "<|im_start|>", displayText: "<|im_start|>", sourceKind: "prompt", weight: 0.6 },
        { contextIndex: 2, tokenId: 9, piece: "<|image_pad|>", displayText: "", sourceKind: "prompt", media: { kind: "image", index: 0 }, weight: 0.1 },
        { contextIndex: 4, tokenId: 11, piece: "Hi", displayText: "Hi", sourceKind: "generated", generatedTokenIndex: 0, weight: 0.2 },
      ],
      capturedLayers: [0, 1],
      capturedHeads: 4,
      normalized: true,
      totalSourceCount: 5,
      retainedSourceCount: 3,
      retainedWeight: 0.9,
      omittedWeight: 0.1,
    };
    const token = { index: 1, tokenId: 12, piece: "!", displayText: "!", reasoningSegment: "answer" } as TokenEvent;

    const influence = liveInfluence(attribution, token, "run-1", 4);

    expect(influence.method).toBe("live_attention");
    expect(influence.sources.map((item) => [item.sourceKind, item.isSpecial])).toEqual([["prompt", null], ["image", null], ["generated", null]]);
    expect(influence.sources[2]?.generatedTokenIndex).toBe(0);
    expect(influence.omittedWeight).toBe(0.1);
    expect(sourceKindLabel(influence.sources[1] as InfluenceSource)).toBe("image 1");
  });
});
