import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { InfluenceSource } from "../api/types";
import { prepareWeb } from "../domain/influence";
import { InfluenceWeb } from "./InfluenceWeb";

function source(contextIndex: number, weight: number, overrides: Partial<InfluenceSource> = {}): InfluenceSource {
  return {
    sourceKind: "prompt",
    contextIndex,
    span: null,
    tokenCount: 1,
    tokenId: contextIndex,
    piece: `piece-${String(contextIndex)}`,
    displayText: `word${String(contextIndex)}`,
    generatedTokenIndex: null,
    mediaIndex: null,
    isSpecial: false,
    weight,
    ...overrides,
  };
}

const sources = [
  source(0, 0.05, { isSpecial: true, displayText: "<|im_start|>" }),
  source(1, 0.42),
  source(2, 0.004),
  source(3, 0.2, { sourceKind: "image", span: [3, 9], tokenCount: 6, tokenId: null, piece: null, displayText: "image 1", mediaIndex: 0 }),
  source(9, 0.33, { sourceKind: "generated", generatedTokenIndex: 0, displayText: " Paris" }),
];

function renderWeb(onSelect = vi.fn(), labelLimit = 64): ReturnType<typeof vi.fn> {
  const web = prepareWeb({ sources, omittedWeight: 0 }, { topN: 24, hideSpecial: false, scale: "share" });
  render(<InfluenceWeb labelLimit={labelLimit} onSelectGeneratedToken={onSelect} scale="share" target={{ label: "!", index: 1 }} title="Influence web" variant="compact" web={web} />);
  return onSelect;
}

describe("InfluenceWeb", () => {
  it("draws one edge per source and writes each weight on its edge", () => {
    renderWeb();
    expect(document.querySelectorAll("[data-edge]")).toHaveLength(5);
    const labels = [...document.querySelectorAll("[data-edge-label] text")].map((item) => item.textContent);
    expect(labels.sort()).toEqual(["0.05", "0.20", "0.33", "0.42", "<0.01"].sort());
  });

  it("labels only the strongest edges when asked to", () => {
    renderWeb(vi.fn(), 2);
    const labels = [...document.querySelectorAll("[data-edge-label] text")].map((item) => item.textContent);
    expect(labels.sort()).toEqual(["0.33", "0.42"]);
  });

  it("makes nodes focusable in sequence order and selects generated tokens with Enter", async () => {
    const user = userEvent.setup();
    const onSelect = renderWeb();
    const nodes = screen.getAllByRole("button");
    expect(nodes.map((node) => node.getAttribute("data-node"))).toEqual(["0", "1", "2", "3", "9"]);

    await user.tab();
    expect(nodes[0]).toHaveFocus();
    await user.tab();
    await user.tab();
    await user.tab();
    await user.tab();
    expect(nodes[4]).toHaveFocus();
    expect(screen.getByRole("tooltip")).toHaveTextContent("generated #0");
    await user.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it("describes each node for assistive technology and shows a tooltip on hover", () => {
    renderWeb();
    const image = screen.getByRole("button", { name: /image 1, image 1 · 6 positions, position 3–8, weight 0\.20, rank 3/ });
    fireEvent.mouseEnter(image);
    const tooltip = screen.getByRole("tooltip");
    expect(tooltip).toHaveTextContent("0.20");
    expect(tooltip).toHaveTextContent("#3");
    // Other edges dim while one source is highlighted.
    expect(document.querySelector('[data-edge="1"]')?.getAttribute("opacity")).toBe("0.08");
    fireEvent.mouseLeave(image);
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("pins a prompt node's highlight with Enter instead of selecting it", async () => {
    const user = userEvent.setup();
    const onSelect = renderWeb();
    const prompt = screen.getAllByRole("button")[1];
    prompt?.focus();
    await user.keyboard("{Enter}");
    prompt?.blur();
    expect(screen.getByRole("tooltip")).toHaveTextContent("word1");
    expect(onSelect).not.toHaveBeenCalled();
  });
});
