import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ModelSummary, RunDetails } from "../api/types";
import { Inspector } from "./Inspector";

const model: ModelSummary = {
  id: "model-1",
  name: "Model",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture",
  lifecycle: "loaded",
  loadedDevice: "cuda",
  capabilities: { raw_logits: { state: "full" }, text_generation: { state: "full" } },
  effectiveContextLimit: 4096,
};

const run: RunDetails = {
  id: "run-1",
  chatId: "chat-1",
  messageId: "assistant-1",
  modelId: "model-1",
  status: "complete",
  createdAt: "2026-09-12T00:00:00Z",
  branchableThroughTokenIndex: 4,
  tokens: [{
    index: 4,
    tokenId: 10,
    piece: "chosen",
    displayText: "chosen",
    reasoningSegment: "answer",
    rawAlternatives: [
      { tokenId: 10, piece: "chosen", probability: 0.6, logProbability: -0.5, logit: 3, rank: 1 },
      { tokenId: 11, piece: "Ġalternate", probability: 0.3, logProbability: -1.2, logit: 2, rank: 2 },
    ],
  }],
};

function view(nerdMode: boolean, onBranchAlternative = vi.fn().mockResolvedValue(undefined)): React.ReactNode {
  return <Inspector activeTab="tokens" branching={false} configuration={null} health={null} model={model} nerdMode={nerdMode} onBranchAlternative={onBranchAlternative} onClose={vi.fn()} onSelectToken={vi.fn()} onTabChange={vi.fn()} open run={run} selectedToken={4} />;
}

describe("alternative-token branching", () => {
  it("only exposes branching controls in Nerd Mode", () => {
    render(view(false));
    expect(screen.getByText("Token details are hidden")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Select token alternate/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Branch out with selected token/i })).not.toBeInTheDocument();
  });

  it("branches from the selected distribution and token", async () => {
    const user = userEvent.setup();
    const onBranchAlternative = vi.fn().mockResolvedValue(undefined);
    render(view(true, onBranchAlternative));

    const alternative = screen.getByRole("button", { name: /Select token alternate from the raw distribution/i });
    expect(alternative).toHaveTextContent("alternate");
    expect(alternative).not.toHaveTextContent("Ġalternate");
    expect(alternative.getAttribute("title")).toContain("Raw tokenizer piece: Ġalternate");
    await user.click(alternative);
    await user.click(screen.getByRole("button", { name: /Branch out with selected token/i }));

    expect(onBranchAlternative).toHaveBeenCalledWith(4, "raw", expect.objectContaining({ tokenId: 11, rank: 2 }));
  });

  it("keeps older runs usable when no attention attribution was captured", () => {
    render(view(true));

    const attentionPanel = screen.getByLabelText("Context attention");
    expect(attentionPanel).toHaveTextContent("Not captured for this token");
    expect(attentionPanel).toHaveTextContent("Full or Expert instrumentation");
    expect(screen.getByText("All generated tokens")).toBeInTheDocument();
  });

  it("hides raw protocol events and their export actions outside Nerd Mode", () => {
    render(<Inspector activeTab="events" branching={false} configuration={null} health={null} model={model} nerdMode={false} onBranchAlternative={vi.fn()} onClose={vi.fn()} onSelectToken={vi.fn()} onTabChange={vi.fn()} open run={{ ...run, rawEvents: [{ version: 1, sequence: 1, type: "token", token: { ...run.tokens[0]!, displayText: "<|end|>" } }] }} selectedToken={4} />);

    expect(screen.getByText("Raw events are hidden")).toBeInTheDocument();
    expect(screen.queryByText(/<\|end\|>/u)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "JSON" })).not.toBeInTheDocument();
  });

  it("only reveals token text in chart tooltips during Nerd Mode", () => {
    const traceRun: RunDetails = {
      ...run,
      tokens: [{
        ...run.tokens[0]!,
        piece: "<|eot_id|>",
        displayText: "<|eot_id|>",
        rawProbability: 0.42,
        samplingProbability: 0.4,
      }],
    };
    const inspector = (nerdMode: boolean): React.ReactNode => (
      <Inspector activeTab="probability" branching={false} configuration={null} health={null} model={model} nerdMode={nerdMode} onBranchAlternative={vi.fn()} onClose={vi.fn()} onSelectToken={vi.fn()} onTabChange={vi.fn()} open run={traceRun} selectedToken={4} />
    );
    const { container, rerender } = render(inspector(false));

    expect(container.querySelector(".chart-point title")).toHaveTextContent("#4 · 0.42");
    expect(container.querySelector(".chart-point title")).not.toHaveTextContent("<|eot_id|>");

    rerender(inspector(true));
    expect(container.querySelector(".chart-point title")).toHaveTextContent("<|eot_id|>");
  });

  it("shows the complete model context with retained attention sources heat-highlighted", async () => {
    const user = userEvent.setup();
    const onSelectToken = vi.fn();
    const attributionBase = {
      method: "mean_causal_self_attention",
      aggregation: "arithmetic_mean_over_layers_and_heads",
      semantics: "attention_weights_not_causal_contributions",
      capturedLayers: [0, 1],
      capturedHeads: 4,
      normalized: true,
      totalSourceCount: 3,
      retainedSourceCount: 2,
      retainedWeight: 0.8,
      omittedWeight: 0.2,
    };
    const attentionRun: RunDetails = {
      ...run,
      metrics: { promptTokens: 2 },
      branchableThroughTokenIndex: 1,
      tokens: [
        {
          index: 0,
          tokenId: 20,
          piece: "prior",
          displayText: "prior",
          reasoningSegment: "answer",
          attentionAttribution: {
            ...attributionBase,
            totalSourceCount: 2,
            sourceTokens: [],
            contextTokens: [
              { contextIndex: 0, tokenId: 1, piece: "system", displayText: "system", sourceKind: "prompt" },
              { contextIndex: 1, tokenId: 2, piece: "question", displayText: "question", sourceKind: "prompt" },
            ],
          },
        },
        {
          index: 1,
          tokenId: 21,
          piece: "answer",
          displayText: "answer",
          reasoningSegment: "answer",
          attentionAttribution: {
            ...attributionBase,
            sourceTokens: [
              { contextIndex: 0, tokenId: 1, piece: "system", displayText: "system", sourceKind: "prompt", weight: 0.5 },
              { contextIndex: 2, tokenId: 20, piece: "prior", displayText: "prior", sourceKind: "generated", generatedTokenIndex: 0, weight: 0.3 },
            ],
          },
        },
      ],
    };

    render(<Inspector activeTab="tokens" branching={false} configuration={null} health={null} model={model} nerdMode onBranchAlternative={vi.fn()} onClose={vi.fn()} onSelectToken={onSelectToken} onTabChange={vi.fn()} open run={attentionRun} selectedToken={1} />);

    const attentionPanel = screen.getByLabelText("Context attention for token 1");
    expect(attentionPanel).toHaveTextContent("2 of 3 positions retained");
    expect(attentionPanel).toHaveTextContent("80% attention mass shown");
    expect(screen.getByLabelText(/system, mean attention 50%/i)).toHaveClass("weighted", "strongest");
    expect(screen.getByLabelText(/question, lower-weight source not individually retained/i)).toHaveClass("unretained");
    const prior = screen.getByRole("button", { name: /prior, mean attention 30%/i });
    await user.click(prior);
    expect(onSelectToken).toHaveBeenCalledWith(0);
    expect(screen.getByRole("note")).toHaveTextContent("not causal contribution scores");
    expect(screen.getByRole("note")).toHaveTextContent("do not prove grounding or hallucination");
  });
});
