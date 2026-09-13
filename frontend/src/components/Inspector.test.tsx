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
      { tokenId: 11, piece: "alternate", probability: 0.3, logProbability: -1.2, logit: 2, rank: 2 },
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

    await user.click(screen.getByRole("button", { name: /Select token alternate from the raw distribution/i }));
    await user.click(screen.getByRole("button", { name: /Branch out with selected token/i }));

    expect(onBranchAlternative).toHaveBeenCalledWith(4, "raw", expect.objectContaining({ tokenId: 11, rank: 2 }));
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
});
