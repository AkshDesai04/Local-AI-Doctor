import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import type { InfluenceSource, ModelSummary, RunDetails, TokenEvent, TokenInfluence } from "../api/types";
import { InfluenceSection } from "./InfluenceSection";

const model: ModelSummary = {
  id: "model-1",
  name: "Model",
  architecture: "Qwen3ForCausalLM",
  task: "text_generation",
  fingerprint: "fixture",
  lifecycle: "loaded",
  capabilities: { token_influence: { state: "partial", reason: "attention allocation and gradient×input saliency; not causal attribution" } },
  effectiveContextLimit: 4096,
};

const token: TokenEvent = {
  index: 2,
  tokenId: 30,
  piece: "Ġthere",
  displayText: " there",
  reasoningSegment: "answer",
  rawAlternatives: [
    { tokenId: 30, piece: "Ġthere", probability: 0.6, logProbability: -0.5, logit: 3, rank: 1 },
    { tokenId: 31, piece: "Ġfriend", probability: 0.3, logProbability: -1.2, logit: 2, rank: 2 },
  ],
};

const run: RunDetails = {
  id: "run-1",
  modelId: "model-1",
  status: "complete",
  createdAt: "2026-10-01T00:00:00Z",
  branchableThroughTokenIndex: 2,
  tokens: [token],
};

function source(contextIndex: number, weight: number, overrides: Partial<InfluenceSource> = {}): InfluenceSource {
  return { sourceKind: "prompt", contextIndex, span: null, tokenCount: 1, tokenId: contextIndex, piece: `p${String(contextIndex)}`, displayText: `word${String(contextIndex)}`, generatedTokenIndex: null, mediaIndex: null, isSpecial: false, weight, ...overrides };
}

function influence(method: TokenInfluence["method"], overrides: Partial<TokenInfluence> = {}): TokenInfluence {
  const sources = [source(0, 0.5, { isSpecial: true }), source(1, 0.3), source(4, 0.2, { sourceKind: "generated", generatedTokenIndex: 0 })];
  return {
    method,
    runId: "run-1",
    tokenIndex: 2,
    cached: false,
    target: { tokenId: 30, piece: "Ġthere", displayText: " there", alternativeTokenId: null, alternativePiece: null },
    contextTokenCount: 6,
    promptTokenCount: 4,
    sources,
    retainedWeight: 1,
    omittedWeight: 0,
    layers: method === "attention" ? [{ layer: 0, sources: [source(1, 1)], retainedWeight: 1, omittedWeight: 0 }, { layer: 1, sources, retainedWeight: 1, omittedWeight: 0 }] : null,
    capturedLayers: method === "attention" ? [0, 1] : [],
    headsPerLayer: 4,
    objective: method === "gradient_x_input" ? "log_probability" : null,
    objectiveValue: method === "gradient_x_input" ? -0.51 : null,
    semantics: "",
    normalization: "sum_to_one",
    modelFingerprint: "fixture",
    durationMs: 42,
    placement: "gpu",
    device: "cuda:0",
    dtype: "bfloat16",
    ...overrides,
  };
}

function edgeLabels(): string[] {
  return [...document.querySelectorAll(".influence-section [data-edge-label] text")].map((item) => item.textContent ?? "").sort();
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("InfluenceSection", () => {
  it("computes attention on request, then switches layer and method", async () => {
    const user = userEvent.setup();
    const request = vi.spyOn(api, "tokenInfluence").mockImplementation((_run, _token, body) => Promise.resolve(influence(body.method)));
    const onSelect = vi.fn();
    render(<InfluenceSection model={model} onSelectToken={onSelect} run={run} token={token} />);

    expect(request).not.toHaveBeenCalled();
    expect(screen.getByRole("note", { name: "Influence caveat" })).toHaveTextContent("allocation, not causation");
    await user.click(screen.getByRole("button", { name: "Compute attention" }));

    await waitFor(() => expect(edgeLabels()).toEqual(["0.20", "0.30", "0.50"]));
    expect(request).toHaveBeenCalledWith("run-1", 2, { method: "attention", layers: "all", sourceLimit: 128 });

    await user.selectOptions(screen.getByRole("combobox", { name: "Attention layer" }), "0");
    expect(edgeLabels()).toEqual(["1.00"]);

    await user.click(screen.getByRole("radio", { name: "Gradient × input" }));
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
    expect(request).toHaveBeenLastCalledWith("run-1", 2, { method: "gradient_x_input", alternativeTokenId: null, sourceLimit: 128 });
    expect(await screen.findByText("log p -0.51")).toBeInTheDocument();
    expect(screen.getByRole("note", { name: "Influence caveat" })).toHaveTextContent("not causal attribution");

    await user.selectOptions(screen.getByRole("combobox", { name: "Gradient target" }), "31");
    await waitFor(() => expect(request).toHaveBeenLastCalledWith("run-1", 2, { method: "gradient_x_input", alternativeTokenId: 31, sourceLimit: 128 }));

    // Back to attention is answered from the client cache.
    await user.click(screen.getByRole("radio", { name: "Attention" }));
    expect(request).toHaveBeenCalledTimes(3);
    await user.selectOptions(screen.getByRole("combobox", { name: "Attention layer" }), "mean");

    // A generated source in the table selects its token.
    await user.click(screen.getByText(/Table view/));
    const table = screen.getByRole("table", { name: "Influence sources by weight" });
    await user.click(within(table).getByRole("row", { name: /generated #0/ }));
    expect(onSelect).toHaveBeenCalledWith(0);
  }, 20_000);

  it("hides special tokens and says the remaining shares were rescaled", async () => {
    const user = userEvent.setup();
    vi.spyOn(api, "tokenInfluence").mockResolvedValue(influence("attention"));
    render(<InfluenceSection model={model} onSelectToken={vi.fn()} run={run} token={token} />);
    await user.click(screen.getByRole("button", { name: "Compute attention" }));
    await waitFor(() => expect(edgeLabels()).toHaveLength(3));

    await user.click(screen.getByRole("switch", { name: /Hide special/ }));

    expect(edgeLabels()).toEqual(["0.40", "0.60"]);
    expect(screen.getByText(/rescaled to the remaining 0.50/)).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "Relative to max" }));
    expect(edgeLabels()).toEqual(["0.67", "1.00"]);
  });

  it("shows the backend's message and hint when an analysis is refused", async () => {
    const user = userEvent.setup();
    vi.spyOn(api, "tokenInfluence").mockRejectedValue(new ApiError("the recorded model fingerprint no longer matches the registered checkpoint", 409, {
      error: { code: "model_fingerprint_changed", message: "the recorded model fingerprint no longer matches the registered checkpoint", hint: "Restore the original checkpoint and refresh the model registry." },
    }));
    render(<InfluenceSection model={model} onSelectToken={vi.fn()} run={run} token={token} />);

    await user.click(screen.getByRole("button", { name: "Compute attention" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("fingerprint no longer matches");
    expect(alert).toHaveTextContent("Restore the original checkpoint");
    expect(within(alert).getByRole("button", { name: "Retry" })).toBeEnabled();
  });

  it("explains why computing is unavailable instead of offering it", () => {
    render(<InfluenceSection model={model} onSelectToken={vi.fn()} run={{ ...run, status: "running" }} token={token} />);
    expect(screen.getByText("Available once the run completes or is cancelled.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Compute/ })).toBeNull();
  });

  it("opens the expanded web in a dialog with every edge labelled", async () => {
    const user = userEvent.setup();
    vi.spyOn(api, "tokenInfluence").mockResolvedValue(influence("attention"));
    render(<InfluenceSection model={model} onSelectToken={vi.fn()} run={run} token={token} />);
    await user.click(screen.getByRole("button", { name: "Compute attention" }));
    await waitFor(() => expect(edgeLabels()).toHaveLength(3));

    await user.click(screen.getByRole("button", { name: "Expand influence web" }));

    const dialog = screen.getByRole("dialog", { name: "Expanded influence web" });
    expect(within(dialog).getByRole("group", { name: /Influence web for token #2/ })).toBeInTheDocument();
    expect(dialog.querySelectorAll("[data-edge-label]")).toHaveLength(3);
    expect(within(dialog).getByRole("table", { name: "Influence sources by weight" })).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "Close influence web" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
