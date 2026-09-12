import { describe, expect, it } from "vitest";
import type { ModelSummary } from "../api/types";
import { capabilityOf, capabilityReason, isUsable, supportsGeneration } from "./capabilities";

const model: ModelSummary = {
  id: "dense",
  name: "Dense fixture",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture",
  lifecycle: "unloaded",
  effectiveContextLimit: 2048,
  capabilities: {
    text_generation: { state: "full" },
    moe_routing: { state: "unsupported", reason: "This is a dense model." },
    cuda: { state: "unavailable_on_backend", reason: "CUDA was not discovered." },
  },
};

describe("capability helpers", () => {
  it("treats full and partial capability states as usable", () => {
    expect(isUsable(model, "text_generation")).toBe(true);
    expect(isUsable({ ...model, capabilities: { embeddings: { state: "partial" } } }, "embeddings")).toBe(true);
  });

  it("preserves precise reasons for disabled UI", () => {
    expect(isUsable(model, "moe_routing")).toBe(false);
    expect(capabilityReason(model, "moe_routing")).toBe("This is a dense model.");
    expect(capabilityOf(model, "audio")).toEqual({
      state: "unsupported",
      reason: "The model adapter did not report this capability.",
    });
  });

  it("accepts encoder-decoder checkpoints as generation models", () => {
    const seq2seq: ModelSummary = {
      ...model,
      task: "encoder_decoder_generation",
      capabilities: {
        text_generation: { state: "unsupported", reason: "Not a causal decoder." },
        encoder_decoder_generation: { state: "full" },
      },
    };
    expect(supportsGeneration(seq2seq)).toBe(true);
  });
});
