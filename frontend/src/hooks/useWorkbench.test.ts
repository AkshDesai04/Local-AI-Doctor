import { describe, expect, it } from "vitest";
import type { ConfigurationSnapshot, Message } from "../api/types";
import { generationSettingsFromConfiguration, resolveGenerationParent } from "./useWorkbench";

const messages: Message[] = [
  { id: "user-root", chatId: "chat", role: "user", content: "First", createdAt: "2026-01-01T00:00:00Z", status: "complete" },
  { id: "assistant-one", chatId: "chat", role: "assistant", content: "One", createdAt: "2026-01-01T00:00:01Z", status: "complete", parentMessageId: "user-root" },
  { id: "user-two", chatId: "chat", role: "user", content: "Second", createdAt: "2026-01-01T00:00:02Z", status: "complete", parentMessageId: "assistant-one" },
  { id: "assistant-two", chatId: "chat", role: "assistant", content: "Two", createdAt: "2026-01-01T00:00:03Z", status: "complete", parentMessageId: "user-two" },
];

describe("generation lineage", () => {
  it("uses the latest assistant for an ordinary continuation", () => {
    expect(resolveGenerationParent(messages, undefined)).toBe("assistant-two");
  });

  it("preserves an explicit branch parent, including the conversation root", () => {
    expect(resolveGenerationParent(messages, "assistant-one")).toBe("assistant-one");
    expect(resolveGenerationParent(messages, null)).toBeNull();
  });
});

describe("configured generation defaults", () => {
  it("derives browser controls from the effective backend configuration", () => {
    const configuration: ConfigurationSnapshot = {
      effective: {
        runtime: { device: "cpu", dtype: "bfloat16" },
        inference: {
          instrumentation: "full",
          deterministic_reference_mode: true,
          defaults: {
            max_output_tokens: 77,
            temperature: 0.2,
            top_k: 7,
            top_p: 0.8,
            min_p: 0.1,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            alternatives: 4,
          },
        },
      },
      precedence: [],
    };

    expect(generationSettingsFromConfiguration(configuration)).toMatchObject({
      device: "cpu",
      dtype: "bfloat16",
      instrumentation: "full",
      deterministic: true,
      reasoning: true,
      maxOutputTokens: 77,
      topK: 7,
      alternatives: 4,
    });
  });
});
