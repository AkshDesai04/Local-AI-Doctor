import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, subscribeToRun } from "../api/client";
import type * as ApiClientModule from "../api/client";
import type { ChatSummary, ModelSummary, RunStreamEvent } from "../api/types";
import { useWorkbench } from "./useWorkbench";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof ApiClientModule>();
  return {
    ...actual,
    api: {
      ...actual.api,
      chats: vi.fn(),
      configuration: vi.fn(),
      generate: vi.fn(),
      health: vi.fn(),
      messages: vi.fn(),
      models: vi.fn(),
    },
    subscribeToRun: vi.fn(),
  };
});

const mockedApi = vi.mocked(api);
const mockedSubscribeToRun = vi.mocked(subscribeToRun);

const model: ModelSummary = {
  id: "generation-fixture",
  name: "Generation fixture",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture-fingerprint",
  lifecycle: "unloaded",
  loadedDevice: null,
  capabilities: { text_generation: { state: "full" }, streaming: { state: "full" } },
  effectiveContextLimit: 4096,
};

const chat: ChatSummary = {
  id: "chat-fixture",
  title: "Fixture chat",
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-01T00:00:00Z",
  pinned: false,
  archived: false,
};

describe("runtime lifecycle synchronization", () => {
  let emitStreamEvent: ((event: RunStreamEvent) => void) | undefined;

  beforeEach(() => {
    emitStreamEvent = undefined;
    mockedApi.health.mockReset()
      .mockResolvedValueOnce({ status: "ok" })
      .mockResolvedValue({ status: "ok", loadedModelId: model.id, loadedDevice: "cpu" });
    mockedApi.models.mockReset().mockResolvedValue([model]);
    mockedApi.chats.mockReset().mockImplementation((archived = false) => Promise.resolve(archived ? [] : [chat]));
    mockedApi.configuration.mockReset().mockResolvedValue({ effective: {}, precedence: [] });
    mockedApi.messages.mockReset().mockResolvedValue([]);
    mockedApi.generate.mockReset().mockResolvedValue({
      runId: "run-fixture",
      messageId: "assistant-fixture",
      userMessageId: "user-fixture",
      modelId: model.id,
    });
    mockedSubscribeToRun.mockReset().mockImplementation((_runId, onEvent) => {
      emitStreamEvent = onEvent;
      return vi.fn();
    });
  });

  it("refreshes health and models after a generation terminal event", async () => {
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.booting).toBe(false));
    expect(result.current.selectedModel?.lifecycle).toBe("unloaded");

    await act(async () => {
      await result.current.submit("Hello locally");
    });
    expect(emitStreamEvent).toBeDefined();

    act(() => {
      emitStreamEvent?.({ version: 1, sequence: 1, type: "completed" });
    });

    await waitFor(() => expect(result.current.selectedModel?.lifecycle).toBe("loaded"));
    expect(result.current.health).toMatchObject({ loadedModelId: model.id, loadedDevice: "cpu" });
    expect(mockedApi.models.mock.calls).toHaveLength(2);
    expect(mockedApi.health.mock.calls).toHaveLength(2);
  });

  it("separates first-token receipt from the first visible answer and uses backend context reservation", async () => {
    const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.booting).toBe(false));

    await act(async () => {
      await result.current.submit("Explain this");
    });
    expect(result.current.selectedRun?.metrics?.context?.reservedOutputTokens).toBeUndefined();

    act(() => {
      emitStreamEvent?.({
        version: 1,
        sequence: 1,
        type: "stage.changed",
        stage: "running",
        metrics: { context: { effectiveLimit: 4096, reservedOutputTokens: 96 } },
      });
    });
    expect(result.current.selectedRun?.metrics?.context?.reservedOutputTokens).toBe(96);

    now.mockReturnValue(1_020);
    act(() => {
      emitStreamEvent?.({
        version: 1,
        sequence: 2,
        type: "token",
        token: { index: 0, tokenId: 1, piece: "<think>", displayText: "<think>", reasoningSegment: "reasoning" },
      });
    });
    expect(result.current.selectedRun?.metrics?.timing).toMatchObject({ clientTtftMs: 20 });
    expect(result.current.selectedRun?.metrics?.timing?.firstVisibleTextMs).toBeUndefined();

    now.mockReturnValue(1_035);
    act(() => {
      emitStreamEvent?.({
        version: 1,
        sequence: 3,
        type: "token",
        token: { index: 1, tokenId: 2, piece: " ", displayText: " ", reasoningSegment: "answer" },
      });
    });
    expect(result.current.selectedRun?.metrics?.timing?.firstVisibleTextMs).toBeUndefined();

    now.mockReturnValue(1_060);
    act(() => {
      emitStreamEvent?.({
        version: 1,
        sequence: 4,
        type: "token",
        token: { index: 2, tokenId: 3, piece: "Answer", displayText: "Answer", reasoningSegment: "answer" },
      });
    });
    expect(result.current.selectedRun?.metrics?.timing).toMatchObject({ clientTtftMs: 20, firstVisibleTextMs: 60 });

    now.mockRestore();
  });

  it("treats unclassified text as visible for models without reasoning segmentation", async () => {
    const now = vi.spyOn(performance, "now").mockReturnValue(2_000);
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.booting).toBe(false));

    await act(async () => {
      await result.current.submit("Answer directly");
    });
    now.mockReturnValue(2_025);
    act(() => {
      emitStreamEvent?.({
        version: 1,
        sequence: 1,
        type: "token",
        token: { index: 0, tokenId: 1, piece: "Answer", displayText: "Answer", reasoningSegment: "unknown" },
      });
    });

    expect(result.current.selectedRun?.metrics?.timing).toMatchObject({ clientTtftMs: 25, firstVisibleTextMs: 25 });
    now.mockRestore();
  });
});
