import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, subscribeToRun } from "../api/client";
import type * as ApiClientModule from "../api/client";
import type { ChatSummary, Message, ModelSummary, RunDetails, RunStreamEvent } from "../api/types";
import { useWorkbench } from "./useWorkbench";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof ApiClientModule>();
  return {
    ...actual,
    api: {
      ...actual.api,
      chats: vi.fn(),
      configuration: vi.fn(),
      branchRun: vi.fn(),
      createChat: vi.fn(),
      generate: vi.fn(),
      health: vi.fn(),
      messages: vi.fn(),
      models: vi.fn(),
      run: vi.fn(),
      runEvents: vi.fn(),
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

const persistedAssistant: Message = {
  id: "assistant-persisted",
  chatId: chat.id,
  role: "assistant",
  content: "Persisted answer",
  createdAt: "2026-01-01T00:00:01Z",
  status: "complete",
  runId: "run-persisted",
};

const persistedRun: RunDetails = {
  id: "run-persisted",
  chatId: chat.id,
  messageId: persistedAssistant.id,
  modelId: model.id,
  status: "complete",
  createdAt: "2026-01-01T00:00:01Z",
  branchableThroughTokenIndex: 0,
  tokens: [{ index: 0, tokenId: 7, piece: "Answer", displayText: "Answer", reasoningSegment: "answer" }],
  metrics: { generatedTokens: 1, peakVramBytes: 2048, timing: { totalMs: 25 } },
};

describe("runtime lifecycle synchronization", () => {
  let emitStreamEvent: ((event: RunStreamEvent) => void) | undefined;

  beforeEach(() => {
    window.localStorage.clear();
    emitStreamEvent = undefined;
    mockedApi.health.mockReset()
      .mockResolvedValueOnce({ status: "ok" })
      .mockResolvedValue({ status: "ok", loadedModelId: model.id, loadedDevice: "cpu" });
    mockedApi.models.mockReset().mockResolvedValue([model]);
    mockedApi.chats.mockReset().mockImplementation((archived = false) => Promise.resolve(archived ? [] : [chat]));
    mockedApi.configuration.mockReset().mockResolvedValue({ effective: {}, precedence: [] });
    mockedApi.messages.mockReset().mockResolvedValue([]);
    mockedApi.run.mockReset().mockResolvedValue(persistedRun);
    mockedApi.runEvents.mockReset().mockResolvedValue([]);
    mockedApi.branchRun.mockReset();
    mockedApi.createChat.mockReset().mockResolvedValue({
      ...chat,
      id: "chat-new",
      title: "New chat",
    });
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

  it("restores the latest persisted run and inspector telemetry when a chat opens", async () => {
    window.localStorage.setItem("local-ai-doctor.client-telemetry.v1", JSON.stringify({
      [persistedRun.id]: {
        storedAt: "2026-01-01T00:00:02Z",
        clientTtftMs: 12,
        firstVisibleTextMs: 18,
        tokenTimings: { 0: { clientInterArrivalMs: 7 } },
      },
    }));
    mockedApi.messages.mockResolvedValue([persistedAssistant]);
    const { result } = renderHook(() => useWorkbench());

    await waitFor(() => expect(result.current.selectedRun?.id).toBe(persistedRun.id));

    expect(mockedApi.run.mock.calls).toContainEqual([persistedRun.id]);
    expect(result.current.selectedRun?.tokens).toHaveLength(1);
    expect(result.current.selectedRun?.metrics).toMatchObject({ peakVramBytes: 2048, timing: { totalMs: 25, clientTtftMs: 12, firstVisibleTextMs: 18 } });
    expect(result.current.selectedRun?.tokens[0]?.timing?.clientInterArrivalMs).toBe(7);
  });

  it("does not let a delayed history restore overwrite a newly submitted run", async () => {
    let resolveRun: ((run: RunDetails) => void) | undefined;
    mockedApi.messages.mockResolvedValue([persistedAssistant]);
    mockedApi.run.mockImplementationOnce(() => new Promise((resolve) => {
      resolveRun = resolve;
    }));
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(resolveRun).toBeDefined());

    await act(async () => {
      await result.current.submit("Start a newer run");
    });
    act(() => resolveRun?.(persistedRun));

    await waitFor(() => expect(result.current.selectedRun?.id).toBe("run-fixture"));
    expect(result.current.runningRunId).toBe("run-fixture");
  });

  it("opens a token branch as a new chat and subscribes to its run", async () => {
    const branchedUser: Message = {
      id: "user-branch",
      chatId: "chat-branch",
      role: "user",
      content: "Persisted prompt",
      createdAt: "2026-01-01T00:00:00Z",
      status: "complete",
    };
    mockedApi.messages
      .mockResolvedValueOnce([persistedAssistant])
      .mockResolvedValueOnce([branchedUser, {
        ...persistedAssistant,
        id: "assistant-branch",
        chatId: "chat-branch",
        content: "",
        status: "streaming",
        runId: "run-branch",
        parentMessageId: branchedUser.id,
      }]);
    mockedApi.branchRun.mockResolvedValue({
      chatId: "chat-branch",
      runId: "run-branch",
      messageId: "assistant-branch",
      modelId: model.id,
      parentRunId: persistedRun.id,
      websocketUrl: "/ws/v1/runs/run-branch",
    });
    mockedApi.chats.mockImplementation((archived = false) => Promise.resolve(archived ? [] : [
      { ...chat, id: "chat-branch", title: "Branched chat" },
      chat,
    ]));
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.selectedRun?.id).toBe(persistedRun.id));

    await act(async () => {
      await result.current.branchFromAlternative(0, "raw", {
        tokenId: 9,
        piece: "Different",
        probability: 0.25,
        logProbability: -1.4,
        logit: 2,
        rank: 2,
      });
    });

    expect(mockedApi.branchRun.mock.calls).toContainEqual([persistedRun.id, { tokenIndex: 0, distribution: "raw", rank: 2, tokenId: 9 }]);
    expect(result.current.activeChatId).toBe("chat-branch");
    expect(result.current.selectedRun?.id).toBe("run-branch");
    expect(mockedSubscribeToRun).toHaveBeenCalledWith("run-branch", expect.any(Function), expect.any(Function), "/ws/v1/runs/run-branch");
    await waitFor(() => expect(result.current.messages.find((item) => item.id === "assistant-branch")?.parentMessageId).toBe(branchedUser.id));
  });

  it("loads persisted raw events only when explicitly requested", async () => {
    mockedApi.messages.mockResolvedValue([persistedAssistant]);
    mockedApi.runEvents.mockResolvedValue([{ version: 1, sequence: 1, type: "completed" }]);
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.selectedRun?.id).toBe(persistedRun.id));
    expect(mockedApi.runEvents.mock.calls).toHaveLength(0);

    await act(async () => {
      await result.current.loadSelectedRunEvents();
    });

    expect(mockedApi.runEvents.mock.calls).toEqual([[persistedRun.id]]);
    expect(result.current.selectedRun?.rawEvents).toEqual([{ version: 1, sequence: 1, type: "completed" }]);

    await act(async () => {
      await result.current.loadSelectedRunEvents();
    });
    expect(mockedApi.runEvents.mock.calls).toHaveLength(1);
  });

  it("keeps a successfully created branch running when its history refresh fails", async () => {
    mockedApi.messages
      .mockResolvedValueOnce([persistedAssistant])
      .mockRejectedValueOnce(new Error("temporary history failure"));
    mockedApi.branchRun.mockResolvedValue({
      chatId: "chat-branch",
      runId: "run-branch",
      messageId: "assistant-branch",
      modelId: model.id,
      websocketUrl: "/ws/v1/runs/run-branch",
    });
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.selectedRun?.id).toBe(persistedRun.id));

    await act(async () => {
      await result.current.branchFromAlternative(0, "sampling", {
        tokenId: 8,
        piece: "Alternative",
        probability: 0.2,
        logProbability: -1.6,
        logit: 1.5,
        rank: 3,
      });
    });

    expect(result.current.activeChatId).toBe("chat-branch");
    expect(result.current.runningRunId).toBe("run-branch");
    expect(mockedSubscribeToRun.mock.calls.some(([runId]) => runId === "run-branch")).toBe(true);
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
