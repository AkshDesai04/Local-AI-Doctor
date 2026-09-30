import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, subscribeToRun } from "../api/client";
import type * as ApiClientModule from "../api/client";
import type { LoadOptions, ModelSummary, ResidentModel, RunStreamEvent } from "../api/types";
import { LOAD_OPTIONS_STORAGE_KEY } from "../domain/residency";
import { useWorkbench } from "./useWorkbench";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof ApiClientModule>();
  return {
    ...actual,
    api: {
      ...actual.api,
      chats: vi.fn(),
      configuration: vi.fn(),
      createChat: vi.fn(),
      generate: vi.fn(),
      health: vi.fn(),
      loadModel: vi.fn(),
      messages: vi.fn(),
      models: vi.fn(),
      residentStatus: vi.fn(),
      run: vi.fn(),
      unloadAll: vi.fn(),
      unloadModel: vi.fn(),
      unloadResident: vi.fn(),
    },
    subscribeToRun: vi.fn(),
  };
});

const mockedApi = vi.mocked(api);
const gib = 1024 ** 3;

function fixtureModel(id: string, name: string): ModelSummary {
  return {
    id,
    name,
    architecture: "FixtureForCausalLM",
    task: "text_generation",
    fingerprint: `${id}-fingerprint`,
    lifecycle: "unloaded",
    capabilities: { text_generation: { state: "full" }, cuda: { state: "full" } },
    effectiveContextLimit: 4096,
  };
}

const first = fixtureModel("model-a", "Model A");
const second = fixtureModel("model-b", "Model B");

function resident(modelId: string, options: Partial<LoadOptions>, key = `key-${modelId}`): ResidentModel {
  const onCpu = options.device === "cpu";
  return {
    modelKey: key,
    modelId,
    displayName: null,
    device: onCpu ? "cpu" : "cuda:0",
    dtype: options.dtype === "auto" || !options.dtype ? "bfloat16" : options.dtype,
    quantization: "none",
    strictVram: options.strictVram ?? true,
    placement: onCpu ? "cpu" : "gpu",
    gpuBytes: onCpu ? 0 : 2 * gib,
    cpuBytes: 0,
    kvReserveBytes: 0,
    loadSeconds: 1,
    lastUsedAt: null,
    inUse: false,
  };
}

describe("resident model set", () => {
  let residents: ResidentModel[];
  let multiModelBackend: boolean;

  beforeEach(() => {
    window.localStorage.clear();
    residents = [];
    multiModelBackend = true;
    mockedApi.health.mockReset().mockImplementation(() => Promise.resolve(multiModelBackend
      ? { status: "ok", loadedModels: residents, loadedModelId: residents.at(-1)?.modelId }
      : { status: "ok", loadedModelId: residents.at(-1)?.modelId, loadedDevice: residents.at(-1)?.device }));
    mockedApi.residentStatus.mockReset().mockImplementation(() => Promise.resolve(multiModelBackend
      ? { models: residents, memory: null, maxLoadedModels: 4 }
      : null));
    mockedApi.models.mockReset().mockResolvedValue([first, second]);
    mockedApi.chats.mockReset().mockResolvedValue([]);
    mockedApi.configuration.mockReset().mockResolvedValue({ effective: {}, precedence: [] });
    mockedApi.messages.mockReset().mockResolvedValue([]);
    mockedApi.run.mockReset().mockRejectedValue(new Error("The durable run is not part of this fixture."));
    mockedApi.createChat.mockReset().mockResolvedValue({ id: "chat-1", title: "New chat", createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z", pinned: false, archived: false });
    mockedApi.loadModel.mockReset().mockImplementation((id, options) => {
      residents = [...residents, resident(id, options)];
      return Promise.resolve({ model: id === first.id ? first : second, modelKey: `key-${id}`, evictedModelKeys: [] });
    });
    mockedApi.unloadResident.mockReset().mockImplementation((key) => {
      residents = residents.filter((item) => item.modelKey !== key);
      return Promise.resolve({ unloadedModelKeys: [key], freedBytes: 2 * gib, leakedBytes: 0 });
    });
    mockedApi.unloadModel.mockReset().mockImplementation((id) => {
      residents = residents.filter((item) => item.modelId !== id);
      return Promise.resolve({ unloadedModelKeys: [], freedBytes: null, leakedBytes: null });
    });
    mockedApi.unloadAll.mockReset().mockImplementation(() => {
      residents = [];
      return Promise.resolve({ unloadedModelKeys: [], freedBytes: 4 * gib, leakedBytes: 0 });
    });
    mockedApi.generate.mockReset().mockResolvedValue({ runId: "run-1", messageId: "assistant-1", userMessageId: "user-1", modelId: first.id });
    vi.mocked(subscribeToRun).mockReset().mockReturnValue(vi.fn());
  });

  async function booted(): Promise<{ current: ReturnType<typeof useWorkbench> }> {
    const { result } = renderHook(() => useWorkbench());
    await waitFor(() => expect(result.current.booting).toBe(false));
    return result;
  }

  it("keeps two models resident at once with their own load options", async () => {
    const result = await booted();
    expect(result.current.models.map((model) => model.lifecycle)).toEqual(["unloaded", "unloaded"]);

    await act(() => result.current.loadModel(first));
    await act(() => result.current.loadModel(second, { device: "cpu", dtype: "float32", strictVram: false }));

    expect(mockedApi.loadModel.mock.calls.map(([id, options]) => [id, options])).toEqual([
      [first.id, { device: "auto", dtype: "auto", strictVram: true }],
      [second.id, { device: "cpu", dtype: "float32", strictVram: false }],
    ]);
    expect(result.current.residents.map((item) => item.modelKey)).toEqual(["key-model-a", "key-model-b"]);
    expect(result.current.models.map((model) => [model.lifecycle, model.loadedDevice])).toEqual([["loaded", "cuda:0"], ["loaded", "cpu"]]);
    expect(result.current.maxLoadedModels).toBe(4);
  });

  it("names the evicted resident in a notice", async () => {
    residents = [resident(first.id, {})];
    mockedApi.loadModel.mockImplementation((id, options) => {
      residents = [resident(id, options)];
      return Promise.resolve({ model: second, modelKey: `key-${id}`, evictedModelKeys: ["key-model-a"] });
    });
    const result = await booted();

    await act(() => result.current.loadModel(second));

    expect(result.current.notice).toBe("Unloaded Model A to make room for Model B.");
    expect(result.current.models.map((model) => model.lifecycle)).toEqual(["unloaded", "loaded"]);
  });

  it("unloads one resident and leaves the other loaded", async () => {
    residents = [resident(first.id, {}), resident(second.id, {})];
    const result = await booted();
    const target = result.current.residents[0];
    if (!target) throw new Error("Expected a resident.");

    await act(() => result.current.unloadResident(target));

    expect(mockedApi.unloadResident.mock.calls).toEqual([["key-model-a"]]);
    expect(result.current.models.map((model) => model.lifecycle)).toEqual(["unloaded", "loaded"]);
    expect(result.current.notice).toBe("Unloaded Model A and freed 2 GiB.");

    await act(() => result.current.unloadAll());
    expect(result.current.residents).toEqual([]);
  });

  it("shows a readable error and marks the model when a load fails", async () => {
    mockedApi.loadModel.mockRejectedValue(new Error("Not enough GPU memory: needs 7.3 GiB, 5.9 GiB available."));
    const result = await booted();

    await act(() => result.current.loadModel(first));

    expect(result.current.error).toBe("Not enough GPU memory: needs 7.3 GiB, 5.9 GiB available.");
    expect(result.current.models[0]?.lifecycle).toBe("error");
  });

  it("persists load options per model and defaults Strict VRAM from configuration", async () => {
    mockedApi.configuration.mockResolvedValue({ effective: { runtime: { strict_vram: false } }, precedence: [] });
    const result = await booted();
    await waitFor(() => expect(result.current.loadOptionsFor(first.id).strictVram).toBe(false));

    act(() => result.current.setLoadOptions(second.id, { device: "cuda", dtype: "bfloat16", strictVram: true }));

    expect(result.current.loadOptionsFor(second.id)).toEqual({ device: "cuda", dtype: "bfloat16", strictVram: true });
    expect(JSON.parse(window.localStorage.getItem(LOAD_OPTIONS_STORAGE_KEY) ?? "{}")).toEqual({
      [second.id]: { device: "cuda", dtype: "bfloat16", strictVram: true },
    });
    const { result: reloaded } = renderHook(() => useWorkbench());
    expect(reloaded.current.loadOptionsFor(second.id)).toEqual({ device: "cuda", dtype: "bfloat16", strictVram: true });
  });

  it("sends a non-resident model's saved load options with a chat message and reuses a resident's placement", async () => {
    window.localStorage.setItem(LOAD_OPTIONS_STORAGE_KEY, JSON.stringify({ [first.id]: { device: "cuda", dtype: "float16", strictVram: false } }));
    const result = await booted();

    await act(() => result.current.submit("Load it for me"));
    expect(mockedApi.generate.mock.calls[0]?.[0].settings).toMatchObject({ device: "cuda", dtype: "float16", strictVram: false });

    residents = [resident(first.id, { device: "cpu", dtype: "float32" })];
    await act(async () => {
      await result.current.synchronizeRuntimeState();
    });
    const events = vi.mocked(subscribeToRun).mock.calls[0]?.[1];
    act(() => events?.({ version: 1, sequence: 1, type: "completed" } satisfies RunStreamEvent));
    await waitFor(() => expect(result.current.runningRunId).toBeNull());

    await act(() => result.current.submit("Reuse the resident copy"));
    expect(mockedApi.generate.mock.calls[1]?.[0].settings).toMatchObject({ device: "cpu", dtype: "float32", strictVram: true });
  });

  it("shows a chat-triggered load as loading and announces its evictions", async () => {
    residents = [resident(first.id, {})];
    mockedApi.generate.mockResolvedValue({ runId: "run-2", messageId: "assistant-2", userMessageId: "user-2", modelId: second.id });
    const result = await booted();
    act(() => result.current.setSelectedModelId(second.id));

    await act(() => result.current.submit("Load the second model"));
    const events = vi.mocked(subscribeToRun).mock.calls[0]?.[1];
    act(() => events?.({ version: 1, sequence: 1, type: "stage.changed", stage: "loading", detail: "model_loading" }));
    expect(result.current.models[1]?.lifecycle).toBe("loading");

    residents = [resident(second.id, {})];
    act(() => events?.({ version: 1, sequence: 2, type: "stage.changed", stage: "running", detail: "running", evictedModelKeys: ["key-model-a"] }));
    expect(result.current.notice).toBe("Unloaded Model A to make room for Model B.");
    await waitFor(() => expect(result.current.models.map((model) => model.lifecycle)).toEqual(["unloaded", "loaded"]));
  });

  it("degrades to health's single loaded model on a backend without residency", async () => {
    multiModelBackend = false;
    residents = [resident(first.id, { device: "cpu" })];
    const result = await booted();

    expect(result.current.models.map((model) => model.lifecycle)).toEqual(["loaded", "unloaded"]);
    expect(result.current.memory).toBeNull();

    await act(() => result.current.loadModel(second));
    expect(mockedApi.loadModel.mock.calls[0]?.[1]).toEqual({ device: "auto", dtype: "auto" });

    await act(() => result.current.submit("Hello"));
    expect(mockedApi.generate.mock.calls[0]?.[0].settings.strictVram).toBeUndefined();
  });
});
