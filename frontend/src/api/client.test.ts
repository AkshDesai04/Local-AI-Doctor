import { describe, expect, it, vi } from "vitest";
import { saveSessionAuthToken } from "./auth";
import { defaultGenerationSettings } from "../hooks/useWorkbench";
import { api, ApiError, subscribeToRun } from "./client";
import type { GenerateRequest } from "./types";

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("API boundary normalization", () => {
  it("normalizes backend model descriptors and capability aliases", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      models: [{
        id: "model-1",
        display_name: "Local model",
        architectures: ["FixtureForCausalLM"],
        task: "text_generation",
        fingerprint: { value: "abcdef", total_weight_bytes: 3_000_000_000 },
        parameter_count: 1_500_000_000,
        effective_context_limit: 4096,
        context_values: [{ source: "config.max_position_embeddings", value: 4096 }],
        capabilities: {
          entries: {
            text_generation: { state: "full" },
            reasoning_channel: { state: "partial", reason: "Only emitted tags are observable." },
            deterministic_seeding: { state: "full" },
          },
        },
      }],
    }));
    vi.stubGlobal("fetch", fetchMock);

    const [model] = await api.models();
    expect(model).toMatchObject({
      id: "model-1",
      name: "Local model",
      architecture: "FixtureForCausalLM",
      task: "text_generation",
      parameterCount: 1_500_000_000,
      weightBytes: 3_000_000_000,
      effectiveContextLimit: 4096,
    });
    expect(model?.capabilities.reasoning_segments).toEqual({
      state: "partial",
      reason: "Only emitted tags are observable.",
      limitations: undefined,
    });
    expect(model?.capabilities.deterministic_seed?.state).toBe("full");
  });

  it("serializes generation settings to the strict snake_case request schema", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      run: { id: "run-1" },
      assistant_message: { id: "message-1" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const request: GenerateRequest = {
      chatId: "chat-1",
      modelId: "model-1",
      content: "Hello",
      attachmentIds: [],
      parentMessageId: "parent-1",
      settings: {
        reasoning: false,
        device: "cpu",
        dtype: "float32",
        instrumentation: "token",
        seed: "18446744073709551615",
        maxOutputTokens: 32,
        temperature: 0,
        topK: 10,
        topP: 0.9,
        minP: 0.05,
        repetitionPenalty: 1.1,
        frequencyPenalty: 0,
        presencePenalty: 0,
        alternatives: 6,
        stopSequences: ["END"],
        deterministic: true,
      },
    };

    await expect(api.generate(request)).resolves.toEqual({ runId: "run-1", messageId: "message-1" });
    const init = fetchMock.mock.calls[0]?.[1];
    if (typeof init?.body !== "string") throw new Error("Expected a JSON request body.");
    const body = JSON.parse(init.body) as Record<string, unknown>;
    expect(body).toMatchObject({
      chat_id: "chat-1",
      model_id: "model-1",
      prompt: "Hello",
      parent_message_id: "parent-1",
      device: "cpu",
      dtype: "float32",
      instrumentation: "token",
      reasoning: false,
      seed: "18446744073709551615",
      deterministic_reference_mode: true,
    });
    expect(body.sampling).toMatchObject({
      max_output_tokens: 32,
      top_k: 10,
      top_p: 0.9,
      alternatives: 6,
      stop_sequences: ["END"],
    });
  });

  it("normalizes and sends the per-chat system prompt", async () => {
    const stored = { id: "chat-1", title: "Chat", created_at: "a", updated_at: "b", pinned: 0, archived: 0 };
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse([{ ...stored, system_prompt: "Be brief." }, stored]))
      .mockResolvedValueOnce(jsonResponse({ ...stored, system_prompt: "Created." }))
      .mockResolvedValueOnce(jsonResponse(stored))
      .mockResolvedValueOnce(jsonResponse({ ...stored, system_prompt: null }));
    vi.stubGlobal("fetch", fetchMock);

    const chats = await api.chats();
    expect(chats.map((chat) => chat.systemPrompt)).toEqual(["Be brief.", null]);

    expect((await api.createChat("Created.")).systemPrompt).toBe("Created.");
    await api.createChat("   ");
    expect(fetchMock.mock.calls[1]?.[1]?.body).toBe(JSON.stringify({ systemPrompt: "Created." }));
    expect(fetchMock.mock.calls[2]?.[1]?.body).toBe("{}");

    expect((await api.updateChat("chat-1", { systemPrompt: null })).systemPrompt).toBeNull();
    expect(fetchMock.mock.calls[3]?.[1]?.body).toBe(JSON.stringify({ systemPrompt: null }));
  });

  it("replays an assistant run without resubmitting its user prompt", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      runId: "run-replay",
      messageId: "assistant-replay",
      parentRunId: "run-source",
      websocketUrl: "/ws/v1/runs/run-replay",
      run: {
        id: "run-replay",
        model_id: "model-1",
        status: "queued",
        created_at: "2026-09-12T00:00:00Z",
      },
      assistant_message: { id: "assistant-replay", parent_id: "user-source" },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.replayRun("run-source");

    expect(result).toMatchObject({
      runId: "run-replay",
      messageId: "assistant-replay",
      modelId: "model-1",
      parentRunId: "run-source",
      websocketUrl: "/ws/v1/runs/run-replay",
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/runs/run-source/replay", expect.objectContaining({ method: "POST" }));
  });

  it("branches a run from an explicitly selected alternative token", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      chatId: "chat-branch",
      runId: "run-branch",
      messageId: "assistant-branch",
      sourceRunId: "run-source",
      websocketUrl: "/ws/v1/runs/run-branch",
      run: {
        id: "run-branch",
        chat_id: "chat-branch",
        model_id: "model-1",
        status: "queued",
        created_at: "2026-09-12T00:00:00Z",
      },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.branchRun("run-source", {
      tokenIndex: 12,
      distribution: "sampling",
      rank: 3,
      tokenId: 456,
    });

    expect(result).toMatchObject({
      chatId: "chat-branch",
      runId: "run-branch",
      messageId: "assistant-branch",
      sourceRunId: "run-source",
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/runs/run-source/branch", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ token_index: 12, distribution: "sampling", rank: 3, token_id: 456 }),
    }));
  });

  it("adds the tab-scoped bearer token to REST requests", async () => {
    saveSessionAuthToken("local-secret");
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ status: "ok" }));
    vi.stubGlobal("fetch", fetchMock);

    await api.health();

    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(headers.get("Authorization")).toBe("Bearer local-secret");
  });

  it("preserves explicit prompt-primed reasoning metadata on messages", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse([{
      id: "assistant-1",
      chat_id: "chat-1",
      role: "assistant",
      content: "emitted reasoning without an opening tag",
      status: "cancelled",
      created_at: "2026-09-12T00:00:00Z",
      metadata: { reasoning_primed: true },
    }]));
    vi.stubGlobal("fetch", fetchMock);

    const [message] = await api.messages("chat-1");

    expect(message?.reasoningPrimed).toBe(true);
  });

  it("normalizes persisted message attachments from the snake-case API payload", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse([{
      id: "user-with-attachment",
      chat_id: "chat-1",
      role: "user",
      content: "Inspect this image",
      status: "complete",
      created_at: "2026-09-12T00:00:00Z",
      attachments: [{
        id: "attachment-1",
        original_name: "scan.png",
        media_type: "image/png",
        size_bytes: 2048,
        preprocessing: { path: "native_model_processor" },
      }],
    }]));
    vi.stubGlobal("fetch", fetchMock);

    const [message] = await api.messages("chat-1");

    expect(message?.attachments).toEqual([expect.objectContaining({
      id: "attachment-1",
      name: "scan.png",
      mimeType: "image/png",
      sizeBytes: 2048,
      kind: "image",
      status: "ready",
      nativeProcessing: true,
    })]);
  });

  it("reconstructs persisted run metrics from backend summaries and phases", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      id: "run-1",
      model_id: "model-1",
      status: "complete",
      created_at: "2026-09-12T00:00:00Z",
      received_at: "2026-09-12T00:00:00.000Z",
      queue_entered_at: "2026-09-12T00:00:00.100Z",
      queue_exited_at: "2026-09-12T00:00:00.350Z",
      first_token_at: "2026-09-12T00:00:00.900Z",
      prompt_token_count: 14,
      generated_token_count: 1,
      branchable_through_token_index: 0,
      model_fingerprint: "abc123",
      effective_config: { instrumentation: "full", inference: { reserved_output_tokens: 384 } },
      settings: { sampling: { temperature: 0.7 } },
      reproducibility: {
        effective_seed: "42",
        backend: "transformers-reference-loop",
        deterministic_kernels: { torch_use_deterministic_algorithms: true },
        device: {
          device_identifier: "cuda:0",
          effective_dtype: "bfloat16",
          reason: "Selected the first compatible accelerator.",
        },
      },
      phases: [
        { phase: "tokenization", duration_ms: 3, details: { context_limit: 4096, reserved_output_tokens: 320 } },
        { phase: "prefill", duration_ms: 8, details: { prompt_tokens_per_second: 120 } },
        { phase: "generation", duration_ms: 25, details: { engine_ttft_ms: 12 } },
      ],
      summary: {
        finish_reason: "eos",
        conditional_response_perplexity: 1.5,
        end_to_end_tokens_per_second: 40,
        memory: { process_rss_bytes: 1024, cuda_peak_allocated_bytes: 2048 },
      },
      tokens: [{
        token_index: 0,
        token_id: 7,
        piece: "A",
        running_perplexity: 1.5,
        reasoning_slices: [{ start: 0, end: 1, classification: "answer", delimiter: false }],
        alternatives: [{ distribution: "raw", rank: 1, token_id: 7, piece: "A", survived_filter: 1 }],
        attention_attribution: {
          method: "mean_causal_self_attention",
          aggregation: "arithmetic_mean_over_layers_and_heads",
          semantics: "attention_weights_not_causal_contributions",
          context_tokens: [
            { context_index: 0, token_id: 1, piece: "Question", display_text: "Question", source_kind: "prompt" },
            { context_index: 1, token_id: 9, piece: "<|image_pad|>", display_text: "<|image_pad|>", source_kind: "prompt", media: { kind: "image", index: 0 } },
            { context_index: 2, token_id: 9, piece: "<|image_pad|>", display_text: "<|image_pad|>", source_kind: "prompt", media: { kind: "audio", index: 0 } },
          ],
          source_tokens: [{ context_index: 0, token_id: 1, piece: "Question", display_text: "Question", source_kind: "prompt", weight: 0.75 }],
          captured_layers: [0, 1],
          captured_heads: 4,
          normalized: true,
          total_source_count: 1,
          retained_source_count: 1,
          retained_weight: 0.75,
          omitted_weight: 0.25,
        },
      }],
    }));
    vi.stubGlobal("fetch", fetchMock);

    const run = await api.run("run-1");
    expect(run.metrics).toMatchObject({
      finishReason: "eos",
      responsePerplexity: 1.5,
      promptTokens: 14,
      generatedTokens: 1,
      peakRamBytes: 1024,
      peakVramBytes: 2048,
      timing: {
        queueMs: 250,
        serverTtftMs: 900,
        tokenizationMs: 3,
        prefillMs: 8,
        engineTtftMs: 12,
        totalMs: 25,
        endToEndTokensPerSecond: 40,
      },
      context: { effectiveLimit: 4096, reservedOutputTokens: 320 },
    });
    expect(run.reproducibility).toMatchObject({
      device: "cuda:0",
      dtype: "bfloat16",
      deviceReason: "Selected the first compatible accelerator.",
      deterministicKernels: true,
    });
    expect(run.tokens[0]?.rawAlternatives?.[0]?.survivedFiltering).toBe(true);
    expect(run.tokens[0]?.reasoningSlices).toEqual([{ start: 0, end: 1, classification: "answer", delimiter: false }]);
    expect(run.tokens[0]?.attentionAttribution).toEqual(expect.objectContaining({
      method: "mean_causal_self_attention",
      capturedLayers: [0, 1],
      capturedHeads: 4,
      totalSourceCount: 1,
      retainedSourceCount: 1,
      retainedWeight: 0.75,
      omittedWeight: 0.25,
      contextTokens: [
        expect.objectContaining({ contextIndex: 0, tokenId: 1, sourceKind: "prompt" }),
        expect.objectContaining({ contextIndex: 1, sourceKind: "prompt", media: { kind: "image", index: 0 } }),
        expect.objectContaining({ contextIndex: 2, sourceKind: "prompt" }),
      ],
      sourceTokens: [expect.objectContaining({ contextIndex: 0, tokenId: 1, sourceKind: "prompt", weight: 0.75 })],
    }));
    // An unknown media kind is dropped rather than guessed.
    expect(run.tokens[0]?.attentionAttribution?.contextTokens?.[2]?.media).toBeUndefined();
    expect(run.branchableThroughTokenIndex).toBe(0);
    expect(run.effectiveSettings).toMatchObject({ instrumentation: "full", sampling: { temperature: 0.7 } });
  });

  it("loads persisted raw events separately from the core run snapshot", async () => {
    const persistedEnvelope = {
      version: 1,
      run_id: "run-1",
      sequence: 4,
      type: "completed",
      monotonic_ns: 98_765_432,
      created_at: "2026-09-12T00:00:04Z",
      payload: {
        finish_reason: "eos",
        metadata: { sampler: "multinomial", stop_token_id: 2 },
      },
    };
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      events: [persistedEnvelope],
    }));
    vi.stubGlobal("fetch", fetchMock);

    const events = await api.runEvents("run-1");

    expect(events).toEqual([expect.objectContaining({ sequence: 4, type: "completed" })]);
    const completed = events[0];
    expect(completed?.type === "completed" ? completed.run?.metrics?.finishReason : undefined).toBe("eos");
    expect(completed?.raw).toEqual(persistedEnvelope);
    expect(JSON.parse(JSON.stringify(completed?.raw))).toEqual(persistedEnvelope);
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/runs/run-1/events", expect.any(Object));
  });

  it("retains compatibility with a boolean deterministic-kernel snapshot", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      id: "run-boolean-determinism",
      model_id: "model-1",
      status: "complete",
      created_at: "2026-09-12T00:00:00Z",
      reproducibility: {
        effective_seed: "9",
        deterministic_kernels: false,
      },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const run = await api.run("run-boolean-determinism");

    expect(run.reproducibility?.deterministicKernels).toBe(false);
  });

  it("normalizes native attachment processing from every backend representation", async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({
        id: "attachment-camel",
        original_name: "camel.png",
        media_type: "image/png",
        size_bytes: 1,
        nativeProcessing: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        id: "attachment-snake",
        original_name: "snake.png",
        media_type: "image/png",
        size_bytes: 1,
        native_processing: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        id: "attachment-path",
        original_name: "path.png",
        media_type: "image/png",
        size_bytes: 1,
        preprocessing: { path: "native_model_processor" },
      }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["x"], "fixture.png", { type: "image/png" });

    const attachments = await Promise.all([
      api.upload(file, "model-1"),
      api.upload(file, "model-1"),
      api.upload(file, "model-1"),
    ]);

    expect(attachments.map((attachment) => attachment.nativeProcessing)).toEqual([true, true, true]);
  });

  it("fetches attachment content through the authenticated API boundary", async () => {
    saveSessionAuthToken("attachment-secret");
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response("image-bytes", {
      status: 200,
      headers: { "Content-Type": "image/png" },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const content = await api.attachmentContent("attachment/one");

    // Response.blob() can return a Blob from the fetch implementation's realm
    // rather than jsdom's realm, so instanceof is not portable across runners.
    expect(Object.prototype.toString.call(content)).toBe("[object Blob]");
    expect(content.size).toBe(11);
    expect(content.type).toBe("image/png");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/attachments/attachment%2Fone/content");
    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(headers.get("Accept")).toBe("*/*");
    expect(headers.get("Authorization")).toBe("Bearer attachment-secret");
  });

  it("uses the persisted resume cursor when a stream requests resynchronization", async () => {
    vi.useFakeTimers();
    saveSessionAuthToken("local-secret");

    class TestSocket {
      static instances: TestSocket[] = [];
      readonly listeners = new Map<string, EventListener[]>();
      readonly url: string;
      readonly protocols: string | string[] | undefined;
      readonly close = vi.fn();

      constructor(url: string, protocols?: string | string[]) {
        this.url = url;
        this.protocols = protocols;
        TestSocket.instances.push(this);
      }

      addEventListener(type: string, listener: EventListener): void {
        this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
      }

      emit(type: string, event: Event): void {
        this.listeners.get(type)?.forEach((listener) => listener(event));
      }
    }

    vi.stubGlobal("WebSocket", TestSocket);
    const events: unknown[] = [];
    const unsubscribe = subscribeToRun("run/one", (event) => events.push(event), vi.fn());
    const first = TestSocket.instances[0];
    expect(first).toBeDefined();
    const initialUrl = new URL(first?.url ?? "ws://invalid");
    expect(initialUrl.pathname).toBe("/ws/v1/runs/run%2Fone");
    expect(initialUrl.searchParams.get("after")).toBe("0");
    expect(initialUrl.searchParams.has("access_token")).toBe(false);
    expect(first?.protocols).toEqual(["lad.events.v1", "lad.auth.bG9jYWwtc2VjcmV0"]);

    first?.emit("message", new MessageEvent("message", {
      data: JSON.stringify({ type: "resync_required", sequence: 999, payload: { resume_after: 12 } }),
    }));
    expect(first?.close).toHaveBeenCalledWith(1012, "resync required");
    first?.emit("close", new CloseEvent("close"));
    await vi.advanceTimersByTimeAsync(250);

    const resumedUrl = new URL(TestSocket.instances[1]?.url ?? "ws://invalid");
    expect(resumedUrl.searchParams.get("after")).toBe("12");
    expect(events).toHaveLength(1);
    unsubscribe();
  });
});

describe("resident models and memory", () => {
  const gib = 1024 ** 3;

  function sentJson(fetchMock: { mock: { calls: Array<Parameters<typeof fetch>> } }): unknown {
    const body = fetchMock.mock.calls[0]?.[1]?.body;
    if (typeof body !== "string") throw new Error("Expected a JSON request body.");
    return JSON.parse(body);
  }

  function errorResponse(status: number, value: unknown): Response {
    return new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
  }

  it("normalizes every resident listed by health and keeps the most recent as loadedModelId", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      status: "ok",
      loaded_model: { model_id: "model-b", device: "cuda:0" },
      loaded_models: [
        { model_key: "key-a", model_id: "model-a", device: "cuda:0", dtype: "bfloat16", quantization: "none", placement: "gpu", strict_vram: true },
        { model_key: "key-b", model_id: "model-b", device: "cuda:0", dtype: "float16", quantization: "none", placement: "offload", strict_vram: false },
      ],
    })));

    const health = await api.health();
    expect(health.loadedModelId).toBe("model-b");
    expect(health.loadedModels).toEqual([
      expect.objectContaining({ modelKey: "key-a", modelId: "model-a", dtype: "bfloat16", placement: "gpu", strictVram: true, gpuBytes: null }),
      expect.objectContaining({ modelKey: "key-b", modelId: "model-b", dtype: "float16", placement: "offload", strictVram: false }),
    ]);
  });

  it("leaves loadedModels undefined for a backend that reports only one loaded model", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ status: "ok", loaded_model: { model_id: "model-a", device: "cpu" } })));
    const health = await api.health();
    expect(health.loadedModels).toBeUndefined();
    expect(health.loadedModelId).toBe("model-a");
  });

  it("normalizes resident status with the memory ledger", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      models: [{
        model_key: "key-a",
        model_id: "model-a",
        display_name: "Model A",
        device: "cuda:0",
        dtype: "bfloat16",
        quantization: "none",
        strict_vram: true,
        placement: "gpu",
        gpu_bytes: 2 * gib,
        cpu_bytes: 64 * 1024 ** 2,
        kv_reserve_bytes: 256 * 1024 ** 2,
        load_seconds: 4.5,
        last_used_at: "2026-10-01T10:00:00Z",
        in_use: true,
      }],
      memory: {
        device: "cuda:0",
        total_bytes: 8 * gib,
        free_bytes: 5 * gib,
        torch_allocated_bytes: 2 * gib,
        torch_reserved_bytes: 2.5 * gib,
        cap_bytes: null,
        process_rss_bytes: gib,
        system_available_bytes: 12 * gib,
        safety_margin_bytes: 512 * 1024 ** 2,
        ledger_age_seconds: 1.5,
        stale: false,
      },
      max_loaded_models: 4,
    }));
    vi.stubGlobal("fetch", fetchMock);

    const status = await api.residentStatus();
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/models/resident");
    expect(status?.maxLoadedModels).toBe(4);
    expect(status?.models).toEqual([{
      modelKey: "key-a",
      modelId: "model-a",
      displayName: "Model A",
      device: "cuda:0",
      dtype: "bfloat16",
      quantization: "none",
      strictVram: true,
      placement: "gpu",
      gpuBytes: 2 * gib,
      cpuBytes: 64 * 1024 ** 2,
      kvReserveBytes: 256 * 1024 ** 2,
      loadSeconds: 4.5,
      lastUsedAt: "2026-10-01T10:00:00Z",
      inUse: true,
    }]);
    expect(status?.memory).toEqual({
      device: "cuda:0",
      totalBytes: 8 * gib,
      freeBytes: 5 * gib,
      torchAllocatedBytes: 2 * gib,
      torchReservedBytes: 2.5 * gib,
      capBytes: null,
      processRssBytes: gib,
      systemAvailableBytes: 12 * gib,
      safetyMarginBytes: 512 * 1024 ** 2,
      ledgerAgeSeconds: 1.5,
      stale: false,
    });
  });

  it("returns null resident status when the backend has no resident endpoint", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(errorResponse(404, { error: { code: "not_found", message: "not found" } })));
    await expect(api.residentStatus()).resolves.toBeNull();
  });

  it("sends load options and surfaces the evicted residents", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      id: "model-c",
      display_name: "Model C",
      task: "text_generation",
      lifecycle: "loaded",
      loaded_device: "cuda:0",
      model_key: "key-c",
      placement: "gpu",
      quantization: "none",
      strict_vram: true,
      evicted_model_keys: ["key-a"],
      load: { ledger: { device: "cuda:0", total_bytes: 8 * gib } },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.loadModel("model-c", { device: "cuda", dtype: "bfloat16", strictVram: true });
    expect(result).toMatchObject({ modelKey: "key-c", placement: "gpu", strictVram: true, evictedModelKeys: ["key-a"], model: { id: "model-c", lifecycle: "loaded" } });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/models/model-c/load");
    expect(sentJson(fetchMock)).toEqual({ device: "cuda", dtype: "bfloat16", strictVram: true });
  });

  it("unloads one resident, one model, or every resident", async () => {
    const fetchMock = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ unloaded_model_keys: ["key/a"], freed_bytes: gib, leaked_bytes: 0 }))
      .mockResolvedValueOnce(jsonResponse({ id: "model-b", lifecycle: "unloaded", unload: { unloaded_model_keys: ["key-b"], freed_bytes: 2 * gib } }))
      .mockResolvedValueOnce(jsonResponse({ unloaded_model_keys: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.unloadResident("key/a")).resolves.toEqual({ unloadedModelKeys: ["key/a"], freedBytes: gib, leakedBytes: 0 });
    await expect(api.unloadModel("model-b")).resolves.toEqual({ unloadedModelKeys: ["key-b"], freedBytes: 2 * gib, leakedBytes: null });
    await api.unloadAll();
    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method])).toEqual([
      ["/api/v1/models/resident/key%2Fa/unload", "POST"],
      ["/api/v1/models/model-b/unload", "POST"],
      ["/api/v1/models/unload", "POST"],
    ]);
  });

  it("turns an out-of-memory envelope into a readable message with the byte counts and hint", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(errorResponse(507, {
      error: {
        code: "out_of_memory",
        message: "insufficient memory",
        retryable: false,
        hint: "Quantize the model, unload another model, or turn off Strict VRAM to allow system-RAM offload.",
        details: {
          required_bytes: 7.3 * gib,
          available_bytes: 5.9 * gib,
          resident_model_keys: ["key-a"],
          pinned_model_keys: ["key-a"],
          evicted_model_keys: [],
        },
      },
    })));

    const failure = await api.loadModel("model-c", { device: "cuda", dtype: "auto", strictVram: true }).catch((cause: unknown) => cause);
    expect(failure).toBeInstanceOf(ApiError);
    const error = failure as ApiError;
    expect(error.status).toBe(507);
    expect(error.code).toBe("out_of_memory");
    expect(error.message).toBe("Not enough GPU memory: needs 7.3 GiB, 5.9 GiB available. Quantize the model, unload another model, or turn off Strict VRAM to allow system-RAM offload.");
    expect(error.details).toMatchObject({ pinned_model_keys: ["key-a"] });
  });

  it("sends Strict VRAM with generation settings and reads evictions from model_loaded events", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ run: { id: "run-1" } }));
    vi.stubGlobal("fetch", fetchMock);
    await api.generate({
      chatId: "chat-1",
      modelId: "model-1",
      content: "Hi",
      attachmentIds: [],
      settings: { ...defaultGenerationSettings, device: "cuda", dtype: "bfloat16", strictVram: false },
    });
    expect(sentJson(fetchMock)).toMatchObject({ device: "cuda", dtype: "bfloat16", strict_vram: false });

    fetchMock.mockResolvedValue(jsonResponse({ events: [
      { version: 1, sequence: 2, type: "model_loaded", payload: { model_key: "key-c", evicted_model_keys: ["key-a"] } },
      { version: 1, sequence: 9, type: "error", payload: { code: "out_of_memory", message: "insufficient memory", details: { required_bytes: 2 * gib, available_bytes: gib, memory_kind: "ram" } } },
    ] }));
    const [loaded, failed] = await api.runEvents("run-1");
    expect(loaded).toMatchObject({ type: "stage.changed", evictedModelKeys: ["key-a"] });
    expect(failed?.type === "error" ? failed.message : "").toMatch(/^Not enough system memory: needs 2 GiB, 1 GiB available\./);
  });

  it("requests token influence in the browser shape and normalizes grouped sources", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      run_id: "run-1",
      token_index: 3,
      method: "gradient_x_input",
      cached: true,
      target: { token_id: 9, piece: "Ġyes", display_text: " yes", alternative_token_id: 12, alternative_piece: "Ġno" },
      context_token_count: 40,
      prompt_token_count: 37,
      sources: [
        { source_kind: "generated", context_index: 38, span: null, token_count: 1, token_id: 5, piece: "Ġa", display_text: " a", generated_token_index: 1, media_index: null, is_special: false, weight: 0.25 },
        { source_kind: "image", context_index: 4, span: [4, 20], token_count: 16, token_id: null, piece: null, display_text: "image 1", generated_token_index: null, media_index: 0, is_special: false, weight: 0.5 },
        { source_kind: "unknown", context_index: 1, weight: 0.1 },
      ],
      retained_weight: 0.75,
      omitted_weight: 0.25,
      layers: null,
      captured_layers: [],
      heads_per_layer: null,
      objective: "logit_difference",
      objective_value: 2.5,
      semantics: "local first-order sensitivity",
      normalization: "sum_to_one",
      model_fingerprint: "abc",
      duration_ms: 31.5,
    }));
    vi.stubGlobal("fetch", fetchMock);

    const influence = await api.tokenInfluence("run 1", 3, { method: "gradient_x_input", alternativeTokenId: 12, sourceLimit: 64 });

    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(url).toBe("/api/v1/runs/run%201/tokens/3/influence");
    expect(JSON.parse(init?.body as string)).toEqual({ method: "gradient_x_input", alternativeTokenId: 12, sourceLimit: 64 });
    expect(influence).toMatchObject({ method: "gradient_x_input", cached: true, objective: "logit_difference", objectiveValue: 2.5, retainedWeight: 0.75, omittedWeight: 0.25, durationMs: 31.5 });
    expect(influence.target).toEqual({ tokenId: 9, piece: "Ġyes", displayText: " yes", alternativeTokenId: 12, alternativePiece: "Ġno" });
    // Unknown kinds are dropped and sources come back in context order.
    expect(influence.sources.map((item) => [item.sourceKind, item.contextIndex, item.span, item.tokenCount])).toEqual([["image", 4, [4, 20], 16], ["generated", 38, null, 1]]);
    expect(influence.sources[1]?.generatedTokenIndex).toBe(1);
  });

  it("sends layers only for attention", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ method: "attention", sources: [], layers: [{ layer: 2, sources: [], retained_weight: 0, omitted_weight: 1 }] }));
    vi.stubGlobal("fetch", fetchMock);

    const influence = await api.tokenInfluence("run-1", 0, { method: "attention", layers: "all" });

    expect(JSON.parse(fetchMock.mock.calls[0]?.[1]?.body as string)).toEqual({ method: "attention", layers: "all" });
    expect(influence.layers).toEqual([{ layer: 2, sources: [], retainedWeight: 0, omittedWeight: 1 }]);
  });
});
