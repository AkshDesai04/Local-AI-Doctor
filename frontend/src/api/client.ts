import type {
  ApiErrorPayload,
  Attachment,
  BranchRunRequest,
  Capability,
  CapabilityKey,
  ChatSummary,
  ConfigurationSnapshot,
  EmbeddingInput,
  EmbeddingRun,
  GenerateRequest,
  GenerateResponse,
  HealthStatus,
  Message,
  ModelInspection,
  ModelSummary,
  ReasoningSlice,
  RunDetails,
  RunStreamEvent,
  TokenAlternative,
  TokenEvent,
} from "./types";
import { AUTH_CHANGED_EVENT, getSessionAuthToken, notifyAuthenticationRequired } from "./auth";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api/v1").replace(/\/$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly requestId?: string;

  constructor(message: string, status: number, payload?: ApiErrorPayload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = payload?.code ?? payload?.error?.code;
    this.requestId = payload?.requestId ?? payload?.error?.request_id;
  }
}

function errorMessage(payload: ApiErrorPayload | undefined, fallback: string): string {
  if (typeof payload?.error?.message === "string") return payload.error.message;
  if (typeof payload?.message === "string") return payload.message;
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg).filter(Boolean).join("; ") || fallback;
  }
  return fallback;
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function asOptionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asOptionalBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (value === 0 || value === 1) return value === 1;
  return undefined;
}

function timestampDeltaMs(later: unknown, earlier: unknown): number | undefined {
  if (typeof later !== "string" || typeof earlier !== "string") return undefined;
  const laterMs = Date.parse(later);
  const earlierMs = Date.parse(earlier);
  const difference = laterMs - earlierMs;
  return Number.isFinite(difference) && difference >= 0 ? difference : undefined;
}

function asNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? String(value) : fallback;
}

function runStatus(value: unknown, fallback: RunDetails["status"] = "failed"): RunDetails["status"] {
  return value === "queued" || value === "loading" || value === "running" || value === "complete" || value === "cancelled" || value === "failed" ? value : fallback;
}

function capabilityState(value: unknown): Capability["state"] {
  return value === "full" || value === "partial" || value === "unavailable_on_backend" ? value : "unsupported";
}

const capabilityAliases: Record<string, CapabilityKey> = {
  text_generation: "text_generation",
  encoder_decoder_generation: "encoder_decoder_generation",
  embeddings: "embeddings",
  vision: "vision",
  audio: "audio",
  video: "video",
  native_file_input: "native_file_input",
  extracted_text_file_input: "extracted_text_input",
  extracted_text_input: "extracted_text_input",
  reasoning_channel: "reasoning_segments",
  reasoning_segments: "reasoning_segments",
  moe_routing: "moe_routing",
  raw_logits: "raw_logits",
  processed_logits: "processed_logits",
  top_k_alternatives: "top_k_alternatives",
  prompt_scoring: "prompt_scoring",
  attention_capture: "attention_capture",
  hidden_state_capture: "hidden_state_capture",
  streaming: "streaming",
  batching: "batching",
  deterministic_seeding: "deterministic_seed",
  deterministic_seed: "deterministic_seed",
  cpu: "cpu",
  cuda: "cuda",
};

function normalizeCapabilities(value: unknown): ModelSummary["capabilities"] {
  const outer = asRecord(value);
  const entries = asRecord(outer.entries ?? outer);
  const capabilities: Partial<Record<CapabilityKey, Capability>> = {};
  for (const [externalKey, externalValue] of Object.entries(entries)) {
    const key = capabilityAliases[externalKey];
    if (!key) continue;
    const item = asRecord(externalValue);
    capabilities[key] = {
      state: capabilityState(item.state),
      reason: typeof item.reason === "string" ? item.reason : undefined,
      limitations: Array.isArray(item.limitations) ? item.limitations.filter((entry): entry is string => typeof entry === "string") : undefined,
    };
  }
  return capabilities;
}

function normalizeModel(value: unknown, fallback?: ModelSummary, lifecycleOverride?: ModelSummary["lifecycle"]): ModelSummary {
  const outer = asRecord(value);
  const raw = Object.keys(asRecord(outer.model)).length ? asRecord(outer.model) : outer;
  const fingerprint = asRecord(raw.fingerprint);
  const architectures = Array.isArray(raw.architectures) ? raw.architectures.filter((item): item is string => typeof item === "string") : [];
  const taskValue = asString(raw.task, "unknown");
  const task: ModelSummary["task"] = taskValue === "text_generation" || taskValue === "encoder_decoder_generation"
    ? taskValue
    : taskValue === "embedding" || taskValue === "multimodal_embedding" ? taskValue : "unknown";
  const diagnostics = Array.isArray(raw.diagnostics) ? raw.diagnostics.map((value) => {
    const diagnostic = asRecord(value);
    const message = typeof diagnostic.message === "string" ? diagnostic.message : "Unknown model diagnostic.";
    return typeof diagnostic.hint === "string" ? `${message} ${diagnostic.hint}` : message;
  }) : undefined;
  const contextValues = Array.isArray(raw.context_values) ? raw.context_values.map((value) => {
    const candidate = asRecord(value);
    return {
      source: asString(candidate.source, "unknown"),
      tokens: asNullableNumber(candidate.value),
      selected: candidate.value === raw.effective_context_limit,
      note: typeof candidate.note === "string" ? candidate.note : undefined,
    };
  }) : undefined;
  const lifecycleValue = lifecycleOverride ?? raw.lifecycle ?? raw.load_state ?? raw.status ?? fallback?.lifecycle ?? "unloaded";
  const lifecycle: ModelSummary["lifecycle"] = lifecycleValue === "loading" || lifecycleValue === "loaded" || lifecycleValue === "unloading" || lifecycleValue === "error" ? lifecycleValue : "unloaded";
  return {
    id: asString(raw.id ?? raw.model_id ?? raw.unloaded_model_id, fallback?.id ?? "unknown"),
    name: asString(raw.name ?? raw.display_name ?? raw.id, fallback?.name ?? "Unknown model"),
    architecture: typeof raw.architecture === "string" ? raw.architecture : architectures[0] ?? fallback?.architecture ?? null,
    task: task === "unknown" ? fallback?.task ?? task : task,
    fingerprint: typeof raw.fingerprint === "string" ? raw.fingerprint : typeof fingerprint.value === "string" ? fingerprint.value : fallback?.fingerprint ?? null,
    parameterCount: asNullableNumber(raw.parameterCount ?? raw.parameter_count) ?? fallback?.parameterCount,
    dtype: typeof raw.dtype === "string" ? raw.dtype : fallback?.dtype ?? null,
    lifecycle,
    loadedDevice: typeof raw.loadedDevice === "string" ? raw.loadedDevice : typeof raw.loaded_device === "string" ? raw.loaded_device : typeof raw.device === "string" ? raw.device : fallback?.loadedDevice ?? null,
    capabilities: Object.keys(normalizeCapabilities(raw.capabilities)).length ? normalizeCapabilities(raw.capabilities) : fallback?.capabilities ?? {},
    effectiveContextLimit: asNullableNumber(raw.effectiveContextLimit ?? raw.effective_context_limit) ?? fallback?.effectiveContextLimit ?? null,
    contextLimits: Array.isArray(raw.contextLimits) ? raw.contextLimits as ModelSummary["contextLimits"] : contextValues ?? fallback?.contextLimits,
    diagnostics: diagnostics ?? fallback?.diagnostics,
    trustRemoteCode: raw.trustRemoteCode === true || raw.trust_decision === "reviewed_bundled_code" || fallback?.trustRemoteCode === true,
  };
}

function normalizeAlternative(value: unknown): TokenAlternative {
  const raw = asRecord(value);
  return {
    tokenId: Number(raw.tokenId ?? raw.token_id ?? -1),
    piece: asString(raw.piece, ""),
    probability: Number(raw.probability ?? 0),
    logProbability: Number(raw.logProbability ?? raw.log_probability ?? 0),
    logit: Number(raw.logit ?? 0),
    rank: Number(raw.rank ?? 0),
    survivedFiltering: asOptionalBoolean(raw.survivedFiltering ?? raw.survived_filter),
  };
}

function normalizeReasoningSlice(value: unknown): ReasoningSlice | null {
  const raw = asRecord(value);
  const start = Number(raw.start);
  const end = Number(raw.end);
  const classification = raw.classification;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) return null;
  return {
    start,
    end,
    classification: classification === "reasoning" || classification === "answer" ? classification : "unknown",
    delimiter: raw.delimiter === true,
  };
}

function normalizeToken(value: unknown): TokenEvent {
  const raw = asRecord(value);
  const alternatives = asRecord(raw.alternatives);
  const alternativesList = Array.isArray(raw.alternatives) ? raw.alternatives : [];
  const storedRawAlternatives = alternativesList.filter((item) => asRecord(item).distribution === "raw");
  const storedSamplingAlternatives = alternativesList.filter((item) => asRecord(item).distribution === "sampling");
  const expertRoutingValue = raw.expertRoutes ?? raw.expert_routing;
  const expertRoutes = Array.isArray(expertRoutingValue) ? expertRoutingValue.map((value) => {
    const route = asRecord(value);
    return {
      layer: Number(route.layer ?? route.layer_index ?? 0),
      selectedExpertIds: (Array.isArray(route.selectedExpertIds) ? route.selectedExpertIds : Array.isArray(route.selected_expert_ids) ? route.selected_expert_ids : []).map(Number),
      executedExpertIds: (Array.isArray(route.executedExpertIds) ? route.executedExpertIds : Array.isArray(route.executed_expert_ids) ? route.executed_expert_ids : []).map(Number),
      gateWeights: (Array.isArray(route.gateWeights) ? route.gateWeights : Array.isArray(route.gate_weights) ? route.gate_weights : []).map(Number),
      routerEntropy: asOptionalNumber(route.routerEntropy ?? route.router_entropy),
      dropped: typeof route.dropped === "boolean" ? route.dropped : undefined,
    };
  }) : undefined;
  const characterSpan = typeof raw.span_start === "number" && typeof raw.span_end === "number"
    ? [raw.span_start, raw.span_end] as [number, number]
    : Array.isArray(raw.characterSpan) && raw.characterSpan.length === 2 ? raw.characterSpan as [number, number] : undefined;
  const segment = raw.reasoningSegment ?? raw.segment;
  const reasoningSlicesValue = raw.reasoningSlices ?? raw.reasoning_slices;
  const reasoningSlices = Array.isArray(reasoningSlicesValue)
    ? reasoningSlicesValue.map(normalizeReasoningSlice).filter((slice): slice is ReasoningSlice => slice !== null)
    : [];
  return {
    index: Number(raw.index ?? raw.token_index ?? 0),
    tokenId: Number(raw.tokenId ?? raw.token_id ?? -1),
    piece: asString(raw.piece, ""),
    bytes: typeof (raw.bytes ?? raw.escaped_bytes) === "string" ? String(raw.bytes ?? raw.escaped_bytes) : undefined,
    displayText: asString(raw.displayText ?? raw.display_text ?? raw.piece, ""),
    replaceFrom: asOptionalNumber(raw.replaceFrom ?? raw.replace_from),
    characterSpan,
    rawLogit: asOptionalNumber(raw.rawLogit ?? raw.raw_logit),
    rawLogProbability: asOptionalNumber(raw.rawLogProbability ?? raw.raw_logprob),
    rawProbability: asOptionalNumber(raw.rawProbability ?? raw.raw_probability),
    rawRank: asOptionalNumber(raw.rawRank ?? raw.raw_rank),
    processedLogit: asOptionalNumber(raw.processedLogit ?? raw.processed_logit),
    samplingLogProbability: asOptionalNumber(raw.samplingLogProbability ?? raw.sample_logprob),
    samplingProbability: asOptionalNumber(raw.samplingProbability ?? raw.sample_probability),
    entropy: asOptionalNumber(raw.entropy),
    surprise: asOptionalNumber(raw.surprise),
    cumulativeLogProbability: asOptionalNumber(raw.cumulativeLogProbability ?? raw.cumulative_logprob),
    runningPerplexity: asOptionalNumber(raw.runningPerplexity ?? raw.running_perplexity),
    reasoningSegment: segment === "reasoning" || segment === "answer" ? segment : "unknown",
    reasoningSlices: reasoningSlices.length > 0 ? reasoningSlices : undefined,
    rawAlternatives: Array.isArray(raw.rawAlternatives) ? raw.rawAlternatives.map(normalizeAlternative) : Array.isArray(alternatives.raw) ? alternatives.raw.map(normalizeAlternative) : storedRawAlternatives.length ? storedRawAlternatives.map(normalizeAlternative) : undefined,
    samplingAlternatives: Array.isArray(raw.samplingAlternatives) ? raw.samplingAlternatives.map(normalizeAlternative) : Array.isArray(alternatives.sampling) ? alternatives.sampling.map(normalizeAlternative) : storedSamplingAlternatives.length ? storedSamplingAlternatives.map(normalizeAlternative) : undefined,
    expertRoutes,
    timing: {
      decodeMs: asOptionalNumber(raw.decode_ms ?? asRecord(raw.timing).decodeMs),
      samplingMs: asOptionalNumber(raw.sample_ms ?? asRecord(raw.timing).samplingMs),
      emissionMs: asOptionalNumber(raw.emit_ms ?? asRecord(raw.timing).emissionMs),
      interTokenMs: asOptionalNumber(raw.inter_token_ms ?? asRecord(raw.timing).interTokenMs),
      cumulativeMs: asOptionalNumber(raw.cumulative_ms ?? asRecord(raw.timing).cumulativeMs),
      instantaneousTps: asOptionalNumber(raw.instantaneous_tps ?? asRecord(raw.timing).instantaneousTps),
      rollingTps: asOptionalNumber(raw.rolling_tps ?? asRecord(raw.timing).rollingTps),
    },
  };
}

function metricsFromPayload(value: unknown): NonNullable<RunDetails["metrics"]> {
  const raw = asRecord(value);
  const memory = asRecord(raw.memory);
  const segmentMetrics = asRecord(raw.segment_metrics);
  const reasoning = asRecord(segmentMetrics.reasoning);
  const answer = asRecord(segmentMetrics.answer);
  const context = asRecord(raw.context);
  const renderedPromptTokens = asOptionalNumber(raw.prompt_tokens ?? raw.renderedPromptTokens ?? context.rendered_prompt_tokens ?? context.renderedPromptTokens);
  const templateTokens = asOptionalNumber(raw.template_tokens ?? raw.templateTokens ?? context.template_tokens ?? context.templateTokens);
  const multimodalPositions = asOptionalNumber(raw.multimodal_positions ?? raw.multimodalPositions ?? context.multimodal_positions ?? context.multimodalPositions);
  const generatedTokens = asOptionalNumber(raw.generated_token_count ?? raw.generatedTokens ?? context.generated_tokens ?? context.generatedTokens);
  const reservedOutputTokens = asOptionalNumber(raw.reserved_output_tokens ?? raw.reservedOutputTokens ?? context.reserved_output_tokens ?? context.reservedOutputTokens);
  const remainingTokens = asOptionalNumber(raw.remaining_tokens ?? raw.remainingTokens ?? context.remaining_tokens ?? context.remainingTokens);
  const effectiveLimitValue = raw.context_limit ?? raw.effectiveLimit ?? context.effective_limit ?? context.effectiveLimit;
  const contextReported = renderedPromptTokens !== undefined
    || templateTokens !== undefined
    || multimodalPositions !== undefined
    || generatedTokens !== undefined
    || reservedOutputTokens !== undefined
    || remainingTokens !== undefined
    || typeof effectiveLimitValue === "number";
  return {
    finishReason: typeof raw.finishReason === "string" ? raw.finishReason : typeof raw.finish_reason === "string" ? raw.finish_reason : undefined,
    responsePerplexity: asOptionalNumber(raw.responsePerplexity ?? raw.conditional_response_perplexity),
    promptTokens: asOptionalNumber(raw.promptTokens ?? raw.prompt_tokens ?? raw.prompt_token_count),
    generatedTokens: asOptionalNumber(raw.generatedTokens ?? raw.generated_token_count),
    reasoningTokens: asOptionalNumber(reasoning.token_count),
    answerTokens: asOptionalNumber(answer.token_count),
    reasoningPerplexity: asOptionalNumber(reasoning.perplexity),
    answerPerplexity: asOptionalNumber(answer.perplexity),
    peakRamBytes: asOptionalNumber(raw.peakRamBytes ?? memory.peak_ram_bytes ?? memory.process_rss_bytes ?? memory.rss_bytes),
    peakVramBytes: asOptionalNumber(raw.peakVramBytes ?? memory.peak_vram_bytes ?? memory.cuda_peak_allocated_bytes ?? memory.cuda_allocated_bytes),
    timing: {
      queueMs: asOptionalNumber(raw.queueMs ?? raw.queue_ms),
      preprocessingMs: asOptionalNumber(raw.preprocessingMs ?? raw.preprocessing_ms),
      templateRenderingMs: asOptionalNumber(raw.templateRenderingMs ?? raw.template_ms),
      tokenizationMs: asOptionalNumber(raw.tokenizationMs ?? raw.tokenization_ms),
      prefillMs: asOptionalNumber(raw.prefillMs ?? raw.prefill_ms),
      serverTtftMs: asOptionalNumber(raw.serverTtftMs ?? raw.server_ttft_ms),
      engineTtftMs: asOptionalNumber(raw.engineTtftMs ?? raw.engine_ttft_ms),
      clientTtftMs: asOptionalNumber(raw.clientTtftMs ?? raw.client_ttft_ms),
      promptTokensPerSecond: asOptionalNumber(raw.promptTokensPerSecond ?? raw.prompt_tokens_per_second),
      totalMs: asOptionalNumber(raw.totalMs ?? raw.total_generation_ms),
      decodeTokensPerSecond: asOptionalNumber(raw.decodeTokensPerSecond ?? raw.decode_tokens_per_second),
      endToEndTokensPerSecond: asOptionalNumber(raw.endToEndTokensPerSecond ?? raw.end_to_end_tokens_per_second),
    },
    context: contextReported ? {
      renderedPromptTokens,
      templateTokens,
      multimodalPositions,
      generatedTokens,
      reservedOutputTokens,
      remainingTokens,
      effectiveLimit: asNullableNumber(effectiveLimitValue),
    } : undefined,
  };
}

function flattenPhaseMetrics(value: unknown): Record<string, unknown> {
  if (!Array.isArray(value)) return {};
  const flattened: Record<string, unknown> = {};
  for (const phaseValue of value) {
    const phase = asRecord(phaseValue);
    Object.assign(flattened, asRecord(phase.details ?? phase.details_json));
    const duration = phase.duration_ms;
    if (typeof duration !== "number") continue;
    if (phase.phase === "chat_template") flattened.template_ms = duration;
    if (phase.phase === "tokenization") flattened.tokenization_ms = duration;
    if (phase.phase === "prefill") flattened.prefill_ms = duration;
    if (phase.phase === "generation") flattened.total_generation_ms = duration;
  }
  return flattened;
}

function normalizeStreamEvent(value: unknown): RunStreamEvent | null {
  const raw = asRecord(value);
  const version = raw.version === 1 ? 1 : null;
  const sequence = asOptionalNumber(raw.sequence);
  if (version === null || sequence === undefined) return null;
  const type = asString(raw.type, "");
  const payload = asRecord(raw.payload);
  const preserveEnvelope = (event: RunStreamEvent): RunStreamEvent => ({
    ...event,
    raw: { ...raw },
  });
  if (type === "run.created" || type === "run_created") return preserveEnvelope({ version, sequence, type: "run.created", run: type === "run.created" ? asRecord(raw.run) : { status: "queued", reproducibility: typeof payload.effective_seed === "string" ? { effectiveSeed: payload.effective_seed } : undefined } });
  if (type === "stage.changed") return preserveEnvelope({ version, sequence, type, stage: runStatus(raw.stage, "running"), detail: typeof raw.detail === "string" ? raw.detail : undefined });
  if (type === "stage" || type === "model_loaded") {
    const stage = asString(payload.stage, "running");
    return preserveEnvelope({ version, sequence, type: "stage.changed", stage: stage.includes("load") ? "loading" : "running", detail: stage, metrics: metricsFromPayload(payload) });
  }
  if (type === "token") return preserveEnvelope({ version, sequence, type, token: normalizeToken(raw.token ?? payload) });
  if (type === "metrics" || type === "metric") {
    const metricPayload = asRecord(raw.metrics ?? payload);
    const samplingPipeline = Array.isArray(metricPayload.sampling_operation_order) ? metricPayload.sampling_operation_order.filter((item): item is string => typeof item === "string") : undefined;
    return preserveEnvelope({
      version,
      sequence,
      type: "metrics",
      metrics: metricsFromPayload(metricPayload),
      samplingPipeline,
      reproducibility: typeof metricPayload.effective_seed === "string" ? {
        effectiveSeed: metricPayload.effective_seed,
        rngAlgorithm: typeof metricPayload.rng_algorithm === "string" ? metricPayload.rng_algorithm : undefined,
        generatorDevice: typeof metricPayload.generator_device === "string" ? metricPayload.generator_device : undefined,
      } : undefined,
    });
  }
  if (type === "warning") return preserveEnvelope({ version, sequence, type, message: asString(raw.message ?? payload.message, "Run warning") });
  if (type === "completed") return preserveEnvelope({ version, sequence, type, run: { status: "complete", metrics: metricsFromPayload(raw.run ?? payload) } });
  if (type === "cancelled") return preserveEnvelope({ version, sequence, type, reason: typeof (raw.reason ?? payload.reason) === "string" ? String(raw.reason ?? payload.reason) : undefined, metrics: metricsFromPayload(payload) });
  if (type === "error") return preserveEnvelope({ version, sequence, type, code: typeof (raw.code ?? payload.code) === "string" ? asString(raw.code ?? payload.code, "") : undefined, message: asString(raw.message ?? payload.message, "Generation failed") });
  return preserveEnvelope({ version, sequence, type: "warning", message: type === "resync_required" ? "The live buffer rolled over; reconnecting from persisted events." : `Ignored unsupported event type “${type}”.` });
}

async function request<T>(path: string, init?: RequestInit, responseType: "json" | "blob" = "json"): Promise<T> {
  const authToken = getSessionAuthToken();
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    if (response.status === 401) notifyAuthenticationRequired();
    let payload: ApiErrorPayload | undefined;
    try {
      payload = (await response.json()) as ApiErrorPayload;
    } catch {
      payload = undefined;
    }
    throw new ApiError(errorMessage(payload, `${response.status} ${response.statusText}`), response.status, payload);
  }
  if (response.status === 204) return undefined as T;
  return (responseType === "blob" ? await response.blob() : await response.json()) as T;
}

function unwrapList<T>(value: T[] | { items?: T[]; models?: T[]; chats?: T[] }): T[] {
  if (Array.isArray(value)) return value;
  return value.items ?? value.models ?? value.chats ?? [];
}

function toChat(chat: ChatSummary & Record<string, unknown>): ChatSummary {
  return {
    ...chat,
    createdAt: String(chat.createdAt ?? chat.created_at ?? new Date(0).toISOString()),
    updatedAt: String(chat.updatedAt ?? chat.updated_at ?? chat.createdAt ?? new Date(0).toISOString()),
    pinned: Boolean(chat.pinned ?? chat.is_pinned),
    archived: Boolean(chat.archived ?? chat.is_archived),
  };
}

function normalizeAttachment(value: unknown): Attachment {
  const raw = asRecord(value);
  const mimeType = asString(raw.mimeType ?? raw.media_type, "application/octet-stream");
  const preprocessing = asRecord(raw.preprocessing);
  const topLevelNative = asOptionalBoolean(raw.nativeProcessing ?? raw.native_processing);
  const preprocessingPath = preprocessing.path;
  const preprocessingStrategy = preprocessing.strategy;
  const nativePreprocessing = preprocessing.native === true
    || preprocessingPath === "native"
    || preprocessingPath === "native_model_processor"
    || preprocessingStrategy === "native"
    || preprocessingStrategy === "native_model_processor";
  const kind: Attachment["kind"] = mimeType.startsWith("image/") ? "image" : mimeType.startsWith("audio/") ? "audio" : mimeType.startsWith("video/") ? "video" : mimeType.startsWith("text/") || mimeType.includes("pdf") || mimeType.includes("document") ? "document" : "unknown";
  return {
    id: asString(raw.id, "unknown"),
    name: asString(raw.name ?? raw.original_name, "attachment"),
    mimeType,
    sizeBytes: Number(raw.sizeBytes ?? raw.size_bytes ?? 0),
    kind,
    status: "ready",
    nativeProcessing: topLevelNative ?? nativePreprocessing,
    detail: typeof preprocessing.summary === "string" ? preprocessing.summary : undefined,
    tokenContribution: asOptionalNumber(raw.tokenContribution ?? raw.token_contribution),
  };
}

function toMessage(message: Message & Record<string, unknown>): Message {
  const statusValue = String(message.status ?? "complete");
  const roleValue = String(message.role ?? "system");
  const metadata = asRecord(message.metadata);
  return {
    ...message,
    chatId: String(message.chatId ?? message.chat_id ?? ""),
    createdAt: String(message.createdAt ?? message.created_at ?? new Date(0).toISOString()),
    runId: typeof message.runId === "string" ? message.runId : typeof message.run_id === "string" ? message.run_id : undefined,
    parentMessageId: typeof (message.parentMessageId ?? message.parent_id) === "string" ? String(message.parentMessageId ?? message.parent_id) : undefined,
    reasoningPrimed: message.reasoningPrimed === true || metadata.reasoning_primed === true,
    attachments: Array.isArray(message.attachments) ? message.attachments.map(normalizeAttachment) : undefined,
    role: roleValue === "user" || roleValue === "assistant" ? roleValue : "system",
    status: statusValue === "cancelled" || statusValue === "failed" ? statusValue : statusValue === "streaming" || statusValue === "pending" ? "streaming" : "complete",
  };
}

function normalizeRun(value: unknown): RunDetails {
  const outer = asRecord(value);
  const raw = asRecord(outer.run ?? outer);
  const reproducibilityRaw = asRecord(raw.reproducibility);
  const environmentRaw = asRecord(raw.environment ?? outer.environment);
  const phaseMetrics = flattenPhaseMetrics(raw.phases ?? outer.phases);
  const summary = asRecord(raw.summary ?? outer.summary);
  const tokensRaw = Array.isArray(outer.tokens) ? outer.tokens : Array.isArray(raw.tokens) ? raw.tokens : [];
  const eventsRaw = Array.isArray(outer.events) ? outer.events : Array.isArray(raw.events) ? raw.events : [];
  const eventSlicesByToken = new Map<number, ReasoningSlice[]>();
  for (const eventValue of eventsRaw) {
    const event = asRecord(eventValue);
    if (event.type !== "token") continue;
    const eventToken = normalizeToken(event.payload ?? event.token);
    if (eventToken.reasoningSlices?.length) eventSlicesByToken.set(eventToken.index, eventToken.reasoningSlices);
  }
  const tokens = tokensRaw.map(normalizeToken).map((token) => token.reasoningSlices?.length
    ? token
    : { ...token, reasoningSlices: eventSlicesByToken.get(token.index) });
  const reportedMetrics = metricsFromPayload({ ...raw, ...phaseMetrics, ...summary, ...asRecord(outer.metrics ?? raw.metrics) });
  const effectiveConfig = asRecord(raw.effective_config);
  const inferenceConfig = asRecord(effectiveConfig.inference);
  const configuredReservedOutputTokens = asOptionalNumber(inferenceConfig.reserved_output_tokens ?? inferenceConfig.reservedOutputTokens);
  if (configuredReservedOutputTokens !== undefined && reportedMetrics.context?.reservedOutputTokens === undefined) {
    reportedMetrics.context = {
      ...reportedMetrics.context,
      reservedOutputTokens: configuredReservedOutputTokens,
      effectiveLimit: reportedMetrics.context?.effectiveLimit ?? null,
    };
  }
  const lastToken = tokens.at(-1);
  const metrics = {
    ...reportedMetrics,
    generatedTokens: reportedMetrics.generatedTokens ?? tokens.length,
    responsePerplexity: reportedMetrics.responsePerplexity ?? lastToken?.runningPerplexity,
    timing: {
      ...reportedMetrics.timing,
      queueMs: reportedMetrics.timing?.queueMs ?? timestampDeltaMs(raw.queue_exited_at, raw.queue_entered_at ?? raw.received_at),
      serverTtftMs: reportedMetrics.timing?.serverTtftMs ?? timestampDeltaMs(raw.first_token_at, raw.received_at),
    },
  };
  const status = runStatus(raw.status);
  const effectiveSeed = reproducibilityRaw.effectiveSeed ?? reproducibilityRaw.effective_seed ?? raw.effective_seed;
  const deviceSelection = asRecord(reproducibilityRaw.device);
  const deterministicKernelsRaw = reproducibilityRaw.deterministic_kernels ?? reproducibilityRaw.deterministicKernels;
  const deterministicKernelSettings = asRecord(deterministicKernelsRaw);
  const deterministicKernels = typeof deterministicKernelsRaw === "boolean"
    ? deterministicKernelsRaw
    : asOptionalBoolean(deterministicKernelSettings.torch_use_deterministic_algorithms ?? deterministicKernelSettings.torchUseDeterministicAlgorithms);
  const deviceWarnings = Array.isArray(deviceSelection.warnings) ? deviceSelection.warnings.filter((item): item is string => typeof item === "string") : [];
  const deviceReasonParts = [typeof deviceSelection.reason === "string" ? deviceSelection.reason : undefined, ...deviceWarnings].filter((item): item is string => Boolean(item));
  const reproducibility = effectiveSeed === undefined ? undefined : {
    requestedSeed: reproducibilityRaw.requestedSeed as string | number | null | undefined ?? reproducibilityRaw.requested_seed as string | number | null | undefined ?? raw.requested_seed as string | number | null | undefined,
    effectiveSeed: asString(effectiveSeed, "not reported"),
    rngAlgorithm: typeof (reproducibilityRaw.rngAlgorithm ?? reproducibilityRaw.rng_algorithm ?? raw.rng_algorithm) === "string" ? String(reproducibilityRaw.rngAlgorithm ?? reproducibilityRaw.rng_algorithm ?? raw.rng_algorithm) : undefined,
    generatorDevice: typeof (reproducibilityRaw.generatorDevice ?? reproducibilityRaw.generator_device ?? raw.generator_device) === "string" ? String(reproducibilityRaw.generatorDevice ?? reproducibilityRaw.generator_device ?? raw.generator_device) : undefined,
    modelFingerprint: typeof raw.model_fingerprint === "string" ? raw.model_fingerprint : typeof asRecord(reproducibilityRaw.model_fingerprint).value === "string" ? String(asRecord(reproducibilityRaw.model_fingerprint).value) : undefined,
    tokenizerFingerprint: typeof (raw.tokenizer_fingerprint ?? reproducibilityRaw.tokenizer_fingerprint) === "string" ? String(raw.tokenizer_fingerprint ?? reproducibilityRaw.tokenizer_fingerprint) : undefined,
    backend: typeof reproducibilityRaw.backend === "string" ? reproducibilityRaw.backend : typeof asRecord(environmentRaw.backend).name === "string" ? String(asRecord(environmentRaw.backend).name) : undefined,
    device: typeof reproducibilityRaw.device === "string" ? reproducibilityRaw.device : typeof deviceSelection.device_identifier === "string" ? deviceSelection.device_identifier : typeof deviceSelection.selected_backend === "string" ? deviceSelection.selected_backend : undefined,
    deviceReason: deviceReasonParts.length ? deviceReasonParts.join(" ") : undefined,
    dtype: typeof reproducibilityRaw.dtype === "string" ? reproducibilityRaw.dtype : typeof deviceSelection.effective_dtype === "string" ? deviceSelection.effective_dtype : undefined,
    quantization: typeof reproducibilityRaw.quantization === "string" ? reproducibilityRaw.quantization : null,
    attentionImplementation: typeof reproducibilityRaw.attention_implementation === "string" ? reproducibilityRaw.attention_implementation : undefined,
    softwareVersions: asRecord(Object.keys(asRecord(reproducibilityRaw.software_versions)).length ? reproducibilityRaw.software_versions : environmentRaw.software) as Record<string, string>,
    deterministicKernels,
    batching: reproducibilityRaw.batching === undefined ? undefined : JSON.stringify(reproducibilityRaw.batching),
  };
  return {
    id: asString(raw.id ?? raw.run_id, "unknown"),
    chatId: typeof (raw.chatId ?? raw.chat_id) === "string" ? String(raw.chatId ?? raw.chat_id) : undefined,
    messageId: typeof (raw.messageId ?? raw.message_id) === "string" ? String(raw.messageId ?? raw.message_id) : undefined,
    modelId: asString(raw.modelId ?? raw.model_id, "unknown"),
    status,
    createdAt: asString(raw.createdAt ?? raw.created_at, new Date(0).toISOString()),
    completedAt: typeof (raw.completedAt ?? raw.completed_at) === "string" ? String(raw.completedAt ?? raw.completed_at) : undefined,
    tokens,
    branchableThroughTokenIndex: asOptionalNumber(raw.branchableThroughTokenIndex ?? raw.branchable_through_token_index),
    metrics,
    reproducibility,
    effectiveSettings: {
      ...asRecord(raw.effective_config),
      ...asRecord(raw.effectiveSettings ?? raw.effective_settings ?? raw.settings),
    },
    samplingPipeline: Array.isArray(raw.samplingPipeline ?? raw.sampling_pipeline ?? phaseMetrics.sampling_operation_order) ? (raw.samplingPipeline ?? raw.sampling_pipeline ?? phaseMetrics.sampling_operation_order) as string[] : undefined,
    warnings: Array.isArray(raw.warnings) ? raw.warnings.filter((item): item is string => typeof item === "string") : undefined,
    rawEvents: eventsRaw.length > 0 ? eventsRaw.map(normalizeStreamEvent).filter((event): event is RunStreamEvent => event !== null) : undefined,
    error: typeof (raw.error ?? raw.error_message) === "string" ? String(raw.error ?? raw.error_message) : undefined,
  };
}

function normalizeEmbedding(value: unknown, inputs: EmbeddingInput[], requestedDimensions: number | undefined): EmbeddingRun {
  const raw = asRecord(value);
  const resultValues = Array.isArray(raw.results) ? raw.results : Array.isArray(raw.vectors) ? raw.vectors : [];
  const vectors = resultValues.map((value) => {
    const item = asRecord(value);
    const statistics = asRecord(item.statistics);
    const values = Array.isArray(item.values) ? item.values : Array.isArray(item.vector) ? item.vector : [];
    return {
      inputId: asString(item.inputId ?? item.input_id, "unknown"),
      values: values.map(Number),
      dimensions: Number(item.dimensions ?? item.output_dimension ?? values.length),
      dtype: asString(item.dtype ?? item.output_dtype, "not reported"),
      l2Norm: Number(item.l2Norm ?? item.l2_norm ?? 0),
      minimum: asOptionalNumber(item.minimum ?? statistics.minimum ?? statistics.min),
      maximum: asOptionalNumber(item.maximum ?? statistics.maximum ?? statistics.max),
      mean: asOptionalNumber(item.mean ?? statistics.mean),
      standardDeviation: asOptionalNumber(item.standardDeviation ?? statistics.standard_deviation ?? statistics.std),
    } satisfies EmbeddingRun["vectors"][number];
  });
  return {
    id: asString(raw.id ?? raw.run_id, "unknown"),
    modelId: asString(raw.modelId ?? raw.model_id, "unknown"),
    status: raw.status === "failed" ? "failed" : raw.status === "queued" || raw.status === "running" ? raw.status : "complete",
    inputs,
    vectors,
    requestedDimensions,
    outputDimensions: Number(raw.outputDimensions ?? raw.output_dimensions ?? vectors[0]?.dimensions ?? 0),
    normalized: raw.normalized !== false,
    pooling: typeof raw.pooling === "string" ? raw.pooling : undefined,
    jointSpace: typeof (raw.jointSpace ?? raw.joint_embedding_space) === "boolean" ? Boolean(raw.jointSpace ?? raw.joint_embedding_space) : undefined,
    truncation: typeof raw.truncation === "string" ? raw.truncation : undefined,
    preprocessingMs: asOptionalNumber(raw.preprocessingMs ?? raw.preprocessing_ms),
    forwardMs: asOptionalNumber(raw.forwardMs ?? raw.forward_ms),
    totalMs: asOptionalNumber(raw.totalMs ?? raw.total_ms),
    tokensPerSecond: asOptionalNumber(raw.tokensPerSecond ?? raw.tokens_per_second),
    itemsPerSecond: asOptionalNumber(raw.itemsPerSecond ?? raw.items_per_second),
    similarityMatrix: Array.isArray(raw.similarityMatrix ?? raw.similarity_matrix) ? (raw.similarityMatrix ?? raw.similarity_matrix) as number[][] : undefined,
    projection: Array.isArray(raw.projection) ? raw.projection as EmbeddingRun["projection"] : undefined,
    error: typeof raw.error === "string" ? raw.error : undefined,
  };
}

function normalizeGenerationResponse(value: unknown): GenerateResponse {
  const raw = asRecord(value);
  const runRaw = asRecord(raw.run);
  const user = asRecord(raw.user_message);
  const assistant = asRecord(raw.assistant_message);
  return {
    chatId: typeof (raw.chatId ?? raw.chat_id ?? runRaw.chat_id) === "string" ? String(raw.chatId ?? raw.chat_id ?? runRaw.chat_id) : undefined,
    runId: asString(raw.runId ?? raw.run_id ?? runRaw.id, ""),
    messageId: typeof (raw.messageId ?? raw.message_id ?? assistant.id) === "string" ? String(raw.messageId ?? raw.message_id ?? assistant.id) : undefined,
    userMessageId: typeof user.id === "string" ? user.id : undefined,
    modelId: typeof runRaw.model_id === "string" ? runRaw.model_id : undefined,
    parentRunId: typeof (raw.parentRunId ?? raw.parent_run_id ?? runRaw.parent_run_id) === "string" ? String(raw.parentRunId ?? raw.parent_run_id ?? runRaw.parent_run_id) : undefined,
    sourceRunId: typeof (raw.sourceRunId ?? raw.source_run_id) === "string" ? String(raw.sourceRunId ?? raw.source_run_id) : undefined,
    websocketUrl: typeof (raw.websocketUrl ?? raw.websocket_url) === "string" ? String(raw.websocketUrl ?? raw.websocket_url) : undefined,
    run: typeof runRaw.model_id === "string" || typeof runRaw.modelId === "string" ? normalizeRun(runRaw) : undefined,
  };
}

export const api = {
  async health(): Promise<HealthStatus> {
    const raw = asRecord(await request<unknown>("/health"));
    const loadedModel = asRecord(raw.loadedModel ?? raw.loaded_model);
    return {
      status: raw.status === "ok" || raw.status === "degraded" ? raw.status : "error",
      version: typeof raw.version === "string" ? raw.version : undefined,
      database: typeof raw.database === "string" ? raw.database : undefined,
      worker: typeof raw.worker === "string" ? raw.worker : undefined,
      loadedModelId: typeof (loadedModel.modelId ?? loadedModel.model_id) === "string" ? String(loadedModel.modelId ?? loadedModel.model_id) : undefined,
      loadedDevice: typeof loadedModel.device === "string" ? loadedModel.device : undefined,
      selectedBackend: typeof (raw.selectedBackend ?? raw.selected_backend) === "string" ? String(raw.selectedBackend ?? raw.selected_backend) : undefined,
      reason: typeof raw.reason === "string" ? raw.reason : undefined,
    };
  },
  async configuration(): Promise<ConfigurationSnapshot> {
    const raw = asRecord(await request<unknown>("/configuration"));
    return {
      effective: asRecord(raw.effective),
      precedence: Array.isArray(raw.precedence) ? raw.precedence.filter((item): item is string => typeof item === "string") : [],
    };
  },
  async models(): Promise<ModelSummary[]> {
    const value = await request<unknown[] | { items?: unknown[]; models?: unknown[] }>("/models");
    return unwrapList(value).map((model) => normalizeModel(model));
  },
  async refreshModels(): Promise<ModelSummary[]> {
    const value = await request<unknown[] | { items?: unknown[]; models?: unknown[] }>("/models/refresh", { method: "POST" });
    return unwrapList(value).map((model) => normalizeModel(model));
  },
  async loadModel(id: string, options: { device: string; dtype: string }, current?: ModelSummary): Promise<ModelSummary> {
    return normalizeModel(await request(`/models/${encodeURIComponent(id)}/load`, { method: "POST", body: JSON.stringify(options) }), current, "loaded");
  },
  async unloadModel(id: string, current?: ModelSummary): Promise<ModelSummary> {
    return normalizeModel(await request(`/models/${encodeURIComponent(id)}/unload`, { method: "POST" }), current, "unloaded");
  },
  async inspectModel(id: string): Promise<ModelInspection> {
    const raw = asRecord(await request(`/models/${encodeURIComponent(id)}/inspect`));
    return {
      tokenizer: Object.keys(asRecord(raw.tokenizer)).length ? asRecord(raw.tokenizer) : undefined,
      generation: Object.keys(asRecord(raw.generation)).length ? asRecord(raw.generation) : undefined,
      specialTokens: Object.keys(asRecord(raw.special_tokens)).length ? asRecord(raw.special_tokens) : undefined,
      chatTemplate: typeof raw.chat_template === "string" ? raw.chat_template : null,
      samplingPipeline: Array.isArray(raw.sampling_pipeline) ? raw.sampling_pipeline.filter((item): item is string => typeof item === "string") : undefined,
    };
  },
  async chats(archived = false): Promise<ChatSummary[]> {
    const value = await request<ChatSummary[] | { items?: ChatSummary[]; chats?: ChatSummary[] }>(`/chats?archived=${String(archived)}`);
    return unwrapList(value).map((chat) => toChat(chat as ChatSummary & Record<string, unknown>));
  },
  async createChat(): Promise<ChatSummary> {
    return toChat(await request<ChatSummary & Record<string, unknown>>("/chats", { method: "POST", body: "{}" }));
  },
  async updateChat(id: string, changes: Partial<Pick<ChatSummary, "title" | "pinned" | "archived">>): Promise<ChatSummary> {
    return toChat(await request<ChatSummary & Record<string, unknown>>(`/chats/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(changes) }));
  },
  deleteChat: (id: string): Promise<void> => request(`/chats/${encodeURIComponent(id)}`, { method: "DELETE" }),
  clearChats: (): Promise<void> => request("/chats?confirm=true&include_archived=true", { method: "DELETE" }),
  async messages(chatId: string): Promise<Message[]> {
    const value = await request<Message[] | { items?: Message[] }>(`/chats/${encodeURIComponent(chatId)}/messages`);
    return unwrapList(value).map((message) => toMessage(message as Message & Record<string, unknown>));
  },
  async generate(body: GenerateRequest): Promise<GenerateResponse> {
    const response = await request<unknown>("/runs", {
      method: "POST",
      body: JSON.stringify({
        chat_id: body.chatId,
        model_id: body.modelId,
        prompt: body.content,
        parent_message_id: body.parentMessageId,
        seed: body.settings.seed.trim() ? body.settings.seed.trim() : null,
        device: body.settings.device,
        dtype: body.settings.dtype,
        instrumentation: body.settings.instrumentation,
        reasoning: body.settings.reasoning,
        deterministic_reference_mode: body.settings.deterministic,
        sampling: {
          max_output_tokens: body.settings.maxOutputTokens,
          temperature: body.settings.temperature,
          top_k: body.settings.topK,
          top_p: body.settings.topP,
          min_p: body.settings.minP,
          repetition_penalty: body.settings.repetitionPenalty,
          frequency_penalty: body.settings.frequencyPenalty,
          presence_penalty: body.settings.presencePenalty,
          stop_sequences: body.settings.stopSequences,
          alternatives: body.settings.alternatives,
        },
        attachment_ids: body.attachmentIds,
      }),
    });
    return normalizeGenerationResponse(response);
  },
  async replayRun(id: string): Promise<GenerateResponse> {
    return normalizeGenerationResponse(await request(`/runs/${encodeURIComponent(id)}/replay`, { method: "POST" }));
  },
  async branchRun(id: string, selection: BranchRunRequest): Promise<GenerateResponse> {
    return normalizeGenerationResponse(await request(`/runs/${encodeURIComponent(id)}/branch`, {
      method: "POST",
      body: JSON.stringify({
        token_index: selection.tokenIndex,
        distribution: selection.distribution,
        rank: selection.rank,
        token_id: selection.tokenId,
      }),
    }));
  },
  cancelRun: (id: string): Promise<void> => request(`/runs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  async run(id: string): Promise<RunDetails> {
    return normalizeRun(await request(`/runs/${encodeURIComponent(id)}`));
  },
  async runEvents(id: string): Promise<RunStreamEvent[]> {
    const raw = asRecord(await request(`/runs/${encodeURIComponent(id)}/events`));
    const events = Array.isArray(raw.events) ? raw.events : [];
    return events.map(normalizeStreamEvent).filter((event): event is RunStreamEvent => event !== null);
  },
  upload: async (file: File, modelId: string): Promise<Attachment> => {
    const body = new FormData();
    body.append("file", file);
    body.append("model_id", modelId);
    return normalizeAttachment(await request("/attachments", { method: "POST", body }));
  },
  attachmentContent: (id: string): Promise<Blob> => request<Blob>(
    `/attachments/${encodeURIComponent(id)}/content`,
    { headers: { Accept: "*/*" } },
    "blob",
  ),
  async embeddings(body: { modelId: string; inputs: EmbeddingInput[]; dimensions?: number; normalize: boolean }): Promise<EmbeddingRun> {
    const result = await request("/embeddings", {
      method: "POST",
      body: JSON.stringify({
        model_id: body.modelId,
        inputs: body.inputs.map((input) => ({
          id: input.id,
          modality: input.kind,
          text: input.text,
          attachment_id: input.attachmentIds?.[0],
        })),
        dimensions: body.dimensions,
        normalize: body.normalize,
        persist_vectors: false,
      }),
    });
    return normalizeEmbedding(result, body.inputs, body.dimensions);
  },
};

export function runExportUrl(runId: string, format: "json" | "jsonl" | "csv"): string {
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/export?format=${format}`;
}

export async function fetchRunExport(runId: string, format: "json" | "jsonl" | "csv"): Promise<Blob> {
  const authToken = getSessionAuthToken();
  const response = await fetch(runExportUrl(runId, format), {
    headers: {
      Accept: format === "csv" ? "text/csv" : "application/json",
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    },
  });
  if (!response.ok) {
    if (response.status === 401) notifyAuthenticationRequired();
    let payload: ApiErrorPayload | undefined;
    try {
      payload = (await response.json()) as ApiErrorPayload;
    } catch {
      payload = undefined;
    }
    throw new ApiError(errorMessage(payload, `${response.status} ${response.statusText}`), response.status, payload);
  }
  return response.blob();
}

function websocketBase(): string {
  if (import.meta.env.VITE_WS_BASE) return import.meta.env.VITE_WS_BASE.replace(/\/$/, "");
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

function absoluteWebSocketUrl(value: string): string {
  if (value.startsWith("ws://") || value.startsWith("wss://")) return value;
  if (value.startsWith("http://")) return `ws://${value.slice("http://".length)}`;
  if (value.startsWith("https://")) return `wss://${value.slice("https://".length)}`;
  return `${websocketBase()}${value.startsWith("/") ? value : `/${value}`}`;
}

function runWebSocketUrl(runId: string, explicitUrl: string | undefined, lastSequence: number): string {
  const fallback = `${websocketBase()}/ws/v1/runs/${encodeURIComponent(runId)}`;
  const url = new URL(explicitUrl ? absoluteWebSocketUrl(explicitUrl) : fallback);
  url.searchParams.set("after", String(lastSequence));
  return url.toString();
}

function websocketProtocols(): string[] {
  const protocols = ["lad.events.v1"];
  const authToken = getSessionAuthToken();
  if (!authToken) return protocols;
  const bytes = new TextEncoder().encode(authToken);
  const binary = Array.from(bytes, (byte) => String.fromCharCode(byte)).join("");
  const encoded = window.btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
  protocols.push(`lad.auth.${encoded}`);
  return protocols;
}

export function subscribeToRun(
  runId: string,
  onEvent: (event: RunStreamEvent) => void,
  onConnectionChange: (connected: boolean) => void,
  explicitUrl?: string,
): () => void {
  let closedByClient = false;
  let lastSequence = 0;
  let retries = 0;
  let socket: WebSocket | null = null;
  let retryTimer: number | undefined;
  let waitingForAuth = false;

  const connect = (): void => {
    socket = new WebSocket(runWebSocketUrl(runId, explicitUrl, lastSequence), websocketProtocols());
    socket.addEventListener("open", () => {
      retries = 0;
      onConnectionChange(true);
    });
    socket.addEventListener("message", (message) => {
      try {
        const decoded: unknown = JSON.parse(String(message.data));
        const decodedRecord = asRecord(decoded);
        if (decodedRecord.type === "resync_required") {
          const resumeAfter = asOptionalNumber(asRecord(decodedRecord.payload).resume_after);
          if (resumeAfter !== undefined) lastSequence = resumeAfter;
          onEvent({ version: 1, sequence: lastSequence, type: "warning", message: "The live buffer rolled over; reconnecting from persisted events." });
          socket?.close(1012, "resync required");
          return;
        }
        const event = normalizeStreamEvent(decoded);
        if (!event || event.sequence <= lastSequence) return;
        lastSequence = event.sequence;
        onEvent(event);
        if (event.type === "completed" || event.type === "cancelled" || event.type === "error") {
          closedByClient = true;
          socket?.close(1000, "run complete");
        }
      } catch {
        onEvent({ version: 1, sequence: lastSequence + 1, type: "warning", message: "Ignored a malformed stream event." });
      }
    });
    socket.addEventListener("close", (event) => {
      onConnectionChange(false);
      if (event.code === 4401) {
        waitingForAuth = true;
        notifyAuthenticationRequired();
        return;
      }
      if (closedByClient || retries >= 5) return;
      const delay = Math.min(4000, 250 * 2 ** retries);
      retries += 1;
      retryTimer = window.setTimeout(connect, delay);
    });
    socket.addEventListener("error", () => onConnectionChange(false));
  };

  const handleAuthChanged = (): void => {
    if (closedByClient || !waitingForAuth) return;
    waitingForAuth = false;
    retries = 0;
    connect();
  };

  window.addEventListener(AUTH_CHANGED_EVENT, handleAuthChanged);
  connect();
  return () => {
    closedByClient = true;
    if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    window.removeEventListener(AUTH_CHANGED_EVENT, handleAuthChanged);
    socket?.close(1000, "client closed");
  };
}
