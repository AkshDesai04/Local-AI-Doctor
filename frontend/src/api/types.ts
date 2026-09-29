export type CapabilityState = "full" | "partial" | "unsupported" | "unavailable_on_backend";

export type CapabilityKey =
  | "text_generation"
  | "encoder_decoder_generation"
  | "embeddings"
  | "vision"
  | "audio"
  | "video"
  | "native_file_input"
  | "extracted_text_input"
  | "reasoning_segments"
  | "moe_routing"
  | "raw_logits"
  | "processed_logits"
  | "top_k_alternatives"
  | "prompt_scoring"
  | "attention_capture"
  | "hidden_state_capture"
  | "streaming"
  | "batching"
  | "deterministic_seed"
  | "cpu"
  | "cuda";

export interface Capability {
  state: CapabilityState;
  reason?: string;
  limitations?: string[];
}

export type ModelTask = "text_generation" | "encoder_decoder_generation" | "embedding" | "multimodal_embedding" | "unknown";
export type ModelLifecycle = "unloaded" | "loading" | "loaded" | "unloading" | "error";

export interface ContextLimitCandidate {
  source: string;
  tokens: number | null;
  selected?: boolean;
  note?: string;
}

export interface ModelSummary {
  id: string;
  name: string;
  architecture: string | null;
  task: ModelTask;
  fingerprint: string | null;
  parameterCount?: number | null;
  dtype?: string | null;
  lifecycle: ModelLifecycle;
  loadedDevice?: string | null;
  capabilities: Partial<Record<CapabilityKey, Capability>>;
  effectiveContextLimit: number | null;
  contextLimits?: ContextLimitCandidate[];
  diagnostics?: string[];
  trustRemoteCode?: boolean;
}

export interface ModelInspection {
  tokenizer?: Record<string, unknown>;
  generation?: Record<string, unknown>;
  specialTokens?: Record<string, unknown>;
  chatTemplate?: string | null;
  samplingPipeline?: string[];
}

export interface HealthStatus {
  status: "ok" | "degraded" | "error";
  version?: string;
  database?: string;
  worker?: string;
  loadedModelId?: string;
  loadedDevice?: string;
  selectedBackend?: string;
  reason?: string;
}

export interface ConfigurationSnapshot {
  effective: Record<string, unknown>;
  precedence: string[];
}

export interface ChatSummary {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  pinned: boolean;
  archived: boolean;
  preview?: string;
  messageCount?: number;
  systemPrompt?: string | null;
}

export interface Attachment {
  id: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  kind: "image" | "audio" | "video" | "document" | "unknown";
  status: "pending" | "processing" | "ready" | "rejected" | "failed";
  nativeProcessing: boolean;
  previewUrl?: string;
  detail?: string;
  tokenContribution?: number;
}

export interface Message {
  id: string;
  chatId: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  status: "complete" | "streaming" | "cancelled" | "failed";
  runId?: string;
  branchId?: string;
  parentMessageId?: string;
  reasoningPrimed?: boolean;
  attachments?: Attachment[];
  error?: string;
}

export interface TokenAlternative {
  tokenId: number;
  piece: string;
  probability: number;
  logProbability: number;
  logit: number;
  rank: number;
  survivedFiltering?: boolean;
}

export interface ExpertRoute {
  layer: number;
  selectedExpertIds: number[];
  executedExpertIds: number[];
  gateWeights: number[];
  routerEntropy?: number;
  dropped?: boolean;
}

export interface TokenTiming {
  decodeMs?: number;
  samplingMs?: number;
  emissionMs?: number;
  interTokenMs?: number;
  clientInterArrivalMs?: number;
  transportOverheadMs?: number;
  cumulativeMs?: number;
  instantaneousTps?: number;
  rollingTps?: number;
  clientReceivedAt?: number;
}

export interface ReasoningSlice {
  start: number;
  end: number;
  classification: "reasoning" | "answer" | "unknown";
  delimiter: boolean;
}

export interface AttentionContextToken {
  contextIndex: number;
  tokenId: number;
  piece: string;
  displayText: string;
  sourceKind: "prompt" | "generated";
  generatedTokenIndex?: number;
}

export interface AttentionSourceToken extends AttentionContextToken {
  weight: number;
}

export interface AttentionAttribution {
  method: string;
  aggregation: string;
  semantics: string;
  sourceTokens: AttentionSourceToken[];
  contextTokens?: AttentionContextToken[];
  capturedLayers: number[];
  capturedHeads: number;
  normalized: boolean;
  totalSourceCount: number;
  retainedSourceCount: number;
  retainedWeight: number;
  omittedWeight: number;
}

export interface TokenEvent {
  index: number;
  tokenId: number;
  piece: string;
  bytes?: string;
  displayText: string;
  replaceFrom?: number;
  characterSpan?: [number, number];
  rawLogit?: number;
  rawLogProbability?: number;
  rawProbability?: number;
  rawRank?: number;
  processedLogit?: number;
  samplingLogProbability?: number;
  samplingProbability?: number;
  entropy?: number;
  surprise?: number;
  cumulativeLogProbability?: number;
  runningPerplexity?: number;
  reasoningSegment: "reasoning" | "answer" | "unknown";
  reasoningSlices?: ReasoningSlice[];
  rawAlternatives?: TokenAlternative[];
  samplingAlternatives?: TokenAlternative[];
  timing?: TokenTiming;
  expertRoutes?: ExpertRoute[];
  attentionAttribution?: AttentionAttribution;
}

export interface ContextUsage {
  renderedPromptTokens?: number;
  templateTokens?: number;
  multimodalPositions?: number;
  generatedTokens?: number;
  reservedOutputTokens?: number;
  remainingTokens?: number;
  effectiveLimit: number | null;
  behavior?: "none" | "truncated" | "sliding_window" | "unknown";
}

export interface RunTiming {
  queueMs?: number;
  preprocessingMs?: number;
  templateRenderingMs?: number;
  tokenizationMs?: number;
  prefillMs?: number;
  serverTtftMs?: number;
  engineTtftMs?: number;
  clientTtftMs?: number;
  firstVisibleTextMs?: number;
  promptTokensPerSecond?: number;
  decodeTokensPerSecond?: number;
  endToEndTokensPerSecond?: number;
  totalMs?: number;
  instrumentationOverheadPercent?: number;
}

export interface RunMetrics {
  finishReason?: string;
  responsePerplexity?: number;
  samplerPerplexity?: number;
  promptPerplexity?: number;
  reasoningPerplexity?: number;
  answerPerplexity?: number;
  promptTokens?: number;
  generatedTokens?: number;
  reasoningTokens?: number;
  answerTokens?: number;
  timing?: RunTiming;
  context?: ContextUsage;
  peakRamBytes?: number;
  peakVramBytes?: number;
  kvCacheBytes?: number;
  telemetryOverhead?: string;
}

export interface ReproducibilitySnapshot {
  requestedSeed?: string | number | null;
  effectiveSeed: string;
  rngAlgorithm?: string;
  generatorDevice?: string;
  modelFingerprint?: string;
  tokenizerFingerprint?: string;
  backend?: string;
  device?: string;
  deviceReason?: string;
  dtype?: string;
  quantization?: string | null;
  attentionImplementation?: string;
  softwareVersions?: Record<string, string>;
  deterministicKernels?: boolean;
  batching?: string;
}

export interface RunDetails {
  id: string;
  chatId?: string;
  messageId?: string;
  modelId: string;
  status: "queued" | "loading" | "running" | "complete" | "cancelled" | "failed";
  createdAt: string;
  completedAt?: string;
  tokens: TokenEvent[];
  branchableThroughTokenIndex?: number;
  metrics?: RunMetrics;
  reproducibility?: ReproducibilitySnapshot;
  effectiveSettings?: Record<string, unknown>;
  samplingPipeline?: string[];
  warnings?: string[];
  rawEvents?: RunStreamEvent[];
  error?: string;
}

export interface GenerationSettings {
  reasoning?: boolean;
  device: "auto" | "cpu" | "cuda";
  dtype: "auto" | "float32" | "float16" | "bfloat16";
  instrumentation: "off" | "basic" | "token" | "full" | "expert";
  seed: string;
  maxOutputTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  repetitionPenalty: number;
  frequencyPenalty: number;
  presencePenalty: number;
  alternatives: number;
  stopSequences: string[];
  deterministic: boolean;
}

export interface GenerateRequest {
  chatId: string;
  modelId: string;
  content: string;
  attachmentIds: string[];
  settings: GenerationSettings;
  parentMessageId?: string | null;
}

export interface GenerateResponse {
  chatId?: string;
  runId: string;
  messageId?: string;
  userMessageId?: string;
  modelId?: string;
  parentRunId?: string;
  sourceRunId?: string;
  websocketUrl?: string;
  run?: RunDetails;
}

export type AlternativeDistribution = "raw" | "sampling";

export interface BranchRunRequest {
  tokenIndex: number;
  distribution: AlternativeDistribution;
  rank: number;
  tokenId: number;
}

/** The untouched JSON object received from the run-event API or websocket. */
export type RawRunEventEnvelope = Readonly<Record<string, unknown>>;

type NormalizedRunStreamEvent =
  | { version: 1; sequence: number; type: "run.created"; run: Partial<RunDetails> }
  | { version: 1; sequence: number; type: "stage.changed"; stage: RunDetails["status"]; detail?: string; metrics?: Partial<RunMetrics> }
  | { version: 1; sequence: number; type: "token"; token: TokenEvent }
  | { version: 1; sequence: number; type: "metrics"; metrics: Partial<RunMetrics>; reproducibility?: Partial<ReproducibilitySnapshot>; samplingPipeline?: string[] }
  | { version: 1; sequence: number; type: "warning"; message: string }
  | { version: 1; sequence: number; type: "completed"; run?: Partial<RunDetails> }
  | { version: 1; sequence: number; type: "cancelled"; reason?: string; metrics?: Partial<RunMetrics> }
  | { version: 1; sequence: number; type: "error"; code?: string; message: string };

export type RunStreamEvent = NormalizedRunStreamEvent & {
  /** Complete server envelope retained for lossless raw-event inspection. */
  raw?: RawRunEventEnvelope;
};

export interface EmbeddingInput {
  id: string;
  kind: "text" | "image" | "video" | "audio" | "mixed";
  text?: string;
  attachmentIds?: string[];
  label?: string;
}

export interface EmbeddingVector {
  inputId: string;
  values: number[];
  dimensions: number;
  dtype: string;
  l2Norm: number;
  minimum?: number;
  maximum?: number;
  mean?: number;
  standardDeviation?: number;
}

export interface EmbeddingRun {
  id: string;
  modelId: string;
  status: "queued" | "running" | "complete" | "failed";
  inputs: EmbeddingInput[];
  vectors: EmbeddingVector[];
  requestedDimensions?: number;
  outputDimensions: number;
  normalized: boolean;
  pooling?: string;
  jointSpace?: boolean;
  truncation?: string;
  preprocessingMs?: number;
  forwardMs?: number;
  totalMs?: number;
  tokensPerSecond?: number;
  itemsPerSecond?: number;
  similarityMatrix?: number[][];
  projection?: Array<{ inputId: string; x: number; y: number }>;
  error?: string;
}

export interface ApiErrorPayload {
  code?: string;
  message?: string;
  detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
  requestId?: string;
  error?: {
    code?: string;
    message?: string;
    hint?: string;
    request_id?: string;
  };
}
