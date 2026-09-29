import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AUTH_CHANGED_EVENT } from "../api/auth";
import { api, ApiError, subscribeToRun } from "../api/client";
import type {
  AlternativeDistribution,
  Attachment,
  ChatSummary,
  ConfigurationSnapshot,
  GenerateRequest,
  GenerationSettings,
  HealthStatus,
  Message,
  ModelSummary,
  RunDetails,
  RunMetrics,
  RunStreamEvent,
  TokenAlternative,
} from "../api/types";
import { isUsable } from "../domain/capabilities";
import { cleanAssistantOutput } from "../utils/markdown";

export const defaultGenerationSettings: GenerationSettings = {
  reasoning: true,
  device: "auto",
  dtype: "auto",
  instrumentation: "token",
  seed: "",
  maxOutputTokens: 512,
  temperature: 0.7,
  topK: 50,
  topP: 0.95,
  minP: 0,
  repetitionPenalty: 1,
  frequencyPenalty: 0,
  presencePenalty: 0,
  alternatives: 10,
  stopSequences: [],
  deterministic: false,
};

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function configuredNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function generationSettingsFromConfiguration(
  configuration: ConfigurationSnapshot,
): GenerationSettings {
  const runtime = asRecord(configuration.effective.runtime);
  const inference = asRecord(configuration.effective.inference);
  const sampling = asRecord(inference.defaults);
  const device = runtime.device;
  const dtype = runtime.dtype;
  const instrumentation = inference.instrumentation;
  return {
    ...defaultGenerationSettings,
    reasoning: inference.reasoning !== false,
    device: device === "cpu" || device === "cuda" ? device : "auto",
    dtype: dtype === "float32" || dtype === "float16" || dtype === "bfloat16" ? dtype : "auto",
    instrumentation: instrumentation === "off" || instrumentation === "basic"
      || instrumentation === "full" || instrumentation === "expert"
      ? instrumentation
      : "token",
    maxOutputTokens: configuredNumber(sampling.max_output_tokens, defaultGenerationSettings.maxOutputTokens),
    temperature: configuredNumber(sampling.temperature, defaultGenerationSettings.temperature),
    topK: configuredNumber(sampling.top_k, defaultGenerationSettings.topK),
    topP: configuredNumber(sampling.top_p, defaultGenerationSettings.topP),
    minP: configuredNumber(sampling.min_p, defaultGenerationSettings.minP),
    repetitionPenalty: configuredNumber(sampling.repetition_penalty, defaultGenerationSettings.repetitionPenalty),
    frequencyPenalty: configuredNumber(sampling.frequency_penalty, defaultGenerationSettings.frequencyPenalty),
    presencePenalty: configuredNumber(sampling.presence_penalty, defaultGenerationSettings.presencePenalty),
    alternatives: configuredNumber(sampling.alternatives, defaultGenerationSettings.alternatives),
    deterministic: inference.deterministic_reference_mode === true,
  };
}

function readableError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.name === "TypeError") return "The local backend is not reachable.";
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}

function reflectLoadedModel(models: ModelSummary[], health: HealthStatus | null): ModelSummary[] {
  if (!health?.loadedModelId) return models;
  return models.map((model) => model.id === health.loadedModelId
    ? { ...model, lifecycle: "loaded", loadedDevice: health.loadedDevice ?? model.loadedDevice }
    : { ...model, lifecycle: "unloaded", loadedDevice: null });
}

function mergeMetrics(current: RunMetrics | undefined, update: Partial<RunMetrics>): RunMetrics {
  return {
    finishReason: update.finishReason ?? current?.finishReason,
    responsePerplexity: update.responsePerplexity ?? current?.responsePerplexity,
    samplerPerplexity: update.samplerPerplexity ?? current?.samplerPerplexity,
    promptPerplexity: update.promptPerplexity ?? current?.promptPerplexity,
    reasoningPerplexity: update.reasoningPerplexity ?? current?.reasoningPerplexity,
    answerPerplexity: update.answerPerplexity ?? current?.answerPerplexity,
    promptTokens: update.promptTokens ?? current?.promptTokens,
    generatedTokens: update.generatedTokens ?? current?.generatedTokens,
    reasoningTokens: update.reasoningTokens ?? current?.reasoningTokens,
    answerTokens: update.answerTokens ?? current?.answerTokens,
    peakRamBytes: update.peakRamBytes ?? current?.peakRamBytes,
    peakVramBytes: update.peakVramBytes ?? current?.peakVramBytes,
    kvCacheBytes: update.kvCacheBytes ?? current?.kvCacheBytes,
    telemetryOverhead: update.telemetryOverhead ?? current?.telemetryOverhead,
    timing: {
      queueMs: update.timing?.queueMs ?? current?.timing?.queueMs,
      preprocessingMs: update.timing?.preprocessingMs ?? current?.timing?.preprocessingMs,
      templateRenderingMs: update.timing?.templateRenderingMs ?? current?.timing?.templateRenderingMs,
      tokenizationMs: update.timing?.tokenizationMs ?? current?.timing?.tokenizationMs,
      prefillMs: update.timing?.prefillMs ?? current?.timing?.prefillMs,
      serverTtftMs: update.timing?.serverTtftMs ?? current?.timing?.serverTtftMs,
      engineTtftMs: update.timing?.engineTtftMs ?? current?.timing?.engineTtftMs,
      clientTtftMs: update.timing?.clientTtftMs ?? current?.timing?.clientTtftMs,
      firstVisibleTextMs: update.timing?.firstVisibleTextMs ?? current?.timing?.firstVisibleTextMs,
      promptTokensPerSecond: update.timing?.promptTokensPerSecond ?? current?.timing?.promptTokensPerSecond,
      decodeTokensPerSecond: update.timing?.decodeTokensPerSecond ?? current?.timing?.decodeTokensPerSecond,
      endToEndTokensPerSecond: update.timing?.endToEndTokensPerSecond ?? current?.timing?.endToEndTokensPerSecond,
      totalMs: update.timing?.totalMs ?? current?.timing?.totalMs,
      instrumentationOverheadPercent: update.timing?.instrumentationOverheadPercent ?? current?.timing?.instrumentationOverheadPercent,
    },
    context: update.context || current?.context ? {
      renderedPromptTokens: update.context?.renderedPromptTokens ?? current?.context?.renderedPromptTokens,
      templateTokens: update.context?.templateTokens ?? current?.context?.templateTokens,
      multimodalPositions: update.context?.multimodalPositions ?? current?.context?.multimodalPositions,
      generatedTokens: update.context?.generatedTokens ?? current?.context?.generatedTokens,
      reservedOutputTokens: update.context?.reservedOutputTokens ?? current?.context?.reservedOutputTokens,
      remainingTokens: update.context?.remainingTokens ?? current?.context?.remainingTokens,
      effectiveLimit: update.context?.effectiveLimit ?? current?.context?.effectiveLimit ?? null,
      behavior: update.context?.behavior ?? current?.context?.behavior,
    } : undefined,
  };
}

const CLIENT_TELEMETRY_STORAGE_KEY = "local-ai-doctor.client-telemetry.v1";

interface PersistedClientTelemetry {
  storedAt: string;
  clientTtftMs?: number;
  firstVisibleTextMs?: number;
  tokenTimings: Record<string, { clientInterArrivalMs?: number; transportOverheadMs?: number }>;
}

function readClientTelemetry(): Record<string, PersistedClientTelemetry> {
  try {
    const decoded: unknown = JSON.parse(window.localStorage.getItem(CLIENT_TELEMETRY_STORAGE_KEY) ?? "{}");
    return asRecord(decoded) as Record<string, PersistedClientTelemetry>;
  } catch {
    return {};
  }
}

function persistClientTelemetry(run: RunDetails): void {
  const tokenTimings = Object.fromEntries(run.tokens.flatMap((token) => {
    const timing = token.timing;
    if (timing?.clientInterArrivalMs === undefined && timing?.transportOverheadMs === undefined) return [];
    return [[String(token.index), {
      clientInterArrivalMs: timing.clientInterArrivalMs,
      transportOverheadMs: timing.transportOverheadMs,
    }]];
  }));
  const clientTtftMs = run.metrics?.timing?.clientTtftMs;
  const firstVisibleTextMs = run.metrics?.timing?.firstVisibleTextMs;
  if (clientTtftMs === undefined && firstVisibleTextMs === undefined && !Object.keys(tokenTimings).length) return;
  try {
    const entries = Object.entries({
      ...readClientTelemetry(),
      [run.id]: { storedAt: new Date().toISOString(), clientTtftMs, firstVisibleTextMs, tokenTimings },
    }).sort(([, left], [, right]) => left.storedAt.localeCompare(right.storedAt)).slice(-100);
    window.localStorage.setItem(CLIENT_TELEMETRY_STORAGE_KEY, JSON.stringify(Object.fromEntries(entries)));
  } catch {
    // Client timing is supplemental; storage limits must never break the workbench.
  }
}

function restoreClientTelemetry(run: RunDetails): RunDetails {
  const cached = readClientTelemetry()[run.id];
  if (!cached) return run;
  return {
    ...run,
    metrics: mergeMetrics(run.metrics, { timing: {
      clientTtftMs: cached.clientTtftMs,
      firstVisibleTextMs: cached.firstVisibleTextMs,
    } }),
    tokens: run.tokens.map((token) => {
      const timing = cached.tokenTimings[String(token.index)];
      return timing ? { ...token, timing: { ...token.timing, ...timing } } : token;
    }),
  };
}

function mergeRefreshedRun(persisted: RunDetails, live: RunDetails): RunDetails {
  const liveTokens = new Map(live.tokens.map((token) => [token.index, token]));
  const finalizedMetrics = mergeMetrics(live.metrics, persisted.metrics ?? {});
  return {
    ...persisted,
    metrics: {
      ...finalizedMetrics,
      timing: {
        ...finalizedMetrics.timing,
        clientTtftMs: live.metrics?.timing?.clientTtftMs ?? finalizedMetrics.timing?.clientTtftMs,
        firstVisibleTextMs: live.metrics?.timing?.firstVisibleTextMs ?? finalizedMetrics.timing?.firstVisibleTextMs,
      },
    },
    tokens: persisted.tokens.map((token) => {
      const liveToken = liveTokens.get(token.index);
      return liveToken ? { ...token, timing: { ...token.timing, ...liveToken.timing } } : token;
    }),
    rawEvents: live.rawEvents ?? persisted.rawEvents,
    warnings: Array.from(new Set([...(persisted.warnings ?? []), ...(live.warnings ?? [])])),
  };
}

function appendTokenContent(current: Message[], messageId: string, text: string, replaceFrom?: number): Message[] {
  return current.map((message) =>
    message.id === messageId ? { ...message, content: replaceFrom === undefined ? `${message.content}${text}` : `${message.content.slice(0, replaceFrom)}${text}`, status: "streaming" } : message,
  );
}

interface StreamVisibilityBuffer {
  rawText: string;
  reasoningPrimed: boolean;
}

function consumeVisibleAnswerText(
  current: StreamVisibilityBuffer | undefined,
  token: RunStreamEvent & { type: "token" },
): { buffer: StreamVisibilityBuffer; visible: boolean } {
  const previous = current ?? { rawText: "", reasoningPrimed: false };
  const rawText = token.token.replaceFrom === undefined
    ? `${previous.rawText}${token.token.displayText}`
    : `${previous.rawText.slice(0, token.token.replaceFrom)}${token.token.displayText}`;
  const reasoningPrimed = previous.reasoningPrimed || token.token.reasoningSegment === "reasoning";
  const buffer = { rawText, reasoningPrimed };
  if (token.token.reasoningSegment === "reasoning") return { buffer, visible: false };
  const visibleText = token.token.reasoningSegment === "answer"
    ? cleanAssistantOutput(token.token.displayText).trim()
    : cleanAssistantOutput(rawText, reasoningPrimed).trim();
  return { buffer, visible: visibleText.length > 0 };
}

export function resolveGenerationParent(messages: Message[], requestedParent: string | null | undefined): string | null | undefined {
  if (requestedParent !== undefined) return requestedParent;
  return [...messages].reverse().find((message) => message.role === "assistant" && message.status !== "streaming")?.id;
}

export interface WorkbenchState {
  health: HealthStatus | null;
  configuration: ConfigurationSnapshot | null;
  connected: boolean;
  booting: boolean;
  models: ModelSummary[];
  chats: ChatSummary[];
  archivedChats: ChatSummary[];
  activeChatId: string | null;
  messages: Message[];
  messagesLoading: boolean;
  selectedModelId: string;
  selectedModel: ModelSummary | null;
  selectedRun: RunDetails | null;
  runningRunId: string | null;
  branching: boolean;
  streamConnected: boolean;
  settings: GenerationSettings;
  defaultSettings: GenerationSettings;
  attachments: Attachment[];
  systemPrompt: string;
  error: string | null;
  notice: string | null;
  setSystemPrompt: (value: string) => void;
  setSelectedModelId: (id: string) => void;
  setSettings: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  setError: (message: string | null) => void;
  selectChat: (id: string) => Promise<void>;
  createChat: () => Promise<void>;
  renameChat: (id: string, title: string) => Promise<void>;
  togglePin: (chat: ChatSummary) => Promise<void>;
  toggleArchive: (chat: ChatSummary) => Promise<void>;
  deleteChat: (id: string) => Promise<void>;
  clearChats: () => Promise<void>;
  refreshModels: () => Promise<void>;
  synchronizeRuntimeState: () => Promise<void>;
  toggleModelLoaded: (model: ModelSummary) => Promise<void>;
  submit: (content: string, parentMessageId?: string | null) => Promise<void>;
  stop: () => Promise<void>;
  inspectRun: (id: string) => Promise<void>;
  loadSelectedRunEvents: () => Promise<void>;
  branchFromAlternative: (tokenIndex: number, distribution: AlternativeDistribution, alternative: TokenAlternative) => Promise<void>;
  addAttachment: (file: File) => Promise<void>;
  removeAttachment: (id: string) => void;
  retryMessage: (message: Message) => Promise<void>;
}

export function useWorkbench(): WorkbenchState {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [configuration, setConfiguration] = useState<ConfigurationSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [booting, setBooting] = useState(true);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [archivedChats, setArchivedChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedRun, setSelectedRun] = useState<RunDetails | null>(null);
  const [runningRunId, setRunningRunId] = useState<string | null>(null);
  const [branching, setBranching] = useState(false);
  const [streamConnected, setStreamConnected] = useState(false);
  const [settings, setSettings] = useState<GenerationSettings>(defaultGenerationSettings);
  const [configuredDefaults, setConfiguredDefaults] = useState<GenerationSettings>(defaultGenerationSettings);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [authRevision, setAuthRevision] = useState(0);
  const [systemPromptDraft, setSystemPromptDraft] = useState<{ chatId: string | null; value: string } | null>(null);
  const streamCleanup = useRef<(() => void) | null>(null);
  const branchPending = useRef(false);
  const activeChatIdRef = useRef<string | null>(null);
  const runSelectionEpoch = useRef(0);
  const locallyCreatedChat = useRef<string | null>(null);
  const clientRequestStarted = useRef<number | null>(null);
  const lastClientReceipt = useRef<number | null>(null);
  const streamVisibility = useRef(new Map<string, StreamVisibilityBuffer>());
  const systemPromptTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const systemPromptPending = useRef<{ chatId: string; value: string } | null>(null);
  const systemPromptSave = useRef<Promise<void>>(Promise.resolve());

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? (selectedModelId ? null : models[0] ?? null),
    [models, selectedModelId],
  );

  // An unsaved edit wins over the stored value; a brand-new chat keeps its draft locally
  // (chatId null) and sends it with the create request.
  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? archivedChats.find((chat) => chat.id === activeChatId) ?? null,
    [activeChatId, archivedChats, chats],
  );
  const systemPrompt = systemPromptDraft?.chatId === activeChatId ? systemPromptDraft.value : activeChat?.systemPrompt ?? "";

  const flushSystemPrompt = useCallback((): Promise<void> => {
    if (systemPromptTimer.current) clearTimeout(systemPromptTimer.current);
    systemPromptTimer.current = null;
    const pending = systemPromptPending.current;
    systemPromptPending.current = null;
    if (!pending) return systemPromptSave.current;
    systemPromptSave.current = systemPromptSave.current.then(async () => {
      try {
        const updated = await api.updateChat(pending.chatId, { systemPrompt: pending.value.trim() ? pending.value : null });
        const store = (chat: ChatSummary): ChatSummary => chat.id === updated.id ? { ...chat, systemPrompt: updated.systemPrompt } : chat;
        setChats((current) => current.map(store));
        setArchivedChats((current) => current.map(store));
        setSystemPromptDraft((current) => current?.chatId === pending.chatId && current.value === pending.value ? null : current);
      } catch (cause) {
        if (!(cause instanceof ApiError && cause.status === 404)) setError(readableError(cause));
      }
    });
    return systemPromptSave.current;
  }, []);

  const setSystemPrompt = useCallback((value: string): void => {
    const chatId = activeChatIdRef.current;
    setSystemPromptDraft({ chatId, value });
    if (chatId === null) return;
    systemPromptPending.current = { chatId, value };
    if (systemPromptTimer.current) clearTimeout(systemPromptTimer.current);
    systemPromptTimer.current = setTimeout(() => void flushSystemPrompt(), 500);
  }, [flushSystemPrompt]);

  useEffect(() => {
    const refreshAfterAuthentication = (): void => setAuthRevision((current) => current + 1);
    window.addEventListener(AUTH_CHANGED_EVENT, refreshAfterAuthentication);
    return () => window.removeEventListener(AUTH_CHANGED_EVENT, refreshAfterAuthentication);
  }, []);

  useEffect(() => {
    let alive = true;
    const bootstrap = async (): Promise<void> => {
      setBooting(true);
      const [healthResult, modelsResult, chatsResult, archivedResult, configurationResult] = await Promise.allSettled([
        api.health(),
        api.models(),
        api.chats(false),
        api.chats(true),
        api.configuration(),
      ]);
      if (!alive) return;
      if (healthResult.status === "fulfilled") {
        setHealth(healthResult.value);
        setConnected(healthResult.value.status !== "error");
      }
      if (modelsResult.status === "fulfilled") {
        const backendHealth = healthResult.status === "fulfilled" ? healthResult.value : null;
        const discovered = reflectLoadedModel(modelsResult.value, backendHealth);
        setModels(discovered);
        setSelectedModelId((current) => current || discovered.find((model) => model.lifecycle === "loaded")?.id || discovered[0]?.id || "");
      }
      if (chatsResult.status === "fulfilled") {
        setChats(chatsResult.value);
        setActiveChatId((current) => current ?? chatsResult.value[0]?.id ?? null);
      }
      if (archivedResult.status === "fulfilled") setArchivedChats(archivedResult.value);
      if (configurationResult.status === "fulfilled") {
        const defaults = generationSettingsFromConfiguration(configurationResult.value);
        setConfiguration(configurationResult.value);
        setConfiguredDefaults(defaults);
        setSettings(defaults);
      }
      const failures = [healthResult, modelsResult, chatsResult].filter((result) => result.status === "rejected");
      const unauthorized = [healthResult, modelsResult, chatsResult, archivedResult, configurationResult].some(
        (result) => result.status === "rejected" && result.reason instanceof ApiError && result.reason.status === 401,
      );
      if (healthResult.status !== "fulfilled" && (modelsResult.status === "fulfilled" || chatsResult.status === "fulfilled")) setConnected(true);
      if (unauthorized) {
        setConnected(false);
        setError("Authentication is required. Enter the local backend token in Generation controls.");
      } else if (failures.length === 3) {
        setConnected(false);
        setError("The local backend is offline. Start it to discover models and load saved chats.");
      } else {
        setError(null);
      }
      setBooting(false);
    };
    void bootstrap();
    return () => {
      alive = false;
    };
  }, [authRevision]);

  useEffect(() => () => streamCleanup.current?.(), []);

  useEffect(() => {
    activeChatIdRef.current = activeChatId;
  }, [activeChatId]);

  useEffect(() => {
    if (selectedRun && ["complete", "cancelled", "failed"].includes(selectedRun.status)) {
      persistClientTelemetry(selectedRun);
    }
  }, [selectedRun]);

  useEffect(() => {
    const restoreEpoch = ++runSelectionEpoch.current;
    if (!activeChatId) {
      setMessages([]);
      setSelectedRun(null);
      return;
    }
    if (locallyCreatedChat.current === activeChatId) {
      setMessagesLoading(false);
      return;
    }
    let alive = true;
    setMessagesLoading(true);
    api.messages(activeChatId)
      .then(async (value) => {
        if (!alive || runSelectionEpoch.current !== restoreEpoch) return;
        setMessages(value);
        const latestRunId = [...value].reverse().find((message) => message.role === "assistant" && message.runId)?.runId;
        if (!latestRunId) {
          setSelectedRun(null);
          return;
        }
        try {
          const run = restoreClientTelemetry(await api.run(latestRunId));
          if (alive && runSelectionEpoch.current === restoreEpoch && activeChatIdRef.current === activeChatId) {
            setSelectedModelId(run.modelId);
            setSelectedRun(run);
          }
        } catch (cause) {
          if (alive && runSelectionEpoch.current === restoreEpoch) setError(readableError(cause));
        }
      })
      .catch((cause: unknown) => {
        if (alive) setError(readableError(cause));
      })
      .finally(() => {
        if (alive) setMessagesLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [activeChatId]);

  const selectChat = useCallback((id: string): Promise<void> => {
    void flushSystemPrompt();
    runSelectionEpoch.current += 1;
    locallyCreatedChat.current = null;
    activeChatIdRef.current = id;
    setActiveChatId(id);
    setSelectedRun(null);
    return Promise.resolve();
  }, [flushSystemPrompt]);

  const createChat = useCallback(async (): Promise<void> => {
    void flushSystemPrompt();
    runSelectionEpoch.current += 1;
    locallyCreatedChat.current = null;
    try {
      const chat = await api.createChat();
      setChats((current) => [chat, ...current]);
      activeChatIdRef.current = chat.id;
      setActiveChatId(chat.id);
      setMessages([]);
      setSelectedRun(null);
      setConnected(true);
    } catch (cause) {
      setError(readableError(cause));
    }
  }, [flushSystemPrompt]);

  const updateChat = useCallback(async (id: string, changes: Partial<Pick<ChatSummary, "title" | "pinned" | "archived">>): Promise<void> => {
    try {
      const updated = await api.updateChat(id, changes);
      if (updated.archived) {
        setChats((current) => current.filter((chat) => chat.id !== id));
        setArchivedChats((current) => [updated, ...current.filter((chat) => chat.id !== id)]);
        if (activeChatIdRef.current === id) {
          runSelectionEpoch.current += 1;
          activeChatIdRef.current = null;
          locallyCreatedChat.current = null;
        }
        setActiveChatId((current) => (current === id ? null : current));
      } else {
        setArchivedChats((current) => current.filter((chat) => chat.id !== id));
        setChats((current) => [updated, ...current.filter((chat) => chat.id !== id)]);
      }
    } catch (cause) {
      setError(readableError(cause));
    }
  }, []);

  const renameChat = useCallback(async (id: string, title: string): Promise<void> => {
    if (!title.trim()) return;
    await updateChat(id, { title: title.trim() });
  }, [updateChat]);

  const togglePin = useCallback(async (chat: ChatSummary): Promise<void> => {
    await updateChat(chat.id, { pinned: !chat.pinned });
  }, [updateChat]);

  const toggleArchive = useCallback(async (chat: ChatSummary): Promise<void> => {
    await updateChat(chat.id, { archived: !chat.archived });
  }, [updateChat]);

  const deleteChat = useCallback(async (id: string): Promise<void> => {
    try {
      await api.deleteChat(id);
      setChats((current) => current.filter((chat) => chat.id !== id));
      setArchivedChats((current) => current.filter((chat) => chat.id !== id));
      if (activeChatIdRef.current === id) {
        runSelectionEpoch.current += 1;
        activeChatIdRef.current = null;
        locallyCreatedChat.current = null;
      }
      setActiveChatId((current) => (current === id ? null : current));
    } catch (cause) {
      setError(readableError(cause));
    }
  }, []);

  const clearChats = useCallback(async (): Promise<void> => {
    try {
      await api.clearChats();
      setChats([]);
      setArchivedChats([]);
      runSelectionEpoch.current += 1;
      activeChatIdRef.current = null;
      locallyCreatedChat.current = null;
      setActiveChatId(null);
      setMessages([]);
    } catch (cause) {
      setError(readableError(cause));
    }
  }, []);

  const synchronizeRuntimeState = useCallback(async (): Promise<void> => {
    try {
      const [discovered, backendHealth] = await Promise.all([api.models(), api.health()]);
      const synchronized = reflectLoadedModel(discovered, backendHealth);
      setHealth(backendHealth);
      setModels(synchronized);
      setSelectedModelId((current) => synchronized.some((model) => model.id === current)
        ? current
        : synchronized.find((model) => model.lifecycle === "loaded")?.id || synchronized[0]?.id || "");
      setConnected(backendHealth.status !== "error");
    } catch {
      // Preserve the completed inference and the last coherent runtime snapshot
      // when the follow-up GETs are temporarily unavailable.
    }
  }, []);

  const refreshModels = useCallback(async (): Promise<void> => {
    setNotice("Scanning configured model roots…");
    try {
      const [scanned, backendHealth] = await Promise.all([api.refreshModels(), api.health()]);
      const discovered = reflectLoadedModel(scanned, backendHealth);
      setHealth(backendHealth);
      setModels(discovered);
      setSelectedModelId((current) => discovered.some((model) => model.id === current)
        ? current
        : discovered.find((model) => model.lifecycle === "loaded")?.id || discovered[0]?.id || "");
      setNotice(`Model scan complete: ${String(discovered.length)} found.`);
      setConnected(true);
    } catch (cause) {
      setError(readableError(cause));
      setNotice(null);
    }
  }, []);

  const toggleModelLoaded = useCallback(async (model: ModelSummary): Promise<void> => {
    setModels((current) => current.map((item) => item.id === model.id ? { ...item, lifecycle: model.lifecycle === "loaded" ? "unloading" : "loading" } : item));
    try {
      const updated = model.lifecycle === "loaded"
        ? await api.unloadModel(model.id, model)
        : await api.loadModel(model.id, { device: settings.device, dtype: settings.dtype }, model);
      setModels((current) => current.map((item) => item.id === updated.id
        ? updated
        : updated.lifecycle === "loaded" ? { ...item, lifecycle: "unloaded", loadedDevice: null } : item));
    } catch (cause) {
      setModels((current) => current.map((item) => item.id === model.id ? { ...item, lifecycle: "error" } : item));
      setError(readableError(cause));
    }
  }, [settings.device, settings.dtype]);

  const handleStreamEvent = useCallback((event: RunStreamEvent, runId: string, assistantId: string, chatId: string): void => {
    const receipt = performance.now();
    const resolvedEvent: RunStreamEvent = event.type === "token" ? {
      ...event,
      token: {
        ...event.token,
        timing: {
          ...event.token.timing,
          clientReceivedAt: receipt,
          clientInterArrivalMs: lastClientReceipt.current === null ? undefined : receipt - lastClientReceipt.current,
        },
      },
    } : event;
    if (event.type === "token") lastClientReceipt.current = receipt;
    const visibleAnswer = resolvedEvent.type === "token"
      ? consumeVisibleAnswerText(streamVisibility.current.get(runId), resolvedEvent)
      : null;
    if (visibleAnswer) streamVisibility.current.set(runId, visibleAnswer.buffer);
    setSelectedRun((current) => {
      if (!current || current.id !== runId) return current;
      const rawEvents = [...(current.rawEvents ?? []), resolvedEvent];
      switch (resolvedEvent.type) {
        case "run.created":
          return { ...current, ...resolvedEvent.run, id: runId, rawEvents };
        case "stage.changed":
          return { ...current, status: resolvedEvent.stage, metrics: resolvedEvent.metrics ? mergeMetrics(current.metrics, resolvedEvent.metrics) : current.metrics, rawEvents };
        case "token":
          {
            const timing: NonNullable<RunMetrics["timing"]> = {};
            if (clientRequestStarted.current !== null) {
              if (current.tokens.length === 0 && current.metrics?.timing?.clientTtftMs === undefined) {
                timing.clientTtftMs = receipt - clientRequestStarted.current;
              }
              if (current.metrics?.timing?.firstVisibleTextMs === undefined && visibleAnswer?.visible === true) {
                timing.firstVisibleTextMs = receipt - clientRequestStarted.current;
              }
            }
            const tokens = [...current.tokens, resolvedEvent.token];
            const generatedTokens = Math.max(
              current.metrics?.generatedTokens ?? 0,
              current.metrics?.context?.generatedTokens ?? 0,
              tokens.length,
              resolvedEvent.token.index + 1,
            );
            const liveTokensPerSecond = resolvedEvent.token.timing?.rollingTps;
            return {
              ...current,
              status: current.status === "complete" || current.status === "cancelled" || current.status === "failed" ? current.status : "running",
              tokens,
              metrics: mergeMetrics(current.metrics, {
                generatedTokens,
                responsePerplexity: resolvedEvent.token.runningPerplexity,
                timing: {
                  ...timing,
                  decodeTokensPerSecond: liveTokensPerSecond,
                },
                context: {
                  ...current.metrics?.context,
                  generatedTokens,
                  effectiveLimit: current.metrics?.context?.effectiveLimit ?? null,
                },
              }),
              rawEvents,
            };
          }
        case "metrics":
          return {
            ...current,
            metrics: mergeMetrics(current.metrics, resolvedEvent.metrics),
            reproducibility: resolvedEvent.reproducibility ? { ...current.reproducibility, ...resolvedEvent.reproducibility, effectiveSeed: resolvedEvent.reproducibility.effectiveSeed ?? current.reproducibility?.effectiveSeed ?? "not reported" } : current.reproducibility,
            samplingPipeline: resolvedEvent.samplingPipeline ?? current.samplingPipeline,
            rawEvents,
          };
        case "warning":
          return { ...current, warnings: [...(current.warnings ?? []), resolvedEvent.message], rawEvents };
        case "completed":
          return { ...current, ...resolvedEvent.run, id: runId, status: "complete", metrics: resolvedEvent.run?.metrics ? mergeMetrics(current.metrics, resolvedEvent.run.metrics) : current.metrics, rawEvents };
        case "cancelled":
          return { ...current, status: "cancelled", metrics: resolvedEvent.metrics ? mergeMetrics(current.metrics, resolvedEvent.metrics) : current.metrics, rawEvents };
        case "error":
          return { ...current, status: "failed", error: resolvedEvent.message, rawEvents };
      }
    });

    if (resolvedEvent.type === "token") setMessages((current) => appendTokenContent(current, assistantId, resolvedEvent.token.displayText, resolvedEvent.token.replaceFrom));
    if (resolvedEvent.type === "completed" || resolvedEvent.type === "cancelled" || resolvedEvent.type === "error") {
      const status = resolvedEvent.type === "completed" ? "complete" : resolvedEvent.type === "cancelled" ? "cancelled" : "failed";
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, status, error: resolvedEvent.type === "error" ? resolvedEvent.message : undefined } : message));
      setRunningRunId(null);
      setStreamConnected(false);
      clientRequestStarted.current = null;
      lastClientReceipt.current = null;
      streamVisibility.current.delete(runId);
      locallyCreatedChat.current = null;
      void synchronizeRuntimeState();
      void api.run(runId).then((persistedRun) => {
        const restored = restoreClientTelemetry(persistedRun);
        setSelectedRun((current) => current?.id === runId
          ? mergeRefreshedRun(restored, current)
          : current);
      }).catch(() => {
        // Live telemetry remains available if the durable snapshot cannot be refreshed yet.
      });
      void Promise.all([api.messages(chatId), api.chats(false)]).then(([persistedMessages, persistedChats]) => {
        if (activeChatIdRef.current === chatId) {
          setMessages(persistedMessages.map((message) => message.id === assistantId ? { ...message, runId: message.runId ?? runId } : message));
        }
        setChats(persistedChats);
      }).catch(() => {
        // The completed stream remains usable even if the persistence refresh is temporarily unavailable.
      });
    }
  }, [synchronizeRuntimeState]);

  const submit = useCallback(async (content: string, parentMessageId?: string | null): Promise<void> => {
    if (!content.trim() || !selectedModel || runningRunId || branchPending.current) return;
    runSelectionEpoch.current += 1;
    setError(null);
    let chatId = activeChatId;
    let temporaryUserId: string | null = null;
    try {
      if (chatId) {
        await flushSystemPrompt();
      } else {
        const chat = await api.createChat(systemPrompt);
        chatId = chat.id;
        setSystemPromptDraft(null);
        setChats((current) => [chat, ...current]);
        locallyCreatedChat.current = chat.id;
        activeChatIdRef.current = chat.id;
        setActiveChatId(chat.id);
      }
      const resolvedChatId = chatId;
      const resolvedParentMessageId = resolveGenerationParent(messages, parentMessageId);
      temporaryUserId = `pending-user-${String(Date.now())}`;
      const userMessage: Message = {
        id: temporaryUserId,
        chatId: resolvedChatId,
        role: "user",
        content: content.trim(),
        createdAt: new Date().toISOString(),
        status: "complete",
        parentMessageId: resolvedParentMessageId ?? undefined,
        attachments,
      };
      setMessages((current) => [...current, userMessage]);
      const requestBody: GenerateRequest = {
        chatId: resolvedChatId,
        modelId: selectedModel.id,
        content: content.trim(),
        attachmentIds: attachments.filter((item) => item.status === "ready").map((item) => item.id),
        settings: {
          ...settings,
          reasoning: isUsable(selectedModel, "reasoning_segments") ? settings.reasoning !== false : undefined,
        },
        parentMessageId: resolvedParentMessageId,
      };
      clientRequestStarted.current = performance.now();
      lastClientReceipt.current = null;
      const response = await api.generate(requestBody);
      if (response.userMessageId) {
        setMessages((current) => current.map((message) => message.id === temporaryUserId ? { ...message, id: response.userMessageId ?? message.id } : message));
      }
      const assistantId = response.messageId ?? `run-message-${response.runId}`;
      const assistantMessage: Message = {
        id: assistantId,
        chatId: resolvedChatId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
        status: "streaming",
        runId: response.runId,
        parentMessageId: response.userMessageId ?? temporaryUserId,
      };
      setMessages((current) => [...current, assistantMessage]);
      setAttachments([]);
      const run: RunDetails = {
        ...response.run,
        id: response.runId,
        chatId: resolvedChatId,
        messageId: assistantId,
        modelId: response.modelId ?? response.run?.modelId ?? selectedModel.id,
        status: response.run?.status ?? "queued",
        createdAt: response.run?.createdAt ?? new Date().toISOString(),
        tokens: response.run?.tokens ?? [],
        metrics: {
          ...response.run?.metrics,
          context: {
            ...response.run?.metrics?.context,
            effectiveLimit: selectedModel.effectiveContextLimit,
            behavior: selectedModel.effectiveContextLimit === null ? "unknown" : "none",
          },
        },
        effectiveSettings: response.run?.effectiveSettings ?? { ...requestBody.settings },
        rawEvents: response.run?.rawEvents ?? [],
      };
      setSelectedRun(run);
      setRunningRunId(response.runId);
      streamCleanup.current?.();
      streamCleanup.current = subscribeToRun(
        response.runId,
        (event) => handleStreamEvent(event, response.runId, assistantId, resolvedChatId),
        setStreamConnected,
        response.websocketUrl,
      );
    } catch (cause) {
      setRunningRunId(null);
      locallyCreatedChat.current = null;
      if (temporaryUserId) setMessages((current) => current.filter((message) => message.id !== temporaryUserId));
      setError(readableError(cause));
    }
  }, [activeChatId, attachments, flushSystemPrompt, handleStreamEvent, messages, runningRunId, selectedModel, settings, systemPrompt]);

  const stop = useCallback(async (): Promise<void> => {
    if (!runningRunId) return;
    try {
      await api.cancelRun(runningRunId);
      setNotice("Cancellation requested. Partial output and telemetry will be kept.");
    } catch (cause) {
      setError(readableError(cause));
    }
  }, [runningRunId]);

  const inspectRun = useCallback(async (id: string): Promise<void> => {
    if (selectedRun?.id === id) return;
    const inspectEpoch = ++runSelectionEpoch.current;
    try {
      const run = restoreClientTelemetry(await api.run(id));
      if (runSelectionEpoch.current === inspectEpoch) setSelectedRun(run);
    } catch (cause) {
      if (runSelectionEpoch.current === inspectEpoch) setError(readableError(cause));
    }
  }, [selectedRun?.id]);

  const loadSelectedRunEvents = useCallback(async (): Promise<void> => {
    const runId = selectedRun?.id;
    if (!runId || selectedRun.rawEvents !== undefined) return;
    try {
      const rawEvents = await api.runEvents(runId);
      setSelectedRun((current) => current?.id === runId ? { ...current, rawEvents } : current);
    } catch (cause) {
      setError(readableError(cause));
    }
  }, [selectedRun?.id, selectedRun?.rawEvents]);

  const branchFromAlternative = useCallback(async (
    tokenIndex: number,
    distribution: AlternativeDistribution,
    alternative: TokenAlternative,
  ): Promise<void> => {
    if (!selectedRun || runningRunId || branchPending.current) return;
    if (selectedRun.status !== "complete") {
      setError("Wait for the current run to finish before branching from a token.");
      return;
    }
    if (selectedRun.branchableThroughTokenIndex === undefined || tokenIndex > selectedRun.branchableThroughTokenIndex) {
      setError("That token was not retained in persisted telemetry and cannot be used as a branch point.");
      return;
    }
    runSelectionEpoch.current += 1;
    branchPending.current = true;
    setBranching(true);
    setError(null);
    setNotice("Creating a new chat from the selected token…");
    clientRequestStarted.current = performance.now();
    lastClientReceipt.current = null;
    try {
      const response = await api.branchRun(selectedRun.id, {
        tokenIndex,
        distribution,
        rank: alternative.rank,
        tokenId: alternative.tokenId,
      });
      const chatId = response.chatId ?? response.run?.chatId;
      if (!chatId) throw new Error("The branched run did not return its new chat.");
      const assistantId = response.messageId ?? `run-message-${response.runId}`;
      const assistantMessage: Message = {
        id: assistantId,
        chatId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
        status: "streaming",
        runId: response.runId,
      };
      const run: RunDetails = {
        ...response.run,
        id: response.runId,
        chatId,
        messageId: assistantId,
        modelId: response.modelId ?? response.run?.modelId ?? selectedRun.modelId,
        status: response.run?.status ?? "queued",
        createdAt: response.run?.createdAt ?? new Date().toISOString(),
        tokens: [],
        rawEvents: [],
      };
      locallyCreatedChat.current = chatId;
      activeChatIdRef.current = chatId;
      setActiveChatId(chatId);
      setMessages([assistantMessage]);
      setSelectedModelId(run.modelId);
      setSelectedRun(run);
      setRunningRunId(response.runId);
      setNotice("Branched into a new chat. Continuing from the selected token…");
      streamCleanup.current?.();
      streamCleanup.current = subscribeToRun(
        response.runId,
        (event) => handleStreamEvent(event, response.runId, assistantId, chatId),
        setStreamConnected,
        response.websocketUrl,
      );
      void api.messages(chatId).then((persistedMessages) => {
        if (activeChatIdRef.current !== chatId) return;
        setMessages((current) => {
          const currentAssistant = current.find((message) => message.id === assistantId) ?? assistantMessage;
          const persistedAssistant = persistedMessages.find((message) => message.id === assistantId);
          const mergedAssistant = persistedAssistant
            ? { ...persistedAssistant, ...currentAssistant, parentMessageId: persistedAssistant.parentMessageId }
            : currentAssistant;
          return persistedMessages.some((message) => message.id === assistantId)
            ? persistedMessages.map((message) => message.id === assistantId ? mergedAssistant : message)
            : [...persistedMessages, mergedAssistant];
        });
      }).catch(() => {
        // The branch remains usable from its stream if the history refresh is transiently unavailable.
      });
      void api.chats(false).then(setChats).catch(() => {
        // The new chat is already active; the sidebar can refresh after the run completes.
      });
    } catch (cause) {
      clientRequestStarted.current = null;
      lastClientReceipt.current = null;
      locallyCreatedChat.current = null;
      setRunningRunId(null);
      setNotice(null);
      setError(readableError(cause));
    } finally {
      branchPending.current = false;
      setBranching(false);
    }
  }, [handleStreamEvent, runningRunId, selectedRun]);

  const addAttachment = useCallback(async (file: File): Promise<void> => {
    if (!selectedModel) return;
    const temporaryId = `upload-${String(Date.now())}`;
    const pending: Attachment = {
      id: temporaryId,
      name: file.name,
      mimeType: file.type || "application/octet-stream",
      sizeBytes: file.size,
      kind: file.type.startsWith("image/") ? "image" : file.type.startsWith("audio/") ? "audio" : file.type.startsWith("video/") ? "video" : "document",
      status: "pending",
      nativeProcessing: false,
    };
    setAttachments((current) => [...current, pending]);
    try {
      const uploaded = await api.upload(file, selectedModel.id);
      setAttachments((current) => current.map((item) => item.id === temporaryId ? uploaded : item));
    } catch (cause) {
      setAttachments((current) => current.map((item) => item.id === temporaryId ? { ...item, status: "failed", detail: readableError(cause) } : item));
    }
  }, [selectedModel]);

  const removeAttachment = useCallback((id: string): void => {
    setAttachments((current) => current.filter((attachment) => attachment.id !== id));
  }, []);

  const retryMessage = useCallback(async (message: Message): Promise<void> => {
    if (!message.runId || runningRunId || branchPending.current) return;
    runSelectionEpoch.current += 1;
    setError(null);
    clientRequestStarted.current = performance.now();
    lastClientReceipt.current = null;
    try {
      const response = await api.replayRun(message.runId);
      const chatId = message.chatId || activeChatId;
      if (!chatId) throw new Error("The source response is not attached to a chat.");
      const assistantId = response.messageId ?? `run-message-${response.runId}`;
      const assistantMessage: Message = {
        id: assistantId,
        chatId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
        status: "streaming",
        runId: response.runId,
        parentMessageId: message.parentMessageId,
      };
      setMessages((current) => [...current, assistantMessage]);
      const run: RunDetails = {
        ...response.run,
        id: response.runId,
        chatId,
        messageId: assistantId,
        modelId: response.modelId ?? response.run?.modelId ?? selectedModel?.id ?? "unknown",
        status: response.run?.status ?? "queued",
        createdAt: response.run?.createdAt ?? new Date().toISOString(),
        tokens: response.run?.tokens ?? [],
        rawEvents: response.run?.rawEvents ?? [],
      };
      setSelectedRun(run);
      setRunningRunId(response.runId);
      streamCleanup.current?.();
      streamCleanup.current = subscribeToRun(
        response.runId,
        (event) => handleStreamEvent(event, response.runId, assistantId, chatId),
        setStreamConnected,
        response.websocketUrl,
      );
    } catch (cause) {
      clientRequestStarted.current = null;
      lastClientReceipt.current = null;
      setRunningRunId(null);
      setError(readableError(cause));
    }
  }, [activeChatId, handleStreamEvent, runningRunId, selectedModel?.id]);

  return {
    health,
    configuration,
    connected,
    booting,
    models,
    chats,
    archivedChats,
    activeChatId,
    messages,
    messagesLoading,
    selectedModelId,
    selectedModel,
    selectedRun,
    runningRunId,
    branching,
    streamConnected,
    settings,
    defaultSettings: configuredDefaults,
    attachments,
    systemPrompt,
    error,
    notice,
    setSelectedModelId,
    setSettings,
    setSystemPrompt,
    setError,
    selectChat,
    createChat,
    renameChat,
    togglePin,
    toggleArchive,
    deleteChat,
    clearChats,
    refreshModels,
    synchronizeRuntimeState,
    toggleModelLoaded,
    submit,
    stop,
    inspectRun,
    loadSelectedRunEvents,
    branchFromAlternative,
    addAttachment,
    removeAttachment,
    retryMessage,
  };
}
