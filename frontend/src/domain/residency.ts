import type {
  ConfigurationSnapshot,
  HealthStatus,
  LoadOptions,
  MemoryLedger,
  ModelSummary,
  ResidentModel,
  ResidentStatus,
  UnloadResult,
} from "../api/types";
import { formatBytes } from "../utils/format";

export const LOAD_OPTIONS_STORAGE_KEY = "local-ai-doctor.load-options.v1";

const MIB = 1024 ** 2;

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function isDevice(value: unknown): value is LoadOptions["device"] {
  return value === "auto" || value === "cpu" || value === "cuda";
}

function isDtype(value: unknown): value is LoadOptions["dtype"] {
  return value === "auto" || value === "float32" || value === "float16" || value === "bfloat16";
}

/** Load defaults from the effective configuration: the runtime device and dtype (auto unless configured) and runtime.strict_vram. */
export function defaultLoadOptions(configuration: ConfigurationSnapshot | null): LoadOptions {
  const runtime = asRecord(configuration?.effective.runtime);
  return {
    device: isDevice(runtime.device) ? runtime.device : "auto",
    dtype: isDtype(runtime.dtype) ? runtime.dtype : "auto",
    strictVram: runtime.strict_vram !== false,
  };
}

/** Saved per-model load options; unreadable or malformed storage yields no saved options. */
export function readStoredLoadOptions(): Record<string, Partial<LoadOptions>> {
  try {
    const decoded = asRecord(JSON.parse(window.localStorage.getItem(LOAD_OPTIONS_STORAGE_KEY) ?? "{}"));
    return Object.fromEntries(Object.entries(decoded).map(([modelId, value]) => {
      const raw = asRecord(value);
      return [modelId, {
        ...(isDevice(raw.device) ? { device: raw.device } : {}),
        ...(isDtype(raw.dtype) ? { dtype: raw.dtype } : {}),
        ...(typeof raw.strictVram === "boolean" ? { strictVram: raw.strictVram } : {}),
      }];
    }));
  } catch {
    return {};
  }
}

export function storeLoadOptions(options: Record<string, Partial<LoadOptions>>): void {
  try {
    window.localStorage.setItem(LOAD_OPTIONS_STORAGE_KEY, JSON.stringify(options));
  } catch {
    // Saved load options are a convenience; storage limits must never break loading.
  }
}

/** The richest available resident list: the resident endpoint, health's list, then its single loaded model. */
export function residentsFrom(health: HealthStatus | null, status: ResidentStatus | null): ResidentModel[] | null {
  if (status) return status.models;
  if (health?.loadedModels) return health.loadedModels;
  if (!health) return null;
  if (!health.loadedModelId) return [];
  return [{
    modelKey: health.loadedModelId,
    modelId: health.loadedModelId,
    displayName: null,
    device: health.loadedDevice ?? "unknown",
    dtype: null,
    quantization: null,
    strictVram: null,
    placement: null,
    gpuBytes: null,
    cpuBytes: null,
    kvReserveBytes: null,
    loadSeconds: null,
    lastUsedAt: null,
    inUse: false,
  }];
}

export function residentsOf(residents: readonly ResidentModel[] | null, modelId: string): ResidentModel[] {
  return residents?.filter((resident) => resident.modelId === modelId) ?? [];
}

/** The most recently used resident; list order breaks ties because the backend lists residents MRU last. */
export function mostRecent(residents: readonly ResidentModel[]): ResidentModel | undefined {
  return residents.reduce<ResidentModel | undefined>(
    (latest, resident) => !latest || (resident.lastUsedAt ?? "") >= (latest.lastUsedAt ?? "") ? resident : latest,
    undefined,
  );
}

export type ModelTransition = "loading" | "unloading" | "error";

/** Derives a model's lifecycle from the resident set; an unknown set keeps the discovery-reported lifecycle. */
export function withResidency(model: ModelSummary, residents: readonly ResidentModel[] | null, transition?: ModelTransition): ModelSummary {
  if (residents === null) return transition ? { ...model, lifecycle: transition } : model;
  const own = residentsOf(residents, model.id);
  const lifecycle = transition === "loading" || transition === "unloading"
    ? transition
    : own.length ? "loaded" : transition ?? "unloaded";
  return { ...model, lifecycle, loadedDevice: mostRecent(own)?.device ?? null };
}

/** The request placement that reuses an existing resident instead of loading a second copy. */
export function residentPlacement(resident: ResidentModel): LoadOptions {
  return {
    device: resident.device.startsWith("cuda") ? "cuda" : resident.device === "cpu" ? "cpu" : "auto",
    dtype: isDtype(resident.dtype) ? resident.dtype : "auto",
    strictVram: resident.strictVram ?? resident.placement !== "offload",
  };
}

const shortDtypes: Record<string, string> = { bfloat16: "bf16", float16: "fp16", float32: "fp32" };

export function shortDtype(dtype: string | null): string | null {
  return dtype ? shortDtypes[dtype] ?? dtype : null;
}

/** "cuda:0 · bf16", or just the device when the dtype is not reported. */
export function residentSummary(resident: ResidentModel): string {
  return [resident.device, shortDtype(resident.dtype)].filter(Boolean).join(" · ");
}

export function residentName(resident: ResidentModel, models: readonly ModelSummary[]): string {
  return resident.displayName ?? models.find((model) => model.id === resident.modelId)?.name ?? resident.modelId;
}

function listNames(names: string[]): string {
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names.at(-1) ?? ""}`;
}

export function evictionNotice(evictedKeys: readonly string[], target: string, residentsBefore: readonly ResidentModel[], models: readonly ModelSummary[]): string {
  const names = evictedKeys.map((key) => {
    const resident = residentsBefore.find((item) => item.modelKey === key);
    return resident ? residentName(resident, models) : "an idle model";
  });
  return `Unloaded ${listNames([...new Set(names)])} to make room for ${target}.`;
}

export function unloadNotice(subject: string, result: UnloadResult): string {
  const freed = result.freedBytes !== null && result.freedBytes >= MIB ? ` and freed ${formatBytes(result.freedBytes)}` : "";
  const leaked = result.leakedBytes !== null && result.leakedBytes >= MIB ? ` ${formatBytes(result.leakedBytes)} was not released.` : "";
  return `Unloaded ${subject}${freed}.${leaked}`;
}

export type LedgerSegmentKind = "resident" | "other" | "reserved" | "margin" | "free";

export interface LedgerSegment {
  id: string;
  kind: LedgerSegmentKind;
  label: string;
  bytes: number;
  /** Share of the bar, 0–100. */
  percent: number;
  /** Categorical colour index (1–7) for residents. */
  color?: number;
  detail: string;
}

/**
 * Splits device memory into resident weights, other use, PyTorch's reserved-but-free cache,
 * the load safety margin, and free memory. Null when the ledger has no totals.
 */
export function ledgerSegments(memory: MemoryLedger, residents: readonly ResidentModel[], models: readonly ModelSummary[]): LedgerSegment[] | null {
  const total = memory.totalBytes;
  const free = memory.freeBytes;
  if (!total || total <= 0 || free === null) return null;
  const onDevice = residents.filter((resident) => (resident.gpuBytes ?? 0) > 0);
  const residentBytes = onDevice.reduce((sum, resident) => sum + (resident.gpuBytes ?? 0), 0);
  const reservedFree = Math.max(0, (memory.torchReservedBytes ?? 0) - (memory.torchAllocatedBytes ?? 0));
  const other = Math.max(0, total - free - residentBytes - reservedFree);
  const margin = Math.min(Math.max(0, memory.safetyMarginBytes ?? 0), Math.max(0, free));
  const segments: Array<Omit<LedgerSegment, "percent">> = [
    ...onDevice.map((resident, index) => ({
      id: `resident-${resident.modelKey}`,
      kind: "resident" as const,
      label: residentName(resident, models),
      bytes: resident.gpuBytes ?? 0,
      color: (index % 7) + 1,
      detail: `${residentSummary(resident)}${resident.placement === "offload" ? " · offloaded" : ""}`,
    })),
    { id: "other", kind: "other", label: "Other GPU use", bytes: other, detail: "CUDA context, active runs, and other processes" },
    { id: "reserved", kind: "reserved", label: "Reserved by PyTorch", bytes: reservedFree, detail: "Cached by the allocator and reusable by loads" },
    { id: "margin", kind: "margin", label: "Safety margin", bytes: margin, detail: "Held back from every load" },
    { id: "free", kind: "free", label: "Free", bytes: Math.max(0, free - margin), detail: "Available to the next load" },
  ];
  // Measurements come from different probes; scale to their sum when they overshoot the device total.
  const scale = Math.max(total, segments.reduce((sum, segment) => sum + segment.bytes, 0));
  return segments.filter((segment) => segment.bytes > 0).map((segment) => ({ ...segment, percent: (segment.bytes / scale) * 100 }));
}

/** True when the ledger describes a CUDA device with a known total. */
export function isCudaLedger(memory: MemoryLedger | null): memory is MemoryLedger {
  return Boolean(memory?.device.startsWith("cuda") && memory.totalBytes);
}
