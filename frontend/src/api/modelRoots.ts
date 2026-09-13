import { getSessionAuthToken, notifyAuthenticationRequired } from "./auth";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api/v1").replace(/\/$/, "");

export interface ModelRootSettings {
  modelRoots: string[];
  writable: boolean;
  source: string;
  reason?: string;
  containerized: boolean;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function normalize(value: unknown): ModelRootSettings {
  const raw = record(value);
  return {
    modelRoots: Array.isArray(raw.model_roots)
      ? raw.model_roots.filter((item): item is string => typeof item === "string")
      : [],
    writable: raw.writable === true,
    source: typeof raw.source === "string" ? raw.source : "user-local configuration",
    reason: typeof raw.reason === "string" ? raw.reason : undefined,
    containerized: raw.containerized === true,
  };
}

async function request(method: "GET" | "PUT", modelRoots?: string[]): Promise<ModelRootSettings> {
  const token = getSessionAuthToken();
  const response = await fetch(`${API_BASE}/configuration/model-roots`, {
    method,
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
      ...(modelRoots ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: modelRoots ? JSON.stringify({ model_roots: modelRoots }) : undefined,
  });
  const payload = await response.json().catch(() => ({})) as unknown;
  if (!response.ok) {
    if (response.status === 401) notifyAuthenticationRequired();
    const outer = record(payload);
    const error = record(outer.error);
    const message = typeof error.message === "string" ? error.message : "Model directory update failed.";
    const hint = typeof error.hint === "string" ? error.hint : undefined;
    throw new Error(hint ? `${message} ${hint}` : message);
  }
  return normalize(payload);
}

export function getModelRootSettings(): Promise<ModelRootSettings> {
  return request("GET");
}

export function updateModelRootSettings(modelRoots: string[]): Promise<ModelRootSettings> {
  return request("PUT", modelRoots);
}
