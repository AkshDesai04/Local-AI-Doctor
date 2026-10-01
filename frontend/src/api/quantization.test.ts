import { afterEach, describe, expect, it, vi } from "vitest";
import { defaultGenerationSettings } from "../hooks/useWorkbench";
import { sendableLoadOptions } from "../domain/capabilities";
import { defaultFlushFolderName, defaultLoadOptions, flushFolderNameError, quantizationLabel, residentPlacement } from "../domain/residency";
import { api } from "./client";
import type { ModelSummary, ResidentModel } from "./types";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
}

function sentJson(fetchMock: { mock: { calls: Array<Parameters<typeof fetch>> } }): unknown {
  const body = fetchMock.mock.calls[0]?.[1]?.body;
  if (typeof body !== "string") throw new Error("Expected a JSON request body.");
  return JSON.parse(body);
}

const derivedDescriptor = {
  id: "qwen3-bnb-nf4-0123",
  display_name: "Qwen3-1.7B-bnb-nf4",
  task: "text_generation",
  parameter_count: null,
  root_index: 1,
  metadata: {
    weight_quantization: { method: "bitsandbytes", bits: 4, quant_type: "nf4" },
    parameter_count_note: "packed quantized tensors",
  },
  derivation: {
    source_model_id: "qwen3-1-7b-abc",
    source_display_name: "Qwen3-1.7B",
    quantization: "bitsandbytes-4bit",
    created_at: "2026-10-01T10:00:00+00:00",
  },
  capabilities: { entries: { weight_quantization: { state: "partial", reason: "pre-quantized bitsandbytes checkpoint; loads as-is; cannot be re-quantized" } } },
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("quantization at the API boundary", () => {
  it("normalizes pre-quantized metadata, provenance, and the model root index", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ models: [derivedDescriptor] })));
    const [model] = await api.models();
    expect(model).toMatchObject({
      weightQuantization: { method: "bitsandbytes", bits: 4, quantType: "nf4" },
      parameterCountNote: "packed quantized tensors",
      derivation: { sourceDisplayName: "Qwen3-1.7B", sourceModelId: "qwen3-1-7b-abc", quantization: "bitsandbytes-4bit" },
      rootIndex: 1,
      capabilities: { weight_quantization: { state: "partial" } },
    });
  });

  it("posts a flush and returns the new model with its provenance", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({
      model: derivedDescriptor,
      folder: "<model-root:1>/Qwen3-1.7B-bnb-nf4",
      bytes_written: 1_360_000_000,
      derivation: derivedDescriptor.derivation,
    }, 201));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.flushResident("key/a", { targetRootIndex: 1, folderName: "Qwen3-1.7B-bnb-nf4" });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/models/resident/key%2Fa/flush");
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBe("POST");
    expect(sentJson(fetchMock)).toEqual({ targetRootIndex: 1, folderName: "Qwen3-1.7B-bnb-nf4" });
    expect(result).toMatchObject({
      folder: "<model-root:1>/Qwen3-1.7B-bnb-nf4",
      bytesWritten: 1_360_000_000,
      model: { id: "qwen3-bnb-nf4-0123", rootIndex: 1 },
      derivation: { sourceDisplayName: "Qwen3-1.7B" },
    });
  });

  it("sends the quantization with loads and with generation", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ id: "m", model_key: "k", quantization: "bitsandbytes-4bit" }));
    vi.stubGlobal("fetch", fetchMock);
    await api.loadModel("m", { device: "cuda", dtype: "auto", strictVram: true, quantization: "bitsandbytes-4bit" });
    expect(sentJson(fetchMock)).toEqual({ device: "cuda", dtype: "auto", strictVram: true, quantization: "bitsandbytes-4bit" });

    fetchMock.mockClear();
    fetchMock.mockResolvedValue(jsonResponse({ run: { id: "run-1" } }));
    await api.generate({ chatId: "c", modelId: "m", content: "Hi", attachmentIds: [], settings: { ...defaultGenerationSettings, quantization: "bitsandbytes-8bit" } });
    expect(sentJson(fetchMock)).toMatchObject({ quantization: "bitsandbytes-8bit" });
  });
});

describe("quantization helpers", () => {
  const usable: ModelSummary = {
    id: "m",
    name: "Model",
    architecture: null,
    task: "text_generation",
    fingerprint: null,
    lifecycle: "unloaded",
    capabilities: { weight_quantization: { state: "partial", reason: "bitsandbytes NF4 4-bit / LLM.int8 8-bit at load time" } },
    effectiveContextLimit: null,
  };

  it("labels modes and builds folder names the backend accepts", () => {
    expect(quantizationLabel("bitsandbytes-4bit")).toBe("4-bit NF4");
    expect(quantizationLabel("bitsandbytes-8bit")).toBe("8-bit LLM.int8");
    expect(quantizationLabel("none")).toBeNull();
    expect(defaultFlushFolderName("Qwen3-1.7B", "bitsandbytes-4bit")).toBe("Qwen3-1.7B-bnb-nf4");
    expect(defaultFlushFolderName("My model (v2)", "bitsandbytes-8bit")).toBe("My-model-v2--bnb-int8");
    expect(flushFolderNameError(defaultFlushFolderName(".hidden name", "bitsandbytes-4bit"))).toBeNull();
    expect(defaultFlushFolderName("x".repeat(120), "bitsandbytes-4bit")).toHaveLength(100);
  });

  it("applies the backend folder-name rules", () => {
    for (const name of ["", ".hidden", "-x", "a b", "a/b", "x".repeat(101), "CON", "con.txt", "Lpt9", "trailing."]) {
      expect(flushFolderNameError(name), name).not.toBeNull();
    }
    for (const name of ["Qwen3-1.7B-bnb-nf4", "CONSOLE", "COM10", "a"]) {
      expect(flushFolderNameError(name), name).toBeNull();
    }
  });

  it("never sends a quantization the model or device cannot use", () => {
    const options = { device: "auto" as const, dtype: "auto" as const, strictVram: true, quantization: "bitsandbytes-4bit" as const };
    expect(sendableLoadOptions(usable, options)).toEqual(options);
    expect(sendableLoadOptions(usable, { ...options, device: "cpu" }).quantization).toBe("none");
    expect(sendableLoadOptions({ ...usable, capabilities: {} }, options).quantization).toBe("none");
    expect(sendableLoadOptions({ ...usable, weightQuantization: { method: "bitsandbytes", bits: 4, quantType: "nf4" } }, options).quantization).toBe("none");
  });

  it("reuses a resident's quantization and reads a configured default", () => {
    const resident: ResidentModel = {
      modelKey: "k", modelId: "m", displayName: null, device: "cuda:0", dtype: "bfloat16", quantization: "bitsandbytes-8bit", strictVram: true,
      placement: "gpu", gpuBytes: 1, cpuBytes: 0, kvReserveBytes: 0, loadSeconds: 1, lastUsedAt: null, inUse: false,
    };
    expect(residentPlacement(resident)).toEqual({ device: "cuda", dtype: "bfloat16", strictVram: true, quantization: "bitsandbytes-8bit" });
    expect(defaultLoadOptions({ effective: { runtime: { quantization: "bitsandbytes-4bit" } }, precedence: [] })).toMatchObject({ quantization: "bitsandbytes-4bit" });
    expect(defaultLoadOptions({ effective: { runtime: { quantization: "none" } }, precedence: [] })).not.toHaveProperty("quantization");
  });
});
