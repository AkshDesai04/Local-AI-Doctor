import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { MemoryLedger, ModelSummary, ResidentModel } from "../api/types";
import { WorkbenchHeader } from "./WorkbenchHeader";

const model: ModelSummary = {
  id: "model-1",
  name: "Local model",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture",
  lifecycle: "unloaded",
  capabilities: { text_generation: { state: "full" } },
  effectiveContextLimit: 4096,
};

const gib = 1024 ** 3;

function resident(overrides: Partial<ResidentModel> = {}): ResidentModel {
  return {
    modelKey: "key-1",
    modelId: model.id,
    displayName: null,
    device: "cuda:0",
    dtype: "bfloat16",
    quantization: "none",
    strictVram: true,
    placement: "gpu",
    gpuBytes: 2 * gib,
    cpuBytes: 0,
    kvReserveBytes: 0,
    loadSeconds: 1,
    lastUsedAt: null,
    inUse: false,
    ...overrides,
  };
}

const cudaLedger: MemoryLedger = {
  device: "cuda:0",
  totalBytes: 8 * gib,
  freeBytes: 3 * gib,
  torchAllocatedBytes: 4 * gib,
  torchReservedBytes: 4.5 * gib,
  capBytes: null,
  processRssBytes: null,
  systemAvailableBytes: null,
  safetyMarginBytes: null,
  ledgerAgeSeconds: 1,
  stale: false,
};

function header(overrides: Partial<React.ComponentProps<typeof WorkbenchHeader>> = {}): React.ReactElement {
  return (
    <WorkbenchHeader
      connected
      controlsOpen={false}
      inspectorOpen
      loadOptions={{ device: "cuda", dtype: "bfloat16", strictVram: true }}
      memory={null}
      models={[model]}
      nerdMode={false}
      onOpenRegistry={vi.fn()}
      onOpenSidebar={vi.fn()}
      onRefresh={vi.fn()}
      onSelectModel={vi.fn()}
      onToggleControls={vi.fn()}
      onToggleInspector={vi.fn()}
      onToggleLoaded={vi.fn()}
      onToggleNerd={vi.fn()}
      residents={[]}
      selectedModel={model}
      selectedModelId={model.id}
      {...overrides}
    />
  );
}

describe("WorkbenchHeader", () => {
  it("shows the lifecycle, names the load options, and exposes Nerd Mode as a switch", async () => {
    const user = userEvent.setup();
    const onToggleNerd = vi.fn();
    render(header({ onToggleNerd }));

    expect(screen.getByRole("combobox", { name: "Selected model" })).toHaveValue("model-1");
    expect(screen.getByText("Not loaded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load" })).toHaveAttribute("title", expect.stringContaining("device cuda, dtype bfloat16, and Strict VRAM on"));
    await user.click(screen.getByRole("switch", { name: "Nerd Mode" }));
    expect(onToggleNerd).toHaveBeenCalledOnce();
  });

  it("moves secondary actions into the overflow menu", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    const onOpenRegistry = vi.fn();
    render(header({ onOpenRegistry, onRefresh }));

    expect(screen.queryByRole("button", { name: /Rescan/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "More actions" }));
    await user.click(screen.getByRole("menuitem", { name: "Rescan model roots" }));
    expect(onRefresh).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "More actions" }));
    await user.click(screen.getByRole("menuitem", { name: "Open model registry" }));
    expect(onOpenRegistry).toHaveBeenCalledOnce();
  });

  it("shows the loaded device and offers Unload once a model is resident", () => {
    render(header({ selectedModel: { ...model, lifecycle: "loaded", loadedDevice: "cuda:0" } }));
    expect(screen.getByText("cuda:0 · ready")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Unload" })).toBeEnabled();
  });

  it("reflects the selected model's resident copies in the status pill and groups resident models", () => {
    const other: ModelSummary = { ...model, id: "model-2", name: "Other model" };
    const loaded = { ...model, lifecycle: "loaded" as const };
    const { rerender } = render(header({ models: [loaded, other], residents: [resident()], selectedModel: loaded }));
    expect(screen.getByText("cuda:0 · bf16").parentElement).toHaveAttribute("title", expect.stringContaining("Resident: cuda:0 · bfloat16 · GPU · Strict VRAM") as unknown);
    expect(screen.getByRole("group", { name: "Resident" })).toContainElement(screen.getByRole("option", { name: "Local model" }));
    expect(screen.getByRole("group", { name: "Not loaded" })).toContainElement(screen.getByRole("option", { name: "Other model" }));

    rerender(header({ models: [loaded, other], residents: [resident(), resident({ modelKey: "key-2", device: "cpu", dtype: "float32", placement: "cpu" })], selectedModel: loaded }));
    expect(screen.getByText("2 resident")).toBeInTheDocument();

    rerender(header({ models: [loaded, other], residents: [resident()], selectedModel: { ...loaded, lifecycle: "unloading" } }));
    expect(screen.getByText("Unloading…", { selector: ".model-status-text" })).toBeInTheDocument();
  });

  it("renders the VRAM mini-bar only for a CUDA ledger", () => {
    const { rerender } = render(header({ memory: cudaLedger, residents: [resident()] }));
    const meter = screen.getByRole("meter", { name: "GPU memory in use" });
    expect(meter).toHaveAttribute("aria-valuetext", "5 GiB of 8 GiB");
    expect(meter.parentElement).toHaveAttribute("title", expect.stringContaining("GPU memory on cuda:0: 5 GiB of 8 GiB in use (63%) · 1 resident copy") as unknown);
    expect(screen.getByText("5/8 GiB")).toBeInTheDocument();

    rerender(header({ memory: { ...cudaLedger, device: "cpu" } }));
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    rerender(header({ memory: { ...cudaLedger, totalBytes: null } }));
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    rerender(header({ memory: null }));
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });
});
