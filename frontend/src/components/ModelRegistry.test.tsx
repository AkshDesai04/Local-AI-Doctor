import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LoadOptions, ModelSummary, ResidentModel } from "../api/types";
import { ModelRegistry } from "./ModelRegistry";

const gib = 1024 ** 3;

const gpuModel: ModelSummary = {
  id: "model-a",
  name: "Model A",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture-a",
  lifecycle: "loaded",
  capabilities: {
    text_generation: { state: "full" },
    cuda: { state: "full" },
    cpu_offload: { state: "partial", reason: "accelerate layer offload when Strict VRAM is off; much slower" },
  },
  effectiveContextLimit: 4096,
};

const cpuOnlyModel: ModelSummary = {
  ...gpuModel,
  id: "model-b",
  name: "Model B",
  lifecycle: "unloaded",
  capabilities: {
    text_generation: { state: "full" },
    cuda: { state: "unavailable_on_backend", reason: "No usable CUDA runtime was discovered on this host." },
    cpu_offload: { state: "unsupported", reason: "CPU offload needs a CUDA device." },
  },
};

function resident(overrides: Partial<ResidentModel>): ResidentModel {
  return {
    modelKey: "key-a",
    modelId: gpuModel.id,
    displayName: "Model A",
    device: "cuda:0",
    dtype: "bfloat16",
    quantization: "none",
    strictVram: true,
    placement: "gpu",
    gpuBytes: 2 * gib,
    cpuBytes: 128 * 1024 ** 2,
    kvReserveBytes: 256 * 1024 ** 2,
    loadSeconds: 4.2,
    lastUsedAt: null,
    inUse: false,
    ...overrides,
  };
}

const defaults: LoadOptions = { device: "auto", dtype: "auto", strictVram: true };

function Registry({ residents = [], onLoad = vi.fn(), onUnloadResident = vi.fn() }: {
  residents?: ResidentModel[];
  onLoad?: (model: ModelSummary, options: LoadOptions) => void;
  onUnloadResident?: (resident: ResidentModel) => void;
}): React.ReactNode {
  const [options, setOptions] = useState<Record<string, LoadOptions>>({});
  return (
    <ModelRegistry
      connected
      loadOptionsFor={(id) => options[id] ?? defaults}
      maxLoadedModels={4}
      memory={null}
      models={[gpuModel, cpuOnlyModel]}
      onLoad={onLoad}
      onLoadOptionsChange={(id, value) => setOptions((current) => ({ ...current, [id]: value }))}
      onRefresh={vi.fn()}
      onSelectModel={vi.fn()}
      onSynchronize={vi.fn()}
      onUnloadAll={vi.fn()}
      onUnloadResident={onUnloadResident}
      residents={residents}
    />
  );
}

describe("model registry load panel", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ model_roots: ["/models"], writable: true }), { status: 200, headers: { "Content-Type": "application/json" } })));
  });

  it("loads with the chosen options and disables Strict VRAM on the CPU", async () => {
    const user = userEvent.setup();
    const onLoad = vi.fn();
    render(<Registry onLoad={onLoad} />);
    const panel = screen.getByRole("group", { name: "Load options for Model A" });
    const strict = within(panel).getByRole("switch", { name: /Strict VRAM/ });
    expect(strict).toBeEnabled();
    expect(strict).toBeChecked();

    await user.selectOptions(within(panel).getByRole("combobox", { name: "Data type" }), "float16");
    await user.click(strict);
    await user.click(within(panel).getByRole("button", { name: "Load" }));
    expect(onLoad).toHaveBeenLastCalledWith(gpuModel, { device: "auto", dtype: "float16", strictVram: false });

    await user.selectOptions(within(panel).getByRole("combobox", { name: "Device" }), "cpu");
    expect(within(panel).getByRole("switch", { name: /Strict VRAM/ })).toBeDisabled();
    await user.click(within(panel).getByRole("button", { name: "Load" }));
    expect(onLoad).toHaveBeenLastCalledWith(gpuModel, { device: "cpu", dtype: "float16", strictVram: false });
  });

  it("disables CUDA with the capability's reason when the backend cannot use it", () => {
    render(<Registry />);
    const panel = screen.getByRole("group", { name: "Load options for Model B" });
    expect(within(panel).getByRole("option", { name: "CUDA (unavailable)" })).toBeDisabled();
    expect(within(panel).getByRole("combobox", { name: "Device" })).toHaveAccessibleDescription("No usable CUDA runtime was discovered on this host.");
  });

  it("lists each resident copy with its placement, usage, and its own Unload", async () => {
    const user = userEvent.setup();
    const onUnloadResident = vi.fn();
    const offloaded = resident({ modelKey: "key-a-fp16", dtype: "float16", strictVram: false, placement: "offload", gpuBytes: 1.5 * gib, cpuBytes: gib });
    render(<Registry onUnloadResident={onUnloadResident} residents={[resident({}), offloaded]} />);

    const copies = within(screen.getByRole("list", { name: "Resident copies of Model A" })).getAllByRole("listitem");
    expect(copies).toHaveLength(2);
    expect(within(copies[0] as HTMLElement).getByText("cuda:0 · bf16 · GPU")).toBeInTheDocument();
    expect(within(copies[0] as HTMLElement).getByText("2 GiB")).toBeInTheDocument();
    expect(within(copies[1] as HTMLElement).getByText("cuda:0 · fp16")).toBeInTheDocument();
    expect(within(copies[1] as HTMLElement).getByText("Offloaded to system RAM")).toBeInTheDocument();
    expect(screen.getByText("2 resident copies")).toBeInTheDocument();

    await user.click(within(copies[1] as HTMLElement).getByRole("button", { name: /^Unload Model A/ }));
    expect(onUnloadResident).toHaveBeenCalledWith(offloaded);
  });

  it("keeps an in-use resident from being unloaded", () => {
    render(<Registry residents={[resident({ inUse: true })]} />);
    expect(screen.getByRole("button", { name: /^Unload Model A/ })).toBeDisabled();
    expect(screen.getByText("In use")).toBeInTheDocument();
  });

  it("reports CPU offload in the capability matrix with its reason", () => {
    render(<Registry />);
    const row = screen.getByRole("button", { name: "CPU offload" }).closest("tr");
    if (!row) throw new Error("Expected a CPU offload matrix row.");
    expect(within(row).getByText("Partial")).toHaveAttribute("title", "accelerate layer offload when Strict VRAM is off; much slower");
    expect(within(row).getByText("None")).toHaveAttribute("title", "CPU offload needs a CUDA device.");
  });
});
