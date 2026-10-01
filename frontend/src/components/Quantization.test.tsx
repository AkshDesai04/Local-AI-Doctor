import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { FlushResult, LoadOptions, ModelSummary, ResidentModel } from "../api/types";
import { ModelRegistry } from "./ModelRegistry";

const gib = 1024 ** 3;

const quantizable: ModelSummary = {
  id: "model-a",
  name: "Model A",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture-a",
  lifecycle: "unloaded",
  rootIndex: 1,
  capabilities: {
    text_generation: { state: "full" },
    cuda: { state: "full" },
    weight_quantization: { state: "partial", reason: "bitsandbytes NF4 4-bit / LLM.int8 8-bit at load time; CUDA only" },
  },
  effectiveContextLimit: 4096,
};

const cpuHost: ModelSummary = {
  ...quantizable,
  id: "model-b",
  name: "Model B",
  capabilities: {
    text_generation: { state: "full" },
    weight_quantization: { state: "unavailable_on_backend", reason: "bitsandbytes quantization needs a usable CUDA runtime" },
  },
};

const derived: ModelSummary = {
  ...quantizable,
  id: "model-c",
  name: "Model-A-bnb-nf4",
  parameterCount: null,
  parameterCountNote: "packed quantized tensors",
  weightQuantization: { method: "bitsandbytes", bits: 4, quantType: "nf4" },
  derivation: { sourceDisplayName: "Model A", sourceModelId: "model-a", quantization: "bitsandbytes-4bit", createdAt: null },
  capabilities: {
    text_generation: { state: "full" },
    cuda: { state: "full" },
    weight_quantization: { state: "partial", reason: "pre-quantized bitsandbytes checkpoint; loads as-is; cannot be re-quantized" },
  },
};

const nf4Resident: ResidentModel = {
  modelKey: "key-nf4",
  modelId: quantizable.id,
  displayName: "Model A",
  device: "cuda:0",
  dtype: "bfloat16",
  quantization: "bitsandbytes-4bit",
  strictVram: true,
  placement: "gpu",
  gpuBytes: 1.5 * gib,
  cpuBytes: 0,
  kvReserveBytes: 0,
  loadSeconds: 26,
  lastUsedAt: null,
  inUse: false,
};

const defaults: LoadOptions = { device: "auto", dtype: "auto", strictVram: true };

function Registry({ residents = [], onLoad = vi.fn(), onFlushResident = vi.fn() }: {
  residents?: ResidentModel[];
  onLoad?: (model: ModelSummary, options: LoadOptions) => void;
  onFlushResident?: (resident: ResidentModel, target: { targetRootIndex: number; folderName: string }) => Promise<FlushResult>;
}): React.ReactNode {
  const [options, setOptions] = useState<Record<string, LoadOptions>>({});
  return (
    <ModelRegistry
      connected
      loadOptionsFor={(id) => options[id] ?? defaults}
      maxLoadedModels={4}
      memory={null}
      models={[quantizable, cpuHost, derived]}
      onFlushResident={onFlushResident}
      onLoad={onLoad}
      onLoadOptionsChange={(id, value) => setOptions((current) => ({ ...current, [id]: value }))}
      onRefresh={vi.fn()}
      onSelectModel={vi.fn()}
      onSynchronize={vi.fn()}
      onUnloadAll={vi.fn()}
      onUnloadResident={vi.fn()}
      residents={residents}
    />
  );
}

describe("load-time quantization in the registry", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ model_roots: ["/models", "/scratch/models"], writable: true }), { status: 200, headers: { "Content-Type": "application/json" } }),
    )));
  });

  it("loads with the chosen quantization and gates it by capability and device", async () => {
    const user = userEvent.setup();
    const onLoad = vi.fn();
    render(<Registry onLoad={onLoad} />);
    const panel = screen.getByRole("group", { name: "Load options for Model A" });
    const select = within(panel).getByRole("combobox", { name: "Quantization" });
    expect(select).toBeEnabled();
    expect(select).toHaveValue("none");

    await user.selectOptions(select, "bitsandbytes-4bit");
    await user.click(within(panel).getByRole("button", { name: "Load" }));
    expect(onLoad).toHaveBeenLastCalledWith(quantizable, { ...defaults, quantization: "bitsandbytes-4bit" });

    await user.selectOptions(within(panel).getByRole("combobox", { name: "Device" }), "cpu");
    const onCpu = within(panel).getByRole("combobox", { name: "Quantization" });
    expect(onCpu).toBeDisabled();
    expect(onCpu).toHaveValue("none");
    expect(onCpu).toHaveAccessibleDescription("bitsandbytes quantization runs on CUDA only.");

    const blocked = within(screen.getByRole("group", { name: "Load options for Model B" })).getByRole("combobox", { name: "Quantization" });
    expect(blocked).toBeDisabled();
    expect(blocked).toHaveAccessibleDescription("bitsandbytes quantization needs a usable CUDA runtime");
  });

  it("marks pre-quantized checkpoints and shows where a flushed model came from", () => {
    render(<Registry />);
    const card = screen.getByRole("article", { name: "Model-A-bnb-nf4" });
    expect(within(card).getByText("Pre-quantized 4-bit nf4")).toBeInTheDocument();
    expect(within(card).getByText("Derived from Model A · 4-bit NF4")).toBeInTheDocument();
    const select = within(card).getByRole("combobox", { name: "Quantization" });
    expect(select).toBeDisabled();
    expect(select).toHaveAccessibleDescription("pre-quantized bitsandbytes checkpoint; loads as-is; cannot be re-quantized");
    expect(within(card).getByTitle("packed quantized tensors")).toHaveTextContent("Unknown");
  });

  it("badges a quantized resident and only lets a GPU copy that is not in use be flushed", () => {
    render(<Registry residents={[nf4Resident, { ...nf4Resident, modelKey: "key-off", placement: "offload" }, { ...nf4Resident, modelKey: "key-busy", inUse: true }]} />);
    const copies = within(screen.getByRole("list", { name: "Resident copies of Model A" })).getAllByRole("listitem");
    expect(within(copies[0] as HTMLElement).getByText("4-bit NF4 · VRAM only")).toBeInTheDocument();
    const buttons = screen.getAllByRole("button", { name: "Flush Model A 4-bit NF4 to storage" });
    expect(buttons.map((button) => (button as HTMLButtonElement).disabled)).toEqual([false, true, true]);
    expect(buttons[1]).toHaveAttribute("title", "Only a copy held entirely in GPU memory can be flushed");
  });

  it("validates the folder name, defaults to the source root, and flushes", async () => {
    const user = userEvent.setup();
    const onFlushResident = vi.fn<(resident: ResidentModel, target: { targetRootIndex: number; folderName: string }) => Promise<FlushResult>>()
      .mockResolvedValue({ model: null, folder: "<model-root:1>/Model-A-bnb-nf4", bytesWritten: 1, derivation: null });
    render(<Registry onFlushResident={onFlushResident} residents={[nf4Resident]} />);
    await user.click(screen.getByRole("button", { name: "Flush Model A 4-bit NF4 to storage" }));

    const dialog = screen.getByRole("dialog", { name: "Flush to storage" });
    const root = within(dialog).getByRole("combobox", { name: "Model root" });
    await waitFor(() => expect(root).toHaveValue("1"));
    expect(within(dialog).getByRole("option", { name: "/scratch/models" })).toBeInTheDocument();
    const folder = within(dialog).getByRole("textbox", { name: "Folder name" });
    expect(folder).toHaveValue("Model-A-bnb-nf4");
    expect(within(dialog).getByText("About 1.5 GiB")).toBeInTheDocument();
    const submit = within(dialog).getByRole("button", { name: "Flush to storage" });

    for (const [name, message] of [["CON", /reserved on Windows/], ["bad name", /letters, digits/], ["trailing.", /end with a dot/]] as const) {
      await user.clear(folder);
      await user.type(folder, name);
      expect(within(dialog).getByRole("alert")).toHaveTextContent(message);
      expect(submit).toBeDisabled();
    }

    await user.clear(folder);
    await user.type(folder, "Model-A-nf4");
    await user.selectOptions(root, "0");
    await user.click(submit);
    expect(onFlushResident).toHaveBeenCalledWith(nf4Resident, { targetRootIndex: 0, folderName: "Model-A-nf4" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Flush to storage" })).not.toBeInTheDocument());
  });

  it("keeps the dialog open with the backend's reason when the flush fails", async () => {
    const user = userEvent.setup();
    const onFlushResident = vi.fn<(resident: ResidentModel, target: { targetRootIndex: number; folderName: string }) => Promise<FlushResult>>()
      .mockRejectedValue(new Error("the model root is not writable"));
    render(<Registry onFlushResident={onFlushResident} residents={[nf4Resident]} />);
    await user.click(screen.getByRole("button", { name: "Flush Model A 4-bit NF4 to storage" }));
    const dialog = screen.getByRole("dialog", { name: "Flush to storage" });
    await waitFor(() => expect(within(dialog).getByRole("combobox", { name: "Model root" })).toBeEnabled());
    await user.click(within(dialog).getByRole("button", { name: "Flush to storage" }));

    expect(await within(dialog).findByRole("alert")).toHaveTextContent("the model root is not writable");
    expect(within(dialog).getByRole("button", { name: "Flush to storage" })).toBeEnabled();
  });
});
