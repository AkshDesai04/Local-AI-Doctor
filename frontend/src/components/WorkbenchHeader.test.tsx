import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ModelSummary } from "../api/types";
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

function header(overrides: Partial<React.ComponentProps<typeof WorkbenchHeader>> = {}): React.ReactElement {
  return (
    <WorkbenchHeader
      connected
      controlsOpen={false}
      inspectorOpen
      loadOptions={{ device: "cuda", dtype: "bfloat16" }}
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
    expect(screen.getByRole("button", { name: "Load" })).toHaveAttribute("title", expect.stringContaining("device cuda and dtype bfloat16"));
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
});
