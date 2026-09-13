import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AUTH_SESSION_KEY } from "../api/auth";
import type { ModelSummary } from "../api/types";
import { defaultGenerationSettings } from "../hooks/useWorkbench";
import { GenerationControls } from "./GenerationControls";

function modelWithCuda(state: "full" | "unavailable_on_backend", reason?: string): ModelSummary {
  return {
    id: "model-1",
    name: "Local model",
    architecture: "FixtureForCausalLM",
    task: "text_generation",
    fingerprint: "fixture",
    lifecycle: "unloaded",
    capabilities: {
      text_generation: { state: "full" },
      cuda: { state, reason },
    },
    effectiveContextLimit: 4096,
  };
}

describe("runtime authentication settings", () => {
  it("stores a write-only token for the tab and clears it explicitly", async () => {
    const user = userEvent.setup();
    render(
      <GenerationControls
        authRequired
        model={null}
        onChange={vi.fn()}
        onClose={vi.fn()}
        open
        settings={defaultGenerationSettings}
      />,
    );

    const input = screen.getByLabelText("Authentication token");
    await user.type(input, "private-local-token");
    await user.click(screen.getByRole("button", { name: "Save for tab" }));

    expect(input).toHaveValue("");
    expect(window.sessionStorage.getItem(AUTH_SESSION_KEY)).toBe("private-local-token");
    expect(screen.queryByText("private-local-token")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(window.sessionStorage.getItem(AUTH_SESSION_KEY)).toBeNull();
  });
});

describe("runtime device settings", () => {
  it("disables CUDA and explains why when the backend reports it unavailable", () => {
    render(
      <GenerationControls
        authRequired={false}
        model={modelWithCuda("unavailable_on_backend", "No usable CUDA runtime was discovered on this host.")}
        onChange={vi.fn()}
        onClose={vi.fn()}
        open
        settings={defaultGenerationSettings}
      />,
    );

    expect(screen.getByRole("option", { name: "CUDA (unavailable)" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Device" })).toHaveAccessibleDescription(
      "No usable CUDA runtime was discovered on this host.",
    );
    expect(screen.getByText("CUDA unavailable")).toBeInTheDocument();
  });

  it("keeps CUDA selectable when the backend reports full support", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <GenerationControls
        authRequired={false}
        model={modelWithCuda("full")}
        onChange={onChange}
        onClose={vi.fn()}
        open
        settings={defaultGenerationSettings}
      />,
    );

    const cudaOption = screen.getByRole("option", { name: "CUDA" });
    expect(cudaOption).toBeEnabled();
    await user.selectOptions(screen.getByRole("combobox", { name: "Device" }), "cuda");

    expect(onChange).toHaveBeenCalledOnce();
    const update = onChange.mock.calls[0]?.[0] as (current: typeof defaultGenerationSettings) => typeof defaultGenerationSettings;
    expect(update(defaultGenerationSettings)).toMatchObject({ device: "cuda" });
  });
});
