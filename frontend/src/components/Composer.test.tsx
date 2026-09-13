import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { GenerationSettings, ModelSummary } from "../api/types";
import { defaultGenerationSettings } from "../hooks/useWorkbench";
import { Composer } from "./Composer";

const model: ModelSummary = {
  id: "model-1",
  name: "Reasoning model",
  architecture: "FixtureForCausalLM",
  task: "text_generation",
  fingerprint: "fixture",
  lifecycle: "loaded",
  capabilities: {
    text_generation: { state: "full" },
    reasoning_segments: { state: "full" },
    extracted_text_input: { state: "full" },
  },
  effectiveContextLimit: 4096,
};

describe("prompt controls", () => {
  it("offers reasoning, temperature, Top K, and Top P before sending", async () => {
    const user = userEvent.setup();
    const onSettingsChange = vi.fn();
    render(
      <Composer
        attachments={[]}
        busy={false}
        connected
        model={model}
        onAttach={vi.fn()}
        onRemoveAttachment={vi.fn()}
        onSettingsChange={onSettingsChange}
        onStop={vi.fn()}
        onSubmit={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Prompt controls" }));

    expect(screen.getByRole("checkbox", { name: "Reason before answering" })).toBeChecked();
    expect(screen.getByRole("slider", { name: "Temperature" })).toHaveValue("0.7");
    expect(screen.getByRole("slider", { name: "Top K" })).toHaveValue("50");
    expect(screen.getByRole("slider", { name: "Top P" })).toHaveValue("0.95");

    await user.click(screen.getByRole("checkbox", { name: "Reason before answering" }));
    fireEvent.change(screen.getByRole("slider", { name: "Temperature" }), { target: { value: "1.25" } });

    const reasoningUpdate = onSettingsChange.mock.calls[0]?.[0] as (current: GenerationSettings) => GenerationSettings;
    const temperatureUpdate = onSettingsChange.mock.calls[1]?.[0] as (current: GenerationSettings) => GenerationSettings;
    expect(reasoningUpdate(defaultGenerationSettings).reasoning).toBe(false);
    expect(temperatureUpdate(defaultGenerationSettings).temperature).toBe(1.25);
  });

  it("closes the prompt controls when generation becomes unavailable", async () => {
    const user = userEvent.setup();
    const props = {
      attachments: [],
      busy: false,
      connected: true,
      model,
      onAttach: vi.fn(),
      onRemoveAttachment: vi.fn(),
      onSettingsChange: vi.fn(),
      onStop: vi.fn(),
      onSubmit: vi.fn(),
      running: false,
      settings: defaultGenerationSettings,
    };
    const { rerender } = render(<Composer {...props} />);
    await user.click(screen.getByRole("button", { name: "Prompt controls" }));
    expect(screen.getByLabelText("Prompt controls panel")).toBeInTheDocument();

    rerender(<Composer {...props} connected={false} />);
    expect(screen.queryByLabelText("Prompt controls panel")).not.toBeInTheDocument();
  });

  it("keeps the attachment and prompt-control popovers mutually exclusive", async () => {
    const user = userEvent.setup();
    render(
      <Composer
        attachments={[]}
        busy={false}
        connected
        model={model}
        onAttach={vi.fn()}
        onRemoveAttachment={vi.fn()}
        onSettingsChange={vi.fn()}
        onStop={vi.fn()}
        onSubmit={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Attach a file" }));
    expect(screen.getByText("Accepted for this model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Prompt controls" }));
    expect(screen.queryByText("Accepted for this model")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Prompt controls panel")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Attach a file" }));
    expect(screen.queryByLabelText("Prompt controls panel")).not.toBeInTheDocument();
    expect(screen.getByText("Accepted for this model")).toBeInTheDocument();
  });

  it("marks reasoning unavailable instead of sending a misleading toggle", async () => {
    const user = userEvent.setup();
    render(
      <Composer
        attachments={[]}
        busy={false}
        connected
        model={{ ...model, capabilities: { text_generation: { state: "full" }, reasoning_segments: { state: "unsupported", reason: "No reasoning protocol was detected." } } }}
        onAttach={vi.fn()}
        onRemoveAttachment={vi.fn()}
        onSettingsChange={vi.fn()}
        onStop={vi.fn()}
        onSubmit={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Prompt controls" }));
    expect(screen.getByRole("checkbox", { name: "Reason before answering" })).toBeDisabled();
    expect(screen.getByTitle("Sampling settings for the next response")).toHaveTextContent("Reasoning unavailable");
  });
});
