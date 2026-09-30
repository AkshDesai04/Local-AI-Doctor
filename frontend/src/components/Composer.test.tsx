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

describe("attachments", () => {
  it("accepts only the media kinds a vision-language generator can read", () => {
    const visionModel: ModelSummary = {
      ...model,
      capabilities: {
        text_generation: { state: "full" },
        vision: { state: "partial", reason: "validated on Qwen3-VL; other processor families best-effort" },
        video: { state: "partial", reason: "validated on Qwen3-VL; other processor families best-effort" },
        audio: { state: "unsupported", reason: "audio chat input is not implemented for this generation path" },
        native_file_input: { state: "partial", reason: "only processor-declared native media types are accepted" },
        extracted_text_input: { state: "unsupported", reason: "no extracted-text inference adapter is currently registered" },
      },
    };
    const { container } = render(
      <Composer
        attachments={[]}
        busy={false}
        connected
        model={visionModel}
        onAttach={vi.fn()}
        onRemoveAttachment={vi.fn()}
        onSettingsChange={vi.fn()}
        onStop={vi.fn()}
        onSubmit={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
      />,
    );

    expect(container.querySelector("input[type=file]")).toHaveAttribute("accept", "image/*,video/*");
    expect(screen.getByRole("button", { name: "Attach a file" })).toBeEnabled();
  });
});

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
        onSystemPromptChange={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
        systemPrompt=""
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
      onSystemPromptChange: vi.fn(),
      running: false,
      settings: defaultGenerationSettings,
      systemPrompt: "",
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
        onSystemPromptChange={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
        systemPrompt=""
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
        onSystemPromptChange={vi.fn()}
        running={false}
        settings={defaultGenerationSettings}
        systemPrompt=""
      />,
    );

    await user.click(screen.getByRole("button", { name: "Prompt controls" }));
    expect(screen.getByRole("checkbox", { name: "Reason before answering" })).toBeDisabled();
    expect(screen.getByTitle("Sampling settings for the next response")).toHaveTextContent("Reasoning unavailable");
  });

  it("edits the system prompt from the prompt controls and flags it while active", async () => {
    const user = userEvent.setup();
    const onSystemPromptChange = vi.fn();
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
      onSystemPromptChange,
      running: false,
      settings: defaultGenerationSettings,
      systemPrompt: "",
    };
    const { rerender } = render(<Composer {...props} />);
    expect(screen.queryByRole("img", { name: "System prompt active" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Prompt controls" }));
    const field = screen.getByRole("textbox", { name: "System prompt" });
    expect(field).toHaveAttribute("placeholder", "Optional instructions applied to every response in this chat");
    expect(field).toHaveAttribute("maxlength", "32000");
    fireEvent.change(field, { target: { value: "Answer briefly." } });
    expect(onSystemPromptChange).toHaveBeenLastCalledWith("Answer briefly.");

    rerender(<Composer {...props} systemPrompt="Answer briefly." />);
    expect(screen.getByRole("textbox", { name: "System prompt" })).toHaveValue("Answer briefly.");
    expect(screen.getByRole("img", { name: "System prompt active" })).toBeInTheDocument();

    rerender(<Composer {...props} systemPrompt="   " />);
    expect(screen.queryByRole("img", { name: "System prompt active" })).not.toBeInTheDocument();
  });
});
