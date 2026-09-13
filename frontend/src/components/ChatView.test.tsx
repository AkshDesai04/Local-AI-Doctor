import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { Message, RunDetails } from "../api/types";
import { ChatView } from "./ChatView";

const raw = "<think>hidden reasoning</think>## Final answer\n\nDone.<｜end▁of▁sentence｜>";
const message: Message = {
  id: "assistant-1",
  chatId: "chat-1",
  role: "assistant",
  content: raw,
  createdAt: "2026-09-12T00:00:00Z",
  status: "complete",
  runId: "run-1",
};

function view(nerdMode: boolean, source: Message = message): React.ReactElement {
  return conversation([source], { nerdMode });
}

function conversation(
  messages: Message[],
  options: {
    nerdMode?: boolean;
    onBranch?: (content: string, parentMessageId: string | null) => void;
    onRetry?: (message: Message) => void;
    run?: RunDetails | null;
  } = {},
): React.ReactElement {
  return (
    <ChatView
      booting={false}
      connected
      messages={messages}
      messagesLoading={false}
      model={null}
      nerdMetric="rawProbability"
      nerdMode={options.nerdMode ?? false}
      onBranch={options.onBranch ?? vi.fn()}
      onInspectRun={vi.fn()}
      onNavigate={vi.fn()}
      onNerdMetricChange={vi.fn()}
      onOpenInspector={vi.fn()}
      onRetry={options.onRetry ?? vi.fn()}
      onSelectToken={vi.fn()}
      run={options.run ?? null}
      runningRunId={null}
      selectedToken={null}
      streamConnected={false}
    />
  );
}

describe("chat response rendering", () => {
  it("shows reasoning in a disclosure and reserves protocol tokens for Nerd Mode", () => {
    const { rerender } = render(view(false));
    expect(screen.getByRole("heading", { name: "Final answer" })).toBeInTheDocument();
    expect(screen.getByText("Thinking…").closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText(/hidden reasoning/i)).toBeInTheDocument();
    expect(screen.queryByText(/end▁of▁sentence/i)).not.toBeInTheDocument();

    rerender(view(true));
    expect(screen.getByLabelText("Start reasoning token")).toHaveTextContent("<think>");
    expect(screen.getByLabelText("End reasoning token")).toHaveTextContent("</think>");
    expect(screen.getByLabelText("Termination tokens")).toHaveTextContent("<｜end▁of▁sentence｜>");
  });

  it("groups clickable reasoning tokens and exposes explicit boundaries in Nerd Mode", () => {
    const run: RunDetails = {
      id: "run-1",
      messageId: "assistant-1",
      modelId: "model-1",
      status: "complete",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [
        { index: 0, tokenId: 10, piece: "<think>", displayText: "<think>", reasoningSegment: "reasoning" },
        { index: 1, tokenId: 11, piece: "consider", displayText: "consider", reasoningSegment: "reasoning" },
        { index: 2, tokenId: 12, piece: "</think>", displayText: "</think>", reasoningSegment: "reasoning" },
        { index: 3, tokenId: 13, piece: "Done.", displayText: "Done.", reasoningSegment: "answer" },
        { index: 4, tokenId: 14, piece: "<|eot_id|>", displayText: "<|eot_id|>", reasoningSegment: "answer" },
      ],
    };
    render(conversation([message], { nerdMode: true, run }));

    const reasoning = screen.getByText(/Thinking…/i).closest("details");
    expect(reasoning).not.toHaveAttribute("open");
    expect(screen.getAllByRole("button", { name: /Start reasoning token 0/i })).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: /End reasoning token 2/i })).toHaveLength(1);
    expect(screen.getByRole("button", { name: /Termination token 4/i })).toBeInTheDocument();
  });

  it("renders token telemetry for a cloned message linked to its source run", () => {
    const clonedMessage = { ...message, id: "assistant-clone", runId: "run-1" };
    const run: RunDetails = {
      id: "run-1",
      messageId: "assistant-original",
      modelId: "model-1",
      status: "complete",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [{ index: 0, tokenId: 13, piece: "Done.", displayText: "Done.", reasoningSegment: "answer" }],
    };

    render(conversation([clonedMessage], { nerdMode: true, run }));

    expect(screen.getByRole("button", { name: /Token 0 Done/i })).toBeInTheDocument();
  });

  it("uses exact reasoning slices so a mixed closing token does not hide answer text", () => {
    const mixedMessage = { ...message, content: "<think>plan</think>Answer<|end|>" };
    const run: RunDetails = {
      id: "run-1",
      messageId: "assistant-1",
      modelId: "model-1",
      status: "complete",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [
        {
          index: 0,
          tokenId: 10,
          piece: "<think>plan",
          displayText: "<think>plan",
          reasoningSegment: "reasoning",
          reasoningSlices: [
            { start: 0, end: 7, classification: "reasoning", delimiter: true },
            { start: 7, end: 11, classification: "reasoning", delimiter: false },
          ],
        },
        {
          index: 1,
          tokenId: 11,
          piece: "</think>Answer",
          displayText: "</think>Answer",
          reasoningSegment: "unknown",
          reasoningSlices: [
            { start: 0, end: 8, classification: "reasoning", delimiter: true },
            { start: 8, end: 14, classification: "answer", delimiter: false },
          ],
        },
        { index: 2, tokenId: 12, piece: "<|end|>", displayText: "<|end|>", reasoningSegment: "answer" },
      ],
    };
    render(conversation([mixedMessage], { nerdMode: true, run }));

    const disclosure = screen.getByText(/Thinking…/i).closest("details");
    expect(screen.getAllByRole("button", { name: /Start reasoning token 0/i })).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: /End reasoning token 1/i })).toHaveLength(1);
    expect(screen.getByRole("button", { name: /Token 1 Answer/i })).not.toBe(disclosure);
    expect(screen.getByRole("button", { name: /Token 1 Answer/i }).closest("details")).toBeNull();
    expect(screen.getByRole("button", { name: /Termination token 2/i })).toBeInTheDocument();
  });

  it("recognizes legacy reasoning delimiters split across token pieces", () => {
    const splitMessage = { ...message, content: "<think>plan</think>Answer" };
    const run: RunDetails = {
      id: "run-split-delimiters",
      messageId: "assistant-1",
      modelId: "model-1",
      status: "complete",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [
        { index: 0, tokenId: 10, piece: "<thi", displayText: "<thi", reasoningSegment: "unknown" },
        { index: 1, tokenId: 11, piece: "nk>plan</th", displayText: "nk>plan</th", reasoningSegment: "unknown" },
        { index: 2, tokenId: 12, piece: "ink>Answer", displayText: "ink>Answer", reasoningSegment: "unknown" },
      ],
    };
    render(conversation([splitMessage], { nerdMode: true, run }));

    const startParts = screen.getAllByRole("button", { name: /Start reasoning token/u });
    const endParts = screen.getAllByRole("button", { name: /End reasoning token/u });
    expect(startParts).toHaveLength(2);
    expect(startParts[0]).toHaveTextContent("<thi");
    expect(startParts[1]).toHaveTextContent("nk>");
    expect(endParts).toHaveLength(2);
    expect(endParts[0]).toHaveTextContent("</th");
    expect(endParts[1]).toHaveTextContent("ink>");
    expect(screen.getByRole("button", { name: /Token 1 plan/u }).closest("details")).not.toBeNull();
    expect(screen.getByRole("button", { name: /Token 2 Answer/u }).closest("details")).toBeNull();
  });

  it("keeps adjacent start and end delimiters distinct for empty reasoning", () => {
    const emptyReasoningMessage = { ...message, content: "<think></think>Answer" };
    const run: RunDetails = {
      id: "run-empty-reasoning",
      messageId: "assistant-1",
      modelId: "model-1",
      status: "complete",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [{
        index: 0,
        tokenId: 10,
        piece: "<think></think>Answer",
        displayText: "<think></think>Answer",
        reasoningSegment: "unknown",
        reasoningSlices: [
          { start: 0, end: 7, classification: "reasoning", delimiter: true },
          { start: 7, end: 15, classification: "reasoning", delimiter: true },
          { start: 15, end: 21, classification: "answer", delimiter: false },
        ],
      }],
    };
    render(conversation([emptyReasoningMessage], { nerdMode: true, run }));

    expect(screen.getByRole("button", { name: /Start reasoning token 0/u })).toHaveTextContent("<think>");
    expect(screen.getByRole("button", { name: /End reasoning token 0/u })).toHaveTextContent("</think>");
    expect(screen.getByRole("button", { name: /Token 0 Answer/u }).closest("details")).toBeNull();
  });

  it("applies backend code-point offsets without splitting non-BMP characters", () => {
    const unicodeMessage = { ...message, content: "<think>😀</think>Answer" };
    const run: RunDetails = {
      id: "run-unicode",
      messageId: "assistant-1",
      modelId: "model-1",
      status: "complete",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [
        {
          index: 0,
          tokenId: 10,
          piece: "<think>",
          displayText: "<think>",
          reasoningSegment: "reasoning",
          reasoningSlices: [{ start: 0, end: 7, classification: "reasoning", delimiter: true }],
        },
        {
          index: 1,
          tokenId: 11,
          piece: "😀</think>Answer",
          displayText: "😀</think>Answer",
          reasoningSegment: "unknown",
          reasoningSlices: [
            { start: 0, end: 1, classification: "reasoning", delimiter: false },
            { start: 1, end: 9, classification: "reasoning", delimiter: true },
            { start: 9, end: 15, classification: "answer", delimiter: false },
          ],
        },
      ],
    };
    const { container } = render(conversation([unicodeMessage], { nerdMode: true, run }));

    expect(screen.getByRole("button", { name: /Token 1 😀/u })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /End reasoning token 1/u })).toHaveTextContent("</think>");
    expect(screen.getByRole("button", { name: /Token 1 Answer/u }).closest("details")).toBeNull();
    expect(container.textContent).not.toContain("�");
  });

  it("labels prompt-primed reasoning honestly without inventing an end token", () => {
    const primedMessage = { ...message, content: "private reasoning", reasoningPrimed: true };
    const run: RunDetails = {
      id: "run-1",
      messageId: "assistant-1",
      modelId: "model-1",
      status: "running",
      createdAt: "2026-09-12T00:00:00Z",
      tokens: [{ index: 0, tokenId: 10, piece: "private reasoning", displayText: "private reasoning", reasoningSegment: "reasoning" }],
    };
    render(conversation([primedMessage], { nerdMode: true, run }));

    expect(screen.getByLabelText("Prompt-primed reasoning start")).toHaveTextContent("no opening token emitted");
    expect(screen.queryByRole("button", { name: /End reasoning token/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("End reasoning token")).not.toBeInTheDocument();
  });

  it("branches an edited user message from that message's parent", async () => {
    const user = userEvent.setup();
    const onBranch = vi.fn();
    render(
      <ChatView
        booting={false}
        connected
        messages={[{ ...message, id: "user-2", role: "user", content: "Original", parentMessageId: "assistant-1" }]}
        messagesLoading={false}
        model={null}
        nerdMetric="rawProbability"
        nerdMode={false}
        onBranch={onBranch}
        onInspectRun={vi.fn()}
        onNavigate={vi.fn()}
        onNerdMetricChange={vi.fn()}
        onOpenInspector={vi.fn()}
        onRetry={vi.fn()}
        onSelectToken={vi.fn()}
        run={null}
        runningRunId={null}
        selectedToken={null}
        streamConnected={false}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Edit and retry" }));
    const editor = screen.getByRole("textbox");
    await user.clear(editor);
    await user.type(editor, "Revised");
    await user.click(screen.getByRole("button", { name: /Send as branch/i }));
    expect(onBranch).toHaveBeenCalledWith("Revised", "assistant-1");
  });

  it("does not offer backend-invalid replay for a cancelled response", () => {
    render(view(false, { ...message, status: "cancelled" }));
    const retry = screen.getByRole("button", { name: "Regenerate response" });
    expect(retry).toBeDisabled();
  });

  it("preserves chronological rendering for a linear conversation", () => {
    const messages: Message[] = [
      { ...message, id: "user-root", role: "user", content: "First question" },
      { ...message, id: "assistant-one", content: "First answer", parentMessageId: "user-root" },
      { ...message, id: "user-two", role: "user", content: "Follow-up", parentMessageId: "assistant-one" },
      { ...message, id: "assistant-two", content: "Second answer", parentMessageId: "user-two" },
    ];
    render(conversation(messages));

    expect(Array.from(document.querySelectorAll(".message-content")).map((node) => node.textContent)).toEqual([
      "First question",
      "First answer",
      "Follow-up",
      "Second answer",
    ]);
    expect(screen.queryByRole("combobox", { name: /branch/i })).not.toBeInTheDocument();
  });

  it("shows one root-to-leaf lineage and lets the user select a sibling branch", async () => {
    const user = userEvent.setup();
    const messages: Message[] = [
      { ...message, id: "user-root", role: "user", content: "Question" },
      { ...message, id: "assistant-old", content: "Older answer", parentMessageId: "user-root" },
      { ...message, id: "user-old", role: "user", content: "Older follow-up", parentMessageId: "assistant-old" },
      { ...message, id: "assistant-new", content: "Newer answer", parentMessageId: "user-root" },
      { ...message, id: "user-new", role: "user", content: "Newer follow-up", parentMessageId: "assistant-new" },
    ];
    render(conversation(messages));

    expect(screen.getByText("Newer answer")).toBeInTheDocument();
    expect(screen.getByText("Newer follow-up")).toBeInTheDocument();
    expect(screen.queryByText("Older answer")).not.toBeInTheDocument();
    const selector = screen.getByRole("combobox", { name: "Choose conversation branch" });
    expect(selector).toHaveValue("assistant-new");

    await user.selectOptions(selector, "assistant-old");
    expect(screen.getByText("Older answer")).toBeInTheDocument();
    expect(screen.getByText("Older follow-up")).toBeInTheDocument();
    expect(screen.queryByText("Newer answer")).not.toBeInTheDocument();
  });

  it("switches to a newly created edit branch when it arrives", async () => {
    const user = userEvent.setup();
    const onBranch = vi.fn();
    const original: Message[] = [
      { ...message, id: "user-root", role: "user", content: "Original question" },
      { ...message, id: "assistant-old", content: "Original answer", parentMessageId: "user-root" },
    ];
    const { rerender } = render(conversation(original, { onBranch }));

    await user.click(screen.getByRole("button", { name: "Edit and retry" }));
    await user.clear(screen.getByRole("textbox"));
    await user.type(screen.getByRole("textbox"), "Revised question");
    await user.click(screen.getByRole("button", { name: /Send as branch/i }));
    expect(onBranch).toHaveBeenCalledWith("Revised question", null);

    rerender(conversation([
      ...original,
      { ...message, id: "user-revised", role: "user", content: "Revised question" },
      { ...message, id: "assistant-revised", content: "Revised answer", parentMessageId: "user-revised" },
    ], { onBranch }));
    expect(screen.getByText("Revised question")).toBeInTheDocument();
    expect(screen.getByText("Revised answer")).toBeInTheDocument();
    expect(screen.queryByText("Original answer")).not.toBeInTheDocument();
  });

  it("switches to the newly regenerated sibling response", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const original: Message[] = [
      { ...message, id: "user-root", role: "user", content: "Question" },
      { ...message, id: "assistant-old", content: "Older answer", parentMessageId: "user-root", runId: "run-old" },
      { ...message, id: "assistant-current", content: "Current answer", parentMessageId: "user-root", runId: "run-current" },
    ];
    const { rerender } = render(conversation(original, { onRetry }));
    await user.selectOptions(screen.getByRole("combobox", { name: "Choose conversation branch" }), "assistant-old");
    await user.click(screen.getByRole("button", { name: "Regenerate response" }));
    expect(onRetry).toHaveBeenCalledWith(expect.objectContaining({ id: "assistant-old" }));

    rerender(conversation([
      ...original,
      { ...message, id: "assistant-regenerated", content: "Regenerated answer", parentMessageId: "user-root", runId: "run-regenerated" },
    ], { onRetry }));
    expect(screen.getByText("Regenerated answer")).toBeInTheDocument();
    expect(screen.queryByText("Older answer")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Choose conversation branch" })).toHaveValue("assistant-regenerated");
  });

  it("loads persisted media through the authenticated content helper and revokes blob URLs", async () => {
    const attachmentContent = vi.spyOn(api, "attachmentContent").mockImplementation((id) => Promise.resolve(new Blob([id], { type: id === "image-1" ? "image/png" : "video/mp4" })));
    const createObjectURL = vi.fn()
      .mockReturnValueOnce("blob:image-preview")
      .mockReturnValueOnce("blob:video-preview");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const source: Message = {
      ...message,
      id: "user-with-media",
      role: "user",
      content: "Inspect these files",
      attachments: [
        { id: "image-1", name: "scan.png", mimeType: "image/png", sizeBytes: 12, kind: "image", status: "ready", nativeProcessing: true },
        { id: "video-1", name: "clip.mp4", mimeType: "video/mp4", sizeBytes: 24, kind: "video", status: "ready", nativeProcessing: false },
        { id: "document-1", name: "notes.pdf", mimeType: "application/pdf", sizeBytes: 36, kind: "document", status: "ready", nativeProcessing: false },
      ],
    };

    const { unmount } = render(conversation([source]));
    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(2));
    expect(attachmentContent).toHaveBeenCalledWith("image-1");
    expect(attachmentContent).toHaveBeenCalledWith("video-1");
    expect(attachmentContent).not.toHaveBeenCalledWith("document-1");
    expect(screen.getByRole("img", { name: "Preview of scan.png" })).toHaveAttribute("src", "blob:image-preview");
    expect(screen.getByLabelText("Preview of clip.mp4")).toHaveAttribute("src", "blob:video-preview");
    expect(screen.getByText("notes.pdf")).toBeInTheDocument();

    unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:image-preview");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:video-preview");
  });
});
