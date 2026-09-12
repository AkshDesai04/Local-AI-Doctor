import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { Message } from "../api/types";
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
      run={null}
      runningRunId={null}
      selectedToken={null}
      streamConnected={false}
    />
  );
}

describe("chat response rendering", () => {
  it("shows cleaned Markdown normally and preserves raw protocol text in Nerd Mode", () => {
    const { rerender } = render(view(false));
    expect(screen.getByRole("heading", { name: "Final answer" })).toBeInTheDocument();
    expect(screen.queryByText(/hidden reasoning/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/end▁of▁sentence/i)).not.toBeInTheDocument();

    rerender(view(true));
    expect(document.querySelector(".nerd-raw-fallback")?.textContent).toBe(raw);
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
