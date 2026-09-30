import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ChatSummary } from "../api/types";
import { Sidebar } from "./Sidebar";

const now = new Date().toISOString();
const chats: ChatSummary[] = [
  { id: "pinned", title: "Pinned chat", createdAt: now, updatedAt: now, pinned: true, archived: false },
  { id: "today", title: "Today chat", createdAt: now, updatedAt: now, pinned: false, archived: false },
  { id: "earlier", title: "Earlier chat", createdAt: "2020-01-01T00:00:00Z", updatedAt: "2020-01-01T00:00:00Z", pinned: false, archived: false },
];

function sidebar(onClear = vi.fn(), onViewChange = vi.fn()): React.ReactElement {
  return (
    <Sidebar
      activeChatId="today"
      archivedChats={[]}
      chats={chats}
      connected
      onClear={onClear}
      onClose={vi.fn()}
      onCreateChat={vi.fn()}
      onDelete={vi.fn()}
      onRename={vi.fn()}
      onSelectChat={vi.fn()}
      onToggleArchive={vi.fn()}
      onTogglePin={vi.fn()}
      onViewChange={onViewChange}
      open={false}
      view="chat"
    />
  );
}

describe("Sidebar", () => {
  it("groups chats as Pinned, Today, and Earlier and marks the active one", () => {
    render(sidebar());
    expect(within(screen.getByRole("region", { name: "Pinned" })).getByText("Pinned chat")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Today" })).getByText("Today chat")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Earlier" })).getByText("Earlier chat")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Today chat/ })).toHaveAttribute("aria-current", "page");
  });

  it("renders workspace navigation from data and reports the chosen view", async () => {
    const user = userEvent.setup();
    const onViewChange = vi.fn();
    render(sidebar(vi.fn(), onViewChange));
    const nav = screen.getByRole("navigation", { name: "Workspaces" });
    expect(within(nav).getAllByRole("button").map((button) => button.textContent)).toEqual(["Chat", "Embeddings", "Models"]);
    await user.click(within(nav).getByRole("button", { name: "Models" }));
    expect(onViewChange).toHaveBeenCalledWith("models");
  });

  it("keeps Clear all data in the settings menu behind a confirmation", async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    render(sidebar(onClear));
    expect(screen.queryByText(/Clear all data/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Workspace settings" }));
    await user.click(screen.getByRole("menuitem", { name: /Clear all data/ }));
    expect(onClear).not.toHaveBeenCalled();

    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Clear all chats" }));
    expect(onClear).toHaveBeenCalledOnce();
  });
});
