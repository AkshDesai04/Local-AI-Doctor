import {
  Archive,
  ArchiveRestore,
  ArrowLeft,
  Boxes,
  Database,
  Ellipsis,
  MessageSquare,
  PanelLeftClose,
  Pencil,
  Pin,
  PinOff,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useMemo, useState } from "react";
import type { ChatSummary } from "../api/types";
import { relativeTime } from "../utils/format";
import { ConfirmDialog } from "./ConfirmDialog";
import { EmptyState, IconButton, MenuButton, MenuItem } from "./ui";

export type WorkspaceView = "chat" | "embeddings" | "models";

/** Workspace navigation. A new workspace is one entry here plus its view in App. */
const WORKSPACES: Array<{ id: WorkspaceView; label: string; icon: LucideIcon }> = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "embeddings", label: "Embeddings", icon: Boxes },
  { id: "models", label: "Models", icon: Database },
];

interface SidebarProps {
  open: boolean;
  chats: ChatSummary[];
  archivedChats: ChatSummary[];
  activeChatId: string | null;
  view: WorkspaceView;
  connected: boolean;
  onClose: () => void;
  onViewChange: (view: WorkspaceView) => void;
  onSelectChat: (id: string) => void;
  onCreateChat: () => void;
  onRename: (id: string, title: string) => void;
  onTogglePin: (chat: ChatSummary) => void;
  onToggleArchive: (chat: ChatSummary) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
}

interface ChatGroup {
  label: string;
  chats: ChatSummary[];
}

function groupChats(chats: ChatSummary[], archived: boolean): ChatGroup[] {
  if (archived) return [{ label: "Archived", chats }];
  const today = new Date().toDateString();
  const unpinned = chats.filter((chat) => !chat.pinned);
  return [
    { label: "Pinned", chats: chats.filter((chat) => chat.pinned) },
    { label: "Today", chats: unpinned.filter((chat) => new Date(chat.updatedAt).toDateString() === today) },
    { label: "Earlier", chats: unpinned.filter((chat) => new Date(chat.updatedAt).toDateString() !== today) },
  ].filter((group) => group.chats.length > 0);
}

interface ChatRowProps {
  chat: ChatSummary;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onTogglePin: () => void;
  onToggleArchive: () => void;
  onDelete: () => void;
}

function ChatRow({ chat, active, onSelect, onRename, onTogglePin, onToggleArchive, onDelete }: ChatRowProps): React.ReactNode {
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState(chat.title);
  const displayTitle = chat.title || "Untitled chat";

  const finishRename = (): void => {
    if (title.trim() && title.trim() !== chat.title) onRename(title);
    setTitle(chat.title);
    setRenaming(false);
  };

  return (
    <div className={`chat-row ${active ? "active" : ""}`}>
      {renaming ? (
        <input
          aria-label="Chat title"
          autoFocus
          className="input input-sm chat-rename-input"
          onBlur={finishRename}
          onChange={(event) => setTitle(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") finishRename();
            if (event.key === "Escape") {
              setTitle(chat.title);
              setRenaming(false);
            }
          }}
          value={title}
        />
      ) : (
        <button aria-current={active ? "page" : undefined} className="chat-row-main" onClick={onSelect} title={displayTitle} type="button">
          <span className="chat-row-title">{displayTitle}</span>
          <span className="chat-row-meta">{relativeTime(chat.updatedAt)}{chat.messageCount ? ` · ${String(chat.messageCount)} messages` : ""}</span>
        </button>
      )}
      {chat.pinned && !renaming && <Pin aria-label="Pinned" className="chat-pin" size={12} />}
      {!renaming && (
        <MenuButton buttonClassName="chat-row-menu" icon={<Ellipsis size={15} />} label={`Actions for ${displayTitle}`} size="sm">
          <MenuItem icon={<Pencil size={14} />} onSelect={() => setRenaming(true)}>Rename</MenuItem>
          <MenuItem icon={chat.pinned ? <PinOff size={14} /> : <Pin size={14} />} onSelect={onTogglePin}>{chat.pinned ? "Unpin" : "Pin"}</MenuItem>
          <MenuItem icon={chat.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />} onSelect={onToggleArchive}>{chat.archived ? "Restore" : "Archive"}</MenuItem>
          <MenuItem danger icon={<Trash2 size={14} />} onSelect={onDelete}>Delete</MenuItem>
        </MenuButton>
      )}
    </div>
  );
}

export function Sidebar({
  open,
  chats,
  archivedChats,
  activeChatId,
  view,
  connected,
  onClose,
  onViewChange,
  onSelectChat,
  onCreateChat,
  onRename,
  onTogglePin,
  onToggleArchive,
  onDelete,
  onClear,
}: SidebarProps): React.ReactNode {
  const [query, setQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<ChatSummary | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const source = showArchived ? archivedChats : chats;
  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return source;
    return source.filter((chat) => `${chat.title} ${chat.preview ?? ""}`.toLocaleLowerCase().includes(normalized));
  }, [query, source]);
  const groups = groupChats(filtered, showArchived);
  const hasData = chats.length > 0 || archivedChats.length > 0;

  const chooseView = (next: WorkspaceView): void => {
    onViewChange(next);
    if (window.matchMedia("(max-width: 760px)").matches) onClose();
  };

  return (
    <>
      {open && <button aria-label="Close navigation" className="sidebar-scrim" onClick={onClose} type="button" />}
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Conversation navigation">
        <div className="brand-row">
          <div aria-hidden="true" className="brand-mark"><Sparkles size={16} /></div>
          <div className="brand-copy"><strong>Local AI Doctor</strong><span>Private inference lab</span></div>
          <IconButton className="collapse-button" icon={<PanelLeftClose size={17} />} label="Collapse sidebar" onClick={onClose} />
        </div>

        <div className="sidebar-actions">
          <button className="new-chat-button" disabled={!connected} onClick={() => { onCreateChat(); chooseView("chat"); }} title={connected ? "Start a new chat" : "Start the local backend to create chats"} type="button">
            <Plus size={16} /><span>New chat</span><kbd>Ctrl K</kbd>
          </button>
        </div>

        <nav className="workspace-nav" aria-label="Workspaces">
          {WORKSPACES.map(({ id, label, icon: Icon }) => (
            <button aria-current={view === id ? "page" : undefined} className="nav-item" key={id} onClick={() => chooseView(id)} type="button">
              <Icon aria-hidden="true" size={16} />{label}
            </button>
          ))}
        </nav>

        <div className="sidebar-section-head">
          <label className="search-box">
            <Search aria-hidden="true" size={14} />
            <input aria-label="Search chats" onChange={(event) => setQuery(event.target.value)} placeholder="Search chats" value={query} />
            {query && <IconButton className="search-clear" icon={<X size={13} />} label="Clear search" onClick={() => setQuery("")} size="sm" />}
          </label>
        </div>

        <div className="chat-list" aria-live="polite">
          {groups.map((group) => (
            <section aria-label={group.label} className="chat-group" key={group.label}>
              <h2 className="chat-group-label">{group.label}</h2>
              {group.chats.map((chat) => (
                <ChatRow
                  active={view === "chat" && chat.id === activeChatId}
                  chat={chat}
                  key={chat.id}
                  onDelete={() => setDeleteTarget(chat)}
                  onRename={(title) => onRename(chat.id, title)}
                  onSelect={() => { onSelectChat(chat.id); chooseView("chat"); }}
                  onToggleArchive={() => onToggleArchive(chat)}
                  onTogglePin={() => onTogglePin(chat)}
                />
              ))}
            </section>
          ))}
          {filtered.length === 0 && (
            <EmptyState icon={query ? Search : MessageSquare} size="compact" title={query ? "No matching chats" : showArchived ? "No archived chats" : "No chats yet"} />
          )}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-footer-row">
            <button className="sidebar-link" onClick={() => setShowArchived((value) => !value)} type="button">
              {showArchived ? <ArrowLeft size={15} /> : <Archive size={15} />}
              <span>{showArchived ? "Back to chats" : "Archived"}</span>
              {!showArchived && <span className="sidebar-count">{String(archivedChats.length)}</span>}
            </button>
            <MenuButton icon={<Settings2 size={15} />} label="Workspace settings" placement="top-end" size="sm">
              <MenuItem danger disabled={!hasData} icon={<Trash2 size={14} />} onSelect={() => setConfirmClear(true)} title={hasData ? "Delete every saved chat and run trace" : "There is no saved conversation data"}>Clear all data…</MenuItem>
            </MenuButton>
          </div>
          <div className={`connection-status ${connected ? "online" : "offline"}`}>
            <span aria-hidden="true" className="status-dot" />
            <span>{connected ? "Local backend connected" : "Backend offline"}</span>
          </div>
        </div>
      </aside>

      <ConfirmDialog
        confirmLabel="Delete chat"
        danger
        description="This removes the chat, its branches, runs, and persisted telemetry. Model files are never touched."
        onCancel={() => setDeleteTarget(null)}
        onConfirm={() => { if (deleteTarget) onDelete(deleteTarget.id); setDeleteTarget(null); }}
        open={deleteTarget !== null}
        title={`Delete “${deleteTarget?.title ?? "chat"}”?`}
      />
      <ConfirmDialog
        confirmLabel="Clear all chats"
        danger
        description="Every saved chat and associated run trace will be deleted. This cannot be undone. Models and model files remain untouched."
        onCancel={() => setConfirmClear(false)}
        onConfirm={() => { onClear(); setConfirmClear(false); }}
        open={confirmClear}
        title="Clear all conversation data?"
      />
    </>
  );
}
