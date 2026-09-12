import {
  Archive,
  ArchiveRestore,
  Boxes,
  ChevronDown,
  Database,
  MessageSquare,
  MoreHorizontal,
  PanelLeftClose,
  Pin,
  PinOff,
  Plus,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";
import type { ChatSummary } from "../api/types";
import { relativeTime } from "../utils/format";
import { ConfirmDialog } from "./ConfirmDialog";

export type WorkspaceView = "chat" | "embeddings" | "models";

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
  const [menuOpen, setMenuOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState(chat.title);

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
          className="chat-rename-input"
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
        <button className="chat-row-main" onClick={onSelect} type="button">
          <span className="chat-row-title">{chat.title || "Untitled chat"}</span>
          <span className="chat-row-meta">{relativeTime(chat.updatedAt)}{chat.messageCount ? ` · ${String(chat.messageCount)} messages` : ""}</span>
        </button>
      )}
      {chat.pinned && <Pin aria-label="Pinned" className="chat-pin" size={12} />}
      <button
        aria-label={`Actions for ${chat.title}`}
        aria-expanded={menuOpen}
        className="chat-row-menu icon-button"
        onClick={() => setMenuOpen((value) => !value)}
        type="button"
      >
        <MoreHorizontal size={16} />
      </button>
      {menuOpen && (
        <div className="popover chat-menu" role="menu">
          <button onClick={() => { setRenaming(true); setMenuOpen(false); }} role="menuitem" type="button">Rename</button>
          <button onClick={() => { onTogglePin(); setMenuOpen(false); }} role="menuitem" type="button">
            {chat.pinned ? <PinOff size={14} /> : <Pin size={14} />} {chat.pinned ? "Unpin" : "Pin"}
          </button>
          <button onClick={() => { onToggleArchive(); setMenuOpen(false); }} role="menuitem" type="button">
            {chat.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />} {chat.archived ? "Restore" : "Archive"}
          </button>
          <button className="danger-text" onClick={() => { onDelete(); setMenuOpen(false); }} role="menuitem" type="button">
            <Trash2 size={14} /> Delete
          </button>
        </div>
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
  const pinned = filtered.filter((chat) => chat.pinned);
  const recent = filtered.filter((chat) => !chat.pinned);

  const chooseView = (next: WorkspaceView): void => {
    onViewChange(next);
    if (window.matchMedia("(max-width: 760px)").matches) onClose();
  };

  return (
    <>
      {open && <button aria-label="Close navigation" className="sidebar-scrim" onClick={onClose} type="button" />}
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Conversation navigation">
        <div className="brand-row">
          <div className="brand-mark"><Sparkles size={17} /></div>
          <div className="brand-copy"><strong>Local AI Doctor</strong><span>Private inference lab</span></div>
          <button aria-label="Collapse sidebar" className="icon-button collapse-button" onClick={onClose} type="button"><PanelLeftClose size={18} /></button>
        </div>

        <button className="new-chat-button" disabled={!connected} onClick={() => { onCreateChat(); chooseView("chat"); }} type="button">
          <Plus size={17} /> New chat <kbd>Ctrl K</kbd>
        </button>

        <nav className="workspace-nav" aria-label="Workspaces">
          <button className={view === "chat" ? "selected" : ""} onClick={() => chooseView("chat")} type="button"><MessageSquare size={16} /> Chat</button>
          <button className={view === "embeddings" ? "selected" : ""} onClick={() => chooseView("embeddings")} type="button"><Boxes size={16} /> Embeddings</button>
          <button className={view === "models" ? "selected" : ""} onClick={() => chooseView("models")} type="button"><Database size={16} /> Model registry</button>
        </nav>

        <div className="sidebar-divider" />
        <label className="search-box">
          <Search size={15} />
          <input aria-label="Search chats" onChange={(event) => setQuery(event.target.value)} placeholder="Search chats" value={query} />
          {query && <button aria-label="Clear search" className="bare-button" onClick={() => setQuery("")} type="button"><X size={13} /></button>}
        </label>

        <div className="chat-list" aria-live="polite">
          {pinned.length > 0 && <p className="section-label">Pinned</p>}
          {pinned.map((chat) => (
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
          {recent.length > 0 && <p className="section-label">{showArchived ? "Archived" : "Recent"}</p>}
          {recent.map((chat) => (
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
          {filtered.length === 0 && (
            <div className="sidebar-empty"><MessageSquare size={18} /><span>{query ? "No matching chats" : showArchived ? "No archived chats" : "No chats yet"}</span></div>
          )}
        </div>

        <div className="sidebar-footer">
          <button className="sidebar-footer-button" onClick={() => setShowArchived((value) => !value)} type="button">
            <Archive size={15} /> {showArchived ? "Back to chats" : `Archived (${String(archivedChats.length)})`} <ChevronDown className={showArchived ? "rotated" : ""} size={14} />
          </button>
          {(chats.length > 0 || archivedChats.length > 0) && (
            <button className="clear-chats" onClick={() => setConfirmClear(true)} type="button"><Trash2 size={14} /> Clear all data</button>
          )}
          <div className="local-status"><span className={`status-dot ${connected ? "online" : "offline"}`} /><span>{connected ? "Local backend connected" : "Backend offline"}</span></div>
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
