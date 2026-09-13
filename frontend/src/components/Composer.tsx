import { ArrowUp, FileText, Image, Mic, Paperclip, Square, Video, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Attachment, ModelSummary } from "../api/types";
import { generationCapabilityReason, isUsable, supportsGeneration } from "../domain/capabilities";
import { formatBytes } from "../utils/format";

interface ComposerProps {
  model: ModelSummary | null;
  connected: boolean;
  busy: boolean;
  running: boolean;
  attachments: Attachment[];
  onSubmit: (value: string) => void;
  onStop: () => void;
  onAttach: (file: File) => void;
  onRemoveAttachment: (id: string) => void;
}

export function Composer({ model, connected, busy, running, attachments, onSubmit, onStop, onAttach, onRemoveAttachment }: ComposerProps): React.ReactNode {
  const [value, setValue] = useState("");
  const [attachmentMenu, setAttachmentMenu] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const generationSupported = supportsGeneration(model);
  const canSubmit = connected && Boolean(model) && generationSupported && value.trim().length > 0;
  const accepted: string[] = [];
  if (isUsable(model, "vision")) accepted.push("image/*");
  if (isUsable(model, "video")) accepted.push("video/*");
  if (isUsable(model, "audio")) accepted.push("audio/*");
  if (isUsable(model, "native_file_input") || isUsable(model, "extracted_text_input")) accepted.push(".txt,.md,.pdf,.docx");

  useEffect(() => {
    if (!textarea.current) return;
    textarea.current.style.height = "0px";
    textarea.current.style.height = `${String(Math.min(textarea.current.scrollHeight, 180))}px`;
  }, [value]);

  const send = (): void => {
    if (!canSubmit || running || busy) return;
    onSubmit(value);
    setValue("");
  };

  return (
    <div className="composer-shell">
      {attachments.length > 0 && (
        <div className="attachment-strip">
          {attachments.map((attachment) => (
            <div className={`attachment-chip ${attachment.status}`} key={attachment.id}>
              <div className="attachment-icon">{attachment.kind === "image" ? <Image size={15} /> : attachment.kind === "audio" ? <Mic size={15} /> : attachment.kind === "video" ? <Video size={15} /> : <FileText size={15} />}</div>
              <div><strong>{attachment.name}</strong><span>{formatBytes(attachment.sizeBytes)} · {attachment.status}</span>{attachment.detail && <span className="attachment-error">{attachment.detail}</span>}</div>
              <button aria-label={`Remove ${attachment.name}`} className="bare-button" onClick={() => onRemoveAttachment(attachment.id)} type="button"><X size={14} /></button>
            </div>
          ))}
        </div>
      )}
      <div className="composer">
        <div className="attach-wrap">
          <button
            aria-expanded={attachmentMenu}
            aria-label="Attach a file"
            className="composer-tool"
            disabled={accepted.length === 0 || running || busy}
            onClick={() => setAttachmentMenu((current) => !current)}
            title={accepted.length ? "Attach supported media or a document" : "This model has no supported attachment path"}
            type="button"
          ><Paperclip size={18} /></button>
          {attachmentMenu && (
            <div className="popover attachment-menu">
              <p>Accepted for this model</p>
              <div className="attachment-capabilities">
                <span className={isUsable(model, "vision") ? "supported" : ""}><Image size={14} /> Image</span>
                <span className={isUsable(model, "video") ? "supported" : ""}><Video size={14} /> Video</span>
                <span className={isUsable(model, "audio") ? "supported" : ""}><Mic size={14} /> Audio</span>
                <span className={isUsable(model, "native_file_input") || isUsable(model, "extracted_text_input") ? "supported" : ""}><FileText size={14} /> Document</span>
              </div>
              <button className="button primary compact" onClick={() => { fileInput.current?.click(); setAttachmentMenu(false); }} type="button">Choose file</button>
            </div>
          )}
          <input
            accept={accepted.join(",")}
            className="visually-hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onAttach(file);
              event.target.value = "";
            }}
            ref={fileInput}
            type="file"
          />
        </div>
        <textarea
          aria-label="Message"
          disabled={!connected || running || busy || !generationSupported}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
          placeholder={!connected ? "Start the local backend to chat" : !model ? "Select a model" : !generationSupported ? "This model does not generate text" : "Message your local model…"}
          ref={textarea}
          rows={1}
          value={value}
        />
        {running ? (
          <button aria-label="Stop generation" className="send-button stop" onClick={onStop} title="Stop and preserve partial output" type="button"><Square fill="currentColor" size={14} /></button>
        ) : (
          <button aria-label="Send message" className="send-button" disabled={!canSubmit || busy} onClick={send} type="button"><ArrowUp size={18} /></button>
        )}
      </div>
      <div className="composer-meta">
        <span>{model ? `${model.name} · ${model.lifecycle}` : "No model selected"}</span>
        {model && !generationSupported && <span className="warning-copy">{generationCapabilityReason(model)}</span>}
        <span className="keyboard-hint"><kbd>Enter</kbd> send · <kbd>Shift Enter</kbd> newline</span>
      </div>
    </div>
  );
}
