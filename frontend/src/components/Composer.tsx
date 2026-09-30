import { ArrowUp, FileText, Image, Mic, Paperclip, ScrollText, SlidersHorizontal, Square, Video, X } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import type { Attachment, GenerationSettings, ModelSummary } from "../api/types";
import { capabilityReason, generationCapabilityReason, isUsable, supportsGeneration } from "../domain/capabilities";
import { SAMPLING_LIMITS } from "../domain/sampling";
import { formatBytes } from "../utils/format";
import { Badge, Button, Field, IconButton, Popover, Slider, Switch, Textarea } from "./ui";

interface ComposerProps {
  model: ModelSummary | null;
  connected: boolean;
  busy: boolean;
  running: boolean;
  attachments: Attachment[];
  settings: GenerationSettings;
  /** Rendered at the end of the toolbar, beside the send button. */
  contextMeter?: ReactNode;
  onSettingsChange: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  onOpenControls?: () => void;
  onSubmit: (value: string) => void;
  onStop: () => void;
  onAttach: (file: File) => void;
  onRemoveAttachment: (id: string) => void;
  systemPrompt: string;
  onSystemPromptChange: (value: string) => void;
}

// UI-only soft cap; the backend bounds the prompt by limits.prompt_bytes.
const SYSTEM_PROMPT_MAX_CHARACTERS = 32_000;

function AttachmentIcon({ kind }: { kind: Attachment["kind"] }): React.ReactNode {
  if (kind === "image") return <Image size={15} />;
  if (kind === "audio") return <Mic size={15} />;
  if (kind === "video") return <Video size={15} />;
  return <FileText size={15} />;
}

export function Composer({ model, connected, busy, running, attachments, settings, contextMeter, onSettingsChange, onOpenControls, onSubmit, onStop, onAttach, onRemoveAttachment, systemPrompt, onSystemPromptChange }: ComposerProps): React.ReactNode {
  const [value, setValue] = useState("");
  const [attachmentMenu, setAttachmentMenu] = useState(false);
  const [promptControlsOpen, setPromptControlsOpen] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const attachButton = useRef<HTMLButtonElement>(null);
  const controlsButton = useRef<HTMLButtonElement>(null);
  const generationSupported = supportsGeneration(model);
  const reasoningSupported = isUsable(model, "reasoning_segments");
  const canSubmit = connected && Boolean(model) && generationSupported && value.trim().length > 0;
  // Native file input covers the media kinds below; documents need text extraction.
  const documentSupported = isUsable(model, "extracted_text_input");
  const accepted: string[] = [];
  if (isUsable(model, "vision")) accepted.push("image/*");
  if (isUsable(model, "video")) accepted.push("video/*");
  if (isUsable(model, "audio")) accepted.push("audio/*");
  if (documentSupported) accepted.push(".txt,.md,.pdf,.docx");

  useEffect(() => {
    if (!textarea.current) return;
    textarea.current.style.height = "0px";
    textarea.current.style.height = `${String(Math.min(textarea.current.scrollHeight, 200))}px`;
  }, [value]);

  useEffect(() => {
    if (!connected || running || busy || accepted.length === 0) setAttachmentMenu(false);
    if (!connected || !generationSupported || running || busy) setPromptControlsOpen(false);
  }, [accepted.length, busy, connected, generationSupported, running]);

  const updateSetting = <K extends keyof GenerationSettings>(key: K, next: GenerationSettings[K]): void => {
    onSettingsChange((current) => ({ ...current, [key]: next }));
  };

  const send = (): void => {
    if (!canSubmit || running || busy) return;
    onSubmit(value);
    setValue("");
  };

  const reasoningSummary = reasoningSupported ? settings.reasoning !== false ? "Reasoning on" : "Reasoning off" : "Reasoning unavailable";

  return (
    <div className="composer-shell">
      {attachments.length > 0 && (
        <div className="attachment-strip">
          {attachments.map((attachment) => (
            <div className={`attachment-chip ${attachment.status}`} key={attachment.id}>
              <span aria-hidden="true" className="attachment-icon"><AttachmentIcon kind={attachment.kind} /></span>
              <div className="attachment-copy">
                <strong>{attachment.name}</strong>
                <span>{formatBytes(attachment.sizeBytes)} · {attachment.status}</span>
                {attachment.detail && <span className="attachment-error" title={attachment.detail}>{attachment.detail}</span>}
              </div>
              <IconButton icon={<X size={13} />} label={`Remove ${attachment.name}`} onClick={() => onRemoveAttachment(attachment.id)} size="sm" />
            </div>
          ))}
        </div>
      )}
      <div className="composer">
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
        <div className="composer-toolbar">
          <div className="composer-tools">
            <IconButton
              aria-expanded={attachmentMenu}
              disabled={accepted.length === 0 || running || busy}
              icon={<Paperclip size={16} />}
              label="Attach a file"
              onClick={() => {
                setAttachmentMenu((current) => !current);
                setPromptControlsOpen(false);
              }}
              ref={attachButton}
              title={accepted.length ? "Attach supported media or a document" : "This model has no supported attachment path"}
            />
            <IconButton
              aria-expanded={promptControlsOpen}
              disabled={!connected || running || busy || !generationSupported}
              icon={<SlidersHorizontal size={16} />}
              label="Prompt controls"
              onClick={() => {
                setPromptControlsOpen((current) => !current);
                setAttachmentMenu(false);
              }}
              ref={controlsButton}
              title="Tune reasoning, sampling, and the system prompt for the next response"
            />
            {systemPrompt.trim() && (
              <span aria-label="System prompt active" className="system-prompt-indicator" role="img" title="System prompt active: applied to every response in this chat">
                <Badge icon={<ScrollText aria-hidden="true" size={12} />} tone="accent">System prompt</Badge>
              </span>
            )}
            <span className="composer-summary" title="Sampling settings for the next response">{reasoningSummary} · T {String(settings.temperature)} · K {String(settings.topK)} · P {String(settings.topP)}</span>
          </div>
          <div className="composer-end">
            {contextMeter}
            {running ? (
              <button aria-label="Stop generation" className="send-button stop" onClick={onStop} title="Stop and preserve partial output" type="button"><Square fill="currentColor" size={12} /></button>
            ) : (
              <button aria-label="Send message" className="send-button" disabled={!canSubmit || busy} onClick={send} title="Send (Enter)" type="button"><ArrowUp size={17} /></button>
            )}
          </div>
        </div>

        <Popover anchorRef={attachButton} className="composer-popover attachment-menu" label="Attachment options" onClose={() => setAttachmentMenu(false)} open={attachmentMenu}>
          <p className="popover-title">Accepted for this model</p>
          <ul className="attachment-capabilities">
            <li className={isUsable(model, "vision") ? "supported" : ""}><Image size={14} /> Image</li>
            <li className={isUsable(model, "video") ? "supported" : ""}><Video size={14} /> Video</li>
            <li className={isUsable(model, "audio") ? "supported" : ""}><Mic size={14} /> Audio</li>
            <li className={documentSupported ? "supported" : ""}><FileText size={14} /> Document</li>
          </ul>
          <Button block onClick={() => { fileInput.current?.click(); setAttachmentMenu(false); }} size="sm" variant="primary">Choose file</Button>
        </Popover>

        <Popover anchorRef={controlsButton} className="composer-popover prompt-controls" label="Prompt controls panel" onClose={() => setPromptControlsOpen(false)} open={promptControlsOpen}>
          <div className="popover-head"><strong>Next response</strong><span>Applied when you send</span></div>
          <Switch
            aria-label="Reason before answering"
            checked={reasoningSupported && settings.reasoning !== false}
            description={reasoningSupported ? "Use the model’s thinking mode" : "Not exposed by this model"}
            disabled={!reasoningSupported}
            label="Reasoning"
            onChange={(event) => updateSetting("reasoning", event.target.checked)}
            title={reasoningSupported ? "Ask the model's chat template to enable or disable its thinking mode." : capabilityReason(model, "reasoning_segments")}
          />
          <Slider {...SAMPLING_LIMITS.temperature} label="Temperature" onValueChange={(next) => updateSetting("temperature", next)} title="Higher values make token selection more varied; zero uses greedy decoding." value={settings.temperature} />
          <Slider {...SAMPLING_LIMITS.topK} label="Top K" onValueChange={(next) => updateSetting("topK", next)} title="Keep only the K highest-scoring tokens. Zero disables this filter." value={settings.topK} />
          <Slider {...SAMPLING_LIMITS.topP} label="Top P" onValueChange={(next) => updateSetting("topP", next)} title="Keep the smallest token set whose cumulative probability reaches this value." value={settings.topP} />
          <Field
            addon={<span className="field-count">{String(systemPrompt.length)} / {String(SYSTEM_PROMPT_MAX_CHARACTERS)}</span>}
            className="system-prompt-field"
            hint="Applies to every response in this chat. Templates without a system role receive it inside the first user message."
            label="System prompt"
          >
            <Textarea maxLength={SYSTEM_PROMPT_MAX_CHARACTERS} onChange={(event) => onSystemPromptChange(event.target.value)} placeholder="Optional instructions applied to every response in this chat" rows={3} value={systemPrompt} />
          </Field>
          {onOpenControls && (
            <button className="popover-link" onClick={() => { setPromptControlsOpen(false); onOpenControls(); }} type="button">All generation controls…</button>
          )}
        </Popover>
        <input
          accept={accepted.join(",")}
          className="visually-hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onAttach(file);
            event.target.value = "";
          }}
          ref={fileInput}
          tabIndex={-1}
          type="file"
        />
      </div>
      <div className="composer-meta">
        <span className="composer-model">{model ? `${model.name} · ${model.lifecycle}` : "No model selected"}</span>
        {model && !generationSupported && <span className="warning-copy">{generationCapabilityReason(model)}</span>}
        <span className="keyboard-hint"><kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> newline</span>
      </div>
    </div>
  );
}
