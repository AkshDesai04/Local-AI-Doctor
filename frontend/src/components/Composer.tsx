import { ArrowUp, BrainCircuit, FileText, Image, Mic, Paperclip, SlidersHorizontal, Square, Video, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Attachment, GenerationSettings, ModelSummary } from "../api/types";
import { capabilityReason, generationCapabilityReason, isUsable, supportsGeneration } from "../domain/capabilities";
import { formatBytes } from "../utils/format";

interface ComposerProps {
  model: ModelSummary | null;
  connected: boolean;
  busy: boolean;
  running: boolean;
  attachments: Attachment[];
  settings: GenerationSettings;
  onSettingsChange: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  onSubmit: (value: string) => void;
  onStop: () => void;
  onAttach: (file: File) => void;
  onRemoveAttachment: (id: string) => void;
}

function PromptRange({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}): React.ReactNode {
  return (
    <label className="prompt-range" title={hint}>
      <span><span>{label}</span><output>{String(value)}</output></span>
      <input aria-label={label} max={max} min={min} onChange={(event) => onChange(Number(event.target.value))} step={step} type="range" value={value} />
    </label>
  );
}

export function Composer({ model, connected, busy, running, attachments, settings, onSettingsChange, onSubmit, onStop, onAttach, onRemoveAttachment }: ComposerProps): React.ReactNode {
  const [value, setValue] = useState("");
  const [attachmentMenu, setAttachmentMenu] = useState(false);
  const [promptControlsOpen, setPromptControlsOpen] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const generationSupported = supportsGeneration(model);
  const reasoningSupported = isUsable(model, "reasoning_segments");
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
            onClick={() => {
              setAttachmentMenu((current) => !current);
              setPromptControlsOpen(false);
            }}
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
        <div className="composer-settings-wrap">
          <button
            aria-expanded={promptControlsOpen}
            aria-label="Prompt controls"
            className={`composer-tool ${promptControlsOpen ? "active" : ""}`}
            disabled={!connected || running || busy || !generationSupported}
            onClick={() => {
              setPromptControlsOpen((current) => !current);
              setAttachmentMenu(false);
            }}
            title="Tune reasoning and sampling for the next response"
            type="button"
          ><SlidersHorizontal size={18} /></button>
          {promptControlsOpen && (
            <div aria-label="Prompt controls panel" className="popover prompt-controls-popover">
              <div className="prompt-controls-heading"><strong>Next response</strong><small>Applied before you send</small></div>
              <label className={`prompt-reasoning-toggle ${reasoningSupported ? "" : "unsupported"}`} title={reasoningSupported ? "Ask the model's chat template to enable or disable its thinking mode." : capabilityReason(model, "reasoning_segments")}>
                <span><BrainCircuit size={15} /><span><strong>Reasoning</strong><small>{reasoningSupported ? "Use the model's thinking mode" : "Not exposed by this model"}</small></span></span>
                <input aria-label="Reason before answering" checked={reasoningSupported && settings.reasoning !== false} disabled={!reasoningSupported} onChange={(event) => updateSetting("reasoning", event.target.checked)} type="checkbox" />
              </label>
              <PromptRange hint="Higher values make token selection more varied; zero uses greedy decoding." label="Temperature" max={5} min={0} onChange={(next) => updateSetting("temperature", next)} step={0.05} value={settings.temperature} />
              <PromptRange hint="Keep only the K highest-scoring tokens. Zero disables this filter." label="Top K" max={Math.max(200, settings.topK)} min={0} onChange={(next) => updateSetting("topK", next)} step={1} value={settings.topK} />
              <PromptRange hint="Keep the smallest token set whose cumulative probability reaches this value." label="Top P" max={1} min={0.01} onChange={(next) => updateSetting("topP", next)} step={0.01} value={settings.topP} />
            </div>
          )}
        </div>
        {running ? (
          <button aria-label="Stop generation" className="send-button stop" onClick={onStop} title="Stop and preserve partial output" type="button"><Square fill="currentColor" size={14} /></button>
        ) : (
          <button aria-label="Send message" className="send-button" disabled={!canSubmit || busy} onClick={send} type="button"><ArrowUp size={18} /></button>
        )}
      </div>
      <div className="composer-meta">
        <span>{model ? `${model.name} · ${model.lifecycle}` : "No model selected"}</span>
        {model && !generationSupported && <span className="warning-copy">{generationCapabilityReason(model)}</span>}
        <span className="prompt-setting-summary" title="Sampling settings for the next response">{reasoningSupported ? settings.reasoning !== false ? "Reasoning on" : "Reasoning off" : "Reasoning unavailable"} · T {String(settings.temperature)} · K {String(settings.topK)} · P {String(settings.topP)}</span>
        <span className="keyboard-hint"><kbd>Enter</kbd> send · <kbd>Shift Enter</kbd> newline</span>
      </div>
    </div>
  );
}
