import {
  Activity,
  Bot,
  Boxes,
  Bug,
  Check,
  Copy,
  Database,
  Edit3,
  Eye,
  FileText,
  Image,
  LoaderCircle,
  GitBranch,
  RefreshCcw,
  ScrollText,
  Sparkles,
  User,
  Video,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { Message, ModelSummary, RunDetails, TokenEvent } from "../api/types";
import { isUsable, supportsGeneration } from "../domain/capabilities";
import { displayTokenText, formatDuration, formatNumber, formatPercent, tokenTextHint } from "../utils/format";
import { cleanAssistantOutput, isTerminationToken, splitAssistantOutput } from "../utils/markdown";
import { MarkdownMessage } from "./MarkdownMessage";
import { Badge, Button, Callout, EmptyState, IconButton, Select, Textarea } from "./ui";

export type NerdMetric = "rawProbability" | "samplingProbability" | "surprise" | "latency" | "reasoning";

interface ChatViewProps {
  connected: boolean;
  booting: boolean;
  model: ModelSummary | null;
  messages: Message[];
  messagesLoading: boolean;
  run: RunDetails | null;
  runningRunId: string | null;
  streamConnected: boolean;
  nerdMode: boolean;
  nerdMetric: NerdMetric;
  selectedToken: number | null;
  onNerdMetricChange: (metric: NerdMetric) => void;
  onSelectToken: (index: number) => void;
  onOpenInspector: () => void;
  onInspectRun: (id: string) => void;
  onRetry: (message: Message) => void;
  onBranch: (content: string, parentMessageId: string | null) => void;
  onNavigate: (view: "embeddings" | "models") => void;
  systemPrompt: string;
}

function nerdValue(token: TokenEvent, metric: NerdMetric, classification: TokenEvent["reasoningSegment"] = token.reasoningSegment): number | undefined {
  if (metric === "rawProbability") return token.rawProbability;
  if (metric === "samplingProbability") return token.samplingProbability;
  if (metric === "surprise") return token.surprise;
  if (metric === "latency") return token.timing?.decodeMs;
  return classification === "reasoning" ? 1 : classification === "answer" ? 0.35 : 0;
}

type ReasoningClass = TokenEvent["reasoningSegment"];
type ReasoningDelimiter = "start" | "end";

interface PositionedTokenPart {
  position: number;
  token: TokenEvent;
  text: string;
  start: number;
  end: number;
  classification: ReasoningClass;
  delimiter: boolean;
  delimiterKind?: ReasoningDelimiter;
  reasoning: boolean;
}

interface NerdTokenGroup {
  reasoning: boolean;
  parts: PositionedTokenPart[];
}

type UnclassifiedTokenPart = Omit<PositionedTokenPart, "reasoning">;

const reasoningDelimiterPattern = /<think(?:\s[^>]*)?>|<\/think\s*>/giu;

function fallbackTokenParts(token: TokenEvent, position: number): UnclassifiedTokenPart[] {
  const text = token.displayText || token.piece;
  return [{ position, token, text, start: 0, end: Array.from(text).length, classification: token.reasoningSegment, delimiter: false }];
}

function exactTokenParts(token: TokenEvent, position: number): UnclassifiedTokenPart[] {
  const text = token.displayText || token.piece;
  if (!token.reasoningSlices?.length) return fallbackTokenParts(token, position);
  const codePoints = Array.from(text);
  const parts: UnclassifiedTokenPart[] = [];
  let cursor = 0;
  const slices = [...token.reasoningSlices].sort((left, right) => left.start - right.start);
  for (const slice of slices) {
    const start = Math.max(cursor, Math.min(codePoints.length, slice.start));
    const end = Math.max(start, Math.min(codePoints.length, slice.end));
    if (start > cursor) parts.push({ position, token, text: codePoints.slice(cursor, start).join(""), start: cursor, end: start, classification: token.reasoningSegment, delimiter: false });
    if (end > start) parts.push({ position, token, text: codePoints.slice(start, end).join(""), start, end, classification: slice.classification, delimiter: slice.delimiter });
    cursor = end;
  }
  if (cursor < codePoints.length) parts.push({ position, token, text: codePoints.slice(cursor).join(""), start: cursor, end: codePoints.length, classification: token.reasoningSegment, delimiter: false });
  return parts.length > 0 ? parts : fallbackTokenParts(token, position);
}

interface ReasoningDelimiterSpan {
  start: number;
  end: number;
  kind: ReasoningDelimiter;
}

function streamDelimiterSpans(text: string): ReasoningDelimiterSpan[] {
  const spans: ReasoningDelimiterSpan[] = [];
  for (const match of text.matchAll(reasoningDelimiterPattern)) {
    const utf16Start = match.index ?? 0;
    const start = Array.from(text.slice(0, utf16Start)).length;
    spans.push({
      start,
      end: start + Array.from(match[0]).length,
      kind: match[0].startsWith("</") ? "end" : "start",
    });
  }
  return spans;
}

function markStreamDelimiters(parts: UnclassifiedTokenPart[], parseProtocol: boolean): UnclassifiedTokenPart[] {
  if (!parseProtocol && !parts.some((part) => part.delimiter)) return parts;
  const explicitRanges: Array<{ start: number; end: number }> = [];
  let explicitCursor = 0;
  for (const part of parts) {
    const partEnd = explicitCursor + Array.from(part.text).length;
    if (part.delimiter) explicitRanges.push({ start: explicitCursor, end: partEnd });
    explicitCursor = partEnd;
  }
  const explicitlyDelimited = (span: ReasoningDelimiterSpan): boolean => {
    let coveredThrough = span.start;
    for (const range of explicitRanges) {
      if (range.end <= coveredThrough) continue;
      if (range.start > coveredThrough) return false;
      coveredThrough = Math.max(coveredThrough, range.end);
      if (coveredThrough >= span.end) return true;
    }
    return false;
  };
  const spans = streamDelimiterSpans(parts.map((part) => part.text).join(""))
    .filter((span) => parseProtocol || explicitlyDelimited(span));
  if (!spans.length) return parts;

  const marked: UnclassifiedTokenPart[] = [];
  let streamCursor = 0;
  for (const part of parts) {
    const characters = Array.from(part.text);
    const streamEnd = streamCursor + characters.length;
    const overlapping = spans.filter((span) => span.start < streamEnd && span.end > streamCursor);
    const boundaries = new Set([streamCursor, streamEnd]);
    for (const span of overlapping) {
      boundaries.add(Math.max(streamCursor, span.start));
      boundaries.add(Math.min(streamEnd, span.end));
    }
    const ordered = [...boundaries].sort((left, right) => left - right);
    for (let index = 0; index < ordered.length - 1; index += 1) {
      const start = ordered[index];
      const end = ordered[index + 1];
      if (start === undefined || end === undefined || end <= start) continue;
      const delimiterSpan = overlapping.find((span) => start >= span.start && end <= span.end);
      const localStart = start - streamCursor;
      const localEnd = end - streamCursor;
      marked.push({
        ...part,
        text: characters.slice(localStart, localEnd).join(""),
        start: part.start + localStart,
        end: part.start + localEnd,
        classification: delimiterSpan ? "reasoning" : part.classification,
        delimiter: delimiterSpan ? true : part.delimiter,
        delimiterKind: delimiterSpan?.kind,
      });
    }
    streamCursor = streamEnd;
  }
  return marked;
}

function positionedTokenParts(tokens: TokenEvent[], reasoningPrimed: boolean): PositionedTokenPart[] {
  const streamText = tokens.map((token) => token.displayText || token.piece).join("");
  const parseFallbackProtocol = reasoningPrimed || /^\s*<think(?:\s[^>]*)?>/iu.test(streamText);
  const rawParts = markStreamDelimiters(
    tokens.flatMap((token, position) => exactTokenParts(token, position)),
    parseFallbackProtocol,
  );
  const parts: PositionedTokenPart[] = [];
  let insideReasoning = reasoningPrimed;
  for (let index = 0; index < rawParts.length;) {
    const part = rawParts[index];
    if (!part) break;
    if (part.delimiter) {
      if (part.delimiterKind) {
        parts.push({ ...part, delimiterKind: part.delimiterKind, reasoning: true });
        insideReasoning = part.delimiterKind === "start";
        index += 1;
        continue;
      }
      let delimiterEnd = index + 1;
      while (rawParts[delimiterEnd]?.delimiter && !rawParts[delimiterEnd]?.delimiterKind) delimiterEnd += 1;
      const delimiterParts = rawParts.slice(index, delimiterEnd);
      const delimiterText = delimiterParts.map((item) => item.text).join("");
      const delimiterKind: ReasoningDelimiter = /<\/think\s*>/iu.test(delimiterText)
        ? "end"
        : /<think(?:\s[^>]*)?>/iu.test(delimiterText)
          ? "start"
          : insideReasoning ? "end" : "start";
      for (const item of delimiterParts) parts.push({ ...item, delimiterKind, reasoning: true });
      insideReasoning = delimiterKind === "start";
      index = delimiterEnd;
      continue;
    }
    const reasoning = part.classification === "reasoning" || (part.classification === "unknown" && insideReasoning);
    parts.push({ ...part, reasoning });
    if (part.classification === "reasoning") insideReasoning = true;
    else if (part.classification === "answer") insideReasoning = false;
    index += 1;
  }
  return parts;
}

function groupNerdTokens(tokens: TokenEvent[], reasoningPrimed: boolean): NerdTokenGroup[] {
  const groups: NerdTokenGroup[] = [];
  for (const part of positionedTokenParts(tokens, reasoningPrimed)) {
    const current = groups.at(-1);
    if (!current || current.reasoning !== part.reasoning) groups.push({ reasoning: part.reasoning, parts: [part] });
    else current.parts.push(part);
  }
  return groups;
}

function nerdTokenLabel(part: PositionedTokenPart): string {
  const label = displayTokenText(part.text || part.token.piece) || "empty token";
  if (part.delimiterKind === "end") return `End reasoning token ${String(part.token.index)} ${label}`;
  if (part.delimiterKind === "start") return `Start reasoning token ${String(part.token.index)} ${label}`;
  if (isTerminationToken(part.text)) return `Termination token ${String(part.token.index)} ${label}`;
  return `Token ${String(part.token.index)} ${label}`;
}

function NerdResponse({
  tokens,
  metric,
  reasoningPrimed,
  selectedToken,
  onSelectToken,
  onOpenInspector,
}: {
  tokens: TokenEvent[];
  metric: NerdMetric;
  reasoningPrimed: boolean;
  selectedToken: number | null;
  onSelectToken: (index: number) => void;
  onOpenInspector: () => void;
}): React.ReactNode {
  const visibleStart = Math.max(0, tokens.length - 1200);
  const groups = groupNerdTokens(tokens, reasoningPrimed);
  const values = groups.flatMap((group) => group.parts.map((part) => nerdValue(part.token, metric, part.classification))).filter((value): value is number => value !== undefined && Number.isFinite(value));
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const range = maximum - minimum || 1;
  const selectedAttribution = tokens.find((token) => token.index === selectedToken)?.attentionAttribution;
  const generatedAttention = new Map(
    selectedAttribution?.sourceTokens
      .filter((source) => source.sourceKind === "generated" && source.generatedTokenIndex !== undefined)
      .map((source) => [source.generatedTokenIndex as number, source.weight]) ?? [],
  );
  const peakAttention = selectedAttribution?.sourceTokens.length
    ? Math.max(...selectedAttribution.sourceTokens.map((source) => source.weight))
    : 0;
  const attentionActive = selectedAttribution !== undefined;
  if (!tokens.length) return <span className="stream-caret" aria-label="Waiting for first token" />;
  const hasEmittedStart = groups.some((group) => group.parts.some((part) => part.delimiterKind === "start"));
  const promptMarkerGroup = reasoningPrimed && !hasEmittedStart ? groups.findIndex((group) => group.reasoning) : -1;
  const renderTokens = (items: PositionedTokenPart[]): React.ReactNode => {
    const collapsed = items.filter(({ position }) => position < visibleStart);
    const visible = items.filter(({ position }) => position >= visibleStart);
    return (
      <>
        {collapsed.length > 0 && <span className="nerd-collapsed-prefix" title="Older tokens remain individually available in the virtualized token table.">{collapsed.map((part) => part.text).join("")}<small>{String(new Set(collapsed.map((part) => part.token.index)).size)} earlier token boundaries collapsed for display performance</small></span>}
        {visible.map((part) => {
          const { token } = part;
          const value = nerdValue(token, metric, part.classification);
          const normalized = value === undefined ? 0 : (value - minimum) / range;
          const attentionWeight = generatedAttention.get(token.index);
          const attentionStrength = attentionWeight === undefined || peakAttention <= 0 ? 0 : attentionWeight / peakAttention;
          const attentionClass = !attentionActive
            ? ""
            : selectedToken === token.index
              ? "attention-target"
              : attentionWeight !== undefined
                ? "attention-source"
                : "attention-muted";
          const tokenLabel = displayTokenText(part.text || token.piece) || "∅";
          const attentionHint = attentionWeight === undefined
            ? ""
            : `\nMean attention weight ${formatNumber(attentionWeight, 8)} (${formatPercent(attentionWeight, 5)} of the normalized row)\nAttention is not a causal contribution score.`;
          return (
            <button
              aria-label={`${nerdTokenLabel(part)}${attentionWeight === undefined ? "" : `, mean attention ${formatPercent(attentionWeight, 5)}`}`}
              aria-pressed={selectedToken === token.index}
              className={`nerd-token ${selectedToken === token.index ? "selected" : ""} ${attentionClass} segment-${part.classification}`}
              key={`${String(token.index)}:${String(part.start)}:${String(part.end)}`}
              onClick={() => { onSelectToken(token.index); onOpenInspector(); }}
              style={{
                "--metric": String(normalized),
                "--attention-fill": `${String(16 + attentionStrength * 70)}%`,
                "--attention-border": `${String(35 + attentionStrength * 65)}%`,
                "--attention-text": `${String(55 + attentionStrength * 45)}%`,
              } as React.CSSProperties}
              title={`Token #${String(token.index)}\n${tokenTextHint(token.piece, part.text || token.displayText)}\nID ${String(token.tokenId)} · raw p ${formatPercent(token.rawProbability, 3)} · sampler p ${formatPercent(token.samplingProbability, 3)}\n${formatDuration(token.timing?.decodeMs)} decode · ${token.reasoningSegment}${attentionHint}\nClick to inspect this token.`}
              type="button"
            ><span className="token-text">{tokenLabel}</span><span className="token-index">{String(token.index)}</span></button>
          );
        })}
      </>
    );
  };
  return (
    <div className={`nerd-response ${attentionActive ? "attention-active" : ""}`} aria-label={`Tokenized response colored by ${attentionActive ? "context attention" : metric}`}>
      {attentionActive && <Callout className="attention-mode-banner" icon={Activity} role="status">Attention view for token #{String(selectedToken)}. Earlier generated tokens are shaded by mean attention weight; the exact prompt and history context map is in the inspector.</Callout>}
      {groups.map((group, groupIndex) => group.reasoning ? (
        <details className="reasoning-disclosure nerd-reasoning-disclosure" key={`reasoning-${String(groupIndex)}`}>
          <summary className="reasoning-summary">Thinking… <small>{String(new Set(group.parts.map((part) => part.token.index)).size)} tokens</small></summary>
          <div className="reasoning-content nerd-reasoning-content">
            {groupIndex === promptMarkerGroup && <code aria-label="Prompt-primed reasoning start" className="nerd-boundary-token start">Reasoning started by the prompt · no opening token emitted</code>}
            <div>{renderTokens(group.parts)}</div>
          </div>
        </details>
      ) : <div className="nerd-answer-segment" key={`answer-${String(groupIndex)}`}>{renderTokens(group.parts)}</div>)}
    </div>
  );
}

function NerdRawResponse({ content, reasoningPrimed }: { content: string; reasoningPrimed: boolean }): React.ReactNode {
  const output = splitAssistantOutput(content, reasoningPrimed);
  return (
    <div className="nerd-raw-response">
      {output.hasReasoning && <details className="reasoning-disclosure nerd-reasoning-disclosure">
        <summary className="reasoning-summary">Thinking…</summary>
        <div className="reasoning-content nerd-reasoning-content">
          {output.openingReasoningToken
            ? <code aria-label="Start reasoning token" className="nerd-boundary-token start">{output.openingReasoningToken}</code>
            : reasoningPrimed && <code aria-label="Prompt-primed reasoning start" className="nerd-boundary-token start">Reasoning started by the prompt · no opening token emitted</code>}
          <pre className="nerd-raw-fallback">{output.reasoning}</pre>
          {output.closingReasoningToken && <code aria-label="End reasoning token" className="nerd-boundary-token end">{output.closingReasoningToken}</code>}
        </div>
      </details>}
      {output.answer && <pre className="nerd-raw-fallback">{output.answer}</pre>}
      {output.terminationTokens.length > 0 && <div className="nerd-termination-tokens" aria-label="Termination tokens">{output.terminationTokens.map((token, index) => <code key={`${token}:${String(index)}`}>{token}</code>)}</div>}
    </div>
  );
}

function AttachmentItem({ attachment }: { attachment: NonNullable<Message["attachments"]>[number] }): React.ReactNode {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const previewable = attachment.status === "ready" && (attachment.kind === "image" || attachment.kind === "video");
  useEffect(() => {
    if (!previewable) return undefined;
    let disposed = false;
    let createdUrl: string | undefined;
    void api.attachmentContent(attachment.id).then((blob) => {
      if (!blob.type.startsWith(`${attachment.kind}/`)) return;
      createdUrl = URL.createObjectURL(blob);
      if (disposed) {
        URL.revokeObjectURL(createdUrl);
        return;
      }
      setObjectUrl(createdUrl);
    }).catch(() => {
      // The compact attachment card remains available if its persisted preview cannot be loaded.
    });
    return () => {
      disposed = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [attachment.id, attachment.kind, previewable]);

  return (
    <div className={`message-attachment ${objectUrl ? "has-preview" : ""}`}>
      {objectUrl && attachment.kind === "image" && <img alt={`Preview of ${attachment.name}`} className="message-attachment-preview" loading="lazy" src={objectUrl} />}
      {objectUrl && attachment.kind === "video" && <video aria-label={`Preview of ${attachment.name}`} className="message-attachment-preview" controls preload="metadata" src={objectUrl} />}
      <div className="message-attachment-meta">{attachment.kind === "image" ? <Image size={14} /> : attachment.kind === "video" ? <Video size={14} /> : <FileText size={14} />}<span>{attachment.name}</span><small>{attachment.nativeProcessing ? "native" : "preprocessed"}</small></div>
    </div>
  );
}

function AttachmentPreview({ message }: { message: Message }): React.ReactNode {
  if (!message.attachments?.length) return null;
  return <div className="message-attachments">{message.attachments.map((attachment) => <AttachmentItem attachment={attachment} key={`${attachment.id}:${attachment.kind}:${attachment.status}`} />)}</div>;
}

const ROOT_BRANCH_KEY = "__conversation_root__";
const EMPTY_BRANCH_SELECTION: Record<string, string> = {};

interface MessageGraph {
  byId: Map<string, Message>;
  childrenByParent: Map<string, Message[]>;
}

interface PendingBranch {
  chatId: string;
  parentKey: string;
  knownChildIds: string[];
  fallbackChildId?: string;
}

interface LineageEntry {
  message: Message;
  parentKey: string;
  siblings: Message[];
}

interface BranchControl {
  accessibleLabel: string;
  options: Array<{ id: string; label: string }>;
  onSelect: (id: string) => void;
}

function buildMessageGraph(messages: Message[]): MessageGraph {
  const byId = new Map(messages.map((message) => [message.id, message]));
  const childrenByParent = new Map<string, Message[]>();
  for (const message of messages) {
    const parentKey = message.parentMessageId && message.parentMessageId !== message.id && byId.has(message.parentMessageId)
      ? message.parentMessageId
      : ROOT_BRANCH_KEY;
    const siblings = childrenByParent.get(parentKey) ?? [];
    siblings.push(message);
    childrenByParent.set(parentKey, siblings);
  }
  return { byId, childrenByParent };
}

function newPendingChildId(graph: MessageGraph, pending: PendingBranch | null, chatId: string | null): string | null {
  if (!pending || pending.chatId !== chatId) return null;
  const knownIds = new Set(pending.knownChildIds);
  const children = graph.childrenByParent.get(pending.parentKey) ?? [];
  for (let index = children.length - 1; index >= 0; index -= 1) {
    const child = children[index];
    if (child && !knownIds.has(child.id)) return child.id;
  }
  return null;
}

function resolveLineage(
  graph: MessageGraph,
  selectedChildByParent: Record<string, string>,
  pending: PendingBranch | null,
  chatId: string | null,
): LineageEntry[] {
  const lineage: LineageEntry[] = [];
  const visited = new Set<string>();
  const pendingChild = newPendingChildId(graph, pending, chatId);
  let parentKey = ROOT_BRANCH_KEY;
  while (true) {
    const siblings = graph.childrenByParent.get(parentKey) ?? [];
    if (!siblings.length) break;
    const selectedId = pending?.chatId === chatId && pending.parentKey === parentKey
      ? pendingChild ?? pending.fallbackChildId
      : selectedChildByParent[parentKey];
    const message = siblings.find((candidate) => candidate.id === selectedId) ?? siblings[siblings.length - 1];
    if (!message || visited.has(message.id)) break;
    lineage.push({ message, parentKey, siblings });
    visited.add(message.id);
    parentKey = message.id;
  }
  return lineage;
}

function MessageRow({
  message,
  run,
  nerdMode,
  nerdMetric,
  selectedToken,
  onSelectToken,
  onOpenInspector,
  onInspectRun,
  onRetry,
  onBranch,
  branchControl,
}: {
  message: Message;
  run: RunDetails | null;
  nerdMode: boolean;
  nerdMetric: NerdMetric;
  selectedToken: number | null;
  onSelectToken: (index: number) => void;
  onOpenInspector: () => void;
  onInspectRun: (id: string) => void;
  onRetry: () => void;
  onBranch: (value: string) => void;
  branchControl?: BranchControl;
}): React.ReactNode {
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  const runMatchesMessage = run?.id === message.runId || run?.messageId === message.id;
  const tokenized = nerdMode && message.role === "assistant" && runMatchesMessage;
  const rawNerdFallback = nerdMode && message.role === "assistant" && !tokenized;
  const reasoningPrimed = message.reasoningPrimed === true || Boolean(
    message.role === "assistant"
    && runMatchesMessage
    && run?.tokens.some((token) => token.reasoningSegment === "reasoning")
  );
  const assistantOutput = message.role === "assistant"
    ? splitAssistantOutput(message.content, reasoningPrimed)
    : null;
  const endedWithoutAnswer = message.status === "complete"
    && assistantOutput?.hasReasoning === true
    && assistantOutput.answer.length === 0;
  const copy = (): void => {
    const copyValue = message.role === "assistant" && !nerdMode ? cleanAssistantOutput(message.content, reasoningPrimed) : message.content;
    void navigator.clipboard.writeText(copyValue).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    });
  };
  return (
    <article className={`message-row ${message.role} ${message.status}`}>
      <div aria-hidden="true" className="message-avatar">{message.role === "assistant" ? <Sparkles size={15} /> : message.role === "user" ? <User size={15} /> : <Bot size={15} />}</div>
      <div className="message-body">
        <div className="message-label">
          <strong>{message.role === "assistant" ? "Local model" : message.role === "user" ? "You" : "System"}</strong>
          {message.status !== "complete" && <Badge className="message-status" icon={message.status === "streaming" ? <LoaderCircle className="spin" size={11} /> : undefined} tone={message.status === "failed" ? "danger" : message.status === "cancelled" ? "warning" : "accent"}>{message.status}</Badge>}
          {branchControl && (
            <span className="branch-selector" title={branchControl.accessibleLabel}>
              <GitBranch aria-hidden="true" size={13} />
              <Select aria-label={branchControl.accessibleLabel} controlSize="sm" onChange={(event) => branchControl.onSelect(event.target.value)} value={message.id} variant="ghost">{branchControl.options.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</Select>
            </span>
          )}
        </div>
        <AttachmentPreview message={message} />
        {editing ? (
          <div className="edit-message">
            <Textarea aria-label="Edit message" autoFocus onChange={(event) => setDraft(event.target.value)} rows={4} value={draft} />
            <div className="edit-message-actions">
              <Button icon={<X size={13} />} onClick={() => { setEditing(false); setDraft(message.content); }} size="sm" variant="ghost">Cancel</Button>
              <Button icon={<GitBranch size={13} />} onClick={() => { if (draft.trim()) onBranch(draft.trim()); setEditing(false); }} size="sm" variant="primary">Send as branch</Button>
            </div>
          </div>
        ) : tokenized ? (
          <NerdResponse metric={nerdMetric} onOpenInspector={onOpenInspector} onSelectToken={onSelectToken} reasoningPrimed={reasoningPrimed} selectedToken={selectedToken} tokens={run?.tokens ?? []} />
        ) : rawNerdFallback ? (
          <NerdRawResponse content={message.content} reasoningPrimed={reasoningPrimed} />
        ) : (
          <div className="message-content"><MarkdownMessage assistant={message.role === "assistant"} content={message.content} reasoningPrimed={reasoningPrimed} />{message.status === "streaming" && <span className="stream-caret" />}</div>
        )}
        {endedWithoutAnswer && (
          <Callout className="assistant-output-warning" role="status" tone="warning">
            {runMatchesMessage && run?.metrics?.finishReason === "length"
              ? "No final answer: this response reached its token limit. Increase Max tokens and regenerate."
              : "No final answer was produced. Regenerate the response to try again."}
          </Callout>
        )}
        {message.error && <Callout className="message-error" role="note" tone="danger">{message.error}</Callout>}
        <div className="message-actions">
          <IconButton icon={copied ? <Check size={14} /> : <Copy size={14} />} label="Copy message" onClick={copy} size="sm" title={copied ? "Copied" : "Copy"} />
          {message.role === "user" && <IconButton icon={<Edit3 size={14} />} label="Edit and retry" onClick={() => setEditing(true)} size="sm" />}
          {message.role === "assistant" && <IconButton disabled={message.status !== "complete" || !message.runId} icon={<RefreshCcw size={14} />} label="Regenerate response" onClick={onRetry} size="sm" title={message.status !== "complete" ? "Only completed runs can be replayed" : message.runId ? "Replay this run" : "No persisted run is attached"} />}
          <IconButton icon={<GitBranch size={14} />} label="Branch from message" onClick={() => setEditing(true)} size="sm" title="Branch" />
          {message.runId && <Button aria-label="Open response details" icon={<Activity size={14} />} onClick={() => { onInspectRun(message.runId ?? ""); onOpenInspector(); }} size="sm" title="Response details" variant="ghost">Details</Button>}
        </div>
      </div>
    </article>
  );
}

function EmptyChat({ connected, booting, model, onNavigate }: { connected: boolean; booting: boolean; model: ModelSummary | null; onNavigate: (view: "embeddings" | "models") => void }): React.ReactNode {
  if (booting) return <EmptyState className="chat-empty booting" description="Reading model capabilities and saved conversations…" headingLevel={1} icon={LoaderCircle} size="page" title="Connecting to your local workbench" />;
  if (!connected) {
    return (
      <EmptyState className="chat-empty" description="Start the local service to discover models and restore persisted chats. This interface will not fall back to a remote inference provider." eyebrow="Local only" headingLevel={1} icon={Database} size="page" title="The backend is offline">
        <code className="empty-command">python -m local_ai_doctor</code>
      </EmptyState>
    );
  }
  if (!model) return <EmptyState actions={<Button onClick={() => onNavigate("models")} variant="primary">Open model registry</Button>} className="chat-empty" description="Configure a read-only model root, then rescan it. Unsupported folders remain visible with actionable diagnostics." eyebrow="Model registry is empty" headingLevel={1} icon={Database} size="page" title="Add a local model to begin" />;
  if (!supportsGeneration(model) && isUsable(model, "embeddings")) return <EmptyState actions={<Button onClick={() => onNavigate("embeddings")} variant="primary">Open embeddings</Button>} className="chat-empty" description="Use the embeddings workspace for vectors, similarity, nearest neighbors, and projections." eyebrow="Embedding model selected" headingLevel={1} icon={Boxes} size="page" title="This model maps meaning, not words" />;
  return (
    <EmptyState className="chat-empty" description="Every run can preserve exact token probabilities, timing, context use, and reproducibility details—when the model and selected instrumentation expose them." eyebrow="Local · private · observable" headingLevel={1} icon={Sparkles} size="page" title="What should we inspect?">
      <ul className="prompt-suggestions">
        <li>Try a short prompt to establish a baseline</li>
        <li>Set a seed, then compare two sampling runs</li>
        <li>Enable Nerd Mode to inspect token boundaries</li>
      </ul>
    </EmptyState>
  );
}

export function ChatView({
  connected,
  booting,
  model,
  messages,
  messagesLoading,
  run,
  runningRunId,
  streamConnected,
  nerdMode,
  nerdMetric,
  selectedToken,
  onNerdMetricChange,
  onSelectToken,
  onOpenInspector,
  onInspectRun,
  onRetry,
  onBranch,
  onNavigate,
  systemPrompt,
}: ChatViewProps): React.ReactNode {
  const bottom = useRef<HTMLDivElement>(null);
  const graph = useMemo(() => buildMessageGraph(messages), [messages]);
  const chatId = messages[0]?.chatId ?? null;
  const [branchSelection, setBranchSelection] = useState<{ chatId: string | null; selectedChildByParent: Record<string, string> }>({ chatId: null, selectedChildByParent: {} });
  const [pendingBranch, setPendingBranch] = useState<PendingBranch | null>(null);
  const selectedChildByParent = branchSelection.chatId === chatId ? branchSelection.selectedChildByParent : EMPTY_BRANCH_SELECTION;
  const lineage = useMemo(
    () => resolveLineage(graph, selectedChildByParent, pendingBranch, chatId),
    [chatId, graph, pendingBranch, selectedChildByParent],
  );
  const resolvedPendingChildId = newPendingChildId(graph, pendingBranch, chatId);

  useEffect(() => {
    if (!chatId || !pendingBranch || !resolvedPendingChildId) return;
    setBranchSelection((current) => ({
      chatId,
      selectedChildByParent: {
        ...(current.chatId === chatId ? current.selectedChildByParent : {}),
        [pendingBranch.parentKey]: resolvedPendingChildId,
      },
    }));
    setPendingBranch(null);
  }, [chatId, pendingBranch, resolvedPendingChildId]);

  useEffect(() => {
    if (runningRunId) bottom.current?.scrollIntoView({ block: "end" });
  }, [messages, run?.tokens.length, runningRunId]);

  const runSummary = useMemo(() => {
    if (!run) return null;
    if (!lineage.some(({ message }) => message.runId === run.id || message.id === run.messageId)) return null;
    return {
      tokens: run.metrics?.generatedTokens ?? run.tokens.length,
      tps: run.metrics?.timing?.decodeTokensPerSecond,
      perplexity: run.metrics?.responsePerplexity,
    };
  }, [lineage, run]);
  const displayedRunIsActive = Boolean(run && runningRunId === run.id);

  const armPendingBranch = (parentKey: string, fallbackChildId?: string): void => {
    if (!chatId) return;
    setPendingBranch({
      chatId,
      parentKey,
      knownChildIds: (graph.childrenByParent.get(parentKey) ?? []).map((message) => message.id),
      fallbackChildId,
    });
  };

  const selectBranch = (parentKey: string, childId: string): void => {
    if (!chatId) return;
    setPendingBranch(null);
    setBranchSelection((current) => ({
      chatId,
      selectedChildByParent: {
        ...(current.chatId === chatId ? current.selectedChildByParent : {}),
        [parentKey]: childId,
      },
    }));
  };

  return (
    <div className="chat-view">
      {nerdMode && (
        <div className="nerd-toolbar">
          <div className="nerd-toolbar-copy"><Bug aria-hidden="true" size={14} /><strong>Nerd Mode</strong><span>Token boundaries are visible. Color is reinforced by index and segment markers.</span></div>
          <label className="nerd-toolbar-metric">
            <span>Color by</span>
            <Select controlSize="sm" onChange={(event) => onNerdMetricChange(event.target.value as NerdMetric)} value={nerdMetric}>
              <option value="rawProbability">Raw probability</option>
              <option value="samplingProbability">Sampler probability</option>
              <option value="surprise">Surprise</option>
              <option value="latency">Decode latency</option>
              <option value="reasoning">Reasoning segment</option>
            </Select>
          </label>
        </div>
      )}
      <div className="messages-scroll">
        {messagesLoading ? <div className="messages-loading"><LoaderCircle className="spin" size={18} /> Loading conversation…</div> : messages.length === 0 ? <EmptyChat booting={booting} connected={connected} model={model} onNavigate={onNavigate} /> : (
          <div className="messages-list">
            {systemPrompt.trim() && (
              <details className="card system-prompt-card">
                <summary className="system-prompt-summary"><ScrollText aria-hidden="true" size={14} />System prompt<small>applies to every response in this chat</small></summary>
                <p className="system-prompt-text">{systemPrompt}</p>
              </details>
            )}
            {lineage.map(({ message, parentKey, siblings }, index) => (
              <MessageRow
                branchControl={siblings.length > 1 ? {
                  accessibleLabel: parentKey === ROOT_BRANCH_KEY ? "Choose starting conversation branch" : "Choose conversation branch",
                  onSelect: (id) => selectBranch(parentKey, id),
                  options: siblings.map((sibling, siblingIndex) => ({ id: sibling.id, label: `Branch ${String(siblingIndex + 1)} of ${String(siblings.length)}` })),
                } : undefined}
                key={message.id}
                message={message}
                nerdMetric={nerdMetric}
                nerdMode={nerdMode}
                onBranch={(value) => {
                  const branchParentKey = message.role === "user" ? parentKey : message.id;
                  const fallbackChildId = message.role === "user" ? message.id : lineage[index + 1]?.message.id;
                  armPendingBranch(branchParentKey, fallbackChildId);
                  onBranch(value, message.role === "user" ? message.parentMessageId ?? null : message.id);
                }}
                onInspectRun={onInspectRun}
                onOpenInspector={onOpenInspector}
                onRetry={() => {
                  armPendingBranch(parentKey, message.id);
                  onRetry(message);
                }}
                onSelectToken={onSelectToken}
                run={run}
                selectedToken={selectedToken}
              />
            ))}
            {runSummary && (
              <button className="run-summary-strip" onClick={onOpenInspector} title="Open the live run inspector for timing, probability, hardware, and token telemetry" type="button">
                <span className="run-summary-label"><Eye aria-hidden="true" size={14} /> Inspect run</span>
                <span className="run-summary-stat">{String(runSummary.tokens)} tokens</span>
                <span className="run-summary-stat">{runSummary.tps === undefined ? "TPS —" : `${formatNumber(runSummary.tps, 2)} tok/s`}</span>
                <span className="run-summary-stat">{runSummary.perplexity === undefined ? "PPL —" : `PPL ${formatNumber(runSummary.perplexity, 3)}`}</span>
                <span className={`stream-indicator ${run?.status ?? ""} ${displayedRunIsActive && streamConnected ? "connected" : ""}`}>{displayedRunIsActive ? `${run?.status ?? "running"} · ${streamConnected ? "live" : "reconnecting"}` : run?.status}</span>
              </button>
            )}
            <div ref={bottom} />
          </div>
        )}
      </div>
    </div>
  );
}
