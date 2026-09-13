import { Eye, Info } from "lucide-react";
import type { AttentionAttribution, AttentionContextToken, AttentionSourceToken } from "../api/types";
import { displayTokenText, formatNumber, formatPercent, tokenTextHint } from "../utils/format";

interface AttentionAttributionViewProps {
  attribution: AttentionAttribution | undefined;
  contextTokens?: AttentionContextToken[];
  targetTokenIndex: number;
  onSelectGeneratedToken: (index: number) => void;
}

interface SourceRange {
  kind: AttentionSourceToken["sourceKind"];
  label: string;
  detail: string;
  start: number;
  end: number;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function sourceTitle(contextToken: AttentionContextToken, source: AttentionSourceToken | undefined, rank: number | undefined): string {
  return [
    tokenTextHint(contextToken.piece, contextToken.displayText),
    source
      ? `Mean attention weight: ${formatNumber(source.weight, 8)} (${formatPercent(source.weight, 5)} of the normalized attention row)`
      : "Below the retained top-source threshold; its exact individual weight was not transmitted.",
    source && rank !== undefined ? `Importance rank among retained sources: #${String(rank)}` : "",
    `Model context position: ${String(contextToken.contextIndex)} · token ID ${String(contextToken.tokenId)}`,
    contextToken.sourceKind === "generated"
      ? `Earlier generated token #${String(contextToken.generatedTokenIndex ?? "unknown")}`
      : "Rendered prompt/history/template token",
    "Attention weight is not a causal contribution score.",
  ].filter(Boolean).join("\n");
}

function AttentionSourceRange({ range, sources, contextTokens, peakWeight, ranks, onSelectGeneratedToken }: {
  range: SourceRange;
  sources: AttentionSourceToken[];
  contextTokens?: AttentionContextToken[];
  peakWeight: number;
  ranks: Map<number, number>;
  onSelectGeneratedToken: (index: number) => void;
}): React.ReactNode {
  const inRange = sources.filter((source) => source.sourceKind === range.kind && source.contextIndex >= range.start && source.contextIndex < range.end);
  const weightedByPosition = new Map(inRange.map((source) => [source.contextIndex, source]));
  const catalog = contextTokens?.filter((token) => token.sourceKind === range.kind && token.contextIndex >= range.start && token.contextIndex < range.end);
  const items: React.ReactNode[] = [];
  const appendToken = (contextToken: AttentionContextToken): void => {
    const source = weightedByPosition.get(contextToken.contextIndex);
    const strength = source === undefined || peakWeight <= 0 ? 0 : Math.min(1, Math.max(0, source.weight / peakWeight));
    const tokenLabel = displayTokenText(contextToken.displayText || contextToken.piece) || "∅";
    const shared = {
      "aria-label": source ? `${tokenLabel}, mean attention ${formatPercent(source.weight, 5)}` : `${tokenLabel}, lower-weight source not individually retained`,
      className: `attention-source-token ${source ? "weighted" : "unretained"} ${strength === 1 ? "strongest" : ""}`,
      style: {
        "--attention-fill": `${String(13 + strength * 72)}%`,
        "--attention-border": `${String(18 + strength * 72)}%`,
        "--attention-text": `${String(52 + strength * 48)}%`,
      } as React.CSSProperties,
      title: sourceTitle(contextToken, source, ranks.get(contextToken.contextIndex)),
    };
    items.push(contextToken.sourceKind === "generated" && contextToken.generatedTokenIndex !== undefined
      ? <button {...shared} key={`source-${String(contextToken.contextIndex)}`} onClick={() => onSelectGeneratedToken(contextToken.generatedTokenIndex ?? 0)} type="button"><span>{tokenLabel}</span><small>{String(contextToken.contextIndex)}</small></button>
      : <span {...shared} key={`source-${String(contextToken.contextIndex)}`} role="listitem"><span>{tokenLabel}</span><small>{String(contextToken.contextIndex)}</small></span>);
  };

  if (catalog?.length) {
    catalog.forEach(appendToken);
  } else {
    let cursor = range.start;
    for (const source of inRange) {
      if (source.contextIndex > cursor) {
        const count = source.contextIndex - cursor;
        items.push(<span aria-label={`${String(count)} lower-weight context positions omitted`} className="attention-source-gap" key={`gap-${range.kind}-${String(cursor)}`}>… {String(count)}</span>);
      }
      appendToken(source);
      cursor = source.contextIndex + 1;
    }
    if (cursor < range.end) {
      const count = range.end - cursor;
      items.push(<span aria-label={`${String(count)} lower-weight context positions omitted`} className="attention-source-gap" key={`gap-${range.kind}-${String(cursor)}`}>… {String(count)}</span>);
    }
  }

  return (
    <div className="attention-source-group">
      <div className="attention-source-heading"><strong>{range.label}</strong><span>{range.detail}</span></div>
      <div aria-label={`${range.label} attention sources`} className="attention-source-stream" role="list">{items}</div>
    </div>
  );
}

export function AttentionAttributionView({ attribution, contextTokens, targetTokenIndex, onSelectGeneratedToken }: AttentionAttributionViewProps): React.ReactNode {
  if (!attribution) {
    return (
      <section className="attention-attribution empty" aria-label="Context attention">
        <div className="attention-attribution-title"><Eye size={15} /><div><strong>Context attention</strong><span>Not captured for this token</span></div></div>
        <p>Generate with Full or Expert instrumentation to capture an attention map.</p>
      </section>
    );
  }

  const sources = [...attribution.sourceTokens].sort((left, right) => left.contextIndex - right.contextIndex);
  const ranked = [...sources].sort((left, right) => right.weight - left.weight);
  const ranks = new Map(ranked.map((source, index) => [source.contextIndex, index + 1]));
  const peakWeight = ranked[0]?.weight ?? 0;
  const generatedBoundaryCandidates = [...(contextTokens ?? []), ...sources]
    .filter((source) => source.sourceKind === "generated" && source.generatedTokenIndex !== undefined)
    .map((source) => source.contextIndex - (source.generatedTokenIndex ?? 0));
  const promptCatalogEnd = contextTokens
    ?.filter((source) => source.sourceKind === "prompt")
    .reduce((maximum, source) => Math.max(maximum, source.contextIndex + 1), 0);
  const inferredPromptLength = generatedBoundaryCandidates[0] ?? promptCatalogEnd
    ?? Math.max(0, attribution.totalSourceCount - targetTokenIndex);
  const promptLength = Math.min(attribution.totalSourceCount, Math.max(0, inferredPromptLength));
  const ranges = ([
    {
      kind: "prompt",
      label: "Rendered prompt and conversation history",
      detail: "Includes chat-template and special tokens",
      start: 0,
      end: promptLength,
    },
    {
      kind: "generated",
      label: "Earlier generated response",
      detail: "Click a retained token to inspect its own telemetry",
      start: promptLength,
      end: attribution.totalSourceCount,
    },
  ] satisfies SourceRange[]).filter((range) => range.end > range.start);

  return (
    <section className="attention-attribution" aria-label={`Context attention for token ${String(targetTokenIndex)}`}>
      <div className="attention-attribution-title">
        <Eye size={15} />
        <div><strong>Context attention</strong><span>for generated token #{String(targetTokenIndex)}</span></div>
      </div>
      <div className="attention-coverage">
        <span><strong>{formatNumber(attribution.retainedSourceCount, 0)}</strong> of {formatNumber(attribution.totalSourceCount, 0)} positions retained</span>
        <span><strong>{formatPercent(attribution.retainedWeight, 2)}</strong> attention mass shown</span>
        <span><strong>{formatNumber(attribution.capturedLayers.length, 0)}</strong> layers × <strong>{formatNumber(attribution.capturedHeads, 0)}</strong> heads</span>
      </div>
      <div className="attention-legend"><span>Lower</span><i /><span>Higher mean attention</span></div>
      <div className="attention-context-map">
        {ranges.map((range) => <AttentionSourceRange contextTokens={contextTokens} key={range.kind} onSelectGeneratedToken={onSelectGeneratedToken} peakWeight={peakWeight} range={range} ranks={ranks} sources={sources} />)}
      </div>
      <dl className="attention-method">
        <div><dt>Measurement</dt><dd>{humanize(attribution.method)}</dd></div>
        <div><dt>Aggregation</dt><dd>{humanize(attribution.aggregation)}</dd></div>
        <div><dt>Omitted mass</dt><dd>{formatPercent(attribution.omittedWeight, 3)}</dd></div>
      </dl>
      <div className="attention-caveat" role="note"><Info size={15} /><span>These are exact post-softmax self-attention weights averaged across the captured layers and heads. The row belongs to the prediction step, not to one vocabulary choice: the same prefix has the same map whichever candidate is sampled. The weights are <strong>not causal contribution scores</strong> and do not prove grounding or hallucination; residual paths, value vectors, MLPs, and later layers also shape the selected token.</span></div>
    </section>
  );
}
