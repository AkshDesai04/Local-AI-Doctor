import { Download, Maximize2, RefreshCw, Share2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { AttentionContextToken, ModelSummary, RunDetails, TokenAlternative, TokenEvent, TokenInfluence } from "../api/types";
import { ApiError, api } from "../api/client";
import { capabilityOf } from "../domain/capabilities";
import {
  formatInfluenceWeight,
  liveInfluence,
  prepareWeb,
  sourceKindLabel,
  sourcePosition,
  sourceText,
  type InfluenceScale,
  type WebModel,
} from "../domain/influence";
import { displayTokenText, downloadBlob, formatDuration, formatNumber, formatPercent } from "../utils/format";
import { AttentionAttributionView } from "./AttentionAttribution";
import { InfluenceWeb, type InfluenceWebVariant } from "./InfluenceWeb";
import { Badge, Button, Callout, Drawer, IconButton, SegmentedControl, Select, Slider, Switch, Tabs } from "./ui";

type MethodKey = "live" | "attention" | "gradient_x_input";
type FetchState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; data: TokenInfluence }
  | { status: "error"; message: string; hint?: string };

const SOURCE_LIMIT = 128;
const COMPACT_LABELS = 12;

const methodLabels: Record<MethodKey, string> = {
  live: "Live attention",
  attention: "Attention",
  gradient_x_input: "Gradient × input",
};

const caveats: Record<MethodKey, string> = {
  live: "Post-softmax self-attention captured while generating, averaged over the captured layers and heads. It shows where this prediction step allocated attention. It is allocation, not causation, and does not prove grounding or hallucination.",
  attention: "Post-softmax self-attention at the position that predicts this token, re-computed from the persisted prefix and averaged over heads (and over layers for the mean). The row is the same for every candidate token at this step. It is allocation, not causation, and does not prove grounding or hallucination.",
  gradient_x_input: "Gradient × input is |Σ gradient · activation| at the residual stream entering the first decoder layer: a local, first-order sensitivity of log p(token), or of logit(token) − logit(alternative). Large values mean small changes there would move this score most. It is not causal attribution and does not prove grounding or hallucination.",
};

interface InfluenceSectionProps {
  run: RunDetails;
  token: TokenEvent;
  /** Prompt catalogue plus earlier generated tokens, for the heatmap view. */
  contextTokens?: AttentionContextToken[];
  /** The selected model; its capability gates computing only when it produced this run. */
  model: ModelSummary | null;
  onSelectToken: (index: number) => void;
}

function unavailableReason(run: RunDetails, token: TokenEvent, model: ModelSummary | null): string | null {
  if (run.status !== "complete" && run.status !== "cancelled") return "Available once the run completes or is cancelled.";
  if (run.branchableThroughTokenIndex === undefined || token.index > run.branchableThroughTokenIndex) {
    return "This token's prefix was not fully persisted, so it cannot be re-run.";
  }
  if (model && model.id === run.modelId) {
    const capability = capabilityOf(model, "token_influence");
    if (capability.state !== "full" && capability.state !== "partial") return capability.reason ?? "Token influence is unavailable for this model.";
  }
  return null;
}

function withLayer(influence: TokenInfluence, layer: "mean" | number): TokenInfluence {
  const entry = layer === "mean" ? undefined : influence.layers?.find((item) => item.layer === layer);
  return entry ? { ...influence, sources: entry.sources, retainedWeight: entry.retainedWeight, omittedWeight: entry.omittedWeight } : influence;
}

/** Serialize the web with its colour tokens resolved, so the file renders anywhere. */
function exportSvg(svg: SVGSVGElement, name: string): void {
  const namespace = "http://www.w3.org/2000/svg";
  const styles = getComputedStyle(document.documentElement);
  const resolve = (value: string): string => value.replace(/var\((--[\w-]+)\)/g, (_match, property: string) => styles.getPropertyValue(property).trim() || "currentColor");
  const clone = svg.cloneNode(true) as SVGSVGElement;
  for (const element of [clone, ...Array.from(clone.querySelectorAll("*"))]) {
    for (const attribute of Array.from(element.attributes)) {
      if (attribute.value.includes("var(")) element.setAttribute(attribute.name, resolve(attribute.value));
    }
  }
  const [, , width = "720", height = "720"] = (clone.getAttribute("viewBox") ?? "").split(" ");
  const background = document.createElementNS(namespace, "rect");
  background.setAttribute("width", width);
  background.setAttribute("height", height);
  background.setAttribute("fill", resolve("var(--bg)"));
  clone.insertBefore(background, clone.firstChild);
  clone.setAttribute("xmlns", namespace);
  downloadBlob(name, new XMLSerializer().serializeToString(clone), "image/svg+xml");
}

function InfluenceTable({ web, scale, open, onSelectToken }: { web: WebModel; scale: InfluenceScale; open: boolean; onSelectToken: (index: number) => void }): React.ReactNode {
  return (
    <details className="influence-table" open={open}>
      <summary>Table view · {web.ranked.length} sources</summary>
      <div className="data-table" role="table" aria-label="Influence sources by weight">
        <div className="data-table-head influence-grid" role="row">
          <span role="columnheader">Rank</span><span role="columnheader">Token</span><span role="columnheader">Position</span><span role="columnheader">Kind</span><span role="columnheader">{scale === "relative" ? "Of max" : "Weight"}</span>
        </div>
        {web.ranked.map((source) => {
          const cells = (
            <>
              <span role="cell">#{source.rank}</span>
              <code role="cell" title={source.piece ?? undefined}>{sourceText(source)}</code>
              <span role="cell">{sourcePosition(source)}</span>
              <span role="cell">{sourceKindLabel(source)}</span>
              <span role="cell">{formatInfluenceWeight(source.display)}</span>
            </>
          );
          return source.sourceKind === "generated" && source.generatedTokenIndex !== null
            ? <button className="data-table-row influence-grid selectable" key={source.contextIndex} onClick={() => onSelectToken(source.generatedTokenIndex ?? 0)} role="row" type="button">{cells}</button>
            : <div className="data-table-row influence-grid" key={source.contextIndex} role="row">{cells}</div>;
        })}
      </div>
    </details>
  );
}

/**
 * Which earlier tokens a generated token depended on, by a stated method: the web,
 * its controls, an accessible table, exports, and the method's caveat. Lives in the
 * Nerd Mode token inspector, so token text is never shown outside Nerd Mode.
 */
export function InfluenceSection({ run, token, contextTokens, model, onSelectToken }: InfluenceSectionProps): React.ReactNode {
  const live = token.attentionAttribution;
  const [view, setView] = useState<"web" | "heatmap">("web");
  const [method, setMethod] = useState<MethodKey>(live ? "live" : "attention");
  const [layer, setLayer] = useState<"mean" | number>("mean");
  const [alternative, setAlternative] = useState<number | null>(null);
  const [topN, setTopN] = useState(24);
  const [hideSpecial, setHideSpecial] = useState(false);
  const [scale, setScale] = useState<InfluenceScale>("share");
  const [armed, setArmed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<FetchState>({ status: "idle" });
  const cache = useRef(new Map<string, TokenInfluence>());
  const serial = useRef(0);
  const compactSvg = useRef<SVGSVGElement>(null);
  const expandedSvg = useRef<SVGSVGElement>(null);

  useEffect(() => {
    setAlternative(null);
  }, [run.id, token.index]);
  useEffect(() => {
    if (method === "live" && !live) setMethod("attention");
  }, [live, method]);

  const unavailable = unavailableReason(run, token, model);
  const computed = method !== "live";
  const runId = run.id;
  const tokenIndex = token.index;

  useEffect(() => {
    if (method === "live" || !armed || unavailable) {
      setState({ status: "idle" });
      return;
    }
    const key = `${runId}:${String(tokenIndex)}:${method}:${method === "gradient_x_input" ? String(alternative) : "all"}`;
    const hit = cache.current.get(key);
    if (hit) {
      setState({ status: "ready", data: hit });
      return;
    }
    serial.current += 1;
    const current = serial.current;
    setState({ status: "loading" });
    const request = method === "attention"
      ? api.tokenInfluence(runId, tokenIndex, { method, layers: "all", sourceLimit: SOURCE_LIMIT })
      : api.tokenInfluence(runId, tokenIndex, { method, alternativeTokenId: alternative, sourceLimit: SOURCE_LIMIT });
    request
      .then((data) => {
        cache.current.set(key, data);
        if (current === serial.current) setState({ status: "ready", data });
      })
      .catch((error: unknown) => {
        if (current !== serial.current) return;
        setState({
          status: "error",
          message: error instanceof Error ? error.message : "The influence analysis failed.",
          hint: error instanceof ApiError ? error.hint : undefined,
        });
      });
  }, [alternative, armed, method, retry, runId, tokenIndex, unavailable]);

  const promptTokens = run.metrics?.promptTokens ?? null;
  const base = useMemo(() => {
    if (method === "live") return live ? liveInfluence(live, token, runId, promptTokens) : null;
    return state.status === "ready" ? state.data : null;
  }, [live, method, promptTokens, runId, state, token]);
  const capturedLayers = useMemo(() => base?.layers?.map((item) => item.layer) ?? [], [base]);
  const effectiveLayer = layer !== "mean" && capturedLayers.includes(layer) ? layer : "mean";
  const influence = useMemo(() => (base ? withLayer(base, method === "attention" ? effectiveLayer : "mean") : null), [base, effectiveLayer, method]);
  const web = useMemo(() => (influence ? prepareWeb(influence, { topN, hideSpecial, scale }) : null), [influence, topN, hideSpecial, scale]);
  const specialFlags = influence?.sources.some((source) => source.isSpecial !== null) ?? false;

  const alternatives = useMemo(() => {
    const seen = new Set<number>([token.tokenId]);
    const options: Array<TokenAlternative & { distribution: string }> = [];
    for (const [distribution, items] of [["raw", token.rawAlternatives], ["sampler", token.samplingAlternatives]] as const) {
      for (const item of items ?? []) {
        if (seen.has(item.tokenId)) continue;
        seen.add(item.tokenId);
        options.push({ ...item, distribution });
      }
    }
    return options;
  }, [token]);

  const methodOptions = (live ? (["live", "attention", "gradient_x_input"] as const) : (["attention", "gradient_x_input"] as const)).map((value) => ({
    value,
    label: methodLabels[value],
    title: value === "live" ? "The attention captured while this token was generated (Full instrumentation)." : value === "attention" ? "Re-run the prefix and read attention per layer." : "Re-run the prefix and take gradient × input at the first decoder layer.",
  }));
  const targetLabel = displayTokenText(token.displayText || token.piece) || "∅";
  const fileStem = `influence-${run.id}-token-${String(token.index)}-${influence?.method ?? method}`;

  const controls = (variant: InfluenceWebVariant): React.ReactNode => (
    <div className={`influence-controls ${variant}`}>
      <SegmentedControl label="Influence method" onChange={(value) => setMethod(value)} options={methodOptions} size="sm" value={method} />
      {method === "gradient_x_input" ? (
        <Select aria-label="Gradient target" controlSize="sm" onChange={(event) => setAlternative(event.target.value === "" ? null : Number(event.target.value))} title="Explain log p(chosen token), or the logit difference to an alternative." value={alternative === null ? "" : String(alternative)}>
          <option value="">Chosen token “{targetLabel}”</option>
          {alternatives.map((item) => <option key={item.tokenId} value={item.tokenId}>vs “{displayTokenText(item.piece) || "∅"}” · {item.distribution} #{item.rank} · {formatPercent(item.probability, 1)}</option>)}
        </Select>
      ) : (
        <Select aria-label="Attention layer" controlSize="sm" disabled={method === "live" || capturedLayers.length === 0} onChange={(event) => setLayer(event.target.value === "mean" ? "mean" : Number(event.target.value))} title={method === "live" ? "Live capture stores only the mean over layers." : "The mean over all captured layers, or one layer."} value={String(effectiveLayer)}>
          <option value="mean">Mean of all layers</option>
          {capturedLayers.map((item) => <option key={item} value={item}>Layer {item}</option>)}
        </Select>
      )}
      <Slider format={(value) => String(value)} label="Top sources" max={64} min={8} onValueChange={setTopN} title="How many of the heaviest sources the web draws; the rest is summarised as other." value={topN} />
      <Switch checked={hideSpecial && specialFlags} disabled={!specialFlags} label="Hide special/template tokens" onChange={(event) => setHideSpecial(event.target.checked)} title={specialFlags ? "Remove tokenizer special tokens and rescale the rest." : "This source did not record special-token flags."} />
      <SegmentedControl label="Weight scale" onChange={setScale} options={[{ value: "share", label: "Share of total", title: "Raw weights; all sources together sum to 1." }, { value: "relative", label: "Relative to max", title: "Each weight divided by the largest shown weight." }]} size="sm" value={scale} />
    </div>
  );

  const summary = influence && web && (
    <div className="influence-summary" role="status">
      <span><strong>{web.shown.length}</strong> of {web.ranked.length} sources shown</span>
      <span>other {web.otherCount} · <strong>{formatInfluenceWeight(web.otherShare)}</strong></span>
      {influence.objective && influence.objectiveValue !== null && (
        <span title={influence.objective === "log_probability" ? "The recomputed log probability of this token." : "The recomputed logit difference to the alternative."}>
          {influence.objective === "log_probability" ? "log p" : "Δ logit"} {formatNumber(influence.objectiveValue, 3)}
        </span>
      )}
      {influence.method !== "live_attention" && <Badge tone={influence.cached ? "neutral" : "info"}>{influence.cached ? "cached" : formatDuration(influence.durationMs)}</Badge>}
      {hideSpecial && specialFlags && <span className="influence-note">Special tokens hidden; shares rescaled to the remaining {formatInfluenceWeight(1 - web.hiddenSpecialWeight)} of the mass.</span>}
    </div>
  );

  const body = (variant: InfluenceWebVariant): React.ReactNode => {
    if (computed && unavailable) return <Callout tone="warning">{unavailable}</Callout>;
    if (computed && !armed) {
      return (
        <div className="influence-compute">
          <p>Re-runs this response&apos;s prefix on its model (loading it if needed) and caches the result.</p>
          <Button icon={<Share2 size={14} />} onClick={() => setArmed(true)} size="sm" variant="primary">Compute {methodLabels[method].toLowerCase()}</Button>
        </div>
      );
    }
    if (state.status === "loading" && computed) return <div className="influence-loading" role="status"><RefreshCw aria-hidden="true" className="spin" size={14} /> Re-running the prefix…</div>;
    if (state.status === "error" && computed) {
      return (
        <Callout tone="danger" title={state.message}>
          {state.hint && <p>{state.hint}</p>}
          <Button icon={<RefreshCw size={13} />} onClick={() => setRetry((value) => value + 1)} size="sm">Retry</Button>
        </Callout>
      );
    }
    if (!influence || !web) return <Callout>Generate with Full or Expert instrumentation to capture live attention, or choose a computed method.</Callout>;
    return (
      <>
        <InfluenceWeb
          labelLimit={variant === "expanded" ? 64 : COMPACT_LABELS}
          onSelectGeneratedToken={onSelectToken}
          scale={scale}
          svgRef={variant === "expanded" ? expandedSvg : compactSvg}
          target={{ label: targetLabel, index: token.index }}
          title={`Influence web for token #${String(token.index)}: ${web.shown.length} sources by ${methodLabels[method].toLowerCase()}`}
          variant={variant}
          web={web}
        />
        {variant === "compact" && web.shown.length > COMPACT_LABELS && <p className="influence-hint">Weights are written on the {COMPACT_LABELS} strongest edges; expand to label all {web.shown.length}.</p>}
      </>
    );
  };

  const exports = influence && (
    <div className="influence-exports">
      <Button icon={<Download size={13} />} onClick={() => { const svg = expanded ? expandedSvg.current : compactSvg.current; if (svg) exportSvg(svg, `${fileStem}.svg`); }} size="sm" title="Download the web as an SVG image">SVG</Button>
      <Button icon={<Download size={13} />} onClick={() => downloadBlob(`${fileStem}.json`, JSON.stringify({ view: { layer: effectiveLayer, topN, hideSpecial, scale }, influence }, null, 2), "application/json")} size="sm" title="Download the analysis as JSON">JSON</Button>
    </div>
  );

  const caveat = <Callout aria-label="Influence caveat" className="influence-caveat">{caveats[method]}</Callout>;

  return (
    <section aria-label={`Influence for token ${String(token.index)}`} className="influence-section">
      <div className="influence-header">
        <Tabs idPrefix="influence" items={[{ id: "web", label: "Influence web" }, { id: "heatmap", label: "Heatmap", disabled: !live, disabledReason: "The heatmap shows live attention, which this token did not capture." }]} label="Influence views" onChange={setView} value={view} />
        <IconButton icon={<Maximize2 size={14} />} label="Expand influence web" onClick={() => { setView("web"); setExpanded(true); }} size="sm" />
      </div>
      <div aria-labelledby={`influence-tab-${view}`} className="influence-panel" id="influence-panel" role="tabpanel">
        {view === "heatmap" ? (
          <AttentionAttributionView attribution={live} contextTokens={contextTokens} onSelectGeneratedToken={onSelectToken} targetTokenIndex={token.index} />
        ) : (
          <>
            {controls("compact")}
            {summary}
            {!expanded && body("compact")}
            {!expanded && web && <InfluenceTable onSelectToken={onSelectToken} open={false} scale={scale} web={web} />}
            {exports}
            {caveat}
          </>
        )}
      </div>
      {expanded && createPortal(<Drawer
        className="influence-dialog"
        closeLabel="Close influence web"
        description={`${methodLabels[method]} · token #${String(token.index)} “${targetLabel}”`}
        label="Expanded influence web"
        onClose={() => setExpanded(false)}
        open={expanded}
        title="Influence web"
      >
        <div className="influence-expanded">
          {controls("expanded")}
          <div className="influence-expanded-grid">
            <div className="influence-expanded-graph">{body("expanded")}</div>
            <aside className="influence-expanded-side">
              {summary}
              {web && <InfluenceTable onSelectToken={onSelectToken} open scale={scale} web={web} />}
              {exports}
              {caveat}
            </aside>
          </div>
        </div>
      </Drawer>, document.body)}
    </section>
  );
}
