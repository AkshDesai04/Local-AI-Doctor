import {
  Activity,
  Binary,
  Box,
  Braces,
  Clock3,
  Cpu,
  FileJson,
  Fingerprint,
  Gauge,
  GitBranch,
  Info,
  Layers3,
  ListOrdered,
  Network,
  Sigma,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { AlternativeDistribution, AttentionContextToken, CapabilityKey, ConfigurationSnapshot, HealthStatus, ModelSummary, RunDetails, TokenAlternative, TokenEvent } from "../api/types";
import { fetchRunExport } from "../api/client";
import { capabilityOf, supportsGeneration } from "../domain/capabilities";
import { displayTokenText, downloadBlob, formatBytes, formatDuration, formatNumber, formatPercent, shortFingerprint, tokenTextHint } from "../utils/format";
import { AttentionAttributionView } from "./AttentionAttribution";
import { type ChartMetric, TraceChart } from "./TraceChart";
import { Badge, type BadgeTone, Button, Callout, Card, EmptyState, IconButton, Select, Stat, Tabs } from "./ui";
import { VirtualTokenTable } from "./VirtualTokenTable";

export type InspectorTab = "overview" | "tokens" | "probability" | "timing" | "experts" | "context" | "embeddings" | "hardware" | "configuration" | "events";

interface TabDefinition {
  id: InspectorTab;
  label: string;
  icon: LucideIcon;
  capability?: CapabilityKey;
}

const tabs: TabDefinition[] = [
  { id: "overview", label: "Overview", icon: Gauge },
  { id: "tokens", label: "Tokens", icon: Binary, capability: "raw_logits" },
  { id: "probability", label: "Probability", icon: Sigma, capability: "raw_logits" },
  { id: "timing", label: "Timing", icon: Clock3, capability: "text_generation" },
  { id: "experts", label: "Experts", icon: Network, capability: "moe_routing" },
  { id: "context", label: "Context", icon: Layers3 },
  { id: "embeddings", label: "Embeddings", icon: Box, capability: "embeddings" },
  { id: "hardware", label: "Hardware", icon: Cpu },
  { id: "configuration", label: "Configuration", icon: Braces },
  { id: "events", label: "Raw events", icon: FileJson },
];

interface InspectorProps {
  open: boolean;
  model: ModelSummary | null;
  run: RunDetails | null;
  health: HealthStatus | null;
  configuration: ConfigurationSnapshot | null;
  activeTab: InspectorTab;
  selectedToken: number | null;
  nerdMode: boolean;
  branching: boolean;
  onBranchAlternative: (tokenIndex: number, distribution: AlternativeDistribution, alternative: TokenAlternative) => Promise<void>;
  onTabChange: (tab: InspectorTab) => void;
  onSelectToken: (index: number) => void;
  onClose: () => void;
}

const runTones: Record<RunDetails["status"], BadgeTone> = {
  queued: "neutral",
  loading: "accent",
  running: "accent",
  complete: "info",
  cancelled: "warning",
  failed: "danger",
};

function tabDisabled(tab: TabDefinition, model: ModelSummary | null): { disabled: boolean; reason?: string } {
  if (tab.id === "timing") return { disabled: !supportsGeneration(model), reason: "Timing traces need a text-generation model." };
  if (!tab.capability) return { disabled: false };
  const capability = capabilityOf(model, tab.capability);
  return { disabled: capability.state !== "full" && capability.state !== "partial", reason: capability.reason };
}

function StatGrid({ children }: { children: React.ReactNode }): React.ReactNode {
  return <div className="stat-grid">{children}</div>;
}

interface SelectedAlternative {
  distribution: AlternativeDistribution;
  alternative: TokenAlternative;
}

function Alternatives({ title, distribution, alternatives, currentTokenId, nerdMode, selected, onSelect }: {
  title: string;
  distribution: AlternativeDistribution;
  alternatives: TokenAlternative[] | undefined;
  currentTokenId: number;
  nerdMode: boolean;
  selected: SelectedAlternative | null;
  onSelect: (selection: SelectedAlternative) => void;
}): React.ReactNode {
  return (
    <section className="alternatives-section">
      <h4 title={distribution === "raw" ? "Probabilities directly from the model before sampling filters are applied." : "Probabilities after temperature, top-k, top-p, and other active sampling filters."}>{title}</h4>
      {!alternatives?.length ? <p className="muted">Not captured at this instrumentation level.</p> : (
        <div className="data-table alternatives-table">
          <div className="data-table-head alternatives-grid"><span title="Probability rank within this distribution">Rank</span><span title="Human-readable token text; hover a row for the exact tokenizer piece">Piece</span><span title="Probability assigned to this token">Probability</span><span title="Natural logarithm of the token probability">Log p</span><span title="Whether this token survived the active sampling filters">Filter</span></div>
          {alternatives.map((item) => {
            const isCurrent = item.tokenId === currentTokenId;
            const isSelected = selected?.distribution === distribution
              && selected.alternative.tokenId === item.tokenId
              && selected.alternative.rank === item.rank;
            const tokenLabel = displayTokenText(item.piece) || "∅";
            return (
            <button
              aria-label={`${isCurrent ? "Current" : "Select"} token ${tokenLabel} from the ${distribution} distribution`}
              aria-pressed={isSelected}
              className={`data-table-row alternatives-grid ${nerdMode && !isCurrent ? "selectable" : ""} ${isSelected ? "selected" : ""} ${isCurrent ? "current" : ""}`}
              disabled={!nerdMode || isCurrent}
              key={`${String(item.rank)}-${String(item.tokenId)}`}
              onClick={() => onSelect({ distribution, alternative: item })}
              title={`${tokenTextHint(item.piece)}\n${isCurrent ? "This is the token the model originally selected." : nerdMode ? "Select this token to prepare a branched response." : "Enable Nerd Mode to branch from an alternative token."}`}
              type="button"
            >
              <span>#{String(item.rank)}</span><code>{tokenLabel}</code><span>{formatPercent(item.probability, 3)}</span><span>{formatNumber(item.logProbability, 4)}</span><span>{item.survivedFiltering === undefined ? "—" : item.survivedFiltering ? "kept" : "removed"}</span>
            </button>
          );})}
        </div>
      )}
    </section>
  );
}

function TokenDetail({ token, contextTokens, nerdMode, branching, canBranch, branchUnavailableReason, onBranchAlternative, onSelectToken }: {
  token: TokenEvent | undefined;
  contextTokens?: AttentionContextToken[];
  nerdMode: boolean;
  branching: boolean;
  canBranch: boolean;
  branchUnavailableReason: string;
  onBranchAlternative: (tokenIndex: number, distribution: AlternativeDistribution, alternative: TokenAlternative) => Promise<void>;
  onSelectToken: (index: number) => void;
}): React.ReactNode {
  const [selectedAlternative, setSelectedAlternative] = useState<SelectedAlternative | null>(null);

  useEffect(() => {
    setSelectedAlternative(null);
  }, [nerdMode, token?.index]);

  if (!token) return <EmptyState description="Select a token in the response, graph, or table to lock its exact details here." title="No token selected" />;
  const branch = async (): Promise<void> => {
    if (!selectedAlternative || branching || !canBranch) return;
    await onBranchAlternative(token.index, selectedAlternative.distribution, selectedAlternative.alternative);
  };
  const selectedTokenLabel = displayTokenText(token.displayText || token.piece) || "∅";
  return (
    <div className="inspector-stack">
      <div className="token-detail-hero" title={tokenTextHint(token.piece, token.displayText)}>
        <Badge className="segment-badge" tone={token.reasoningSegment === "reasoning" ? "info" : "neutral"}>{token.reasoningSegment}</Badge>
        <code>{selectedTokenLabel}</code>
        <span>token #{String(token.index)}</span>
      </div>
      <AttentionAttributionView attribution={token.attentionAttribution} contextTokens={contextTokens} onSelectGeneratedToken={onSelectToken} targetTokenIndex={token.index} />
      <dl className="kv-grid">
        <div><dt>Token ID</dt><dd className="mono">{String(token.tokenId)}</dd></div>
        <div><dt>Bytes</dt><dd className="mono">{token.bytes ?? "Not captured"}</dd></div>
        <div><dt>Character span</dt><dd>{token.characterSpan ? `${String(token.characterSpan[0])}–${String(token.characterSpan[1])}` : "Not captured"}</dd></div>
        <div><dt>Decoded display</dt><dd className="mono">{displayTokenText(token.displayText) || "∅"}</dd></div>
        <div><dt>Raw logit</dt><dd>{formatNumber(token.rawLogit, 5)}</dd></div>
        <div><dt>Processed logit</dt><dd>{formatNumber(token.processedLogit, 5)}</dd></div>
        <div><dt>Raw probability</dt><dd>{formatPercent(token.rawProbability, 5)}</dd></div>
        <div><dt>Sampler probability</dt><dd>{formatPercent(token.samplingProbability, 5)}</dd></div>
        <div><dt>Raw log p</dt><dd>{formatNumber(token.rawLogProbability, 5)}</dd></div>
        <div><dt>Sampler log p</dt><dd>{formatNumber(token.samplingLogProbability, 5)}</dd></div>
        <div><dt>Exact rank</dt><dd>{formatNumber(token.rawRank, 0)}</dd></div>
        <div><dt>Entropy</dt><dd>{formatNumber(token.entropy, 5)}</dd></div>
        <div><dt>Surprise</dt><dd>{formatNumber(token.surprise, 5)}</dd></div>
        <div><dt>Cumulative log p</dt><dd>{formatNumber(token.cumulativeLogProbability, 5)}</dd></div>
        <div><dt>Running perplexity</dt><dd>{formatNumber(token.runningPerplexity, 4)}</dd></div>
        <div><dt>Decode / sample</dt><dd>{formatDuration(token.timing?.decodeMs)} / {formatDuration(token.timing?.samplingMs)}</dd></div>
        <div><dt>Inter-token</dt><dd>{formatDuration(token.timing?.interTokenMs)}</dd></div>
        <div><dt>Rolling throughput</dt><dd>{formatNumber(token.timing?.rollingTps, 2)} tok/s</dd></div>
      </dl>
      <Alternatives alternatives={token.rawAlternatives} currentTokenId={token.tokenId} distribution="raw" nerdMode={nerdMode} onSelect={setSelectedAlternative} selected={selectedAlternative} title="Top alternatives · raw model distribution" />
      <Alternatives alternatives={token.samplingAlternatives} currentTokenId={token.tokenId} distribution="sampling" nerdMode={nerdMode} onSelect={setSelectedAlternative} selected={selectedAlternative} title="Top alternatives · sampler distribution" />
      {nerdMode && selectedAlternative && (
        <div className="branch-token-action">
          <span>Continue from token <code title={tokenTextHint(selectedAlternative.alternative.piece)}>{displayTokenText(selectedAlternative.alternative.piece) || "∅"}</code> in a new chat.</span>
          <Button block disabled={!canBranch} icon={<GitBranch size={14} />} loading={branching} onClick={() => void branch()} title={!canBranch ? branchUnavailableReason : "Create a new chat from this point and force the selected alternative before continuing generation."} variant="primary">{branching ? "Creating branch…" : canBranch ? "Branch out with selected token" : branchUnavailableReason}</Button>
        </div>
      )}
      <Callout>Alternatives are tokens with high probability under a distribution—not model thoughts or hidden reasoning.</Callout>
    </div>
  );
}

function OverviewPanel({ run, model }: { run: RunDetails | null; model: ModelSummary | null }): React.ReactNode {
  if (!run) return <EmptyState description="Run a prompt or open response details to inspect telemetry. No placeholder measurements are shown." icon={Gauge} title="No run selected" />;
  const timing = run.metrics?.timing;
  return (
    <div className="inspector-stack">
      <div className="run-status-row"><Badge tone={runTones[run.status]}>{run.status}</Badge><span>{run.metrics?.finishReason ? `finish: ${run.metrics.finishReason}` : "No finish reason yet"}</span></div>
      {run.warnings?.map((warning, index) => <Callout key={index} tone="warning">{warning}</Callout>)}
      <StatGrid>
        <Stat caption="request → first server token" label="Server TTFT" value={formatDuration(timing?.serverTtftMs)} />
        <Stat caption="steady-state decode" label="Decode rate" value={timing?.decodeTokensPerSecond === undefined ? "—" : `${formatNumber(timing.decodeTokensPerSecond, 2)} tok/s`} />
        <Stat caption="conditional generated response" label="Perplexity" value={formatNumber(run.metrics?.responsePerplexity, 4)} />
        <Stat caption="prompt + completion" label="Total latency" value={formatDuration(timing?.totalMs)} />
        <Stat label="Prompt tokens" value={formatNumber(run.metrics?.promptTokens, 0)} />
        <Stat label="Generated tokens" value={formatNumber(run.metrics?.generatedTokens ?? run.tokens.length, 0)} />
      </StatGrid>
      <Card icon={Fingerprint} title="Reproducibility">
        {!run.reproducibility ? <p className="muted">The backend has not returned a reproducibility snapshot.</p> : (
          <dl className="kv-grid">
            <div><dt>Requested seed</dt><dd>{run.reproducibility.requestedSeed === null || run.reproducibility.requestedSeed === undefined || run.reproducibility.requestedSeed === "" ? "Generated" : String(run.reproducibility.requestedSeed)}</dd></div>
            <div><dt>Effective seed</dt><dd className="mono">{run.reproducibility.effectiveSeed}</dd></div>
            <div><dt>RNG</dt><dd>{run.reproducibility.rngAlgorithm ?? "Not reported"}</dd></div>
            <div><dt>Generator</dt><dd>{run.reproducibility.generatorDevice ?? "Not reported"}</dd></div>
            <div><dt>Model fingerprint</dt><dd className="mono">{shortFingerprint(run.reproducibility.modelFingerprint ?? model?.fingerprint)}</dd></div>
            <div><dt>Backend</dt><dd>{run.reproducibility.backend ?? "Not reported"}</dd></div>
            <div><dt>Device / dtype</dt><dd title={run.reproducibility.deviceReason}><span>{run.reproducibility.device ?? "—"} / {run.reproducibility.dtype ?? "—"}</span>{run.reproducibility.deviceReason && <small className="kv-detail">{run.reproducibility.deviceReason}</small>}</dd></div>
            <div><dt>Deterministic kernels</dt><dd>{run.reproducibility.deterministicKernels === undefined ? "Not reported" : run.reproducibility.deterministicKernels ? "Enabled" : "Disabled"}</dd></div>
          </dl>
        )}
        <p className="footnote">A seed cannot be recovered from output. Exact replay also depends on unchanged files, tokenizer, device, dtype, kernels, software, and batching.</p>
      </Card>
      {run.metrics?.telemetryOverhead && <Callout icon={Activity}>Instrumentation overhead: {run.metrics.telemetryOverhead}</Callout>}
    </div>
  );
}

const probabilityMetrics: ChartMetric[] = [
  { key: "raw-p", label: "Raw model probability", color: "var(--viz-1)", value: (token) => token.rawProbability },
  { key: "sample-p", label: "Sampler probability", color: "var(--viz-2)", value: (token) => token.samplingProbability },
];
const logProbabilityMetrics: ChartMetric[] = [
  { key: "raw-log-p", label: "Raw model log probability", color: "var(--viz-1)", value: (token) => token.rawLogProbability },
  { key: "sample-log-p", label: "Sampler log probability", color: "var(--viz-2)", value: (token) => token.samplingLogProbability },
];
const uncertaintyMetrics: ChartMetric[] = [
  { key: "entropy", label: "Entropy", color: "var(--viz-3)", value: (token) => token.entropy },
  { key: "surprise", label: "Surprise −ln(p)", color: "var(--viz-5)", value: (token) => token.surprise },
];
const rankMetrics: ChartMetric[] = [
  { key: "rank", label: "Exact full-vocabulary rank", color: "var(--viz-4)", value: (token) => token.rawRank },
];
const perplexityMetrics: ChartMetric[] = [
  { key: "perplexity", label: "Running response perplexity", color: "var(--viz-6)", value: (token) => token.runningPerplexity },
];
const timingMetrics: ChartMetric[] = [
  { key: "decode", label: "Decode forward", unit: "ms", color: "var(--viz-1)", value: (token) => token.timing?.decodeMs },
  { key: "sampling", label: "Sampling", unit: "ms", color: "var(--viz-2)", value: (token) => token.timing?.samplingMs },
];
const arrivalMetrics: ChartMetric[] = [
  { key: "server-arrival", label: "Server inter-token", unit: "ms", color: "var(--viz-3)", value: (token) => token.timing?.interTokenMs },
  { key: "client-arrival", label: "Client inter-arrival", unit: "ms", color: "var(--viz-5)", value: (token) => token.timing?.clientInterArrivalMs },
];
const cumulativeTimingMetrics: ChartMetric[] = [
  { key: "cumulative", label: "Cumulative time", unit: "ms", color: "var(--viz-6)", value: (token) => token.timing?.cumulativeMs },
];
const throughputMetrics: ChartMetric[] = [
  { key: "instant", label: "Instantaneous", unit: "tok/s", color: "var(--viz-1)", value: (token) => token.timing?.instantaneousTps },
  { key: "rolling", label: "Rolling", unit: "tok/s", color: "var(--viz-2)", value: (token) => token.timing?.rollingTps },
];

const expertSelectionMetrics: ChartMetric[] = [
  { key: "expert-id", label: "First selected expert ID", color: "var(--viz-3)", value: (token) => token.expertRoutes?.[0]?.selectedExpertIds[0] },
  { key: "expert-weight", label: "First expert weight", color: "var(--viz-1)", value: (token) => token.expertRoutes?.[0]?.gateWeights[0] },
];

function ExpertsPanel({ run, selectedToken, nerdMode, onSelectToken }: { run: RunDetails | null; selectedToken: number | null; nerdMode: boolean; onSelectToken: (index: number) => void }): React.ReactNode {
  const routed = run?.tokens.filter((token) => token.expertRoutes?.length) ?? [];
  if (!routed.length) return <EmptyState description="The selected adapter supports routing, but this run did not expose per-token routes at its instrumentation level." icon={Network} title="No expert route events" />;
  return (
    <div className="inspector-stack">
      <TraceChart metrics={expertSelectionMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Expert selection and weight" tokens={run?.tokens ?? []} />
      <Card flush icon={Network} title="Selected vs. executed experts">
        <div className="data-table expert-route-table">
          <div className="data-table-head expert-grid"><span>Token</span><span>Layer</span><span>Selected</span><span>Executed</span><span>Weights</span></div>
          {routed.flatMap((token) => token.expertRoutes?.map((route) => <button className="data-table-row expert-grid selectable" key={`${String(token.index)}-${String(route.layer)}`} onClick={() => onSelectToken(token.index)} type="button"><span>#{String(token.index)}</span><span>{String(route.layer)}</span><span>{route.selectedExpertIds.join(", ")}</span><span>{route.executedExpertIds.join(", ")}</span><span>{route.gateWeights.map((weight) => formatNumber(weight, 3)).join(", ")}</span></button>) ?? [])}
        </div>
      </Card>
      <Callout>Selected and actually executed experts are intentionally distinct. Dense gated MLPs are never presented as MoE routing.</Callout>
    </div>
  );
}

function ContextPanel({ run, model }: { run: RunDetails | null; model: ModelSummary | null }): React.ReactNode {
  const context = run?.metrics?.context;
  const limit = context?.effectiveLimit ?? model?.effectiveContextLimit ?? null;
  const parts = [
    { label: "Rendered prompt", value: context?.renderedPromptTokens ?? 0, className: "prompt" },
    { label: "Template + special", value: context?.templateTokens ?? 0, className: "template" },
    { label: "Multimodal positions", value: context?.multimodalPositions ?? 0, className: "media" },
    { label: "Generated", value: context?.generatedTokens ?? 0, className: "generated" },
    { label: "Reserved output", value: context?.reservedOutputTokens ?? 0, className: "reserved" },
  ];
  const used = parts.reduce((sum, part) => sum + part.value, 0);
  const remaining = context?.remainingTokens ?? (limit === null ? undefined : Math.max(0, limit - used));
  if (!run && !model) return <EmptyState description="Select a model to see its discovered context metadata." icon={Layers3} title="No context metadata" />;
  return (
    <div className="inspector-stack">
      <StatGrid>
        <Stat label="Effective limit" value={limit === null ? "Unknown" : formatNumber(limit, 0)} />
        <Stat label="Used + reserved" value={context ? formatNumber(used, 0) : "No run"} />
        <Stat label="Remaining" value={remaining === undefined ? "—" : formatNumber(remaining, 0)} />
      </StatGrid>
      <Card title="Composition">
        {context && limit !== null && <div className="context-bar" aria-label={`Context used ${formatPercent(used / limit)}`} role="img">{parts.map((part) => <i className={part.className} key={part.label} style={{ width: `${String((part.value / limit) * 100)}%` }} />)}</div>}
        <ul className="context-legend">{parts.map((part) => <li key={part.label}><i aria-hidden="true" className={part.className} /><span>{part.label}</span><strong>{context ? formatNumber(part.value, 0) : "—"}</strong></li>)}</ul>
      </Card>
      <Callout>Text tokens and expanded multimodal positions are kept separate. Remaining capacity includes the reserved output budget.</Callout>
      <Card flush icon={Layers3} title="Discovered limits">
        {!model?.contextLimits?.length ? <p className="muted card-inset">The adapter did not expose context-limit candidates.</p> : (
          <ul className="candidate-list">
            {model.contextLimits.map((candidate) => <li className={candidate.selected ? "selected" : ""} key={candidate.source}><span className="mono">{candidate.source}</span><strong>{candidate.tokens === null ? "Unknown" : formatNumber(candidate.tokens, 0)}</strong>{candidate.note && <small>{candidate.note}</small>}</li>)}
          </ul>
        )}
      </Card>
      <dl className="kv-grid"><div><dt>Truncation behavior</dt><dd>{context?.behavior?.replaceAll("_", " ") ?? "Not observed"}</dd></div><div><dt>Selection</dt><dd>{limit === null ? "Conflicting or absent metadata" : "Selected effective limit"}</dd></div></dl>
    </div>
  );
}

function HardwarePanel({ run, health }: { run: RunDetails | null; health: HealthStatus | null }): React.ReactNode {
  return (
    <div className="inspector-stack">
      <StatGrid>
        <Stat caption="selected by hardware discovery" label="Backend" value={run?.reproducibility?.backend ?? health?.selectedBackend ?? "Not reported"} />
        <Stat label="Device" value={run?.reproducibility?.device ?? "Not reported"} />
        <Stat label="Peak RAM" value={formatBytes(run?.metrics?.peakRamBytes)} />
        <Stat label="Peak VRAM" value={formatBytes(run?.metrics?.peakVramBytes)} />
        <Stat label="KV cache" value={formatBytes(run?.metrics?.kvCacheBytes)} />
        <Stat label="Attention" value={run?.reproducibility?.attentionImplementation ?? "Not reported"} />
      </StatGrid>
      <Callout>Memory peaks are exact allocator measurements where available. Utilization samples are not exact per-token measurements.</Callout>
      {!run && <EmptyState description="Hardware discovery is available above. Per-run memory and cache measurements appear after selecting a run." icon={Cpu} size="compact" title="No run hardware snapshot" />}
    </div>
  );
}

export function Inspector({ open, model, run, health, configuration, activeTab, selectedToken, nerdMode, branching, onBranchAlternative, onTabChange, onSelectToken, onClose }: InspectorProps): React.ReactNode {
  const selected = run?.tokens.find((token) => token.index === selectedToken);
  const [exporting, setExporting] = useState<"json" | "jsonl" | "csv" | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const activeDefinition = tabs.find((tab) => tab.id === activeTab) ?? tabs[0];
  const activeCapability = activeDefinition?.capability ? capabilityOf(model, activeDefinition.capability) : null;
  const activeCapabilityDisabled = activeDefinition ? tabDisabled(activeDefinition, model).disabled : false;
  // Never render (or fetch for) a tab the selected model cannot support, even for the
  // single render before the parent state falls back to the overview.
  const visibleTab: InspectorTab = activeCapabilityDisabled ? "overview" : activeTab;
  const tabItems = tabs.map((tab) => {
    const state = tabDisabled(tab, model);
    return { id: tab.id, label: tab.label, icon: tab.icon, disabled: state.disabled, disabledReason: state.reason ?? `${tab.label} is not available for this model.` };
  });

  useEffect(() => {
    if (activeCapabilityDisabled) onTabChange("overview");
  }, [activeCapabilityDisabled, onTabChange]);

  const exportRun = useCallback(async (format: "json" | "jsonl" | "csv"): Promise<void> => {
    if (!run) return;
    setExporting(format);
    setExportError(null);
    try {
      const blob = await fetchRunExport(run.id, format);
      const extension = format === "jsonl" ? "jsonl" : format;
      downloadBlob(`run-${run.id}.${extension}`, blob, blob.type || (format === "csv" ? "text/csv" : "application/json"));
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "The trace export failed.");
    } finally {
      setExporting(null);
    }
  }, [run]);

  const panel = useMemo((): React.ReactNode => {
    const chart = (title: string, metrics: ChartMetric[]): React.ReactNode => <TraceChart metrics={metrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title={title} tokens={run?.tokens ?? []} />;
    switch (visibleTab) {
      case "overview": return <OverviewPanel model={model} run={run} />;
      case "tokens": {
        const runComplete = run?.status === "complete";
        const tokenPersisted = selected !== undefined
          && run?.branchableThroughTokenIndex !== undefined
          && selected.index <= run.branchableThroughTokenIndex;
        const branchUnavailableReason = !runComplete
          ? "Available when run completes"
          : "Selected token was not retained in persisted telemetry";
        const promptContextTokens = run?.tokens.find((token) => token.index === 0)?.attentionAttribution?.contextTokens ?? [];
        const promptTokenCount = run?.metrics?.promptTokens ?? promptContextTokens.length;
        const generatedContextTokens: AttentionContextToken[] = (run?.tokens ?? [])
          .filter((token) => selected !== undefined && token.index < selected.index)
          .map((token) => ({
            contextIndex: promptTokenCount + token.index,
            tokenId: token.tokenId,
            piece: token.piece,
            displayText: token.displayText,
            sourceKind: "generated",
            generatedTokenIndex: token.index,
          }));
        const attentionContextTokens = promptContextTokens.length > 0
          ? [...promptContextTokens, ...generatedContextTokens]
          : undefined;
        return nerdMode
          ? <div className="inspector-stack"><TokenDetail branchUnavailableReason={branchUnavailableReason} branching={branching} canBranch={runComplete && tokenPersisted} contextTokens={attentionContextTokens} nerdMode onBranchAlternative={onBranchAlternative} onSelectToken={onSelectToken} token={selected} /><Card flush icon={ListOrdered} title="All generated tokens"><VirtualTokenTable onSelectToken={onSelectToken} selectedToken={selectedToken} tokens={run?.tokens ?? []} /></Card></div>
          : <EmptyState description="Enable Nerd Mode to inspect raw token boundaries, protocol markers, and alternative distributions." icon={Binary} title="Token details are hidden" />;
      }
      case "probability": return <div className="inspector-stack">{chart("Chosen-token probability", probabilityMetrics)}{chart("Chosen-token log probability", logProbabilityMetrics)}{chart("Uncertainty", uncertaintyMetrics)}{chart("Exact model rank", rankMetrics)}{chart("Running perplexity", perplexityMetrics)}</div>;
      case "timing": return (
        <div className="inspector-stack">
          {run?.tokens[0] && <Callout>The first token includes prefill and is not treated as steady-state decode.</Callout>}
          {chart("Token compute", timingMetrics)}
          {chart("Inter-token arrival", arrivalMetrics)}
          {chart("Decode throughput", throughputMetrics)}
          {chart("Cumulative generation time", cumulativeTimingMetrics)}
          <StatGrid><Stat label="Queue" value={formatDuration(run?.metrics?.timing?.queueMs)} /><Stat label="Tokenization" value={formatDuration(run?.metrics?.timing?.tokenizationMs)} /><Stat label="Prefill" value={formatDuration(run?.metrics?.timing?.prefillMs)} /><Stat label="Client TTFT" value={formatDuration(run?.metrics?.timing?.clientTtftMs)} /></StatGrid>
        </div>
      );
      case "experts": return <ExpertsPanel nerdMode={nerdMode} onSelectToken={onSelectToken} run={run} selectedToken={selectedToken} />;
      case "context": return <ContextPanel model={model} run={run} />;
      case "embeddings": return <EmptyState description="Create or select an embedding run in the Embeddings workspace to inspect vectors, similarity, and projections." icon={Box} title="No embedding run selected" />;
      case "hardware": return <HardwarePanel health={health} run={run} />;
      case "configuration": return (
        <div className="inspector-stack">
          <Card icon={Braces} title={run?.effectiveSettings ? "Effective run settings" : "Effective application configuration"}>
            {run?.effectiveSettings || configuration?.effective ? <pre className="code-block">{JSON.stringify(run?.effectiveSettings ?? configuration?.effective, null, 2)}</pre> : <p className="muted">No redacted effective configuration has been returned.</p>}
          </Card>
          <Card icon={Activity} title={run ? "Sampling operation order" : "Configuration precedence"}>
            {(run?.samplingPipeline ?? configuration?.precedence)?.length ? <ol className="ordered-list">{(run?.samplingPipeline ?? configuration?.precedence ?? []).map((step) => <li key={step}>{step}</li>)}</ol> : <p className="muted">The configuration order was not reported.</p>}
          </Card>
        </div>
      );
      case "events": return nerdMode
        ? (
          <div className="inspector-stack">
            <div className="raw-events-heading">
              <span>{String(run?.rawEvents?.length ?? 0)} live events</span>
              {run && (
                <div className="raw-events-actions">
                  <Button disabled={exporting !== null} loading={exporting === "json"} onClick={() => void exportRun("json")} size="sm">JSON</Button>
                  <Button disabled={exporting !== null} loading={exporting === "jsonl"} onClick={() => void exportRun("jsonl")} size="sm">JSONL</Button>
                  <Button disabled={exporting !== null} loading={exporting === "csv"} onClick={() => void exportRun("csv")} size="sm">Token CSV</Button>
                </div>
              )}
            </div>
            {exportError && <Callout tone="danger">{exportError}</Callout>}
            {run?.rawEvents?.length ? <pre className="code-block event-log">{run.rawEvents.map((event) => JSON.stringify(event)).join("\n")}</pre> : <EmptyState description="Raw protocol events appear here as they arrive. Use the server export above for the complete persisted trace." icon={FileJson} title="No live events captured" />}
          </div>
        )
        : <EmptyState description="Enable Nerd Mode to inspect raw protocol events and export token-level traces." icon={FileJson} title="Raw events are hidden" />;
    }
  }, [visibleTab, branching, configuration, exportError, exportRun, exporting, health, model, nerdMode, onBranchAlternative, onSelectToken, run, selected, selectedToken]);

  return (
    <aside className={`inspector ${open ? "open" : ""}`} aria-label="Run inspector">
      <header className="inspector-header">
        <div className="inspector-title"><h2>Run inspector</h2>{run ? <Badge tone={runTones[run.status]}>{run.status}</Badge> : <Badge>no run</Badge>}</div>
        <IconButton icon={<X size={16} />} label="Close inspector" onClick={onClose} />
      </header>
      <div className="inspector-nav">
        <Tabs className="inspector-tabs" idPrefix="inspector" items={tabItems} label="Inspector sections" onChange={onTabChange} value={visibleTab} />
        <div className="inspector-mobile-select">
          <Select aria-label="Inspector section" onChange={(event) => onTabChange(event.target.value as InspectorTab)} value={visibleTab}>
            {tabItems.map((tab) => <option disabled={tab.disabled} key={tab.id} value={tab.id}>{tab.disabled ? `${tab.label} (unavailable)` : tab.label}</option>)}
          </Select>
        </div>
      </div>
      <div aria-labelledby={`inspector-tab-${visibleTab}`} className="inspector-content" id="inspector-panel" role="tabpanel">{panel}</div>
      {activeCapability && activeCapability.state === "partial" && <div className="capability-footer"><Info aria-hidden="true" size={13} /><span>Partial support: {activeCapability.reason ?? "Some fields may be unavailable."}</span></div>}
    </aside>
  );
}
