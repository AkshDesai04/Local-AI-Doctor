import {
  Activity,
  Binary,
  Box,
  Braces,
  ChevronDown,
  Clock3,
  Cpu,
  FileJson,
  Fingerprint,
  Gauge,
  GitBranch,
  Info,
  Layers3,
  Network,
  Sigma,
  Sparkles,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { AlternativeDistribution, CapabilityKey, ConfigurationSnapshot, HealthStatus, ModelSummary, RunDetails, TokenAlternative, TokenEvent } from "../api/types";
import { fetchRunExport } from "../api/client";
import { capabilityOf, supportsGeneration } from "../domain/capabilities";
import { displayTokenText, downloadBlob, formatBytes, formatDuration, formatNumber, formatPercent, shortFingerprint, tokenTextHint } from "../utils/format";
import { type ChartMetric, TraceChart } from "./TraceChart";
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

function DataCard({ label, value, detail }: { label: string; value: React.ReactNode; detail?: string }): React.ReactNode {
  return <div className="data-card"><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>;
}

function EmptyInspector({ icon: Icon = Info, title, body }: { icon?: LucideIcon; title: string; body: string }): React.ReactNode {
  return <div className="inspector-empty"><div><Icon size={19} /></div><strong>{title}</strong><p>{body}</p></div>;
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
        <div className="alternatives-table">
          <div className="alternatives-head"><span title="Probability rank within this distribution">Rank</span><span title="Human-readable token text; hover a row for the exact tokenizer piece">Piece</span><span title="Probability assigned to this token">Probability</span><span title="Natural logarithm of the token probability">Log p</span><span title="Whether this token survived the active sampling filters">Filter</span></div>
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
              className={`alternatives-row ${nerdMode && !isCurrent ? "selectable" : ""} ${isSelected ? "selected" : ""}`}
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

function TokenDetail({ token, nerdMode, branching, canBranch, branchUnavailableReason, onBranchAlternative }: {
  token: TokenEvent | undefined;
  nerdMode: boolean;
  branching: boolean;
  canBranch: boolean;
  branchUnavailableReason: string;
  onBranchAlternative: (tokenIndex: number, distribution: AlternativeDistribution, alternative: TokenAlternative) => Promise<void>;
}): React.ReactNode {
  const [selectedAlternative, setSelectedAlternative] = useState<SelectedAlternative | null>(null);

  useEffect(() => {
    setSelectedAlternative(null);
  }, [nerdMode, token?.index]);

  if (!token) return <EmptyInspector body="Select a token in the response, graph, or table to lock its exact details here." title="No token selected" />;
  const branch = async (): Promise<void> => {
    if (!selectedAlternative || branching || !canBranch) return;
    await onBranchAlternative(token.index, selectedAlternative.distribution, selectedAlternative.alternative);
  };
  const selectedTokenLabel = displayTokenText(token.displayText || token.piece) || "∅";
  return (
    <div className="token-detail">
      <div className="token-detail-hero" title={tokenTextHint(token.piece, token.displayText)}><span className={`segment-badge ${token.reasoningSegment}`}>{token.reasoningSegment}</span><code>{selectedTokenLabel}</code><span>token #{String(token.index)}</span></div>
      <dl className="definition-grid">
        <div><dt>Token ID</dt><dd>{String(token.tokenId)}</dd></div>
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
          <button disabled={branching || !canBranch} onClick={() => void branch()} title={!canBranch ? branchUnavailableReason : "Create a new chat from this point and force the selected alternative before continuing generation."} type="button"><GitBranch size={14} />{branching ? "Creating branch…" : canBranch ? "Branch out with selected token" : branchUnavailableReason}</button>
        </div>
      )}
      <div className="metric-note"><Info size={14} /><span>Alternatives are tokens with high probability under a distribution—not model thoughts or hidden reasoning.</span></div>
    </div>
  );
}

function OverviewPanel({ run, model }: { run: RunDetails | null; model: ModelSummary | null }): React.ReactNode {
  if (!run) return <EmptyInspector body="Run a prompt or open response details to inspect telemetry. No placeholder measurements are shown." title="No run selected" />;
  const timing = run.metrics?.timing;
  return (
    <div className="inspector-stack">
      <div className="run-status-row"><span className={`run-state ${run.status}`}><i />{run.status}</span><span>{run.metrics?.finishReason ?? "No finish reason yet"}</span></div>
      {run.warnings?.map((warning, index) => <div className="inline-warning" key={index}><Info size={14} /><span>{warning}</span></div>)}
      <div className="data-grid">
        <DataCard detail="request → first server token" label="Server TTFT" value={formatDuration(timing?.serverTtftMs)} />
        <DataCard detail="steady-state decode" label="Decode rate" value={timing?.decodeTokensPerSecond === undefined ? "—" : `${formatNumber(timing.decodeTokensPerSecond, 2)} tok/s`} />
        <DataCard detail="conditional generated response" label="Perplexity" value={formatNumber(run.metrics?.responsePerplexity, 4)} />
        <DataCard detail="prompt + completion" label="Total latency" value={formatDuration(timing?.totalMs)} />
        <DataCard label="Prompt tokens" value={formatNumber(run.metrics?.promptTokens, 0)} />
        <DataCard label="Generated tokens" value={formatNumber(run.metrics?.generatedTokens ?? run.tokens.length, 0)} />
      </div>
      <section className="inspector-section">
        <div className="section-title"><Fingerprint size={15} /><h3>Reproducibility</h3></div>
        {!run.reproducibility ? <p className="muted">The backend has not returned a reproducibility snapshot.</p> : (
          <dl className="definition-grid compact-definitions">
            <div><dt>Requested seed</dt><dd>{run.reproducibility.requestedSeed === null || run.reproducibility.requestedSeed === undefined || run.reproducibility.requestedSeed === "" ? "Generated" : String(run.reproducibility.requestedSeed)}</dd></div>
            <div><dt>Effective seed</dt><dd className="mono">{run.reproducibility.effectiveSeed}</dd></div>
            <div><dt>RNG</dt><dd>{run.reproducibility.rngAlgorithm ?? "Not reported"}</dd></div>
            <div><dt>Generator</dt><dd>{run.reproducibility.generatorDevice ?? "Not reported"}</dd></div>
            <div><dt>Model fingerprint</dt><dd>{shortFingerprint(run.reproducibility.modelFingerprint ?? model?.fingerprint)}</dd></div>
            <div><dt>Backend</dt><dd>{run.reproducibility.backend ?? "Not reported"}</dd></div>
            <div><dt>Device / dtype</dt><dd title={run.reproducibility.deviceReason}><span>{run.reproducibility.device ?? "—"} / {run.reproducibility.dtype ?? "—"}</span>{run.reproducibility.deviceReason && <small className="definition-detail">{run.reproducibility.deviceReason}</small>}</dd></div>
            <div><dt>Deterministic kernels</dt><dd>{run.reproducibility.deterministicKernels === undefined ? "Not reported" : run.reproducibility.deterministicKernels ? "Enabled" : "Disabled"}</dd></div>
          </dl>
        )}
        <p className="footnote">A seed cannot be recovered from output. Exact replay also depends on unchanged files, tokenizer, device, dtype, kernels, software, and batching.</p>
      </section>
      {run.metrics?.telemetryOverhead && <div className="metric-note"><Activity size={14} /><span>Instrumentation overhead: {run.metrics.telemetryOverhead}</span></div>}
    </div>
  );
}

const probabilityMetrics: ChartMetric[] = [
  { key: "raw-p", label: "Raw model probability", color: "#60a5fa", value: (token) => token.rawProbability },
  { key: "sample-p", label: "Sampler probability", color: "#b49cff", value: (token) => token.samplingProbability },
];
const logProbabilityMetrics: ChartMetric[] = [
  { key: "raw-log-p", label: "Raw model log probability", color: "#60a5fa", value: (token) => token.rawLogProbability },
  { key: "sample-log-p", label: "Sampler log probability", color: "#b49cff", value: (token) => token.samplingLogProbability },
];
const uncertaintyMetrics: ChartMetric[] = [
  { key: "entropy", label: "Entropy", color: "#5bb8ef", value: (token) => token.entropy },
  { key: "surprise", label: "Surprise −ln(p)", color: "#f3a769", value: (token) => token.surprise },
];
const rankMetrics: ChartMetric[] = [
  { key: "rank", label: "Exact full-vocabulary rank", color: "#f3a769", value: (token) => token.rawRank },
];
const perplexityMetrics: ChartMetric[] = [
  { key: "perplexity", label: "Running response perplexity", color: "#b49cff", value: (token) => token.runningPerplexity },
];
const timingMetrics: ChartMetric[] = [
  { key: "decode", label: "Decode forward", unit: "ms", color: "#60a5fa", value: (token) => token.timing?.decodeMs },
  { key: "sampling", label: "Sampling", unit: "ms", color: "#b49cff", value: (token) => token.timing?.samplingMs },
];
const arrivalMetrics: ChartMetric[] = [
  { key: "server-arrival", label: "Server inter-token", unit: "ms", color: "#f3a769", value: (token) => token.timing?.interTokenMs },
  { key: "client-arrival", label: "Client inter-arrival", unit: "ms", color: "#68bce8", value: (token) => token.timing?.clientInterArrivalMs },
];
const cumulativeTimingMetrics: ChartMetric[] = [
  { key: "cumulative", label: "Cumulative time", unit: "ms", color: "#60a5fa", value: (token) => token.timing?.cumulativeMs },
];
const throughputMetrics: ChartMetric[] = [
  { key: "instant", label: "Instantaneous", unit: "tok/s", color: "#5bb8ef", value: (token) => token.timing?.instantaneousTps },
  { key: "rolling", label: "Rolling", unit: "tok/s", color: "#b49cff", value: (token) => token.timing?.rollingTps },
];

const expertSelectionMetrics: ChartMetric[] = [
  { key: "expert-id", label: "First selected expert ID", color: "#b49cff", value: (token) => token.expertRoutes?.[0]?.selectedExpertIds[0] },
  { key: "expert-weight", label: "First expert weight", color: "#60a5fa", value: (token) => token.expertRoutes?.[0]?.gateWeights[0] },
];

function ExpertsPanel({ run, selectedToken, nerdMode, onSelectToken }: { run: RunDetails | null; selectedToken: number | null; nerdMode: boolean; onSelectToken: (index: number) => void }): React.ReactNode {
  const routed = run?.tokens.filter((token) => token.expertRoutes?.length) ?? [];
  if (!routed.length) return <EmptyInspector body="The selected adapter supports routing, but this run did not expose per-token routes at its instrumentation level." icon={Network} title="No expert route events" />;
  return (
    <div className="inspector-stack">
      <TraceChart metrics={expertSelectionMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Expert selection and weight" tokens={run?.tokens ?? []} />
      <section className="inspector-section">
        <div className="section-title"><Network size={15} /><h3>Selected vs. executed experts</h3></div>
        <div className="expert-route-table"><div><strong>Token</strong><strong>Layer</strong><strong>Selected</strong><strong>Executed</strong><strong>Weights</strong></div>{routed.flatMap((token) => token.expertRoutes?.map((route) => <button key={`${String(token.index)}-${String(route.layer)}`} onClick={() => onSelectToken(token.index)} type="button"><span>#{String(token.index)}</span><span>{String(route.layer)}</span><span>{route.selectedExpertIds.join(", ")}</span><span>{route.executedExpertIds.join(", ")}</span><span>{route.gateWeights.map((weight) => formatNumber(weight, 3)).join(", ")}</span></button>) ?? [])}</div>
      </section>
      <div className="metric-note"><Info size={14} /><span>Selected and actually executed experts are intentionally distinct. Dense gated MLPs are never presented as MoE routing.</span></div>
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
  if (!run && !model) return <EmptyInspector body="Select a model to see its discovered context metadata." title="No context metadata" />;
  return (
    <div className="inspector-stack">
      <div className="context-hero"><div><span className="eyebrow">Effective model limit</span><strong>{limit === null ? "Unknown" : formatNumber(limit, 0)}</strong></div><div><span className="eyebrow">Used + reserved</span><strong>{context ? formatNumber(used, 0) : "No run"}</strong></div><div><span className="eyebrow">Remaining</span><strong>{remaining === undefined ? "—" : formatNumber(remaining, 0)}</strong></div></div>
      {context && limit !== null && <div className="context-bar" aria-label={`Context used ${formatPercent(used / limit)}`}>{parts.map((part) => <i className={part.className} key={part.label} style={{ width: `${String((part.value / limit) * 100)}%` }} />)}</div>}
      <div className="context-legend">{parts.map((part) => <div key={part.label}><i className={part.className} /><span>{part.label}</span><strong>{context ? formatNumber(part.value, 0) : "—"}</strong></div>)}</div>
      <div className="metric-note"><Info size={14} /><span>Text tokens and expanded multimodal positions are kept separate. Remaining capacity includes the reserved output budget.</span></div>
      <section className="inspector-section">
        <div className="section-title"><Layers3 size={15} /><h3>Discovered limits</h3></div>
        {!model?.contextLimits?.length ? <p className="muted">The adapter did not expose context-limit candidates.</p> : model.contextLimits.map((candidate) => <div className={`context-candidate ${candidate.selected ? "selected" : ""}`} key={candidate.source}><span>{candidate.source}</span><strong>{candidate.tokens === null ? "Unknown" : formatNumber(candidate.tokens, 0)}</strong>{candidate.note && <small>{candidate.note}</small>}</div>)}
      </section>
      <dl className="definition-grid compact-definitions"><div><dt>Truncation behavior</dt><dd>{context?.behavior?.replaceAll("_", " ") ?? "Not observed"}</dd></div><div><dt>Selection</dt><dd>{limit === null ? "Conflicting or absent metadata" : "Conservative effective limit"}</dd></div></dl>
    </div>
  );
}

function HardwarePanel({ run, health }: { run: RunDetails | null; health: HealthStatus | null }): React.ReactNode {
  return (
    <div className="inspector-stack">
      <div className="data-grid">
        <DataCard detail="selected by hardware discovery" label="Backend" value={run?.reproducibility?.backend ?? health?.selectedBackend ?? "Not reported"} />
        <DataCard label="Device" value={run?.reproducibility?.device ?? "Not reported"} />
        <DataCard label="Peak RAM" value={formatBytes(run?.metrics?.peakRamBytes)} />
        <DataCard label="Peak VRAM" value={formatBytes(run?.metrics?.peakVramBytes)} />
        <DataCard label="KV cache" value={formatBytes(run?.metrics?.kvCacheBytes)} />
        <DataCard label="Attention" value={run?.reproducibility?.attentionImplementation ?? "Not reported"} />
      </div>
      <div className="metric-note"><Info size={14} /><span>Memory peaks are exact allocator measurements where available. Utilization samples are not exact per-token measurements.</span></div>
      {!run && <EmptyInspector body="Hardware discovery is available above. Per-run memory and cache measurements appear after selecting a run." title="No run hardware snapshot" />}
    </div>
  );
}

export function Inspector({ open, model, run, health, configuration, activeTab, selectedToken, nerdMode, branching, onBranchAlternative, onTabChange, onSelectToken, onClose }: InspectorProps): React.ReactNode {
  const selected = run?.tokens.find((token) => token.index === selectedToken);
  const [mobileTabsOpen, setMobileTabsOpen] = useState(false);
  const [exporting, setExporting] = useState<"json" | "jsonl" | "csv" | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const activeDefinition = tabs.find((tab) => tab.id === activeTab) ?? tabs[0];
  const ActiveIcon = activeDefinition?.icon;
  const activeCapability = activeDefinition?.capability ? capabilityOf(model, activeDefinition.capability) : null;
  const activeCapabilityDisabled = activeDefinition?.id === "timing"
    ? !supportsGeneration(model)
    : activeCapability !== null && activeCapability.state !== "full" && activeCapability.state !== "partial";

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
    switch (activeTab) {
      case "overview": return <OverviewPanel model={model} run={run} />;
      case "tokens": {
        const runComplete = run?.status === "complete";
        const tokenPersisted = selected !== undefined
          && run?.branchableThroughTokenIndex !== undefined
          && selected.index <= run.branchableThroughTokenIndex;
        const branchUnavailableReason = !runComplete
          ? "Available when run completes"
          : "Selected token was not retained in persisted telemetry";
        return nerdMode
          ? <div className="inspector-stack"><TokenDetail branchUnavailableReason={branchUnavailableReason} branching={branching} canBranch={runComplete && tokenPersisted} nerdMode onBranchAlternative={onBranchAlternative} token={selected} /><section className="inspector-section"><div className="section-title"><Binary size={15} /><h3>All generated tokens</h3></div><VirtualTokenTable onSelectToken={onSelectToken} selectedToken={selectedToken} tokens={run?.tokens ?? []} /></section></div>
          : <EmptyInspector body="Enable Nerd Mode to inspect raw token boundaries, protocol markers, and alternative distributions." icon={Binary} title="Token details are hidden" />;
      }
      case "probability": return <div className="inspector-stack"><TraceChart metrics={probabilityMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Chosen-token probability" tokens={run?.tokens ?? []} /><TraceChart metrics={logProbabilityMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Chosen-token log probability" tokens={run?.tokens ?? []} /><TraceChart metrics={uncertaintyMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Uncertainty" tokens={run?.tokens ?? []} /><TraceChart metrics={rankMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Exact model rank" tokens={run?.tokens ?? []} /><TraceChart metrics={perplexityMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Running perplexity" tokens={run?.tokens ?? []} /></div>;
      case "timing": return <div className="inspector-stack">{run?.tokens[0] && <div className="metric-note"><Info size={14} /><span>The first token includes prefill and is not treated as steady-state decode.</span></div>}<TraceChart metrics={timingMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Token compute" tokens={run?.tokens ?? []} /><TraceChart metrics={arrivalMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Inter-token arrival" tokens={run?.tokens ?? []} /><TraceChart metrics={throughputMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Decode throughput" tokens={run?.tokens ?? []} /><TraceChart metrics={cumulativeTimingMetrics} onSelectToken={onSelectToken} revealTokenText={nerdMode} selectedToken={selectedToken} title="Cumulative generation time" tokens={run?.tokens ?? []} /><div className="data-grid"><DataCard label="Queue" value={formatDuration(run?.metrics?.timing?.queueMs)} /><DataCard label="Tokenization" value={formatDuration(run?.metrics?.timing?.tokenizationMs)} /><DataCard label="Prefill" value={formatDuration(run?.metrics?.timing?.prefillMs)} /><DataCard label="Client TTFT" value={formatDuration(run?.metrics?.timing?.clientTtftMs)} /></div></div>;
      case "experts": return <ExpertsPanel nerdMode={nerdMode} onSelectToken={onSelectToken} run={run} selectedToken={selectedToken} />;
      case "context": return <ContextPanel model={model} run={run} />;
      case "embeddings": return <EmptyInspector body="Create or select an embedding run in the Embeddings workspace to inspect vectors, similarity, and projections." icon={Box} title="No embedding run selected" />;
      case "hardware": return <HardwarePanel health={health} run={run} />;
      case "configuration": return <div className="inspector-stack"><section className="json-section"><div className="section-title"><Braces size={15} /><h3>{run?.effectiveSettings ? "Effective run settings" : "Effective application configuration"}</h3></div>{run?.effectiveSettings || configuration?.effective ? <pre>{JSON.stringify(run?.effectiveSettings ?? configuration?.effective, null, 2)}</pre> : <p className="muted">No redacted effective configuration has been returned.</p>}</section><section className="json-section"><div className="section-title"><Activity size={15} /><h3>{run ? "Sampling operation order" : "Configuration precedence"}</h3></div>{(run?.samplingPipeline ?? configuration?.precedence)?.length ? <ol>{(run?.samplingPipeline ?? configuration?.precedence ?? []).map((step) => <li key={step}>{step}</li>)}</ol> : <p className="muted">The configuration order was not reported.</p>}</section></div>;
      case "events": return nerdMode
        ? <div className="inspector-stack"><div className="raw-events-heading"><span>{String(run?.rawEvents?.length ?? 0)} live events</span>{run && <div className="server-export-actions"><button disabled={exporting !== null} onClick={() => void exportRun("json")} type="button">{exporting === "json" ? "Exporting…" : "JSON"}</button><button disabled={exporting !== null} onClick={() => void exportRun("jsonl")} type="button">{exporting === "jsonl" ? "Exporting…" : "JSONL"}</button><button disabled={exporting !== null} onClick={() => void exportRun("csv")} type="button">{exporting === "csv" ? "Exporting…" : "Token CSV"}</button></div>}</div>{exportError && <p className="inline-error" role="alert">{exportError}</p>}{run?.rawEvents?.length ? <pre className="event-log">{run.rawEvents.map((event) => JSON.stringify(event)).join("\n")}</pre> : <EmptyInspector body="Raw protocol events appear here as they arrive. Use the server export above for the complete persisted trace." title="No live events captured" />}</div>
        : <EmptyInspector body="Enable Nerd Mode to inspect raw protocol events and export token-level traces." icon={FileJson} title="Raw events are hidden" />;
    }
  }, [activeTab, branching, configuration, exportError, exportRun, exporting, health, model, nerdMode, onBranchAlternative, onSelectToken, run, selected, selectedToken]);

  return (
    <aside className={`inspector ${open ? "open" : ""}`} aria-label="Run inspector">
      <div className="inspector-header">
        <div><span className="eyebrow">Observability</span><h2>Run inspector</h2></div>
        <div className="inspector-header-actions"><span className={`run-mini-status ${run?.status ?? "idle"}`}>{run?.status ?? "no run"}</span><button aria-label="Close inspector" className="icon-button" onClick={onClose} type="button"><X size={18} /></button></div>
      </div>
      <button className="mobile-tab-select" onClick={() => setMobileTabsOpen((value) => !value)} type="button"><span>{ActiveIcon && <ActiveIcon size={15} />} {activeDefinition?.label}</span><ChevronDown size={15} /></button>
      <div className={`inspector-tabs ${mobileTabsOpen ? "mobile-open" : ""}`} role="tablist">
        {tabs.map((tab) => {
          const capability = tab.capability ? capabilityOf(model, tab.capability) : null;
          const disabled = tab.id === "timing"
            ? !supportsGeneration(model)
            : capability !== null && capability.state !== "full" && capability.state !== "partial";
          const Icon = tab.icon;
          return (
            <button
              aria-disabled={disabled}
              aria-selected={activeTab === tab.id}
              className={`${activeTab === tab.id ? "selected" : ""} ${disabled ? "disabled" : ""}`}
              key={tab.id}
              onClick={() => { if (!disabled) { onTabChange(tab.id); setMobileTabsOpen(false); } }}
              role="tab"
              title={disabled ? capability?.reason ?? `${tab.label} is not available for this model.` : tab.label}
              type="button"
            ><Icon size={15} /><span>{tab.label}</span>{disabled && <i />}</button>
          );
        })}
      </div>
      <div className="inspector-content" role="tabpanel">{panel}</div>
      {activeCapability && activeCapability.state === "partial" && <div className="capability-footer"><Sparkles size={13} /><span>Partial support: {activeCapability.reason ?? "Some fields may be unavailable."}</span></div>}
    </aside>
  );
}
