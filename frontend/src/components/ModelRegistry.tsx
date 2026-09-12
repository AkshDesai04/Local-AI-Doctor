import {
  AlertCircle,
  Check,
  ChevronDown,
  CircleDashed,
  CircleOff,
  Cpu,
  Database,
  Fingerprint,
  HardDrive,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldCheck,
  Square,
  TriangleAlert,
} from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import type { CapabilityKey, CapabilityState, ModelInspection, ModelSummary } from "../api/types";
import { capabilityKeys, capabilityLabel, capabilityOf } from "../domain/capabilities";
import { formatNumber, shortFingerprint } from "../utils/format";

interface ModelRegistryProps {
  models: ModelSummary[];
  connected: boolean;
  onRefresh: () => void;
  onToggleLoaded: (model: ModelSummary) => void;
  onSelectModel: (id: string) => void;
}

const matrixKeys: CapabilityKey[] = ["text_generation", "embeddings", "vision", "audio", "video", "reasoning_segments", "moe_routing", "raw_logits", "prompt_scoring", "streaming", "cpu", "cuda"];

function StateIcon({ state }: { state: CapabilityState }): React.ReactNode {
  if (state === "full") return <Check size={13} />;
  if (state === "partial") return <CircleDashed size={13} />;
  if (state === "unavailable_on_backend") return <AlertCircle size={13} />;
  return <CircleOff size={13} />;
}

function ModelCard({ model, onToggleLoaded, onSelectModel }: { model: ModelSummary; onToggleLoaded: (model: ModelSummary) => void; onSelectModel: (id: string) => void }): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const [inspection, setInspection] = useState<ModelInspection | null>(null);
  const [inspectionOpen, setInspectionOpen] = useState(false);
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  const busy = model.lifecycle === "loading" || model.lifecycle === "unloading";
  const fullCount = capabilityKeys.filter((key) => capabilityOf(model, key).state === "full").length;
  const inspectFiles = async (): Promise<void> => {
    if (inspection) {
      setInspectionOpen((value) => !value);
      return;
    }
    setInspectionLoading(true);
    setInspectionError(null);
    try {
      setInspection(await api.inspectModel(model.id));
      setInspectionOpen(true);
    } catch (cause) {
      setInspectionError(cause instanceof Error ? cause.message : "Model inspection failed.");
    } finally {
      setInspectionLoading(false);
    }
  };
  return (
    <article className={`registry-model-card ${model.lifecycle}`}>
      <div className="registry-model-main">
        <button className="registry-model-summary" onClick={() => setExpanded((value) => !value)} type="button">
          <div className="model-glyph">{model.task.includes("embedding") ? <Fingerprint size={20} /> : <Cpu size={20} />}<span className={`model-dot ${model.lifecycle}`} /></div>
          <div className="registry-model-copy"><div><h2>{model.name}</h2><span className={`task-badge ${model.task}`}>{model.task.replaceAll("_", " ")}</span></div><p>{model.architecture ?? "Architecture not identified"}</p><div className="model-mini-meta"><span>{model.parameterCount ? `${formatNumber(model.parameterCount / 1_000_000_000, 2)}B params` : "Parameter count unknown"}</span><span>{model.dtype ?? "dtype unknown"}</span><span>{model.effectiveContextLimit ? `${formatNumber(model.effectiveContextLimit, 0)} context` : "context unknown"}</span><span>{String(fullCount)} full capabilities</span></div></div>
          <ChevronDown className={expanded ? "rotated" : ""} size={17} />
        </button>
        <button className={`button ${model.lifecycle === "loaded" ? "secondary" : "primary"}`} disabled={busy} onClick={() => { onSelectModel(model.id); onToggleLoaded(model); }} type="button">{busy ? <LoaderCircle className="spin" size={14} /> : model.lifecycle === "loaded" ? <Square size={12} /> : <Play size={14} />}{model.lifecycle === "loaded" ? "Unload" : busy ? model.lifecycle : "Load"}</button>
      </div>
      {expanded && (
        <div className="registry-model-detail">
          <dl className="definition-grid compact-definitions">
            <div><dt>Fingerprint</dt><dd className="mono" title={model.fingerprint ?? undefined}>{shortFingerprint(model.fingerprint)}</dd></div>
            <div><dt>Lifecycle</dt><dd>{model.lifecycle}{model.loadedDevice ? ` on ${model.loadedDevice}` : ""}</dd></div>
            <div><dt>Remote code</dt><dd>{model.trustRemoteCode ? "Narrowly enabled" : "Disabled"}</dd></div>
            <div><dt>Task selection</dt><dd>{model.task.replaceAll("_", " ")}</dd></div>
          </dl>
          {model.contextLimits?.length ? <div className="model-context-candidates"><span className="eyebrow">Context metadata</span>{model.contextLimits.map((candidate) => <div className={candidate.selected ? "selected" : ""} key={candidate.source}><span>{candidate.source}</span><strong>{candidate.tokens === null ? "Unknown" : formatNumber(candidate.tokens, 0)}</strong>{candidate.note && <small>{candidate.note}</small>}</div>)}</div> : null}
          {model.diagnostics?.length ? <div className="model-diagnostics"><span className="eyebrow">Adapter diagnostics</span>{model.diagnostics.map((diagnostic, index) => <p key={index}><TriangleAlert size={14} />{diagnostic}</p>)}</div> : <div className="model-diagnostic-ok"><ShieldCheck size={15} /> No adapter diagnostics reported</div>}
          <div className="model-inspection-control"><button className="button secondary compact" disabled={inspectionLoading} onClick={() => void inspectFiles()} type="button">{inspectionLoading ? <LoaderCircle className="spin" size={13} /> : <Fingerprint size={13} />} {inspectionOpen ? "Hide tokenizer details" : "Inspect tokenizer & template"}</button>{inspectionError && <span>{inspectionError}</span>}</div>
          {inspectionOpen && inspection && <div className="model-inspection"><section><span className="eyebrow">Chat template</span>{inspection.chatTemplate ? <pre>{inspection.chatTemplate}</pre> : <p>Not present in the inspected model folder.</p>}</section><section><span className="eyebrow">Special tokens</span>{inspection.specialTokens ? <pre>{JSON.stringify(inspection.specialTokens, null, 2)}</pre> : <p>Not reported.</p>}</section><section><span className="eyebrow">Sampling order</span>{inspection.samplingPipeline?.length ? <ol>{inspection.samplingPipeline.map((step) => <li key={step}>{step}</li>)}</ol> : <p>Not reported.</p>}</section></div>}
        </div>
      )}
    </article>
  );
}

export function ModelRegistry({ models, connected, onRefresh, onToggleLoaded, onSelectModel }: ModelRegistryProps): React.ReactNode {
  return (
    <main className="model-registry">
      <header className="workspace-titlebar registry-titlebar">
        <div><span className="eyebrow">Read-only discovery</span><h1>Model registry</h1><p>Capabilities are adapter claims with reasons—not guesses based on architecture names.</p></div>
        <button className="button primary" disabled={!connected} onClick={onRefresh} type="button"><RefreshCw size={15} /> Rescan roots</button>
      </header>
      {!models.length ? (
        <div className="registry-empty"><div className="empty-orbit"><Database size={23} /></div><h2>{connected ? "No model folders discovered" : "Backend offline"}</h2><p>{connected ? "Check the effective configuration for a readable model root, then rescan. Model roots are never modified." : "Start the local backend before scanning configured roots."}</p></div>
      ) : (
        <>
          <section className="registry-overview">
            <div><HardDrive size={17} /><span>Discovered</span><strong>{String(models.length)}</strong></div>
            <div><Cpu size={17} /><span>Loaded</span><strong>{String(models.filter((model) => model.lifecycle === "loaded").length)}</strong></div>
            <div><Fingerprint size={17} /><span>Fingerprint ready</span><strong>{String(models.filter((model) => model.fingerprint).length)}</strong></div>
          </section>
          <section className="registry-list"><div className="registry-section-heading"><h2>Local models</h2><span>Load one at a time by default</span></div>{models.map((model) => <ModelCard key={model.id} model={model} onSelectModel={onSelectModel} onToggleLoaded={onToggleLoaded} />)}</section>
          <section className="capability-matrix-section">
            <div className="registry-section-heading"><div><h2>Capability matrix</h2><p>Non-full states always retain their adapter explanation.</p></div><div className="matrix-legend"><span className="full"><Check size={11} /> Full</span><span className="partial"><CircleDashed size={11} /> Partial</span><span className="unavailable_on_backend"><AlertCircle size={11} /> Backend</span><span className="unsupported"><CircleOff size={11} /> Unsupported</span></div></div>
            <div className="capability-table-wrap"><table className="capability-table"><thead><tr><th>Capability</th>{models.map((model) => <th key={model.id}>{model.name}</th>)}</tr></thead><tbody>{matrixKeys.map((key) => <tr key={key}><th>{capabilityLabel(key)}</th>{models.map((model) => { const capability = capabilityOf(model, key); return <td key={model.id}><span className={`capability-state ${capability.state}`} title={capability.reason ?? capability.state}><StateIcon state={capability.state} />{capability.state === "unavailable_on_backend" ? "backend" : capability.state}</span>{capability.reason && <small>{capability.reason}</small>}</td>; })}</tr>)}</tbody></table></div>
          </section>
        </>
      )}
    </main>
  );
}
