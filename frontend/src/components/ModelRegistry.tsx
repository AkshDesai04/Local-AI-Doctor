import {
  Check,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CircleDashed,
  CircleOff,
  Cpu,
  Database,
  Fingerprint,
  Play,
  RefreshCw,
  ShieldCheck,
  Square,
} from "lucide-react";
import { Fragment, useState } from "react";
import { api } from "../api/client";
import type { CapabilityKey, CapabilityState, ModelInspection, ModelSummary } from "../api/types";
import { capabilityKeys, capabilityLabel, capabilityOf } from "../domain/capabilities";
import { formatBytes, formatNumber, shortFingerprint } from "../utils/format";
import { ModelRootSettings } from "./ModelRootSettings";
import { Badge, type BadgeTone, Button, Callout, Card, EmptyState, IconButton, Stat } from "./ui";

interface ModelRegistryProps {
  models: ModelSummary[];
  connected: boolean;
  onRefresh: () => void;
  onSynchronize: () => void;
  onToggleLoaded: (model: ModelSummary) => void;
  onSelectModel: (id: string) => void;
}

const matrixKeys: CapabilityKey[] = ["text_generation", "embeddings", "vision", "audio", "video", "reasoning_segments", "moe_routing", "raw_logits", "prompt_scoring", "streaming", "cpu", "cuda"];

const stateLabels: Record<CapabilityState, string> = {
  full: "Full",
  partial: "Partial",
  unavailable_on_backend: "Backend",
  unsupported: "None",
};

const stateDescriptions: Record<CapabilityState, string> = {
  full: "Supported by this model and backend.",
  partial: "Supported with limitations.",
  unavailable_on_backend: "Supported by the model but unavailable on this backend.",
  unsupported: "Not supported by this model.",
};

const taskTones: Record<ModelSummary["task"], BadgeTone> = {
  text_generation: "accent",
  encoder_decoder_generation: "accent",
  embedding: "info",
  multimodal_embedding: "info",
  unknown: "neutral",
};

function taskLabel(task: ModelSummary["task"]): string {
  const text = task.replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function StateIcon({ state }: { state: CapabilityState }): React.ReactNode {
  if (state === "full") return <Check aria-hidden="true" size={13} />;
  if (state === "partial") return <CircleDashed aria-hidden="true" size={13} />;
  if (state === "unavailable_on_backend") return <CircleAlert aria-hidden="true" size={13} />;
  return <CircleOff aria-hidden="true" size={13} />;
}

/** The effective context, or "declared X · tokenizer Y" when the tokenizer reports a different length. */
function contextSummary(model: ModelSummary): { text: string; conflict: boolean } {
  const limit = model.effectiveContextLimit;
  if (!limit) return { text: "Unknown", conflict: false };
  const tokenizer = model.contextLimits?.find((candidate) => candidate.source === "tokenizer_config.model_max_length" && candidate.tokens !== null && candidate.tokens !== limit);
  return tokenizer?.tokens
    ? { text: `declared ${formatNumber(limit, 0)} · tokenizer ${formatNumber(tokenizer.tokens, 0)}`, conflict: true }
    : { text: formatNumber(limit, 0), conflict: false };
}

function ModelCard({ model, onToggleLoaded, onSelectModel }: { model: ModelSummary; onToggleLoaded: (model: ModelSummary) => void; onSelectModel: (id: string) => void }): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const [inspection, setInspection] = useState<ModelInspection | null>(null);
  const [inspectionOpen, setInspectionOpen] = useState(false);
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  const busy = model.lifecycle === "loading" || model.lifecycle === "unloading";
  const loaded = model.lifecycle === "loaded";
  const fullCount = capabilityKeys.filter((key) => capabilityOf(model, key).state === "full").length;
  const context = contextSummary(model);
  const detailId = `model-detail-${model.id}`;
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
    <article aria-label={model.name} className={`model-card ${model.lifecycle}`}>
      <div className="model-card-main">
        <div aria-hidden="true" className="model-glyph">{model.task.includes("embedding") ? <Fingerprint size={18} /> : <Cpu size={18} />}<span className={`model-dot ${model.lifecycle}`} /></div>
        <div className="model-card-copy">
          <div className="model-card-title">
            <h3 title={model.name}>{model.name}</h3>
            <Badge tone={taskTones[model.task]}>{taskLabel(model.task)}</Badge>
            {loaded && <Badge tone="accent">Loaded{model.loadedDevice ? ` · ${model.loadedDevice}` : ""}</Badge>}
            {model.lifecycle === "error" && <Badge tone="danger">Load error</Badge>}
          </div>
          <p className="model-card-arch">{model.architecture ?? "Architecture not identified"}</p>
          <dl className="model-facts">
            <div><dt>Parameters</dt><dd>{model.parameterCount ? `${formatNumber(model.parameterCount / 1_000_000_000, 2)}B` : "Unknown"}</dd></div>
            <div><dt>Context</dt><dd className={context.conflict ? "conflict" : ""} title={context.conflict ? "The architecture declares a different length than the tokenizer's model_max_length; the declared value is used." : undefined}>{context.text}</dd></div>
            <div><dt>Weights dtype</dt><dd>{model.dtype ?? "Unknown"}</dd></div>
            <div><dt>Weight size</dt><dd>{formatBytes(model.weightBytes)}</dd></div>
            <div><dt>Capabilities</dt><dd>{String(fullCount)} full</dd></div>
          </dl>
        </div>
        <div className="model-card-actions">
          <Button
            disabled={busy}
            icon={loaded ? <Square size={11} /> : <Play size={13} />}
            loading={busy}
            onClick={() => { onSelectModel(model.id); onToggleLoaded(model); }}
            size="sm"
            variant={loaded ? "secondary" : "primary"}
          >{loaded ? "Unload" : busy ? (model.lifecycle === "loading" ? "Loading…" : "Unloading…") : "Load"}</Button>
          <IconButton aria-controls={detailId} aria-expanded={expanded} icon={<ChevronDown className={expanded ? "rotated" : ""} size={16} />} label={expanded ? `Hide details for ${model.name}` : `Show details for ${model.name}`} onClick={() => setExpanded((value) => !value)} size="sm" title={expanded ? "Hide details" : "Show details"} />
        </div>
      </div>
      {expanded && (
        <div className="model-card-detail" id={detailId}>
          <dl className="kv-grid">
            <div><dt>Fingerprint</dt><dd className="mono" title={model.fingerprint ?? undefined}>{shortFingerprint(model.fingerprint)}</dd></div>
            <div><dt>Lifecycle</dt><dd>{model.lifecycle}{model.loadedDevice ? ` on ${model.loadedDevice}` : ""}</dd></div>
            <div><dt>Remote code</dt><dd>{model.trustRemoteCode ? "Narrowly enabled" : "Disabled"}</dd></div>
            <div><dt>Task selection</dt><dd>{model.task.replaceAll("_", " ")}</dd></div>
          </dl>
          {model.contextLimits?.length ? (
            <Card flush title="Context metadata">
              <ul className="candidate-list">{model.contextLimits.map((candidate) => <li className={candidate.selected ? "selected" : ""} key={candidate.source}><span className="mono">{candidate.source}</span><strong>{candidate.tokens === null ? "Unknown" : formatNumber(candidate.tokens, 0)}</strong>{candidate.note && <small>{candidate.note}</small>}</li>)}</ul>
            </Card>
          ) : null}
          {model.diagnostics?.length
            ? <Callout title="Adapter diagnostics" tone="warning"><ul className="diagnostic-list">{model.diagnostics.map((diagnostic, index) => <li key={index}>{diagnostic}</li>)}</ul></Callout>
            : <Callout icon={ShieldCheck}>No adapter diagnostics reported.</Callout>}
          <div className="model-inspection-control">
            <Button icon={<Fingerprint size={13} />} loading={inspectionLoading} onClick={() => void inspectFiles()} size="sm">{inspectionOpen ? "Hide tokenizer details" : "Inspect tokenizer & template"}</Button>
            {inspectionError && <span className="text-danger" role="alert">{inspectionError}</span>}
          </div>
          {inspectionOpen && inspection && (
            <div className="model-inspection">
              <Card title="Chat template">{inspection.chatTemplate ? <pre className="code-block">{inspection.chatTemplate}</pre> : <p className="muted">Not present in the inspected model folder.</p>}</Card>
              <Card title="Special tokens">{inspection.specialTokens ? <pre className="code-block">{JSON.stringify(inspection.specialTokens, null, 2)}</pre> : <p className="muted">Not reported.</p>}</Card>
              <Card title="Sampling order">{inspection.samplingPipeline?.length ? <ol className="ordered-list">{inspection.samplingPipeline.map((step) => <li key={step}>{step}</li>)}</ol> : <p className="muted">Not reported.</p>}</Card>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function CapabilityMatrix({ models }: { models: ModelSummary[] }): React.ReactNode {
  const [openRow, setOpenRow] = useState<CapabilityKey | null>(null);
  return (
    <div className="matrix-wrap">
      <table className="capability-matrix">
        <thead>
          <tr><th scope="col">Capability</th>{models.map((model) => <th key={model.id} scope="col" title={model.name}>{model.name}</th>)}</tr>
        </thead>
        <tbody>
          {matrixKeys.map((key) => {
            const open = openRow === key;
            return (
              <Fragment key={key}>
                <tr className={open ? "open" : ""}>
                  <th scope="row">
                    <button aria-controls={`matrix-detail-${key}`} aria-expanded={open} className="matrix-row-toggle" onClick={() => setOpenRow(open ? null : key)} title={open ? "Hide the reasons" : "Show every model's reason"} type="button">
                      <ChevronRight aria-hidden="true" className={open ? "open" : ""} size={13} />{capabilityLabel(key)}
                    </button>
                  </th>
                  {models.map((model, index) => {
                    const capability = capabilityOf(model, key);
                    const reason = capability.reason ?? stateDescriptions[capability.state];
                    const reasonId = `capability-${key}-${String(index)}`;
                    return (
                      <td key={model.id}>
                        <span aria-describedby={reasonId} className={`capability-state ${capability.state}`} tabIndex={0} title={reason}>
                          <StateIcon state={capability.state} />{stateLabels[capability.state]}
                        </span>
                        <span className="visually-hidden" id={reasonId}>{reason}</span>
                      </td>
                    );
                  })}
                </tr>
                {open && (
                  <tr className="matrix-detail" id={`matrix-detail-${key}`}>
                    <td colSpan={models.length + 1}>
                      <ul>
                        {models.map((model) => {
                          const capability = capabilityOf(model, key);
                          return <li key={model.id}><span className={`capability-state ${capability.state}`}><StateIcon state={capability.state} />{stateLabels[capability.state]}</span><strong>{model.name}</strong><span>{capability.reason ?? stateDescriptions[capability.state]}</span></li>;
                        })}
                      </ul>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ModelRegistry({ models, connected, onRefresh, onSynchronize, onToggleLoaded, onSelectModel }: ModelRegistryProps): React.ReactNode {
  return (
    <main className="workspace model-registry">
      <header className="page-header">
        <div><h1>Model registry</h1><p>Read-only discovery. Capabilities are adapter claims with reasons—not guesses based on architecture names.</p></div>
        <Button disabled={!connected} icon={<RefreshCw size={14} />} onClick={onRefresh}>Rescan roots</Button>
      </header>
      <ModelRootSettings connected={connected} modelLoaded={models.some((model) => model.lifecycle === "loaded")} onRefresh={onSynchronize} />
      {!models.length ? (
        <EmptyState description={connected ? "Check the effective configuration for a readable model root, then rescan. Model roots are never modified." : "Start the local backend before scanning configured roots."} icon={Database} title={connected ? "No model folders discovered" : "Backend offline"} />
      ) : (
        <>
          <div className="stat-row">
            <Stat label="Discovered" value={String(models.length)} />
            <Stat label="Loaded" value={String(models.filter((model) => model.lifecycle === "loaded").length)} />
            <Stat label="Fingerprinted" value={String(models.filter((model) => model.fingerprint).length)} />
          </div>
          <section aria-labelledby="registry-models" className="page-section">
            <div className="section-head"><h2 id="registry-models">Local models</h2><p>One model is resident at a time.</p></div>
            <div className="model-list">{models.map((model) => <ModelCard key={model.id} model={model} onSelectModel={onSelectModel} onToggleLoaded={onToggleLoaded} />)}</div>
          </section>
          <section aria-labelledby="registry-matrix" className="page-section">
            <div className="section-head">
              <div><h2 id="registry-matrix">Capability matrix</h2><p>Hover or focus a cell for its reason; expand a row to compare every model.</p></div>
              <ul aria-label="Legend" className="matrix-legend">
                {(Object.keys(stateLabels) as CapabilityState[]).map((state) => <li className={`capability-state ${state}`} key={state} title={stateDescriptions[state]}><StateIcon state={state} />{stateLabels[state]}</li>)}
              </ul>
            </div>
            <CapabilityMatrix models={models} />
          </section>
        </>
      )}
    </main>
  );
}
