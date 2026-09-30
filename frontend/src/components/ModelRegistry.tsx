import {
  AlertTriangle,
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
import type { CapabilityKey, CapabilityState, LoadOptions, MemoryLedger, ModelInspection, ModelSummary, ResidentModel } from "../api/types";
import { capabilityKeys, capabilityLabel, capabilityOf } from "../domain/capabilities";
import { residentColor, residentsOf, residentSummary } from "../domain/residency";
import { formatBytes, formatNumber, shortFingerprint } from "../utils/format";
import { LoadOptionsFields } from "./LoadOptionsFields";
import { MemoryLedgerCard } from "./MemoryLedger";
import { ModelRootSettings } from "./ModelRootSettings";
import { Badge, type BadgeTone, Button, Callout, Card, EmptyState, IconButton, Stat } from "./ui";

interface ModelRegistryProps {
  models: ModelSummary[];
  residents: ResidentModel[];
  memory: MemoryLedger | null;
  maxLoadedModels: number | null;
  connected: boolean;
  loadOptionsFor: (modelId: string) => LoadOptions;
  onLoadOptionsChange: (modelId: string, options: LoadOptions) => void;
  onLoad: (model: ModelSummary, options: LoadOptions) => void;
  onUnloadResident: (resident: ResidentModel) => void;
  onUnloadAll: () => void;
  onRefresh: () => void;
  onSynchronize: () => void;
  onSelectModel: (id: string) => void;
}

const matrixKeys: CapabilityKey[] = ["text_generation", "embeddings", "vision", "audio", "video", "reasoning_segments", "moe_routing", "raw_logits", "prompt_scoring", "streaming", "cpu", "cuda", "cpu_offload"];

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

const placementLabels: Record<NonNullable<ResidentModel["placement"]>, string | null> = { gpu: "GPU", cpu: "CPU", offload: null };

function ResidentRow({ model, resident, color, busy, onUnload }: { model: ModelSummary; resident: ResidentModel; color: number | null; busy: boolean; onUnload: (resident: ResidentModel) => void }): React.ReactNode {
  const summary = [residentSummary(resident), resident.placement ? placementLabels[resident.placement] : null].filter(Boolean).join(" · ");
  const usage: Array<[string, string]> = [
    ...(resident.gpuBytes !== null ? [["GPU", formatBytes(resident.gpuBytes)] as [string, string]] : []),
    ...(resident.cpuBytes !== null ? [["RAM", formatBytes(resident.cpuBytes)] as [string, string]] : []),
    ...(resident.kvReserveBytes ? [["KV reserve", formatBytes(resident.kvReserveBytes)] as [string, string]] : []),
    ...(resident.loadSeconds !== null ? [["Load", `${formatNumber(resident.loadSeconds, 1)} s`] as [string, string]] : []),
  ];
  return (
    <li className="resident-row">
      <div className="resident-badges">
        {color !== null && <i aria-hidden="true" className="resident-swatch" style={{ "--swatch": `var(--viz-${String(color)})` } as React.CSSProperties} title="Colour of this copy in the GPU memory bar" />}
        <Badge tone="accent">{summary}</Badge>
        {resident.placement === "offload" && <Badge icon={<AlertTriangle aria-hidden="true" size={11} />} title="Layers that did not fit in VRAM run from system RAM, which is much slower." tone="warning">Offloaded to system RAM</Badge>}
        {resident.strictVram && resident.placement === "gpu" && <Badge title="Loaded with Strict VRAM: this copy never spills into system RAM.">Strict VRAM</Badge>}
        {resident.quantization && resident.quantization !== "none" && <Badge>{resident.quantization}</Badge>}
        {resident.inUse && <Badge tone="info">In use</Badge>}
      </div>
      {usage.length > 0 && <dl className="resident-usage">{usage.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>}
      <Button
        aria-label={`Unload ${model.name} on ${summary}`}
        disabled={resident.inUse}
        icon={<Square size={11} />}
        loading={busy}
        onClick={() => onUnload(resident)}
        size="sm"
        title={resident.inUse ? "In use by a running job; unload it when the job finishes" : "Unload this copy and free its memory"}
      >Unload</Button>
    </li>
  );
}

interface ModelCardProps {
  model: ModelSummary;
  residents: ResidentModel[];
  colorOf: (modelKey: string) => number | null;
  connected: boolean;
  loadOptions: LoadOptions;
  onLoadOptionsChange: (modelId: string, options: LoadOptions) => void;
  onLoad: (model: ModelSummary, options: LoadOptions) => void;
  onUnloadResident: (resident: ResidentModel) => void;
  onSelectModel: (id: string) => void;
}

function ModelCard({ model, residents, colorOf, connected, loadOptions, onLoadOptionsChange, onLoad, onUnloadResident, onSelectModel }: ModelCardProps): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const [inspection, setInspection] = useState<ModelInspection | null>(null);
  const [inspectionOpen, setInspectionOpen] = useState(false);
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  const busy = model.lifecycle === "loading" || model.lifecycle === "unloading";
  const loadable = model.task !== "unknown";
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
            {residents.length > 0 && <Badge tone="accent">{residents.length === 1 ? "Resident" : `${String(residents.length)} resident copies`}</Badge>}
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
          <IconButton aria-controls={detailId} aria-expanded={expanded} icon={<ChevronDown className={expanded ? "rotated" : ""} size={16} />} label={expanded ? `Hide details for ${model.name}` : `Show details for ${model.name}`} onClick={() => setExpanded((value) => !value)} size="sm" title={expanded ? "Hide details" : "Show details"} />
        </div>
      </div>
      {loadable && (
        <div aria-label={`Load options for ${model.name}`} className="model-load-panel" role="group">
          <LoadOptionsFields disabled={!connected || busy} model={model} onChange={(options) => onLoadOptionsChange(model.id, options)} size="sm" value={loadOptions} />
          <Button
            disabled={!connected || model.lifecycle === "unloading"}
            icon={<Play size={13} />}
            loading={model.lifecycle === "loading"}
            onClick={() => { onSelectModel(model.id); onLoad(model, loadOptions); }}
            size="sm"
            title={connected ? `Load ${model.name} with these options; a resident copy with the same options is reused` : "Start the local backend first"}
            variant={residents.length ? "secondary" : "primary"}
          >{model.lifecycle === "loading" ? "Loading…" : "Load"}</Button>
        </div>
      )}
      {residents.length > 0 && (
        <ul aria-label={`Resident copies of ${model.name}`} className="resident-list">
          {residents.map((resident) => <ResidentRow busy={model.lifecycle === "unloading"} color={colorOf(resident.modelKey)} key={resident.modelKey} model={model} onUnload={onUnloadResident} resident={resident} />)}
        </ul>
      )}
      {expanded && (
        <div className="model-card-detail" id={detailId}>
          <dl className="kv-grid">
            <div><dt>Fingerprint</dt><dd className="mono" title={model.fingerprint ?? undefined}>{shortFingerprint(model.fingerprint)}</dd></div>
            <div><dt>Lifecycle</dt><dd>{model.lifecycle}{residents.length ? ` · ${String(residents.length)} resident` : ""}</dd></div>
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

export function ModelRegistry({ models, residents, memory, maxLoadedModels, connected, loadOptionsFor, onLoadOptionsChange, onLoad, onUnloadResident, onUnloadAll, onRefresh, onSynchronize, onSelectModel }: ModelRegistryProps): React.ReactNode {
  return (
    <main className="workspace model-registry">
      <header className="page-header">
        <div><h1>Model registry</h1><p>Read-only discovery. Capabilities are adapter claims with reasons—not guesses based on architecture names.</p></div>
        <Button disabled={!connected} icon={<RefreshCw size={14} />} onClick={onRefresh}>Rescan roots</Button>
      </header>
      {connected && <MemoryLedgerCard connected={connected} maxLoadedModels={maxLoadedModels} memory={memory} models={models} onUnloadAll={onUnloadAll} residents={residents} />}
      <ModelRootSettings connected={connected} modelLoaded={residents.length > 0} onRefresh={onSynchronize} />
      {!models.length ? (
        <EmptyState description={connected ? "Check the effective configuration for a readable model root, then rescan. Model roots are never modified." : "Start the local backend before scanning configured roots."} icon={Database} title={connected ? "No model folders discovered" : "Backend offline"} />
      ) : (
        <>
          <div className="stat-row">
            <Stat label="Discovered" value={String(models.length)} />
            <Stat caption={maxLoadedModels ? `of ${String(maxLoadedModels)} allowed` : undefined} label="Resident" value={String(residents.length)} />
            <Stat label="Fingerprinted" value={String(models.filter((model) => model.fingerprint).length)} />
          </div>
          <section aria-labelledby="registry-models" className="page-section">
            <div className="section-head"><div><h2 id="registry-models">Local models</h2><p>Several models can stay resident. A load that needs room unloads the least recently used idle model first.</p></div></div>
            <div className="model-list">
              {models.map((model) => (
                <ModelCard
                  colorOf={(key) => residentColor(residents, key)}
                  connected={connected}
                  key={model.id}
                  loadOptions={loadOptionsFor(model.id)}
                  model={model}
                  onLoad={onLoad}
                  onLoadOptionsChange={onLoadOptionsChange}
                  onSelectModel={onSelectModel}
                  onUnloadResident={onUnloadResident}
                  residents={residentsOf(residents, model.id)}
                />
              ))}
            </div>
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
