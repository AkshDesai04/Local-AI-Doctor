import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Boxes,
  Check,
  ChevronDown,
  Copy,
  Download,
  FileJson,
  Image,
  LoaderCircle,
  Plus,
  Search,
  Table2,
  Trash2,
  Upload,
  Video,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { EmbeddingInput, EmbeddingRun, EmbeddingVector, ModelSummary } from "../api/types";
import { capabilityReason, isUsable } from "../domain/capabilities";
import { downloadBlob, formatDuration, formatNumber } from "../utils/format";

type EmbeddingTab = "vectors" | "similarity" | "projection";

interface EmbeddingsWorkspaceProps {
  models: ModelSummary[];
  selectedModelId: string;
  connected: boolean;
  onSelectModel: (id: string) => void;
  onRuntimeStateChange: () => Promise<void>;
}

function freshInput(index: number): EmbeddingInput {
  return { id: `input-${String(Date.now())}-${String(index)}`, kind: "text", text: "", label: `Item ${String(index + 1)}` };
}

function cosine(left: number[], right: number[]): number {
  const length = Math.min(left.length, right.length);
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm) || 1);
}

function similarityFor(run: EmbeddingRun): number[][] {
  if (run.similarityMatrix) return run.similarityMatrix;
  return run.vectors.map((left) => run.vectors.map((right) => cosine(left.values, right.values)));
}

function vectorLabel(vector: EmbeddingVector, run: EmbeddingRun): string {
  return run.inputs.find((input) => input.id === vector.inputId)?.label ?? vector.inputId;
}

function npyBlob(run: EmbeddingRun): Blob {
  const rows = run.vectors.length;
  const columns = run.outputDimensions;
  const dictionary = `{'descr': '<f4', 'fortran_order': False, 'shape': (${String(rows)}, ${String(columns)}), }`;
  const preambleLength = 10;
  const padding = (16 - ((preambleLength + dictionary.length + 1) % 16)) % 16;
  const header = `${dictionary}${" ".repeat(padding)}\n`;
  const buffer = new ArrayBuffer(preambleLength + header.length + rows * columns * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00], 0);
  bytes[8] = header.length & 0xff;
  bytes[9] = (header.length >> 8) & 0xff;
  bytes.set(new TextEncoder().encode(header), preambleLength);
  const data = new DataView(buffer, preambleLength + header.length);
  let offset = 0;
  for (const vector of run.vectors) {
    for (let column = 0; column < columns; column += 1) {
      data.setFloat32(offset, vector.values[column] ?? Number.NaN, true);
      offset += 4;
    }
  }
  return new Blob([buffer], { type: "application/octet-stream" });
}

function VectorRow({ vector, run }: { vector: EmbeddingVector; run: EmbeddingRun }): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  return (
    <article className="vector-card">
      <button className="vector-summary" onClick={() => setExpanded((value) => !value)} type="button">
        <div><span className="vector-index">{run.vectors.findIndex((item) => item.inputId === vector.inputId) + 1}</span><div><strong>{vectorLabel(vector, run)}</strong><span>{String(vector.dimensions)} dimensions · {vector.dtype}</span></div></div>
        <div className="vector-stats"><span>‖x‖ <strong>{formatNumber(vector.l2Norm, 5)}</strong></span><span>μ <strong>{formatNumber(vector.mean, 5)}</strong></span><ChevronDown className={expanded ? "rotated" : ""} size={15} /></div>
      </button>
      {expanded && (
        <div className="vector-expanded">
          <div className="vector-toolbar"><span>Full vector · index:value</span><button onClick={() => { void navigator.clipboard.writeText(JSON.stringify(vector.values)); setCopied(true); window.setTimeout(() => setCopied(false), 1000); }} type="button">{copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy"}</button></div>
          <div className="vector-values">{vector.values.map((value, index) => <code key={index}><i>{String(index)}</i>{formatNumber(value, 7)}</code>)}</div>
        </div>
      )}
    </article>
  );
}

export function EmbeddingsWorkspace({ models, selectedModelId, connected, onSelectModel, onRuntimeStateChange }: EmbeddingsWorkspaceProps): React.ReactNode {
  const embeddingModels = models.filter((model) => isUsable(model, "embeddings"));
  const selectedModel = embeddingModels.find((model) => model.id === selectedModelId) ?? embeddingModels[0] ?? null;
  const [inputs, setInputs] = useState<EmbeddingInput[]>([freshInput(0), freshInput(1)]);
  const [dimensions, setDimensions] = useState(2048);
  const [normalize, setNormalize] = useState(true);
  const [run, setRun] = useState<EmbeddingRun | null>(null);
  const [activeTab, setActiveTab] = useState<EmbeddingTab>("vectors");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadTarget, setUploadTarget] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const validInputs = inputs.filter((input) => Boolean(input.text?.trim()) || Boolean(input.attachmentIds?.length));
  const similarity = useMemo(() => run ? similarityFor(run) : [], [run]);

  const updateInput = (id: string, changes: Partial<EmbeddingInput>): void => setInputs((current) => current.map((input) => input.id === id ? { ...input, ...changes } : input));
  const execute = async (): Promise<void> => {
    if (!selectedModel || !validInputs.length) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.embeddings({ modelId: selectedModel.id, inputs: validInputs, dimensions, normalize });
      setRun(result);
      setActiveTab("vectors");
      await onRuntimeStateChange();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : cause instanceof Error ? cause.message : "Embedding request failed.");
    } finally {
      setLoading(false);
    }
  };

  const upload = async (file: File): Promise<void> => {
    if (!selectedModel || !uploadTarget) return;
    setLoading(true);
    try {
      const attachment = await api.upload(file, selectedModel.id);
      const kind = attachment.kind === "image" || attachment.kind === "video" || attachment.kind === "audio" ? attachment.kind : "mixed";
      updateInput(uploadTarget, { kind, attachmentIds: [attachment.id], label: file.name });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
    } finally {
      setLoading(false);
      setUploadTarget(null);
    }
  };

  const exportCsv = (): void => {
    if (!run) return;
    const header = ["input_id", ...Array.from({ length: run.outputDimensions }, (_, index) => `d${String(index)}`)].join(",");
    const rows = run.vectors.map((vector) => [JSON.stringify(vector.inputId), ...vector.values.map(String)].join(","));
    downloadBlob(`embeddings-${run.id}.csv`, [header, ...rows].join("\n"), "text/csv");
  };

  if (!embeddingModels.length) {
    return <main className="workspace-empty"><div className="empty-orbit"><Boxes size={23} /></div><span className="eyebrow">Embeddings workspace</span><h1>No embedding-capable model</h1><p>{models.length ? "Discovered models do not expose embeddings through their current adapters." : connected ? "Scan a configured model root to discover local models." : "Start the local backend to discover models."}</p>{models[0] && <div className="inline-warning"><AlertTriangle size={15} />{capabilityReason(models[0], "embeddings")}</div>}</main>;
  }

  return (
    <main className="embeddings-workspace">
      <header className="workspace-titlebar">
        <div><span className="eyebrow">Vector laboratory</span><h1>Embeddings</h1><p>Inspect exact local vectors, similarity, and model-provided projections.</p></div>
        <label className="workspace-model-select"><span>Embedding model</span><div><select onChange={(event) => onSelectModel(event.target.value)} value={selectedModel?.id ?? ""}>{embeddingModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select><ChevronDown size={14} /></div></label>
      </header>

      <div className="embeddings-layout">
        <section className="embedding-input-panel">
          <div className="panel-title-row"><div><h2>Inputs</h2><span>{String(inputs.length)} items · batch order preserved</span></div><button className="button secondary compact" onClick={() => setInputs((current) => [...current, freshInput(current.length)])} type="button"><Plus size={14} /> Add text</button></div>
          <div className="embedding-input-list">
            {inputs.map((input, index) => (
              <article className="embedding-input-card" key={input.id}>
                <div className="embedding-input-number">{String(index + 1).padStart(2, "0")}</div>
                <div className="embedding-input-main">
                  <div className="embedding-input-heading"><input aria-label={`Label for item ${String(index + 1)}`} onChange={(event) => updateInput(input.id, { label: event.target.value })} value={input.label ?? ""} /><span>{input.kind}</span></div>
                  <textarea aria-label={`Text for ${input.label ?? `item ${String(index + 1)}`}`} onChange={(event) => updateInput(input.id, { text: event.target.value, kind: input.attachmentIds?.length ? "mixed" : "text" })} placeholder="Enter text to embed…" rows={3} value={input.text ?? ""} />
                  {input.attachmentIds?.length ? <div className="attached-media"><Check size={13} /> Local media attached <button onClick={() => updateInput(input.id, { attachmentIds: [], kind: "text" })} type="button">Remove</button></div> : <div className="input-media-actions"><button disabled={!isUsable(selectedModel, "vision")} onClick={() => { setUploadTarget(input.id); fileInput.current?.click(); }} title={capabilityReason(selectedModel, "vision")} type="button"><Image size={14} /> Image</button><button disabled={!isUsable(selectedModel, "video")} onClick={() => { setUploadTarget(input.id); fileInput.current?.click(); }} title={capabilityReason(selectedModel, "video")} type="button"><Video size={14} /> Video</button></div>}
                </div>
                <button aria-label={`Remove item ${String(index + 1)}`} className="icon-button remove-input" disabled={inputs.length === 1} onClick={() => setInputs((current) => current.filter((item) => item.id !== input.id))} type="button"><Trash2 size={15} /></button>
              </article>
            ))}
          </div>
          <input accept="image/*,video/*,audio/*" className="visually-hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ""; }} ref={fileInput} type="file" />
          <div className="embedding-controls">
            <label><span>Output dimensions</span><input max={16384} min={1} onChange={(event) => setDimensions(Number(event.target.value))} type="number" value={dimensions} /></label>
            <div className="dimension-presets">{[64, 256, 1024, 2048].map((value) => <button className={dimensions === value ? "selected" : ""} key={value} onClick={() => setDimensions(value)} type="button">{String(value)}</button>)}</div>
            <label className="compact-switch"><input checked={normalize} onChange={(event) => setNormalize(event.target.checked)} type="checkbox" /><span>Re-normalize output</span></label>
            <button className="button primary run-embedding" disabled={!connected || loading || !validInputs.length} onClick={() => void execute()} type="button">{loading ? <LoaderCircle className="spin" size={15} /> : <ArrowRight size={15} />} {loading ? "Running locally…" : `Embed ${String(validInputs.length)} item${validInputs.length === 1 ? "" : "s"}`}</button>
          </div>
          <div className="metric-note"><Upload size={14} /><span>Media is accepted only when the selected adapter reports native support. Audio remains disabled when unsupported.</span></div>
          {error && <div className="inline-warning"><AlertTriangle size={15} /><span>{error}</span></div>}
        </section>

        <section className="embedding-results-panel">
          <div className="embedding-result-tabs">
            <button className={activeTab === "vectors" ? "selected" : ""} onClick={() => setActiveTab("vectors")} type="button"><Table2 size={14} /> Vectors</button>
            <button className={activeTab === "similarity" ? "selected" : ""} onClick={() => setActiveTab("similarity")} type="button"><Search size={14} /> Similarity</button>
            <button className={activeTab === "projection" ? "selected" : ""} onClick={() => setActiveTab("projection")} type="button"><BarChart3 size={14} /> Projection</button>
          </div>
          {!run ? <div className="embedding-results-empty"><div><Boxes size={20} /></div><strong>No embedding run yet</strong><p>Results will appear exactly as returned by the local model. Nothing here is demonstration data.</p></div> : (
            <>
              <div className="embedding-run-summary"><div><span>Run</span><code>{run.id}</code></div><div><span>Shape</span><strong>{String(run.vectors.length)} × {String(run.outputDimensions)}</strong></div><div><span>Pooling</span><strong>{run.pooling ?? "Not reported"}</strong></div><div><span>Total</span><strong>{formatDuration(run.totalMs)}</strong></div><div><span>Normalized</span><strong>{run.normalized ? "Yes" : "No"}</strong></div></div>
              <div className="export-row"><span>{run.jointSpace === undefined ? "Joint space not reported" : run.jointSpace ? "Joint multimodal space" : "Modality-specific space"}</span><div><button onClick={() => downloadBlob(`embeddings-${run.id}.json`, JSON.stringify(run, null, 2), "application/json")} type="button"><FileJson size={13} /> JSON</button><button onClick={exportCsv} type="button"><Download size={13} /> CSV</button><button onClick={() => { const blob = npyBlob(run); const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `embeddings-${run.id}.npy`; anchor.click(); URL.revokeObjectURL(url); }} type="button"><Download size={13} /> NumPy</button></div></div>
              {activeTab === "vectors" && <div className="vector-list">{run.vectors.map((vector) => <VectorRow key={vector.inputId} run={run} vector={vector} />)}</div>}
              {activeTab === "similarity" && <div className="similarity-wrap"><div className="similarity-matrix" style={{ gridTemplateColumns: `minmax(90px, 1fr) repeat(${String(run.vectors.length)}, minmax(64px, 1fr))` }}><div /><>{run.vectors.map((vector, index) => <strong key={vector.inputId}>{String(index + 1)}<small>{vectorLabel(vector, run)}</small></strong>)}</>{run.vectors.flatMap((row, rowIndex) => [<strong key={`row-${row.inputId}`}>{String(rowIndex + 1)}<small>{vectorLabel(row, run)}</small></strong>, ...run.vectors.map((column, columnIndex) => { const value = similarity[rowIndex]?.[columnIndex]; return <span key={`${row.inputId}-${column.inputId}`} style={{ "--similarity": String(Math.max(0, value ?? 0)) } as React.CSSProperties}>{formatNumber(value, 4)}</span>; })])}</div><p className="footnote">Cosine similarity is computed from returned vectors when the backend does not provide a matrix.</p></div>}
              {activeTab === "projection" && (run.projection?.length ? <div className="projection-view"><svg viewBox="0 0 640 400">{run.projection.map((point, index) => <g key={point.inputId} transform={`translate(${String(40 + point.x * 560)} ${String(360 - point.y * 320)})`}><circle r="7" /><text x="11" y="4">{run.inputs.find((input) => input.id === point.inputId)?.label ?? String(index + 1)}</text></g>)}</svg><p className="footnote">Model/backend-provided projection; axes do not preserve original feature meaning.</p></div> : <div className="embedding-results-empty"><div><BarChart3 size={19} /></div><strong>No projection returned</strong><p>PCA or UMAP coordinates must be explicitly computed and labeled by the backend. The UI will not invent coordinates.</p></div>)}
            </>
          )}
        </section>
      </div>
    </main>
  );
}
