import {
  ArrowRight,
  BarChart3,
  Boxes,
  Check,
  ChevronDown,
  Copy,
  Download,
  FileJson,
  Image,
  Plus,
  Search,
  Table2,
  Trash2,
  Upload,
  Video,
  X,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { EmbeddingInput, EmbeddingRun, EmbeddingVector, ModelSummary } from "../api/types";
import { capabilityReason, isUsable } from "../domain/capabilities";
import { downloadBlob, formatDuration, formatNumber } from "../utils/format";
import { Badge, Button, Callout, EmptyState, Field, IconButton, NumberInput, SegmentedControl, Select, Stat, Switch, Tabs } from "./ui";

type EmbeddingTab = "vectors" | "similarity" | "projection";

interface EmbeddingsWorkspaceProps {
  models: ModelSummary[];
  selectedModelId: string;
  connected: boolean;
  onSelectModel: (id: string) => void;
  onRuntimeStateChange: () => Promise<void>;
}

const resultTabs = [
  { id: "vectors" as const, label: "Vectors", icon: Table2 },
  { id: "similarity" as const, label: "Similarity", icon: Search },
  { id: "projection" as const, label: "Projection", icon: BarChart3 },
];

const dimensionPresets = ["64", "256", "1024", "2048"].map((value) => ({ value, label: value, title: `${value} output dimensions` }));

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

function npyBuffer(run: EmbeddingRun): ArrayBuffer {
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
  return buffer;
}

function VectorRow({ vector, run }: { vector: EmbeddingVector; run: EmbeddingRun }): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  return (
    <li className={`vector-row ${expanded ? "expanded" : ""}`}>
      <button aria-expanded={expanded} className="vector-summary" onClick={() => setExpanded((value) => !value)} type="button">
        <span className="vector-index">{run.vectors.findIndex((item) => item.inputId === vector.inputId) + 1}</span>
        <span className="vector-copy"><strong>{vectorLabel(vector, run)}</strong><span>{String(vector.dimensions)} dimensions · {vector.dtype}</span></span>
        <span className="vector-stats"><span>‖x‖ <strong>{formatNumber(vector.l2Norm, 5)}</strong></span><span>μ <strong>{formatNumber(vector.mean, 5)}</strong></span></span>
        <ChevronDown aria-hidden="true" className={expanded ? "rotated" : ""} size={15} />
      </button>
      {expanded && (
        <div className="vector-expanded">
          <div className="vector-toolbar">
            <span>Full vector · index:value</span>
            <Button icon={copied ? <Check size={13} /> : <Copy size={13} />} onClick={() => { void navigator.clipboard.writeText(JSON.stringify(vector.values)); setCopied(true); window.setTimeout(() => setCopied(false), 1000); }} size="sm" variant="ghost">{copied ? "Copied" : "Copy"}</Button>
          </div>
          <div className="vector-values">{vector.values.map((value, index) => <code key={index}><i>{String(index)}</i>{formatNumber(value, 7)}</code>)}</div>
        </div>
      )}
    </li>
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
    return (
      <main className="workspace">
        <EmptyState
          description={models.length ? "Discovered models do not expose embeddings through their current adapters." : connected ? "Scan a configured model root to discover local models." : "Start the local backend to discover models."}
          eyebrow="Embeddings workspace"
          headingLevel={1}
          icon={Boxes}
          size="page"
          title="No embedding-capable model"
        >
          {models[0] && <Callout tone="warning">{capabilityReason(models[0], "embeddings")}</Callout>}
        </EmptyState>
      </main>
    );
  }

  return (
    <main className="workspace embeddings-workspace">
      <header className="page-header">
        <div><h1>Embeddings</h1><p>Inspect exact local vectors, similarity, and model-provided projections.</p></div>
        <Field className="page-header-field" label="Embedding model">
          <Select onChange={(event) => onSelectModel(event.target.value)} value={selectedModel?.id ?? ""}>{embeddingModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</Select>
        </Field>
      </header>

      <div className="embeddings-layout">
        <section aria-labelledby="embedding-inputs-title" className="panel embedding-inputs">
          <header className="panel-head">
            <div><h2 id="embedding-inputs-title">Inputs</h2><p>{String(inputs.length)} items · batch order preserved</p></div>
            <Button icon={<Plus size={14} />} onClick={() => setInputs((current) => [...current, freshInput(current.length)])} size="sm">Add text</Button>
          </header>
          <ol className="embedding-input-list">
            {inputs.map((input, index) => (
              <li className="embedding-input" key={input.id}>
                <div className="embedding-input-head">
                  <span aria-hidden="true" className="embedding-input-number">{String(index + 1).padStart(2, "0")}</span>
                  <input aria-label={`Label for item ${String(index + 1)}`} className="embedding-label-input" onChange={(event) => updateInput(input.id, { label: event.target.value })} value={input.label ?? ""} />
                  <Badge>{input.kind}</Badge>
                  <IconButton disabled={inputs.length === 1} icon={<Trash2 size={14} />} label={`Remove item ${String(index + 1)}`} onClick={() => setInputs((current) => current.filter((item) => item.id !== input.id))} size="sm" title={inputs.length === 1 ? "At least one input is required" : "Remove this input"} />
                </div>
                <textarea aria-label={`Text for ${input.label ?? `item ${String(index + 1)}`}`} className="input textarea" onChange={(event) => updateInput(input.id, { text: event.target.value, kind: input.attachmentIds?.length ? "mixed" : "text" })} placeholder="Enter text to embed…" rows={3} value={input.text ?? ""} />
                {input.attachmentIds?.length ? (
                  <div className="attached-media"><Check aria-hidden="true" size={13} /> Local media attached <Button icon={<X size={12} />} onClick={() => updateInput(input.id, { attachmentIds: [], kind: "text" })} size="sm" variant="ghost">Remove</Button></div>
                ) : (
                  <div className="input-media-actions">
                    <Button disabled={!isUsable(selectedModel, "vision")} icon={<Image size={14} />} onClick={() => { setUploadTarget(input.id); fileInput.current?.click(); }} size="sm" title={capabilityReason(selectedModel, "vision")} variant="ghost">Image</Button>
                    <Button disabled={!isUsable(selectedModel, "video")} icon={<Video size={14} />} onClick={() => { setUploadTarget(input.id); fileInput.current?.click(); }} size="sm" title={capabilityReason(selectedModel, "video")} variant="ghost">Video</Button>
                  </div>
                )}
              </li>
            ))}
          </ol>
          <input accept="image/*,video/*,audio/*" className="visually-hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ""; }} ref={fileInput} tabIndex={-1} type="file" />
          <div className="embedding-controls">
            <Field label="Output dimensions">
              <NumberInput max={16384} min={1} onValueChange={setDimensions} value={dimensions} />
            </Field>
            <SegmentedControl label="Dimension presets" onChange={(value) => setDimensions(Number(value))} options={dimensionPresets} value={dimensionPresets.some((preset) => preset.value === String(dimensions)) ? String(dimensions) : null} />
            <Switch checked={normalize} label="Re-normalize output" onChange={(event) => setNormalize(event.target.checked)} />
            <Button className="run-embedding" disabled={!connected || !validInputs.length} icon={<ArrowRight size={15} />} loading={loading} onClick={() => void execute()} variant="primary">{loading ? "Running locally…" : `Embed ${String(validInputs.length)} item${validInputs.length === 1 ? "" : "s"}`}</Button>
          </div>
          <div className="panel-notes">
            <Callout icon={Upload}>Media is accepted only when the selected adapter reports native support. Audio remains disabled when unsupported.</Callout>
            {error && <Callout tone="danger">{error}</Callout>}
          </div>
        </section>

        <section aria-label="Embedding results" className="panel embedding-results">
          <Tabs className="embedding-tabs" idPrefix="embedding-results" items={resultTabs} label="Embedding result views" onChange={setActiveTab} value={activeTab} />
          <div aria-labelledby={`embedding-results-tab-${activeTab}`} className="embedding-results-body" id="embedding-results-panel" role="tabpanel">
            {!run ? <EmptyState description="Results will appear exactly as returned by the local model. Nothing here is demonstration data." icon={Boxes} title="No embedding run yet" /> : (
              <>
                <div className="embedding-summary">
                  <Stat label="Run" title={run.id} value={<code>{run.id}</code>} />
                  <Stat label="Shape" value={`${String(run.vectors.length)} × ${String(run.outputDimensions)}`} />
                  <Stat label="Pooling" value={run.pooling ?? "Not reported"} />
                  <Stat label="Total" value={formatDuration(run.totalMs)} />
                  <Stat label="Normalized" value={run.normalized ? "Yes" : "No"} />
                </div>
                <div className="export-row">
                  <span>{run.jointSpace === undefined ? "Joint space not reported" : run.jointSpace ? "Joint multimodal space" : "Modality-specific space"}</span>
                  <div>
                    <Button icon={<FileJson size={13} />} onClick={() => downloadBlob(`embeddings-${run.id}.json`, JSON.stringify(run, null, 2), "application/json")} size="sm" variant="ghost">JSON</Button>
                    <Button icon={<Download size={13} />} onClick={exportCsv} size="sm" variant="ghost">CSV</Button>
                    <Button icon={<Download size={13} />} onClick={() => downloadBlob(`embeddings-${run.id}.npy`, npyBuffer(run), "application/octet-stream")} size="sm" variant="ghost">NumPy</Button>
                  </div>
                </div>
                {activeTab === "vectors" && <ol className="vector-list">{run.vectors.map((vector) => <VectorRow key={vector.inputId} run={run} vector={vector} />)}</ol>}
                {activeTab === "similarity" && (
                  <div className="similarity-wrap">
                    <div className="similarity-matrix" style={{ gridTemplateColumns: `minmax(96px, 1fr) repeat(${String(run.vectors.length)}, minmax(64px, 1fr))` }}>
                      <div />
                      {run.vectors.map((vector, index) => <strong key={vector.inputId} title={vectorLabel(vector, run)}>{String(index + 1)}<small>{vectorLabel(vector, run)}</small></strong>)}
                      {run.vectors.flatMap((row, rowIndex) => [
                        <strong key={`row-${row.inputId}`} title={vectorLabel(row, run)}>{String(rowIndex + 1)}<small>{vectorLabel(row, run)}</small></strong>,
                        ...run.vectors.map((column, columnIndex) => {
                          const value = similarity[rowIndex]?.[columnIndex];
                          return <span key={`${row.inputId}-${column.inputId}`} style={{ "--similarity": String(Math.max(0, value ?? 0)) } as React.CSSProperties}>{formatNumber(value, 4)}</span>;
                        }),
                      ])}
                    </div>
                    <p className="footnote">Cosine similarity is computed from returned vectors when the backend does not provide a matrix.</p>
                  </div>
                )}
                {activeTab === "projection" && (run.projection?.length ? (
                  <div className="projection-view">
                    <svg viewBox="0 0 640 400">{run.projection.map((point, index) => <g key={point.inputId} transform={`translate(${String(40 + point.x * 560)} ${String(360 - point.y * 320)})`}><circle r="7" /><text x="12" y="5">{run.inputs.find((input) => input.id === point.inputId)?.label ?? String(index + 1)}</text></g>)}</svg>
                    <p className="footnote">Model/backend-provided projection; axes do not preserve original feature meaning.</p>
                  </div>
                ) : <EmptyState description="PCA or UMAP coordinates must be explicitly computed and labeled by the backend. The UI will not invent coordinates." icon={BarChart3} title="No projection returned" />)}
              </>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
