import { HardDriveDownload } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import { ApiError } from "../api/client";
import { getModelRootSettings } from "../api/modelRoots";
import type { FlushResult, ModelSummary, ResidentModel } from "../api/types";
import { defaultFlushFolderName, flushFolderNameError, quantizationLabel } from "../domain/residency";
import { formatBytes } from "../utils/format";
import { Button, Callout, Drawer, Field, Input, Select } from "./ui";

interface FlushDialogProps {
  resident: ResidentModel;
  model: ModelSummary;
  onClose: () => void;
  onFlush: (resident: ResidentModel, target: { targetRootIndex: number; folderName: string }) => Promise<FlushResult>;
}

/** Chooses where Flush to storage writes a quantized resident: a configured root and a new folder name. */
export function FlushDialog({ resident, model, onClose, onFlush }: FlushDialogProps): React.ReactNode {
  const [roots, setRoots] = useState<string[] | null>(null);
  const [rootIndex, setRootIndex] = useState(model.rootIndex ?? 0);
  const [folderName, setFolderName] = useState(() => defaultFlushFolderName(model.name, resident.quantization));
  const [writing, setWriting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nameError = flushFolderNameError(folderName);
  const label = quantizationLabel(resident.quantization) ?? "quantized";
  const formId = `flush-form-${resident.modelKey}`;

  useEffect(() => {
    let active = true;
    void getModelRootSettings()
      .then((settings) => { if (active) setRoots(settings.modelRoots); })
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Could not read the configured model roots."); });
    return () => {
      active = false;
    };
  }, []);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (nameError) return;
    setWriting(true);
    setError(null);
    try {
      await onFlush(resident, { targetRootIndex: rootIndex, folderName });
      onClose();
    } catch (cause) {
      const hint = cause instanceof ApiError && cause.hint ? ` ${cause.hint}` : "";
      setError(cause instanceof Error ? `${cause.message}${hint}` : "Flush to storage failed.");
      setWriting(false);
    }
  };

  return (
    <Drawer
      closeLabel="Close flush dialog"
      description={`Saves the ${label} copy of ${model.name} exactly as it is held in GPU memory. The source folder is never modified.`}
      footer={(
        <>
          <Button disabled={writing} onClick={onClose}>Cancel</Button>
          <Button disabled={nameError !== null || roots === null} form={formId} icon={<HardDriveDownload size={14} />} loading={writing} type="submit" variant="primary">
            {writing ? "Writing…" : "Flush to storage"}
          </Button>
        </>
      )}
      label="Flush to storage"
      onClose={writing ? () => undefined : onClose}
      open
      title="Flush to storage"
    >
      <form className="flush-form" id={formId} onSubmit={(event) => void submit(event)}>
        <Field hint="New folders are written only inside a configured model root." label="Model root">
          <Select disabled={writing || roots === null} onChange={(event) => setRootIndex(Number(event.target.value))} value={String(rootIndex)}>
            {roots === null
              ? <option value={String(rootIndex)}>Loading roots…</option>
              : roots.map((root, index) => <option key={root} value={String(index)}>{root}</option>)}
          </Select>
        </Field>
        <Field error={nameError ?? undefined} hint={nameError ? undefined : "A new folder; an existing one is never overwritten."} label="Folder name">
          <Input autoComplete="off" disabled={writing} onChange={(event) => setFolderName(event.target.value)} spellCheck={false} value={folderName} />
        </Field>
        <dl className="kv-grid flush-facts">
          <div><dt>Format</dt><dd>{label} · SafeTensors</dd></div>
          <div><dt>Estimated size</dt><dd title="The measured GPU memory of this copy; the files are about the same size">{resident.gpuBytes ? `About ${formatBytes(resident.gpuBytes)}` : "Unknown"}</dd></div>
        </dl>
        {writing && <p className="muted" role="status">Writing {label} weights to disk. Large models take a minute.</p>}
        {error && <Callout title="Nothing was saved" tone="danger">{error}</Callout>}
      </form>
    </Drawer>
  );
}
