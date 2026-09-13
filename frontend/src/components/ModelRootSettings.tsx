import { FolderCog, LoaderCircle, Save } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import {
  getModelRootSettings,
  updateModelRootSettings,
} from "../api/modelRoots";

interface ModelRootSettingsProps {
  connected: boolean;
  modelLoaded?: boolean;
  onRefresh: () => void;
}

export function ModelRootSettings({ connected, modelLoaded = false, onRefresh }: ModelRootSettingsProps): React.ReactNode {
  const [paths, setPaths] = useState("");
  const [writable, setWritable] = useState(false);
  const [containerized, setContainerized] = useState(false);
  const [reason, setReason] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!connected) {
      setWritable(false);
      setReason("Start the backend before changing model directories.");
      return;
    }
    let active = true;
    setLoading(true);
    setError(null);
    void getModelRootSettings()
      .then((settings) => {
        if (!active) return;
        setPaths(settings.modelRoots.join("\n"));
        setWritable(settings.writable);
        setContainerized(settings.containerized);
        setReason(settings.reason);
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Could not read model directory settings.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [connected]);

  const save = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const roots = paths.split(/\r?\n/u).map((path) => path.trim()).filter(Boolean);
    if (!roots.length) {
      setError("Enter at least one model directory.");
      return;
    }
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const settings = await updateModelRootSettings(roots);
      setPaths(settings.modelRoots.join("\n"));
      setWritable(settings.writable);
      setReason(settings.reason);
      setMessage("Saved to the user-local configuration and rescanned.");
      onRefresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save model directories.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section aria-labelledby="model-root-settings-title" className="model-root-settings">
      <div className="model-root-settings-copy">
        <FolderCog size={18} />
        <div>
          <h2 id="model-root-settings-title">Local model directories</h2>
          <p>Saved in the existing user-local configuration file. Model files stay read-only.</p>
        </div>
      </div>
      <form onSubmit={(event) => void save(event)}>
        <label htmlFor="model-root-paths" title="Directories are scanned for compatible local model checkpoints.">Backend-visible absolute paths, one per line</label>
        <textarea
          disabled={!connected || loading || !writable || modelLoaded}
          id="model-root-paths"
          onChange={(event) => setPaths(event.target.value)}
          placeholder={containerized ? "/models" : "C:/models"}
          rows={Math.max(1, Math.min(4, paths.split(/\r?\n/u).length))}
          spellCheck={false}
          title={containerized ? "Use a path mounted inside the backend container, normally /models." : "Use an absolute directory path readable by the backend process."}
          value={paths}
        />
        <button className="button primary" disabled={!connected || loading || saving || !writable || modelLoaded} title={modelLoaded ? "Unload the active model before changing model directories" : "Persist these paths in the user-local config file and rescan models"} type="submit">
          {saving ? <LoaderCircle className="spin" size={14} /> : <Save size={14} />}
          {saving ? "Saving…" : "Save & rescan"}
        </button>
      </form>
      <p className="model-root-hint">
        {containerized
          ? "Docker users must enter the mounted container path, normally /models—not a Windows host path."
          : "The path must exist and be readable by the backend process."}
      </p>
      {reason && <p className="model-root-warning" role="status">{reason}</p>}
      {modelLoaded && <p className="model-root-warning" role="status">Unload the active model before changing model directories.</p>}
      {error && <p className="model-root-error" role="alert">{error}</p>}
      {message && !error && <p className="model-root-success" role="status">{message}</p>}
    </section>
  );
}
