import { FolderCog, Save } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import {
  getModelRootSettings,
  updateModelRootSettings,
} from "../api/modelRoots";
import { Button, Callout, Card, Field, Textarea } from "./ui";

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
    <Card
      aria-label="Local model directories"
      className="model-root-settings"
      description="Saved in the existing user-local configuration file. Model files stay read-only."
      icon={FolderCog}
      title="Local model directories"
      titleAs="h2"
    >
      <form className="model-root-form" onSubmit={(event) => void save(event)}>
        <Field
          hint={containerized
            ? "Docker users must enter the mounted container path, normally /models—not a Windows host path."
            : "The path must exist and be readable by the backend process."}
          id="model-root-paths"
          label="Backend-visible absolute paths, one per line"
        >
          <Textarea
            className="mono"
            disabled={!connected || loading || !writable || modelLoaded}
            onChange={(event) => setPaths(event.target.value)}
            placeholder={containerized ? "/models" : "C:/models"}
            rows={Math.max(1, Math.min(4, paths.split(/\r?\n/u).length))}
            spellCheck={false}
            title={containerized ? "Use a path mounted inside the backend container, normally /models." : "Use an absolute directory path readable by the backend process."}
            value={paths}
          />
        </Field>
        <Button disabled={!connected || loading || !writable || modelLoaded} icon={<Save size={14} />} loading={saving} title={modelLoaded ? "Unload the active model before changing model directories" : "Persist these paths in the user-local config file and rescan models"} type="submit" variant="primary">
          {saving ? "Saving…" : "Save & rescan"}
        </Button>
      </form>
      {(reason || modelLoaded || error || (message && !error)) && (
        <div className="model-root-messages">
          {reason && <Callout role="status" tone="warning">{reason}</Callout>}
          {modelLoaded && <Callout role="status" tone="warning">Unload the active model before changing model directories.</Callout>}
          {error && <Callout tone="danger">{error}</Callout>}
          {message && !error && <Callout role="status">{message}</Callout>}
        </div>
      )}
    </Card>
  );
}
