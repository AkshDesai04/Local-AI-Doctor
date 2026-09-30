import { KeyRound, RotateCcw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { AUTH_CHANGED_EVENT, clearSessionAuthToken, hasSessionAuthToken, saveSessionAuthToken } from "../api/auth";
import type { GenerationSettings, ModelSummary } from "../api/types";
import { SAMPLING_LIMITS } from "../domain/sampling";
import { Badge, Button, Callout, Drawer, Field, Input, NumberInput, Select, Switch, Textarea } from "./ui";

interface GenerationControlsProps {
  open: boolean;
  model: ModelSummary | null;
  settings: GenerationSettings;
  defaultSettings?: GenerationSettings;
  authRequired: boolean;
  onChange: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  onClose: () => void;
}

type NumericKey = keyof typeof SAMPLING_LIMITS;

export function GenerationControls({ open, model, settings, defaultSettings, authRequired, onChange, onClose }: GenerationControlsProps): React.ReactNode {
  const [tokenInput, setTokenInput] = useState("");
  const [hasToken, setHasToken] = useState(hasSessionAuthToken);

  useEffect(() => {
    const syncTokenState = (): void => setHasToken(hasSessionAuthToken());
    window.addEventListener(AUTH_CHANGED_EVENT, syncTokenState);
    return () => window.removeEventListener(AUTH_CHANGED_EVENT, syncTokenState);
  }, []);

  const update = <K extends keyof GenerationSettings>(key: K, value: GenerationSettings[K]): void => {
    onChange((current) => ({ ...current, [key]: value }));
  };
  const numberField = (key: NumericKey, label: string, title?: string): React.ReactNode => (
    <Field label={label}>
      <NumberInput {...SAMPLING_LIMITS[key]} onValueChange={(value) => update(key, value)} title={title} value={settings[key]} />
    </Field>
  );
  const profiling = settings.instrumentation === "full" || settings.instrumentation === "expert";
  const cudaCapability = model?.capabilities.cuda;
  const cudaUnavailable = cudaCapability?.state === "unavailable_on_backend";
  const cudaUnavailableReason = cudaCapability?.reason ?? "No usable CUDA runtime was discovered on this backend.";
  const summary = `${model?.name ?? "No model"} · ${settings.device}/${settings.dtype} · ${settings.instrumentation} · ${settings.temperature === 0 ? "greedy" : `temp ${String(settings.temperature)}`}`;

  return (
    <Drawer
      className="controls-drawer"
      closeLabel="Close controls"
      description="Settings for the next response and for model loads."
      footer={(
        <>
          <p className="controls-summary" title={summary}><span>Next run</span><code>{summary}</code></p>
          <Button icon={<RotateCcw size={14} />} onClick={() => onChange({ ...(defaultSettings ?? settings), seed: "", stopSequences: [] })} size="sm" variant="ghost">Reset to defaults</Button>
        </>
      )}
      label="Generation settings"
      onClose={onClose}
      open={open}
      title="Generation controls"
    >
      {authRequired && (
        <Callout className="controls-auth-alert" role="alert" tone="warning">The backend requires authentication. Enter its access token under Connection to reconnect.</Callout>
      )}

      <section aria-labelledby="controls-next-response" className="controls-section">
        <header className="controls-section-head">
          <h3 id="controls-next-response">Next response</h3>
          <p>Sampling and telemetry for the next message you send. The composer’s quick controls edit the same values.</p>
        </header>
        <div className="form-grid">
          {numberField("maxOutputTokens", "Maximum output")}
          {numberField("temperature", "Temperature")}
          {numberField("topK", "Top-K")}
          {numberField("topP", "Top-P")}
          {numberField("minP", "Min-P", "Adapter-dependent")}
          {numberField("repetitionPenalty", "Repetition penalty")}
          {numberField("frequencyPenalty", "Frequency penalty")}
          {numberField("presencePenalty", "Presence penalty")}
          {numberField("alternatives", "Alternatives", "Top alternatives captured per distribution")}
          <Field hint="Blank generates one" label="Seed">
            <Input inputMode="numeric" onChange={(event) => update("seed", event.target.value)} placeholder="Unsigned 64-bit" value={settings.seed} />
          </Field>
          <Field className="span-2" hint="Token captures full-vocabulary probabilities and alternatives; Full and Expert add attention capture and synchronized timing." label="Instrumentation">
            <Select onChange={(event) => update("instrumentation", event.target.value as GenerationSettings["instrumentation"])} value={settings.instrumentation}>
              <option value="off">Off</option>
              <option value="basic">Basic</option>
              <option value="token">Token</option>
              <option value="full">Full</option>
              <option value="expert">Expert</option>
            </Select>
          </Field>
          {profiling && <Callout className="span-2" tone="warning">Synchronized profiling can alter throughput. The run will report measured overhead.</Callout>}
          <Field className="span-2" hint="One per line" label="Stop sequences">
            <Textarea onChange={(event) => update("stopSequences", event.target.value.split("\n").filter(Boolean))} placeholder="Optional" rows={3} value={settings.stopSequences.join("\n")} />
          </Field>
          <Switch
            checked={settings.deterministic}
            className="span-2"
            description="Uses the most reproducible kernel path available; exact replay still depends on the full environment."
            label="Deterministic reference mode"
            onChange={(event) => update("deterministic", event.target.checked)}
          />
        </div>
      </section>

      <section aria-labelledby="controls-model-loading" className="controls-section">
        <header className="controls-section-head">
          <h3 id="controls-model-loading">Model loading</h3>
          <p>Used by Load and by the next response. Choosing a different device or data type reloads the model before it runs.</p>
        </header>
        <div className="form-grid">
          <Field addon={cudaUnavailable ? <Badge tone="warning">CUDA unavailable</Badge> : undefined} hint={cudaUnavailable ? cudaUnavailableReason : undefined} label="Device">
            <Select onChange={(event) => update("device", event.target.value as GenerationSettings["device"])} value={settings.device}>
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option disabled={cudaUnavailable} value="cuda">{cudaUnavailable ? "CUDA (unavailable)" : "CUDA"}</option>
            </Select>
          </Field>
          <Field label="Data type">
            <Select onChange={(event) => update("dtype", event.target.value as GenerationSettings["dtype"])} value={settings.dtype}>
              <option value="auto">Auto</option>
              <option value="float32">FP32</option>
              <option value="float16">FP16</option>
              <option value="bfloat16">BF16</option>
            </Select>
          </Field>
        </div>
      </section>

      <section aria-labelledby="controls-connection" className="controls-section">
        <header className="controls-section-head">
          <h3 id="controls-connection"><KeyRound aria-hidden="true" size={14} />Connection</h3>
          <Badge tone={hasToken ? "accent" : "neutral"}>{hasToken ? "Saved for tab" : "No token"}</Badge>
        </header>
        <p className="controls-copy">The token stays in memory and session storage for this browser tab only. It is never added to the app build or configuration display.</p>
        <form className="auth-token-row" onSubmit={(event) => {
          event.preventDefault();
          if (!tokenInput.trim()) return;
          saveSessionAuthToken(tokenInput);
          setTokenInput("");
        }}>
          <Field label="Access token">
            <Input aria-label="Authentication token" autoComplete="off" onChange={(event) => setTokenInput(event.target.value)} placeholder={hasToken ? "Enter a replacement token" : "Enter token"} spellCheck={false} type="password" value={tokenInput} />
          </Field>
          <div className="auth-actions">
            <Button disabled={!tokenInput.trim()} size="md" type="submit" variant="primary">Save for tab</Button>
            {hasToken && <Button icon={<Trash2 size={13} />} onClick={() => { clearSessionAuthToken(); setTokenInput(""); }} variant="ghost">Clear</Button>}
          </div>
        </form>
      </section>
    </Drawer>
  );
}
