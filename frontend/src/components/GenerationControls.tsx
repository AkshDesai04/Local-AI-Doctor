import { AlertTriangle, ChevronDown, Gauge, KeyRound, RotateCcw, SlidersHorizontal, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { AUTH_CHANGED_EVENT, clearSessionAuthToken, hasSessionAuthToken, saveSessionAuthToken } from "../api/auth";
import type { GenerationSettings, ModelSummary } from "../api/types";
import { SAMPLING_LIMITS } from "../domain/sampling";

interface GenerationControlsProps {
  open: boolean;
  model: ModelSummary | null;
  settings: GenerationSettings;
  defaultSettings?: GenerationSettings;
  authRequired: boolean;
  onChange: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  onClose: () => void;
}

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint?: string;
  onChange: (value: number) => void;
}): React.ReactNode {
  return (
    <label className="control-field number-control">
      <span>{label}{hint && <small>{hint}</small>}</span>
      <input max={max} min={min} onChange={(event) => onChange(Number(event.target.value))} step={step} type="number" value={value} />
    </label>
  );
}

export function GenerationControls({ open, model, settings, defaultSettings, authRequired, onChange, onClose }: GenerationControlsProps): React.ReactNode {
  const [tokenInput, setTokenInput] = useState("");
  const [hasToken, setHasToken] = useState(hasSessionAuthToken);

  useEffect(() => {
    const syncTokenState = (): void => setHasToken(hasSessionAuthToken());
    window.addEventListener(AUTH_CHANGED_EVENT, syncTokenState);
    return () => window.removeEventListener(AUTH_CHANGED_EVENT, syncTokenState);
  }, []);

  if (!open) return null;
  const update = <K extends keyof GenerationSettings>(key: K, value: GenerationSettings[K]): void => {
    onChange((current) => ({ ...current, [key]: value }));
  };
  const profiling = settings.instrumentation === "full" || settings.instrumentation === "expert";
  const cudaCapability = model?.capabilities.cuda;
  const cudaUnavailable = cudaCapability?.state === "unavailable_on_backend";
  const cudaUnavailableReason = cudaCapability?.reason ?? "No usable CUDA runtime was discovered on this backend.";

  return (
    <div className="controls-panel" aria-label="Generation settings">
      <div className="panel-heading">
        <div><span className="eyebrow">Run configuration</span><h2>Generation controls</h2></div>
        <button aria-label="Close controls" className="icon-button" onClick={onClose} type="button"><X size={18} /></button>
      </div>
      <div className="controls-scroll">
        <section className={`control-section auth-section ${authRequired ? "required" : ""}`}>
          <div className="control-section-title"><KeyRound size={15} /><span>Connection authentication</span><span className={`auth-state ${hasToken ? "saved" : ""}`}>{hasToken ? "Saved for tab" : "No token"}</span></div>
          {authRequired && <div className="inline-warning" role="alert"><AlertTriangle size={15} /><span>The backend requires authentication. Enter its access token to reconnect.</span></div>}
          <p className="auth-copy">The token stays in memory and session storage for this browser tab only. It is never added to the app build or configuration display.</p>
          <form className="auth-token-row" onSubmit={(event) => {
            event.preventDefault();
            if (!tokenInput.trim()) return;
            saveSessionAuthToken(tokenInput);
            setTokenInput("");
          }}>
            <label className="control-field"><span>Access token</span><input aria-label="Authentication token" autoComplete="off" onChange={(event) => setTokenInput(event.target.value)} placeholder={hasToken ? "Enter a replacement token" : "Enter token"} spellCheck={false} type="password" value={tokenInput} /></label>
            <div className="auth-actions"><button disabled={!tokenInput.trim()} type="submit">Save for tab</button>{hasToken && <button className="clear-auth" onClick={() => { clearSessionAuthToken(); setTokenInput(""); }} type="button"><Trash2 size={12} /> Clear</button>}</div>
          </form>
        </section>
        <section className="control-section">
          <div className="control-section-title"><Gauge size={15} /><span>Runtime</span></div>
          <div className="control-grid two">
            <label className="control-field">
              <span>Device{cudaUnavailable && <small>CUDA unavailable</small>}</span>
              <div className="select-wrap">
                <select
                  aria-describedby={cudaUnavailable ? "cuda-unavailable-reason" : undefined}
                  aria-label="Device"
                  onChange={(event) => update("device", event.target.value as GenerationSettings["device"])}
                  value={settings.device}
                >
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option disabled={cudaUnavailable} value="cuda">{cudaUnavailable ? "CUDA (unavailable)" : "CUDA"}</option>
                </select>
                <ChevronDown size={13} />
              </div>
              {cudaUnavailable && <small id="cuda-unavailable-reason" role="status">{cudaUnavailableReason}</small>}
            </label>
            <label className="control-field"><span>Data type</span><div className="select-wrap"><select onChange={(event) => update("dtype", event.target.value as GenerationSettings["dtype"])} value={settings.dtype}><option value="auto">Auto</option><option value="float32">FP32</option><option value="float16">FP16</option><option value="bfloat16">BF16</option></select><ChevronDown size={13} /></div></label>
            <label className="control-field"><span>Instrumentation</span><div className="select-wrap"><select onChange={(event) => update("instrumentation", event.target.value as GenerationSettings["instrumentation"])} value={settings.instrumentation}><option value="off">Off</option><option value="basic">Basic</option><option value="token">Token</option><option value="full">Full</option><option value="expert">Expert</option></select><ChevronDown size={13} /></div></label>
            <label className="control-field"><span>Seed <small>blank = generated</small></span><input inputMode="numeric" onChange={(event) => update("seed", event.target.value)} placeholder="Unsigned 64-bit" value={settings.seed} /></label>
          </div>
          {profiling && <div className="inline-warning"><AlertTriangle size={15} /><span>Synchronized profiling can alter throughput. The run will report measured overhead.</span></div>}
        </section>
        <section className="control-section">
          <div className="control-section-title"><SlidersHorizontal size={15} /><span>Sampling pipeline</span></div>
          <div className="control-grid two">
            <NumberControl label="Maximum output" {...SAMPLING_LIMITS.maxOutputTokens} onChange={(value) => update("maxOutputTokens", value)} value={settings.maxOutputTokens} />
            <NumberControl label="Temperature" {...SAMPLING_LIMITS.temperature} onChange={(value) => update("temperature", value)} value={settings.temperature} />
            <NumberControl label="Top-K" {...SAMPLING_LIMITS.topK} onChange={(value) => update("topK", value)} value={settings.topK} />
            <NumberControl label="Top-P" {...SAMPLING_LIMITS.topP} onChange={(value) => update("topP", value)} value={settings.topP} />
            <NumberControl hint="adapter-dependent" label="Min-P" {...SAMPLING_LIMITS.minP} onChange={(value) => update("minP", value)} value={settings.minP} />
            <NumberControl label="Repetition penalty" {...SAMPLING_LIMITS.repetitionPenalty} onChange={(value) => update("repetitionPenalty", value)} value={settings.repetitionPenalty} />
            <NumberControl label="Frequency penalty" {...SAMPLING_LIMITS.frequencyPenalty} onChange={(value) => update("frequencyPenalty", value)} value={settings.frequencyPenalty} />
            <NumberControl label="Presence penalty" {...SAMPLING_LIMITS.presencePenalty} onChange={(value) => update("presencePenalty", value)} value={settings.presencePenalty} />
            <NumberControl hint="per distribution" label="Alternatives" {...SAMPLING_LIMITS.alternatives} onChange={(value) => update("alternatives", value)} value={settings.alternatives} />
          </div>
          <label className="control-field full"><span>Stop sequences <small>one per line</small></span><textarea onChange={(event) => update("stopSequences", event.target.value.split("\n").filter(Boolean))} placeholder="Optional" rows={3} value={settings.stopSequences.join("\n")} /></label>
          <label className="switch-row">
            <span><strong>Deterministic reference mode</strong><small>Uses the most reproducible available kernel path; exact replay still depends on the full environment.</small></span>
            <input checked={settings.deterministic} onChange={(event) => update("deterministic", event.target.checked)} type="checkbox" />
          </label>
        </section>
        <section className="effective-settings">
          <span className="eyebrow">Effective for next run</span>
          <code>{model?.name ?? "No model"} · {settings.device}/{settings.dtype} · {settings.instrumentation} · {settings.temperature === 0 ? "greedy" : `temp ${String(settings.temperature)}`}</code>
        </section>
      </div>
      <button className="reset-settings" onClick={() => onChange({ ...(defaultSettings ?? settings), seed: "", stopSequences: [] })} type="button"><RotateCcw size={14} /> Reset configured defaults</button>
    </div>
  );
}
