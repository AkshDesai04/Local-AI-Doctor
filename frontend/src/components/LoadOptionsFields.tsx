import type { LoadOptions, ModelSummary, QuantizationMode } from "../api/types";
import { quantizationBlockedReason } from "../domain/capabilities";
import { Badge, Field, Select, Switch } from "./ui";

interface LoadOptionsFieldsProps {
  model: ModelSummary | null;
  value: LoadOptions;
  onChange: (value: LoadOptions) => void;
  size?: "sm" | "md";
  disabled?: boolean;
}

/** Device, dtype, quantization, and Strict VRAM for one model's loads. Renders a fragment so the parent owns the layout. */
export function LoadOptionsFields({ model, value, onChange, size = "md", disabled = false }: LoadOptionsFieldsProps): React.ReactNode {
  const cuda = model?.capabilities.cuda;
  const cudaBlocked = cuda !== undefined && cuda.state !== "full" && cuda.state !== "partial";
  const cudaReason = cuda?.reason ?? "No usable CUDA runtime was discovered on this backend.";
  const onCpu = value.device === "cpu";
  const quantizationBlocked = quantizationBlockedReason(model, value.device);
  const prequantized = model?.weightQuantization;
  return (
    <>
      <Field addon={cudaBlocked ? <Badge tone="warning">CUDA unavailable</Badge> : undefined} className="load-device" hint={cudaBlocked ? cudaReason : undefined} label="Device">
        <Select controlSize={size} disabled={disabled} onChange={(event) => onChange({ ...value, device: event.target.value as LoadOptions["device"] })} value={value.device}>
          <option value="auto">Auto</option>
          <option disabled={cudaBlocked} value="cuda">{cudaBlocked ? "CUDA (unavailable)" : "CUDA"}</option>
          <option value="cpu">CPU</option>
        </Select>
      </Field>
      <Field className="load-dtype" label="Data type">
        <Select controlSize={size} disabled={disabled} onChange={(event) => onChange({ ...value, dtype: event.target.value as LoadOptions["dtype"] })} value={value.dtype}>
          <option value="auto">Auto</option>
          <option value="bfloat16">bfloat16</option>
          <option value="float16">float16</option>
          <option value="float32">float32</option>
        </Select>
      </Field>
      <Field
        addon={prequantized ? <Badge tone="info">Pre-quantized</Badge> : undefined}
        className="load-quantization"
        hint={quantizationBlocked ?? undefined}
        label="Quantization"
      >
        <Select
          controlSize={size}
          disabled={disabled || quantizationBlocked !== null}
          onChange={(event) => onChange({ ...value, quantization: event.target.value as QuantizationMode })}
          title={quantizationBlocked ?? "Quantized in GPU memory at load time; nothing is written to disk until you flush it"}
          value={quantizationBlocked === null ? value.quantization ?? "none" : "none"}
        >
          <option value="none">None</option>
          <option value="bitsandbytes-4bit">4-bit (NF4)</option>
          <option value="bitsandbytes-8bit">8-bit (LLM.int8)</option>
        </Select>
      </Field>
      <Switch
        checked={value.strictVram}
        className="load-strict"
        description={onCpu ? "Applies to GPU loads only." : "Keep the model entirely in GPU memory; fail instead of spilling into system RAM"}
        disabled={disabled || onCpu}
        label="Strict VRAM"
        onChange={(event) => onChange({ ...value, strictVram: event.target.checked })}
        title={onCpu ? "Strict VRAM applies to GPU loads only" : "Keep the model entirely in GPU memory; fail instead of spilling into system RAM"}
      />
    </>
  );
}
