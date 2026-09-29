import { ChevronDown } from "lucide-react";
import {
  createContext,
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
  useContext,
  useId,
} from "react";

interface FieldContextValue {
  id: string;
  describedBy?: string;
  invalid: boolean;
}

const FieldContext = createContext<FieldContextValue | null>(null);

interface FieldProps {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  /** Extra content beside the label, such as a status badge. */
  addon?: ReactNode;
  id?: string;
  className?: string;
  children: ReactNode;
}

/** Labels one control and wires its hint and error through `aria-describedby`. */
export function Field({ label, hint, error, addon, id, className, children }: FieldProps): React.ReactNode {
  const generated = useId();
  const controlId = id ?? generated;
  const hintId = hint ? `${controlId}-hint` : undefined;
  const errorId = error ? `${controlId}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(" ") || undefined;
  return (
    <div className={`field ${className ?? ""}`}>
      <div className="field-label-row"><label className="field-label" htmlFor={controlId}>{label}</label>{addon}</div>
      <FieldContext.Provider value={{ id: controlId, describedBy, invalid: Boolean(error) }}>{children}</FieldContext.Provider>
      {hint && <p className="field-hint" id={hintId}>{hint}</p>}
      {error && <p className="field-error" id={errorId} role="alert">{error}</p>}
    </div>
  );
}

function useFieldControl(id: string | undefined, describedBy: string | undefined, invalid: boolean | "true" | "false" | "grammar" | "spelling" | undefined): {
  id?: string;
  "aria-describedby"?: string;
  "aria-invalid"?: boolean | "true" | "false" | "grammar" | "spelling";
} {
  const field = useContext(FieldContext);
  return {
    id: id ?? field?.id,
    "aria-describedby": [describedBy, field?.describedBy].filter(Boolean).join(" ") || undefined,
    "aria-invalid": invalid ?? (field?.invalid ? true : undefined),
  };
}

type ControlSize = "sm" | "md";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  controlSize?: ControlSize;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input({ controlSize = "md", className, id, ...rest }, ref) {
  const control = useFieldControl(id, rest["aria-describedby"], rest["aria-invalid"]);
  return <input {...rest} {...control} className={`input input-${controlSize} ${className ?? ""}`} ref={ref} />;
});

interface NumberInputProps extends Omit<InputProps, "type" | "value" | "onChange"> {
  value: number;
  onValueChange: (value: number) => void;
}

export function NumberInput({ value, onValueChange, ...rest }: NumberInputProps): React.ReactNode {
  return <Input {...rest} inputMode="decimal" onChange={(event) => onValueChange(Number(event.target.value))} type="number" value={value} />;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, id, ...rest }, ref) {
  const control = useFieldControl(id, rest["aria-describedby"], rest["aria-invalid"]);
  return <textarea {...rest} {...control} className={`input textarea ${className ?? ""}`} ref={ref} />;
});

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  controlSize?: ControlSize;
  /** `ghost` drops the border for use inside toolbars and headers. */
  variant?: "default" | "ghost";
  wrapperClassName?: string;
}

/** A native select with consistent chrome; keyboard and screen-reader behaviour stay native. */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { controlSize = "md", variant = "default", className, wrapperClassName, id, children, ...rest },
  ref,
) {
  const control = useFieldControl(id, rest["aria-describedby"], rest["aria-invalid"]);
  return (
    <span className={`select-wrap select-${variant} select-${controlSize} ${wrapperClassName ?? ""}`}>
      <select {...rest} {...control} className={`select ${className ?? ""}`} ref={ref}>{children}</select>
      <ChevronDown aria-hidden="true" className="select-chevron" size={controlSize === "sm" ? 13 : 14} />
    </span>
  );
});

interface SwitchProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "role"> {
  label: ReactNode;
  description?: ReactNode;
  /** Visually hides the label while keeping it as the accessible name. */
  hideLabel?: boolean;
}

/** A native checkbox exposed with the switch role, so Space toggles it and forms still work. */
export function Switch({ label, description, hideLabel = false, className, title, ...rest }: SwitchProps): React.ReactNode {
  return (
    <label className={`switch-field ${hideLabel ? "label-hidden" : ""} ${className ?? ""}`} title={title}>
      <span className={hideLabel ? "visually-hidden" : "switch-text"}>
        <span className="switch-label">{label}</span>
        {description && <span className="switch-description">{description}</span>}
      </span>
      <input {...rest} className="switch" role="switch" type="checkbox" />
    </label>
  );
}

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "value" | "onChange" | "min" | "max"> {
  label: string;
  value: number;
  min: number;
  max: number;
  onValueChange: (value: number) => void;
  /** Formats the visible readout; defaults to the raw number. */
  format?: (value: number) => string;
}

export function Slider({ label, value, min, max, onValueChange, format = String, title, className, id, ...rest }: SliderProps): React.ReactNode {
  const generated = useId();
  const controlId = id ?? generated;
  const fill = max > min ? Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)) : 0;
  return (
    <div className={`slider-field ${className ?? ""}`} title={title}>
      <div className="slider-head"><label htmlFor={controlId}>{label}</label><output htmlFor={controlId}>{format(value)}</output></div>
      <input
        {...rest}
        className="slider"
        id={controlId}
        max={max}
        min={min}
        onChange={(event) => onValueChange(Number(event.target.value))}
        style={{ "--fill": `${String(fill)}%` } as React.CSSProperties}
        type="range"
        value={value}
      />
    </div>
  );
}
