import { LoaderCircle } from "lucide-react";
import { type ButtonHTMLAttributes, forwardRef, type ReactNode } from "react";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ControlSize = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ControlSize;
  /** Leading icon; replaced by a spinner while `loading`. */
  icon?: ReactNode;
  loading?: boolean;
  block?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon, loading = false, block = false, className, disabled, children, type = "button", ...rest },
  ref,
) {
  const classes = ["btn", `btn-${variant}`, `btn-${size}`, block ? "btn-block" : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <button {...rest} aria-busy={loading || undefined} className={classes} disabled={disabled || loading} ref={ref} type={type}>
      {loading ? <LoaderCircle aria-hidden="true" className="spin" size={14} /> : icon}
      {children}
    </button>
  );
});

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label" | "children"> {
  /** Accessible name; also used as the hover hint unless `title` overrides it. */
  label: string;
  icon: ReactNode;
  size?: ControlSize;
  variant?: "ghost" | "secondary";
  pressed?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, size = "md", variant = "ghost", pressed, className, title, type = "button", ...rest },
  ref,
) {
  const classes = ["icon-btn", `icon-btn-${variant}`, `icon-btn-${size}`, className ?? ""].filter(Boolean).join(" ");
  return (
    <button {...rest} aria-label={label} aria-pressed={pressed} className={classes} ref={ref} title={title ?? label} type={type}>
      {icon}
    </button>
  );
});
