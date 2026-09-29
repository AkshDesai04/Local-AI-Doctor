import { AlertTriangle, X } from "lucide-react";
import { useEffect } from "react";
import { Button, IconButton } from "./ui";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps): React.ReactNode {
  useEffect(() => {
    if (!open) return undefined;
    const escape = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [onCancel, open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <section
        aria-describedby="confirm-description"
        aria-labelledby="confirm-title"
        aria-modal="true"
        className="dialog-card"
        onMouseDown={(event) => event.stopPropagation()}
        role="alertdialog"
      >
        <IconButton className="dialog-close" icon={<X size={16} />} label="Close dialog" onClick={onCancel} />
        <div className={`dialog-icon ${danger ? "danger" : ""}`}><AlertTriangle size={18} /></div>
        <h2 id="confirm-title">{title}</h2>
        <p id="confirm-description">{description}</p>
        <div className="dialog-actions">
          <Button autoFocus onClick={onCancel} variant="secondary">Cancel</Button>
          <Button onClick={onConfirm} variant={danger ? "danger" : "primary"}>{confirmLabel}</Button>
        </div>
      </section>
    </div>
  );
}
