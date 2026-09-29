import { AlertTriangle, Info, OctagonAlert, X } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { AriaRole, ReactNode } from "react";
import { IconButton } from "./Button";

interface CardProps {
  title?: ReactNode;
  icon?: LucideIcon;
  description?: ReactNode;
  actions?: ReactNode;
  /** Removes body padding for tables and lists that run edge to edge. */
  flush?: boolean;
  titleAs?: "h2" | "h3";
  className?: string;
  children?: ReactNode;
  "aria-label"?: string;
}

export function Card({ title, icon: Icon, description, actions, flush = false, titleAs: Title = "h3", className, children, ...rest }: CardProps): React.ReactNode {
  return (
    <section aria-label={rest["aria-label"]} className={`card ${flush ? "card-flush" : ""} ${className ?? ""}`}>
      {(title || actions) && (
        <header className="card-header">
          <div className="card-heading">
            {Icon && <Icon aria-hidden="true" className="card-icon" size={15} />}
            <div>
              {title && <Title className="card-title">{title}</Title>}
              {description && <p className="card-description">{description}</p>}
            </div>
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children !== undefined && <div className="card-body">{children}</div>}
    </section>
  );
}

interface StatProps {
  label: ReactNode;
  value: ReactNode;
  caption?: ReactNode;
  title?: string;
  className?: string;
}

export function Stat({ label, value, caption, title, className }: StatProps): React.ReactNode {
  return (
    <div className={`stat ${className ?? ""}`} title={title}>
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {caption && <span className="stat-caption">{caption}</span>}
    </div>
  );
}

export type BadgeTone = "neutral" | "accent" | "warning" | "danger" | "info";

interface BadgeProps {
  tone?: BadgeTone;
  icon?: ReactNode;
  title?: string;
  className?: string;
  children: ReactNode;
}

export function Badge({ tone = "neutral", icon, title, className, children }: BadgeProps): React.ReactNode {
  return <span className={`badge badge-${tone} ${className ?? ""}`} title={title}>{icon}{children}</span>;
}

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: ReactNode;
  eyebrow?: string;
  actions?: ReactNode;
  /** `page` is the large centred hero used when a whole view is empty. */
  size?: "compact" | "panel" | "page";
  headingLevel?: 1 | 2 | 3;
  className?: string;
  children?: ReactNode;
}

export function EmptyState({ icon: Icon = Info, title, description, eyebrow, actions, size = "panel", headingLevel = 3, className, children }: EmptyStateProps): React.ReactNode {
  const Heading = `h${String(headingLevel)}` as "h1" | "h2" | "h3";
  return (
    <div className={`empty-state empty-${size} ${className ?? ""}`}>
      <div aria-hidden="true" className="empty-icon"><Icon size={size === "page" ? 22 : 18} /></div>
      {eyebrow && <span className="empty-eyebrow">{eyebrow}</span>}
      <Heading className="empty-title">{title}</Heading>
      {description && <p className="empty-description">{description}</p>}
      {children}
      {actions && <div className="empty-actions">{actions}</div>}
    </div>
  );
}

export type CalloutTone = "info" | "warning" | "danger";

const calloutIcons: Record<CalloutTone, LucideIcon> = { info: Info, warning: AlertTriangle, danger: OctagonAlert };

interface CalloutProps {
  tone?: CalloutTone;
  title?: ReactNode;
  icon?: LucideIcon | null;
  /** Defaults to `alert` for danger and `note` otherwise. */
  role?: AriaRole;
  onDismiss?: () => void;
  dismissLabel?: string;
  className?: string;
  children?: ReactNode;
  "aria-label"?: string;
}

export function Callout({ tone = "info", title, icon, role, onDismiss, dismissLabel = "Dismiss", className, children, ...rest }: CalloutProps): React.ReactNode {
  const Icon = icon === null ? null : icon ?? calloutIcons[tone];
  return (
    <div aria-label={rest["aria-label"]} className={`callout callout-${tone} ${className ?? ""}`} role={role ?? (tone === "danger" ? "alert" : "note")}>
      {Icon && <Icon aria-hidden="true" className="callout-icon" size={15} />}
      <div className="callout-body">
        {title && <strong className="callout-title">{title}</strong>}
        {children}
      </div>
      {onDismiss && (
        <IconButton className="callout-dismiss" icon={<X size={14} />} label={dismissLabel} onClick={onDismiss} size="sm" />
      )}
    </div>
  );
}
