import { X } from "lucide-react";
import { createContext, type ReactNode, type RefObject, useContext, useEffect, useRef, useState } from "react";
import { IconButton } from "./Button";
import { moveFocus } from "./focus";

interface PopoverProps {
  open: boolean;
  onClose: () => void;
  /** The control that opened the popover: clicks on it are not "outside", and Escape returns focus to it. */
  anchorRef: RefObject<HTMLElement | null>;
  label: string;
  role?: "dialog" | "menu";
  className?: string;
  children: ReactNode;
}

/**
 * A lightweight non-modal layer. It closes on Escape (returning focus to its
 * anchor) and on pointer presses outside itself and its anchor. Placement is
 * left to CSS, relative to the nearest positioned ancestor.
 */
export function Popover({ open, onClose, anchorRef, label, role = "dialog", className, children }: PopoverProps): React.ReactNode {
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return undefined;
    const pressOutside = (event: MouseEvent): void => {
      const target = event.target as Node;
      if (panel.current?.contains(target) || anchorRef.current?.contains(target)) return;
      onClose();
    };
    const escape = (event: KeyboardEvent): void => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      onClose();
      anchorRef.current?.focus();
    };
    document.addEventListener("mousedown", pressOutside);
    // Capture phase so an enclosing drawer sees the Escape as already handled.
    document.addEventListener("keydown", escape, true);
    return () => {
      document.removeEventListener("mousedown", pressOutside);
      document.removeEventListener("keydown", escape, true);
    };
  }, [anchorRef, onClose, open]);

  useEffect(() => {
    if (open && role === "menu") panel.current?.querySelector<HTMLElement>('[role="menuitem"]:not(:disabled)')?.focus();
  }, [open, role]);

  if (!open) return null;
  return (
    <div
      aria-label={label}
      className={`popover ${className ?? ""}`}
      onKeyDown={role === "menu" ? (event) => { moveFocus(event, '[role="menuitem"]:not(:disabled)', "vertical"); } : undefined}
      ref={panel}
      role={role}
    >
      {children}
    </div>
  );
}

const MenuCloseContext = createContext<() => void>(() => undefined);

interface MenuItemProps {
  onSelect: () => void;
  icon?: ReactNode;
  danger?: boolean;
  disabled?: boolean;
  title?: string;
  className?: string;
  children: ReactNode;
}

export function MenuItem({ onSelect, icon, danger = false, disabled = false, title, className, children }: MenuItemProps): React.ReactNode {
  const close = useContext(MenuCloseContext);
  return (
    <button
      className={`menu-item ${danger ? "danger" : ""} ${className ?? ""}`}
      disabled={disabled}
      onClick={() => { close(); onSelect(); }}
      role="menuitem"
      tabIndex={-1}
      title={title}
      type="button"
    >{icon}<span>{children}</span></button>
  );
}

interface MenuButtonProps {
  label: string;
  icon: ReactNode;
  /** Positions the menu; see `.menu-*` placement classes. */
  placement?: "bottom-end" | "bottom-start" | "top-start" | "top-end";
  size?: "sm" | "md";
  className?: string;
  buttonClassName?: string;
  children: ReactNode;
}

/** An icon button that toggles a keyboard-navigable action menu. */
export function MenuButton({ label, icon, placement = "bottom-end", size = "md", className, buttonClassName, children }: MenuButtonProps): React.ReactNode {
  const [open, setOpen] = useState(false);
  const anchor = useRef<HTMLButtonElement>(null);
  const close = (): void => setOpen(false);
  return (
    <div className={`menu-anchor ${className ?? ""}`}>
      <IconButton
        aria-expanded={open}
        aria-haspopup="menu"
        className={buttonClassName}
        icon={icon}
        label={label}
        onClick={() => setOpen((value) => !value)}
        ref={anchor}
        size={size}
      />
      <MenuCloseContext.Provider value={close}>
        <Popover anchorRef={anchor} className={`menu menu-${placement}`} label={label} onClose={close} open={open} role="menu">{children}</Popover>
      </MenuCloseContext.Provider>
    </div>
  );
}

interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  label: string;
  closeLabel?: string;
  side?: "left" | "right";
  footer?: ReactNode;
  className?: string;
  children: ReactNode;
}

/**
 * A side panel. It takes focus when it opens and returns it to the opener when it
 * closes; Escape closes it. A scrim is shown only at narrow widths (see CSS).
 */
export function Drawer({ open, onClose, title, description, label, closeLabel = "Close panel", side = "right", footer, className, children }: DrawerProps): React.ReactNode {
  const panel = useRef<HTMLElement>(null);
  // Read through a ref so a new onClose identity on re-render does not re-run the focus handling.
  const close = useRef(onClose);
  useEffect(() => {
    close.current = onClose;
  });

  useEffect(() => {
    if (!open) return undefined;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    panel.current?.focus();
    const escape = (event: KeyboardEvent): void => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      close.current();
    };
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("keydown", escape);
      if (opener?.isConnected) opener.focus();
    };
  }, [open]);

  if (!open) return null;
  return (
    <>
      <div aria-hidden="true" className="drawer-scrim" onClick={onClose} />
      <aside aria-label={label} className={`drawer drawer-${side} ${className ?? ""}`} ref={panel} role="dialog" tabIndex={-1}>
        <header className="drawer-header">
          <div>
            <h2 className="drawer-title">{title}</h2>
            {description && <p className="drawer-description">{description}</p>}
          </div>
          <IconButton icon={<X size={16} />} label={closeLabel} onClick={onClose} />
        </header>
        <div className="drawer-body">{children}</div>
        {footer && <footer className="drawer-footer">{footer}</footer>}
      </aside>
    </>
  );
}
