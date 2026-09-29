import type { LucideIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { moveFocus } from "./focus";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  icon?: LucideIcon;
  disabled?: boolean;
  /** Shown as the hover hint and read as the tab description while disabled. */
  disabledReason?: string;
}

interface TabsProps<T extends string> {
  items: Array<TabItem<T>>;
  value: T;
  onChange: (value: T) => void;
  label: string;
  /** Prefix for tab and panel ids so a tabpanel can reference its tab. */
  idPrefix: string;
  className?: string;
}

/**
 * Horizontal tab list with roving tabindex and automatic activation. Labels stay
 * visible; when the list overflows it scrolls and fades the clipped edge.
 */
export function Tabs<T extends string>({ items, value, onChange, label, idPrefix, className }: TabsProps<T>): React.ReactNode {
  const list = useRef<HTMLDivElement>(null);
  const [overflow, setOverflow] = useState({ start: false, end: false });

  useEffect(() => {
    const element = list.current;
    if (!element) return undefined;
    const measure = (): void => setOverflow({
      start: element.scrollLeft > 1,
      end: element.scrollLeft + element.clientWidth < element.scrollWidth - 1,
    });
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    observer?.observe(element);
    return () => {
      element.removeEventListener("scroll", measure);
      observer?.disconnect();
    };
  }, []);

  useEffect(() => {
    list.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [value]);

  return (
    <div className={`tabs ${overflow.start ? "fade-start" : ""} ${overflow.end ? "fade-end" : ""} ${className ?? ""}`}>
      <div
        aria-label={label}
        className="tabs-list"
        onKeyDown={(event) => {
          const index = moveFocus(event, '[role="tab"]');
          const item = index === null ? undefined : items[index];
          if (item && !item.disabled) onChange(item.id);
        }}
        ref={list}
        role="tablist"
      >
        {items.map((item) => {
          const selected = item.id === value;
          const Icon = item.icon;
          return (
            <button
              aria-controls={`${idPrefix}-panel`}
              aria-disabled={item.disabled || undefined}
              aria-selected={selected}
              className="tab"
              id={`${idPrefix}-tab-${item.id}`}
              key={item.id}
              onClick={() => { if (!item.disabled) onChange(item.id); }}
              role="tab"
              tabIndex={selected ? 0 : -1}
              title={item.disabled ? item.disabledReason ?? `${item.label} is unavailable` : undefined}
              type="button"
            >
              {Icon && <Icon aria-hidden="true" size={14} />}
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  title?: string;
}

interface SegmentedControlProps<T extends string> {
  options: Array<SegmentedOption<T>>;
  /** `null` when the current value matches none of the options. */
  value: T | null;
  onChange: (value: T) => void;
  label: string;
  size?: "sm" | "md";
  className?: string;
}

/** A compact single-choice control exposed as a radio group. */
export function SegmentedControl<T extends string>({ options, value, onChange, label, size = "md", className }: SegmentedControlProps<T>): React.ReactNode {
  const focusable = options.some((option) => option.value === value) ? value : options[0]?.value;
  return (
    <div
      aria-label={label}
      className={`segmented segmented-${size} ${className ?? ""}`}
      onKeyDown={(event) => {
        const index = moveFocus(event, '[role="radio"]');
        const option = index === null ? undefined : options[index];
        if (option) onChange(option.value);
      }}
      role="radiogroup"
    >
      {options.map((option) => (
        <button
          aria-checked={option.value === value}
          className="segment"
          key={option.value}
          onClick={() => onChange(option.value)}
          role="radio"
          tabIndex={option.value === focusable ? 0 : -1}
          title={option.title}
          type="button"
        >{option.label}</button>
      ))}
    </div>
  );
}
