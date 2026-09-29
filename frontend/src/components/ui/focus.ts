import type { KeyboardEvent } from "react";

/**
 * Roving focus for composite widgets: moves focus between the container's
 * matching descendants with arrow, Home, and End keys (wrapping at the ends).
 * Returns the newly focused index, or null when the key is not a navigation key.
 */
export function moveFocus(event: KeyboardEvent<HTMLElement>, selector: string, orientation: "horizontal" | "vertical" = "horizontal"): number | null {
  const [previousKey, nextKey] = orientation === "vertical" ? ["ArrowUp", "ArrowDown"] : ["ArrowLeft", "ArrowRight"];
  if (event.key !== previousKey && event.key !== nextKey && event.key !== "Home" && event.key !== "End") return null;
  const items = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(selector));
  if (!items.length) return null;
  const current = items.indexOf(document.activeElement as HTMLElement);
  const last = items.length - 1;
  const next = event.key === "Home" ? 0
    : event.key === "End" ? last
      : event.key === previousKey ? (current <= 0 ? last : current - 1)
        : (current >= last ? 0 : current + 1);
  event.preventDefault();
  items[next]?.focus();
  return next;
}
