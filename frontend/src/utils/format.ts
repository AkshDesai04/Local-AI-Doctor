export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${formatNumber(value * 100, digits)}%`;
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  if (ms < 1) return `${formatNumber(ms, 2)} ms`;
  if (ms < 1000) return `${formatNumber(ms, 1)} ms`;
  if (ms < 60_000) return `${formatNumber(ms / 1000, 2)} s`;
  const minutes = Math.floor(ms / 60_000);
  return `${String(minutes)}m ${formatNumber((ms % 60_000) / 1000, 0)}s`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${formatNumber(bytes / 1024 ** exponent, exponent === 0 ? 0 : 1)} ${units[exponent] ?? "B"}`;
}

export function relativeTime(value: string): string {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "";
  const deltaSeconds = Math.round((timestamp - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (Math.abs(deltaSeconds) < 60) return formatter.format(deltaSeconds, "second");
  const minutes = Math.round(deltaSeconds / 60);
  if (Math.abs(minutes) < 60) return formatter.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return formatter.format(hours, "hour");
  const days = Math.round(hours / 24);
  if (Math.abs(days) < 7) return formatter.format(days, "day");
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(timestamp);
}

export function shortFingerprint(value: string | null | undefined): string {
  if (!value) return "Not fingerprinted";
  return value.length > 14 ? `${value.slice(0, 7)}…${value.slice(-6)}` : value;
}

export function escapeToken(value: string): string {
  return value
    .replaceAll("\\", "\\\\")
    .replaceAll("\n", "\\n")
    .replaceAll("\r", "\\r")
    .replaceAll("\t", "\\t")
    .replaceAll(" ", "·");
}

const tokenizerWhitespaceMarkers: Readonly<Record<string, string>> = {
  "Ġ": " ",
  "▁": " ",
  "Ċ": "\n",
  "č": "\r",
  "ĉ": "\t",
};

/**
 * Produces a compact, human-facing label for a tokenizer piece. Byte-level BPE
 * and SentencePiece expose whitespace using private-looking glyphs such as
 * `Ġ`, `Ċ`, and `▁`; those remain available in the raw metadata tooltip,
 * but should not leak into the primary UI. Control whitespace is represented on
 * one line so a newline-only token cannot stretch a token chip vertically.
 */
export function displayTokenText(value: string): string {
  const characters = Array.from(value);
  let markerEnd = 0;
  let decodedPrefix = "";
  while (markerEnd < characters.length) {
    const marker = tokenizerWhitespaceMarkers[characters[markerEnd] ?? ""];
    if (marker === undefined) break;
    decodedPrefix += marker;
    markerEnd += 1;
  }
  const decoded = `${decodedPrefix}${characters.slice(markerEnd).join("")}`;
  if (!decoded) return "";
  if (/^\s+$/u.test(decoded)) {
    if (/\r|\n/u.test(decoded)) return decoded.replace(/\r\n|\r|\n/gu, "↵").replace(/[ \t]/gu, "");
    if (decoded.includes("\t")) return "⇥";
    return "␠";
  }
  return decoded
    .trimStart()
    .replace(/\r\n|\r|\n/gu, "↵")
    .replaceAll("\t", "⇥");
}

/** Keeps exact tokenizer data discoverable without making it the visible label. */
export function tokenTextHint(piece: string, displayText?: string): string {
  const visible = displayTokenText(displayText ?? piece) || "∅";
  return `Displayed text: ${visible}\nRaw tokenizer piece: ${escapeToken(piece) || "∅"}`;
}

export function downloadBlob(name: string, value: BlobPart, type: string): void {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}
