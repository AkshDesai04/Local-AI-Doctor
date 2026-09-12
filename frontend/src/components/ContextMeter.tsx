import { AlertTriangle, Info } from "lucide-react";
import type { ContextUsage } from "../api/types";
import { formatNumber } from "../utils/format";

interface ContextMeterProps {
  context: ContextUsage | undefined;
  fallbackLimit: number | null;
  onOpen: () => void;
}

function contextTotal(context: ContextUsage | undefined): number {
  if (!context) return 0;
  return (context.renderedPromptTokens ?? 0) + (context.templateTokens ?? 0) + (context.multimodalPositions ?? 0) + (context.generatedTokens ?? 0) + (context.reservedOutputTokens ?? 0);
}

export function ContextMeter({ context, fallbackLimit, onOpen }: ContextMeterProps): React.ReactNode {
  const limit = context?.effectiveLimit ?? fallbackLimit;
  const used = contextTotal(context);
  const percent = limit && limit > 0 ? Math.min(100, (used / limit) * 100) : null;
  const circumference = 2 * Math.PI * 19;
  const offset = percent === null ? circumference : circumference * (1 - percent / 100);
  const warning = percent !== null && percent >= 85;

  return (
    <button
      aria-label={percent === null ? "Context usage unknown" : `Context usage ${formatNumber(percent, 0)} percent`}
      className={`context-meter ${warning ? "warning" : ""}`}
      onClick={onOpen}
      title="Open context inspector"
      type="button"
    >
      <svg height="48" viewBox="0 0 48 48" width="48">
        <circle className="meter-track" cx="24" cy="24" fill="none" r="19" strokeWidth="4" />
        <circle
          className="meter-progress"
          cx="24"
          cy="24"
          fill="none"
          r="19"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          strokeWidth="4"
          transform="rotate(-90 24 24)"
        />
      </svg>
      <span className="meter-value">{percent === null ? <Info size={14} /> : warning ? <AlertTriangle size={13} /> : `${formatNumber(percent, 0)}%`}</span>
      <span className="meter-tooltip">
        <strong>{percent === null ? "Context unknown" : `${formatNumber(used, 0)} / ${formatNumber(limit, 0)}`}</strong>
        <small>{context?.behavior === "truncated" ? "Input was truncated" : context?.behavior === "sliding_window" ? "Sliding window active" : "tokens incl. reserve"}</small>
      </span>
    </button>
  );
}
