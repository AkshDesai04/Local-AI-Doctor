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

const radius = 7;
const circumference = 2 * Math.PI * radius;

/** A compact context-usage ring for the composer toolbar; details show on hover or focus. */
export function ContextMeter({ context, fallbackLimit, onOpen }: ContextMeterProps): React.ReactNode {
  const limit = context?.effectiveLimit ?? fallbackLimit;
  const used = contextTotal(context);
  const percent = limit && limit > 0 ? Math.min(100, (used / limit) * 100) : null;
  const offset = percent === null ? circumference : circumference * (1 - percent / 100);
  const warning = percent !== null && percent >= 85;

  return (
    <button
      aria-describedby="context-meter-detail"
      aria-label={percent === null ? "Context usage unknown" : `Context usage ${formatNumber(percent, 0)} percent`}
      className={`context-meter ${warning ? "warning" : ""} ${percent === null ? "unknown" : ""}`}
      onClick={onOpen}
      type="button"
    >
      <svg aria-hidden="true" height="18" viewBox="0 0 18 18" width="18">
        <circle className="meter-track" cx="9" cy="9" fill="none" r={radius} strokeWidth="2.5" />
        <circle
          className="meter-progress"
          cx="9"
          cy="9"
          fill="none"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          strokeWidth="2.5"
          transform="rotate(-90 9 9)"
        />
      </svg>
      <span aria-hidden="true" className="meter-value">{percent === null ? "—" : `${formatNumber(percent, 0)}%`}</span>
      <span className="meter-tooltip" id="context-meter-detail" role="tooltip">
        <strong>{percent === null ? "Context unknown" : `${formatNumber(used, 0)} / ${formatNumber(limit, 0)}`}</strong>
        <small>{context?.behavior === "truncated" ? "Input was truncated" : context?.behavior === "sliding_window" ? "Sliding window active" : "tokens incl. reserve · open the context inspector"}</small>
      </span>
    </button>
  );
}
