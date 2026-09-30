import { useEffect, useRef, useState } from "react";
import type { TokenEvent } from "../api/types";
import { displayTokenText, formatDuration, formatNumber, formatPercent, tokenTextHint } from "../utils/format";

interface VirtualTokenTableProps {
  tokens: TokenEvent[];
  selectedToken: number | null;
  onSelectToken: (index: number) => void;
}

const rowHeight = 32;
const viewportHeight = 360;
const overscan = 6;

export function VirtualTokenTable({ tokens, selectedToken, onSelectToken }: VirtualTokenTableProps): React.ReactNode {
  const [scrollTop, setScrollTop] = useState(0);
  const viewport = useRef<HTMLDivElement>(null);
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const count = Math.ceil(viewportHeight / rowHeight) + overscan * 2;
  const end = Math.min(tokens.length, start + count);
  const visible = tokens.slice(start, end);

  useEffect(() => {
    if (selectedToken === null || !viewport.current) return;
    const selectedPosition = tokens.findIndex((token) => token.index === selectedToken);
    if (selectedPosition < 0) return;
    const top = selectedPosition * rowHeight;
    const currentTop = viewport.current.scrollTop;
    if (top < currentTop || top + rowHeight > currentTop + viewportHeight) {
      viewport.current.scrollTo({ top: Math.max(0, top - viewportHeight / 2), behavior: "smooth" });
    }
  }, [selectedToken, tokens]);

  if (!tokens.length) return <p className="muted card-inset">Token events were not captured for this run.</p>;
  return (
    <div className="token-table-shell">
      <div className="data-table-head token-table-grid" role="row">
        <span title="Generation order">#</span><span title="Human-readable token text">Piece</span><span title="Tokenizer vocabulary ID">ID</span><span title="Probability before sampling filters">Model p</span><span title="Probability after sampling filters">Sampler p</span><span title="Raw model probability rank">Rank</span><span title="Time spent decoding this token">Latency</span>
      </div>
      <div
        aria-label="Token event table"
        className="token-table-viewport"
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        ref={viewport}
        role="table"
        style={{ height: viewportHeight }}
        tabIndex={0}
      >
        <div style={{ height: tokens.length * rowHeight, position: "relative" }}>
          {visible.map((token, localIndex) => (
            <button
              aria-selected={selectedToken === token.index}
              className={`data-table-row token-table-grid token-table-row selectable ${selectedToken === token.index ? "selected" : ""}`}
              key={token.index}
              onClick={() => onSelectToken(token.index)}
              role="row"
              style={{ height: rowHeight, top: (start + localIndex) * rowHeight }}
              title={`${tokenTextHint(token.piece, token.displayText)}\nClick to inspect this token.`}
              type="button"
            >
              <span>{String(token.index)}</span>
              <span className="mono token-piece">{displayTokenText(token.displayText || token.piece) || "∅"}</span>
              <span className="mono">{String(token.tokenId)}</span>
              <span>{formatPercent(token.rawProbability, 3)}</span>
              <span>{formatPercent(token.samplingProbability, 3)}</span>
              <span>{formatNumber(token.rawRank, 0)}</span>
              <span>{formatDuration(token.timing?.decodeMs)}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
