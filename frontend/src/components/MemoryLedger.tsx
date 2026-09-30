import { Gauge, Square } from "lucide-react";
import { useState } from "react";
import type { MemoryLedger, ModelSummary, ResidentModel } from "../api/types";
import { isCudaLedger, type LedgerSegment, ledgerSegments } from "../domain/residency";
import { formatBytes, formatNumber } from "../utils/format";
import { ConfirmDialog } from "./ConfirmDialog";
import { Badge, Button, Card } from "./ui";

interface MemoryLedgerCardProps {
  memory: MemoryLedger | null;
  residents: ResidentModel[];
  models: ModelSummary[];
  maxLoadedModels: number | null;
  connected: boolean;
  onUnloadAll: () => void;
}

function segmentStyle(segment: LedgerSegment): React.CSSProperties {
  return { "--swatch": segment.color ? `var(--viz-${String(segment.color)})` : undefined } as React.CSSProperties;
}

function segmentText(segment: LedgerSegment): string {
  return `${segment.label}: ${formatBytes(segment.bytes)} (${formatNumber(segment.percent, 1)}%) · ${segment.detail}`;
}

function LedgerBar({ memory, segments }: { memory: MemoryLedger; segments: LedgerSegment[] }): React.ReactNode {
  const [active, setActive] = useState<string | null>(null);
  const activeSegment = segments.find((segment) => segment.id === active);
  const total = memory.totalBytes ?? 0;
  const capPercent = memory.capBytes && total && memory.capBytes < total ? (memory.capBytes / total) * 100 : null;
  return (
    <div className="ledger" onMouseLeave={() => setActive(null)}>
      <div aria-hidden="true" className={`ledger-bar ${active ? "has-active" : ""}`}>
        {segments.map((segment) => (
          <span
            className={`ledger-seg ${segment.kind} ${segment.id === active ? "active" : ""}`}
            key={segment.id}
            onMouseEnter={() => setActive(segment.id)}
            style={{ ...segmentStyle(segment), width: `${String(segment.percent)}%` }}
            title={segmentText(segment)}
          />
        ))}
        {capPercent !== null && <span className="ledger-cap" style={{ left: `${String(capPercent)}%` }} title={`Allocator cap for Strict VRAM: ${formatBytes(memory.capBytes)}`} />}
      </div>
      <ul aria-label="GPU memory breakdown" className="ledger-legend">
        {segments.map((segment) => (
          <li
            className={segment.id === active ? "active" : ""}
            key={segment.id}
            onBlur={() => setActive(null)}
            onFocus={() => setActive(segment.id)}
            onMouseEnter={() => setActive(segment.id)}
            tabIndex={0}
            title={segment.detail}
          >
            <i aria-hidden="true" className={`ledger-swatch ${segment.kind}`} style={segmentStyle(segment)} />
            <span className="ledger-legend-label">{segment.label}</span>
            <strong>{formatBytes(segment.bytes)}</strong>
          </li>
        ))}
      </ul>
      <p className="ledger-detail">{activeSegment ? segmentText(activeSegment) : "Hover or focus a segment for details."}</p>
    </div>
  );
}

/** Device memory by resident, PyTorch cache, safety margin, and free space, with the resident count and Unload all. */
export function MemoryLedgerCard({ memory, residents, models, maxLoadedModels, connected, onUnloadAll }: MemoryLedgerCardProps): React.ReactNode {
  const [confirming, setConfirming] = useState(false);
  const cuda = isCudaLedger(memory);
  const segments = cuda ? ledgerSegments(memory, residents, models) : null;
  const used = cuda && memory.freeBytes !== null ? (memory.totalBytes ?? 0) - memory.freeBytes : null;
  const loadable = cuda && memory.freeBytes !== null
    ? Math.max(0, memory.freeBytes + Math.max(0, (memory.torchReservedBytes ?? 0) - (memory.torchAllocatedBytes ?? 0)) - (memory.safetyMarginBytes ?? 0))
    : null;
  const count = `${String(residents.length)}${maxLoadedModels ? ` of ${String(maxLoadedModels)}` : ""} resident`;
  const age = memory?.ledgerAgeSeconds;
  return (
    <Card
      actions={(
        <>
          <Badge title={maxLoadedModels ? `At most ${String(maxLoadedModels)} models stay resident at once` : undefined} tone={residents.length ? "accent" : "neutral"}>{count}</Badge>
          <Button disabled={!connected || residents.length === 0} icon={<Square size={11} />} onClick={() => setConfirming(true)} size="sm">Unload all</Button>
        </>
      )}
      aria-label="Resident memory"
      className="ledger-card"
      description={(
        <>
          {memory ? memory.device : "Not reported by this backend"}
          {age !== null && age !== undefined ? ` · measured ${formatNumber(age, 1)} s ago` : ""}
          {memory?.stale && <Badge className="ledger-stale" title="The worker was busy, so this is the last measurement it reported." tone="warning">Stale</Badge>}
        </>
      )}
      icon={Gauge}
      title={cuda ? "GPU memory" : "Memory"}
    >
      {cuda && segments ? (
        <>
          <p className="ledger-summary">
            <span><strong>{formatBytes(used)}</strong> of <strong>{formatBytes(memory.totalBytes)}</strong> in use</span>
            <span><strong>{formatBytes(loadable)}</strong> available to the next load</span>
            {memory.capBytes !== null && <span>Strict VRAM cap <strong>{formatBytes(memory.capBytes)}</strong></span>}
          </p>
          <LedgerBar memory={memory} segments={segments} />
        </>
      ) : (
        <p className="ledger-summary">
          {memory
            ? <><span>No CUDA device ledger; models run on the CPU.</span>{memory.processRssBytes !== null && <span>Process RSS <strong>{formatBytes(memory.processRssBytes)}</strong></span>}{memory.systemAvailableBytes !== null && <span>System RAM available <strong>{formatBytes(memory.systemAvailableBytes)}</strong></span>}</>
            : <span>This backend does not report a memory ledger.</span>}
        </p>
      )}
      <ConfirmDialog
        confirmLabel="Unload all"
        description={`Unload ${String(residents.length)} resident ${residents.length === 1 ? "copy" : "copies"} and free their GPU and system memory. Chats and run history are kept.`}
        onCancel={() => setConfirming(false)}
        onConfirm={() => { setConfirming(false); onUnloadAll(); }}
        open={confirming}
        title="Unload every resident model?"
      />
    </Card>
  );
}
