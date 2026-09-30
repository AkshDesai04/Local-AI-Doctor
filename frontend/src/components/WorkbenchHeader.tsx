import { Bug, CircleOff, Database, Ellipsis, Menu, PanelRight, Play, RefreshCw, SlidersHorizontal, Square } from "lucide-react";
import type { LoadOptions, MemoryLedger, ModelSummary, ResidentModel } from "../api/types";
import { isCudaLedger, residentsOf, residentSummary } from "../domain/residency";
import { formatBytes, formatNumber } from "../utils/format";
import { Button, IconButton, MenuButton, MenuItem, Select, Switch } from "./ui";

interface WorkbenchHeaderProps {
  models: ModelSummary[];
  selectedModel: ModelSummary | null;
  selectedModelId: string;
  /** The selected model's saved load options, used by Load. */
  loadOptions: LoadOptions;
  /** Every resident copy of every model. */
  residents: ResidentModel[];
  memory: MemoryLedger | null;
  nerdMode: boolean;
  inspectorOpen: boolean;
  controlsOpen: boolean;
  connected: boolean;
  onOpenSidebar: () => void;
  onSelectModel: (id: string) => void;
  onRefresh: () => void;
  onOpenRegistry: () => void;
  onToggleLoaded: (model: ModelSummary) => void;
  onToggleControls: () => void;
  onToggleNerd: () => void;
  onToggleInspector: () => void;
}

function residentLabel(resident: ResidentModel): string {
  const label = resident.dtype ? residentSummary(resident) : `${resident.device} · ready`;
  return resident.placement === "offload" ? `${label} · offload` : label;
}

function modelStatusLabel(model: ModelSummary | null, own: ResidentModel[]): string {
  if (!model) return "No model";
  if (model.lifecycle === "loading") return "Loading…";
  if (model.lifecycle === "unloading") return "Unloading…";
  if (own.length > 1) return `${String(own.length)} resident`;
  if (own[0]) return residentLabel(own[0]);
  if (model.lifecycle === "loaded") return `${model.loadedDevice ?? "device"} · ready`;
  if (model.lifecycle === "error") return "Load error";
  return "Not loaded";
}

function residentDetail(resident: ResidentModel): string {
  const placement = resident.placement === "offload" ? "offloaded to system RAM" : resident.placement === "cpu" ? "CPU" : resident.placement === "gpu" ? "GPU" : null;
  return [resident.device, resident.dtype, placement, resident.strictVram ? "Strict VRAM" : null].filter(Boolean).join(" · ");
}

/** Compact used/total GPU memory; rendered only for a CUDA ledger. */
function VramMiniBar({ memory, residentCount }: { memory: MemoryLedger | null; residentCount: number }): React.ReactNode {
  if (!isCudaLedger(memory) || memory.freeBytes === null) return null;
  const total = memory.totalBytes ?? 0;
  const used = Math.max(0, total - memory.freeBytes);
  const share = Math.min(100, (used / total) * 100);
  const text = `${formatBytes(used)} of ${formatBytes(total)}`;
  const title = `GPU memory on ${memory.device}: ${text} in use (${formatNumber(share, 0)}%) · ${String(residentCount)} resident ${residentCount === 1 ? "copy" : "copies"}${memory.stale ? " · last reported measurement" : ""}`;
  return (
    <span className={`vram-mini ${share >= 90 ? "high" : ""}`} title={title}>
      <span aria-label="GPU memory in use" aria-valuemax={total} aria-valuemin={0} aria-valuenow={used} aria-valuetext={text} className="vram-mini-track" role="meter">
        <span className="vram-mini-fill" style={{ width: `${String(share)}%` }} />
      </span>
      <span aria-hidden="true" className="vram-mini-text">{formatNumber(used / 1024 ** 3, 1)}/{formatNumber(total / 1024 ** 3, 0)} GiB</span>
    </span>
  );
}

export function WorkbenchHeader({
  models,
  selectedModel,
  selectedModelId,
  loadOptions,
  residents,
  memory,
  nerdMode,
  inspectorOpen,
  controlsOpen,
  connected,
  onOpenSidebar,
  onSelectModel,
  onRefresh,
  onOpenRegistry,
  onToggleLoaded,
  onToggleControls,
  onToggleNerd,
  onToggleInspector,
}: WorkbenchHeaderProps): React.ReactNode {
  const lifecycle = selectedModel?.lifecycle ?? "none";
  const own = selectedModel ? residentsOf(residents, selectedModel.id) : [];
  const busy = lifecycle === "loading" || lifecycle === "unloading";
  const loaded = lifecycle === "loaded";
  const status = modelStatusLabel(selectedModel, own);
  const statusTitle = own.length
    ? `${own.length === 1 ? "Resident" : `${String(own.length)} resident copies`}: ${own.map(residentDetail).join("; ")}`
    : status;
  const loadLabel = loaded ? "Unload" : lifecycle === "error" ? "Retry load" : busy ? (lifecycle === "loading" ? "Loading…" : "Unloading…") : "Load";
  const loadHint = !connected
    ? "Start the local backend first"
    : !selectedModel
      ? "Select a model first"
      : loaded
        ? `Unload every resident copy of ${selectedModel.name} and free its memory`
        : `Load ${selectedModel.name} with device ${loadOptions.device}, dtype ${loadOptions.dtype}, and Strict VRAM ${loadOptions.strictVram && loadOptions.device !== "cpu" ? "on" : "off"} (Model registry or Generation controls › Model loading)`;
  const toggleLoaded = (): void => { if (selectedModel) onToggleLoaded(selectedModel); };
  const residentIds = new Set(residents.map((resident) => resident.modelId));
  const residentModels = models.filter((model) => residentIds.has(model.id));
  const idleModels = models.filter((model) => !residentIds.has(model.id));

  return (
    <header className="workbench-header">
      <div className="header-leading">
        <IconButton className="mobile-menu-button" icon={<Menu size={18} />} label="Open navigation" onClick={onOpenSidebar} />
        <div className="model-picker">
          <label className="visually-hidden" htmlFor="header-model-select">Selected model</label>
          <Select
            disabled={!connected || models.length === 0}
            id="header-model-select"
            onChange={(event) => onSelectModel(event.target.value)}
            title={selectedModel?.name}
            value={selectedModelId}
            variant="ghost"
            wrapperClassName="model-select"
          >
            {models.length === 0 && <option value="">No models discovered</option>}
            {residentModels.length > 0 ? (
              <>
                <optgroup label="Resident">{residentModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</optgroup>
                {idleModels.length > 0 && <optgroup label="Not loaded">{idleModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</optgroup>}
              </>
            ) : models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </Select>
          <span className={`model-status ${lifecycle} ${own.some((resident) => resident.placement === "offload") ? "offload" : ""}`} title={statusTitle}>
            <i aria-hidden="true" /><span className="model-status-text">{status}</span>
          </span>
          <VramMiniBar memory={memory} residentCount={residents.length} />
        </div>
      </div>
      <div className="header-actions">
        <Button
          className="header-load"
          disabled={!selectedModel || !connected}
          icon={loaded ? <Square size={11} /> : lifecycle === "error" ? <CircleOff size={14} /> : <Play size={13} />}
          loading={busy}
          onClick={toggleLoaded}
          size="sm"
          title={loadHint}
          variant={loaded ? "secondary" : "primary"}
        >{loadLabel}</Button>
        <span aria-hidden="true" className="header-divider" />
        <Button aria-expanded={controlsOpen} aria-label="Generation controls" className="header-controls" icon={<SlidersHorizontal size={15} />} onClick={onToggleControls} size="sm" title="Generation controls" variant="ghost"><span className="header-controls-text">Controls</span></Button>
        <Switch
          checked={nerdMode}
          className="nerd-switch"
          label={<><Bug aria-hidden="true" size={14} /><span className="nerd-label-text">Nerd Mode</span></>}
          onChange={onToggleNerd}
          title="Show raw tokens, alternatives, and protocol events"
        />
        <IconButton icon={<PanelRight size={17} />} label="Toggle inspector" onClick={onToggleInspector} pressed={inspectorOpen} title="Toggle run inspector" />
        <MenuButton icon={<Ellipsis size={17} />} label="More actions">
          <MenuItem className="narrow-only" disabled={!selectedModel || !connected || busy} icon={loaded ? <Square size={13} /> : <Play size={14} />} onSelect={toggleLoaded} title={loadHint}>{loadLabel === "Load" ? "Load model" : loadLabel === "Unload" ? "Unload model" : loadLabel}</MenuItem>
          <MenuItem disabled={!connected} icon={<RefreshCw size={14} />} onSelect={onRefresh} title="Rescan configured model roots">Rescan model roots</MenuItem>
          <MenuItem icon={<Database size={14} />} onSelect={onOpenRegistry}>Open model registry</MenuItem>
        </MenuButton>
      </div>
    </header>
  );
}
