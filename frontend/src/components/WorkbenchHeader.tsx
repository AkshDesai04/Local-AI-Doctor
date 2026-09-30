import { Bug, CircleOff, Database, Ellipsis, Menu, PanelRight, Play, RefreshCw, SlidersHorizontal, Square } from "lucide-react";
import type { GenerationSettings, ModelSummary } from "../api/types";
import { Button, IconButton, MenuButton, MenuItem, Select, Switch } from "./ui";

interface WorkbenchHeaderProps {
  models: ModelSummary[];
  selectedModel: ModelSummary | null;
  selectedModelId: string;
  loadOptions: Pick<GenerationSettings, "device" | "dtype">;
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

function modelStatusLabel(model: ModelSummary | null): string {
  if (!model) return "No model";
  if (model.lifecycle === "loaded") return `${model.loadedDevice ?? "device"} · ready`;
  if (model.lifecycle === "loading") return "Loading…";
  if (model.lifecycle === "unloading") return "Unloading…";
  if (model.lifecycle === "error") return "Load error";
  return "Not loaded";
}

export function WorkbenchHeader({
  models,
  selectedModel,
  selectedModelId,
  loadOptions,
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
  const busy = lifecycle === "loading" || lifecycle === "unloading";
  const loaded = lifecycle === "loaded";
  const loadLabel = loaded ? "Unload" : lifecycle === "error" ? "Retry load" : busy ? (lifecycle === "loading" ? "Loading…" : "Unloading…") : "Load";
  const loadHint = !connected
    ? "Start the local backend first"
    : !selectedModel
      ? "Select a model first"
      : loaded
        ? `Unload ${selectedModel.name} and free its memory`
        : `Load ${selectedModel.name} with device ${loadOptions.device} and dtype ${loadOptions.dtype} (Generation controls › Model loading)`;
  const toggleLoaded = (): void => { if (selectedModel) onToggleLoaded(selectedModel); };

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
            {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </Select>
          <span className={`model-status ${lifecycle}`} title={selectedModel?.lifecycle === "loaded" ? `Loaded on ${selectedModel.loadedDevice ?? "an unreported device"}` : modelStatusLabel(selectedModel)}>
            <i aria-hidden="true" /><span className="model-status-text">{modelStatusLabel(selectedModel)}</span>
          </span>
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
