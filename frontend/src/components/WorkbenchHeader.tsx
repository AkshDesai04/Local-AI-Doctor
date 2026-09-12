import {
  Bug,
  ChevronDown,
  CircleOff,
  Menu,
  PanelRight,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Square,
} from "lucide-react";
import type { ModelSummary } from "../api/types";

interface WorkbenchHeaderProps {
  models: ModelSummary[];
  selectedModel: ModelSummary | null;
  selectedModelId: string;
  nerdMode: boolean;
  inspectorOpen: boolean;
  connected: boolean;
  onOpenSidebar: () => void;
  onSelectModel: (id: string) => void;
  onRefresh: () => void;
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
  nerdMode,
  inspectorOpen,
  connected,
  onOpenSidebar,
  onSelectModel,
  onRefresh,
  onToggleLoaded,
  onToggleControls,
  onToggleNerd,
  onToggleInspector,
}: WorkbenchHeaderProps): React.ReactNode {
  const busy = selectedModel?.lifecycle === "loading" || selectedModel?.lifecycle === "unloading";
  return (
    <header className="workbench-header">
      <div className="header-leading">
        <button aria-label="Open navigation" className="icon-button mobile-menu-button" onClick={onOpenSidebar} type="button"><Menu size={19} /></button>
        <label className="model-select-wrap">
          <span className={`model-presence ${selectedModel?.lifecycle ?? "none"}`} />
          <span className="visually-hidden">Selected model</span>
          <select disabled={!connected || models.length === 0} onChange={(event) => onSelectModel(event.target.value)} value={selectedModelId}>
            {models.length === 0 && <option value="">No models discovered</option>}
            {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </select>
          <ChevronDown size={14} />
        </label>
        <span className={`model-state-label ${selectedModel?.lifecycle ?? "none"}`}>{modelStatusLabel(selectedModel)}</span>
      </div>
      <div className="header-actions">
        <button aria-label="Refresh models" className="icon-button optional-action" disabled={!connected} onClick={onRefresh} title="Rescan configured model roots" type="button"><RefreshCw size={17} /></button>
        <button
          className={`button load-button ${selectedModel?.lifecycle === "loaded" ? "secondary" : "primary"}`}
          disabled={!selectedModel || busy || !connected}
          onClick={() => { if (selectedModel) onToggleLoaded(selectedModel); }}
          type="button"
        >{selectedModel?.lifecycle === "loaded" ? <Square size={12} /> : selectedModel?.lifecycle === "error" ? <CircleOff size={14} /> : <Play size={14} />}{selectedModel?.lifecycle === "loaded" ? "Unload" : selectedModel?.lifecycle === "error" ? "Retry load" : "Load"}</button>
        <button aria-label="Generation controls" className="header-tool" onClick={onToggleControls} title="Generation controls" type="button"><SlidersHorizontal size={17} /><span>Controls</span></button>
        <button aria-pressed={nerdMode} className={`nerd-toggle ${nerdMode ? "active" : ""}`} onClick={onToggleNerd} type="button"><Bug size={16} /><span>Nerd Mode</span><i /></button>
        <button aria-label="Toggle inspector" aria-pressed={inspectorOpen} className={`icon-button inspector-toggle ${inspectorOpen ? "active" : ""}`} onClick={onToggleInspector} title="Toggle run inspector" type="button"><PanelRight size={18} /></button>
      </div>
    </header>
  );
}
