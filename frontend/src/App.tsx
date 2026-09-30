import { AlertTriangle, CheckCircle2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { ChatView, type NerdMetric } from "./components/ChatView";
import { Composer } from "./components/Composer";
import { ContextMeter } from "./components/ContextMeter";
import { CudaRuntimeNotice } from "./components/CudaRuntimeNotice";
import { EmbeddingsWorkspace } from "./components/EmbeddingsWorkspace";
import { GenerationControls } from "./components/GenerationControls";
import { Inspector, type InspectorTab } from "./components/Inspector";
import { ModelRegistry } from "./components/ModelRegistry";
import { Sidebar, type WorkspaceView } from "./components/Sidebar";
import { WorkbenchHeader } from "./components/WorkbenchHeader";
import { IconButton } from "./components/ui";
import { AUTH_CHANGED_EVENT, AUTH_REQUIRED_EVENT } from "./api/auth";
import { isUsable } from "./domain/capabilities";
import { useWorkbench } from "./hooks/useWorkbench";

export default function App(): React.ReactNode {
  const workbench = useWorkbench();
  const { connected, createChat, loadSelectedRunEvents, models, selectedModel, setSelectedModelId } = workbench;
  const [view, setView] = useState<WorkspaceView>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(() => !window.matchMedia("(max-width: 760px)").matches);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const [nerdMode, setNerdMode] = useState(false);
  const [nerdMetric, setNerdMetric] = useState<NerdMetric>("rawProbability");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("overview");
  const [selectedToken, setSelectedToken] = useState<number | null>(null);

  useEffect(() => {
    setSelectedToken(null);
  }, [workbench.selectedRun?.id]);

  useEffect(() => {
    if (inspectorOpen && nerdMode && inspectorTab === "events") {
      void loadSelectedRunEvents();
    }
  }, [inspectorOpen, inspectorTab, loadSelectedRunEvents, nerdMode]);

  useEffect(() => {
    const requireAuthentication = (): void => {
      setAuthRequired(true);
      setControlsOpen(true);
    };
    const authenticationChanged = (): void => setAuthRequired(false);
    window.addEventListener(AUTH_REQUIRED_EVENT, requireAuthentication);
    window.addEventListener(AUTH_CHANGED_EVENT, authenticationChanged);
    return () => {
      window.removeEventListener(AUTH_REQUIRED_EVENT, requireAuthentication);
      window.removeEventListener(AUTH_CHANGED_EVENT, authenticationChanged);
    };
  }, []);

  useEffect(() => {
    if (view !== "embeddings" || isUsable(selectedModel, "embeddings")) return;
    const embeddingModel = models.find((model) => isUsable(model, "embeddings"));
    if (embeddingModel) setSelectedModelId(embeddingModel.id);
  }, [view, models, selectedModel, setSelectedModelId]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent): void => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "k" || !connected) return;
      event.preventDefault();
      setView("chat");
      setSidebarOpen(false);
      void createChat();
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [connected, createChat]);

  const selectedLoadOptions = workbench.loadOptionsFor(selectedModel?.id ?? "");
  const openInspector = (): void => setInspectorOpen(true);
  const inspectToken = (index: number): void => {
    setSelectedToken(index);
    setInspectorTab("tokens");
  };

  return (
    <div className="app-shell">
      <Sidebar
        activeChatId={workbench.activeChatId}
        archivedChats={workbench.archivedChats}
        chats={workbench.chats}
        connected={workbench.connected}
        onClear={() => void workbench.clearChats()}
        onClose={() => setSidebarOpen(false)}
        onCreateChat={() => void workbench.createChat()}
        onDelete={(id) => void workbench.deleteChat(id)}
        onRename={(id, title) => void workbench.renameChat(id, title)}
        onSelectChat={(id) => void workbench.selectChat(id)}
        onToggleArchive={(chat) => void workbench.toggleArchive(chat)}
        onTogglePin={(chat) => void workbench.togglePin(chat)}
        onViewChange={setView}
        open={sidebarOpen}
        view={view}
      />
      <section className={`workbench-frame ${inspectorOpen ? "inspector-visible" : ""}`}>
        <WorkbenchHeader
          connected={workbench.connected}
          controlsOpen={controlsOpen}
          inspectorOpen={inspectorOpen}
          loadOptions={selectedLoadOptions}
          memory={workbench.memory}
          models={workbench.models}
          nerdMode={nerdMode}
          onOpenRegistry={() => setView("models")}
          onOpenSidebar={() => setSidebarOpen(true)}
          onRefresh={() => void workbench.refreshModels()}
          onSelectModel={workbench.setSelectedModelId}
          onToggleControls={() => setControlsOpen((value) => !value)}
          onToggleInspector={() => setInspectorOpen((value) => !value)}
          onToggleLoaded={(model) => void workbench.toggleModelLoaded(model)}
          onToggleNerd={() => setNerdMode((value) => !value)}
          residents={workbench.residents}
          selectedModel={workbench.selectedModel}
          selectedModelId={workbench.selectedModel?.id ?? workbench.selectedModelId}
        />
        <div className="workbench-body">
          <div className="main-pane">
            <CudaRuntimeNotice connected={workbench.connected} />
            {view === "chat" && (
              <>
                <ChatView
                  booting={workbench.booting}
                  connected={workbench.connected}
                  messages={workbench.messages}
                  messagesLoading={workbench.messagesLoading}
                  model={workbench.selectedModel}
                  nerdMetric={nerdMetric}
                  nerdMode={nerdMode}
                  onBranch={(content, parentMessageId) => void workbench.submit(content, parentMessageId)}
                  onInspectRun={(id) => void workbench.inspectRun(id)}
                  onNavigate={setView}
                  onNerdMetricChange={setNerdMetric}
                  onOpenInspector={openInspector}
                  onRetry={(message) => void workbench.retryMessage(message)}
                  onSelectToken={inspectToken}
                  run={workbench.selectedRun}
                  runningRunId={workbench.runningRunId}
                  selectedToken={selectedToken}
                  streamConnected={workbench.streamConnected}
                  systemPrompt={workbench.systemPrompt}
                />
                <Composer
                  attachments={workbench.attachments}
                  busy={workbench.branching}
                  connected={workbench.connected}
                  contextMeter={(
                    <ContextMeter
                      context={workbench.selectedRun?.metrics?.context}
                      fallbackLimit={workbench.selectedModel?.effectiveContextLimit ?? null}
                      onOpen={() => { setInspectorTab("context"); setInspectorOpen(true); }}
                    />
                  )}
                  model={workbench.selectedModel}
                  onAttach={(file) => void workbench.addAttachment(file)}
                  onOpenControls={() => setControlsOpen(true)}
                  onRemoveAttachment={workbench.removeAttachment}
                  onSettingsChange={workbench.setSettings}
                  onStop={() => void workbench.stop()}
                  onSubmit={(content) => void workbench.submit(content)}
                  onSystemPromptChange={workbench.setSystemPrompt}
                  running={Boolean(workbench.runningRunId)}
                  settings={workbench.settings}
                  systemPrompt={workbench.systemPrompt}
                />
              </>
            )}
            {view === "embeddings" && (
              <EmbeddingsWorkspace
                connected={workbench.connected}
                models={workbench.models}
                onRuntimeStateChange={workbench.synchronizeRuntimeState}
                onSelectModel={workbench.setSelectedModelId}
                selectedModelId={workbench.selectedModel?.id ?? workbench.selectedModelId}
              />
            )}
            {view === "models" && (
              <ModelRegistry
                connected={workbench.connected}
                loadOptionsFor={workbench.loadOptionsFor}
                maxLoadedModels={workbench.maxLoadedModels}
                memory={workbench.memory}
                models={workbench.models}
                onLoad={(model, options) => void workbench.loadModel(model, options)}
                onLoadOptionsChange={workbench.setLoadOptions}
                onRefresh={() => void workbench.refreshModels()}
                onSelectModel={workbench.setSelectedModelId}
                onSynchronize={() => void workbench.synchronizeRuntimeState()}
                onUnloadAll={() => void workbench.unloadAll()}
                onUnloadResident={(resident) => void workbench.unloadResident(resident)}
                residents={workbench.residents}
              />
            )}
          </div>
          <Inspector
            activeTab={inspectorTab}
            branching={workbench.branching}
            configuration={workbench.configuration}
            health={workbench.health}
            model={workbench.selectedModel}
            nerdMode={nerdMode}
            onBranchAlternative={workbench.branchFromAlternative}
            onClose={() => setInspectorOpen(false)}
            onSelectToken={setSelectedToken}
            onTabChange={setInspectorTab}
            open={inspectorOpen}
            run={workbench.selectedRun}
            selectedToken={selectedToken}
          />
        </div>
        <GenerationControls
          authRequired={authRequired}
          defaultSettings={workbench.defaultSettings}
          loadOptions={selectedLoadOptions}
          model={workbench.selectedModel}
          onChange={workbench.setSettings}
          onClose={() => setControlsOpen(false)}
          onLoadOptionsChange={(options) => { if (selectedModel) workbench.setLoadOptions(selectedModel.id, options); }}
          open={controlsOpen}
          settings={workbench.settings}
        />
      </section>
      {workbench.error && (
        <div className="toast error" role="alert"><AlertTriangle size={16} /><span>{workbench.error}</span><IconButton icon={<X size={14} />} label="Dismiss error" onClick={() => workbench.setError(null)} size="sm" /></div>
      )}
      {workbench.notice && !workbench.error && (
        <div className="toast notice" role="status"><CheckCircle2 size={16} /><span>{workbench.notice}</span></div>
      )}
    </div>
  );
}
