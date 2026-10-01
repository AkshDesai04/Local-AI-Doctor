# Local AI Doctor agent guide

This file is the repository-level operating guide for coding agents and maintainers working on Local AI Doctor. It consolidates the product brief, the owner's decisions from the project conversation, and the current implementation contract. Read it before changing code, configuration, deployment, tests, or releases.

The filename is intentionally `AGENT.md` because that is the artifact requested by the project owner.

## 1. How to interpret this guide

Use these labels mentally when reading the document:

- **Current behavior** describes code that exists now and should be preserved unless the task changes it.
- **Required invariant** describes an owner decision that must not regress, even when the implementation is refactored.
- **Capability-gated goal** describes a supported product direction that may still be partial or unavailable for a particular model/backend.
- **Known limitation** describes something the project must report honestly rather than simulate.

When information disagrees:

1. Follow the latest explicit request from the project owner.
2. Preserve security, data-integrity, privacy, and legal constraints.
3. Treat tests and executable source as the authority for current behavior.
4. Treat this guide as the authority for product intent and non-negotiable decisions.
5. Consult the focused documents under `docs/` for full API, metric, deployment, and model-evidence details.

Do not silently convert a requirement into a claim that it is already implemented. Update this file when a change materially alters architecture, commands, persistent data, public contracts, user-visible behavior, or release policy.

## 2. Project identity and purpose

Local AI Doctor is a local-first inference and observability workbench for model directories containing SafeTensors checkpoints and their associated configuration, tokenizer, processor, template, and pooling metadata.

Despite its name, this is a model diagnostics and experimentation tool. It is **not** a medical product and must not be represented or used as a diagnostic or clinical system.

The product is intended to provide:

- Local text generation for supported causal and encoder-decoder models.
- Local text, image, video, and mixed-input embeddings for models whose processors genuinely expose those modalities.
- Evidence-based discovery, model fingerprints, diagnostics, and capability matrices.
- Persistent ChatGPT-style conversations with branching and replay.
- Deep token, sampling, timing, context, hardware, reasoning, and optional routing observability.
- CPU execution everywhere it is feasible and CUDA acceleration when genuinely available.
- Native Windows/WSL execution, hardened WSL-hosted Docker deployment, and an installed Windows Electron application.
- An adapter-oriented path for adding architectures without pretending that unknown models are supported.

Normal inference must run locally. Never add a hidden or automatic external inference provider. Dependency installation and container/image downloads are build-time network activity; they are not permission to transmit prompts, outputs, uploads, telemetry, or model data during normal operation.

## 3. Owner decisions that must not regress

The following requirements came directly from the project conversation and are binding unless the owner explicitly changes them.

### 3.1 Ports and startup ordering

- Docker and Electron production paths permanently use frontend `127.0.0.1:6969` and backend/API/WebSocket `127.0.0.1:6767`.
- Do not make those production ports configurable and do not work around conflicts by selecting different ports.
- Native CLI development may continue to use backend port `8000`; Vite development may continue to use frontend port `5173`.
- The backend must become fully ready first. Require the health contract to report `status: ok`, `database: ready`, and `worker: ready`; only then start the frontend. Do not add a fixed stabilization timer after readiness.
- Preserve both readiness gates: Compose dependency health ordering and the frontend entrypoint's independent complete-readiness check. Electron may show its native startup window immediately, but it must perform equivalent ordering before starting or displaying the production frontend.
- If a WSL Docker container owns port 6767 or 6969, identify the exact owner and stop only that conflicting container before starting the requested profile. Do not kill an arbitrary process or change the ports.
- Electron intentionally refuses to start if either port is occupied; it must not kill the owner automatically.

### 3.2 Local models, configuration, and hardware

- Model roots are user-configured and external. Model roots are read-only except for the explicit, user-initiated Flush-to-storage action, which writes a new sibling folder atomically and never modifies existing files. Never modify, relocate, delete, or copy model weights into this repository, a release, or a container image.
- Load-time quantization (bitsandbytes NF4 or LLM.int8) happens in GPU memory only; nothing is written to disk until the user presses Flush to storage. Docker keeps `/models` read-only, so flushing there fails with `model_root_read_only`.
- The current workstation has a separate host model directory, but its absolute Windows path, WSL path, user name, and WSL distribution are machine-local facts. Keep them only in ignored local configuration. Portable code may use the container-internal `/models` path.
- The Model registry settings UI is the user-facing wrapper for model-root configuration. It must persist backend-visible absolute paths into the writable user-local config and rescan.
- Choosing CUDA must result in actual CUDA execution or a clear failure. The CPU Docker profile must never accept an explicit CUDA request and silently run it on CPU.
- Verify CUDA with the installed PyTorch build, `torch.cuda.is_available()`, the selected device recorded by the backend, container smoke checks, and run metadata. A UI dropdown alone is not evidence.
- One worker process, N resident models bounded by VRAM and `runtime.max_loaded_models`, one admitted job at a time (a comparison is one job). Loading a model that does not fit evicts least-recently-used idle residents first; residents the running job uses are never evicted. If it still does not fit, Strict VRAM (the default, chosen per load) fails clearly with a 507, and only an explicit Strict VRAM off may offload layers to system RAM. Unload releases every reference and backend cache. Do not kill unrelated GPU processes just because VRAM is in use.

### 3.3 Reasoning and response presentation

- In normal mode, emitted reasoning appears in a collapsed disclosure labeled `Thinking…`; the final answer remains outside it.
- Reasoning start/end markers and termination tokens are visible only in Nerd Mode. Normal mode hides protocol markers from display and from copied answer text.
- Nerd Mode preserves the raw content, token boundaries, reasoning markers, and termination token visibility needed for inspection.
- Parsing must handle delimiters split across token pieces, prompt-primed reasoning, mixed delimiter/answer tokens, Unicode code-point slices, and tokenizer boundary revisions. Do not revert to naive string splitting.
- A completed response that contains reasoning but no answer must never look like a successful blank response. The backend may use one context-clipped extra answer window for explicitly tagged reasoning models; the UI must show an honest warning if no answer still exists.

### 3.4 Token inspection, branching, and live telemetry

- Token labels must not expose unexplained tokenizer whitespace glyphs before visible text. Use the shared display formatter while preserving the exact raw piece in metadata/tooltips.
- Token chips have a fixed visual height; whitespace, newline, or fallback tokens must not stretch the row.
- The `Inspect run` strip and inspector update while a response streams: token count, running tokens/second, running perplexity when available, live/reconnecting state, and terminal status.
- Opening an existing chat must restore persisted run telemetry. Inspecting a response must load that response's durable run rather than losing statistics on navigation.
- Hover hints must explain unfamiliar controls, disabled capability reasons, token encodings, alternatives, and model-root behavior.
- Before sending, the composer-side popover offers reasoning, temperature, top-k, and top-p controls for the next response. Full controls continue to expose all supported generation settings.
- In Nerd Mode, selecting a generated token exposes raw-model and post-sampler alternatives as distinct distributions.
- Choosing an eligible non-current alternative offers `Branch out with selected token`. The new continuation uses the original response prefix through that position, forces the selected alternative, continues generation, and is saved as a **new chat**. Never mutate the source chat or run.

### 3.5 Attention attribution

- For compatible decoder-only models at `full` or `expert` instrumentation, clicking a generated token can show earlier rendered-prompt, conversation-history, and generated-prefix tokens retained by the model's causal self-attention summary.
- The selected target is highlighted, unrelated tokens are muted, and retained source tokens use intensity proportional to their stored mean attention weight. The most strongly weighted retained source is brightest.
- Hovering a source shows the mathematical value, percentage, rank, token identity, context index, and source kind.
- This feature must always be described as **post-softmax self-attention allocation averaged across captured layers and heads**. It is not causal attribution, not proof of grounding, not the probability that a source caused the selected token, and not a hallucination detector by itself.
- The attention row belongs to the prediction step/prefix and does not depend on which vocabulary candidate the sampler ultimately selected. Residual paths, value-vector content, MLPs, normalization, and the output projection also affect the result.

### 3.6 Interface design

- The canonical primary accent is blue (`#60a5fa` in the current theme), not green.
- Important UI text must remain readable. Do not reintroduce microscopic labels; dense telemetry generally has a 10–11 px floor and answer text is larger.
- Preserve keyboard access, focus-visible outlines, ARIA state, status/alert semantics, reduced-motion support, and narrow/mobile layouts.
- Unsupported features remain visible when useful but disabled with a precise reason. Never populate them with fake or demonstration telemetry in production.

### 3.7 Desktop and releases

- The desktop distribution is Electron and produces one per-user Windows installer `.exe`, not a model bundle. Installation extracts the runtime once; subsequent launches use the installed payload.
- Root `VERSION` is the stable `X.Y.Z` source of truth. Python, frontend, desktop, and lock manifests must stay synchronized through `scripts/release-version.mjs`.
- An ordinary completed feature or fix receives a patch bump. Use a minor or major bump only when the owner explicitly asks for that release level.
- A successful `dev` push creates a beta prerelease `vX.Y.Z-beta.<workflow-run>` with a matching beta-suffixed EXE.
- A successful `main` push creates the immutable stable `vX.Y.Z` release. Promotion normally occurs by merging the tested `dev` revision into `main`.
- The release workflow uploads exactly one custom asset: `Local-AI-Doctor-<version>.exe`. GitHub's automatic source ZIP/TAR links cannot be removed and are not custom release assets.
- The GitHub-built EXE is CPU-only because the complete CUDA runtime exceeds GitHub's 2 GiB asset limit. A local EXE built from the pinned CUDA environment may use CUDA; native and Docker CUDA paths remain available.
- The EXE is currently unsigned and may trigger SmartScreen. Installation can take time because the local CUDA payload is large, but normal launches must not re-extract it.

### 3.8 Git and delivery

- Normal development work belongs on `dev`. Do not create, switch, rebase, merge, push, or force-push another branch unless the owner explicitly requests the relevant operation. `main` is the stable release branch.
- Create coherent, tested milestone commits and push validated work to the configured `dev` upstream when the task authorizes project delivery.
- Use only the owner's existing Git identity and authentication. Never change Git identity, expose credentials, add AI/generated-by text, or add co-author trailers.
- Keep model weights, personal config, credentials, databases, uploads, caches, environments, logs, release artifacts, and large telemetry out of Git.
- Do not select or add an open-source license without explicit owner instruction.
- Update the root `README.md` in the same change whenever code, configuration, deployment, installation, startup behavior, user workflow, or release behavior changes. Keep its commands and user-facing claims synchronized with the implementation.

### 3.9 Docker/WSL storage hygiene

- All Docker work on the Windows workstation runs through WSL2 and Linux Docker, never Windows `docker.exe`.
- Preserve named application volumes and the external model bind. Do not use blanket `docker system prune --volumes`.
- A prior incident grew the WSL virtual disk beyond 230 GiB, primarily through build cache and obsolete Docker artifacts. Inspect Docker disk use regularly and remove proven-unused builder cache, dangling images, and exact obsolete stopped containers as appropriate.
- Deleting files inside WSL does not necessarily shrink the Windows VHD. Compaction or export/import is a separate, potentially destructive operation requiring exact-target verification, adequate free space, a verified backup/export, and explicit authorization.

## 4. Current implementation status

The current 0.1-series implementation includes:

- Bounded read-only model discovery and SafeTensors header validation.
- Stable directory fingerprints and complete reason-bearing capability matrices.
- Reference causal and generic encoder-decoder generation loops with cache reuse.
- SentenceTransformers-based embedding execution for supported inputs.
- One spawned model worker holding several resident models (least-recently-used eviction, Strict VRAM preflight and allocator cap, optional layer offload), round-robin generation sessions within one admitted job, and serialized job admission.
- CPU and CUDA selection with explicit device/dtype reporting.
- Streaming REST/WebSocket run orchestration, cancellation, replay, and persisted partial results.
- Tiered token telemetry, raw and sampler distributions, timing, perplexity, and bounded alternatives.
- Bounded causal self-attention summaries for compatible decoder-only `full`/`expert` runs.
- Reasoning-tag segmentation and an extra-answer-window recovery path for supported tagged models.
- SQLite chats, branches, runs, events, tokens, alternatives, attention data, attachments, and embedding records.
- A responsive React workbench with chat, model registry, embeddings, live inspector, Nerd Mode, token branching, settings, and durable run restoration.
- Hardened CPU/NVIDIA Compose profiles and an Electron installer build/release pipeline.

Do not overstate these boundaries:

- Discovery is not proof that a checkpoint can execute.
- The adapter registry contracts exist, but the running worker still owns explicit built-in Transformers/SentenceTransformers paths.
- Only one worker process and one active/runnable job are supported; the worker holds up to `max_loaded_models` residents and interleaves up to `max_concurrent_runs` generation sessions inside that job. `queue_limit` controls additional waiters.
- ROCm, Metal, and general multi-GPU placement are not implemented runtime paths. Layer offload to system RAM exists only for generation models on CUDA with Strict VRAM off.
- Weight quantization is bitsandbytes only (`bitsandbytes-4bit` NF4, `bitsandbytes-8bit` LLM.int8), at load time, for decoder-only text generators on CUDA with the CUDA extra installed. Quantized outputs differ from the checkpoint dtype. Pre-quantized bitsandbytes checkpoints load as stored and cannot be re-quantized; the legacy `int4`/`int8` values are rejected.
- MoE schemas/interfaces exist, but no production router hook is currently composed. The supplied checkpoints are dense.
- Hidden-state probes, activation probes, logit lens, KV-cache inspection, and continuous hardware sampling are not implemented production telemetry.
- Attention capture is partial, bounded, decoder-only, and unavailable when the model cannot return alignable eager attention tensors.
- Audio embedding is unsupported for the supplied Qwen checkpoint. PDF/text extraction is not an implemented upload adapter.
- Nearest-neighbor indexing and persisted PCA/UMAP are not implemented services.
- The application is single-user and loopback-oriented, not multi-tenant or internet-ready.

The canonical limitation list is `docs/limitations.md`. When adding a capability, update discovery, runtime execution, API types, UI gating, tests, documentation, and the model capability report together.

## 5. Repository map and ownership

```text
AGENT.md                         this guide
README.md                        user-facing overview and quick starts
CONTRIBUTING.md                  contribution, validation, version policy
SECURITY.md                      threat model and reporting policy
VERSION                          canonical stable release version
pyproject.toml                   Python package, tools, and test markers
requirements*.lock              reproducible backend/ML/desktop dependencies
compose.yaml                     hardened CPU/NVIDIA/maintenance topology
Dockerfile                       multi-stage CPU, NVIDIA, and frontend images
.env.example                     ignored deployment configuration template
config/default.yaml              portable schema-v1 defaults/profiles
config/local.example.yaml        template for ignored user-local configuration
backend/local_ai_doctor/         Python application package
  adapters/                      architecture-neutral contracts/registry
  api/                           strict boundary schemas and body limits
  discovery/                     scanning, fingerprints, capabilities, headers
  domain/                        immutable model/capability domain objects
  hardware/                      inventory and device/dtype selection
  persistence/                   SQLite connection, migrations, repository
  reasoning/                     emitted reasoning-segment classification
  sampling/                      exact sampler metrics and math
  services/                      models, runs, events, uploads, workspaces
  workers/                       admission, spawned runtime, supervisor
frontend/                        React/TypeScript/Vite application
  src/api/                       auth, public types, boundary normalization
  src/components/                workbench UI and inspector views
  src/hooks/useWorkbench.ts      browser orchestration and durable state merge
  src/utils/                     token formatting and safe Markdown helpers
  tests/                         Playwright fixture-driven browser tests
desktop/                         Electron shell and PyInstaller packaging
docker/                          entrypoints, preflight, Nginx, SQLite utilities
docs/                            architecture/API/metrics/deployment/evidence
scripts/wsl-docker.ps1           WSL-only Docker wrapper
scripts/release-version.mjs      synchronized semantic-version helper
tests/                           Python unit/integration/security suites
release/                         generated EXE output; never commit
runtime/ and data/               local runtime state; never commit
```

### 5.1 Canonical focused references

- `docs/architecture.md`: runtime topology, lifecycle, generation flow, persistence, trust boundaries.
- `docs/api.md`: REST endpoint and WebSocket protocol contract.
- `docs/configuration.md`: schema, precedence, profiles, every setting group.
- `docs/metrics.md`: metric formulas, timing clocks, instrumentation, attention semantics.
- `docs/adapters.md`: adding and validating a model family.
- `docs/deployment.md`: WSL Docker commands, profiles, backup/restore, operations.
- `docs/benchmarking.md`: valid benchmark protocols and evidence requirements.
- `docs/benchmark-results.md`: dated real-model results.
- `docs/model-capability-report.md`: supplied-checkpoint inspection and runtime evidence.
- `docs/limitations.md`: current gaps and non-claims.

Do not duplicate a large contract in code comments when one of these documents is the better home. Do update both the focused reference and this guide when an owner-level invariant changes.

## 6. Runtime architecture

```text
Browser/Electron window
  |  static UI + same-origin /api/v1 + /ws/v1
  v
Frontend reverse proxy/static server (:6969 in Docker/Electron)
  |  private REST/WebSocket traffic
  v
FastAPI application (:6767 in Docker/Electron; :8000 native default)
  |-- typed settings and transport security
  |-- model discovery/registry
  |-- FIFO inference admission and run orchestration
  |-- replayable event broker
  |-- chat/workspace/upload services
  |-- SQLite repository + batched telemetry writer
  |
  `-- ModelWorkerSupervisor
        | bounded serializable multiprocessing messages
        v
      spawned worker process
        |-- local Transformers generation/sampling/attention
        `-- local SentenceTransformers embeddings

Read-only model roots              Writable application data
config/tokenizer/processor/        SQLite/uploads/cache/exports/backups
templates/SafeTensors
```

Heavy model objects, tokenizer/processor instances, CUDA state, KV caches, and raw full-vocabulary tensors stay inside the worker. Only bounded, serializable commands/results/events cross the process boundary. A worker exception or process exit becomes a structured failure and must not leak an arbitrary traceback or local path through the public API.

The application is optimized for one local user, one web process, one worker process holding several resident models, batch size one, and one inference lease at a time. Do not increase web/model workers merely to chase throughput without validating SQLite ordering, GPU ownership, admission semantics, and shutdown behavior.

## 7. Configuration contract

All runtime settings flow through `AppSettings` and `SettingsLoader`. Do not scatter direct `LAD_` reads through business logic.

### 7.1 Precedence

From lowest to highest:

1. Pydantic defaults.
2. Portable configuration `defaults`.
3. Selected built-in profile.
4. Selected profile in the portable configuration.
5. User-local configuration `defaults`, then its selected profile.
6. `LAD_` environment overrides.
7. Repeated command-line `--set` overrides.

Profile selection is CLI `--profile`, then `LAD_PROFILE`, then portable `active_profile`, then `development`.

Supported profiles are:

- `development`
- `test`
- `native-windows`
- `native-wsl`
- `container-cpu`
- `container-nvidia`
- `production`

Configuration schema version is 1. Unknown fields fail. Future schema versions fail. The loader can migrate the early version-0 `models_path` and `database_path` keys.

### 7.2 Local configuration rules

- `config/default.yaml` is portable and committed.
- `config/local.example.yaml` is committed and contains no personal path.
- `config/local.yaml` is writable, ignored, and machine-local.
- `.env.example` is committed; `.env` is ignored and machine-local.
- Native model roots must be absolute paths visible to the backend process.
- Docker's model-root setting normally uses `/models`; the host-side bind source belongs only in ignored `.env`.
- The web model-root editor is disabled when a higher-precedence environment or CLI override would make its write misleading.
- Effective-configuration views redact secrets, private paths, and WSL identity.
- Persist an inference-relevant settings snapshot and digest with every run.

### 7.3 Major setting groups

- `paths`: model roots, database, uploads, cache, exports, backups.
- `server`: host, port, origin allowlist, external-access opt-in, auth token, CSRF.
- `runtime`: backend/device/fallback, CPU threads, RAM/VRAM budgets, low-memory loading, placement, dtype, quantization, attention backend, model/batch/concurrency/queue limits.
- `inference`: context limits, reserved output, prompt limit, local-only/trust policy, deterministic mode, instrumentation, sampler defaults.
- `limits`: upload/prompt/trace/event/media bounds.
- `workers`: count and startup/load/inference/shutdown timeouts.
- `telemetry`: persistence, retention, routing bounds, hardware interval, write batching.
- `logging`: level, JSON mode, redaction, prompt/output logging prohibitions.
- `features`: experimental attention/hidden-state/activation/logit-lens/router/multi-GPU flags.
- `platform`: portable container paths/profile hints.

Only claim a typed option works when the runtime path supports it. For example, `runtime.quantization` supports `none`, `bitsandbytes-4bit`, and `bitsandbytes-8bit` (CUDA text generators only) while the legacy `int4`/`int8` values are rejected at load, and `runtime.cpu_offload: true` is rejected as superseded by `runtime.strict_vram: false`.

Several fields are forward-looking or only partially honored today, including general backend selection, device placement, multi-worker/multi-model concurrency, scheduled retention, router controls, periodic utilization sampling, and most experimental probes. `features.attention_probe` is not the current capture gate: choosing `full` or `expert` instrumentation requests compatible causal attention capture.

Portable defaults worth knowing when reviewing limits are:

- Fallback context limit 4,096, used only when a checkpoint declares no length; maximum prompt tokens 32,768; reserved output 512. Prompt size is bounded by `max_prompt_tokens`, not by the fallback.
- Default output 512, temperature 0.7, top-k 50, top-p 0.95, min-p 0, penalties 1/0/0, and ten alternatives.
- Default instrumentation `token` and queue limit 32.
- Prompt body/content 4 MiB; upload 100 MiB; per-run trace 128 MiB; 100,000 token events; 16 attachments.
- Worker startup 120 seconds, model load 600 seconds, inference 3,600 seconds, shutdown grace 15 seconds.

Check `config/default.yaml` before relying on these values; a profile or higher-precedence source may override them.

## 8. Model discovery and capability truth

Discovery scans each configured root to depth two without following directory symlinks. A directory becomes a candidate when it has `config.json` or a direct SafeTensors file. Discovery must:

- Read bounded JSON metadata.
- Parse and validate SafeTensors headers, tensor shapes/dtypes, offsets, and byte ranges without deserializing weights.
- Detect required config, tokenizer, chat template, generation config, processor, pooling, and bundled custom-code components.
- Infer task and modalities from architecture, tensor, and packaging evidence—not a marketing model name.
- Identify MoE only through genuine router/expert evidence. Ordinary `gate_proj` tensors in a dense SwiGLU MLP are not experts.
- Record every discovered context-length candidate and select the most authoritative one: the sentence-transformers truncation length, then the architecture's declared positional capacity, then the tokenizer's `model_max_length`, and only then the configured application fallback. Never select a pre-scaling RoPE base length, a per-layer attention span, or an unbounded tokenizer sentinel; record them as evidence. Selecting a minimum across all candidates is a defect: it floors every checkpoint to the application value and hides the real context.
- Produce diagnostics for missing, malformed, contradictory, or unsupported components. Pickle-only weights are blocked with `pickle_weights_only` and conversion guidance, never unpickled. Reviewed bundled code must also pass a static import check (`bundled_code_incompatible` otherwise); a review approves what code does, not that it still imports.
- Discover reasoning delimiters only from chat-template text. Added-token lists name `<think>` for whole tokenizer families, including embedding checkpoints.
- Report the SafeTensors header dtype as the model dtype and keep the configuration's claim as evidence.
- Produce a complete capability matrix. Every non-`full` state needs a reason. A checkpoint with any error-severity diagnostic is not loadable and reports every capability `unsupported`, naming the blocking diagnostic.
- Compute a stable manifest fingerprint. The default quick policy hashes metadata fully and SafeTensors headers/sizes, not every weight byte; it is identity evidence, not a full checkpoint-integrity digest. The reported weight size counts only files a Transformers load reads (index-listed shards, else `model.safetensors`), so alternate copies such as `original/consolidated.00.pth` do not inflate budget checks.

Current defensive scan bounds include 32 MiB JSON metadata, 256 MiB SafeTensors headers, and 1,000,000 tensor records. Preserve bounded reads and failure diagnostics when extending discovery.

Unknown or incomplete folders remain visible with `task=unknown`/`loadable=false` and actionable diagnostics. Do not make discovery hide them and do not guess them into a working task. Folders that cannot be candidates at all (empty, GGUF-only) are reported as root diagnostics with root-relative names; dot-directories are skipped.

Model deletion in the registry may remove metadata, never underlying files. A changed fingerprint creates new identity evidence; historical runs retain the recorded fingerprint.

### 8.1 Capability states

Use exactly the evidence-oriented states:

- `full`
- `partial`
- `unsupported`
- `unavailable_on_backend`

Cover generation, encoder-decoder generation, embeddings, modalities, native/extracted input, reasoning exposure, MoE routing, logits/alternatives, scoring, attention/hidden states, streaming, batching, deterministic seeding, and hardware backends.

### 8.2 Supplied checkpoint facts

The current local model inventory is external and may change. Never commit its host paths. The last reviewed checkpoints were:

**DeepSeek-R1-Distill-Qwen-1.5B**

- Dense BF16 causal text-generation checkpoint using `Qwen2ForCausalLM`.
- Approximately 1.78B parameters; not MoE.
- Emits reasoning text using `<think>...</think>` conventions.
- Context sources and BOS guidance can disagree. Use the bundled tokenizer, chat template, and generation config, expose all candidates, and select the most authoritative context source rather than the smallest.
- System-prompt/template guidance may warrant a warning, not invented special-token reconstruction.

**Qwen3-VL-Embedding-2B**

- Dense BF16 multimodal embedding checkpoint, approximately 2.13B parameters.
- The declared architecture may resemble conditional generation, but the checkpoint has no LM head. It is not a generation model.
- Reviewed path supports text, image, video, and mixed inputs; it does not support audio.
- Native vector width is 2,048 with last-token pooling and L2 normalization.
- Reviewed Matryoshka dimensions are 64 through 2,048; truncate leading dimensions and re-normalize.
- The runtime applies the reviewed checkpoint key mapping for the outer `model.` prefix. Do not remove it without revalidating exact weight binding.

Treat `docs/model-capability-report.md` and `docs/benchmark-results.md` as dated evidence. Re-run current hardware/container checks before repeating an old availability conclusion. In particular, an older report records a missing WSL NVIDIA CDI prerequisite, while later project work established a healthy NVIDIA Docker path; current smoke tests are authoritative.

The last recorded real-model workstation evidence used an RTX 4060 Laptop GPU with about 8 GiB VRAM and a PyTorch 2.8 CUDA 12.8 runtime. This is dated evidence, never a portable hardware assumption.

### 8.3 Hardware selection and loading

- `auto` chooses the first PyTorch-usable CUDA device, otherwise CPU when fallback is allowed.
- CPU `auto` dtype resolves to FP32; explicit CPU FP16 is rejected.
- CUDA `auto` dtype resolves to BF16 on compute capability 8 or newer, otherwise FP16.
- Physical NVIDIA hardware reported by `nvidia-smi` does not prove that the installed PyTorch runtime can use it.
- Docker CPU and NVIDIA profiles both disable CPU fallback; an unavailable explicit CUDA request must fail clearly.
- The worker loads local files only, calls evaluation mode, and always places the model through a device map (never `.to()`): the exact selected device, CPU, or accelerate layer offload when Strict VRAM is off.
- A resident is reused only when its key matches: model ID, fingerprint, device, effective dtype, and quantization. A Strict VRAM request re-places an offloaded resident on the GPU.
- A non-`none` quantization passes a `BitsAndBytesConfig` (NF4 with double quantization and the selected compute dtype, or LLM.int8 with threshold 6.0) through the same device map; the LM head, and a multimodal model's vision tower, stay at the compute dtype. A checkpoint whose `config.quantization_config` is bitsandbytes loads with no quantization argument.
- Flush to storage (`POST /models/resident/{modelKey}/flush`, a lifecycle operation) saves a load-time quantized GPU resident with `save_pretrained` into `<root>/.lad-staging-<uuid>`, adds the tokenizer or processor, licence files, and a path-free `local_ai_doctor_derivation.json`, renames the staging folder to the new name, and rescans. A failure removes only the staging folder.
- A worker timeout poisons IPC state. The supervisor retires/replaces the worker and queues atomically; if the old process cannot be terminated, fail closed.

## 9. Backend lifecycle and execution

### 9.1 Startup

FastAPI lifespan startup performs these operations in order:

1. Open SQLite, enable foreign keys and WAL, and apply append-only migrations. Back up an existing database before a pending migration.
2. Start the asynchronous telemetry writer.
3. Spawn the isolated model worker.
4. Scan configured model roots and persist descriptors/capabilities.
5. Mark runs left `queued`, `loading`, or `running` by a prior process as failed with `interrupted_by_restart`.
6. Expose composed services to routes.

Shutdown cancels orchestration, asks the worker to unload and exit, drains telemetry, and closes SQLite. If the worker misses the grace period, terminate only that child after bounded waiting.

### 9.2 Admission and model lifecycle

- Generation, embeddings, and prompt scoring share one FIFO execution lease.
- Capacity is one runnable/active inference plus `runtime.queue_limit` waiters.
- Queue overflow returns structured HTTP 429 before generation creates chat/run state.
- Explicit load/unload conflicts while inference is admitted so lifecycle changes cannot overtake queued work.
- Loading another model keeps existing residents when they fit. The worker preflight estimates weights at the compute dtype plus a KV reserve and compares them with free memory less `vram_safety_margin_bytes` (clipped to the configured budgets) before `from_pretrained`; the registry evicts least-recently-used residents the running job does not use, then fails with 507 under Strict VRAM or retries once with layer offload. It is an estimate, not a peak-memory guarantee.
- OOMs, corrupt files, unsupported dtypes/devices, worker exits, and timeouts become bounded structured errors.

### 9.3 Generation flow

1. Validate request shape, model capability, prompt byte limit, chat/parent, seed, sampling, and admission.
2. Resolve hardware/device/dtype, then persist the user message, pending assistant, queued run, model fingerprint, and reproducibility snapshot.
3. Emit `run_created`, load/reuse the selected model, and transition the run to `running`.
4. Reconstruct the selected conversation branch, lead it with the chat's system prompt when set (unless the branch already begins with a system message), and render the tokenizer chat template with `add_generation_prompt=true`, or use the documented deterministic fallback when no template exists. A template that raises on or drops the system role gets it prepended to the first user message instead, with a `system_prompt_merged` warning. The applied prompt is snapshotted in `settings.system_prompt`.
5. Tokenize with `add_special_tokens=false` because the template owns special-token behavior. The plain-text fallback has no template, so it prepends the BOS token the tokenizer inserts by default.
6. For causal generation, prefill and reuse KV cache, passing absolute `cache_position` to every decoder call (multimodal-RoPE families such as Qwen3-VL derive decode positions from it). For encoder-decoder generation, encode source once, resolve the decoder-start token, and reuse encoder outputs/decoder cache.
7. Run the owned sampling pipeline, emit bounded events, persist according to policy, checkpoint partial message text every eight tokens, and check cancellation between bounded units.
8. Finalize run/message on completion, cancellation, or failure while preserving partial output and accepted telemetry.

Do not replace the reference loop with `generate()` if doing so hides logits, timings, cancellation, sampler state, reasoning slices, or attention data promised by the product.

### 9.4 Sampling order

The fixed order is:

1. Repetition penalty.
2. Frequency penalty.
3. Presence penalty.
4. Temperature, or deterministic argmax at temperature zero.
5. Top-k.
6. Top-p.
7. Min-p.
8. Renormalization.
9. Sampling.

Persist the effective settings and operation order. Raw-model and post-sampler probabilities are different quantities and must remain separately named throughout backend, API, UI, and exports.

Stop-sequence detection currently occurs after selecting and decoding a token, so matched stop text remains in stored/displayed output. Document and test any future change to that contract.

### 9.5 Seed and reproducibility

- Seed `0` is valid.
- If omitted, generate and persist an effective unsigned 64-bit seed before inference.
- Return requested/effective seed, RNG algorithm/device, model/tokenizer fingerprint, sampling, backend, device, dtype, quantization, attention implementation, software versions, deterministic-kernel state, and batching state.
- A seed cannot be recovered from output. Never claim otherwise.
- Replay is a new forward pass and only promises assistance within an unchanged environment, not byte identity across changed hardware, dtype, kernels, libraries, tokenizer, or batching.
- Greedy decoding must state that seed did not affect token selection.
- Deterministic-reference mode toggles deterministic PyTorch algorithms per request, sets the reviewed cuBLAS workspace behavior, and disables cuDNN benchmarking. Restore process-global settings after the run.

### 9.6 Reasoning recovery

Only segment reasoning that is explicitly emitted through tags, special tokens, a model-defined channel, template metadata, or reviewed configuration. Never infer hidden chain-of-thought.

The request's reasoning preference is forwarded as `enable_thinking` only to compatible templates. If disabling reasoning encounters a template that hardcodes an opening marker, add a closing marker only when history-only rendering proves that opening marker came from the generation prompt rather than user text.

For supported tagged reasoning models, if the configured output budget ends before visible answer text, the runtime may allocate one additional output window bounded by both the configured size and remaining model context. Emit a warning and terminal fields describing activation and extra-token use. Do not loop indefinitely; if context is exhausted or the model continues reasoning, finalize honestly and let the UI show the no-answer warning.

### 9.7 Replay and alternative-token branching

Normal replay:

- Requires a completed generation, intact model ID/fingerprint, and valid message lineage.
- Reconstructs the source branch and recorded settings, including the recorded `settings.system_prompt` snapshot rather than the chat's current prompt.
- Creates a sibling assistant branch linked by `parent_run_id`.

Alternative-token branching:

- Validates the exact source run, token index, distribution (`raw` or `sampler`), alternative rank, and token ID.
- Reconstructs the prefix through the selected decision point, forces the chosen alternative, and continues locally.
- Creates a new chat and new run; source records remain immutable. The new chat inherits the source run's system prompt snapshot.
- Must preserve reproducibility metadata and make the forced-token decision explicit in persisted state/telemetry.

The canonical route is `POST /api/v1/runs/{run_id}/branch`. Its request identifies `tokenIndex`, `distribution`, `rank`, and `tokenId`. The source must be complete, its model/fingerprint must still match, token rows through the branch point must be contiguous and durable, and the requested alternative must match a persisted raw or sampler alternative exactly. The runtime forces the original prefix through token `N-1`, substitutes the selected token at `N`, advances cache through the forced prefix, then samples the suffix normally. The new run intentionally has no cross-chat `parent_run_id`; provenance is stored separately so workspace export does not create a dangling cross-chat foreign key.

## 10. Metrics and observability contract

Metrics describe observable model/runtime behavior. They do not reveal consciousness, intent, hidden deliberation, or a complete causal explanation. Define a new metric before displaying it: identify its distribution, units, clock, population, exclusions, exact/sampled/estimated status, and unavailable state.

### 10.1 Token identity and rendered text

For each generated token, preserve where available:

- Zero-based `token_index`, the stable graph/table x-axis.
- Vocabulary `token_id`.
- Exact tokenizer `piece`.
- UTF-8 `escaped_bytes`.
- Newly visible decoded `display_text`.
- `replace_from`, because decoding a complete prefix may revise a prior visible suffix.
- Half-open rendered `span_start`/`span_end`.
- Reasoning/answer/unknown classification plus code-point `reasoning_slices` where a token crosses classes.

Never reconstruct visible text by concatenating tokenizer pieces. Apply `replace_from` and `display_text` in token order. Conversely, do not discard raw pieces merely because the UI shows friendlier whitespace labels.

### 10.2 Raw model distribution

For raw logits `z` over the full vocabulary:

```text
raw_log_probability(i) = z_i - logsumexp(z)
raw_probability(i)     = exp(raw_log_probability(i))
```

Detailed tiers compute normalization in float32 over the complete vocabulary. Do not normalize only the displayed alternatives and label the result exact.

Exact raw rank is:

```text
rank(i) = 1
        + count(z_j > z_i)
        + count(z_j = z_i and j < i)
```

Ascending token ID breaks exact-logit ties deterministically.

### 10.3 Sampler distribution

The sampler distribution is the renormalized distribution after the fixed penalty/temperature/filter pipeline. The chosen sampling probability/log probability refers to what is passed to `torch.multinomial`; greedy selection has sampling probability 1 and log probability 0.

Raw alternatives are the highest raw logits with exact full-distribution probabilities and a `survived_filter` flag. Sampler alternatives are the highest surviving post-filter values. These are bounded views, not model thoughts and not standalone normalized Top-K distributions. Say “top alternatives under this distribution.”

Always store the chosen token in the main token row even when it falls outside the configured alternative count.

### 10.4 Likelihood and uncertainty

- `entropy = -sum(p_i * ln(p_i))` over the surviving sampler distribution.
- `surprise = -ln(p_selected)` under the sampler distribution.
- `cumulative_logprob` is the sum of selected **raw model** log probabilities so far.
- `running_perplexity = exp(-cumulative_raw_logprob / generated_token_count)`.
- Conditional response perplexity is `exp(-mean(raw model log probability of every generated token))`.
- Prompt perplexity is a separate teacher-forced causal scoring pass using shifted logits; exclude the undefined first position and masked/non-text positions.

Perplexity is model/tokenizer-specific likelihood, not factuality, reasoning quality, safety, or answer usefulness. Do not compare values casually across tokenizers or inclusion rules.

Reasoning- and answer-segment perplexity/throughput are product goals where segmentation and persisted fields support them. Show unavailable rather than deriving a misleading value.

### 10.5 Timing and throughput

Use monotonic clocks for server/worker durations. Keep the following concepts distinct:

- Request receipt and queue entry/exit.
- Model loading and any warm-up.
- Attachment preprocessing.
- Template rendering and tokenization.
- Prefill start/end and prompt throughput.
- Decode forward, sampling, emission, and inter-token latency.
- Browser receipt and client inter-arrival.
- Engine TTFT, server TTFT, client TTFT, and time to first visible answer text.
- Decode TPS, rolling TPS, end-to-end TPS, cumulative time, completion/cancel/failure.

The first token includes prefill and is not an ordinary steady-state decode point. Browser clocks are not directly comparable to server monotonic timestamps as absolute time. The current `emit_ms` is worker-enqueue bookkeeping, not a measured shared-clock network transport duration.

### 10.6 Instrumentation tiers

- `off`: lightweight chosen logit, sampler likelihood, identity/display, and timing.
- `basic`: currently the same lightweight capture as `off`.
- `token`: exact raw likelihood/rank, entropy/perplexity, and bounded alternatives.
- `full`: detailed token data, CUDA synchronization where relevant, and bounded attention request.
- `expert`: full detail plus expert-oriented capability paths where implemented.

Detailed profiling can alter kernels, synchronization, memory, and throughput. Warn the user and record the effective attention/instrumentation path. Do not claim the historical target of less than 5% lightweight overhead unless a controlled paired benchmark actually demonstrates it.

### 10.7 Bounded self-attention summary

For generated token `y_t`, a compatible causal model may return final-query post-softmax attention rows `A[l,h,t,j]` over earlier source positions. The stored visualization statistic is:

```text
mean_attention[t,j]
  = (1 / captured_head_rows) * sum over captured layers/heads A[l,h,t,j]
```

Implementation requirements:

- Capture the eager forward pass that produced that step's logits.
- Align the final causal query row to rendered prompt/history/generated-prefix positions.
- Clamp only numerical underflow below zero and normalize the mean back to unit mass.
- Retain at most the 128 highest-weight source positions for each generated token.
- Persist exact omitted mass and capture metadata.
- Persist the complete rendered-prompt token catalogue once on generated token 0; reconstruct generated source positions by prior token events.
- Do not guess conversation roles for prompt positions: arbitrary templates can rewrite/inject text. Context index and token identity are exact; role attribution may be unknown.
- At prompt prefill, avoid storing a quadratic full-prompt matrix. Prefill the prefix and use the final prompt token as the eager query predicting generated token 0.
- Continue the run with a warning when alignable attention is unavailable; do not fail otherwise valid generation.

The frontend heat scale is normalized against the strongest retained source across prompt and generated positions together. Unretained sources are gray. Coverage displays retained/total positions, retained/omitted mass, layers × heads, method, and aggregation.

### 10.8 MoE telemetry

Only expose routing when the architecture and runtime path genuinely provide it. Desired fields include router logits/probabilities, selected and executed experts, normalized weights, shared experts, overflow/drops, entropy, counts, load, imbalance, and drop rate.

Keep “selected” and “executed” separate. Dense models show routing as not applicable. If hooks disable a fused/compiled path, report that overhead. Prompt traces must be bounded and aggregates are the default.

### 10.9 Embedding telemetry

Embedding runs do not have generated-token alternatives, generation perplexity, reasoning tokens, or decode TPS. Show those as not applicable.

Record input modality, processor/tokenizer inputs, truncation, effective length, pooling, normalization, requested/output dimensions, dtype, vector norm/statistics, preprocessing/forward/total timing, items or tokens per second where defined, peak memory, and joint-versus-modality-specific space when known.

The current returned similarity matrix is a dot product. It is cosine similarity only when vectors are normalized.

### 10.10 Performance objectives

- Deliver a normal local UI update within 100 ms after the browser receives an engine token event.
- Keep cancellation checks between bounded decode units and stop promptly.
- Keep SQLite writes and export formatting off the decode hot path.
- Do not retain full-vocabulary logits per token; compute exact detailed metrics transiently and persist only the chosen token and bounded alternatives.
- Bound trace bytes, event counts, subscriber queues, prompt/upload sizes, and retained attention.
- Keep disabled/basic instrumentation close to the same engine path. The original product target is no more than 5% measured overhead, but publish that number only after controlled paired benchmarks.
- Release GPU memory on unload and keep long chats/traces responsive through virtualization/lazy loading.

## 11. Persistence, events, and data ownership

### 11.1 SQLite

SQLite uses one WAL-mode connection, foreign keys, explicit transactions, indexed queries, append-only migrations, and a write lock. Existing databases are backed up before pending migrations.

Current migrations cover the initial schema, reasoning slices, token attention attribution, and the per-chat system prompt (`chats.system_prompt`). Never edit an already released migration; add the next migration and test upgrade, backup, restart, and failure behavior.

Persistent concepts include:

- Models, fingerprints, diagnostics, and capabilities.
- Chats (with an optional system prompt), branched messages, and attachment links.
- Content-addressed attachment metadata.
- Generation runs and replay parentage.
- Environment/configuration snapshots and phase metrics.
- Token rows, alternatives, reasoning slices, and optional attention summaries.
- Router records/aggregates for future supported paths.
- Embedding runs/inputs and optionally vectors.
- Raw versioned protocol events.

Token/alternative writes are batched away from the decode hot path. Publishing is serialized per run as `assign sequence -> apply persistence policy -> fan out`. Before terminal protocol persistence, flush accepted batched token writes. A browser disconnect does not cancel inference.

### 11.2 Partial results and restart recovery

- Checkpoint assistant content every eight tokens and at terminal state.
- Cancellation, failure, and disconnect-related terminal states preserve partial message content and telemetry.
- A sudden process loss can leave message text behind durable event/token rows; startup marks the run failed rather than regenerating it.
- Frontend restoration treats the backend snapshot as authoritative and merges only supplemental browser timing.

### 11.3 WebSocket ordering and recovery

Protocol subprotocol is `lad.events.v1`. Each ordinary event has schema version 1, run ID, monotonically increasing per-run sequence, type, worker/server timing, and payload.

Recognized event families include `run_created`, `stage`, `model_loaded`, `metric`, `token`, `warning`, `completed`, `cancelled`, and `error`.

- Subscribe before replay to close the replay/live race.
- Resume with `after=<last_processed_durable_sequence>`.
- Filter duplicate ordinary sequences.
- Persistence-disabled or limit-truncated token events cannot be recovered after disconnect; refresh current REST message/run state.
- Subscriber queues are bounded. Overflow clears that subscriber queue and emits synthetic `resync_required` with `payload.resume_after` equal to the last event actually delivered.
- Do not advance the durable cursor to the synthetic event's sequence.

Browser bearer credentials travel in a private `lad.auth.<base64url-token>` WebSocket subprotocol value, never a URL query parameter. The server selects only `lad.events.v1`.

### 11.4 Portable workspaces and exports

Chat workspace export/import uses strict path-free schema `local-ai-doctor/chat-workspace`, version 1. It contains chat metadata, branched messages, runs, token rows, bounded alternatives, and persisted bounded attention data. Import validates unknown fields, versions, identifiers, graph cycles, parentage, duplicate token positions/alternatives, and remaps all IDs.

It deliberately excludes:

- Attachments and upload bytes.
- Filesystem paths.
- Raw protocol events.
- Environment and phase rows.
- Embedding inputs/vectors.
- Executable models.

Non-terminal imported snapshots become failed records rather than resumed work. Missing destination model IDs become null while the source fingerprint is retained.

Per-run exports support generation JSON, raw-event JSONL, and token CSV.

### 11.5 Retention and deletion

Telemetry retention is an explicit confirmed API operation with a timezone-aware cutoff. It deletes entire terminal runs before the cutoff and cascades dependent telemetry while preserving chats/messages, models, attachments, upload bytes, backups, exports, and active runs. `telemetry.retention_days` is not an automatic scheduler.

Deleting a chat cascades its database messages/runs/telemetry, but content-addressed upload garbage collection is not currently implemented. Never imply that chat deletion erases every copy, backup, export, or orphan upload.

Deleting model metadata never deletes model files.

## 12. REST and WebSocket boundary

The versioned REST base is `/api/v1`. `backend/local_ai_doctor/main.py` and strict schemas are the executable route authority; keep `docs/api.md` synchronized. Canonical fields are `snake_case`; the bundled browser boundary may normalize documented camel-case compatibility. Unknown JSON fields normally return 422. Do not add unchecked casts in components to compensate for boundary drift.

Development-only interactive OpenAPI is `/api/docs`; the schema is `/api/v1/openapi.json` in every profile.

### 12.1 Endpoint inventory

Service/configuration/storage:

- `GET /health`
- `GET /configuration`
- `GET /configuration/model-roots`
- `PUT /configuration/model-roots`
- `GET /hardware`
- `GET /storage`
- `DELETE /storage/telemetry?before=...&confirm=true`

Models:

- `GET /models`
- `POST /models/refresh`
- `POST /models/{model_id}/load`
- `GET /models/resident`
- `POST /models/resident/{model_key}/unload`
- `POST /models/unload`
- `POST /models/{model_id}/unload`
- `GET /models/{model_id}/inspect`

Chats/messages:

- `POST /chats`
- `POST /chats/import`
- `GET /chats`
- `GET /chats/{chat_id}`
- `GET /chats/{chat_id}/export`
- `GET /chats/{chat_id}/messages`
- `PATCH /chats/{chat_id}`
- `DELETE /chats/{chat_id}`
- `DELETE /chats?confirm=true&include_archived=false`
- `POST /chats/{chat_id}/messages`

Runs/scoring:

- `POST /runs` and `/runs/generation`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/events`
- `POST /runs/{run_id}/cancel`
- `POST /runs/{run_id}/replay`
- `POST /runs/{run_id}/branch`
- `GET /runs/{run_id}/export?format=json|jsonl|csv`
- `GET /runs/compare/summary?ids=...`
- `POST /runs/prompt-score`

Attachments/embeddings:

- `POST /attachments` and `/uploads`
- `GET /attachments/{attachment_id}/content`
- `POST /embeddings` and `/runs/embeddings`

Live events:

- `GET ws(s)://<host>/ws/v1/runs/{run_id}?after=<sequence>` using `lad.events.v1`.

### 12.2 Errors and body limits

Normal application errors use a bounded structure with `code`, safe `message`, `retryable`, optional `hint`, and bounded `details`. Validation uses `invalid_request`. Some direct resource/conflict errors still use FastAPI's `detail` shape, so clients handle both.

Bound bodies before route parsing. Non-multipart bodies allow at most `limits.prompt_bytes + 1 MiB`; multipart uploads at most `limits.upload_bytes + 1 MiB`, followed by exact inner limits. Oversize envelopes return 413 `limit_exceeded`.

Do not leak arbitrary exception text, local paths, prompts, outputs, tokens, or credentials through errors.

## 13. Uploads and multimodal processing

Uploads are streamed under configured byte limits, identified by content signature rather than client MIME/name, hashed, and stored under content-addressed names outside SQLite blobs.

Currently recognized signatures include PNG, JPEG, GIF, WebP, MP4, WebM, WAV, FLAC, MP3, PDF, and UTF-8 text. Recognition is not acceptance:

- Accept a native modality only if the selected model and runtime adapter support it.
- The supplied embedding model supports text/image/video/mixed, not audio.
- Plain text/PDF upload inference is rejected because no extracted-text adapter is registered.
- Generation accepts image and video attachments only when the selected causal generator reports `vision`/`video` as `full` or `partial` (processor-backed; validated on Qwen3-VL). Rejection happens before chat/run rows are written. The worker renders media turns with the checkpoint processor's chat template, decodes only upload-store files the parent resolved, and passes pixel inputs to the prefill forward only. Earlier messages' attachments are re-sent when history is rendered, including replay and token branches. Audio chat input is unsupported.

Validate images and fully decode bounded media before recording an attachment. Current portable defaults include:

- 100 MiB upload limit.
- 40,000,000 image pixels.
- 256 video frames.
- 8,500,000 pixels per video frame.
- 500,000,000 cumulative decoded pixels.
- 600-second media duration.

Canonicalize paths, reduce upload names to basenames, reject traversal/symlinks, never fetch media URLs, never extract archives, and always revoke frontend object URLs.

## 14. Frontend architecture

The frontend is React 18, strict TypeScript 5.7, Vite 6, DM Sans for general UI, and JetBrains Mono for token/data/code surfaces.

### 14.1 State ownership

- `frontend/src/main.tsx` loads fonts/styles and renders `App` under `StrictMode`.
- `App.tsx` owns presentation state: workspace, mobile sidebar, inspector/drawer, Nerd Mode, color metric, inspector tab, and selected token.
- `useWorkbench.ts` is the single application orchestration hook for bootstrap, API state, model/chat/run state, optimistic messages, WebSocket streams, restoration, replay/branching, uploads, errors, and notices.
- There is no Redux/global domain store.
- `api/client.ts` and `api/types.ts` own external normalization/serialization. Components receive normalized typed data.

Main workspaces/components:

- Chat: `ChatView`, `Composer`, `ContextMeter`.
- Embeddings: `EmbeddingsWorkspace`.
- Models/settings: `ModelRegistry`, `ModelRootSettings`.
- Navigation/runtime: `Sidebar`, `WorkbenchHeader`.
- Observability: `Inspector`, `VirtualTokenTable`, `TraceChart`, `AttentionAttribution`.
- Full generation/auth settings: `GenerationControls`.

The product-level layout contract is:

- Left sidebar: new chat, persisted recent/archived chats, automatic titles, rename, search, pin, archive, confirmed delete, and confirmed clear-all.
- Main chat: streaming output, stop, regenerate/replay, edit-and-retry, copy, branch selection, attachment previews, and response details.
- Collapsible/right-side inspector: Overview, Tokens, Probability, Timing, Experts, Context, Embeddings, Hardware, Configuration, and Raw Events.
- Compact circular context meter in the composer toolbar (it must never overlap the message input).

Do not remove an unsupported inspector area merely to make the UI appear complete. Keep it visible/disabled with the capability reason when that helps users understand the model/backend boundary.

### 14.2 API and authentication

`VITE_API_BASE` defaults to same-origin `/api/v1`; `VITE_WS_BASE` is optional. Vite development listens on `127.0.0.1:5173` and proxies to native backend `127.0.0.1:8000`. Production Docker/Electron uses the fixed 6969/6767 contract.

The browser token is tab-scoped write-only state kept in memory and `sessionStorage` under `lad.auth_token`. It is added to REST by `api/auth.ts`, never displayed or persisted in app config. HTTP 401 triggers `lad:auth-required`; saving/clearing triggers `lad:auth-changed` and reboots data queries.

### 14.3 Bootstrap and model truth

Bootstrap uses `Promise.allSettled` for health, models, active chats, archived chats, and configuration. Distinguish fully offline, partial endpoint success, and auth-required states. A single failed optional endpoint must not fabricate “backend offline.”

The health response is lifecycle truth: mark its loaded model loaded and all others unloaded. Load actions use current device/dtype. Entering Embeddings may select the first usable embedding model. Ctrl/Cmd+K creates a chat only while connected.

### 14.4 Live generation

The optimistic flow is:

1. Add a temporary user message.
2. POST the strict generation request.
3. Replace the temporary user ID if the backend returns a durable one.
4. Add the streaming assistant/run.
5. Subscribe to WebSocket events.

Selection epochs and active-chat references protect against delayed history/run fetches overwriting a newer selection or submission.

Each token immediately updates decoded assistant text (including `replaceFrom` revisions), token telemetry, count, context generated count, rolling decode TPS, and running perplexity. Stage/metric/warning/terminal events merge into the selected run. The run strip exists before token one.

Track browser first-token receipt separately from first visible answer text so hidden reasoning does not masquerade as visible-answer TTFT. Measure client inter-arrival locally.

On terminal events, preserve partial output, finalize status, clear active stream state, refresh health/model/chat data, fetch the durable run, and merge supplemental client timing. Browser timing is capped to the latest 100 runs in `local-ai-doctor.client-telemetry.v1`; the backend remains authoritative.

### 14.5 Chat branches and restored inspection

`ChatView` renders one selected root-to-leaf lineage from a parent/child message graph. Sibling controls switch branches without flattening chronology.

- Ordinary continuation uses the latest completed assistant as parent unless an explicit parent/root is chosen.
- Editing a message starts a new generation with the correct parent.
- Regenerate/replay requires a completed assistant with a persisted run ID and creates a sibling branch.
- Details on any response fetches that response's persisted run.
- Opening a chat restores the latest assistant run, selected model, and inspector telemetry.
- A selected run summary is shown only when the run belongs to the currently selected lineage. Never label an unrelated historical run as live.
- Changing the selected run clears selected-token state.

Raw events are lazy-loaded only when the inspector is open, Nerd Mode is on, and Raw Events is selected.

### 14.6 Normal answer and Nerd Mode

Normal message rendering uses safe Markdown (`react-markdown`, GFM, KaTeX) with raw HTML skipped. Never enable model-supplied HTML.

Normal mode:

- Hides reasoning protocol and termination markers.
- Shows emitted reasoning inside a collapsed native `<details>` labeled `Thinking…`.
- Leaves final answer outside the disclosure.
- Copy strips protocol/reasoning/termination markup.

Nerd Mode:

- Preserves raw content and protocol markers.
- Enables token details/raw events/alternative branching.
- Supports color by raw probability, sampler probability, surprise, latency, or reasoning segment without relying on color alone.
- Hover provides a quick hint; click locks token/inspector selection.
- Synchronizes transcript tokens, virtual table, charts, and inspector.
- May collapse more than 1,200 old transcript boundaries for rendering performance, while the full virtual table remains addressable.

Use `displayTokenText` and `tokenTextHint`. Friendly labels may show `␠`, `↵`, or `⇥`; hover must retain the exact raw piece. Token chips remain 30 px high.

### 14.7 Inspector

Tabs include Overview, Tokens, Probability, Timing, Experts, Context, Embeddings placeholder, Hardware, Configuration, and Raw Events. Capability-gate every tab and show exact reasons for unavailable data.

Token inspection includes identity/span/bytes, raw and processed logits, model/sampler probability and log probability, rank, entropy, surprise, cumulative log probability, running perplexity, timing/throughput, alternatives, virtual token table, and bounded attention when captured.

Charts cover probability/log probability, entropy/surprise, rank, running perplexity, decode/sample time, server/client arrivals, instantaneous/rolling TPS, cumulative timing, and MoE where available. Use generated token index as x-axis. Windows of 128/512/all and display-point downsampling must not mutate source telemetry or selection.

Outside Nerd Mode, chart tooltips must not leak hidden token text. Configuration displays only redacted effective settings, precedence, and sampler order.

### 14.8 Composer and generation controls

The composer-side prompt popover applies to the next response and exposes:

- Reasoning toggle, capability-gated.
- Temperature slider.
- Top-k slider.
- Top-p slider.
- A "System prompt" textarea for the active chat. Edits are saved with a debounced `PATCH` and flushed before a send; a chat that does not exist yet keeps the text locally and sends it with the create request. A composer icon labelled "System prompt active" and a collapsed "System prompt" card at the top of the conversation show when one is set.

It is mutually exclusive with the attachment popover and closes while disconnected, unsupported, running, or branching. Enter sends; Shift+Enter inserts a newline. Stop explicitly preserves partial output.

Full controls expose:

- Tab-scoped bearer token.
- Device `auto`/CPU/CUDA.
- Dtype auto/FP32/FP16/BF16.
- Instrumentation `off`/`basic`/`token`/`full`/`expert`.
- Seed and deterministic-reference mode.
- Max output, temperature, top-k, top-p, min-p.
- Repetition, frequency, and presence penalties.
- Alternatives count and stop sequences.
- Effective-settings summary and reset to backend defaults.

Do not disable CUDA merely from a frontend assumption; use the backend capability reason. Defaults are seeded from backend configuration. Current portable defaults are max 512, temperature 0.7, top-k 50, top-p 0.95, min-p 0, penalties 1/0/0, ten alternatives, auto device/dtype, token instrumentation, and nondeterministic mode.

### 14.9 Model registry and embeddings UI

The registry shows actual architecture, task, parameters, dtype, context candidates, fingerprint, lifecycle, diagnostics, trust decision, tokenizer/template/special tokens/sampling details, and a reason-bearing capability matrix. Never infer capability from a filename.

Model-root settings accept one backend-visible absolute directory per line, persist via the backend, and rescan. Disable edits when offline, unwritable/overridden, saving/loading, or a model is loaded.

The backend rejects duplicates, filesystem roots, missing/unreadable directories, and changes while inference/model lifecycle is busy. It writes the selected user-local YAML/JSON profile with atomic replacement where supported and a guarded fsync fallback for Windows/WSL bind mounts.

The embeddings workspace supports capability-gated batches of text/media inputs, optional reviewed dimensions and normalization, returned vectors/statistics, similarity, optional backend projections, and JSON/CSV/valid little-endian Float32 `.npy` export. Do not invent projection coordinates. If calculating a client fallback, label and compute it correctly.

### 14.10 Styling and accessibility

- Primary accent is blue `#60a5fa`. Green is never used, including for success states (`--success` is a sky-blue variant) and chart series.
- Every colour, space, font size, radius, shadow, focus ring, and duration comes from the tokens in `frontend/src/styles/tokens.css`. Do not add literal hex values elsewhere; add or reuse a token. Charts read the categorical `--viz-1`…`--viz-8` and sequential `--seq-*` tokens through CSS variables.
- Styles live in four files imported by `main.tsx` in order: `tokens.css` (variables), `base.css` (element defaults and utilities), `components.css` (primitives), and `views.css` (layout and per-view rules, including breakpoints and workspace container queries).
- Build UI from the typed primitives in `frontend/src/components/ui` (Button, IconButton, Tabs, SegmentedControl, Field/Input/Textarea/NumberInput/Select, Switch, Slider, Badge, Card, Stat, EmptyState, Callout, Popover, MenuButton, Drawer) instead of one-off buttons, tabs, selects, section titles, empty states, or callouts. `IconButton` requires a label, which also becomes its hover hint.
- 11px is the minimum size for any text, including token-chip index labels, chart axis labels, and badges. Preserve the readability floor and do not shrink important text to fit.
- Maintain global `:focus-visible`, ARIA labels/states, live status, alerts, visually hidden labels, and disabled explanations.
- Respect `prefers-reduced-motion`.
- Preserve current responsive behavior around 1320, 1120, 900, 760, and 480 px.
- At 760 px the sidebar becomes a drawer and inspector becomes full-screen; prompt controls must remain inside very narrow viewports.
- Inspector defaults open on desktop and closed at 760 px or narrower.
- Errors are dismissible `role=alert`; notices use `role=status`.
- Add native `title` or an accessible equivalent for unfamiliar/compact controls and meaningful disabled reasons.

### 14.11 Context meter and long traces

The context meter should distinguish, when reported:

- Rendered prompt tokens.
- Template/special tokens.
- Expanded multimodal positions separately from text tokens.
- Generated tokens.
- Reserved output budget.
- Remaining capacity and percentage used.
- Effective model limit.
- Truncation or sliding-window behavior.

Show `Unknown` when conflicting metadata cannot be resolved honestly. Do not derive a confident context number from one convenient config field while ignoring contradictory tokenizer/model-card evidence.

Long token tables are virtualized, charts downsample display points, and transcript token boundaries may collapse after the documented threshold. Keep source telemetry and selection stable so 10,000-token traces remain inspectable. Provide accessible table/text alternatives to graph-only data.

### 14.12 Cross-cutting inspection features

Preserve and extend these product surfaces without overstating them:

- Model refresh, load/unload, descriptor diagnostics, and capability matrix.
- Prompt/chat-template, tokenizer/special-token, and sampling-pipeline inspection.
- Run comparison for two to eight runs; the current comparison is metadata-oriented, not a complete paired statistical analysis.
- Replay with recorded settings/seed and explicit environment caveats.
- Cold-versus-warm benchmark workflows through documented protocols; do not auto-run expensive benchmarks.
- Hardware/memory snapshots and redacted effective configuration.
- JSON/JSONL/CSV run export and portable chat import/export.
- Database size and explicit telemetry-retention controls.
- Bounded experimental attention, hidden-state, activation, and logit-lens views only when the runtime really supports them. Today only the bounded causal attention view is implemented from that group.

## 15. Security and privacy boundaries

### 15.1 Default network posture

- Bind to loopback by default.
- Reject non-loopback binding unless `allow_external_access=true` and an authentication token is configured.
- Enforce exact Host and Origin checks, strict CORS, state-changing CSRF checks, CSP, `nosniff`, no-referrer, and disabled camera/microphone/geolocation permissions.
- The application provides no TLS, accounts, roles, sessions, rate limits, lockouts, token rotation, or multi-tenant isolation. Any remote deployment requires a reviewed TLS/auth proxy and restricted network boundary.
- Proxies must not log WebSocket subprotocol headers because they can contain the tab-scoped credential.

### 15.2 Model and file trust

- Treat model metadata, tokenizer templates, filenames, uploads, chat text, model output, and API values as untrusted.
- Force Hugging Face/Transformers offline operation, `local_files_only=true`, and `trust_remote_code=false`.
- Do not load pickle checkpoints as a fallback.
- If bundled custom code becomes unavoidable, audit exact files, constrain by fingerprint, document why built-ins cannot load it, and execute only in the isolated worker. Never enable repository code globally.
- Worker process isolation is fault containment, not an OS sandbox.
- Keep model mounts read-only and verify them at container preflight.

### 15.3 Sensitive data

SQLite, uploads, caches, exports, and backups can contain prompts, outputs, rendered templates, token telemetry, vectors, and private content. Logs must not include prompts, uploads, generated tokens, credentials, or unredacted local paths.

Backups and exported workspaces/runs are not encrypted by the application. Treat them as sensitive. A quick model fingerprint is not a provenance-grade full-file digest.

Never print, store, interpolate, pipe, or embed passwords/tokens/private keys. Never weaken WSL sudo policy, Docker socket permissions, credential storage, or group membership to avoid an authentication prompt.

## 16. Native development workflows

### 16.1 Versions

- Python `>=3.12,<3.14`; Python 3.12 is the validated development/release target.
- Node.js 22 or newer.
- Current frontend: React 18, Vite 6, TypeScript 5.7.
- Current desktop: Electron 44 and electron-builder 26.
- Important ML/runtime versions are pinned in lock files; never install an arbitrary CUDA/CPU mixture and assume it is valid.

### 16.2 Native Windows setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock -r requirements-ml.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --editable .
Copy-Item config\local.example.yaml config\local.yaml
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\local-ai-doctor.exe serve --profile native-windows
```

Open `http://127.0.0.1:8000`. For live UI development, leave the backend on 8000 and run `npm --prefix frontend run dev`; open `http://127.0.0.1:5173`.

For native CUDA, install the exact pinned CUDA wheel from `requirements-cuda.lock` and verify the version, CUDA availability, and device name before running a model.

### 16.3 Native WSL setup

Use a separate Python 3.12 Linux environment; do not reuse a Windows virtual environment. Put only the Linux-visible model root into ignored `config/local.yaml` and run the same package/frontend build with `--profile native-wsl`.

The repository may be on a mounted Windows filesystem, which slows build-context traversal and many small file operations. Do not relocate it without owner approval.

## 17. WSL-only Docker operations

### 17.1 Mandatory execution path

All Docker builds, Compose operations, tests, inspection, health checks, and logs on Windows must run through `scripts/wsl-docker.ps1`, which invokes Linux `docker` inside a validated WSL2 distribution and user context.

Never:

- Invoke Windows `docker.exe` or a Windows Docker context.
- Guess a WSL distribution or hardcode a personal Linux user in committed code.
- Use `sudo`, pass a password, or loosen the Docker socket from automation.
- Relocate the repository or model directory without owner approval.

The wrapper chooses an unambiguous WSL2 distribution/default user or uses ignored `.env`/explicit parameters, validates Docker connectivity, translates paths safely, and invokes `docker compose` without credentials. If the user lacks Docker access, stop that operation and report the exact interactive/admin step; do not bypass host security.

### 17.2 Local deployment setup

```powershell
Copy-Item .env.example .env
Copy-Item config\local.example.yaml config\local.yaml
```

In ignored `.env`:

- `MODEL_PATH` is a Linux path visible inside the selected WSL distribution.
- `APP_CONFIG_DIR` is the local writable config directory.
- WSL distribution/user fields are optional local selectors.
- Resource, image/container, and explicit volume names are deployment settings.

The model bind is host `MODEL_PATH` -> `/models:ro`. The settings UI inside Docker normally stores `/models`, never the host Windows or WSL source path.

### 17.3 Compose topology

CPU profile:

- `preflight-cpu`
- `app-cpu`
- `frontend-cpu`

NVIDIA profile:

- `preflight-nvidia`
- `app-nvidia`
- `frontend-nvidia`

Maintenance profile:

- `database-maintenance`

Dockerfile targets are `cpu`, `nvidia`, and `frontend`. The CPU final image contains CPU PyTorch and verifies that no NVIDIA Python package leaked in. The NVIDIA target uses the pinned CUDA 12.8 PyTorch/Torchvision locks. The frontend is minimal Nginx and proxies `/api/` and `/ws/` to backend port 6767.

All final services preserve these hardening properties:

- UID/GID 10001, no root.
- Read-only root filesystem.
- All capabilities dropped.
- `no-new-privileges`.
- Bounded tmpfs, CPU, memory, PID, and shared-memory resources.
- Loopback-only host publication.
- Rotating JSON logs, 10 MiB × 3.
- SIGTERM and bounded graceful stop.

Do not weaken these settings merely to make a failing build convenient.

### 17.4 Persistent mounts

The model root is a read-only bind. The user-config directory is a read-write bind so the API can atomically replace `local.yaml`. Portable defaults remain read-only in the image.

Named volumes:

| Purpose | Container path | Default explicit volume name |
| --- | --- | --- |
| SQLite | `/data/database` | `local-ai-doctor-database` |
| Uploads | `/data/uploads` | `local-ai-doctor-uploads` |
| Framework cache | `/data/cache` | `local-ai-doctor-cache` |
| Exports | `/data/exports` | `local-ai-doctor-exports` |
| Backups | `/data/backups` | `local-ai-doctor-backups` |

`HF_HOME`, `TORCH_HOME`, and `XDG_CACHE_HOME` point into the cache volume. Explicit names survive Compose project renames. Changing a `*_VOLUME_NAME` selects different state and is a migration, not a cosmetic rename.

`Down` preserves volumes. Never add `--volumes` unless the owner explicitly requests permanent deletion of all corresponding data and exact targets have been verified.

### 17.5 Normal wrapper actions

Check and render configuration:

```powershell
.\scripts\wsl-docker.ps1 -Action Check
.\scripts\wsl-docker.ps1 -Action CheckNvidia
.\scripts\wsl-docker.ps1 -Action ConfigCpu
.\scripts\wsl-docker.ps1 -Action ConfigNvidia
```

Build and start updated source:

```powershell
.\scripts\wsl-docker.ps1 -Action BuildCpu
.\scripts\wsl-docker.ps1 -Action UpCpu
.\scripts\wsl-docker.ps1 -Action BuildNvidia
.\scripts\wsl-docker.ps1 -Action UpNvidia
```

`UpCpu`/`UpNvidia` include build/recreate behavior and stop the opposite Local AI Doctor profile so the fixed ports have one owner. They also start/reuse a named idle Linux keepalive process because WSL may otherwise stop despite systemd/Docker services, causing localhost ports to disappear.

Health, state, and logs:

```powershell
.\scripts\wsl-docker.ps1 -Action Ps
.\scripts\wsl-docker.ps1 -Action HealthCpu
.\scripts\wsl-docker.ps1 -Action HealthNvidia
.\scripts\wsl-docker.ps1 -Action NvidiaSmoke
.\scripts\wsl-docker.ps1 -Action LogsCpu
.\scripts\wsl-docker.ps1 -Action LogsNvidia
```

Stop without deleting data:

```powershell
.\scripts\wsl-docker.ps1 -Action StopCpu
.\scripts\wsl-docker.ps1 -Action StopNvidia
.\scripts\wsl-docker.ps1 -Action Down
```

URLs:

- UI: `http://127.0.0.1:6969/`
- Backend health: `http://127.0.0.1:6767/api/v1/health`
- Frontend health: `http://127.0.0.1:6969/healthz`

### 17.6 Backend-first readiness gate

Compose `frontend-*` depends on healthy `app-*`. `docker/frontend-entrypoint.sh` then:

1. Polls the backend health URL.
2. Requires `status: ok`, `database: ready`, and `worker: ready`.
3. Starts Nginx immediately after that complete readiness contract succeeds.

Preserve this behavior across daemon/container restarts. Worker readiness requires an IPC handshake from the spawned inference process; a live PID is insufficient. The health route is exposed only after database initialization, worker startup, model discovery, and incomplete-run recovery finish, so an additional timer is neither necessary nor authoritative.

### 17.7 NVIDIA profile and VRAM

NVIDIA execution requires all of:

- WSL GPU passthrough with `nvidia-smi` working inside the selected distribution.
- NVIDIA Container Toolkit configured for that Linux Docker daemon.
- A CUDA PyTorch build in the NVIDIA image.
- `gpus: all` or the reviewed `NVIDIA_VISIBLE_DEVICES` restriction.
- `LAD_RUNTIME__DEVICE=cuda` and no CPU fallback.

`UpNvidia` performs host/container CUDA preflight before stopping a working CPU profile, then starts the NVIDIA services and re-verifies CUDA and both health endpoints.

A loaded 1.5B-class BF16 checkpoint legitimately reserves roughly 3.5–4.3 GiB in the recorded environment. To release project-owned VRAM:

1. Use the UI Unload control, `POST /api/v1/models/resident/{model_key}/unload` for one resident, or `POST /api/v1/models/unload` for all of them.
2. Verify the runtime released references and called CUDA cache/IPC collection.
3. If the service is stale, gracefully stop the exact NVIDIA service.
4. Recheck `nvidia-smi` and ownership before touching any remaining process.

Never kill arbitrary GPU PIDs based only on memory size.

### 17.8 Database backup and restore

Create an online integrity-checked backup:

```powershell
.\scripts\wsl-docker.ps1 -Action Backup
```

Restore only while the active app is stopped and provide only the printed backup basename:

```powershell
.\scripts\wsl-docker.ps1 -Action StopNvidia
.\scripts\wsl-docker.ps1 -Action Restore -BackupFile workbench-YYYYMMDDTHHMMSSZ.sqlite3
.\scripts\wsl-docker.ps1 -Action UpNvidia
.\scripts\wsl-docker.ps1 -Action HealthNvidia
```

Use the CPU actions instead if CPU owns the database. Restore rejects paths outside `/data/backups`, integrity-checks source, restores through a temporary database, and atomically replaces the stopped database.

### 17.9 Safe Docker and WSL garbage collection

The historical incident grew the WSL VHD to about 234.76 GiB. Inspection found roughly 49 GiB of Docker build cache plus obsolete images/containers and small package/log caches; models were external and not duplicated in WSL. Docker-aware cleanup followed by a verified safe compaction reduced the physical VHD to about 20.76 GiB while preserving application volumes, uploads, backups, and models. These are incident figures, not expected steady-state sizes. The incident established this operational sequence.

Inspect first:

```powershell
.\scripts\wsl-docker.ps1 -Action Ps
```

Then, through the exact selected WSL distribution/user, inspect:

```bash
docker system df -v
docker ps -a --size
docker images
docker volume ls
docker builder du
```

Distinguish expected large CUDA/runtime layers from unique unused bytes; image virtual sizes share layers and are not directly additive. Verify that models remain an external bind rather than a WSL/image copy.

Potential routine cleanup candidates after inspection:

- Unused BuildKit cache with `docker builder prune` or, when intended, `docker builder prune -af`.
- Dangling images with `docker image prune`.
- Exact obsolete stopped test containers and their proven-unused images.
- Package-manager caches, pip caches, and old journal entries.

Do not:

- Use `docker system prune --volumes`.
- Delete active images merely because they are large.
- Delete named database/uploads/backups volumes.
- Delete or write the external model bind.
- Manually remove Docker/containerd storage directories.
- Unregister a WSL distribution based on a guessed name/path.

Free blocks inside a dynamic WSL VHD may not reduce the Windows file's physical size. Shut WSL down before any host-side compaction. Export/rebuild/import/unregister is destructive and requires explicit owner approval, exact resolved targets, adequate free disk, a verified export archive, preservation of the original default UID/user configuration, and post-import Docker/data/health checks. Delete a recovery archive only after successful verification.

## 18. Electron desktop application

Electron owns two loopback services:

- Packaged Python/FastAPI backend at `127.0.0.1:6767`.
- Electron static server/reverse proxy at `127.0.0.1:6969`.

`desktop/lib/lifecycle.cjs` contains the port constants and readiness validation. `desktop/main.cjs`:

1. Enforces a single instance.
2. Checks both fixed ports are free.
3. Starts the packaged backend.
4. Polls `/api/v1/health` for `status: ok`, `database: ready`, and `worker: ready`.
5. Confirms the backend did not exit after reporting ready.
6. Starts the frontend server/proxy immediately.
7. Navigates the already-visible native startup window to the production frontend.

The lightweight native startup window may be created before the gate so users
receive immediate feedback. Do not start or navigate to the production frontend
before complete backend readiness, and do not replace the readiness contract
with an arbitrary renderer timer.

Desktop persistent state uses Electron `app.getPath('userData')`, including writable `config/local.yaml`, SQLite, uploads, exports, backups, caches, runtime shutdown marker, and backend log. Models remain external. Uninstalling the application does not delete user data unless that behavior is explicitly changed.

Packaged shutdown writes a private per-launch random marker so a backend watcher can trigger normal Uvicorn lifespan cleanup. A Windows `taskkill` fallback targets the exact child tree only after the 20-second grace. Preserve graceful cleanup.

The desktop shell requests automatic device selection. Actual capability comes from the PyTorch runtime bundled at build time: GitHub releases bundle CPU PyTorch, while a local build from the pinned CUDA environment can select CUDA. Never force every packaged build to CPU, and never advertise CUDA when the bundled runtime cannot import or use it.

### 18.1 Local desktop build

Use a Python 3.12 environment with the locked base, matching PyTorch/Torchvision flavor, ML, desktop, and current project packages installed. GitHub uses CPU wheels; a local CUDA-capable build uses the pinned CUDA wheels. Then:

```powershell
node scripts/release-version.mjs check
npm --prefix frontend ci
npm --prefix desktop ci
npm --prefix desktop test
npm --prefix desktop run dist
.\desktop\scripts\smoke-backend.ps1
.\desktop\scripts\smoke-desktop.ps1
```

The two fixed ports must be free during desktop smoke tests, so stop the Docker stack first and restore it afterward if needed.

The only distributable result is `release/Local-AI-Doctor-<version>.exe`. The build checks installed project/version agreement and removes other generated items only within the bounded `release` directory.

## 19. Versioning and GitHub release automation

### 19.1 Version source

`VERSION` contains the canonical stable semantic version. Use:

```powershell
node scripts/release-version.mjs current
node scripts/release-version.mjs check
node scripts/release-version.mjs patch
node scripts/release-version.mjs minor
node scripts/release-version.mjs major
```

The helper synchronizes `pyproject.toml`, frontend package/lock, and desktop package/lock. Never edit one manifest independently.

Default rule:

- Completed ordinary feature or fix: `patch`.
- `minor`: only explicit owner request.
- `major`: only explicit owner request.
- Documentation-only work does not automatically require a version bump unless it changes the shipped product/release or the owner requests a release.

### 19.2 Branch channels

- `dev` push -> beta prerelease `vX.Y.Z-beta.<github.run_number>` and `Local-AI-Doctor-X.Y.Z-beta.<run>.exe`.
- `main` push -> stable immutable `vX.Y.Z` and `Local-AI-Doctor-X.Y.Z.exe`.

The release job runs only after backend, frontend, and desktop validation. It uses Windows and the CPU runtime, builds/smokes the EXE, enforces less than 2 GiB, exact filename, SHA-256 digest, target commit, channel, and exactly one uploaded custom asset.

Published releases/tags are immutable. An interrupted draft may be recovered and verified by immutable release ID, target SHA, unique title, asset digest, and integrity marker. Do not manually replace a published asset or retarget a tag. If a stable tag points elsewhere, bump `VERSION` and publish a new release.

## 20. Validation matrix

Run targeted tests while iterating and the complete relevant matrix before commit/push.

### 20.1 Backend

From repository root with the development environment installed:

```powershell
.\.venv\Scripts\ruff.exe format --check backend tests desktop/backend_launcher.py
.\.venv\Scripts\ruff.exe check backend tests desktop/backend_launcher.py
.\.venv\Scripts\mypy.exe backend
.\.venv\Scripts\pytest.exe -q -m "not real_model and not gpu and not performance"
```

CI installs locked development and ML dependencies plus explicit CPU PyTorch/Torchvision, verifies the `+cpu` build, and runs tests excluding `real_model` and `gpu`. The local default also excludes `performance` unless intentionally benchmarking.

Python expectations:

- Strict mypy.
- Ruff format and lint.
- Pytest strict config/markers.
- Public boundaries typed and validated.
- Test imports for optional ML modules such as Pillow/AV are covered by reviewed mypy overrides; do not remove runtime dependencies just to silence typing.

Backend test areas are split intentionally: unit coverage for config/discovery/adapters/hardware/sampling/reasoning/runtime/admission/repository; integration coverage for API/database lifecycle/event ordering/terminal durability/workspace replay and token branching; security coverage for body limits, transport/auth/origin/host, uploads, traversal, signatures, and expansion bounds.

### 20.2 Frontend

```powershell
npm --prefix frontend ci
npm --prefix frontend run lint
npm --prefix frontend run test:run
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

For CI browser setup, install Chromium with Playwright first. TypeScript is strict with unchecked-index and unused/floating-promise discipline. Update `package-lock.json` with dependency changes.

Important suites cover:

- Boundary normalization, strict requests, auth, attachment Blob handling, reconnect/resume.
- Bootstrap/offline/auth and race behavior.
- Live metrics, persisted restoration, TTFT, branching.
- Reasoning/protocol display, token formatting/height, attention heat, live strip.
- Nerd gating, alternative branch, attention, raw events, chart privacy.
- Composer, full controls, model roots, Markdown, context, capabilities, formatting.
- Desktop and Pixel 7/very narrow browser layouts using deterministic intercepted fixtures.

Playwright fixtures are not real inference evidence. Browser workers are intentionally bounded for stable Windows Vite transforms.

### 20.3 Desktop/release

```powershell
node scripts/release-version.mjs check
npm --prefix desktop ci
npm --prefix desktop test
npm audit --package-lock-only --prefix desktop --audit-level=high
```

For release-affecting changes, also build the payload and run packaged backend/startup-order smoke scripts.

### 20.4 Real-model and GPU evidence

Real model tests are opt-in and read-only:

```powershell
$env:LAD_REAL_MODEL_ROOT = 'X:\path\to\models'
.\.venv\Scripts\pytest.exe -m real_model
Remove-Item Env:LAD_REAL_MODEL_ROOT
```

Never automatically download a large substitute and never mutate the root. Mark GPU/performance tests correctly. State whether evidence came from deterministic fixtures, metadata/header inspection, or a real forward pass.

A valid benchmark records model fingerprint, hardware, backend/device/dtype, attention implementation, software versions, settings, instrumentation, cold/warm state, raw run IDs, and inclusion rules. Follow `docs/benchmarking.md`.

### 20.5 Pre-commit review

Before every milestone commit:

1. Confirm the active branch and upstream.
2. Inspect `git status`, full diff, and `git diff --check`.
3. Run the complete relevant checks.
4. Inspect for secrets, personal paths, model files, SQLite, uploads, caches, logs, telemetry, screenshots, and release artifacts.
5. Stage only the coherent task.
6. Use a concise professional commit message without AI/generated/co-author text.
7. Push only the intended branch through existing credentials.

## 21. Engineering rules

- Keep domain, inference, adapters, persistence, transport, configuration, and UI responsibilities separate.
- Prefer small cohesive modules and explicit dependency ownership.
- Keep heavyweight state in the worker and serialized messages bounded.
- Use typed boundary objects and structured safe errors.
- Keep high-volume writes off the decode hot path.
- Preserve ordering, cancellation, restart, and partial-result semantics in every refactor.
- Use dependency injection at hardware, clock, storage, and runtime seams where it improves deterministic tests.
- Comment non-obvious reasoning, not line-by-line mechanics.
- Do not leave dead code, fake production data, unbounded collections, abandoned TODOs, unexplained suppressions, or architecture-name hacks.
- Prefer stable metadata/probes over model-name allowlists.
- Keep APIs/events versioned and backwards-normalized deliberately.
- Pin important dependencies and update locks reproducibly.
- Use `apply_patch` for hand-authored repository edits and preserve unrelated user changes.
- Do not destructively reset, checkout, clean, or rewrite a dirty worktree.
- Search with `rg`/`rg --files` and inspect exact targets before mutations.

## 22. Common troubleshooting

### 22.1 “Backend offline” in the UI

1. Run the matching WSL `Ps`, logs, and health actions.
2. Confirm the backend health JSON rather than relying only on frontend text.
3. Inspect failed preflight: model bind exists/readable/read-only, `local.yaml` exists and is atomically writable, volumes are writable, database migration succeeded.
4. Use `UpCpu`/`UpNvidia` to rebuild/recreate updated source and establish the WSL keepalive.
5. Account for backend initialization; the frontend starts immediately after the complete readiness contract passes.
6. If only an optional bootstrap endpoint failed, fix frontend offline-state classification instead of hiding backend health.

### 22.2 Port conflict

- On Windows inspect listeners for 6767/6969.
- In the selected WSL Docker daemon inspect published ports.
- Resolve the exact owning process/container and stop only the in-scope conflict.
- Do not change permanent ports.
- Docker and Electron cannot run simultaneously because both own the same pair.

### 22.3 CUDA selected but CPU used

1. Confirm WSL `nvidia-smi`.
2. Run `CheckNvidia` and `NvidiaSmoke`.
3. Confirm the container has the pinned CUDA PyTorch build and `torch.cuda.is_available()`.
4. Confirm the NVIDIA profile, `LAD_RUNTIME__DEVICE=cuda`, GPU assignment, and no-fallback setting.
5. Inspect backend health, load result, run environment snapshot, and logs for the actual selected device.
6. Treat UI selection as intent, not proof.

### 22.4 Roughly 4 GiB of VRAM in use

A loaded supplied checkpoint can legitimately reserve this. Unload through the API/UI, wait for cleanup, then stop the exact NVIDIA backend if stale. Use `nvidia-smi` ownership evidence before touching anything else.

### 22.5 Updated source not visible

Use `UpCpu` or `UpNvidia`, which rebuilds/recreates; a plain restart can keep an old image. Verify health and then hard-refresh the browser. In native development, rebuild `frontend/dist` unless using Vite.

### 22.6 Frontend absent after backend health

Inspect frontend logs for the complete readiness gate. If readiness never succeeds, inspect backend startup and health details rather than adding a timer.

### 22.7 Reasoning ends without an answer

Inspect the finish reason, context remaining, warning events, reasoning slices, and terminal extra-answer fields. The backend should use at most one bounded recovery window for explicitly tagged models. The UI should show a no-answer/token-limit warning rather than blank success. Do not fabricate an answer or strip evidence.

### 22.8 Frontend Blob test fails only in CI

JSDOM/Node may provide different Blob realms, so `toBeInstanceOf(Blob)` can fail even for a correct Blob-like result. Preserve behavior-based checks (size, MIME, bytes/API contract) or ensure the implementation constructs the browser realm's Blob. Do not weaken authenticated attachment-boundary coverage.

### 22.9 Mypy cannot find Pillow or AV

Install the locked ML dependencies in the checking environment and retain reviewed `ignore_missing_imports` overrides for `PIL`, `av`, and heavy ML modules. Do not add fake stubs that mask runtime absence, and do not remove upload validation dependencies.

### 22.10 WSL disk grows unexpectedly

Use the inspection and safe-cleanup sequence in section 17.9. Builder cache is a common cause. Preserve volumes/models, then separately decide whether the shut-down VHD needs safe compaction.

### 22.11 Model-root save works in production but fails in Vite development

The model-root update route uses HTTP `PUT`. Same-origin production does not require a browser CORS preflight, while the Vite development origin can. Keep FastAPI `CORSMiddleware.allow_methods` synchronized with implemented state-changing methods and cover the `PUT` preflight in transport tests; do not bypass Origin/CSRF validation.

## 23. Known limitations that must remain visible

- One local user, one worker process (several resident models bounded by VRAM), one admitted job at a time.
- No universal Transformers compatibility guarantee.
- No production MoE router capture for current supplied models.
- No hidden-chain-of-thought access.
- Attention weights are bounded allocation summaries, not causal explanations.
- No full-vocabulary logit persistence; exact values are transient and bounded alternatives are retained.
- No continuous per-token GPU utilization or exact CPU attribution.
- No general hidden-state, activation, logit-lens, KV-cache, or prompt-cache inspection.
- No automatic telemetry-retention scheduler.
- No upload-file garbage collection after chat deletion.
- No general document text extraction, PDF inference, remote URL fetch, or archive extraction.
- No vector index/nearest-neighbor service or persisted dimensionality-reduction service.
- No TLS, identity system, multi-tenancy, or production internet exposure.
- CPU-only GitHub desktop binary; unsigned installer EXE.
- No selected open-source license.

If a UI section represents one of these goals, keep it disabled/explained. Never substitute fixture/demo values in a real session.

## 24. Definition of done for changes

A change is done only when all applicable items are true:

- The behavior is implemented end to end, not just specified or mocked.
- Current owner invariants in section 3 still hold.
- Configuration remains centralized and portable.
- Capability discovery and runtime support agree.
- Backend boundary, persisted schema/event order, frontend types, and UI state are synchronized.
- Failure, cancellation, reconnect, restart, and partial-result paths were considered.
- Security, path confinement, privacy, and data preservation were considered.
- Relevant unit, integration, security, frontend, browser, Docker, or desktop tests pass.
- Visual changes were checked at desktop and narrow widths.
- Documentation and limitation/capability claims are updated.
- Semantic version is bumped at the correct level when the completed work is a shipped feature/fix.
- Diff/status are reviewed and contain no private/generated artifacts.
- A coherent commit is created on `dev` and pushed only when authorized by the task/workflow.
- The working tree is clean, or every intentional uncommitted file is explained.

## 25. Guidance for future feature work

When adding a model family:

1. Define bounded discovery evidence and negative cases.
2. Produce a complete capability matrix with reasons.
3. Implement the worker path using local-only, no-remote-code loading.
4. Preserve prompt/template/special-token semantics and owned sampler math.
5. Add processor/instrumentation only where genuinely supported.
6. Wire typed parent/API/frontend boundaries and reason-preserving UI gates.
7. Add deterministic fixture coverage and optional real-model evidence.
8. Update adapter, API, metrics, limitations, and capability documentation.

When adding a metric:

1. Define the exact mathematical population/distribution and units.
2. Define the clock and synchronization behavior.
3. Define exclusions and unavailable cases.
4. Bound transient and persisted data.
5. Measure instrumentation overhead.
6. Add backend math tests, schema/event persistence tests, frontend labels/tooltips, exports, and docs.

When changing UI:

1. Preserve server-authoritative durable state and API normalization.
2. Preserve normal/Nerd privacy and protocol-marker boundaries.
3. Keep live and restored run behavior consistent.
4. Add accessible hints and capability reasons.
5. Test desktop, narrow, keyboard, and reduced-motion behavior.
6. Keep the blue visual identity and readability floor.

When changing deployment/release:

1. Preserve fixed ports and complete backend-readiness ordering in Docker and Electron.
2. Preserve model read-only and data-volume persistence.
3. Validate CPU and NVIDIA paths separately without silent fallback.
4. Preserve WSL-only Docker execution and hardening.
5. Validate version synchronization, desktop tests/smokes, one-EXE output, digest, and immutable release rules.

## 26. Final cautions

- Do not call the project complete merely because fixture tests pass; separate fixture, metadata, and real-forward evidence.
- Do not claim attention, perplexity, or alternatives prove truth or explain internal reasoning.
- Do not solve a port, GPU, permission, or storage problem by weakening the permanent contracts.
- Do not destroy user state, named volumes, model files, releases, tags, Git history, or a WSL distribution without explicit authority and exact-target verification.
- Do not allow convenience refactors to hide backend health, CPU fallback, capability reasons, persistence gaps, or unsupported modalities.
- When blocked by an optional capability, document the exact blocker, complete safe independent work, and report the smallest next action.
