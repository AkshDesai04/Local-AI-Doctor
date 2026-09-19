# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Orientation

Local AI Doctor is a local-first **inference and observability workbench** for local SafeTensors model directories. It generates text and embeddings entirely on the local machine and exposes the details: tokens, raw and sampler probabilities, timing, reasoning segments, and attention. It is a model-diagnostics tool. It is **not** a medical product. Nothing may call an external inference provider.

The codebase has four parts:
- **Backend**: Python 3.12, FastAPI, SQLite, and a spawned PyTorch/Transformers worker process (`backend/local_ai_doctor/`).
- **Frontend**: React 18, strict TypeScript 5.7, Vite 6 (`frontend/`).
- **Deployment**: hardened Docker Compose (CPU and NVIDIA profiles) that runs **only through WSL2** (`compose.yaml`, `Dockerfile`, `docker/`, `scripts/wsl-docker.ps1`).
- **Desktop**: an Electron Windows installer wrapping a PyInstaller-built backend (`desktop/`).

Version `0.1.3` (from `VERSION`).

### Sources of truth, in priority order
1. The owner's latest explicit request.
2. Security, data integrity, and privacy.
3. Tests and executable code. These define **current behavior**.
4. `AGENT.md`, the owner's ~1,500-line binding guide. It defines **product intent** and the invariants that must not regress (its §3). Read the relevant section before changing an area.
5. `docs/`: `architecture.md`, `api.md` (REST/WS contract), `configuration.md`, `metrics.md` (formulas), `adapters.md` (adding a model family), `deployment.md`, `benchmarking.md`, `limitations.md` (the canonical non-claims), `model-capability-report.md`, and `benchmark-results.md` (dated evidence, not current truth).

Never describe a partial or goal feature as implemented. When you add a capability, update the code, tests, `docs/limitations.md`, `docs/api.md`, `README.md`, and `AGENT.md` together.

## 2. Environment gotchas (read first)

- **Windows host.** The primary shell is PowerShell. The venv is `.\.venv\Scripts\*.exe`.
- **Git ownership mismatch.** Plain `git` fails with "dubious ownership" because the checkout is owned by a different Windows account. Use `git -c safe.directory=E:/Code/Personal/Local-AI-Doctor <cmd>` on every call. Don't change the global git config unless the user asks.
- **Run backend commands from the repo root.** `cli.py` and `main.py` resolve `config/default.yaml`, `config/local.yaml`, and `frontend/dist` relative to `Path.cwd()`.
- `config/default.yaml` paths are relative to the config dir: `../models` and `../data/...` point to repo-root `models/` and `data/`, and both are gitignored.
- Ignored and machine-local, never commit: `.env`, `config/local*.yaml`, `data/`, `runtime/`, `release/`, `models/`, `*.safetensors`, `*.sqlite3`, `*.npy`, `frontend/dist`, and caches.
- **Docker on Windows runs only through `scripts/wsl-docker.ps1`.** Never call Windows `docker.exe`. Never run `docker system prune --volumes`, and never pass `--volumes` to `down`. Never use sudo or loosen Docker socket permissions.

## 3. Commands

### Setup
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock -r requirements-ml.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --editable .
# Native CUDA only: afterwards install the exact wheels in requirements-cuda.lock (torch 2.8.0+cu128)
npm --prefix frontend ci
npm --prefix desktop ci
Copy-Item config\local.example.yaml config\local.yaml   # only if you need custom model roots
```
The lock files are exact pins: `requirements.lock` (web/runtime), `-ml` (torch/transformers/sentence-transformers/Pillow/av), `-dev`, `-cuda`, and `-desktop` (pyinstaller). `pyproject.toml` mirrors them as extras. Never mix CPU and CUDA torch builds casually.

### Backend checks (identical to CI)
```powershell
.\.venv\Scripts\ruff.exe format --check backend tests desktop/backend_launcher.py
.\.venv\Scripts\ruff.exe check backend tests desktop/backend_launcher.py
.\.venv\Scripts\mypy.exe backend                     # strict mode + pydantic plugin
.\.venv\Scripts\pytest.exe -q -m "not real_model and not gpu and not performance"
```
Single tests:
```powershell
.\.venv\Scripts\pytest.exe tests/backend/test_sampling.py -q
.\.venv\Scripts\pytest.exe tests/backend/test_sampling.py::<test_name> -q
.\.venv\Scripts\pytest.exe tests/integration -k branch -q
```
- Pytest config (`pyproject.toml`): `pythonpath=backend`, `asyncio_mode=auto`, `--strict-markers --strict-config`.
- Markers: `real_model`, `gpu`, `performance`.
- CI sets `LAD_PROFILE=test` and excludes `real_model` and `gpu`.
- Ruff: line length 100 (E501 ignored), rules `E,F,I,UP,B,SIM,ASYNC,RUF`, target py312.
- Mypy overrides ignore missing imports for `PIL`, `av`, `transformers.*`, `torch.*`, and `sentence_transformers.*`. Do not add fake stubs.

Real-model tests are opt-in and read-only. They never download substitutes.
```powershell
$env:LAD_REAL_MODEL_ROOT = 'X:\path\to\models'; .\.venv\Scripts\pytest.exe -m real_model; Remove-Item Env:LAD_REAL_MODEL_ROOT
```

### Frontend checks
```powershell
npm --prefix frontend run lint        # eslint, type-checked rules, --max-warnings=0
npm --prefix frontend run test:run    # vitest + jsdom, src/**/*.test.ts(x)
npm --prefix frontend run build       # tsc -b && vite build -> frontend/dist
npm --prefix frontend run test:e2e    # Playwright; starts Vite on :4173; projects desktop-chromium + mobile-chromium (Pixel 7); workers=2
npm --prefix frontend run test:run -- src/components/ChatView.test.tsx        # single vitest file
npm --prefix frontend run test:e2e -- -g "branch" --project desktop-chromium  # single Playwright test
```
- CI runs `npx playwright install --with-deps chromium` first.
- Playwright intercepts `**/api/v1/**` with fixtures, so e2e results are **not** evidence of real inference.
- `npm run screenshots` regenerates the `docs/screenshots/` images.

### Run natively
```powershell
.\.venv\Scripts\local-ai-doctor.exe serve --profile native-windows   # API + built UI on http://127.0.0.1:8000
npm --prefix frontend run dev                                        # http://127.0.0.1:5173, proxies /api and /ws to :8000
.\.venv\Scripts\local-ai-doctor.exe config    # redacted effective config JSON
.\.venv\Scripts\local-ai-doctor.exe scan      # discovery report JSON (no weights loaded)
.\.venv\Scripts\local-ai-doctor.exe serve --set runtime.device=cpu --set inference.instrumentation=full
```
- CLI flags: `serve|scan|config`, `--profile`, `--config`, `--user-config`, and repeated `--set a.b=value`.
- Interactive OpenAPI at `/api/docs` exists only under the `development` profile. The schema at `/api/v1/openapi.json` is always available.

### Docker (via WSL only)
```powershell
.\scripts\wsl-docker.ps1 -Action Check          # also: CheckNvidia, ConfigCpu, ConfigNvidia
.\scripts\wsl-docker.ps1 -Action UpCpu          # build+recreate, stops the other profile, starts WSL keepalive
.\scripts\wsl-docker.ps1 -Action UpNvidia       # CUDA preflight first
.\scripts\wsl-docker.ps1 -Action HealthCpu      # also: HealthNvidia, NvidiaSmoke, LogsCpu, LogsNvidia, Ps
.\scripts\wsl-docker.ps1 -Action StopCpu        # also: StopNvidia, Down (both preserve volumes)
.\scripts\wsl-docker.ps1 -Action Backup
.\scripts\wsl-docker.ps1 -Action Restore -BackupFile workbench-YYYYMMDDTHHMMSSZ.sqlite3   # only while the app is stopped
```
- Setup: copy `.env.example` to `.env` and set `MODEL_PATH`, a WSL-visible Linux path.
- URLs: UI `http://127.0.0.1:6969/`, API health `http://127.0.0.1:6767/api/v1/health`, frontend health `:6969/healthz`.
- A plain restart can keep an old image. Use `UpCpu`/`UpNvidia` to pick up source changes.

### Desktop and release
```powershell
node scripts/release-version.mjs check | current | patch | minor | major
npm --prefix desktop test                       # node --test desktop/test/*.test.cjs
npm --prefix desktop start                      # dev Electron; uses .venv python or $env:LAD_DESKTOP_PYTHON; needs frontend/dist
npm --prefix desktop run dist                   # frontend build + PyInstaller + NSIS -> release/Local-AI-Doctor-<ver>.exe
.\desktop\scripts\smoke-backend.ps1; .\desktop\scripts\smoke-desktop.ps1   # ports 6767/6969 must be free (stop Docker first)
npm audit --package-lock-only --prefix desktop --audit-level=high
```
- `$env:PYTHON` selects the packaging interpreter.
- `build-backend.ps1` refuses to package if the installed `local-ai-doctor` version differs from `VERSION`.

## 4. Repository map

```
backend/local_ai_doctor/
  main.py            create_app(): lifespan composition, middleware, ALL REST + WS routes, SPA fallback
  cli.py             `local-ai-doctor` entry (serve/scan/config)
  config.py          AppSettings (strict pydantic), SettingsLoader, BUILTIN_PROFILES, persist_user_model_roots
  errors.py          WorkbenchError hierarchy -> {"error":{code,message,retryable,hint?,details?}}
  api/schemas.py     strict request models (extra="forbid") + browser camelCase shims; chat-workspace export schema
  api/body_limit.py  ASGI byte-counting body limit (413 limit_exceeded)
  domain/            ModelDescriptor, ModelTask, Capability enum, CapabilityMatrix (non-full needs reason)
  discovery/         scanner.py (bounded read-only scan), safetensors.py (header parse), fingerprint.py, capabilities.py
  hardware/          probe.py (psutil/torch/nvidia-smi inventory), selection.py (device/dtype choice)
  workers/           supervisor.py (async side, spawn ctx), runtime.py (in-worker model code), admission.py (FIFO lease)
  services/          models.py (ModelRegistry), runs.py (RunManager), events.py (EventBroker), uploads.py, workspace.py
  persistence/       database.py (aiosqlite, migrations, TelemetryWriter batcher), repository.py (all SQL), migrations/*.sql
  sampling/metrics.py  numpy fp32 reference math for distributions/ranks/perplexity (see §5.7)
  reasoning/segments.py  streaming tag segmenter (<think>…</think>) with code-point slices
  adapters/          ModelAdapter/GenerationAdapter/... ABCs + AdapterRegistry (contracts only; not wired into worker)
frontend/src/
  main.tsx, App.tsx  App.tsx = presentation state only (view, drawers, Nerd Mode, inspector tab, selected token)
  hooks/useWorkbench.ts  THE orchestration hook (bootstrap, chats, runs, streaming, restore, branch, uploads)
  api/client.ts      all fetch + normalization (snake_case -> typed), subscribeToRun WS client; api/types.ts types
  api/auth.ts        tab-scoped token (sessionStorage `lad.auth_token`), events lad:auth-required / lad:auth-changed
  api/modelRoots.ts  GET/PUT /configuration/model-roots
  components/        ChatView, Composer, ContextMeter, Inspector, VirtualTokenTable, TraceChart, AttentionAttribution,
                     GenerationControls, ModelRegistry, ModelRootSettings, EmbeddingsWorkspace, Sidebar, WorkbenchHeader
  domain/capabilities.ts  isUsable() etc. for capability gating
  utils/format.ts    displayTokenText / tokenTextHint (whitespace glyphs ␠ ↵ ⇥), formatting, downloadBlob
  utils/markdown.ts  splitAssistantOutput(): reasoning/answer/termination-marker split for normal mode
  tests/workbench.spec.ts  Playwright with intercepted fixture API
desktop/             main.cjs, lib/lifecycle.cjs (ports, readiness), lib/frontend-server.cjs, backend_launcher.py, backend.spec
docker/              entrypoint.sh (uvicorn), frontend-entrypoint.sh (readiness gate), nginx.conf, preflight.py, sqlite_backup.py
tests/backend        unit: config, discovery, adapters, hardware, sampling, reasoning, worker_runtime, admission, repository
tests/integration    API workbench, DB lifecycle, event broker, terminal ordering, workspace portability, real-model
tests/security       body limits, transport (host/origin/auth/CORS), upload store
```

## 5. Backend architecture

### 5.1 Topology and process boundary
```
Browser/Electron -> static UI + same-origin /api/v1 + /ws/v1
  -> (Docker: nginx :6969 | Electron: frontend-server.cjs :6969 | dev: Vite :5173 | native: FastAPI serves frontend/dist)
  -> FastAPI (:6767 Docker/Electron, :8000 native) -> ModelWorkerSupervisor
       --multiprocessing "spawn" Queues (bounded dicts only)--> worker process: workers/runtime.py WorkerRuntime
```
- Model, tokenizer, processor, CUDA state, KV caches, and full-vocabulary tensors exist **only** in the worker.
- Worker commands are dicts with `op` set to one of `load | unload | generate | embed | score_prompt | shutdown`.
- The worker sends back `ready`, `reply` (`request_id`, `ok`, `payload|error`), `run_event` (`run_id`, `event_type`, `payload`), and `diagnostic` (a bounded traceback that stays in logs).
- Errors crossing the boundary go through `_safe_error`, which returns a generic code/message plus the exception class name only, with no paths or exception text.
- The worker sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `CUBLAS_WORKSPACE_CONFIG` before torch is imported.
- Supervisor details: `available` requires the IPC ready handshake, not just a live PID. A timeout "poisons" the worker, which gets retired and replaced. If it cannot be terminated, the supervisor fails closed. One active run at a time.

### 5.2 Startup and shutdown (`main.py` lifespan). The order matters.
1. `Database.initialize()` opens SQLite with WAL and foreign keys. It backs up an existing DB to `paths.backups` before applying pending migrations.
2. `TelemetryWriter.start()` starts batched writes (`write_batch_size`, `write_flush_interval_ms`).
3. `EventBroker` is created with `persist_event` and replay from `raw_events`.
4. `ModelWorkerSupervisor.start()` spawns the worker and waits for the handshake.
5. `ModelRegistry.refresh()` probes hardware, scans roots, and upserts models and capabilities.
6. `repository.recover_incomplete_runs()` marks queued/loading/running runs failed with `interrupted_by_restart`.
7. `RunManager`, `UploadStore`, and `WorkspaceService` are built into `app.state.services` (`ApplicationServices` dataclass). Routes get them via `_services(request)`.

Shutdown runs in reverse: `runs.close()`, worker close with the grace period, `telemetry.stop()`, `database.close()`. `/api/v1/health` returns `{"status":"ok","database":"ready","worker":"ready","loaded_model":…,"protocol_version":1}`, or 503 `degraded` when the worker is unavailable. The Docker and Electron readiness gates parse exactly these fields.

### 5.3 Request pipeline (`main.py`)
Middleware order, outermost first:
1. `trusted_host_transport`: Host allow-list against DNS rebinding. Allowed hosts are the server host, the hosts from `allowed_origins`, and loopback aliases when the server host is loopback. Otherwise 400 `host_rejected`.
2. `CORSMiddleware`: allowed `allow_origins`, methods `GET, POST, PATCH, DELETE`.
3. `RequestBodyLimitMiddleware`: JSON ≤ `limits.prompt_bytes + 1 MiB`, multipart ≤ `upload_bytes + 1 MiB`, else 413.
4. `secure_transport`:
   - Bearer auth when `server.authentication_token` is set. Static-shell GETs are exempt. Otherwise 401 `authentication_required`.
   - CSRF Origin check on POST/PATCH/PUT/DELETE. Otherwise 403 `origin_rejected`.
   - Security headers: CSP, `nosniff`, `DENY`, `no-referrer`, Permissions-Policy, and `no-store` on `/api/`.

Error shapes:
- `WorkbenchError` → `{"error": exc.to_dict()}` with the class's `http_status`. See `errors.py` for the code→status map, e.g. `capability_unavailable` 409, `limit_exceeded` 429/413, `out_of_memory` 507, `worker_busy` 409.
- Validation → 422 `invalid_request` with `details.issues`.
- Uncaught errors → 500 `internal_error` with no text.
- Some routes raise plain `HTTPException`, giving a `{"detail": …}` shape. Clients handle both shapes.

The WebSocket `/ws/v1/runs/{run_id}?after=N` checks Host (close 4403), auth via the `lad.auth.<base64url>` subprotocol or Bearer (close 4401), and Origin (4403). It then accepts with subprotocol `lad.events.v1`. A disconnect does **not** cancel the run.

Routes (all under `/api/v1`):
- **Service:** `health`, `configuration`, `GET/PUT configuration/model-roots`, `hardware`, `storage`, `DELETE storage/telemetry?before=<tz-aware>&confirm=true`.
- **Models:** `models`, `models/refresh`, `models/{id}/load` (body `{device?,dtype?}`), `models/unload`, `models/{id}/unload`, `models/{id}/inspect`.
- **Chats:** `chats` (POST, GET `?search=&archived=`), `chats/import`, `chats/{id}` (GET, PATCH, DELETE), `chats/{id}/export`, `chats/{id}/messages` (GET, POST), `DELETE chats?confirm=true&include_archived=`.
- **Runs:**
  - `runs` or `runs/generation` (202; returns `runId`, `messageId`, `websocketUrl` plus `run`/`user_message`/`assistant_message`)
  - `runs/{id}` (run with `tokens`, `branchable_through_token_index`, `phases`, `environment`, `summary`, `warnings`)
  - `runs/{id}/events`, `runs/{id}/cancel`, `runs/{id}/replay`
  - `runs/{id}/branch` (body `{tokenIndex, distribution: "raw"|"sampling", rank, tokenId}`)
  - `runs/{id}/export?format=json|jsonl|csv`, `runs/compare/summary?ids=` (2 to 8), `runs/prompt-score`
- **Media:** `attachments` or `uploads` (multipart `model_id` + `file`), `attachments/{id}/content`, `embeddings` or `runs/embeddings`.

Anything outside `/api` and `/ws` falls back to `frontend/dist/index.html`.

### 5.4 Configuration (`config.py`)
- `AppSettings` is strict (`extra="forbid"`) at schema version 1. Groups: `paths, server, runtime, inference, limits, workers, telemetry, logging, features, platform`.
- Precedence, lowest to highest:
  1. pydantic defaults
  2. `config/default.yaml` `defaults:`
  3. `BUILTIN_PROFILES[profile]`
  4. `default.yaml` `profiles.<profile>`
  5. user-local file (`config/local.yaml`): `defaults`, then its profile
  6. `LAD_SECTION__FIELD` env vars (double underscore nests; values parsed as scalars/JSON, e.g. `LAD_SERVER__ALLOWED_ORIGINS='["..."]'`)
  7. `--set section.field=value`
- Profile selection: `--profile` > `LAD_PROFILE` > `active_profile` > `development`.
- Profiles: `development, test, native-windows, native-wsl, container-cpu, container-nvidia, production`.
- Reserved env vars: `LAD_PROFILE`, `LAD_CONFIG` (portable file path), `LAD_USER_CONFIG` (writable user file path).
- **Only `config.py` reads `LAD_*` variables.** `main.py` has narrow exceptions for locating the user config and detecting model-root overrides.
- Relative paths resolve against the directory of the portable config.
- `redacted_effective_config()` hides paths, the WSL distribution, and the token. `inference_snapshot()` (with a sha256 digest) is persisted on every run.
- `PUT /configuration/model-roots`:
  - Validates that roots are absolute, existing, readable, not a filesystem root, and not duplicates.
  - Refuses while a model is loaded or while lifecycle/admission is busy.
  - Refuses while `LAD_PATHS__MODEL_ROOTS` or `--set paths.model_roots` overrides are active; the UI shows the reason.
  - Writes the active profile section of the user file atomically, falling back to a guarded fsync overwrite on bind mounts, then rescans.
- Typed-but-unimplemented options: non-`none` quantization, `cpu_offload`, multi-GPU, most `features.*` probes, and the `retention_days` scheduler. `features.attention_probe` is **not** the attention gate; instrumentation `full`/`expert` is.
- Portable defaults:
  - Context 4096 (fallback only, when a checkpoint declares none), max prompt tokens 32768, reserved output 512.
  - Sampling: max output 512, temperature 0.7, top-k 50, top-p 0.95, min-p 0, penalties 1/0/0, 10 alternatives.
  - Instrumentation `token`, queue limit 32.
  - Limits: prompt 4 MiB, upload 100 MiB, 16 attachments, 100k token events and 128 MiB of trace per run.
  - Worker timeouts: startup 120 s, load 600 s, inference 3600 s, shutdown 15 s.

### 5.5 Discovery and capabilities (`discovery/`)
- **Scan:** depth 2 below each root, without following directory symlinks. A candidate folder has `config.json` or a SafeTensors file.
- **Bounds:** JSON ≤ 32 MiB, SafeTensors header ≤ 256 MiB, ≤ 1,000,000 tensors. Offsets and dtypes are validated without reading weights.
- **Task detection** (`_detect_task`) works from metadata:
  - SentenceTransformers files or architecture names containing embedding/sentence/featureextraction → `embedding`, or `multimodal_embedding` when there is more than one modality.
  - `is_encoder_decoder`/seq2seq → `encoder_decoder_generation`.
  - `*ForCausalLM`/`*ForConditionalGeneration` → `text_generation`.
  - Anything else → `unknown` with `loadable=false`. Unknown models stay visible with diagnostics.
- **MoE:** only config keys (`num_experts`, `num_local_experts`, `n_routed_experts`, …) or router tensors count. A dense `gate_proj` is **not** MoE.
- **Context:** every context candidate is recorded, and the most authoritative one is selected by tier: `sentence_bert_config.max_seq_length` → the architecture's declared capacity (`max_position_embeddings`, plus the family aliases `max_seq_len`/`n_positions`/`seq_length`/`max_sequence_length`/`n_ctx`, also under `text_config`) → `tokenizer_config.model_max_length` → `inference.conservative_context_limit` (fallback only, never a cap). `sliding_window`, `rope_scaling.original_max_position_embeddings`, and values above 10,000,000 are evidence only. Prompt size is bounded by `inference.max_prompt_tokens`.
- **Reasoning delimiters** come from tokenizer metadata.
- **Capabilities:** `CapabilityMatrix` must contain every `Capability` enum member. States are `full | partial | unsupported | unavailable_on_backend`, and every non-full state carries a reason. `build_capability_matrix(ModelEvidence)` never promotes unknowns.
- **Fingerprint** (quick policy): hashes metadata files fully, plus SafeTensors headers and sizes. It is identity evidence, not a full-file integrity digest. A changed fingerprint means a new identity; historical runs keep their fingerprint.
- Deleting model metadata never touches files. Model roots are read-only, always.

### 5.6 Hardware, admission, lifecycle
- `select_hardware`: `cpu` → CPU, FP32 by default (explicit CPU FP16 is rejected). With `auto`/`cuda`, the first torch-usable CUDA device is chosen, with BF16 on compute capability ≥ 8 and FP16 otherwise. If CUDA is requested but unavailable and `allow_cpu_fallback=false`, it raises `BackendUnavailableError`. `nvidia-smi` presence is not proof of a usable CUDA runtime.
- Docker sets `LAD_RUNTIME__ALLOW_CPU_FALLBACK=false` for **both** profiles. Built-in `container-cpu` alone would allow fallback; compose overrides that.
- `SingleWorkerAdmission`: generation, embeddings, and prompt scoring share one FIFO lease. Capacity is 1 active plus `queue_limit` waiters. Overflow returns 429 **before** any chat or run rows are written. `admission.lifecycle(op)` serializes load, unload, and model-root changes against admitted work.
- `ModelRegistry.load_reserved` reuses a load only when model identity and the complete runtime selection match. A different model is unloaded first. RAM/VRAM budget checks are lower-bound preflights only.

### 5.7 Generation (`services/runs.py` RunManager + `workers/runtime.py`)
`create_generation` runs these steps:
1. Validates the task, rejects attachments (the generation adapter has no media), checks that parent lineage is in the same chat, and checks rendered history bytes.
2. `reserve_inference`.
3. Chooses hardware and the effective seed (`secrets.randbits(64)` if omitted; seed 0 is valid). Merges only caller-set sampling fields over config defaults (`_effective_sampling`).
4. Builds the `reproducibility` record.
5. Calls `repository.create_generation_setup`, which writes the user message, pending assistant, queued run, and environment snapshot in one transaction and auto-titles "New chat" from the prompt.
6. Starts the background task `_execute_generation`.

`_execute_generation` then:
1. Publishes `run_created`.
2. Waits for the lease and sets the run to `loading`.
3. Publishes `stage: model_loading`, loads the model, and publishes `model_loaded`.
4. Sets the run to `running`.
5. Streams worker events:
   - `stage: prefill` persists `rendered_prompt`, `prompt_token_count`, `reasoning_primed`, and the template/tokenization phase metrics.
   - A `metric` with `prefill_ms` records the prefill phase.
   - Each `token` applies `output = output[:replace_from] + display_text`. Tokens are persisted via `TelemetryWriter` within the event/byte limits; hitting a limit emits one `telemetry_persistence_truncated` warning. The assistant text is checkpointed every **8** tokens.
   - `completed`/`cancelled`/`error`: records the generation phase, **flushes telemetry**, then finalizes the message and run.
6. On any exception it flushes, marks the message and run failed, and publishes exactly one `error`. Partial output is always preserved.

Cancel works in two ways. A run still waiting in the queue is cancelled in place (`cancelled_before_start`). A running run is cancelled through the worker cancel event, which is checked between decode steps.

Inside the worker (`_generate_impl`):
1. Renders the chat template with `add_generation_prompt=True`. The request's `reasoning` is forwarded as `enable_thinking` only to compatible templates. Without a template, a deterministic "Role: content" fallback is used and a `chat_template_unavailable` warning is emitted.
2. Tokenizes with `add_special_tokens=False`.
3. Causal models prefill, then reuse the KV cache. Encoder-decoder models encode once and resolve the decoder start token.
4. The owned sampler runs, penalties first (`_apply_penalties`), then `_filter_distribution`. Fixed order: repetition → frequency → presence → temperature (0 = argmax) → top-k → top-p → min-p → renormalize → `torch.multinomial` with a seeded `torch.Generator`.
5. Stop sequences are detected **after** a token is decoded, so the matched stop text stays in the output.

- **Two sampler implementations exist.** The live path is the torch code in `workers/runtime.py`. `sampling/metrics.py` is a numpy fp32 reference (unit-tested in `test_sampling.py`; `SamplingSettings` is also used by the adapter contracts). If you change sampler semantics, keep both consistent and test both.
- **Instrumentation tiers:**
  - `off`/`basic`: lightweight chosen-token data and timing.
  - `token`: exact full-vocabulary raw log-prob and rank, entropy, perplexity, and bounded alternatives.
  - `full`/`expert`: adds CUDA sync and bounded attention capture.
- **Raw vs sampler.** The raw distribution is the full-vocab fp32 log-softmax of the model logits. The sampler distribution is the post-pipeline renormalized one. Keep them separately named everywhere. In the DB and API the sampler distribution is the string `"sampling"`; in UI copy it is "sampler".
- **Alternative lists** are bounded views and are never renormalized. The chosen token is always in its own token row.
- **Metric formulas:**
  - `raw_rank = 1 + #(z_j > z_i) + #(z_j == z_i and j < i)`.
  - `entropy`/`surprise` use the sampler distribution. `cumulative_logprob` and `running_perplexity = exp(-cum_raw_logprob / n)` use the **raw** distribution.
  - Full definitions are in `docs/metrics.md`.
- **Reasoning** (`reasoning/segments.py`):
  - `TagReasoningSegmenter` classifies streamed text into `reasoning | answer | unknown`, emitting code-point `reasoning_slices` for tokens that cross a boundary. It handles delimiters split across tokens and prompt-primed `<think>`.
  - It segments only emitted markers. Never infer hidden chain-of-thought.
  - If a tagged model runs out of budget before any visible answer, **one** extra answer window is allowed (clipped to the remaining context) and `reasoning_answer_allowance_activated` is emitted. If there is still no answer, `reasoning_answer_missing` is emitted and the UI shows a warning, never a blank success.
- **Attention attribution** (full/expert, decoder-only):
  - Takes the eager-attention final-query row from the forward pass that produced each token's logits and averages it over captured layers and heads (`_mean_causal_self_attention`).
  - Keeps the top **128** sources (`_ATTENTION_SOURCE_LIMIT`) and records the exact omitted mass.
  - The prompt token catalogue is stored once, on generated token 0.
  - If capture fails, it emits `attention_capture_unavailable` and generation continues. After any failed request the worker restores the attention implementation.
  - Always describe it as "post-softmax self-attention allocation averaged across captured layers and heads". It is not causal attribution.
- **Deterministic reference mode:** a per-request `torch.use_deterministic_algorithms`, with cuBLAS workspace `:4096:8` and cudnn benchmark off. Global state is restored afterward.
- **Replay** (`replay_generation`): requires a complete source run with an unchanged fingerprint. It rebuilds the lineage and settings and creates a sibling assistant branch with `parent_run_id` set.
- **Token branch** (`branch_generation`):
  - Validates `tokenIndex`/`distribution`/`rank`/`tokenId` against a persisted alternative, and requires contiguous durable tokens through the branch point (`branchable_through_token_index`).
  - Forces the original prefix through N-1 (`forced_prefix_token_ids`), substitutes the alternative at N, and samples the rest normally.
  - Creates a **new chat and run**. The source stays immutable, the new run has no cross-chat `parent_run_id`, and provenance is stored separately.
- **Embeddings** (`create_embedding` → worker `_embed`, SentenceTransformers):
  - The route checks modality per item: text must have text only, mixed needs text plus one attachment, media must have an attachment only.
  - Optional Matryoshka `dimensions` truncate the vector and re-normalize it.
  - The route computes `similarityMatrix` as a **dot product**, which is cosine only when vectors are normalized.
  - The Qwen3-VL embedding loader applies a key mapping `^model\.` → `""` (`_embedding_model_kwargs`). Don't remove it without revalidation.
- **Prompt scoring:** a teacher-forced causal pass that excludes the first position.

### 5.8 Events (`services/events.py`)
- Envelope: `{version:1, run_id, sequence (per-run, from 1), type, monotonic_ns, server_time, payload}`.
- Types: `run_created, stage, model_loaded, metric, token, warning, completed, cancelled, error`, plus the synthetic `resync_required` sent to slow subscribers. Its `payload.resume_after` is the last delivered sequence; don't advance the cursor to the synthetic sequence.
- `publish` is serialized per run in this order: assign sequence → persist → fan out.
- `persist_event` in `main.py` sends `token` events to the batched writer (bounded per run) and commits all other events immediately. Before a terminal event it flushes pending token writes.
- `subscribe(run_id, after)` registers first, then replays from the DB, then streams live, de-duplicating by sequence. The subscriber queue size is 256.

### 5.9 Persistence (`persistence/`)
- One aiosqlite connection, WAL, foreign keys, a write lock, and explicit transactions. **All SQL lives in `repository.py`**, except the token and alternative INSERTs in `runs.py` and the raw-event INSERT in `main.py`, which go through `TelemetryWriter.submit`.
- Migrations: `0001_initial.sql`, `0002_token_reasoning_slices.sql` (`reasoning_slices_json`), `0003_token_attention_attribution.sql` (`attention_attribution_json`). They are **append-only**: add `000N_*.sql` and never edit an applied one.
- Tables:
  - `models` and `model_capabilities`
  - `chats` (pinned, archived) and `messages` (a parent_id tree with `branch_index`; status `pending|streaming|complete|cancelled|failed`)
  - `attachments` (sha256, content-addressed) and `message_attachments`
  - `inference_runs` (kind `generation|embedding|prompt_score|benchmark`; status `queued|loading|running|complete|cancelled|failed|disconnected`; seeds, settings/effective_config/reproducibility JSON, timestamps)
  - `environment_snapshots` and `phase_metrics`
  - `token_events` (PK run_id + token_index; all per-token metrics and `segment`) and `token_alternatives` (distribution `raw|sampling`)
  - `router_events` and `router_aggregates` (unused today)
  - `embedding_runs` and `embedding_inputs`
  - `raw_events` (PK run_id + sequence)
- Retention runs only through the explicit `DELETE /storage/telemetry` call. It removes whole terminal runs and keeps chats and messages. Upload files are never garbage-collected, even after the chat is deleted.

### 5.10 Uploads and workspaces
- `UploadStore`:
  - Streams uploads under a byte limit and sniffs content signatures: PNG, JPEG, GIF, WebP, MP4, WebM, WAV, FLAC, MP3, PDF, UTF-8 text.
  - Fully decodes images and video within the pixel, frame, and duration limits, then stores files as `<sha256>.<ext>`.
  - Reduces names to basenames and rejects traversal and symlinks.
  - Recognizing a type does not mean accepting it. Acceptance requires the model's modalities, and text/PDF extraction is not implemented.
- `WorkspaceService`: path-free chat export/import (`schema: "local-ai-doctor/chat-workspace"`, `schema_version: 1`, `ChatWorkspaceDocument` in `schemas.py`). Import validates the parent graph and cycles, remaps every ID, turns non-terminal runs into failed ones, and nulls missing model IDs while keeping the fingerprint. Exports exclude attachments, paths, raw events, environment snapshots, and embeddings.

## 6. Frontend architecture
- There is no global store.
  - `useWorkbench()` owns the domain state: health, config, models, chats (active and archived), messages, `selectedRun`, `runningRunId`, streaming, settings and their configured defaults, attachments, errors, and notices.
  - `App.tsx` owns only the UI state.
  - Components receive normalized, typed data.
  - All API normalization lives in `api/client.ts`, via `normalizeRun`, `normalizeToken`, `normalizeStreamEvent`, `normalizeModel`, and similar functions. **Don't cast raw API data inside components.**
- **Bootstrap** calls `Promise.allSettled` over health, models, active chats, archived chats, and configuration. It distinguishes offline, partial failure, and auth-required. One optional endpoint failing must not show "backend offline". Health is lifecycle truth for which model is loaded.
- **Live generation:**
  1. Optimistic user message.
  2. POST.
  3. Swap in the durable IDs.
  4. Streaming assistant.
  5. `subscribeToRun` over WS with `after=<last seq>`, reconnect, and de-duplication.
  - `runSelectionEpoch` and `activeChatIdRef` guard against stale fetches.
  - Each token updates the text (honoring `replaceFrom`), token count, rolling TPS, and running perplexity.
  - Browser first-token time and first-visible-answer time are tracked separately.
  - On a terminal event the hook refetches the durable run and merges client timing. Client timing is stored in localStorage `local-ai-doctor.client-telemetry.v1`, capped at 100 runs. The backend stays authoritative.
- **Branches:** ChatView renders one root-to-leaf lineage and offers sibling switchers. Regenerate calls replay and creates a sibling. Editing a message regenerates with the correct parent. Opening a chat restores the latest run into the inspector. Raw events are lazy-loaded only when the inspector is open, Nerd Mode is on, and the Raw Events tab is selected.
- **Normal mode vs Nerd Mode:**
  - Normal mode uses safe Markdown: `react-markdown`, GFM, math/KaTeX, and raw HTML is **never** rendered. `splitAssistantOutput` hides protocol and termination markers and puts reasoning in a collapsed `<details>` labeled `Thinking…`. Copy strips markers.
  - Nerd Mode shows raw tokens colored by raw probability, sampler probability, surprise, latency, or segment. Clicking a token locks the inspector selection, which exposes raw and sampler alternatives and "Branch out with selected token". Very old token boundaries collapse past 1,200.
  - Token chips are a fixed 30 px high and use `displayTokenText`/`tokenTextHint`, keeping the raw piece in the tooltip.
- **Inspector tabs:** Overview, Tokens, Probability, Timing, Experts, Context, Embeddings, Hardware, Configuration, Raw Events. Each is capability-gated with an exact reason, and none is ever filled with fake data. Chart tooltips outside Nerd Mode must not leak hidden token text.
- **Composer:** a popover sets reasoning, temperature, top-k, and top-p for the next response. Enter sends and Shift+Enter inserts a newline. `GenerationControls` exposes the full settings: token, device, dtype, instrumentation, seed, deterministic mode, all sampler knobs, stop sequences, and reset to backend defaults. Ctrl/Cmd+K creates a new chat when connected.
- **Styling and accessibility:**
  - Accent blue `#60a5fa`, never green. DM Sans for UI, JetBrains Mono for data.
  - Dense text has a 10–11 px floor.
  - Breakpoints are 1320, 1120, 900, 760, and 480 px. At ≤760 px the sidebar becomes a drawer and the inspector goes full-screen (closed by default).
  - Keep focus-visible, ARIA, `role=alert` errors, `role=status` notices, `prefers-reduced-motion`, and `title` hints on compact or disabled controls.
- **TypeScript and ESLint:** TS `strict`, `noUncheckedIndexedAccess`, and `noUnused*`. ESLint `recommendedTypeChecked`, `no-floating-promises` (use `void`), `consistent-type-imports`.
- **Vitest:** `src/test/setup.ts` mocks `matchMedia` and `scrollIntoView` and clears the auth token. Blob realm differences between jsdom and Node mean tests should assert on behavior (size, type, bytes), not `toBeInstanceOf(Blob)`.
- **Env:** `VITE_API_BASE` (default `/api/v1`) and `VITE_WS_BASE` (optional).

## 7. Deployment
### Docker (`Dockerfile`, `compose.yaml`)
- **Dockerfile targets:**
  - `cpu`: CPU torch. The build asserts `+cpu` and that no `nvidia-*` packages are installed.
  - `nvidia`: torch cu128.
  - `frontend`: nginx serving the built SPA.
  - The default target is CPU.
- **Compose profiles:**
  - `cpu`: `preflight-cpu` → `app-cpu` → `frontend-cpu`
  - `nvidia`: the same chain with `gpus: all`
  - `maintenance`: `database-maintenance` (sqlite backup)
- **Preflight** (`docker/preflight.py`, no network) checks that `/models` is readable **and mounted read-only**, that the config files are readable, and that the data volumes are writable.
- **App container:**
  - Env: `LAD_CONFIG=/app/config/default.yaml`, `LAD_USER_CONFIG=/app/user-config/local.yaml` (bind of `APP_CONFIG_DIR`), `LAD_PROFILE=container-*`, `LAD_SERVER__HOST=127.0.0.1`, `LAD_SERVER__PORT=6767`.
  - uvicorn is started directly by `docker/entrypoint.sh`, not the CLI, bound to `0.0.0.0` inside the container.
  - The healthcheck requires ok/ready/ready.
- **Frontend container:**
  - `frontend-entrypoint.sh` polls backend health with `Host: 127.0.0.1:6767`. It must send that Host because the backend Host allow-list rejects the `backend` alias.
  - It starts nginx only after full readiness. Compose `depends_on: service_healthy` is a second gate. Keep **both** gates and never replace them with timers.
  - nginx proxies `/api/` and `/ws/` to `backend:6767` and forwards the WS subprotocol header.
- **Hardening (keep all of it):** UID 10001, read-only rootfs, `cap_drop: ALL`, `no-new-privileges`, bounded tmpfs/cpu/mem/pids, loopback-only publishing, and json logs of 10 MiB × 3.
- **Mounts:**
  - Model bind `MODEL_PATH`→`/models:ro`.
  - Named volumes `local-ai-doctor-{database,uploads,cache,exports,backups}`. `HF_HOME`, `TORCH_HOME`, and XDG point into the cache volume.
  - Renaming a volume is a data migration.
- **Ports are fixed:** UI `127.0.0.1:6969`, API `127.0.0.1:6767`. They are not configurable. For conflicts, stop only the exact owning container.

### Electron (`desktop/`)
- **Startup** (`main.cjs`):
  1. Single-instance lock.
  2. Refuse to start if 6767 or 6969 is busy (never kill the owner).
  3. Show the native `startup.html` window immediately.
  4. Spawn the backend: the packaged PyInstaller `backend_launcher.py`, or in development `.venv` python / `LAD_DESKTOP_PYTHON`.
     - Env: `LAD_CONFIG`, `LAD_USER_CONFIG`=`userData/config/local.yaml`, `LAD_PROFILE=native-windows`, host and port 127.0.0.1:6767, allowed origins, `LAD_PATHS__*` under `userData/data`.
     - Packaged builds also set `LAD_RUNTIME__DEVICE=auto`.
  5. Poll health (`lib/lifecycle.cjs` `isBackendReady`).
  6. Confirm the backend hasn't exited.
  7. Start `lib/frontend-server.cjs` (static server and proxy on 6969).
  8. Navigate the window.
- **Shutdown** writes a private per-launch marker file (`LAD_DESKTOP_SHUTDOWN_FILE`). The launcher watches it and triggers uvicorn lifespan cleanup. After 20 s it falls back to `taskkill` of the exact child tree.
- Logs go to `userData/logs`.
- `multiprocessing.freeze_support()` in the launcher is required so spawned workers don't start a second server.
- GitHub builds bundle CPU torch because of the 2 GiB asset limit. Local builds bundle whatever torch the build environment has. The installer is unsigned per-user NSIS. Models are never bundled.

### CI and release (`.github/workflows/ci.yml`)
- Runs on pushes to `dev`/`main` and on PRs. Jobs:
  - `backend` (py3.12, CPU torch asserted, ruff, mypy, pytest)
  - `frontend` (node 22: lint, vitest, build, playwright)
  - `desktop` (version check, node tests, npm audit)
- `desktop-release` runs on push only, on Windows:
  - `dev` → prerelease `vX.Y.Z-beta.<run_number>`.
  - `main` → stable, immutable `vX.Y.Z`.
  - Exactly one asset, `Local-AI-Doctor-<version>.exe`, under 2 GiB, verified by SHA-256 and an integrity marker in the release body.
  - If a tag already points at another commit, the job fails and you must bump `VERSION`.
- `VERSION` is the source of truth. `scripts/release-version.mjs` syncs `pyproject.toml` and the frontend and desktop `package.json`/lock files, so never hand-edit a manifest version. A completed feature or fix gets a **patch** bump. **minor/major** only on explicit owner request. Docs-only changes don't need a bump.

## 8. Invariants that must not regress (from AGENT.md §3, condensed)
- Fixed ports 6969/6767 with backend-ready-first ordering everywhere (Compose, frontend entrypoint, Electron). No timers. Native dev may use 8000/5173.
- Model roots are external and read-only. Never copy, modify, or commit weights, host paths, WSL distro or user names. Docker stores `/models`.
- Local-only loading: offline HF, `local_files_only=true`, `trust_remote_code=false` (a validator forbids enabling it globally), no pickle fallback.
- A CUDA request runs on CUDA or fails clearly. There is no silent CPU fallback in Docker. The UI's device selection is intent, not proof.
- One worker, one resident model, one active inference. Don't raise worker counts without revalidating ordering and shutdown.
- Reasoning presentation rules (normal vs Nerd Mode). Never rebuild visible text by concatenating tokenizer pieces.
- Attention is described only as allocation, never causation. Perplexity is not factuality. Alternatives are "top alternatives under this distribution".
- Unsupported features stay visible and disabled with a reason. No demo or fake telemetry in real sessions.
- Errors and logs never contain prompts, outputs, tokens, credentials, or unredacted paths (`logging.log_prompts`/`log_model_output` must stay false).
- Blue accent, readability floor, 30 px token chips, accessibility.
- Docker hardening and named volumes are preserved. WSL VHD cleanup follows AGENT.md §17.9: inspect first; never prune volumes; never compact or unregister without explicit approval.

## 9. Change checklists (verified against code paths)
- **New/changed REST endpoint:** route in `main.py` → strict model in `api/schemas.py` (camelCase shim if the browser sends it) → `WorkbenchError` subclasses for failures → tests in `tests/integration/test_api_workbench.py` (plus `tests/security/` if it touches transport or bodies) → `frontend/src/api/client.ts` + `types.ts` → `docs/api.md`. New HTTP methods also need to go into `CORSMiddleware.allow_methods`.
- **New per-token field:** worker payload in `runtime.py` → new migration adding the column → `TOKEN_INSERT_SQL` and `_persist_token` in `runs.py` → `repository.list_run_tokens` → `ChatExportToken` in `schemas.py` plus `workspace.py` export/import → CSV `fields` in `main.py` `export_run` if useful → `TokenEvent` in `types.ts` and `normalizeToken` in `client.ts` → UI → `docs/metrics.md` (definition, units, clock, unavailable state).
- **Schema change:** add `persistence/migrations/000N_*.sql`, then test upgrade from an existing DB (pre-migration backup happens automatically) and export/import round-trips (`tests/integration/test_workspace_portability.py`, `test_database_lifecycle.py`).
- **New capability or model family:** follow `docs/adapters.md`. Discovery evidence and negative tests (`tests/backend/test_discovery.py` uses the synthetic SafeTensors helpers in `tests/backend/conftest.py`), then a capability-matrix reason, the worker path in `runtime.py`, API, frontend gating (`domain/capabilities.ts`), docs, and `docs/model-capability-report.md`.
- **Sampler change:** `runtime.py` (`_apply_penalties`, `_filter_distribution`, sampling loop), `sampling/metrics.py` reference plus `test_sampling.py`, the `sampling_pipeline` list in `inspect_model`, and `docs/metrics.md`.
- **UI change:** check desktop and ≤760/480 px widths, keyboard, reduced motion, and normal vs Nerd privacy. Update Playwright fixtures in `frontend/tests/workbench.spec.ts` when the API shape changes.
- **Config change:** field on the relevant `*Settings` model → `config/default.yaml` → `docs/configuration.md`, and `config/local.example.yaml` if user-facing. Run-affecting fields must appear in `inference_snapshot()`.
- **Every shipped feature or fix:** patch bump via `release-version.mjs`, plus `README.md` updates whenever setup, config, deployment, startup, workflow, or release behavior changes. `CHANGELOG.md` uses the `Unreleased` section.

## 10. Git and delivery rules (owner-specified)
- Work on `dev`. `main` is the stable release branch. Don't create, switch, merge, rebase, or push other branches unless asked. Commit and push only when asked.
- **No AI/"generated by" text and no `Co-Authored-By` trailers in commits.** Use the existing git identity and never change it. This owner rule overrides default attribution.
- Before committing: check `git status`, the full diff, and `git diff --check`. Run the relevant check suites. Make sure there are no secrets, personal paths, model files, DBs, uploads, logs, screenshots, or release artifacts. Stage only the coherent task.
- Don't add or change a license (none is selected). Report vulnerabilities privately per `SECURITY.md`.
- Never destructively reset or clean a dirty worktree. Preserve unrelated user changes.

## 11. Known discrepancies / open issues found while reviewing (verify before relying)
- **CORS lacks `PUT`.** `main.py` has `CORSMiddleware(allow_methods=["GET","POST","PATCH","DELETE"])`, but `PUT /configuration/model-roots` exists. AGENT.md §22.11 says `PUT` must be allowed and covered by a preflight test, yet `tests/integration/test_api_workbench.py` only preflights `POST`. The likely symptom is that saving model roots from Vite dev (`:5173` → `:8000`) fails CORS preflight, while same-origin production works.
- **Distribution naming.** AGENT.md §9.7 says the branch `distribution` is `raw` or `sampler`, but the code, DB CHECK constraint, and TS type all use `"raw" | "sampling"`. The code is authoritative.
- **`AdapterRegistry` and the adapter ABCs are contracts only.** `WorkerRuntime` hard-codes the Transformers and SentenceTransformers paths.
- `test` profile model roots point at `tests/fixtures/models`, which does not exist. Tests build their own fixtures in `tmp_path` via conftest.
- AGENT.md §21 mentions `apply_patch` (a Codex tool). In Claude Code, use Edit/Write.
