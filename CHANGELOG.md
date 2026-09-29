# Changelog

This project follows a Keep a Changelog-style structure. Versions and dates are added when a release is cut; unreleased work remains under `Unreleased`.

## Unreleased

### Added

- Typed, schema-versioned configuration with local files, profiles, environment/CLI overrides, redacted effective views, and per-run inference snapshots.
- Read-only SafeTensors model discovery, component diagnostics, fingerprints, context evidence, hardware discovery, and complete capability matrices.
- Isolated spawned worker with local-only causal and generic encoder-decoder Transformers generation, cache-aware decode and sampling, cancellation, causal prompt scoring, SentenceTransformers embeddings, and model unload.
- Cost-tiered token instrumentation, exact detailed-tier raw/post-sampler likelihood data, bounded alternatives, emitted reasoning segmentation, phase/timing metrics, and reproducibility/environment records.
- SQLite migrations, WAL mode, foreign keys, pre-migration backups, durable partial runs, batched token writes, and replayable version-1 run events.
- Persistent chat management, content-addressed uploads, generation/embedding APIs, JSON/JSONL/CSV run exports, path-free chat workspace import/export, branch-aware generation replay, confirmed terminal-run retention, and storage statistics.
- React workbench with chat, model registry, generation controls, Nerd Mode, linked token tables/graphs, inspector views, context meter, and embedding comparisons.
- CPU and NVIDIA multi-stage images, hardened Compose profiles, persistent volumes, health checks, WSL2 PowerShell orchestration, and database maintenance utilities.
- Unit, integration, security, real-model classification, frontend component, and browser test foundations, with a checked-in CPU CI workflow.
- Architecture, adapter, configuration, API, metrics, benchmarking, deployment, security, contribution, model-capability, and limitations documentation.

### Fixed

- Discovery no longer treats `config.sliding_window` as a context candidate. It is a per-layer attention span, so sliding-window checkpoints such as Gemma 3 (512) and Phi-3-mini-4k (2047) previously reported a context far below their real limit.
- Discovery selects the context length by source authority instead of taking the minimum of every candidate. The old minimum always included the application fallback, so every checkpoint reported that fallback (4,096 by default) no matter what it declared: a 131,072-token Llama 3.2 and a 262,144-token Qwen3-VL were both capped to 4,096, and the worker budgeted prompts against that floor. The declared positional capacity is now selected, the application value applies only when a checkpoint declares no length at all, and prompt safety continues to come from `inference.max_prompt_tokens`.
- `config.rope_scaling.original_max_position_embeddings` is recorded as evidence but is no longer selectable. It is the pre-scaling base length, so RoPE-scaled checkpoints such as Llama 3.2 (8,192 base, 131,072 scaled) previously reported the unscaled value.
- Discovery reads the family-specific capacity keys `max_seq_len`, `n_positions`, `seq_length`, `max_sequence_length`, and `n_ctx` alongside `max_position_embeddings`, so MPT-style checkpoints such as Krutrim-1 report a declared context instead of falling back.
- Discovery reads `sentence_bert_config.json` `max_seq_length` and ranks it above the positional capacity, because the sentence-transformers pipeline truncates every input at that length. XLM-R embedding checkpoints allocate 514 positions while truncating at 128.
- `conflicting_context_metadata` now fires only when two selectable sources genuinely disagree, and records which source was selected.
- A bundled `auto_map` is no longer an automatic blocking diagnostic. Discovery now asks the installed Transformers build whether a reviewed built-in configuration reads the checkpoint with `trust_remote_code=false`, so checkpoints published before their family was upstreamed (for example Phi-3) load, while checkpoints that genuinely need their bundled code stay rejected with the reason recorded.
- Task detection falls back to the installed Transformers auto mappings when the architecture name is unfamiliar, so a natively supported family is no longer reported as an unsupported architecture.
- The worker resolves the generation Auto class from the loaded configuration instead of always using `AutoModelForCausalLM`. Multimodal generators registered only under the image-text-to-text head, such as Qwen3-VL, now load and generate.
- `chat_template.json` is read alongside `chat_template.jinja` for chat-template presence and reasoning-delimiter discovery.
- CORS preflights for `PUT` are now answered, so saving model roots from the Vite dev origin (`:5173` to `:8000`) no longer fails; `PUT` was missing from the allowed methods.
- Worker failures on model load and unload, embeddings, and prompt scoring now return the standard error envelope with the worker's error code instead of a generic HTTP 500: out-of-memory is 507 (`out_of_memory`), timeouts are 504, a load state mismatch is 409, and any other worker failure is 502.
- Starting a generation in a chat that does not exist now returns HTTP 404 with code `chat_not_found`; it previously fell through to a generic 500.
- Chat and run 404s, run-cancel and confirmation 409s, unloading a model by ID while a different model is resident, the embedding attachment-count 413, and unknown `/api` paths now return the standard `{"error": {...}}` envelope with stable codes (`chat_not_found`, `run_not_found`, `run_not_cancellable`, `confirmation_required`, `model_not_loaded`, `not_found`). They previously returned FastAPI's `{"detail": "..."}` shape. Status codes are unchanged.
- Abandoning a generation stream before its terminal event (the consumer closing it, raising, or being cancelled) now cancels the worker and waits for it to stop, instead of leaving an orphaned run generating in the background and blocking the next run. The run manager now closes the worker stream explicitly when the streaming loop fails.
- Editing a user message (a new user message under the same parent) now gets the next `branch_index` among its siblings instead of always `0`, matching how replay numbers assistant branches. Root messages are numbered per chat.
- The browser-shaped `settings` object on `POST /runs` no longer overwrites an explicit `sampling` object or the top-level `deterministic_reference_mode` (`settings` now only fills fields that are not already set), and unknown or snake_case keys inside `settings` are rejected with 422 `invalid_request` instead of being silently dropped.
- Unloading a model uses its own `workers.unload_timeout_seconds` (default 60, minimum 1) instead of `workers.shutdown_grace_seconds` (15). A slow unload previously timed out early, which recycled the whole worker process.
- Uploads rejected because of the file itself (unrecognized signature, invalid filename, MIME type that contradicts the signature, invalid UTF-8, or media that fails decoding) now return HTTP 415 `unsupported_media_type` or 422 `invalid_upload` in the standard error envelope; they previously surfaced as a generic 500.
- `POST /runs/{run_id}/cancel` on a run that does not exist now returns 404 `run_not_found`; it returned 409 `run_not_cancellable`, which is still used for a run that exists but has already finished.
- `POST /chats/{chat_id}/messages` for a chat that does not exist now returns 404 `chat_not_found`; it returned 422 `invalid_request`.

### Security

- Loopback binding by default; explicit token-authenticated opt-in for non-loopback service.
- Origin enforcement, restrictive response headers, constant-time token comparison, local-only model loading, global remote-code prohibition, safe error redaction, and no prompt/output logging.
- Signature-validated bounded uploads with basename handling, content-addressed storage, and traversal/symlink confinement.
- Non-root, capability-dropped, read-only container execution with a read-only model mount.

### Known limitations

- Dynamic production adapter composition, scheduled telemetry retention, MoE routing hooks, advanced probes, peak-memory enforcement, multi-GPU/offload/quantization, richer portable-workspace coverage, and benchmark automation remain future work.
- No license has been selected; adding one is required before presenting the project as open source.
