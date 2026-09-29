# Architecture

## Design goals

Local AI Doctor separates model-specific execution from application state and transport. The web process remains responsive when a model is slow, model files remain read-only, high-volume telemetry is batched, and every public capability is backed by discovered evidence rather than a model-name allowlist.

The current release optimizes for one local user, one isolated model worker, one loaded model, and one active inference operation at a time. A bounded FIFO admission layer permits one runnable operation plus a configured number of waiters. Interfaces anticipate additional adapters and hardware backends, but the implementation does not claim universal model coverage.

## Runtime topology

```text
Browser
  |  UI assets + same-origin REST /api/v1 + WebSocket /ws/v1             :6969
  v
Frontend reverse proxy
  |  private REST/WebSocket traffic
  v
FastAPI application                                                            :6767
  |-- typed settings + security middleware
  |-- model registry and read-only discovery
  |-- FIFO inference admission, run orchestration, and replayable event broker
  |-- portable chat workspace import/export and branch replay
  |-- upload store
  |-- SQLite repository ---- batched telemetry writer
  |
  `-- ModelWorkerSupervisor
        |  bounded multiprocessing queues
        v
      spawned worker process
        |-- Transformers generation + KV cache + sampler
        `-- SentenceTransformers embeddings

Configured model roots (read-only)       Application data (read/write)
config/tokenizer/processor/SafeTensors   SQLite/uploads/cache/exports/backups
```

In the container deployment, Compose waits for the FastAPI health check before starting the separate
frontend reverse proxy. Its entrypoint independently requires `status: ok`, `database: ready`, and
`worker: ready` before Nginx begins listening, including after Docker daemon restarts. The browser
uses that proxy for same-origin API and WebSocket traffic. Native profiles continue to let FastAPI serve
the built static assets directly.

Worker readiness is established by an IPC handshake from the spawned inference process after it has
initialized and entered its command loop; a live process identifier alone is not considered ready.

Only serializable commands and events cross the worker boundary. PyTorch modules, tokenizer instances, CUDA state, and raw logits stay inside the spawned child process. A Python exception or process exit is translated into a structured worker failure; it does not intentionally expose a traceback through the API.

## Package responsibilities

| Package | Responsibility |
| --- | --- |
| `config` | Schema-versioned Pydantic settings, profiles, precedence, path resolution, and redaction. |
| `discovery` | Bounded metadata reads, SafeTensors header validation, fingerprints, task/modality inference, diagnostics, and capabilities. |
| `domain` | Immutable model descriptors and complete capability matrices. |
| `hardware` | Host/runtime discovery plus explainable device and dtype selection. |
| `adapters` | Architecture-neutral extension interfaces and deterministic adapter resolution contracts. |
| `workers` | Spawned process lifecycle, local-only loading, generation, sampling, embeddings, prompt scoring, memory snapshots, and cancellation. |
| `services` | Single-worker admission, model lifecycle, run orchestration/replay, portable workspaces, uploads, durable event sequencing, and application-level validation. |
| `persistence` | SQLite initialization/migrations, repository queries, transactions, backup-before-migration, and telemetry batching. |
| `api` and `main` | Validated HTTP/WebSocket boundary, security middleware, errors, static frontend, and dependency composition. |
| `frontend` | Chat and model workflows, stream resumption, capability gating, token/metric inspection, and embeddings UI. |

## Startup and shutdown

Application lifespan startup performs these operations in order:

1. Open SQLite, enable foreign keys and WAL mode, and apply append-only migrations. An existing database is backed up before a pending migration.
2. Start the asynchronous telemetry writer.
3. Spawn the isolated model worker.
4. Scan configured model roots and persist model descriptors and capabilities.
5. Mark runs left in `queued`, `loading`, or `running` by a previous process as failed with `interrupted_by_restart`.
6. Expose the application services to request handlers.

Shutdown cancels active run orchestration, asks the worker to unload and exit, drains telemetry, and closes SQLite. If the worker misses the configured grace period, the parent terminates it.

## Model discovery and identity

Discovery walks each configured root to a bounded depth without following model metadata symlinks. A candidate has a `config.json` or a SafeTensors file. The scanner:

- reads size-bounded JSON metadata;
- validates SafeTensors headers, shapes, dtypes, and byte ranges without deserializing weights;
- detects tokenizer, template, generation, processor, pooling, and custom-code components;
- infers task and modalities from architecture and packaging evidence;
- detects MoE only from explicit expert/router evidence, never from ordinary `gate_proj` tensors;
- records every discovered context value and selects the most authoritative one: the sentence-transformers truncation length, then the architecture's declared positional capacity, then `tokenizer_config.model_max_length`, and only then the configured application fallback. Pre-scaling RoPE base lengths, per-layer attention spans, and unbounded tokenizer sentinels are recorded as evidence but never selected;
- creates a SHA-256 manifest fingerprint and a stable ID containing its prefix. The default quick policy hashes metadata files fully and SafeTensors headers plus file sizes (or bounded endpoint samples for other weight formats), not every weight byte. The reported weight size counts only the files a Transformers load reads;
- blocks pickle-only checkpoints (`pickle_weights_only`) and reviewed bundled code whose imports no longer resolve (`bundled_code_incompatible`, a static AST check); a blocked checkpoint reports every capability as `unsupported`;
- reports empty and GGUF-only folders as root diagnostics rather than candidates, and skips dot-directories.

A changed fingerprint represents a new model identity, but an unchanged quick fingerprint is not proof that every weight byte is unchanged. Refresh replaces the registry row for the same canonical path while historical runs retain their stored fingerprint. Deleting metadata never deletes a model file.

## Generation flow

1. The API validates the request shape, capability, prompt byte limit, chat, and sampling settings, then reserves admission. A full queue returns 429 before chat or run state is created.
2. Once admitted, it resolves the requested hardware selection, then creates user and pending assistant messages and persists a queued run with its reproducibility snapshot.
3. A background task emits `run_created`, loads or reuses the selected model, and changes the run to `running`.
4. The worker renders the complete branch (led by the chat's system prompt when one is set) with the tokenizer chat template and `add_generation_prompt=true`, or a deterministic plain-text fallback when no template is available, then tokenizes with `add_special_tokens=false`. The fallback alone prepends the BOS token the tokenizer inserts by default.
5. For a causal model, prompt prefill produces token-0 logits and a KV cache. Every decoder call receives absolute `cache_position` values, which multimodal-RoPE families such as Qwen3-VL require for decode positions. During compatible `full`/`expert` attention capture, the worker prefills the prompt prefix and then runs its final token as the single eager query that predicts token 0, avoiding a quadratic full-prompt attention result. For an encoder-decoder model, the source is encoded once, the decoder starts from a resolved decoder-start token, and both encoder outputs and decoder cache are reused. Each selected token produces the following distribution without an unnecessary final forward pass.
6. The application sampler applies penalties, temperature, and filters in a fixed order. Every token event carries chosen-token/sampler fields, timing, and emitted reasoning classification; `token`, `full`, and `expert` additionally carry exact raw likelihood/rank, uncertainty/perplexity, and bounded distribution views. On supported causal decoders, `full` and `expert` also reduce each generated token's post-softmax attention rows to a mean across captured layers/heads, retain the top 128 source positions with explicit omitted mass, and attach the complete rendered-prompt token catalogue to token 0.
7. Subject to persistence settings and per-run event/byte limits, token rows go to the telemetry writer and raw protocol events are persisted before fan-out. Live delivery continues when token persistence is disabled or truncated. Partial message text is checkpointed every eight tokens and at the terminal event.
8. Completion, cancellation, or failure finalizes both message and run state while preserving partial data.

Generation, embeddings, and prompt scoring share one FIFO execution lease. Capacity is one active/runnable inference plus `runtime.queue_limit` waiting reservations. Queued generation creation is asynchronous, so clients should follow the returned run ID over WebSocket or poll the run resource. Explicit load/unload operations reject while any inference is admitted, preventing lifecycle changes from overtaking queued work.

## Replay and portable workspaces

`POST /runs/{run_id}/replay` accepts only a completed generation whose model is still registered with the recorded fingerprint. It reconstructs that message's branch lineage, creates a sibling assistant branch, links `parent_run_id`, and schedules a new forward pass with the recorded seed, sampling, instrumentation, and deterministic-mode settings. Replay is reproducibility assistance, not a promise of byte-identical output across changed software, device, dtype, kernels, tokenizer files, or environment.

Chat workspace export is a strict, versioned, path-free JSON document. It contains chat metadata, branched messages, runs, token rows, bounded token alternatives, and any persisted bounded attention attribution. Import validates identifiers and parent graphs, remaps all message/run IDs, and converts non-terminal snapshots to failed records. Attachments, upload bytes, raw protocol events, environment/phase rows, and embedding vectors are outside the version-1 portable document.

## Event ordering and recovery

Each run has a monotonically increasing live sequence. Publishing is serialized per run in the order `assign sequence -> apply persistence policy -> fan out`; terminal persistence flushes accepted batched token writes first. A subscriber registers before replaying persisted events, then filters duplicates by sequence. If token persistence is disabled or reaches a configured limit, those live token sequences are intentionally absent from later replay.

A slow subscriber has a bounded queue. When it overflows, the broker clears that subscriber queue and emits `resync_required` with the last sequence actually delivered. The client reconnects using that cursor and recovers durable events. A browser disconnect never cancels generation by itself.

## Embedding flow

The API resolves content-addressed attachments inside the upload root and rejects unsupported model modalities. The worker passes text or local image/video payloads to the loaded SentenceTransformers graph, returns float32 vectors, optionally retains the first requested dimensions, and re-normalizes truncated vectors when normalization is enabled.

Run metadata, input previews, norms, and value statistics are persisted. Full vectors are persisted only when `persist_vectors=true`; they are always returned to the requesting client. Similarity in the current API response is a dot product, which equals cosine similarity only for normalized vectors.

## Persistence model

SQLite tables cover model metadata/capabilities, chats and branched messages, content-addressed attachments, inference runs, environment and phase extension tables, token events (including optional bounded attention attribution) and alternatives, router traces/aggregates, embedding runs/inputs, and raw protocol events. Foreign-key cascades remove dependent chat/run telemetry; model deletion uses `SET NULL` for historical runs. Uploaded files are not stored as database blobs.

One WAL-mode connection and a write lock serialize transactions. Token events and alternatives enter a bounded async writer and are grouped by SQL statement. Terminal protocol persistence explicitly flushes that writer first. This prioritizes durable ordering over maximum write throughput.

Retention is an explicit API operation rather than a background scheduler. After confirmation and a timezone-aware cutoff, it flushes pending telemetry and deletes terminal run rows completed before the cutoff; dependent environment, phase, token, alternative, router, embedding, and raw-event rows cascade. Chats, messages, models, attachments, active/incomplete runs, and upload files remain.

## Trust boundaries

- Model roots are read-only inputs. Transformers loads with `local_files_only=true` and `trust_remote_code=false`.
- Model configuration, filenames, tokenizer templates, uploads, and generated output are untrusted data.
- Upload names are reduced to basenames; content signatures determine supported MIME type; stored names are SHA-256 based; resolution rejects traversal and symlinks.
- Normal API errors redact arbitrary exception detail and local paths. Worker tracebacks stay private to the supervisor.
- The default listener is loopback. A non-loopback listener requires explicit external access and bearer authentication configuration.
- The browser relies on React text rendering and does not intentionally inject model HTML.

See [SECURITY.md](../SECURITY.md) for the operational threat model.

## Extension points

The abstract adapter contracts describe generation, embedding, processing, reasoning segmentation, routing instrumentation, hardware backends, and telemetry sinks. `AdapterRegistry` resolves the unique highest-confidence probe and explains conflicts or no-match cases.

The current production worker has built-in Transformers/SentenceTransformers task paths rather than dynamically instantiating `ModelAdapter` implementations. Adding an architecture therefore requires both a discovery/capability change and a reviewed worker integration. The [adapter guide](adapters.md) documents that boundary explicitly.
