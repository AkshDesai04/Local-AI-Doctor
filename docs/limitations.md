# Known limitations

This page describes the implemented 0.1-series workbench, not the eventual adapter contract. A capability matrix on each discovered model is the authority for a particular model/backend pair.

## Model and backend coverage

- Production execution has explicit Transformers reference paths for causal and generic encoder-decoder generation plus SentenceTransformers embeddings. The encoder-decoder path resolves a decoder-start token, encodes the source once, and reuses encoder outputs/decoder cache; it is fixture-tested, not a claim that every real sequence-to-sequence checkpoint has been validated. The abstract `ModelAdapter` registry is not yet dynamically composed into the running application.
- An architecture name recognized during discovery is not a guarantee that its exact forward signature, cache representation, processor, or chat template has passed real inference.
- Only one spawned model worker and one resident model are used. Generation, embeddings, and prompt scoring are serialized. Admission permits one runnable inference plus `queue_limit` waiters; lifecycle changes reject while inference is admitted.
- CPU and CUDA are selectable. ROCm, MPS, multi-GPU placement, and device maps are not implemented runtime paths. CPU offload and non-`none` weight quantization are typed configuration choices but are explicitly rejected by the reference loader.
- Extracted-document inference and attention/hidden-state trace capture are not registered adapters. The capability matrix reports them as unsupported; framework-level tensor availability is not presented as product support.
- RAM/VRAM budget preflight compares discovered checkpoint weight bytes with the selected backend's configured budget. It does not estimate or cap parameters after dtype conversion, activations, KV cache, allocator overhead, or peak runtime memory, so a later backend OOM remains possible.
- Custom checkpoint code is not supported. `trust_remote_code=true` is globally rejected.

## Observability gaps

- `off` and `basic` currently have the same lightweight capture: chosen logits, selected sampler likelihood, token/display identity, and timing. They skip exact raw full-vocabulary normalization/rank, entropy/perplexity, and alternatives; `token` adds those detailed metrics, while `full`/`expert` also add CUDA synchronization.
- No measured instrumentation-overhead percentage is produced. It must be established through controlled paired benchmarks.
- Attention maps, hidden-state/activation norms, logit-lens probes, prompt-cache/KV-cache shape and utilization, and per-token hardware utilization are not captured.
- GPU memory values are exact allocator snapshots when PyTorch exposes them, not continuous utilization. CPU memory is process/system snapshot data, not per-token attribution.
- `emit_ms` is currently zero-valued worker enqueue bookkeeping; transport overhead is not measured by a shared server/client clock.
- Phase metrics cover template rendering, tokenization, prefill, and generation. Warm-up is not a separate phase, and model load remains in its load payload rather than a phase row.
- Reasoning duration/throughput is not reported. For models with an explicitly discovered delimiter pair, the production worker preserves delimiters that span decoded token suffixes and records per-token character slices. A token containing more than one class is marked `unknown` at token level; no hidden or inferred chain-of-thought is exposed.
- MoE schema and interfaces exist, but no production router hooks or routed fixture adapter are composed. Both supplied checkpoints are dense and correctly report routing as not applicable.
- The separate tokenizer-fingerprint storage field is not populated. The default model-directory fingerprint hashes metadata fully and SafeTensors headers/sizes rather than all weight bytes; it is useful identity evidence, not a dedicated tokenizer fingerprint or full checkpoint integrity hash.

## Generation behavior

- Generation uses batch size one and a reference loop. It prioritizes observability over optimized continuous batching or serving throughput.
- The selected conservative context is the minimum plausible discovered/configured value, currently capped by the application default. It is not proof that the limit is safe on every device.
- `max_prompt_tokens` is enforced after rendering/tokenization. Causal generation also subtracts `reserved_output_tokens` when admitting the prompt and clips output to remaining context. Encoder-decoder source and decoder lengths are treated separately; this is a conservative generic policy, not an architecture-specific memory guarantee.
- Prompt scoring enforces `limits.prompt_bytes` but does not separately apply `inference.max_prompt_tokens`; an over-context scoring input is left to the tokenizer/model path to reject.
- Stop-sequence detection occurs after a token has been selected and decoded. The matching text remains in the stored/displayed output.
- Tagged reasoning models can use one additional, context-clipped output window when the configured token budget ends before visible answer text. This recovery is reported in warning and terminal telemetry; it cannot produce an answer when no model context remains or when the model continues reasoning through the entire allowance.
- DeepSeek system-prompt guidance generates a warning but is not a hard rejection.
- A WebSocket disconnect does not stop a run. Explicit cancellation is required.
- Partial message content is checkpointed every eight tokens and at terminal events. A sudden process loss can leave the message text behind the durable token/event rows; restart marks the run failed rather than regenerating it.
- Run comparison currently returns metadata for two to eight IDs; it is not a full side-by-side statistical comparison or paired trace UI.
- Server-side replay supports completed generation runs only. It requires the original model ID/fingerprint to remain registered, creates a sibling assistant branch, and performs a new forward pass; it cannot guarantee identical output after an environment, tokenizer, kernel, or software change.
- Version-1 chat workspace import/export carries messages, branch/run relationships, token rows, and alternatives. It does not carry attachments, upload bytes, raw events, environment/phase records, embedding inputs/vectors, or an executable model bundle. Per-run JSON/JSONL/CSV exports remain generation-focused.

## Sampling and metrics

- Exact probabilities/ranks are computed from the current full-vocabulary float32 tensor, but implementation uses PyTorch `log_softmax`; values can differ slightly by backend/kernel.
- Only bounded alternatives are persisted. Full-vocabulary logits/distributions are deliberately not retained.
- Run-level post-sampler perplexity is not returned, although selected sampling log probabilities are stored.
- Perplexity is model/tokenizer-specific likelihood, not answer quality.
- Browser receipt timings reflect the browser clock and scheduling and are not directly comparable to server monotonic timestamps as absolute times.

## Embeddings and media

- The embeddings endpoint is synchronous and returns full vectors in its response. Large batches/dimensions increase response size and memory.
- Full vectors are persisted only with `persist_vectors=true`. There is no vector index, nearest-neighbor service, or persisted PCA/UMAP projection.
- The current UI provides batch inputs, vector inspection/copying, a similarity matrix, a lightweight projection view, and client-side JSON/CSV/NumPy (`.npy`) export. A dedicated nearest-neighbor workspace is not implemented.
- Image and video are passed through the SentenceTransformers model path when capability metadata allows. The built-in worker does not accept a standalone audio embedding item.
- File signatures can identify UTF-8 text and PDF, but the current upload route rejects both because no extracted-text inference adapter is registered. Detection must not be confused with native embedding ingestion.
- Upload validation verifies image containers and fully decodes video/audio streams under configurable pixel, frame, and duration caps. Attachment metadata records decoder facts, but embedding telemetry does not separately report preprocessing latency, expanded visual positions, processor resizing/truncation, or sampled-frame selection.
- Dimension requests are runtime-validated against vector width. The reviewed Qwen3-VL path permits leading-dimension truncation from 64 through its native width; generic embedding paths accept only the native width until a model-specific truncation contract is implemented.
- The similarity matrix is a dot product. It is cosine only for normalized vectors.

## Persistence and retention

- SQLite uses migrations, WAL, foreign keys, transactions, and batched telemetry. It is designed for a local single application process, not a multi-node service.
- Confirmed terminal-run retention is implemented through `DELETE /storage/telemetry` with a timezone-aware cutoff. It deletes qualifying run rows and cascading telemetry while preserving chats/messages and non-run data; retained replay runs lose `parent_run_id` when their deleted source was the parent. `telemetry.retention_days` is not scheduled automatically, and periodic hardware sampling remains inactive.
- Token persistence suppression and per-run token event/byte limits are enforced. Router persistence controls have no effect until a production routing hook exists.
- Deleting a chat cascades database records, but content-addressed attachment garbage collection is not implemented; uploaded files can remain on disk.
- `/storage` reports database size and basic row counts, not total upload/cache/export/backup usage.
- Container backup/restore is implemented; native backup is automatic before migrations but has no dedicated CLI restore command.

## Security and deployment

- The intended deployment is loopback. The application does not provide TLS, user accounts, rate limiting, audit-log administration, token rotation, or multi-tenant isolation.
- Bearer authentication is a single configured token. Browser WebSockets carry its base64url encoding in a private `lad.auth.*` subprotocol header because the WebSocket API cannot set an `Authorization` header; it is never added to the URL. Header logging should still remain disabled at proxies.
- Origin and Host checks resist browser cross-origin requests and DNS rebinding. They are defense in depth, not a substitute for authentication on non-loopback deployments.
- Worker isolation is process-level fault containment, not a hardened OS sandbox. A malicious local checkpoint is untrusted input; only SafeTensors and built-in code paths should be used.
- File-signature checks are intentionally narrow. Supported media is decoder-validated under configured expansion limits, but there is no archive extraction, URL fetch, or antivirus scan.
- Normal operation sets framework offline modes, but installation/build requires package registries and base image registries.
- Containers use a read-only root filesystem, dropped capabilities, non-root UID, and a read-only model mount. Native execution inherits the current user's filesystem permissions.

## User interface and accessibility

- Long token tables are virtualized and charts downsample display points, but extremely long chats and full vectors still consume browser memory.
- Unsupported inspector sections remain present with explanations where capability data is available; some planned advanced controls have no backend implementation.
- Browser E2E tests exercise desktop and narrow layouts with fixtures. Real GPU/model performance and OS-specific font/rendering still require manual inspection.

## Publishing

No license has been selected. Until the owner chooses and adds one, third parties do not receive an open-source license despite the repository's portability goals. License selection is a publishing prerequisite.
