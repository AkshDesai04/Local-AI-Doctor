# REST and WebSocket API

## Conventions

The versioned REST base is `/api/v1`. JSON request models are strict: unknown fields normally return 422. Canonical API fields use `snake_case`; generation and embedding endpoints also accept the bundled browser's documented camel-case shape.

Interactive OpenAPI documentation is available at `/api/docs` only under the `development` profile. The schema itself is `/api/v1/openapi.json` for every profile.

When `server.authentication_token` is configured, every API request requires:

```http
Authorization: Bearer <token>
```

The static application shell remains readable so a browser can show the tab-scoped token entry screen; it contains no private API data. The default loopback configuration has no token. Request hosts are checked against the configured listener and origin hosts to resist DNS rebinding. When CSRF protection is enabled, state-changing requests carrying an `Origin` header accept a trusted request target's own origin or an entry in `server.allowed_origins` and reject other origins. WebSocket origin matching treats `ws` as HTTP and `wss` as HTTPS. Absence of an Origin is not treated as browser proof; bearer authentication remains the external-access boundary.

Structured application errors use:

```json
{
  "error": {
    "code": "capability_unavailable",
    "message": "the selected model does not expose embeddings",
    "retryable": false,
    "hint": "optional safe guidance",
    "details": {}
  }
}
```

Validation errors use code `invalid_request` with a bounded `details.issues` array. Resource and conflict errors use the same envelope: `chat_not_found` (404, including creating a message or run in a chat that does not exist) and `run_not_found` (404), `not_found` for an unknown `/api` path (404), `run_not_cancellable`, `confirmation_required`, and `model_not_resident` (409, unloading a model or resident key that is not resident), `worker_busy` (409, including when `runtime.max_loaded_models` residents are all pinned by the running job), and `limit_exceeded` (413). Errors raised by the router itself use the envelope too: an unsupported method on a known or unknown path returns 405 `method_not_allowed` (with the `Allow` header), and a missing static asset returns 404 `not_found`; any other framework status uses code `http_error`. The message is the canned HTTP status phrase, never request text. Client-side application routes still fall back to the frontend shell for `GET`. The bundled client also accepts the older `{"detail":"..."}` shape.

Failures reported by the model worker (model load and unload, embeddings, prompt scoring) use the same envelope with the worker's error code and a status chosen from it. The message and hint are the worker's fixed, redacted text; exception text and paths never appear.

| Worker code | HTTP status | Envelope `code` |
| --- | --- | --- |
| `model_out_of_memory`, `out_of_memory`, `insufficient_memory` | 507 | `out_of_memory` (the original worker code is kept in `details.worker_code`) |
| `model_worker_timeout`, `inference_timeout` | 504 | unchanged |
| `model_worker_state_mismatch`, `model_not_resident`, `worker_busy` | 409 | unchanged |
| any other code (for example `cuda_runtime_error`, `model_worker_error`) | 502 | unchanged |

Out-of-memory and timeout failures are marked `retryable`. A worker's `insufficient_memory` report carries only codes and numbers (`memory_kind` `vram` or `ram`, `required_bytes`, `available_bytes`, and `estimate` `{weights, kv_reserve, margin}`), which the envelope returns in `details`.

A model load that does not fit after evicting idle residents returns 507 `out_of_memory` built by the registry:

```json
{
  "error": {
    "code": "out_of_memory",
    "message": "the model does not fit the available memory",
    "retryable": true,
    "hint": "Quantize the model, unload another model, or turn off Strict VRAM to allow system-RAM offload.",
    "details": {
      "memory_kind": "vram",
      "required_bytes": 8053063680,
      "available_bytes": 6442450944,
      "estimate": {"weights": 7642398720, "kv_reserve": 410664960, "margin": 536870912},
      "strict_vram": true,
      "placement": "gpu_only",
      "resident_model_keys": ["3f1c..."],
      "pinned_model_keys": [],
      "evicted_model_keys": ["9ab2..."]
    }
  }
}
```

`evicted_model_keys` lists residents that were unloaded while trying to make room; a CPU (`memory_kind: "ram"`) failure has a different hint.

POST, PUT, and PATCH bodies are bounded before route parsing. Non-multipart bodies may use at most `limits.prompt_bytes + 1 MiB`; upload multipart bodies may use at most `limits.upload_bytes + 1 MiB`. The inner services separately enforce the exact prompt/content and uploaded-file limits. A declared or streamed envelope overrun returns HTTP 413 with code `limit_exceeded`.

## Endpoint index

### Service, configuration, and storage

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/health` | API, database, worker, resident-model, and protocol status. |
| `GET` | `/configuration` | Redacted effective configuration and precedence list. |
| `GET` | `/hardware` | Hardware inventory and explainable current selection. |
| `GET` | `/storage` | SQLite size and row counts for core tables. |
| `DELETE` | `/storage/telemetry?before=...&confirm=true` | Delete terminal runs completed before a timezone-aware cutoff and return per-table counts. |

### Models

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/models` | Current startup/refresh scan with redacted roots, descriptors, diagnostics, and capability matrices. |
| `POST` | `/models/refresh` | Re-scan configured roots and return the new report. Does not load weights. |
| `POST` | `/models/{model_id}/load` | Make a checkpoint resident (or reuse its resident) on optional `device`/`dtype`, evicting least-recently-used idle residents only when needed. |
| `GET` | `/models/resident` | Every resident model plus the worker memory ledger. |
| `POST` | `/models/resident/{model_key}/unload` | Unload one resident; 409 `model_not_resident` for an unknown key. |
| `POST` | `/models/resident/{model_key}/flush` | Flush to storage: write a load-time quantized resident, exactly as held in VRAM, to a new folder in a model root. Returns 201. |
| `POST` | `/models/unload` | Unload every resident. |
| `POST` | `/models/{model_id}/unload` | Unload every resident of that model; 409 `model_not_resident` when it has none. |
| `GET` | `/models/{model_id}/inspect` | Redacted descriptor plus bounded tokenizer/generation/special-token metadata, chat template, and sampler order. |

Load body, all fields optional:

```json
{"device":"cuda","dtype":"bfloat16","quantization":"none","strictVram":true}
```

`device` is `auto`, `cpu`, or `cuda`; dtype is `auto`, `float32`, `float16`, or `bfloat16`. `strictVram` (or `strict_vram`) defaults to `runtime.strict_vram`. `quantization` defaults to `runtime.quantization` and is one of:

| Value | Load |
| --- | --- |
| `none` | The checkpoint's own weights at the selected dtype. A pre-quantized bitsandbytes checkpoint loads as stored. |
| `bitsandbytes-4bit` | bitsandbytes NF4 with double quantization, `uint8` storage, and the selected dtype as compute dtype. |
| `bitsandbytes-8bit` | bitsandbytes LLM.int8 with outlier threshold 6.0. |

Quantization happens in GPU memory while the weights load; nothing is written to disk. The LM head stays at the compute dtype, and multimodal generators also keep their vision tower and projector unquantized. A request returns 409 `capability_unavailable` with the reason in `message` when the model is not a decoder-only text generator, the selected device is not CUDA, bitsandbytes is not installed, the checkpoint is already bitsandbytes-quantized (it cannot be re-quantized), or the value is the legacy `int4`/`int8`. A pre-quantized checkpoint also refuses a CPU device. Quantization is part of the resident key, so the same checkpoint at 4-bit and unquantized are separate residents, and the preflight estimate counts 2-D linear weights at 0.5 × 1.03 bytes per parameter (NF4) or 1 byte (LLM.int8).

#### Resident models

One worker process holds several resident models at once. A resident is identified by `model_key`, the first 20 hex characters of `sha256(json([model_id, fingerprint, device, effective_dtype, quantization]))`, so the same checkpoint on another device or dtype is a separate resident. A load with the same key reuses the resident and marks it most recently used, except that a Strict VRAM request for a resident that was offloaded to system RAM unloads and re-places it on the GPU.

Before loading, the worker estimates the resident bytes (weights at the compute dtype plus a KV reserve of `runtime.kv_reserve_tokens`) and compares them with free device memory plus the allocator's unused cache, less `runtime.vram_safety_margin_bytes` and clipped to `runtime.vram_budget_bytes` when set. When the model does not fit, the registry evicts least-recently-used residents that the running job does not use, but only when their measured `gpu_bytes` can make room, then retries. If it still does not fit, Strict VRAM returns the 507 above; with Strict VRAM off, a generation model retries once with accelerate layer offload (`device_map="auto"` bounded by `max_memory`), which is much slower. Residents beyond `runtime.max_loaded_models` are evicted the same way. While any strict CUDA resident exists, the worker caps its allocator with `torch.cuda.set_per_process_memory_fraction`, so an allocation beyond free memory fails instead of spilling into system RAM through the driver.

The load response keeps its previous fields (the descriptor, `lifecycle`, `loaded_device`, `selection`, and the worker's `load` payload) and adds:

```json
{
  "model_key": "3f1c0d9e8b7a6c5d4e3f",
  "placement": "gpu",
  "quantization": "none",
  "strict_vram": true,
  "evicted_model_keys": [],
  "load": {
    "model_key": "3f1c0d9e8b7a6c5d4e3f",
    "placement": "gpu",
    "gpu_bytes": 2006974464,
    "cpu_bytes": 104857600,
    "kv_reserve_bytes": 109051904,
    "device_map_summary": {"cuda:0": 1},
    "estimate": {"weights": 1999994880, "kv_reserve": 109051904, "margin": 536870912},
    "ledger": {}
  }
}
```

`placement` is `gpu`, `offload` (layers split between GPU and system RAM), or `cpu`, read back from the device map the load produced. `gpu_bytes` is the allocator growth measured across the load; `cpu_bytes` is the process resident-set growth and is approximate.

`GET /models/resident` response (`models` is ordered least recently used first):

```json
{
  "models": [
    {
      "model_key": "3f1c0d9e8b7a6c5d4e3f",
      "model_id": "MODEL_ID",
      "display_name": "gemma-3-1b-it",
      "device": "cuda:0",
      "dtype": "bfloat16",
      "quantization": "none",
      "strict_vram": true,
      "placement": "gpu",
      "gpu_bytes": 2006974464,
      "cpu_bytes": 104857600,
      "kv_reserve_bytes": 109051904,
      "load_seconds": 3.4,
      "last_used_at": "2026-10-01T00:00:00+00:00",
      "in_use": false
    }
  ],
  "memory": {
    "device": "cuda:0",
    "total_bytes": 8585216000,
    "free_bytes": 5100273664,
    "torch_allocated_bytes": 2006974464,
    "torch_reserved_bytes": 2147483648,
    "cap_bytes": 6710886400,
    "process_rss_bytes": 1932735283,
    "system_available_bytes": 17179869184,
    "safety_margin_bytes": 536870912,
    "ledger_age_seconds": 0.01,
    "stale": false
  },
  "max_loaded_models": 4
}
```

`in_use` marks residents the currently admitted job has loaded. The worker ledger is refreshed only while nothing is admitted or running, because a timed-out worker request recycles the worker; otherwise the last ledger is served with `stale: true`. `cap_bytes` is `null` while no strict CUDA resident exists. Before the worker has used CUDA, the device totals are `null` rather than creating a CUDA context just to report them; on a CPU-only host `device` is `cpu` and the totals describe system RAM.

#### Flush to storage

`POST /models/resident/{model_key}/flush` body (camelCase; `target_root_index`/`folder_name` also accepted):

```json
{"targetRootIndex": 0, "folderName": "Qwen3-1.7B-bnb-nf4"}
```

`targetRootIndex` indexes `paths.model_roots` (the order of `GET /configuration/model-roots`). `folderName` must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$`, must not end with a dot, and must not be a Windows reserved name (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`, any case, with or without an extension); otherwise 422 `invalid_request`. An index outside the configured roots is also 422.

The route runs as a lifecycle operation, so it returns 409 `worker_busy` while a run is admitted. It then requires:

| Condition | Error |
| --- | --- |
| The key names a resident. | 409 `model_not_resident` |
| The resident was quantized at load time (`bitsandbytes-4bit` or `bitsandbytes-8bit`) and its placement is `gpu`. | 409 `flush_requires_quantized_resident` |
| `<root>/<folderName>` does not exist yet. | 409 `target_exists` |
| A staging folder can be created in the root. | 409 `model_root_read_only`, hint "The model root is read-only (for example Docker mounts /models read-only). Flush from the native app or choose a writable root." |

The backend creates `<root>/.lad-staging-<uuid>`; the worker writes `model.save_pretrained(safe_serialization=True, max_shard_size="2GB")`, the processor or tokenizer, any source `chat_template.json`/`chat_template.jinja` the save did not write, the source `LICENSE*`, `NOTICE*`, and `USE_POLICY*` files, and `local_ai_doctor_derivation.json`. The staging folder is then renamed to its final name (retried up to three times on `PermissionError`, never replacing an existing path) and the roots are rescanned; the scanner skips dot-directories, so a staging folder is never listed. On any failure only the staging folder is removed. Existing files are never modified.

Response (201):

```json
{
  "model": {"id": "qwen3-1-7b-bnb-nf4-1a2b3c4d5e6f", "display_name": "Qwen3-1.7B-bnb-nf4", "path": "<model-root>/Qwen3-1.7B-bnb-nf4", "root_index": 0, "derivation": {}},
  "folder": "<model-root:0>/Qwen3-1.7B-bnb-nf4",
  "bytes_written": 1363148800,
  "derivation": {
    "schema": "local-ai-doctor/derivation",
    "schema_version": 1,
    "source_model_id": "qwen3-1-7b-0123456789ab",
    "source_fingerprint": "SHA256",
    "source_display_name": "Qwen3-1.7B",
    "quantization": "bitsandbytes-4bit",
    "quantization_config": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": true, "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_storage": "uint8", "llm_int8_enable_fp32_cpu_offload": false},
    "software": {"torch": "2.8.0+cu128", "transformers": "4.57.6", "bitsandbytes": "0.50.2"},
    "created_at": "2026-10-01T12:00:00+00:00"
  }
}
```

`model` is the rescanned public descriptor of the new folder (abbreviated above). It is discovered as a pre-quantized checkpoint: `metadata.weight_quantization` is `{"method": "bitsandbytes", "bits": 4, "quant_type": "nf4"}`, `parameter_count` is `null` with `metadata.parameter_count_note: "packed quantized tensors"`, and loading it needs no quantization argument. The derivation file holds identity and software versions only, never host paths.

`POST /models/{model_id}/unload` returns the descriptor with `lifecycle: "unloaded"`, `loaded_device: null`, and `unload: {unloaded_model_id, unloaded_model_keys, freed_bytes, leaked_bytes, ledger}`. The key and unload-all routes return the worker payload: `unloaded_model_id`, `unloaded_model_ids`, `unloaded_model_keys`, `freed_bytes` (allocator bytes actually returned), `leaked_bytes` (measured load bytes that were not returned), `memory_before`, `memory_after`, and `ledger`.

`GET /health` keeps `status`, `database`, `worker`, `loaded_model` (the most recently used resident, or `null`), and `protocol_version: 1`, and adds `loaded_models`, ordered least recently used first:

```json
{
  "status": "ok",
  "database": "ready",
  "worker": "ready",
  "loaded_model": {"model_key": "3f1c0d9e8b7a6c5d4e3f", "model_id": "MODEL_ID"},
  "loaded_models": [
    {
      "model_key": "3f1c0d9e8b7a6c5d4e3f",
      "model_id": "MODEL_ID",
      "device": "cuda:0",
      "dtype": "bfloat16",
      "quantization": "none",
      "placement": "gpu",
      "strict_vram": true
    }
  ],
  "protocol_version": 1
}
```

Every descriptor also carries `root_index` (the configured model root it was found under) and `derivation` (the path-free contents of `local_ai_doctor_derivation.json` for a folder written by Flush to storage, else `null`). A checkpoint whose `config.json` has a `quantization_config` reports it as `metadata.weight_quantization` (`method`, `bits`, `quant_type`); a bitsandbytes checkpoint additionally gets `parameter_count: null` (packed tensors hold several parameters per stored byte), a dtype read from its unpacked tensors, and an `error` diagnostic `quantization_backend_missing` when bitsandbytes is not installed.

In the scan report, each descriptor's `dtype` is the SafeTensors header dtype that stores the most parameters (`null` when no header exists, for example pickle-only folders); the configuration's claim is `metadata.declared_dtype`. `fingerprint.total_weight_bytes` counts only the files a Transformers load reads. A descriptor with any `error` diagnostic has `loadable: false` and every capability `unsupported`. Each `roots[]` entry carries root-level diagnostics, including `empty_model_directory` and `gguf_only_directory` for folders that cannot be candidates, with root-relative names only.

### Chats and messages

| Method | Path | Result |
| --- | --- | --- |
| `POST` | `/chats` | Create a chat; body is `{}` or `{"title":"...","systemPrompt":"..."}`. |
| `POST` | `/chats/import` | Validate and import a version-1 portable chat workspace, remapping message/run IDs. |
| `GET` | `/chats?search=&archived=false` | Up to 100 chats, pinned first and then most recently updated. |
| `GET` | `/chats/{chat_id}` | Chat and ordered messages. |
| `GET` | `/chats/{chat_id}/export` | Download a path-free version-1 chat workspace JSON document. |
| `GET` | `/chats/{chat_id}/messages` | Ordered message list only. |
| `PATCH` | `/chats/{chat_id}` | Change any of `title`, `pinned`, `archived`, or `systemPrompt`. |
| `DELETE` | `/chats/{chat_id}` | Delete a chat and cascading messages/runs/telemetry. Returns 204. |
| `DELETE` | `/chats?confirm=true&include_archived=false` | Explicit bulk deletion; returns count. |
| `POST` | `/chats/{chat_id}/messages` | Create a standalone system/user/assistant/tool message with optional parent, branch index, and attachment IDs. |

Generated messages are normally created through the run endpoint so run identity, status, and partial output remain linked.

Chat objects carry `system_prompt` (`null` when unset). `POST /chats` and `PATCH /chats/{chat_id}` accept it as `system_prompt` or `systemPrompt`. On `PATCH`, an absent field leaves the prompt unchanged, while `null`, an empty string, or whitespace-only text clears it; any other text is stored exactly as sent. The UTF-8 byte length is bounded by `limits.prompt_bytes` and a larger value is rejected with a structured 413 `limit_exceeded`. The prompt is chat-level state, not a message: it does not appear in `messages`.

The portable document uses `"schema":"local-ai-doctor/chat-workspace"` and `"schema_version":1`. It carries chat metadata (including the optional `system_prompt`, validated like `PATCH`), branched messages, runs, token rows, and bounded token alternatives. It deliberately excludes filesystem paths, attachments/upload bytes, raw WebSocket events, environment/phase rows, and embedding vectors. Import rejects unknown fields, duplicate IDs, missing parents, cycles, duplicate token indices/alternatives, and unsupported versions. All message/run IDs are regenerated; source IDs are retained only as import provenance. A run whose model ID is absent from the destination registry is imported with `model_id=null` while its recorded fingerprint remains. Runs/messages captured in a non-terminal state are imported as failed snapshots rather than resumed.

### Generation and scoring

| Method | Path | Result |
| --- | --- | --- |
| `POST` | `/runs` or `/runs/generation` | Persist and schedule a generation; returns 202 immediately. |
| `GET` | `/runs/{run_id}` | Run record, token rows/alternatives, phase metrics, environment snapshot, and terminal summary when present. |
| `POST` | `/runs/{run_id}/cancel` | Request cancellation for queued/loading/running work. An unknown run returns 404 `run_not_found`; a run that has already finished returns 409 `run_not_cancellable`. |
| `POST` | `/runs/{run_id}/replay` | Schedule a new generation from a completed run as a sibling assistant branch. |
| `GET` | `/runs/{run_id}/export?format=json` | Run metadata and token rows. |
| `GET` | `/runs/{run_id}/export?format=jsonl` | Durable raw protocol events as NDJSON. |
| `GET` | `/runs/{run_id}/export?format=csv` | Selected token identity, likelihood, timing, and segment columns. |
| `GET` | `/runs/compare/summary?ids=A&ids=B` | Metadata for two through eight run IDs and any missing IDs. |
| `POST` | `/runs/prompt-score` | Synchronous teacher-forced text perplexity. |

Canonical generation request:

```json
{
  "chat_id": "CHAT_UUID",
  "model_id": "MODEL_ID",
  "prompt": "What is 17 + 25?",
  "seed": 0,
  "device": "auto",
  "dtype": "auto",
  "instrumentation": "token",
  "deterministic_reference_mode": false,
  "strict_vram": true,
  "quantization": "none",
  "sampling": {
    "max_output_tokens": 64,
    "temperature": 0.6,
    "top_k": 50,
    "top_p": 0.95,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop_sequences": [],
    "alternatives": 10
  },
  "attachment_ids": []
}
```

The endpoint also accepts a browser-shaped body: `chatId`, `modelId`, `content`, `parentMessageId`, `attachmentIds`, and a nested `settings` object with `device`, `dtype`, `instrumentation`, `reasoning`, `seed`, `deterministic`, `strictVram`, `quantization`, `temperature`, `alternatives`, `maxOutputTokens`, `topK`, `topP`, `minP`, `repetitionPenalty`, `frequencyPenalty`, `presencePenalty`, and `stopSequences`. Canonical top-level fields (including `deterministic_reference_mode`) and an explicit `sampling` object always win; `settings` only fills what is not already set. An unrecognized key inside `settings` (including a snake_case one) or a `settings` value that is not an object is rejected with 422 `invalid_request`.

The response includes camel-case convenience fields and canonical objects:

```json
{
  "runId": "RUN_UUID",
  "messageId": "ASSISTANT_MESSAGE_UUID",
  "websocketUrl": "/ws/v1/runs/RUN_UUID",
  "run": {"id":"RUN_UUID","status":"queued"},
  "user_message": {},
  "assistant_message": {}
}
```

When the chat has a system prompt, generation prepends it as a `system` message to the reconstructed branch (unless the branch already begins with a system message, which then wins). The rendered-history byte check includes it, and the applied text is snapshotted into the run as `settings.system_prompt` (`null` when none was applied). Replay and token branching use the source run's snapshot, never the chat's current prompt, so editing the prompt later does not change a replay; a token branch's new chat inherits the snapshot. The JSON run export shows it under `run.settings`. When the model's chat template has no system role (it raises, or its output omits the system text), the worker re-renders with the prompt prepended to the first user message and emits a `warning` event with code `system_prompt_merged`; the `prefill` stage reports `prompt_renderer` as `chat_template_system_merged`.

`attachment_ids` (browser alias `attachmentIds`) may name stored uploads for the new user message. Each must be an image or video, and the selected model must report the matching `vision` or `video` capability as `full` or `partial`; otherwise the request fails with 409 `capability_unavailable` (details carry `attachment_id`, `media_kind`, and the capability `reason`) before any chat or run row is written, and an unknown ID fails with 404 `attachment_not_found`. Accepted attachments are linked to the user message, and attachments on earlier messages in the selected branch are re-sent whenever that history is rendered, including replay and token branches. The worker renders media turns with the checkpoint processor's chat template and decodes only files resolved inside the upload store; the `stage` event's `media` field reports `images`, `videos`, and `placeholder_tokens` (null for text-only prompts), and `prompt_tokens` includes those placeholder positions. Vision-language support is validated on Qwen3-VL; other processor families are best-effort.

`seed` accepts the full unsigned 64-bit range and zero is valid. JavaScript clients should preserve large returned seeds as strings.

`sampling.max_output_tokens` is the normal generated-token limit. For explicitly tagged reasoning models only, if the whole configured budget is spent before any visible answer text, the worker can use one additional context-clipped answer window of at most the configured size. Warning events and the terminal summary expose whether that allowance activated and how many additional tokens were used.

`strict_vram` and `quantization` choose how the run's model is made resident; when omitted they fall back to `runtime.strict_vram` and `runtime.quantization`. They are recorded in `settings` and `reproducibility` together with the real `placement` and `model_key` once the model is resident, and `reproducibility.scheduling` (`{concurrent_sessions_max, interleaved}`) once the run ends. Replay and token branching reuse the source run's recorded choices.

Generation, embeddings, and prompt scoring share single-worker admission. At most one job is runnable/active and `runtime.queue_limit` additional operations may wait. Queue overflow is a structured 429 and occurs before a generation creates chat/run state. Explicit model load/unload returns a worker-busy conflict while any inference is admitted.

Replay requires a completed generation, an attached assistant/user branch, and a currently registered model with the recorded fingerprint. It creates a new assistant sibling and run linked through `parent_run_id`, reconstructs that branch's messages, and executes with the recorded seed, sampler, instrumentation, and deterministic-mode settings. The response has the same `runId`, `messageId`, `websocketUrl`, `run`, and `assistant_message` fields as creation, plus `parentRunId`. Replay is a new forward pass; matching output still depends on the recorded environment being reproducible.

Prompt-score body:

```json
{"model_id":"MODEL_ID","text":"Text to score","device":"auto","dtype":"auto"}
```

Prompt scoring is available only for causal generation models. Generic encoder-decoder generation is supported, but scoring it would require separate source and target inputs that this endpoint does not expose.

### Attachments and embeddings

| Method | Path | Result |
| --- | --- | --- |
| `POST` | `/attachments` or `/uploads` | Multipart upload with `model_id` and `file`; returns content-addressed attachment metadata. |
| `GET` | `/attachments/{attachment_id}/content` | Authenticated inline attachment bytes, resolved from confined content-addressed storage. |
| `POST` | `/embeddings` or `/runs/embeddings` | Synchronous embedding run and vectors. |

The upload route determines type from content signature, not filename. It can identify PNG, JPEG, GIF, WebP, MP4, WebM, WAV, FLAC, MP3, PDF, and UTF-8 text. Identification is not acceptance: the current route accepts only native image/video/audio media advertised by the selected model. Plain-text and PDF uploads are rejected because no extracted-text inference adapter is registered; PDF extraction is not installed. Native embedding execution currently consumes local image/video paths, and audio is rejected for the supplied Qwen embedding model.

Accepted media is decoder-validated before the attachment record is created. Images undergo container verification plus dimension and animation-frame checks; video and audio streams are decoded while their bounds are enforced. Defaults cap an image at 40,000,000 pixels, video at 256 decoded frames and 8,500,000 pixels per frame, cumulative animated-image/video expansion at 500,000,000 pixels, and video/audio duration at 600 seconds. These values are configurable under `limits`.

Upload failures caused by the file itself are client errors, not 500s: an unrecognized signature returns 415 `unsupported_media_type`; an invalid filename, a declared MIME type that contradicts the signature, invalid UTF-8 text, or media that fails decoder validation returns 422 `invalid_upload`. Oversized files and decode limits return 413 `limit_exceeded`, and a file the selected model cannot process returns 409 `capability_unavailable`.

Embedding request:

```json
{
  "model_id": "MODEL_ID",
  "inputs": [
    {"id":"a","modality":"text","text":"a red bicycle"},
    {"id":"b","modality":"image","attachment_id":"ATTACHMENT_UUID"},
    {"id":"c","modality":"mixed","text":"the same object","attachment_id":"ATTACHMENT_UUID"}
  ],
  "dimensions": 256,
  "normalize": true,
  "persist_vectors": false
}
```

The response includes both worker-native result objects and browser convenience fields such as `vectors`, `similarityMatrix`, `outputDimensions`, and durations. Vectors are returned even when persistence is false. The current similarity matrix is a dot product; use normalization for cosine semantics. The reviewed Qwen3-VL embedding adapter permits leading-dimension truncation from 64 through its native width; generic embedding adapters accept only their native width unless a model-specific truncation contract is implemented.

## Telemetry retention

Retention is explicit and destructive:

```http
DELETE /api/v1/storage/telemetry?before=2026-01-01T00:00:00Z&confirm=true
```

The cutoff must include a timezone. The server flushes pending telemetry, converts the cutoff to UTC, then deletes entire terminal runs (`complete`, `cancelled`, `failed`, or `disconnected`) whose `completed_at` is earlier. Dependent environment, phase, token, alternative, router, embedding, and raw-event rows cascade. A retained replay whose deleted source was its parent keeps the run but has `parent_run_id` cleared by the foreign key. The response contains the normalized cutoff and counts by table. Chats, messages, models, attachments, upload files, and active/incomplete runs are preserved. `telemetry.retention_days` is not an automatic scheduler.

## WebSocket protocol v1

Connect to:

```text
ws://127.0.0.1:8000/ws/v1/runs/{run_id}?after={last_durable_sequence}
Sec-WebSocket-Protocol: lad.events.v1
```

Non-browser clients may send the Bearer header. Browser clients cannot set arbitrary WebSocket headers, so the bundled client offers a second subprotocol named `lad.auth.<base64url-encoded-token>`. The server selects only `lad.events.v1`; the credential never appears in the request URL. Do not record WebSocket headers at a reverse proxy. Query-string credentials are rejected. Production remote use should provide TLS and a reviewed token-delivery design.

Every ordinary event has this envelope:

```json
{
  "version": 1,
  "run_id": "RUN_UUID",
  "sequence": 7,
  "type": "token",
  "monotonic_ns": 123456789,
  "server_time": "2026-09-13T00:00:00+00:00",
  "payload": {}
}
```

Sequence numbers are per run and begin at one. They order the live stream and, for events accepted by persistence policy, identify durable replay position. Token sequences may be absent from later replay when `persist_token_events=false` or per-run event/byte limits are reached. Recognized event types are:

| Type | Important payload |
| --- | --- |
| `run_created` | queued status and effective seed. |
| `stage` | `model_loading` or `prefill`; the latter includes rendered prompt, prompt count, context, template and tokenization durations, and the resident's `model_key`, `placement`, `quantization`, and `concurrent_sessions`. |
| `model_loaded` | the load response above: lifecycle, device/dtype selection, `model_key`, `placement`, `evicted_model_keys`, load duration, and memory. |
| `metric` | sampler order, seed/RNG, prefill, and prompt throughput. |
| `token` | token identity, selected sampler likelihood, timing, decoded replacement, and segment; exact raw likelihood/rank, entropy/perplexity, and alternatives are populated only at `token`/`full`/`expert` instrumentation. |
| `warning` | bounded model/runtime warning, for example `system_prompt_merged`. |
| `completed` | finish reason, response/segment metrics, throughput, memory, the worker `ledger`, `scheduling` (`concurrent_sessions_max`, `interleaved`), and routing applicability. |
| `cancelled` | same terminal summary shape with cancellation finish reason. |
| `error` | structured failure code, message, and optional hint. |

Terminal events close the server iterator. A disconnect does not cancel the run.

### Reconnection and backpressure

Reconnect with `after=N` to replay all persisted events whose sequence is greater than `N`, then continue live. Store the last ordinary event sequence only after processing that event. Persistence-disabled or truncated token events cannot be recovered by reconnecting; obtain the current message/run state over REST when needed.

Each subscriber queue is bounded. If a browser falls behind, it receives a synthetic `resync_required` event whose `payload.resume_after` is the last sequence actually delivered to that subscriber. Do not advance the durable cursor to the synthetic event's `sequence`; reconnect using `resume_after`. Registration occurs before replay and duplicate sequence filtering closes the replay/live race.

## Minimal command-line example

```bash
BASE=http://127.0.0.1:8000/api/v1
curl -s "$BASE/health"
curl -s -X POST -H 'Content-Type: application/json' -d '{"title":"API run"}' "$BASE/chats"
curl -s "$BASE/models"
```

When authentication is configured, add `-H "Authorization: Bearer $LAD_TOKEN"` without placing the token in a committed script or shell history.
