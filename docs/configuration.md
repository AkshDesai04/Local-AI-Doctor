# Configuration reference

## Sources and precedence

Configuration is parsed once through `SettingsLoader` and validated as `AppSettings`. From lowest to highest precedence:

1. Pydantic model defaults.
2. The portable configuration's `defaults` section.
3. The selected built-in profile.
4. The selected profile in the portable configuration.
5. The user-local configuration's `defaults`, then its selected profile.
6. `LAD_` environment overrides.
7. Repeated command-line `--set` overrides.

The checked-in portable file is `config/default.yaml`. Native commands automatically use the ignored `config/local.yaml` when it exists. Select alternatives with `--config`, `--user-config`, `LAD_CONFIG`, and `LAD_USER_CONFIG`.

Profile selection uses `--profile`, then `LAD_PROFILE`, then `active_profile` in the portable configuration, then `development`. Available profiles are `development`, `test`, `native-windows`, `native-wsl`, `container-cpu`, `container-nvidia`, and `production`.

All documents require schema version 1. The loader can migrate the pre-release version-0 keys `models_path` and `database_path`; a file with a future schema version fails with an actionable error.

## File shape

```yaml
schema_version: 1
active_profile: native-windows

defaults:
  paths:
    model_roots:
      - X:/models
  runtime:
    device: auto

profiles:
  native-windows:
    runtime:
      cpu_threads: 8
```

A document without a `defaults` key is also accepted as a flat settings document. Unknown fields are rejected. Relative application paths are resolved against the portable configuration directory, so checked-in `../data/...` defaults resolve at the repository root. Prefer absolute model roots in local configuration to avoid ambiguity.

Copy `config/local.example.yaml` to `config/local.yaml`; never edit the example with personal paths and never commit the local file.

The Model registry screen can update `paths.model_roots` in the active profile of that same
user-local YAML or JSON file and rescan without restarting. Paths must be absolute, existing,
readable directories in the backend environment. In Docker, enter the mounted container path
(normally `/models`), not the Windows or WSL host-side bind source. A higher-precedence
`LAD_PATHS__MODEL_ROOTS` or `--set paths.model_roots=...` override intentionally disables web edits
so the displayed setting cannot pretend to be durable.

## Environment and CLI values

Environment variables use `LAD_` and double underscores between nested keys. Values are parsed as safe YAML scalars, so booleans, numbers, lists, and `null` retain their types.

```powershell
$env:LAD_PROFILE = 'native-windows'
$env:LAD_RUNTIME__DEVICE = 'cpu'
$env:LAD_PATHS__MODEL_ROOTS = '["X:/models","Y:/more-models"]'
$env:LAD_SERVER__PORT = '8080'
local-ai-doctor config
```

CLI keys use dots and may be repeated:

```powershell
local-ai-doctor serve `
  --profile native-windows `
  --set runtime.device=cuda `
  --set runtime.dtype=bfloat16 `
  --set inference.defaults.max_output_tokens=256
```

`local-ai-doctor config` prints the effective settings while replacing secrets, local paths, and the WSL distribution identifier with redaction markers. `GET /api/v1/configuration` returns the same safe view. Each run stores an inference-relevant snapshot and SHA-256 configuration digest.

## Settings

### `paths`

| Field | Portable default | Current behavior |
| --- | --- | --- |
| `model_roots` | `../models` | One or more read-only roots scanned at startup and refresh. Duplicates and an empty list are rejected. |
| `database` | `../data/workbench.sqlite3` | SQLite database path. Parent directories are created. |
| `uploads` | `../data/uploads` | Content-addressed attachment storage. |
| `cache` | `../data/cache` | Reserved native cache location; containers separately point framework cache variables at `/data/cache`. |
| `exports` | `../data/exports` | Reserved for persisted exports; current HTTP exports are generated responses. |
| `backups` | `../data/backups` | Automatic pre-migration SQLite backups. |

### `server`

| Field | Default | Notes |
| --- | --- | --- |
| `host` | `127.0.0.1` | Loopback is the safe default. |
| `port` | `8000` | 1 through 65535. |
| `allowed_origins` | Vite origins on `127.0.0.1:5173` and `localhost:5173` | Exact additional HTTP Origin allowlist for CORS, state-changing requests, and WebSockets. The request target's own origin is also accepted, including the HTTP-equivalent origin for WS/WSS. |
| `allow_external_access` | `false` | Must be explicitly true for a non-loopback host. |
| `authentication_token` | `null` | Required for a non-loopback host. HTTP APIs use `Authorization: Bearer`; the browser encodes the same tab-scoped token in a `lad.auth.*` WebSocket subprotocol so it never appears in a URL. |
| `csrf_protection` | `true` | When enabled, state-changing requests with an Origin must match the allowlist. CORS remains configured independently. |

Non-loopback configuration is rejected unless external access and a token are both present. Origins must be exact HTTP(S) origins without paths or credentials; duplicates and wildcards are rejected in every profile. HTTP Host values are limited to the configured listener and the hostnames in `allowed_origins`; list every reviewed browser-facing hostname there. The application does not provide TLS, account management, rate limiting, or token rotation; place a reviewed authenticated TLS proxy in front if remote access is genuinely required.

### `runtime`

| Field | Default | Notes |
| --- | --- | --- |
| `backend` | `auto` | Reserved selector; the current worker uses its reference Transformers/SentenceTransformers paths. |
| `device` | `auto` | `auto`, `cpu`, or `cuda`. Auto chooses the first usable CUDA device, otherwise CPU. |
| `allow_cpu_fallback` | `true` | Allows an unavailable CUDA request to fall back to CPU. |
| `cpu_threads` | logical CPU count | Passed to `torch.set_num_threads`; test and container profiles override it. |
| `ram_budget_bytes`, `vram_budget_bytes` | `null` | Optional caps used by the worker's load preflight. `vram_budget_bytes` bounds everything the worker allocates on the GPU, so a new resident must fit in the budget less the bytes already allocated; `ram_budget_bytes` bounds the worker's resident set for CPU and offloaded loads. The estimate counts weights and the KV reserve, not peak activations. |
| `low_memory_loading` | `true` | Passed to Transformers model loading. |
| `cpu_offload` | `false` | Superseded and rejected when `true`; set `strict_vram: false` to allow layer offload instead. |
| `device_placement` | `sequential` | Reserved for adapter placement policies. |
| `dtype` | `auto` | `float32`, `float16`, or `bfloat16`; auto uses BF16 on CUDA capability 8+, FP16 on older CUDA, and FP32 on CPU. CPU FP16 is rejected. |
| `quantization` | `none` | Default weight quantization for loads that do not choose one (the load body, generation `settings.quantization`, or `quantization` override it). `bitsandbytes-4bit` (NF4, double quantization, compute dtype = the selected dtype) and `bitsandbytes-8bit` (LLM.int8, threshold 6.0) quantize in GPU memory at load time; they need a CUDA device, a decoder-only text generator, and the `bitsandbytes` package from `requirements-cuda.lock`, and are otherwise refused with 409 `capability_unavailable`. The legacy `int4`/`int8` values remain in the schema but are rejected at load. Part of the resident key and of `inference_snapshot()`. |
| `attention_backend` | `auto` | `eager`, `sdpa`, or `flash-attention-2` is passed to Transformers when selected. A compatible decoder-only `full`/`expert` run temporarily selects eager attention so post-softmax rows can be returned, then restores this configured implementation. The run warns and continues without attribution when switching is unavailable. |
| `strict_vram` | `true` | Default for each load (the load body, generation `settings.strictVram`, or `strict_vram` override it). Strict VRAM never spills weights into system RAM: a model that does not fit after evicting idle residents fails with 507 `out_of_memory`, and while any strict CUDA resident exists the worker caps its CUDA allocator (`torch.cuda.set_per_process_memory_fraction`) at the memory free when the cap is computed, less the safety margin, so the driver cannot page allocations into system RAM. When `false`, a generation model that does not fit retries once with accelerate layer offload (`device_map="auto"` with `max_memory`), which is much slower. The cap is process-wide, so a non-strict resident next to a strict one is capped too. |
| `vram_safety_margin_bytes` | `536870912` (512 MiB) | Kept free by the load preflight and the allocator cap; `0` through 16 GiB. Also subtracted from available system RAM for CPU and offloaded loads. |
| `kv_reserve_tokens` | `4096` | Tokens of KV cache reserved per generation model at load time: `2 x layers x kv_heads x head_dim x dtype_bytes x tokens`, read from the checkpoint's `text_config` when present. `0` disables the reserve. |
| `load_one_model_at_a_time` | `false` | When `true`, requires `max_loaded_models=1`, so loading another checkpoint evicts the resident one. |
| `max_loaded_models` | `4` | Maximum resident models in the worker. Loading beyond it evicts the least recently used resident that the running job does not use; if every resident is in use, the load returns 409 `worker_busy`. |
| `max_batch_size` | `1` | Used as the maximum SentenceTransformers encode batch; generation remains batch one. |
| `max_concurrent_runs` | `2` | Maximum generation sessions the worker interleaves (one decode step per session per round). The admission lease still runs one job at a time, so this bounds the sessions inside one job; a generation beyond it is refused with `worker_busy`. |
| `queue_limit` | `32` | Number of inference reservations allowed to wait beyond the one runnable/active slot. `0` permits one inference with no waiter. Overflow returns a structured 429 before chat/run mutation. Explicit model load/unload rejects while any inference is admitted. |

### `inference`

| Field | Default | Notes |
| --- | --- | --- |
| `conservative_context_limit` | `4096` | Fallback context length used only when a checkpoint declares none. It is recorded among the discovered candidates but no longer caps a checkpoint that declares its own capacity; bound prompt size with `max_prompt_tokens` instead. |
| `reserved_output_tokens` | `512` | For causal generation, subtracted from the effective context when computing the maximum admitted rendered-prompt length. Encoder-decoder source and decoder lengths are treated separately, so it is not subtracted from the source limit. |
| `max_prompt_tokens` | `32768` | Enforced for generation after chat rendering/tokenization. The prompt limit is the minimum of this value and the architecture-specific context allowance. Requested output is also clipped to remaining context. |
| `local_files_only` | `true` | The worker currently enforces local-only loading regardless of override. |
| `trust_remote_code` | `false` | `true` is rejected globally. |
| `deterministic_reference_mode` | `false` | Default policy; each generation request may opt in separately. |
| `instrumentation` | `token` | Default request level: `off`, `basic`, `token`, `full`, or `expert`. See [Metrics](metrics.md). |
| `defaults.*` | 512 tokens, temperature 0.7, top-k 50, top-p 0.95 | Central sampling defaults. An omitted request field inherits the corresponding effective configured value. |

Sampling also includes `min_p=0`, repetition penalty `1`, frequency/presence penalties `0`, and 10 alternatives.

### `limits`

| Field | Default | Enforcement |
| --- | --- | --- |
| `upload_bytes` | 100 MiB | Exact uploaded-file limit, enforced while streaming to temporary storage. Multipart request bodies are independently capped at this value plus a 1 MiB framing allowance. |
| `prompt_bytes` | 4 MiB | Enforced on rendered generation prompts, embedding text, prompt scoring, and standalone message content. Non-multipart POST/PUT/PATCH bodies are independently capped at this value plus a 1 MiB JSON/form envelope allowance. |
| `trace_bytes_per_run` | 128 MiB | Caps persisted per-token payload bytes for token rows/raw token events. Live inference continues and emits a warning when token-row persistence is truncated. |
| `attachment_count` | 16 | Enforced for standalone-message and generation attachment IDs, plus attachment-backed embedding inputs. Generation accepts image/video attachments only for generators whose `vision`/`video` capability is usable. |
| `telemetry_events_per_run` | 100,000 | Caps persisted token events per run. It does not cap live WebSocket delivery or stop inference. |
| `image_pixels` | 40,000,000 | Maximum decoded width times height for an uploaded image. |
| `video_frames` | 256 | Maximum decoded frames for video and maximum frames for an animated image. |
| `video_frame_pixels` | 8,500,000 | Maximum decoded width times height for each video frame. |
| `decoded_media_pixels` | 500,000,000 | Maximum cumulative decoded pixels for a video or animated image. |
| `media_duration_seconds` | 600 | Maximum declared or decoded duration for video and audio. |

Declared `Content-Length` and streamed bytes are both checked. Oversized request envelopes or decoded media return HTTP 413. Supported image, video, and audio uploads are decoder-validated before their attachment record is created; the byte limit alone is not treated as sufficient protection from compressed media expansion.

### `workers`

| Field | Default | Notes |
| --- | --- | --- |
| `count` | `1` | Reserved; application composition currently starts exactly one spawned worker. |
| `startup_timeout_seconds` | `120` | Maximum wait for the spawned inference worker's IPC readiness handshake. |
| `load_timeout_seconds` | `600` | Model-load request timeout. |
| `inference_timeout_seconds` | `3600` | Generation, embedding, and prompt-score timeout. |
| `unload_timeout_seconds` | `60` | Model-unload request timeout (at least `1`). A timeout recycles the worker, so this is kept separate from the shutdown grace. |
| `shutdown_grace_seconds` | `15` | Grace for the worker to exit cleanly when the application shuts down. |

### `telemetry`

`write_batch_size` (128) and `write_flush_interval_ms` (100) configure asynchronous SQLite batching. Terminal events force a flush before persistence. `persist_token_events=false` suppresses token rows and durable raw token events while leaving live delivery and terminal state intact. `retention_days` expresses profile policy but is not run by a scheduler; old terminal runs are removed only through the confirmed, timezone-aware `DELETE /api/v1/storage/telemetry` operation. `persist_router_traces`, `router_trace_token_limit`, and `hardware_sample_interval_seconds` remain inactive because production router capture and periodic utilization sampling are not implemented.

### `logging`

`level` controls Uvicorn log level. `json`, `redact_paths`, `log_prompts`, and `log_model_output` express the logging contract. Enabling prompt or model-output logging is rejected. The current application does not yet install a separate structured JSON logger; transport errors and worker diagnostics are deliberately bounded and redacted.

### `features`

`attention_probe`, `hidden_state_probe`, `activation_probe`, `logit_lens`, `full_router_traces`, and `multi_gpu` default to false. These fields remain reserved compatibility gates. In particular, `attention_probe` does not control the implemented causal attention view: selecting `full` or `expert` instrumentation requests that capture automatically for a compatible decoder-only generation model. The other probe flags do not activate production probes in the current worker.

### `platform`

`wsl_distribution` records a local WSL selection, while `container_model_root`, `container_data_root`, and `container_gpu_profile` describe container intent. Docker orchestration uses `.env` and `scripts/wsl-docker.ps1`; application business logic does not invoke WSL or Docker.

## Profile behavior

- `development`: debug, human-readable logs and seven-day retention intent. Interactive API docs are available at `/api/docs`.
- `test`: isolated test paths, CPU with one thread, full instrumentation, and a 1,024-token conservative context.
- `native-windows` and `native-wsl`: automatic device selection.
- `container-cpu`: `/models` and `/data` paths, explicit CPU, JSON-log intent.
- `container-nvidia`: container paths, required CUDA without CPU fallback, GPU profile marker.
- `production`: info/JSON-log intent and 30-day retention intent. Interactive docs are disabled.

Compose supplies additional deployment controls such as port publication, CPU/memory/PID limits, volume names, and NVIDIA device visibility. Those are documented in [deployment.md](deployment.md), not parsed as application settings.

The installed Electron application requests automatic device selection. The
GitHub workflow bundles CPU-only PyTorch to remain within its release-asset
limit, while a local installer built from the pinned CUDA environment can
discover and use CUDA. Capability reporting always reflects the runtime that
was actually packaged.
