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
- Desktop installers always ship the pinned CUDA 12.8 PyTorch runtime or fail to build. Packaging previously used whatever PyTorch the selected Python had (CPU from `requirements-ml.lock` unless the CUDA override was applied), only printed it, and GitHub releases deliberately shipped CPU PyTorch under a name that did not say so. Packaging now uses a dedicated `.venv-desktop` created by `desktop/scripts/prepare-build-env.ps1`, asserts `+cu128`/CUDA 12.8 (or `+cpu` for an explicit `LAD_DESKTOP_TORCH_VARIANT=cpu` build) before and after PyInstaller, and names the installer `Local-AI-Doctor-<version>-cuda.exe` or `-cpu.exe`. GitHub releases now build the CUDA variant.
- The packaged backend no longer bundles a CUDA DLL from a local CUDA Toolkit on `PATH`. A mismatched NVRTC 12.4 `nvrtc64_120_0.dll` had leaked into `_internal`; Toolkit directories are now removed from `PATH` while packaging, the spec drops CUDA runtime DLLs that do not come from the pinned wheels, and the build fails if one appears outside `torch\lib`, `torchvision`, or `bitsandbytes`.
- The desktop build verifies the packaged backend with `local-ai-doctor-backend.exe --self-check`, which reports the bundled torch build and device without starting the server. On machines with `nvidia-smi` it also runs a CUDA matmul, a bfloat16 attention call, and an NVRTC-compiled kernel against CPU results, and both smoke tests require the application to select the GPU.
- The installer shrank by dropping `cusolverMg64_11.dll` and `nvrtc64_120_0.alt.dll`, which no bundled binary or module imports or loads by name, the leaked Toolkit NVRTC, and `accelerate.test_utils`; bitsandbytes, when present, contributes only its CPU and CUDA 12.8 libraries. The 0.1.5 CUDA installer went from 2,115,333,309 to 2,027,244,324 bytes. The build fails when the installer reaches 1.95 GiB, leaving room under GitHub's 2 GiB asset limit and the NSIS payload limit.
- The desktop shell no longer forces `LAD_RUNTIME__DEVICE=auto`, which overrode a `runtime.device` chosen in the desktop user configuration. The `native-windows` profile default is already `auto`.
- The desktop build scripts no longer combine `$ErrorActionPreference = "Stop"` with stderr redirection of native commands, which Windows PowerShell 5.1 turns into a fatal error on harmless warnings, and never fall back to a `python` found on `PATH`.

### Security

- Loopback binding by default; explicit token-authenticated opt-in for non-loopback service.
- Origin enforcement, restrictive response headers, constant-time token comparison, local-only model loading, global remote-code prohibition, safe error redaction, and no prompt/output logging.
- Signature-validated bounded uploads with basename handling, content-addressed storage, and traversal/symlink confinement.
- Non-root, capability-dropped, read-only container execution with a read-only model mount.

### Known limitations

- Dynamic production adapter composition, scheduled telemetry retention, MoE routing hooks, advanced probes, peak-memory enforcement, multi-GPU/offload/quantization, richer portable-workspace coverage, and benchmark automation remain future work.
- No license has been selected; adding one is required before presenting the project as open source.
