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

### Security

- Loopback binding by default; explicit token-authenticated opt-in for non-loopback service.
- Origin enforcement, restrictive response headers, constant-time token comparison, local-only model loading, global remote-code prohibition, safe error redaction, and no prompt/output logging.
- Signature-validated bounded uploads with basename handling, content-addressed storage, and traversal/symlink confinement.
- Non-root, capability-dropped, read-only container execution with a read-only model mount.

### Known limitations

- Dynamic production adapter composition, scheduled telemetry retention, MoE routing hooks, advanced probes, peak-memory enforcement, multi-GPU/offload/quantization, richer portable-workspace coverage, and benchmark automation remain future work.
- No license has been selected; adding one is required before presenting the project as open source.
