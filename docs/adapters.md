# Adapter development

## Scope

An adapter turns discovered model evidence into a safe, explicit runtime path. It must never make a checkpoint appear more capable than the code can execute. This guide covers the public contracts in `backend/local_ai_doctor/adapters`, the discovery layer, and the current worker integration point.

There are two related layers:

1. `ModelAdapter` and its subordinate abstract interfaces define the intended architecture-neutral contract and deterministic selection rules.
2. `WorkerRuntime` currently contains the production built-in load/execute paths for generation and SentenceTransformers embeddings.

The adapter registry is tested infrastructure, but it is not yet dynamically composed into `create_app`. Registering a class in `AdapterRegistry` alone does not make it available through the API. A production adapter change must wire its serializable configuration and worker-side implementation as described below.

## Contracts

| Contract | Owns |
| --- | --- |
| `ModelAdapter` | Metadata-only probe, load/unload, and access to subordinate adapters. |
| `GenerationAdapter` | Prefill, cached decode, sampling, cancellation, generation events, prompt scoring. |
| `EmbeddingAdapter` | Batched embedding and output dimension/normalization behavior. |
| `ProcessorAdapter` | Bounded text/media preprocessing and modality-position/truncation metadata. |
| `ReasoningSegmentAdapter` | Explicit emitted-channel segmentation without rewriting tokens. |
| `RouterInstrumentationAdapter` | Selected versus executed experts, weights, entropy, and drop data. |
| `HardwareBackend` | Inventory, selection, and backend memory release. |
| `TelemetrySink` | Non-blocking/batched event handoff and flush lifecycle. |

Objects containing heavyweight runtime state stay inside the worker. Parent-side contracts and commands must remain serializable.

## Resolution rules

`AdapterRegistry` evaluates every adapter in sorted name order. A probe returns:

- `supported`: whether this model/backend pair is eligible;
- `confidence`: 0 through 100;
- a human-readable reason;
- explicit limitations.

The unique supported adapter with the highest confidence wins. Equal highest-confidence probes are an error, not a registration-order tiebreaker. If none support the model, the structured error includes every probe reason. Prefer a narrow high-confidence probe for an exact architecture/package and a lower-confidence generic fallback.

Probes must inspect the immutable `ModelDescriptor` and `HardwareSelection` only. They must not import a checkpoint's Python files or load weights.

## Adding a model family

### 1. Define evidence and classification

Update discovery only when stable local metadata or tensor names justify the decision. Keep task classification separate from a marketing model name. At minimum verify:

- architecture and `model_type`;
- required config, tokenizer/template, processor, and pooling files;
- SafeTensors tensor presence and shape evidence for the intended head;
- modalities actually accepted by the processor;
- context candidates and any conflicts;
- custom Python files and the trust decision;
- dense versus MoE evidence.

Missing, corrupt, or contradictory evidence should produce a diagnostic. Unknown models should remain discoverable with `task=unknown` and `loadable=false`.

### 2. Produce a complete capability matrix

Every `Capability` enum member must have a state. Use:

- `full` only when the implemented adapter exposes the capability without a known material gap;
- `partial` when the path works with a documented limitation;
- `unsupported` when the model/package does not provide it;
- `unavailable_on_backend` when the model could provide it but this backend cannot.

Every state other than `full` requires a reason. Add limitations for meaningful constraints such as missing prompt traces, a disabled fast path, sampled rather than exact telemetry, or CPU-only support.

### 3. Implement the worker path

Model execution belongs in the spawned worker. A new path should:

- use only local files and default to `trust_remote_code=false`;
- honor the selected device, dtype, attention backend, thread count, and configured limits;
- load one checkpoint at a time and release references/backend caches on unload;
- translate expected failures, including OOM, into bounded structured errors;
- check cancellation between bounded units of work;
- avoid returning model objects, tensors, raw paths, or unbounded distributions to the parent;
- measure phases with monotonic clocks;
- return enough load metadata to prove the effective device, dtype, trust mode, and memory state.

For generation, preserve the exact tokenizer chat template and special-token behavior. Own the sampling order and report the actual post-filter distribution. Do not call `generate()` if doing so hides the logits, timings, cancellation boundary, or sampler state the workbench promises.

Keep causal and encoder-decoder state flows explicit. A generic encoder-decoder path must resolve and report the decoder-start token, encode the source once, reuse encoder outputs and decoder cache, and treat source/decoder context limits separately. Do not expose teacher-forced prompt scoring for it through the current one-text endpoint; that operation needs a reviewed source/target contract.

For embeddings, document the processor input shape, pooling, normalization, native dimension, allowed Matryoshka dimensions, and whether modalities share a space. A decoder-style backbone without an LM head is not a generation model.

### 4. Add processing and instrumentation narrowly

Processor adapters must accept confined local paths, never network URLs, and return explicit truncation and modality-position information. Decoder failures must be errors; do not silently replace failed media with empty text.

Reasoning classification may use emitted tags, special-token metadata, a model-defined output field, or a reviewed template convention. It may not infer hidden chain-of-thought. When delimiters span tokenizer pieces, keep the original pieces and attach character slices or an `unknown` token-level classification.

Router instrumentation must prove the architecture is MoE. Ordinary gated MLP projections are not routers. Keep selected and executed experts separate. If hooks disable compilation, fusion, or another fast path, report that fact and bound prompt traces.

### 5. Wire transport and UI capability gating

Add the adapter to parent-side composition and pass only reviewed settings/evidence across the process boundary. Do not branch on a path supplied by an API caller. Extend boundary schemas only with validated, versionable fields.

The UI should keep an unsupported section visible but disabled with the capability reason. Embedding views must not show generation perplexity or token alternatives. Dense models must show expert routing as not applicable.

### 6. Test the complete boundary

Add fixture tests before real-model tests:

- positive and negative discovery evidence;
- complete capabilities and non-full reasons;
- unique probe resolution, no-match, and equal-confidence conflict;
- CPU load/execute/unload with a tiny deterministic fixture;
- invalid/missing components and corrupt SafeTensors metadata;
- device/dtype rejection and OOM translation;
- cancellation and worker exit;
- exact sampling calculations against small known logits;
- API validation, persistence, event replay, and UI gating;
- traversal/symlink/media-decoder failure paths for processors.

Real-model tests are opt-in. Record model fingerprints and never modify or download into a configured model root during a test.

## Custom model code

Global remote code is prohibited by settings validation. If a checkpoint genuinely requires bundled Python:

1. Audit the exact local files and fingerprint them.
2. Document why built-in Transformers classes cannot load the model.
3. Constrain support to that reviewed fingerprint or a deliberately maintained rule.
4. Execute only inside the worker with offline mode and confined inputs.
5. Expose `reviewed_bundled_code` as the trust decision.
6. Add malicious/import-side-effect and upgrade regression tests.

Do not turn on `trust_remote_code` for a whole model root or accept code from a network repository.

## Review checklist

- Does discovery evidence support the task, modalities, and MoE classification?
- Is every capability state honest for every supported backend?
- Are all non-full states explained?
- Are model roots never written?
- Is all heavyweight state isolated in the child process?
- Are full-vocabulary metrics computed transiently and bounded before persistence?
- Are media limits, context limits, memory cost, and instrumentation side effects visible?
- Can cancellation, unload, restart recovery, and error translation be demonstrated?
- Do CPU fixture tests pass without a GPU or network?
- Are documentation and the capability report updated?
