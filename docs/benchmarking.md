# Benchmarking and real-model validation

## What a valid result contains

A throughput number without its execution identity is not a reproducible benchmark. Record at least:

- UTC date and whether the run was cold or warm;
- model ID, full fingerprint, parameter count, and model-root revision when known;
- rendered prompt or a content hash plus prompt token count;
- requested/effective seed and complete sampling settings;
- instrumentation level and deterministic-reference setting;
- OS, CPU, RAM, GPU, total/free VRAM, and power mode;
- backend, device, dtype, quantization, and attention implementation;
- Python, PyTorch, Transformers, Tokenizers, SafeTensors, NumPy, and driver/runtime versions;
- batch and concurrency state;
- template, tokenization, prefill, engine TTFT, decode, end-to-end, total, and model-load timings;
- generated token count and finish reason;
- process RSS and CUDA allocated/reserved/peak memory where available;
- warnings, fallbacks, and observed instrumentation side effects.

`GET /api/v1/runs/{id}` includes the persisted model/config/reproducibility record, environment snapshot, phase metrics, tokens, and terminal summary. `GET /api/v1/hardware` captures the current inventory. Export both with the test notes; do not rely on a screenshot alone.

## Benchmark modes

Use distinct labels:

- **Cold load**: the target checkpoint is not resident. Report load time separately from inference.
- **Warm model, cold prompt**: model resident, first execution of the prompt/shape. This may include backend initialization.
- **Warm steady state**: model resident after at least one discarded warm-up with the same shape and instrumentation.
- **End-to-end browser**: includes request, queue, WebSocket, browser scheduling, and rendering. Report alongside, not instead of, engine values.

The application does not yet provide an automated benchmark-runner endpoint. Use the ordinary API with a controlled script, store run IDs, and calculate aggregate statistics from exports. Never relabel a single interactive run as a stable benchmark.

## Generation protocol

1. Stop unrelated GPU/CPU-heavy work and record the system power profile.
2. Start from a clean application process for a cold-load measurement.
3. Refresh discovery and save the selected descriptor/fingerprint.
4. Explicitly load the model with the intended device and dtype. Save `load_seconds` and memory.
5. Create a fresh chat for each independent run so prior context is identical.
6. Use a fixed short prompt, seed, output budget, and sampler. For sampling tests, ensure the model reaches the same token count or report finish-reason differences.
7. Run one warm-up, then at least five measured warm repetitions. More repetitions are appropriate when variance is high.
8. Export each run as JSON and token CSV. Confirm nonnegative timings, contiguous token indexes, and the chosen finish reason.
9. Report median and p95 latency plus mean and sample standard deviation for throughput. Keep raw run IDs/results available.
10. Unload and verify process/CUDA memory before changing checkpoint, dtype, backend, or instrumentation.

A useful short validation request is a deterministic arithmetic prompt with seed 0 and a modest output limit. It validates plumbing; it is not a quality evaluation.

### Cold versus warm reporting

Do not include model load in engine TTFT. Report:

```text
cold_user_wait = HTTP scheduling + queue + model load + template + tokenization + prefill + first emission
engine_ttft     = prefill start through token-0 emission
steady_decode   = token intervals after token 0
```

Token 0 has `includes_prefill=true` and its `decode_ms` is prefill. Exclude it from steady-state decode statistics. The terminal `decode_tokens_per_second` already uses only intervals after the first emission; `end_to_end_tokens_per_second` includes prefill.

## Instrumentation overhead

Compare levels using the same process, loaded checkpoint, prompt tokens, output token count, sampler, seed, and repetition order. Alternate levels rather than running all samples of one level first.

```text
overhead_percent = 100 * (median_profiled_time / median_baseline_time - 1)
```

Choose and name the baseline. `off` and `basic` retain chosen-token/sampler fields and timing but skip exact raw full-vocabulary normalization/rank, uncertainty/perplexity, and alternatives. `token` collects those detailed fields; `full`/`expert` also add explicit CUDA synchronization. The comparison therefore measures relative modes of the same reference loop, not zero-observability versus observability. Full CUDA timing can be more accurate and slower because synchronization changes execution.

## Determinism checks

For a within-environment replay, call `POST /api/v1/runs/{source_run_id}/replay`; it creates a sibling assistant branch and new run using the recorded configuration. Then:

1. keep the same app process or record a process restart;
2. use the same model ID/fingerprint, complete chat, seed (including 0), settings, device, dtype, and instrumentation;
3. set `deterministic_reference_mode=true`;
4. compare token IDs, not only decoded text;
5. compare the stored reproducibility and environment snapshots before interpreting a mismatch.

Repeatability on one CPU/backend is not evidence of CUDA/CPU or cross-version identity. Floating-point reductions, kernels, drivers, library versions, and device-specific sampling can change results. Greedy decoding is seed-independent but can still diverge if logits differ or exact ties are handled differently by a backend.

## Supplied generation checkpoint validation

For DeepSeek-R1-Distill-Qwen-1.5B, validate all of the following in one short run before increasing context:

- the bundled chat template is used with no reconstructed BOS;
- the rendered prompt ends in the expected generation prefix;
- the tokenizer's configured EOS terminates correctly;
- `<think>` and `</think>` are preserved in exact token/display data and segmentation;
- the model reports dense routing as not applicable;
- raw probabilities are finite and between zero and one;
- raw rank is positive and alternatives identify sampler survivors;
- cumulative raw log probability and running perplexity agree with a recomputation;
- token 0 includes prefill and later decode timings are nonnegative;
- cancellation preserves partial message/token/event data;
- unload reduces allocator ownership before loading the next model.

Begin well below the declared context length, at a bounded prompt size you know the host tolerates. A model-file maximum is not a tested hardware-safe limit, and discovery now reports what the checkpoint declares rather than a conservative floor, so `inference.max_prompt_tokens` is the knob that bounds the sweep. Increase prompt length in bounded steps while recording prefill time and peak VRAM; stop before the host is forced into unstable paging or OOM recovery.

## Embedding protocol

For each model/dimension/modality combination:

1. unload any generation model, then load the embedding model explicitly;
2. run at least one warm-up at batch one;
3. use an identical-pair, a related-pair, and an unrelated-pair input set;
4. verify vector shape, float32 finite values, and L2 norm near one when normalization is enabled;
5. verify identical-input dot similarity is near one and record—not assume—the related/unrelated ordering;
6. test the native dimension and every advertised Matryoshka dimension after truncation/re-normalization;
7. record forward, total, items/s, and memory for batch sizes that fit configured limits;
8. for images/videos, record source dimensions/duration and the exact asset hash;
9. test unsupported audio and malformed media as expected failures;
10. unload and verify memory before the next model.

For the supplied Qwen checkpoint, use dimensions 64, 256, 1,024, and 2,048. Start media validation with a low-resolution image and a short low-resolution video. The current API does not separately report media preprocessing time, expanded positions, or truncation, so preserve those test-input facts externally and do not claim they were measured.

## CPU and CUDA comparison

CPU tests should use explicit `device=cpu`, `dtype=float32`, a fixed thread count, and a stable power profile. CUDA tests should use explicit `device=cuda` and dtype, confirm no CPU fallback in the selection record, and record driver plus free VRAM before load. Do not compare CPU and GPU runs that use different prompt or output lengths.

CPU execution of multi-billion-parameter models can take minutes and consume substantial RAM. Routine CI uses tiny deterministic fixtures and does not produce benchmark results. Real-model CPU smoke runs should use very short prompts/output and carry a time/memory warning.

## Result record template

```markdown
### <model> — <cold|warm> — <device>/<dtype> — <instrumentation>

- Date (UTC):
- Model ID / fingerprint:
- Run IDs:
- Hardware and power mode:
- Software / driver:
- Prompt token count / output budget:
- Sampling / requested and effective seed:
- Deterministic reference mode:
- Load seconds:
- Template / tokenization / prefill ms:
- Engine TTFT median / p95:
- Decode tok/s median / p95:
- End-to-end tok/s median / p95:
- Process RSS / CUDA allocated, reserved, peak:
- Finish reason and generated-token count:
- Warnings, fallbacks, anomalies:
```

Keep raw benchmark output out of Git unless it is intentionally curated, small, path-redacted, and reviewed for prompts or private metadata. `benchmarks/results/` is ignored by default.
