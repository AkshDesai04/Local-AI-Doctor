# Metrics and instrumentation

## Principles

Metrics describe observable model outputs and measured application phases. They do not reveal hidden chain-of-thought, intent, consciousness, or tokens a model “considered.” A displayed alternative is only a high-probability token under a named distribution.

All probability logarithms use the natural logarithm and entropy is reported in nats. Logits used by the sampler are converted to float32. At `token`, `full`, and `expert`, the raw full vocabulary is also normalized for exact selected-token likelihood/rank and bounded alternatives; those tensors are discarded after the event is built.

## Token identity and text

| Field | Definition |
| --- | --- |
| `token_index` | Zero-based generated-token position. It is the stable x-axis for tables and graphs. |
| `token_id` | Vocabulary ID selected by the sampler. |
| `piece` | Exact result of the tokenizer's ID-to-token conversion. It may contain a special marker or byte-fallback representation. |
| `escaped_bytes` | UTF-8 bytes of the newly displayed text, rendered as `\xNN`. |
| `display_text` | Newly visible suffix after decoding the complete generated prefix. |
| `replace_from` | Character offset where the client replaces its previous decoded text. This permits a tokenizer decode to revise a suffix. |
| `span_start`, `span_end` | Half-open character span in the current fully decoded generated text. |

Tokenizer pieces and visible characters are not assumed to align. Reconstruct display output by applying `replace_from` and `display_text` in token-index order, not by concatenating `piece`.

## Two distributions

For next-token raw logits `z` over the entire vocabulary:

```text
raw_log_probability(i) = z_i - logsumexp(z)
raw_probability(i)     = exp(raw_log_probability(i))
```

The raw distribution is the model output before application sampling transformations. Its selected `raw_logit` is present at every instrumentation level; exact raw log probability, probability, and rank are populated only at `token`, `full`, and `expert`.

The sampling pipeline is applied in this order:

1. repetition penalty;
2. frequency penalty;
3. presence penalty;
4. temperature, or deterministic argmax when temperature is zero;
5. top-k;
6. top-p;
7. min-p;
8. renormalization;
9. sampling.

The selected sampling log probability and probability refer to the renormalized post-filter distribution actually passed to `torch.multinomial` and are present at every instrumentation level. For greedy decoding the selected token has sampling probability 1 and log probability 0; the seed does not affect selection.

`processed_logit` is the chosen raw logit after repetition/frequency/presence penalties and before temperature or vocabulary filters. The token's `filters` field records active temperature/top-k/top-p/min-p steps, or `greedy_argmax`; the complete fixed pipeline, including penalties, renormalization, and sampling, is recorded in the run metric event.

### Rank and alternatives

Raw rank is exact over the full vocabulary:

```text
rank(i) = 1
        + count(z_j > z_i)
        + count(z_j = z_i and j < i)
```

Ascending token ID breaks exact-logit ties deterministically. Raw alternatives are the configured number of highest raw logits with their exact full-distribution probabilities and a `survived_filter` flag. Sampling alternatives are the highest surviving post-filter log probabilities. These lists are bounded views, not separately normalized Top-K distributions. Rank and alternatives are collected only at `token`, `full`, and `expert`; `off` and `basic` store empty alternative lists and null raw rank/probability fields. The chosen token remains stored in the main token fields even when it is outside the alternatives list.

## Uncertainty and likelihood

| Metric | Definition |
| --- | --- |
| `entropy` | `-sum(p_i * ln(p_i))` over all surviving tokens in the sampling distribution; detailed tiers only. |
| `surprise` | `-ln(p_selected)` under the sampling distribution; detailed tiers only. |
| `cumulative_logprob` | Sum of raw selected-token log probabilities from token 0 through the current token; detailed tiers only. |
| `running_perplexity` | `exp(-cumulative_raw_logprob / generated_token_count_so_far)`; detailed tiers only. |
| Conditional response perplexity | `exp(-mean(raw model log probability of every generated token))`; detailed tiers only. |

Lower perplexity means the generated sequence was more likely under this exact model/tokenizer conditioning. It is not a measure of truth, safety, reasoning quality, or usefulness, and values are not generally comparable across tokenizers.

The core math library also defines post-sampler perplexity, but the production token schema currently persists the raw running/response values and the selected token's sampler likelihood rather than a run-level post-sampler perplexity.

## Prompt scoring

`POST /api/v1/runs/prompt-score` performs a separate teacher-forced forward pass for causal generation models only. Encoder-decoder scoring is rejected because this endpoint has one text field rather than distinct source and target inputs. For tokens `x_0 ... x_(n-1)`, the logit row at position `t-1` scores target `x_t`. The first token has no preceding in-sequence distribution and is excluded.

```text
prompt_perplexity = exp(-mean(log p(x_t | x_<t))) for included t >= 1
```

The response reports `included_token_count`, `excluded_first_token`, `excluded_masked_tokens`, `excluded_non_text_tokens`, and a position-aligned list where excluded entries are `null`. The current endpoint accepts text only, so non-text exclusions are zero; multimodal prompt scoring is not implemented.

## Reasoning segmentation

Reasoning is classified only from an explicitly discovered emitted delimiter pair or adapter-provided evidence. The production worker's streaming segmenter preserves exact decoded text and recognizes opening or closing delimiters even when they span token suffixes. It records character slices for the precise boundary and never rewrites tokens. Token-level classes are:

- `reasoning`: emitted content inside explicit reasoning delimiters, including delimiter text;
- `answer`: content after a closed reasoning segment;
- `unknown`: no supported channel or a token whose character slices span multiple classes.

Run completion includes token count for reasoning, answer, and unknown groups. Mean raw log probability and perplexity are populated at `token`, `full`, and `expert` and are null for groups without detailed raw likelihood. No hidden or latent reasoning is exposed. Duration and throughput by segment are not yet returned by the production worker.

For a model with an explicit reasoning delimiter pair, `max_output_tokens` is the normal generation limit. If that entire budget is consumed by emitted reasoning before any visible answer text, the worker may continue for one additional window of at most the same size, clipped to the remaining model context. The stage and terminal payloads report `configured_max_output_tokens`, `reasoning_answer_allowance`, and `reasoning_answer_allowance_used`, and activation is recorded as a warning event. Non-reasoning runs and runs that already produced answer text retain the configured hard limit.

## Timing

All engine phase measurements use a monotonic nanosecond clock. Persisted lifecycle timestamps such as `received_at` are UTC wall-clock values and should not be subtracted for precision benchmarking.

| Metric | Current definition |
| --- | --- |
| Template time | Rendering stored messages through the tokenizer chat template or deterministic plain-text fallback. |
| Tokenization time | Converting the rendered prompt to tensors and moving them to the selected device. |
| Prefill time | Causal prompt forward, or encoder-decoder source encode plus decoder-start forward, producing cache state and token-0 logits. |
| Prompt tokens/s | Prompt token count divided by prefill duration. |
| `decode_ms` for token 0 | Prefill duration. `includes_prefill=true`; this point is not steady-state decode. |
| `decode_ms` for token N > 0 | Forward pass on token N-1 that produced token N's logits. |
| `sample_ms` | Penalties, filtering, selection, and decode-to-display work; at detailed tiers it also includes exact raw full-distribution metrics and alternative extraction. |
| `emit_ms` | Worker-side post-compute enqueue bookkeeping. It is currently recorded as 0 rather than claimed as transport latency. |
| `inter_token_ms` | Time between consecutive worker emission timestamps; undefined for token 0. |
| `cumulative_ms` | Time from prefill start to this token's worker emission. |
| `instantaneous_tps` | `1000 / inter_token_ms`; undefined for token 0. |
| `rolling_tps` | For up to the most recent 10 emitted tokens, interval count divided by first-to-last interval duration. Token 0 uses one divided by elapsed engine time. |
| Engine TTFT | Prefill start to worker emission of token 0. |
| Decode tokens/s | Tokens after token 0 divided by first-to-last emission interval; undefined for a one-token run. |
| End-to-end tokens/s | All generated tokens divided by prefill-start-to-completion duration. |
| Client TTFT/inter-arrival | Browser `performance.now()` measurements from request/send and WebSocket receipt. They are client observations, not persisted engine clocks. |

The run resource also carries wall-clock request, queue, start, first-token, and completion timestamps plus persisted template, tokenization, prefill, and generation phase rows. Model-load duration and exact allocator memory snapshots are returned in load/terminal payloads but are not continuous utilization samples.

On CUDA, `full` and `expert` levels synchronize before and after forward passes for more accurate durations. Synchronization changes scheduling and therefore performance. Other levels avoid those explicit synchronizations, so CUDA timings can include asynchronous work from a different phase or under-report work not yet synchronized.

## Instrumentation levels

The API accepts `off`, `basic`, `token`, `full`, and `expert`. The important cost boundary is between lightweight and detailed tiers:

| Level | Current behavior |
| --- | --- |
| `off` | Emits token identity/display text, selected raw/processed logits, selected sampler likelihood, sampler/filter identity, timing, throughput, and segment. Skips exact raw full-vocabulary normalization/rank, entropy, surprise, perplexity, and alternatives. |
| `basic` | Same current capture as `off`. It is a reserved semantic tier even though its present fields are identical. |
| `token` | Adds exact selected raw log probability/probability/rank, sampler entropy/surprise, cumulative likelihood/perplexity, segment likelihood summaries, and up to the requested number of bounded raw/sampling alternatives. |
| `full` | `token` capture plus explicit CUDA synchronization around forward timing. |
| `expert` | Current `full` behavior. Router fields are emitted as not applicable for dense models; no production MoE hook is composed. |

`off` and `basic` still perform the sampling distribution work required to choose a token, so they are not zero-overhead modes. No measured overhead percentage is returned. Benchmark levels independently before comparing throughput. Attention maps, hidden-state norms, activation probes, logit lens, and routed-expert traces are capability placeholders, not active captures.

## Reproducibility

Every generation records the requested seed (including valid seed 0), an effective random 64-bit seed, `torch.Generator`, generator device, sampling settings, model fingerprint, selected backend/device/dtype, quantization and attention settings, batching state, software versions, and inference configuration digest. Hardware/software/backend snapshots are stored per run.

Reference mode enables deterministic PyTorch algorithms with warnings and disables cuDNN benchmarking. This improves repeatability but is not a cross-device guarantee. The replay endpoint creates a new assistant branch and new forward pass using the recorded seed, sampler, instrumentation, deterministic mode, and recorded device/dtype where still selectable; it first requires the same registered model fingerprint. Exact output identity still requires unchanged model and tokenizer files, rendered branch, software versions, backend, device, dtype, attention kernels, batching, and sampler implementation. The current discovery fingerprint covers model-directory evidence; a separate `tokenizer_fingerprint` field exists in storage but is not yet populated.

## Embedding metrics

Embedding responses report:

- input ID and modality;
- float32 vector and output dimension;
- discovered pooling label (`last-token` for the supplied Qwen3-VL embedding checkpoint, otherwise the model-declared value or `not_reported`);
- requested normalization and measured L2 norm;
- minimum, maximum, mean, standard deviation, and finite-value status;
- forward and total duration, items per second, joint-space claim, and memory snapshot.

For a model-specific adapter that advertises dimension truncation, the worker keeps the requested leading dimensions and normalizes the truncated vector again when `normalize=true`. The reviewed Qwen3-VL adapter permits 64 through its native width; generic adapters require the native width. The response similarity matrix is a dot product. Treat it as cosine similarity only when both vectors are normalized. Preprocessing latency, token throughput, truncation detail, and peak memory are not separately measured yet.

Embedding runs have no generated-token alternatives, response perplexity, reasoning segments, or decode throughput. Those metrics are not applicable.
