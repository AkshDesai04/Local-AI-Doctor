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

## Context attention attribution

For a decoder-only generation run at `full` or `expert` instrumentation, each generated token can carry a bounded summary of the causal self-attention rows used by the forward pass that produced its logits. If generated token `y_t` is predicted from source positions `j` in the rendered prompt and already-generated prefix, the worker reads the final query row returned by each usable layer `l` and attention head `h`:

```text
A[l,h,t,j] = exp(S[l,h,t,j]) / sum over allowed k of exp(S[l,h,t,k])

mean_attention[t,j]
    = (1 / captured_head_rows) * sum over captured (l,h) of A[l,h,t,j]
```

Here `S` is the model's actual masked pre-softmax attention score, including its architecture-specific scaling and position treatment; for ordinary scaled dot-product attention it contains `QK^T / sqrt(d_head)` plus the causal mask. The worker clamps only numerical underflow below zero and normalizes the resulting mean back to unit mass. The stored values are therefore the post-softmax attention allocations returned by the eager forward pass that actually produced that step's logits, summarized by an arithmetic mean across all captured layer/head rows. The mean is a deliberate visualization statistic, not a uniquely correct explanation of the model, and the individual per-layer/per-head matrices are not persisted.

The attention row is specific to the decoding step and its prefix, not to one vocabulary candidate. With an unchanged prefix, it is the same row whether the sampler chooses the observed token or a different candidate from that step's logits. Attention weight is therefore not the probability that a source token caused the selected token, and it is unrelated to the selected token's raw or sampler probability. It does not include the direction or magnitude of value vectors, attention output projections, residual-stream state, layer normalization, MLP contributions, or the final vocabulary projection. High attention can accompany little downstream effect, and low direct attention does not rule out information already mixed into another position. Consequently this view can guide investigation, but it cannot by itself prove grounding, factuality, or hallucination.

The context positions have these semantics:

- `prompt` covers the exact tokenizer input after chat-template rendering. It includes system text, earlier conversation turns, the current user prompt, and any template/control tokens. Arbitrary templates can inject or rewrite text, so prompt positions are not guessed back into message/role boundaries.
- `generated` covers earlier tokens from the same assistant response. The token being explained is never one of its own sources because causal decoding predicts it from the preceding context.
- `context_index` is the absolute zero-based position in that combined model context. A generated source also carries its zero-based `generated_token_index`.
- A prompt position that holds image or video features (the processor's placeholder tokens) stays a `prompt` source and additionally carries `media: {"kind": "image"|"video", "index": n}`, where `index` is the zero-based attachment of that kind in conversation order. `index` is omitted when the processor's patch grid does not account for every placeholder. Attention to such a position is allocation to one patch-group embedding, not to a human-readable token.

Persisting every source weight for every generated token would grow quadratically. Each attribution therefore retains at most the 128 highest-weight source positions, keeps their original normalized-row weights, and reports `retained_weight` plus `omitted_weight`; retained weights are not renormalized. Retained entries are stored in context order after selection. Token 0 additionally stores the complete prompt-token catalogue once, and the client combines it with persisted generated-token rows so lower-weight positions can remain visible but dimmed. Heat intensity is scaled relative to the strongest retained source for legibility; hover text reports the exact stored mean weight and its percentage of the normalized row.

Capture is supported only for decoder-only causal text generation. Encoder-decoder models have distinct decoder self-attention and encoder cross-attention distributions, so the workbench does not merge them into a misleading single score. A model must also support switching to eager attention and expose usable per-layer tensors. Otherwise the run emits an `attention_capture_unavailable` warning and continues without attribution.

## Token influence (on demand)

`POST /runs/{run_id}/tokens/{token_index}/influence` answers, for one generated token `y_t`, how much each earlier context position weighed in its prediction by one stated method. Both methods recompute from the persisted run instead of reading the live capture, so they work at any instrumentation level once the run is `complete` or `cancelled`.

**The re-run sequence.** The worker re-tokenizes the persisted `rendered_prompt` exactly as generation did (`_encode_prompt`: the tokenizer with `add_special_tokens=False`, the plain-text fallback's BOS when the run's recorded `prompt_renderer` is `plain_text_fallback`, and the processor with the conversation's image/video attachments for media runs). If that does not reproduce the run's `prompt_token_count`, the request fails with 409 `influence_prompt_mismatch` instead of analyzing a different sequence. The analyzed sequence is `ids = prompt_ids + generated_ids[0..t-1]` of length `S`; position `S-1` is the one whose logits predicted `y_t`.

**Attention.** As live capture does, an efficient-kernel forward fills the cache for positions `0..S-2`, then one eager step at `S-1` returns each layer's post-softmax rows. For layer `l` with `H_l` heads:

```text
layer_row[l,j] = (1 / H_l) * sum over h of A[l,h,S-1,j]          (sliding-window layers left-padded with 0)
mean_row[j]    = sum over l,h of A[l,h,S-1,j] / sum over l of H_l  (the live statistic)
selected[j]    = sum over l in L of H_l * layer_row[l,j] / sum over l in L of H_l
```

Every row is clamped at zero for float underflow and renormalized to unit mass. `layers` may be `"mean"`, `"all"` (the mean plus one entry per captured layer), or a list of layer indices (their head-weighted mean plus one entry per layer). The rows are the same whichever vocabulary candidate is the target, and they are allocation, not causation: the caveats of [Context attention attribution](#context-attention-attribution) apply unchanged. Up to kernel floating-point differences the mean equals the live capture for the same token.

**Gradient x input.** Let `h_i` (dimension `d`) be the residual stream entering the first text-decoder layer at position `i`: the token embeddings after any family-specific scaling, with image/video features already merged. A forward pre-hook replaces that input with a detached leaf, so `input_ids` (and therefore RoPE/mRoPE positions and media merging) are unchanged. With `z` the final-position logits in fp32:

```text
f = log_softmax(z)[target]                         (default: the chosen token)
f = z[target] - z[alternative]                     (when alternativeTokenId is set)
score_i  = | sum over d of (df / dh[i,d]) * h[i,d] |
weight_i = score_i / sum over k of score_k
```

The model stays in eval mode (no dropout) with every weight frozen; one forward and one backward pass run over the whole sequence and the graph is freed afterwards. The response reports `objective` (`log_probability` in nats, or `logit_difference` in logits) and its recomputed `objective_value`; for the chosen token the latter should match the run's recorded raw log probability up to kernel differences. This is the first-order Taylor term of `f` along each position's input vector: a local sensitivity. The absolute value drops the sign, saturated or non-linear effects are invisible to it, and it is not causal attribution. Features a model injects after the first decoder layer (for example Qwen3-VL's deepstack visual features) bypass `h` and are not attributed.

**Sources and normalization.** Contiguous image (or video) placeholder positions that belong to one attachment are grouped into a single source (`source_kind` `image`/`video`, `span` `[start, end)`, `token_count`, weight = the sum of its positions) before ranking, so one image competes as one source. Every other position is a `prompt` or `generated` source. `is_special` is true exactly for ids in the tokenizer's `all_special_ids`; template text such as role names is not special, and media groups are never special. The `source_limit` (8 to 512, default 128) heaviest sources are kept and reported in context order with their original weights; `retained_weight` is their exact sum and `omitted_weight = 1 - retained_weight`. Retained weights are not renormalized. Generated sources carry the text the run streamed rather than a one-token decode. All weights are dimensionless shares that sum to one across the full sequence (`normalization: "sum_to_one"`).

**In the UI.** Edge labels show the weight with two decimals, or `<0.01` below one hundredth. "Relative to max" divides by the largest shown weight. Hiding special tokens divides each remaining weight by `1 - (sum of retained special weights)`, counting positions beyond the retained limit as visible because their status is unknown; the view states that rescale. The live-attention option shows the stored live capture: it has no special-token flags and keeps media placeholders as single positions, because only the top retained positions were stored.

**Unavailable states.** Encoder-decoder and embedding models, runs that are not `complete`/`cancelled`, and models that cannot switch to eager attention return 409 `influence_unavailable`; a token beyond the contiguous persisted prefix, a bad method/field combination, an out-of-range layer, or an alternative equal to the chosen token returns 422 `invalid_request`; a changed or unregistered checkpoint returns 409 `model_fingerprint_changed`; gradient x input on a layer-offloaded model returns 409 `influence_offload_unsupported`, and on a sequence longer than `inference.influence_max_gradient_tokens` 409 `influence_sequence_too_long`; running out of device memory returns 507. Results are cached per run, token, method, canonical parameters, and model fingerprint.

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
| `full` | `token` capture plus explicit CUDA synchronization around forward timing and bounded causal self-attention attribution for compatible decoder-only models. |
| `expert` | Current `full` behavior, including attention attribution. Router fields are emitted as not applicable for dense models; no production MoE hook is composed. |

`off` and `basic` still perform the sampling distribution work required to choose a token, so they are not zero-overhead modes. No measured overhead percentage is returned. Benchmark levels independently before comparing throughput. `full` and `expert` temporarily select eager attention for compatible causal decoders and restore the model's configured implementation afterward. To avoid materializing a prompt-by-prompt attention matrix, the worker prefills every prompt token except the last into the KV cache and captures only the last prompt query row that predicts token 0; later decode steps already have one query position. Eager kernels, the split prefill, tensor capture, CPU transfer, persistence, and CUDA synchronization can materially reduce throughput and can produce floating-point differences from SDPA or Flash Attention. Hidden-state norms, activation probes, logit lens, and production routed-expert traces remain capability placeholders rather than active captures.

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
