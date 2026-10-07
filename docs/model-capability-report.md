# Supplied model capability report

Audit date: 2026-09-13

This report covers the two checkpoints found in the configured local model root. It distinguishes static checkpoint evidence, deterministic fixture coverage, and real-checkpoint execution. A successful short run applies only to the model, modality, dimension, backend, device, dtype, and limits stated here; it is not evidence of universal support.

Neither checkpoint directory was modified. SafeTensors headers and payload layouts were validated, but the multi-gigabyte payloads were not re-hashed in full. The workbench discovery fingerprint and the upstream Git/LFS identities are recorded below.

## Full local inventory audit (2026-09-30)

The configured root later grew to eleven folders. Every row below is **Metadata/Header verified** by the discovery scan; rows marked *runtime* passed the opt-in matrix in `tests/integration/test_real_model_matrix.py` on the RTX 4060 Laptop GPU (8 GiB, CUDA BF16) described under hardware below. The generator matrix loads through the worker runtime, repeats a 20-token greedy run, compares it with Transformers `generate()` on the same prompt ids, runs the `full` tier (split eager prefill plus attention capture), checks three sampling seeds at temperature 1.0, toggles `enable_thinking` where the template supports it, and confirms CUDA memory returns to baseline after unload. It is a correctness check, not a quality or throughput evaluation.

| Checkpoint | Task | Loadable | Context selected (other candidates) | Reasoning | Modalities | Loader weights / 8 GiB GPU | Blocking reason | Matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-1.5B | text generation | yes | 131,072 `config.max_position_embeddings` (tokenizer 16,384) | `<think>` (template-primed) | text | 3.31 GiB BF16, fits | — | runtime: pass |
| gemma-3-1b-it | text generation | yes | 32,768 `config.max_position_embeddings` (tokenizer sentinel ignored) | none | text | 1.86 GiB BF16, fits | — | runtime: pass |
| Krutrim-1-instruct | text generation | no | 4,096 `config.max_seq_len` (tokenizer 4,096) | none | text | 13.68 GiB BF16, does not fit | `bundled_code_incompatible`: the reviewed MPT code imports `LlamaDynamicNTKScalingRotaryEmbedding` and `LlamaLinearScalingRotaryEmbedding`, which Transformers 4.57.6 no longer defines, and needs `einops`, which is not installed | scan only |
| Krutrim-2-instruct | text generation | no | 1,024,000 `config.max_position_embeddings` (tokenizer 4,096) | none | text | 45.63 GiB pickle shards (stored FP32; config claims BF16), does not fit | `pickle_weights_only`: ten `pytorch_model-*.bin` shards and no SafeTensors; convert offline with a trusted tool into a new folder | scan only |
| Llama-3.2-1B | text generation (base) | yes | 131,072 `config.max_position_embeddings` (RoPE base 8,192 evidence only; tokenizer 131,072) | none | text | 2.30 GiB BF16, fits (the `original/consolidated.00.pth` copy is no longer counted) | — | runtime: pass; plain-text fallback prompt starts with BOS 128000 |
| Ministral-3-3B-Reasoning-2512-GGUF | — | — | — | — | — | empty folder | root diagnostic `empty_model_directory` | scan only |
| Phi-3-mini-4k-instruct | text generation | yes | 4,096 `config.max_position_embeddings` (tokenizer 4,096) | none | text | 7.12 GiB BF16; exceeds the ~6.9 GiB free VRAM before activations | — | matrix skipped (does not fit on the GPU); multi-model runtime: Strict VRAM refuses it before loading, and with Strict VRAM off it loads with layer offload to system RAM and generates 8 greedy tokens; quantized to 4-bit NF4 it fits under Strict VRAM (2.11 GiB) and generates |
| Qwen3-1.7B | text generation | yes | 40,960 `config.max_position_embeddings` (tokenizer 131,072) | `<think>` (`enable_thinking` template) | text | 3.78 GiB BF16, fits | — | runtime: pass, including reasoning on/off |
| Qwen3-VL-2B-Thinking | text generation | yes | 262,144 `config.text_config.max_position_embeddings` (tokenizer 262,144) | `<think>` (template-primed) | text, image, video | 3.96 GiB BF16, fits | — | runtime: pass (text, image, and short-video chat input) |
| Qwen3-VL-Embedding-2B | multimodal embedding | yes | 262,144 `config.text_config.max_position_embeddings` (tokenizer 262,144) | none (no longer inferred from added tokens) | text, image, video | 3.96 GiB BF16, fits | — | runtime: pass, 2,048-wide normalized vectors |
| Vyakyarth | text embedding | yes | 128 `sentence_bert_config.max_seq_length` (positions 514, tokenizer 128) | none | text | 1.04 GiB FP32, fits | — | runtime: pass, 768-wide vectors (run in BF16 on CUDA) |

The Qwen3-VL row records a fixed defect. Before the worker passed `cache_position`, Qwen3-VL placed every decoded token at position zero. On this checkpoint the old decode loop diverged from `generate()` at the fifth generated token for the prompt "Name three primary colors." and then repeated itself; the fixed loop matches `generate()` for all 20 tokens. The other runtime-verified generators also match `generate()` with the change in place. The Phi-3 matrix skip is a memory limit of this 8 GiB card, not a compatibility result.

## Several resident models (2026-10-01)

`tests/integration/test_real_multi_model.py` ran on the same RTX 4060 Laptop GPU (8 GiB, BF16) with the opt-in markers. It passed all five checks:

- gemma-3-1b-it and Llama-3.2-1B stay resident together in one worker and each generates; unloading one returns its measured `gpu_bytes` to within 64 MiB and the worker's allocated memory returns to its baseline after unloading both.
- Sessions interleaved in one worker (the two different residents, and two sessions on gemma) produce exactly the greedy token ids of their solo runs.
- With a strict resident loaded, the allocator cap raises `torch.OutOfMemoryError` for an allocation that fits physically, and the same allocation succeeds once the cap is lifted.
- Phi-3-mini-4k-instruct under Strict VRAM is refused by the preflight with `insufficient_memory` before any weights load; with Strict VRAM off it loads with accelerate layer offload (part of the model in system RAM) and generates 8 tokens.
- Through the real supervisor, spawned worker, and registry, a `vram_budget_bytes` limit that leaves room for Qwen3-1.7B only after one eviction unloads the least recently used resident (Llama-3.2-1B) and keeps gemma-3-1b-it.

These are placement and correctness checks, not throughput measurements; offloaded generation is much slower than VRAM-resident generation.

Chat media on Qwen3-VL-2B-Thinking was exercised through the worker with synthetic inputs and reasoning disabled: a 64×48 red PNG became 70 image placeholder tokens (95 prompt tokens) and the greedy answer was "Red"; a 24-frame, 3-second 64×64 clip that brightens from black was sampled by the processor to 6 frames, became 12 video placeholder tokens, and was described as going from black to brighter. The opt-in matrix repeats this with a 96×96 PNG and a 16-frame clip and checks that every media position is labelled in the `full`-tier attention catalogue. These are smoke results for the preprocessing and prefill path, not a visual-quality evaluation.

## Load-time quantization and Flush to storage (2026-10-01)

`tests/integration/test_real_quantization.py` ran on the same RTX 4060 Laptop GPU (8 GiB) with torch 2.8.0+cu128, Transformers 4.57.6, and bitsandbytes 0.50.2, BF16 compute, Strict VRAM on. It passed all five checks:

- Qwen3-1.7B at 4-bit NF4 stays under 1.6 GiB of measured `gpu_bytes`, generates, and answers "Paris" to "What is the capital of France? Answer in one word." with reasoning off.
- Qwen3-1.7B at 8-bit LLM.int8 loads entirely on the GPU and generates.
- Phi-3-mini-4k-instruct at 4-bit NF4 fits under Strict VRAM (it does not unquantized) and generates 8 tokens.
- Qwen3-VL-2B-Thinking at 4-bit keeps every module under `visual` and the LM head unquantized, quantizes the language model, and answers "Paris".
- Through the real supervisor, spawned worker, and registry, a 4-bit Qwen3-1.7B resident flushed into a temporary model root (never the real one) is rediscovered as a pre-quantized bitsandbytes checkpoint (`parameter_count` null, `weight_quantization` 4-bit nf4, capability `partial`) with a path-free derivation file and the copied `LICENSE`; reloaded without any quantization argument, it produces the same 16 greedy token ids as the in-memory quantized model.

Measured on the same card (one load at a time, `kv_reserve_tokens` 512):

| Checkpoint | Quantization | Measured `gpu_bytes` | Preflight weight estimate | Answer |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | none (BF16) | 3.21 GiB | 3.20 GiB | Paris |
| Qwen3-1.7B | 4-bit NF4 | 1.26 GiB | 1.26 GiB | Paris |
| Qwen3-1.7B | 8-bit LLM.int8 | 1.89 GiB | 1.89 GiB | Paris |
| Phi-3-mini-4k-instruct | 4-bit NF4 | 2.11 GiB | 2.11 GiB | Paris |
| Phi-3-mini-4k-instruct | 8-bit LLM.int8 | 3.75 GiB | 3.74 GiB | Paris |
| Qwen3-VL-2B-Thinking | 4-bit NF4 (vision tower BF16) | 2.02 GiB | 2.01 GiB | Paris |

The estimate counts a tied LM head that the checkpoint also stores (Qwen3) once; before that fix the Qwen3-1.7B estimates were about 0.58 GiB too high. These are memory and correctness checks, not quality evaluations: quantized outputs differ from the BF16 checkpoint, and a one-word answer says nothing about longer-form quality.

## Evidence labels

| Label | Meaning |
| --- | --- |
| **Header verified** | SafeTensors dtype, shape, byte count, and data spans were checked without loading tensor payloads. |
| **Metadata verified** | Local model, tokenizer, generation, processor, chat-template, SentenceTransformers, and model-card files were inspected. |
| **Fixture verified** | The workbench path passed deterministic tests with a small local fixture, not either supplied checkpoint. |
| **Runtime verified** | A supplied checkpoint was loaded and exercised on the stated hardware and backend. |
| **Runtime pending** | The interface or metadata indicates support, but no valid supplied-checkpoint run covers that exact path. |

## Capability matrix

| Capability | DeepSeek-R1-Distill-Qwen-1.5B | Qwen3-VL-Embedding-2B |
| --- | --- | --- |
| Text generation | `full` — real BF16 generation verified on CUDA and a two-token BF16 CPU smoke run completed | `unsupported` — checkpoint has no LM head and is packaged for feature extraction |
| Encoder-decoder generation | `unsupported` — decoder-only Qwen2 checkpoint | `unsupported` — embedding-only use of a decoder-style multimodal backbone |
| Text embeddings | `unsupported` as a first-class capability — no pooling/normalization contract | `full` — corrected-weight CUDA/BF16 runs returned finite normalized embeddings at 64, 256, 1,024, and 2,048 dimensions |
| Image embeddings | `unsupported` | `full` — corrected-weight CUDA/BF16 image path returned finite normalized embeddings |
| Video embeddings | `unsupported` | `full` — corrected-weight CUDA/BF16 short-video path returned a finite normalized 64-dimensional embedding |
| Mixed embeddings | `unsupported` | `partial` — text/image mixed input is runtime verified; every video-containing mixture has not been exercised |
| Audio | `unsupported` — no audio processor | `unsupported` — no audio modality is declared; the API returned a structured capability error |
| Joint cross-modal similarity | `not_applicable` | `partial` — a corrected 64-dimensional image/text comparison ran in the shared normalized space; broader retrieval quality was not evaluated |
| Matryoshka dimensions | `not_applicable` | `full` for the advertised validation set — 64, 256, 1,024, and native 2,048 dimensions passed truncation, re-normalization, finiteness, and shape checks |
| Reasoning segmentation | `partial` — explicit `<think>...</think>` output is segmented while exact raw tokens are retained; there is no hidden reasoning channel | `unsupported` for embedding operations |
| Hidden chain-of-thought access | `unsupported` | `unsupported` |
| MoE routing | `not_applicable` — dense SwiGLU; `gate_proj` is not a router | `not_applicable` — dense text and vision blocks |
| Generated-token likelihood/perplexity | `full` on the reference loop — exact detailed instrumentation and conditional response perplexity were runtime verified | `not_applicable` |
| Teacher-forced prompt scoring | `full` for causal text — a real 17-token prompt was scored | `not_applicable` |
| Sampler alternatives and processed likelihood | `partial` — application sampler and detailed telemetry are implemented; deterministic temperature/top-k/top-p sampling is runtime verified, but not every filter combination | `not_applicable` |
| Attention/hidden-state capture | `partial` for attention — bounded mean self-attention capture at `full`/`expert` is fixture verified and runtime verified on the supplied DeepSeek Qwen2 checkpoint; hidden-state capture remains unsupported | `unsupported` as a trace feature — the pooled embedding uses the final hidden state internally, but the adapter does not expose hidden-state or attention traces |
| Token influence (attention, gradient x input) | `partial` — reported for decoder-only generation; runtime evidence comes from Qwen3-1.7B and Qwen3-VL-2B-Thinking (below), not from this checkpoint | `unsupported` — embedding model |
| Streaming decode | `full` for the tested causal path — token events and terminal state were observed and persisted | `not_applicable` |
| Batch execution | `partial` — supplied generation run used batch one | `partial` — text batch and two-item image/mixed execution are verified; larger batches and media stress are pending |
| Deterministic seeding | `partial` — two same-environment sampled runs produced identical 32-token ID sequences; this is not a cross-device guarantee | `not_applicable` to token selection |
| Deterministic replay | `full` for the tested same-environment path — corrected replay preserved branch lineage and reproduced the rendered prompt, token IDs, and text exactly | `not_applicable` |
| CPU on this machine | `full` for a short DeepSeek BF16 smoke path; performance and operation coverage remain deliberately narrow | `runtime pending` |
| CUDA on this machine | `full` for the validated DeepSeek causal path | `full` for the tested text, image, mixed text/image, and short-video embedding paths |

The workbench also has fixture-verified generic encoder-decoder generation, including one-time encoder execution, cached decoder steps, EOS handling, streaming, telemetry, and cancellation. Neither supplied checkpoint is encoder-decoder, so that implementation is not labeled real-checkpoint verified. Prompt scoring is intentionally causal-only because the current API has no separate encoder source and decoder target fields.

The attention path is fixture verified with a small Qwen2 causal model and runtime verified with a short `full`-instrumentation run on the supplied DeepSeek checkpoint. That live run persisted attribution for both generated tokens over a 14-token rendered prompt and captured all 28 layers with 12 heads per layer. For each generated token the runtime captures the eager post-softmax causal self-attention row, averages across captured layers and heads, retains the 128 highest-weight source positions with explicit omitted mass, and persists the full rendered-prompt token catalogue once on token 0. These weights describe attention allocation, not causal contribution or hallucination evidence; the smoke run is not a performance benchmark.

Token influence is runtime verified by `tests/integration/test_real_influence.py` on a short greedy Qwen3-1.7B BF16 CUDA run: all 28 per-layer attention rows sum to one, gradient x input is finite, non-negative, normalized, recomputes the run's own raw log probability, and changes for an alternative target with its graph freed, and the endpoint serves a repeated request from its cache. On Qwen3-VL-2B-Thinking a synthetic 128x128 PNG's placeholders group into one image source for both methods.

## DeepSeek-R1-Distill-Qwen-1.5B

### Identity and tensor evidence

| Property | Verified value |
| --- | --- |
| Workbench model ID | `deepseek-r1-distill-qwen-1-5b-7ab2d923d545` |
| Workbench fingerprint | SHA-256 `7ab2d923d54590e3515bfd20bd97141630a7ffbfc3471052734a01e58fa4a905` |
| Git revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Git LFS object | `sha256:58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945` |
| SafeTensors file size | 3,554,214,621 bytes (3.310 GiB) |
| Header size / metadata | 38,613 bytes / `format=pt` |
| Tensor count / dtype | 339 / all BF16 |
| Exact parameter count | 1,777,088,000 |
| Header/payload validation | Shape-derived byte counts and contiguous offsets match exactly |

Representative tensors are:

- `model.embed_tokens.weight`: `[151936, 1536]`
- `lm_head.weight`: `[151936, 1536]`
- 28 each of `q_proj` and `o_proj`: `[1536, 1536]`
- 28 each of `k_proj` and `v_proj`: `[256, 1536]`
- 28 each of `gate_proj` and `up_proj`: `[8960, 1536]`
- 28 `down_proj`: `[1536, 8960]`

There are no tensor names containing `expert` or `router`. Each `gate_proj` is the gate branch of an ordinary dense SwiGLU MLP.

### Architecture, context, and template

The local configuration declares `Qwen2ForCausalLM`: 28 layers, hidden size 1,536, intermediate size 8,960, 12 query heads, 2 key/value heads, SiLU activation, RMS normalization, untied embeddings, BF16 weights, and KV caching. `sliding_window=4096` and `max_window_layers=21` are present, but `use_sliding_window=false`; they are not active behavior.

| Context source | Value | Interpretation |
| --- | ---: | --- |
| `config.json` | 131,072 | Architectural maximum-position declaration |
| `tokenizer_config.json` | 16,384 | Bundled tokenizer limit |
| Model card | 32,768 | Reported evaluation generation length, not necessarily total tested context |
| Workbench default | 4,096 | Conservative operational budget, not a model maximum |

The lowest bundled limit, 16,384 total tokens, remains the conservative unvalidated semantic ceiling. No boundary run justifies raising the workbench default. Calculated BF16 KV-cache cost is about 448 MiB at 16K and 896 MiB at 32K for one sequence. Full prefill logits at 16K across a 151,936-token vocabulary would be about 4.64 GiB, so generation consumes only the final prefill position and prompt scoring avoids retaining a full-sequence vocabulary matrix.

The bundled files disagree about BOS: `config.json` uses ID `151643`, while generation/tokenizer metadata use `151646`; ID `151643` is consistently EOS. The validated path follows the tokenizer and generation configuration, applies the bundled chat template with `add_generation_prompt=true`, and tokenizes the rendered text with `add_special_tokens=false` to avoid duplicate BOS insertion.

The template ends the generation prompt with `<｜Assistant｜><think>\n`. `<think>` and `</think>` are token IDs `151648` and `151649`; they are added tokens with `special=false`. The implementation therefore preserves their exact token/text data and layers segmentation on top.

### Loading path

The runtime uses built-in `AutoTokenizer` and `AutoModelForCausalLM`, `local_files_only=true`, `trust_remote_code=false`, `low_cpu_mem_usage=true`, no quantization, and one loaded model at a time. The recorded attention selection was `auto`; the curated run record does not assert a more specific effective kernel.

The raw BF16 weights are 3.310 GiB. Short CUDA execution fits the 8 GiB GPU, but long-context activations, detailed instrumentation, and allocator growth still need measured budgets. CPU BF16 works for the short smoke path and emitted a warning that operation-level BF16 kernel support is architecture dependent.

### Runtime-verified CUDA/BF16 generation

Two cold-load observations were **6.438 s** and **8.406 s**. The latest load snapshot recorded 3,556,011,520 CUDA bytes allocated and 3,753,902,080 bytes reserved. Load time is reported separately from inference and should not be inferred from engine TTFT.

Run `cfc1c2b7-a6fb-4f98-9cce-3a2beb240e3b` used the arithmetic prompt `17*23`, deterministic reference mode, seed 0, greedy decoding, full instrumentation, and a 384-token output budget:

| Measurement | Result |
| --- | ---: |
| Rendered prompt tokens | 29 |
| Generated tokens | 266 |
| Finish reason | EOS |
| Final arithmetic result | 391 |
| Segments | 99 reasoning tokens; 167 answer tokens; closing `</think>` present |
| Prefill | 219 ms |
| Engine TTFT | 344 ms |
| Steady decode | 25.1638 tokens/s |
| End-to-end | 24.4598 tokens/s |
| Conditional generated-response perplexity | 1.0724 |

Stored token IDs re-decoded to the displayed stream and timing fields passed nonnegative/monotonic checks. This is a correctness run, not a quality evaluation or maximum-context benchmark.

Teacher-forced causal prompt scoring was separately verified with 17 prompt tokens, 16 included next-token predictions, one excluded first token, and perplexity **7.87005**.

Two same-process sampled runs, `b1d6c4cc-25d8-4618-9972-02fe6cdeb1d3` and `c5d15051-0333-4bce-b470-e9b02f741fc8`, used seed 42, deterministic algorithms, temperature 0.6, top-k 20, top-p 0.95, and a 32-token budget. Their token IDs and decoded text were identical. This validates same-environment repeatability only.

The corrected replay endpoint was subsequently exercised with source run `0a398d6f-4a30-4c0b-9765-ecf394752f4b` and replay run `9dd7f310-03af-4514-9290-23ecc18079fb`. Both completed with the same 19-token rendered prompt, seed 42 settings, 32 generated/stored tokens, token ID sequence, and decoded text. The replay persisted the source ID as `parent_run_id`. Source and replay generation times were 1,578 ms and 1,141 ms respectively; deterministic replay requires output identity under the recorded environment, not timing identity.

Cancellation was exercised on run `a12ef21a-1c67-44ab-8ecd-34e7c6235b6c`. It stopped after 28 of a possible 512 tokens, persisted the partial assistant message and token events with terminal status `cancelled`, and retained that state across application restart.

### Runtime-verified CPU/BF16 smoke

A deliberately short CPU run proved the fallback path without attempting a throughput comparison against the longer CUDA run:

| Measurement | Result |
| --- | ---: |
| Cold load | 4.313 s |
| Prompt / generated tokens | 12 / 2 |
| Prefill | 609 ms |
| Engine TTFT | 625 ms |
| Steady decode | 10.6383 tokens/s |
| End-to-end | 2.7816 tokens/s |
| Process RSS snapshot | 3,794,923,520 bytes |
| Finish reason | Length |

The selected backend was explicitly CPU with BF16 and no fallback. The worker warned that BF16 kernel support can vary by CPU operation. The two-token sample establishes validity, not stable CPU performance.

## Qwen3-VL-Embedding-2B

### Identity and tensor evidence

| Property | Verified value |
| --- | --- |
| Workbench model ID | `qwen3-vl-embedding-2b-def7ecb282ac` |
| Workbench fingerprint | SHA-256 `def7ecb282ac77bafe77394dad6e1e678a16700063f8b834d5a1445f43c43ce8` |
| Git revision | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` |
| Git LFS model object | `sha256:c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` |
| SafeTensors file size | 4,255,140,312 bytes (3.963 GiB) |
| Header size / metadata | 76,240 bytes / `format=pt` |
| Tensor count / dtype | 625 / all BF16 |
| Exact parameter count | 2,127,532,032 |
| Parameter split | 1,720,574,976 language (80.87%); 406,957,056 vision (19.13%) |
| Header/payload validation | Shape-derived byte counts and contiguous offsets match exactly |

The text tower has 28 dense layers with hidden size 2,048, intermediate size 6,144, 16 query heads, and 8 key/value heads. The vision tower has 24 blocks, hidden size 1,024, 16 heads, patch size 16, temporal patch size 2, and spatial merge size 2. There is no `lm_head`, `expert`, or `router` tensor. Although `config.json` names `Qwen3VLForConditionalGeneration`, the weight layout and SentenceTransformers metadata make this checkpoint an embedding model.

### Embedding contract and limits

The packaged SentenceTransformers pipeline is multimodal feature extraction to `last_hidden_state`, last attended-token pooling with `include_prompt=true`, and L2 normalization. Native output width is 2,048. Matryoshka output keeps the leading requested dimensions from 64 through 2,048 and normalizes again. The API enforces that range: a 32-dimensional request returned structured HTTP 422 instead of executing an invalid path.

The default instruction is `Represent the user's input.` and cosine is the declared similarity function. Metadata supports text, image, video, and structured message inputs. There is no audio contract; an audio upload returned a structured `capability_unavailable` response listing text, image, and video as supported modalities.

| Text-limit source | Value |
| --- | ---: |
| `text_config.max_position_embeddings` | 262,144 |
| `tokenizer_config.model_max_length` | 262,144 |
| Model card / model table | 32,768 |
| Bundled reference script `MAX_LENGTH` | 8,192 |

The adapter default remains 8,192 until longer real runs are measured. Media metadata also conflicts: the image processor declares 4,096 to 1,310,720 pixels while the bundled script permits 1,843,200; video metadata declares 2 FPS and up to 768 frames while the script uses 1 FPS, at most 64 frames, and an approximately 7.5-megapixel total budget. The lower image bound and script's conservative video budget remain operational guidance. The current API does not separately time preprocessing or report expanded visual positions.

### Corrected loading path

The runtime uses built-in `Qwen3VLModel` and `Qwen3VLProcessor`, local-files-only loading, `trust_remote_code=false`, BF16 CUDA, no quantization, and attention selection `auto`. The checkpoint keys begin with `model.` because it was saved from an outer task wrapper; the built-in base model expects `language_model.` and `visual.`. The validated loader therefore applies `key_mapping={r"^model\.": ""}`.

An earlier attempt omitted that mapping and left weights unbound. No results from that attempt count as model validation. The corrected loader emitted no missing-weight warning and exact comparison of a loaded embedding tensor against the checkpoint tensor passed.

The latest cold CUDA/BF16 load completed in **5.750 s**, with 4,256,128,000 CUDA bytes allocated and 4,261,412,864 bytes reserved.

### Runtime-verified embeddings

All vectors in the table were finite and L2-normalized after any truncation. Text rows used four inputs containing an identical pair, a semantically related item, and an unrelated item.

| Dimensions | Identical cosine | Related cosine | Unrelated cosine | Total time |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.999999854 | 0.867071 | 0.129546 | 921 ms |
| 256 | 0.999999935 | 0.762189 | 0.186077 | 172 ms |
| 1,024 | 1.000000050 | 0.714048 | 0.227925 | 172 ms |
| 2,048 | 1.000000036 | 0.692967 | 0.224739 | 156 ms |

The first 64-dimensional request included first-request overhead and should not be compared directly with the following warm requests. Values slightly above 1 are floating-point rounding around unit norm, not evidence of a similarity greater than the mathematical cosine bound.

A two-item image/mixed text-image request returned finite, unit-normalized 64-dimensional vectors with cosine similarity **0.813329** in 656 ms. A short-video request returned a finite, unit-normalized 64-dimensional vector in 110 ms. These prove the tested preprocessing and forward paths; they do not establish retrieval quality, maximum media limits, or every possible mixed combination.

## Hardware and software used for runtime evidence

| Component | Observed value |
| --- | --- |
| OS | Windows 11, 64-bit |
| CPU | Intel Core i7-13620H, 10 cores / 16 logical processors |
| System memory | 31.7 GiB total |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8,585,216,000 bytes (8,187.5 MiB) |
| CUDA capability / driver | Compute capability 8.9 / driver 610.88 |
| Python | 3.12.9 |
| PyTorch | 2.8.0+cu128 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| SafeTensors / NumPy | 0.8.0 / 2.5.3 |

`nvidia-smi` is the authoritative VRAM source. Windows WMI's legacy `AdapterRAM` field under-reports this adapter and is not used as the memory budget.

Native CUDA is verified. WSL sees the GPU, but Docker GPU execution is currently blocked by the host's missing NVIDIA Container Toolkit/CDI integration (`failed to discover GPU vendor from CDI`). That deployment prerequisite does not invalidate native Windows CUDA evidence.

## Workbench-level evidence and remaining scope

The following remain fixture verified rather than supplied-checkpoint runtime verified:

- generic encoder-decoder generation;
- one-active-run admission, bounded queueing, overflow rejection, queued cancellation, and lifecycle conflict handling;
- HTTP origin/bearer enforcement and WebSocket authentication/replay behavior.

Replay branch construction, `parent_run_id`, recorded-setting reuse, and exact same-environment output identity now have both fixture and supplied-DeepSeek runtime evidence. This remains a within-environment guarantee, not a claim of identity across devices, dtypes, drivers, or library versions.

Remaining real-checkpoint work is intentionally bounded:

1. Run Qwen on CPU and record load, execution, memory, and unload behavior.
2. Stress increasing DeepSeek context, Qwen text/media sizes, and batches under explicit RAM/VRAM budgets before changing conservative defaults.
3. Benchmark the eager-kernel, persistence, throughput, and memory overhead of bounded attention attribution on longer supplied-checkpoint runs; hidden-state probes remain unimplemented.
4. Validate generic encoder-decoder generation against a real local checkpoint when one is available.
5. Install/configure NVIDIA Container Toolkit and CDI in WSL before labeling the NVIDIA Compose profile available.

Only the exact paths above should be displayed as runtime verified. Header/metadata evidence and deterministic fixtures must not be promoted to real-checkpoint claims.
