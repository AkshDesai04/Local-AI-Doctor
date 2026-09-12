# Benchmark and real-model validation results

Measurement date: 2026-09-13 (runs recorded 2026-09-12 UTC)

These results are a curated validation snapshot from one local machine. They establish that specific inference and observability paths work; they are not model-quality scores, maximum-capacity claims, or portable performance promises. Times are persisted engine measurements unless explicitly labeled as model load or request total.

## Execution identity

| Item | Value |
| --- | --- |
| Host OS | Windows 11, 64-bit |
| CPU | Intel Core i7-13620H, 10 cores / 16 logical processors |
| RAM | 31.7 GiB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes (8,187.5 MiB) |
| GPU driver / compute capability | 610.88 / 8.9 |
| Python | 3.12.9 |
| PyTorch | 2.8.0+cu128 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| SafeTensors / NumPy | 0.8.0 / 2.5.3 |
| Runtime | Isolated local worker; Transformers reference decode loop |
| Attention selection | `auto`; a more specific effective kernel was not persisted in the curated run record |
| Quantization | None |
| Concurrency | One active inference; one loaded model at a time |

Model identities:

| Model | Workbench ID | SHA-256 discovery fingerprint | Parameters / weights |
| --- | --- | --- | ---: |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-r1-distill-qwen-1-5b-7ab2d923d545` | `7ab2d923d54590e3515bfd20bd97141630a7ffbfc3471052734a01e58fa4a905` | 1,777,088,000 / BF16 |
| Qwen3-VL-Embedding-2B | `qwen3-vl-embedding-2b-def7ecb282ac` | `def7ecb282ac77bafe77394dad6e1e678a16700063f8b834d5a1445f43c43ce8` | 2,127,532,032 / BF16 |

See the [model capability report](model-capability-report.md) for upstream revisions, SafeTensors evidence, loader details, context conflicts, and exact validation scope.

## Cold model loads

Load time is excluded from engine TTFT and generation totals.

| Model | Device / dtype | Cold-load observations | Post-load memory snapshot |
| --- | --- | ---: | --- |
| DeepSeek | CUDA / BF16 | 6.438 s; 8.406 s | latest: 3,556,011,520 CUDA bytes allocated; 3,753,902,080 reserved |
| DeepSeek | CPU / BF16 | 4.313 s | 3,794,923,520-byte process RSS during the subsequent short run |
| Qwen embedding | CUDA / BF16 | 5.750 s | 4,256,128,000 CUDA bytes allocated; 4,261,412,864 reserved |

Cold-load variation is expected from filesystem cache, process state, and framework initialization. Two DeepSeek CUDA samples are shown rather than presenting either as a stable distribution. The CPU load and CUDA loads are different device/dtype paths and should not be compared as inference throughput.

## DeepSeek correctness and telemetry run

Run `cfc1c2b7-a6fb-4f98-9cce-3a2beb240e3b` used deterministic reference mode, explicit CUDA/BF16, attention selection `auto`, no quantization, full instrumentation, seed 0, and greedy sampling (`temperature=0`, `top_k=0`, `top_p=1`). The arithmetic prompt `17*23` rendered to 29 tokens; the output budget was 384.

| Metric | Result |
| --- | ---: |
| Generated tokens | 266 |
| Finish reason | EOS |
| Final answer | 391 |
| Reasoning / answer tokens | 99 / 167 |
| Prefill | 219 ms |
| Engine TTFT | 344 ms |
| Steady decode | 25.1638 tokens/s |
| End-to-end | 24.4598 tokens/s |
| Conditional generated-response perplexity | 1.0724 |

The output contained a closing `</think>` boundary, stored token IDs reconstructed the exact displayed stream, and timing fields were nonnegative and monotonic. This was one long correctness sample, not an aggregate throughput benchmark.

Teacher-forced prompt scoring was separately validated on a 17-token causal prompt. Sixteen next-token predictions were included, the first token was correctly excluded, and prompt perplexity was **7.87005**.

## DeepSeek CPU smoke

Run `1bf0660d-2ce2-4981-a556-4508e7081033` explicitly selected CPU/BF16 with no fallback, deterministic reference mode, basic instrumentation, seed 0, greedy decoding, 12 prompt tokens, and a 2-token output budget.

| Metric | Result |
| --- | ---: |
| Cold load | 4.313 s |
| Prefill | 609 ms |
| Engine TTFT | 625 ms |
| Steady decode | 10.6383 tokens/s |
| End-to-end | 2.7816 tokens/s |
| Process RSS | 3,794,923,520 bytes |
| Finish reason | Length after 2 tokens |

The runtime warned that CPU BF16 kernel support depends on processor architecture and operation. A two-token run establishes a valid execution path; it is too short to characterize sustained CPU throughput and is not directly comparable to the 266-token CUDA run.

## Instrumentation overhead

The controlled series used the same 24-token rendered prompt for every run on one loaded DeepSeek CUDA/BF16 model. Every measured run generated exactly 64 tokens with deterministic reference mode, seed 12345, greedy sampling (`temperature=0`, `top_k=0`, `top_p=1`), no quantization, and attention selection `auto`. One first-run warm-up was excluded. `off` and `basic` each have six observations; `token` and `full` each have three. The reported statistic is the median of each metric, and overhead uses the median generation duration:

```text
overhead_percent = 100 * (profiled_median_ms / off_median_ms - 1)
```

| Instrumentation | n | Generation ms | Prefill ms | TTFT ms | Decode tok/s | End-to-end tok/s | Median peak CUDA bytes | Overhead vs `off` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 6 | 2,086 | 31 | 31 | 30.6574 | 30.6812 | 3,602,586,624 | baseline |
| `basic` | 6 | 2,047 | 39.5 | 47 | 31.2500 | 31.2653 | 3,602,586,624 | -1.870% |
| `token` | 3 | 2,172 | 31 | 47 | 29.4255 | 29.4659 | 3,602,586,624 | +4.123% |
| `full` | 3 | 2,219 | 47 | 47 | 28.8066 | 28.8418 | 3,602,586,624 | +6.376% |

The negative `basic` delta is measurement noise, not evidence that instrumentation accelerates inference. Its observed overhead is within the 5% acceptance threshold. Detailed `token` collection performs full-distribution probability/rank/entropy work; `full` additionally synchronizes CUDA for more accurate profiling, so their measured overhead is expected and is surfaced rather than hidden. Representative runs are `7173fd3b-e3ae-42b7-bc7f-2b1ea7fe2920` (`off`), `99a13296-8f95-491e-ad48-3b1664dae0ac` (`basic`), `730618a4-053e-40de-9786-8a99cdf0303d` (`token`), and `be439fe9-1b5a-488d-bf52-a7c59f268a54` (`full`).

## Qwen embedding validation

The corrected loader used the real Qwen checkpoint on CUDA/BF16 with no quantization, local-files-only loading, and the required `model.` key-prefix mapping. The first 64-dimensional request included first-request overhead; later dimensions were warm requests. Every returned vector was float32-compatible, finite, and unit-normalized after truncation.

| Dimensions | Batch | Identical cosine | Related cosine | Unrelated cosine | Request total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 0.999999854 | 0.867071 | 0.129546 | 921 ms |
| 256 | 4 | 0.999999935 | 0.762189 | 0.186077 | 172 ms |
| 1,024 | 4 | 1.000000050 | 0.714048 | 0.227925 | 172 ms |
| 2,048 | 4 | 1.000000036 | 0.692967 | 0.224739 | 156 ms |

The cosine values slightly above 1 are ordinary floating-point rounding around unit norm. The related/unrelated ordering passed at every tested dimension; this is a semantic smoke check, not a retrieval benchmark.

Media validation:

| Input | Dimensions / batch | Result | Request total |
| --- | --- | --- | ---: |
| Image plus mixed text/image | 64 / 2 | finite, unit-normalized; cross-item cosine 0.813329 | 656 ms |
| Short video | 64 / 1 | finite, unit-normalized | 110 ms |
| Audio upload | — | structured `capability_unavailable`; no model execution | — |
| Requested dimension 32 | — | structured HTTP 422; valid range 64–2,048 | — |

The API does not yet isolate media preprocessing latency from forward time, so these totals must not be interpreted as pure model-forward benchmarks. Source media size/frame metadata was not retained in this curated report; the results validate only the exact smoke assets used during the run.

## Determinism, cancellation, and replay

Runs `b1d6c4cc-25d8-4618-9972-02fe6cdeb1d3` and `c5d15051-0333-4bce-b470-e9b02f741fc8` used the same loaded checkpoint, CUDA/BF16 backend, deterministic reference mode, seed 42, temperature 0.6, top-k 20, top-p 0.95, and a 32-token budget. Both completed at the length limit with identical token ID sequences and decoded text. Their observed steady decode rates were 26.4505 and 27.1930 tokens/s. This proves same-environment repeatability, not CPU/CUDA or cross-version identity.

Cancellation run `a12ef21a-1c67-44ab-8ecd-34e7c6235b6c` was stopped after 28 tokens of a 512-token budget. The partial message and token events were retained, terminal status was `cancelled`, and the same state was present after application restart.

Corrected replay was then validated with the supplied DeepSeek checkpoint. Source run `0a398d6f-4a30-4c0b-9765-ecf394752f4b` and replay run `9dd7f310-03af-4514-9290-23ecc18079fb` both completed with 19 prompt tokens, 32 generated/stored tokens, seed 42, deterministic reference mode, temperature 0.6, top-k 20, and top-p 0.95. The replay points to the source through `parent_run_id`; rendered prompts, token ID sequences, and decoded text were identical. Source generation took 1,578 ms and replay took 1,141 ms. Timing equality is not expected or required for deterministic output replay.

## Result disposition

| Claim | Disposition |
| --- | --- |
| Native DeepSeek CUDA/BF16 generation | Runtime verified for the recorded short and 266-token paths |
| DeepSeek CPU execution | Runtime verified as a two-token BF16 smoke path |
| Causal prompt scoring | Runtime verified on one 17-token prompt |
| Basic instrumentation overhead within 5% of off | Met in a controlled fixed-prompt series: -1.870%, interpreted as noise |
| Detailed instrumentation overhead | Measured in the same controlled series: +4.123% `token`, +6.376% `full` |
| Qwen text Matryoshka embeddings | Runtime verified at 64, 256, 1,024, and 2,048 dimensions |
| Qwen image/mixed/video smoke paths | Runtime verified for the tested assets and dimensions |
| Qwen CPU | Not tested |
| Real-checkpoint encoder-decoder generation | Not tested; architecture path is fixture verified only |
| Corrected real replay | Runtime verified with identical rendered prompt, token IDs, and decoded text |
| NVIDIA Docker GPU path | Blocked by host CDI/toolkit integration; native CUDA results are unaffected |

No maximum context, maximum batch, long-video, sustained CPU, thermal/power, or multi-user throughput claim is made from these samples.
