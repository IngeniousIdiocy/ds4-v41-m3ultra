# V4.1 prefill on Metal

## What changed

Batch embedding/HC initialization, fold existing BF16 rounding into producers, and reuse F16 weight staging across two token rows without changing each row's reduction order. Use lean rb16 attention staging for supported large batches. Start bounded Engram read-ahead across prefill sweeps, with 32 readers and owned token IDs checked before reuse.

Existing short-row verification fusions remain selected. New producer paths are guarded to supported local, nonstreaming normal Metal execution. Diagnostic `DS4_METAL_DISABLE_V41_PREFILL_*` switches retain comparison paths; the optimizations are enabled by default.

## Results

M3 Ultra, 80 GPU cores, 512 GB; native MXFP4 target/drafter, chunk 8192. Empty KV with warmed weights. The control and candidate share the accepted scalar code; only the prefill bundle changes in the paired tests.

| Prompt | Control | Complete bundle |
|---|---:|---:|
| 8192 tokens | 658.46 | 736.67 t/s |
| 63488 tokens | 741.33 | 845.75 t/s |

The 8k candidate is the final release repeat; earlier paired qualification was 658.46→736.22. The 62k paired qualification has four repeats per arm: best 741.59→846.05 t/s. Full persisted target/drafter state and final logits match. This is a prefill measurement, not a disk-cache restore rate.

Isolated 8k screens: producers 658.35→678.50, attention 658.46→675.90, Engram concurrency/read-ahead 658.12→694.43 t/s. Do not add these percentages.

Earlier complete-bundle append qualification (mean t/s):

| Append | Control | Bundle |
|---|---:|---:|
| 2048 after 8192 | 373.68 | 409.11 |
| 2048 after 63488 | 301.80 | 325.59 |
| 63 after 8192 | 88.41 | 90.61 |
| 63 after 63488 | 83.53 | 88.05 |

## Validation and reproduction

Randomized producer, embedding and attention gates include complete outputs, buffer guards, repeated inputs and changed-input checks. Full native snapshot comparisons cover fresh prompts and appends. Each extracted commit builds Metal host code and CPU engine objects. CUDA/distributed model execution was not tested.

```sh
make tests/test_v41_embed_batch tests/test_v41_prefill_folds tests/test_v41_prefill_attention
./tests/test_v41_embed_batch 50000
./tests/test_v41_prefill_attention 50000
```

Use the same model, corpus (`speed-bench/promessi_sposi.txt`), chunk size, allocated context and drafter configuration for native comparisons. Empty the session KV for full-prefill tests, keep weights warm, and report cached appends separately. Generation sweep timings cannot substitute for full-prompt prefill.

## Experiments and measurement corrections

A three-row Engram projection reused weights but was approximately 11% slower in the microbenchmark and had no fixed-input native gain. It was removed; the accepted two-row F16 producer path is separate.

An initial attention fixture used an absolute position where the API expects a ring slot; the API rejected it before dispatch. A first-use one-token append included setup overhead and was excluded from steady-state latency claims. Neither result supports a speed claim.
