# DeepSeek V4.1 on Metal: runtime configuration

This page lists everything a V4.1 deployment configures, and nothing else. Every
setting a server or client integration needs is a command-line option. The
environment switches at the end are diagnostics: a normal launch sets none of
them, and the defaults they turn off are the configuration that was measured
(see [RECEIPTS-V41-1017.md](RECEIPTS-V41-1017.md)).

## 1. A serving launch

Run from the repository checkout, or pass `--chdir` (see section 4):

```
./ds4-server -m DeepSeek-V4.1-Flash-MXFP4.gguf --metal \
    --host 127.0.0.1 --port 8195 -c 393216 --prefill-chunk 8192 \
    --mtp-model DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark --mtp-exact-sampling \
    --kv-disk-dir /path/to/kvcache --kv-disk-space-mb 98304 \
    --kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096 \
    --default-top-p 0.95 --default-min-p 0 \
    --utility-ctx 32768 --serial-reasoning
```

This is the production launch on the M3 Ultra, minus local paths. Nothing in
the environment is required.

Project Trench needs `--serial-reasoning`. Its earlier server kept V4.1
reasoning serial unconditionally and drafted only answers and tool calls. Here
that policy is an option, off by default. Without it, DSpark also drafts
reasoning spans.

## 2. Options

### Model and decode

| Option | Default | What it does |
| --- | --- | --- |
| `-c N` | — | Allocated context. Think Max needs `-c 393216` or more. |
| `--prefill-chunk N` | 4096 | Prefill chunk; V4.1 clamps at 8192. |
| `--mtp-model FILE --dspark` | off | Load the DSpark drafter and its admission controller. |
| `--mtp-exact-sampling` | off | Keep the sampled distribution exact under drafting. |
| `--v41-sidecar FILE` | none | Optional router BF16, Engram FP8 or indexer FP8 sidecar, one per family (repeat the option). The engine identifies each file by its `general.architecture`. Without sidecars, the narrow decode copies are built at load from the GGUF. |

The decode path needs no options. On resident single-host Metal it builds the
narrow copies at load and makes them resident. It also takes the blended
readers, the GPU top-p selector, the pre-encoded next-token buffer and the
other defaults recorded in the receipts. The narrow-copy build prints one line
at load, and each blend reader family prints one line on its first dispatch.

### Server policy (Project Trench and other integrations)

| Option | Default | What it does |
| --- | --- | --- |
| `--utility-ctx N` | 0 (off) | A side session of N tokens for `"ds4_utility": true` requests (titles, tags, follow-ups). It never thinks, never drafts and never touches the disk KV cache. Not with `--batched-session`. |
| `--serial-reasoning` | off | V4.1 reasoning always decodes serially; DSpark drafts only answers and tool calls. Trench runs with this. |
| `--default-reasoning-effort E` | high | Effort for a thinking request that names none: `low`, `medium`, `high` or `max`. A request's own effort, thinking switches and model aliases still win. |
| `--default-top-p F`, `--default-min-p F` | 1, 0.05 | Sampler defaults for requests that set none. |

### Disk KV cache

| Option | What it does |
| --- | --- |
| `--kv-disk-dir DIR` | Enable disk KV checkpoints in DIR. |
| `--kv-disk-space-mb N` | Disk budget for DIR (default 4096 when enabled). |
| `--kv-cache-cold-max-tokens N` | Save cold first prompts up to N tokens (default 30000; 0 disables). |
| `--kv-cache-continued-interval-tokens N` | Save aligned continued frontiers every N tokens (default 10000; 0 disables). |

`./ds4-server --help kv-cache` lists the rest (minimum tokens, boundary trim and
alignment).

The disk cache turns on only after a numerics canary completes. The canary is a
4,096-token prefill whose hash binds every entry to this build's prefill
numerics. The server listens and serves at once; until the canary completes it
neither loads nor stores disk entries, and it logs one line saying so. The
canary takes nine turns at the model (session setup, then eight 512-token
steps; about 12 s of model time in all on the M3 Ultra), one turn in each gap
between requests, so a request waits for at most
one step. Under unbroken load, with a request always queued, the canary waits,
and the disk cache stays off until enough gaps have passed. With one client
sending requests back to back this took 42 to 51 s after start, against 12 s on
an idle server. `KV disk cache on: canary ...` in the log marks the moment the
cache turns on. If the canary fails, the cache stays off for the life of the
process, and a warning says why.

## 3. Benchmarking

`ds4-bench` reproduces the decode numbers:

- `--temp 1 --top-p 0.95 --seed N` samples through the GPU selector.
- `--token-csv PATH` writes per-token times, plus per-frontier prefix and
  final-logit digests to `PATH.runs.csv`.
- `--jobs FILE` keeps one engine resident and runs save, load, run and quit
  job lines (see `ds4_bench.c`). Saved caches are snapshot format 4
  (`ds4_snapstore.h`): a load checks the producer identity, then the
  payload's size and hash, before restoring; `canary=N` on a save also binds
  the cache to the prefill numerics. Format-3 caches still load.

`DS4_BENCH_SHA_DIR` names a directory of recorded `*.sha256` files for the
job runner's cache identity.

## 4. Metal sources

The engine compiles its Metal library at startup from `metal/*.metal`,
resolved against the working directory, as upstream does. The GPU top-p
selector (`metal/sampler_selector.metal`) follows the same rule. So:

- run the binaries from the checkout root, or
- pass `ds4-server --chdir /path/to/checkout`, or
- point one file elsewhere with its `DS4_METAL_*_SOURCE` variable (for the
  selector, `DS4_METAL_SAMPLER_SELECTOR_SOURCE`).

If the selector source is missing, decode stays correct: one line says the
selector is unavailable, and sampling stays on the CPU.

## 5. Diagnostic and experimental switches

Nothing here is needed for normal use. Each opt-out restores the path the
default replaced, for A/B checks and bisection.

### Opt-outs of shipped V4.1 defaults (set to 1)

| Switch | Turns off |
| --- | --- |
| `DS4_METAL_DISABLE_V41_BLEND` | blended Engram, router and indexer q_b readers |
| `DS4_METAL_DISABLE_V41_ROUTER_BF16` | BF16 router copy built at load |
| `DS4_METAL_DISABLE_V41_ENGRAM_FP8` | Engram E4M3 copy built at load |
| `DS4_METAL_DISABLE_V41_INDEXER_FP8` | indexer q_b E4M3 copy built at load |
| `DS4_METAL_DISABLE_V41_ENGRAM_ADD_TG` | one 256-thread group per stream for the Engram gate (EADD, the one lever that changes arithmetic) |
| `DS4_METAL_DISABLE_V41_SAMPLER_GPU` | GPU top-p selector |
| `DS4_METAL_DISABLE_V41_PREENCODE` | pre-encoding the next token's first command buffer |
| `DS4_METAL_DISABLE_V41_GPU_BRIDGE` | finite GPU bridge behind the decode head |
| `DS4_METAL_DISABLE_V41_CAPTURE_IN_HEAD` | DSpark capture inside the head's buffer |
| `DS4_METAL_DISABLE_V41_IRQ_TABLE` | fused index-query RoPE and quantization |
| `DS4_METAL_DISABLE_V41_ROPE_TAB` | per-step RoPE coefficient table |
| `DS4_METAL_DISABLE_V41_HEADS_ONCE` | attention heads transformed once in the reduce |
| `DS4_METAL_DISABLE_V41_LOW_STATIC`, `_HC_CONSUMER_STATIC`, `_HC_JOIN_LITERAL`, `_QA_KV_PAIR_LITERAL`, `_GU_ENTRY` | literal-shape decode entries |
| `DS4_METAL_DISABLE_V41_INDEX_CANDIDATES` | candidate-block index scoring |
| `DS4_METAL_DISABLE_V41_INDEX_CAND_ONCE` | building the candidate list once per token |
| `DS4_METAL_DISABLE_V41_TOPK_SORT512` | 512-wide sort of the top-k winners |
| `DS4_METAL_DISABLE_V41_INDEX_MMA` | register-resident prefill indexer scores |
| `DS4_QWEN4_Q8_PREFILL_UNPACK=0` | unpack-then-F16 GEMM for wide BF16 Q8 prefill matmuls |
| `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY` | aligned tensor copies in the open compute encoder |
| `DS4_METAL_MODEL_TRACKED` | untracked V4.1 model views (`DS4_METAL_MODEL_UNTRACKED=1` still forces untracked views for any model) |

`DS4_METAL_V41_GPU_BRIDGE_ITERS=N` overrides the bridge length (default
17,591, tuned on the M3 Ultra; 0 turns it off).

### Experiments, off by default

| Switch | What it is | Status |
| --- | --- | --- |
| `DS4_V41_S1TG=ckrh` | single-group latency kernels: split HC collapse (c), fused top-k merge ladder (k), ballot router (r), eight heads per threadgroup in single-row decode attention (h) | experimental, opt-in. Bit-identical in micros; per-letter stack exactness pending. The hoisted paired norm (former letter n) was removed: it was not exact |
| `DS4_V41_ENGRAM_DECODE=shift` | Ivan Fioravanti's F16 bit-pattern decode for the non-blend Engram reader | exact; superseded by the blend |
| `DS4_V41_SUFFIX_ALL=1` | exact decoder-suffix pruning for short sweeps and appends (Ivan Fioravanti) | awaiting a digest comparison |
| `DS4_METAL_V41_PREFILL_MMA=1` | simdgroup-matrix prefill attention (Ivan Fioravanti) | reassociated sums, not byte-identical; awaiting quality evidence |

### Diagnostics

| Switch | What it does |
| --- | --- |
| `DS4_METAL_V41_READER_COUNTS=1` | one exit line of decode reader dispatches per family and source |
| `DS4_V41_ROUTER_BF16_SIDECAR`, `DS4_V41_ENGRAM_FP8_SIDECAR`, `DS4_V41_INDEXER_FP8_SIDECAR` | name a sidecar for a family no `--v41-sidecar` named |
| `DS4_METAL_*_SOURCE` | load one Metal source file from another path |
