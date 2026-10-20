# DeepSeek V4.1 on Metal: runtime configuration

This page lists everything a V4.1 deployment configures, and nothing else. Every
setting a server or client integration needs is a command-line option. The
environment switches at the end are diagnostics: a normal launch sets none of
them, and the defaults they turn off are the configuration that was measured
(see [RECEIPTS-V41-1017.md](RECEIPTS-V41-1017.md)).

## 1. A serving launch

Run from the repository checkout, or pass `--chdir` (see section 4):

```
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --metal \
    --host 127.0.0.1 --port 8195 -c 393216 --prefill-chunk 8192 \
    --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark --mtp-exact-sampling \
    --kv-disk-dir /path/to/kvcache --kv-disk-space-mb 98304 \
    --kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096 \
    --default-top-p 0.95 --default-min-p 0 \
    --utility-ctx 32768 --serial-reasoning
```

This is the production launch on the M3 Ultra, with the model paths that
the README's Prepare step writes. Nothing in
the environment is required.

Project Trench needs `--serial-reasoning`. Its earlier server kept V4.1
reasoning serial unconditionally and drafted only answers and tool calls. Here
that policy is an option, off by default. Without it, DSpark also drafts
reasoning spans.

## 2. Options

### Model and decode

| Option | Default | What it does |
| --- | --- | --- |
| `-c N` | — | Allocated context. V4.1 keeps Think Max at any context; V4 models drop it to high below `-c 393216`. |
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

The disk cache turns on only after a numerics canary passes. The canary is a
one-shot 4,096-token prefill in a side session; its logits hash binds every
entry to this build's prefill numerics. It runs once, as soon as the server
listens, ahead of any request. The server accepts requests meanwhile and queues
them, and none runs until the canary's verdict, so no request is served cold
just because the cache is not on yet: once the canary passes, a queued request
whose prefix is saved on disk restores it.

This has a startup cost. For about the canary's duration after listen (one
4,096-token prefill, about 7.1 s on the M3 Ultra: 7.04 to 7.18 s across the
release's validation windows), every request waits for the verdict,
including requests that have no disk hit. A fresh 11-token prompt sent right
after a restart took 7.3 s to first token.
`KV disk cache on: canary ...` in the log marks the verdict and gives its time.
If the canary fails, the queued requests run with the cache off for the life of
the process, and a warning says why. Entries stay on disk and are still evicted
under the space budget. Entries saved under a different canary are refused and
evicted the same way. That includes entries written by the stepped canary of
earlier development builds of this branch (canary `e8c0517e0b2d6638`).

A request that restores a prefix from disk and then extends it saves its own
deeper checkpoint. When the prompt reaches at least one continued interval
(`--kv-cache-continued-interval-tokens`) past the restored prefix, and is no
longer than `--kv-cache-cold-max-tokens`, the server stores a checkpoint at
the aligned prompt length (or a later chat anchor), as it does for a cold
first prompt. The log line `kv cache extends restored prefix cached=N
store=M prompt=P` marks it. Earlier builds stored nothing at the new depth
in this case, because the cold store ran only when nothing was restored and
V4.1 cannot stage a checkpoint in the middle of a prefill; every later
request then restored only the shorter prefix. In the release validation, a
67,314-token prompt that extended a restored 8,192-token checkpoint stored
65,536 tokens; after a restart it restored those 65,536 tokens, 5.6 s to
first token instead of 72.2 s, with the same output.

To exercise the failure path, `make ds4-server-testhooks` builds a separate
test binary in which `DS4_KV_CANARY_FAIL_TEST=1` turns a completed canary into
a failed verdict. `ds4-server` does not read that variable.

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

Every `DS4_METAL_DISABLE_V41_*` switch in the source is listed here. Each
turns off one default and restores the path it replaced. Several apply only
to one configuration (prefill, DSpark verify, SSD streaming or two-host
tensor parallelism); the heading or the line says so.

Decode step:

| Switch | Turns off |
| --- | --- |
| `DS4_METAL_DISABLE_V41_BLEND` | blended Engram, router and indexer q_b readers |
| `DS4_METAL_DISABLE_V41_ROUTER_BF16` | BF16 router copy built at load |
| `DS4_METAL_DISABLE_V41_ENGRAM_FP8` | Engram E4M3 copy built at load |
| `DS4_METAL_DISABLE_V41_INDEXER_FP8` | indexer q_b E4M3 copy built at load |
| `DS4_METAL_DISABLE_V41_ENGRAM_ADD_TG` | the exact Engram gate with one 256-thread group per stream, in decode and DSpark verify (bit-identical to the 32-lane kernel) |
| `DS4_METAL_DISABLE_V41_SPLIT_COLLAPSE` | the decode HC collapse split over 32 + 10 groups (one 1024-thread group instead) |
| `DS4_METAL_DISABLE_V41_TOPK_FUSED_MERGE` | the decode top-k's staged and fused merge ladder (one dispatch per merge level instead) |
| `DS4_METAL_DISABLE_V41_SAMPLER_GPU` | GPU top-p selector |
| `DS4_METAL_DISABLE_V41_DECODE_QUEUE` | queuing a decode step's layers in one command stream with periodic flushes |
| `DS4_METAL_DISABLE_V41_QUEUED_HEAD` | the output head in that queued stream |
| `DS4_METAL_DISABLE_V41_PREENCODE` | pre-encoding the next token's first command buffer |
| `DS4_METAL_DISABLE_V41_GPU_BRIDGE` | finite GPU bridge behind the decode head |
| `DS4_METAL_DISABLE_V41_CAPTURE_IN_HEAD` | DSpark capture inside the head's buffer |
| `DS4_METAL_DISABLE_V41_IRQ_TABLE` | fused index-query RoPE and quantization |
| `DS4_METAL_DISABLE_V41_ROPE_TAB` | per-step RoPE coefficient table |
| `DS4_METAL_DISABLE_V41_ROPE_QUANTIZE` | fused RoPE and quantization of a row written to the cache |
| `DS4_METAL_DISABLE_V41_HEADS_ONCE` | attention heads transformed once in the reduce |
| `DS4_METAL_DISABLE_V41_DEFERRED_HC` | deferring the attention HC collapse behind the query projection |
| `DS4_METAL_DISABLE_V41_DEFERRED_FFN_HC` | the same deferral on the FFN side, behind the router |
| `DS4_METAL_DISABLE_V41_QUERY_KV` | storing the KV row inside the deferred-collapse query kernel |
| `DS4_METAL_DISABLE_V41_HC_BLOCK_INPUT` | fused HC block input (mix, norm and split in one kernel) |
| `DS4_METAL_DISABLE_V41_EXPAND_FUSION` | attention output-B and shared-expert down matmuls expanded straight into the HC streams |
| `DS4_METAL_DISABLE_V41_CONCURRENT_FFN` | the shared expert in the routed experts' concurrent encoder |
| `DS4_METAL_DISABLE_V41_Q8_SHORT` | short-K Q8 kernels for the 1280→32768 and 2304→5120 matvecs and the shared-expert down |
| `DS4_METAL_DISABLE_V41_LINEAR_BF16` | the linear BF16 rounding kernel (the generic quantize kernel instead) |
| `DS4_METAL_DISABLE_V41_ROUTER_ONE` | the fused router selection (also used for a few batched rows) |
| `DS4_METAL_DISABLE_V41_ROUTER_HIER` | the hierarchical one-row 384→6 router select |
| `DS4_METAL_DISABLE_V41_LOW_STATIC`, `_HC_CONSUMER_STATIC`, `_HC_JOIN_LITERAL`, `_QA_KV_PAIR_LITERAL`, `_GU_ENTRY` | literal-shape decode entries |

Sparse attention indexer and top-k:

| Switch | Turns off |
| --- | --- |
| `DS4_METAL_DISABLE_V41_INDEX_BF16` | the BF16 view of the index keys (scorers read the F32 keys) |
| `DS4_METAL_DISABLE_V41_SCORE_SPLIT` | several lanes per key in the BF16 full-row scorer below 65,536 keys |
| `DS4_METAL_DISABLE_V41_INDEX_SCORE_WIDE` | the wide index scorer |
| `DS4_METAL_DISABLE_V41_INDEX_MASKED` | the masked wide index scorer |
| `DS4_METAL_DISABLE_V41_INDEX_CANDIDATES` | candidate-block index scoring |
| `DS4_METAL_DISABLE_V41_INDEX_CAND_ONCE` | building the candidate list once per token |
| `DS4_METAL_DISABLE_V41_CAND_SPLIT` | two lanes per key in the candidate scorer |
| `DS4_METAL_DISABLE_V41_CAND_SELECT_ONE` | the candidate select in one threadgroup (the radix select instead) |
| `DS4_METAL_DISABLE_V41_TOPK_SELECT` | radix select for wide rows (the merge ladder instead) |
| `DS4_METAL_DISABLE_V41_TOPK_GATHER` | finishing a one-row radix select from one gathered pass |
| `DS4_METAL_DISABLE_V41_TOPK_SORT512` | 512-wide sort of the top-k winners |
| `DS4_METAL_DISABLE_V41_TOPK_PREFIX` | the ordered top-512 prefix merge (and with it the fused ladder) |
| `DS4_METAL_DISABLE_V41_TOPK_SHUFFLE` | the SIMD-shuffle block sort |
| `DS4_METAL_DISABLE_V41_VERIFY_INDEX_BF16` | scoring DSpark verify rows over the BF16 key view |
| `DS4_METAL_DISABLE_V41_VERIFY_TILED_INDEX` | the tiled index scorer for 2 to 7 verify rows |

DSpark verify and batched sessions:

| Switch | Turns off |
| --- | --- |
| `DS4_METAL_DISABLE_V41_PUBLIC_CONTROLLER` | the public fork's DSpark admission algorithm on single-host V4.1 (the engine's established controller instead) |
| `DS4_METAL_DISABLE_V41_SESSION_BATCH` | decoding several sessions in one batched pass |
| `DS4_METAL_DISABLE_V41_ROWS_MMA` | matrix kernels for 3 to 8 row Q8 matmuls |
| `DS4_METAL_DISABLE_V41_MOE_DEDUP` | reading each distinct expert once for 2 to 8 MXFP4 rows |
| `DS4_METAL_DISABLE_V41_MXFP4_TINY_BATCH` | the fused decode MoE kernels for batches of up to 8 rows |

Prefill:

| Switch | Turns off |
| --- | --- |
| `DS4_METAL_DISABLE_V41_INDEX_MMA` | register-resident prefill indexer scores |
| `DS4_METAL_DISABLE_V41_LAYER_PREFILL` | layer-major batched prefill (one token at a time instead) |
| `DS4_METAL_DISABLE_V41_WIDE_CHUNK` | prefill chunks above 2,048 rows |
| `DS4_METAL_DISABLE_V41_8K_CHUNK` | prefill chunks above 4,096 rows |
| `DS4_METAL_DISABLE_V41_WIDE_PREFILL` | wide prefill sweeps that carry rows across chunks |
| `DS4_METAL_DISABLE_V41_SHORT_SWEEP` | one sweep for prompts under 8,192 rows, keeping the final partial tile |
| `DS4_METAL_DISABLE_V41_DECODER_SUFFIX` | pruning the decoder pass to the rows later layers need |
| `DS4_METAL_DISABLE_V41_DEFER_DECODER` | deferring the decoder pass of a long prefill to the next sweep |
| `DS4_METAL_DISABLE_V41_COMPACT_CARRY` | compact (BF16) carry rows between sweeps |
| `DS4_METAL_DISABLE_V41_PREFILL_ALIAS` | aliasing prefill scratch buffers |
| `DS4_METAL_DISABLE_V41_ENGRAM_PREFETCH` | overlapping Engram reads with prefill compute |
| `DS4_METAL_DISABLE_V41_ENGRAM_PIPELINE` | pipelining those Engram reads through the sweep |
| `DS4_METAL_DISABLE_V41_ENGRAM_XSWEEP` | carrying an Engram prefetch across sweeps |
| `DS4_METAL_DISABLE_V41_BATCH_ENGRAM` | batched Engram reads (per-row reads instead; also turns off the prefetch) |
| `DS4_METAL_DISABLE_V41_BATCH_ATTN` | batched attention projections |
| `DS4_METAL_DISABLE_V41_BATCH_CORE` | batched attention core |
| `DS4_METAL_DISABLE_V41_BATCH_MOE` | batched MoE |
| `DS4_METAL_DISABLE_V41_BATCH_HC` | batched HC mix and expand |
| `DS4_METAL_DISABLE_V41_BATCH_INDEX` | batched index scoring |
| `DS4_METAL_DISABLE_V41_BATCH_INDEX_PROJ` | batched index projection |
| `DS4_METAL_DISABLE_V41_BATCH_COMPRESS` | batched compressed-KV publication on KV-source layers |
| `DS4_METAL_DISABLE_V41_BATCH_PUBLISH` | batched KV publication in the decoder pass |
| `DS4_METAL_DISABLE_V41_BATCH_TOPK` | batched top-k across rows |
| `DS4_METAL_DISABLE_V41_BATCH_HEAD` | batched output head over the prefill rows |
| `DS4_METAL_DISABLE_V41_PACKED_INDEX` | packed index scoring for 8 or more rows |
| `DS4_METAL_DISABLE_V41_PACKED_M32N128` | the packed 32×128 routed-expert matmul for 8,192 or more rows (IQ2_XXS / Q2_K experts) |
| `DS4_METAL_DISABLE_V41_ROUTER_BATCH` | the batched router matvec for 32 or more rows |
| `DS4_METAL_DISABLE_V41_ROUTER_MM` | the router as a matrix multiply for 256 or more rows |
| `DS4_METAL_DISABLE_V41_PREFILL_ADD`, `_PREFILL_EMBED`, `_PREFILL_EXPAND`, `_PREFILL_F16_ROWS2`, `_PREFILL_NORM`, `_PREFILL_SUM` | folded multi-row prefill kernels for the FFN add, embedding, HC expand, F16 projections, norm and HC sum |
| `DS4_METAL_DISABLE_V41_PREFILL_RB16` | the 16-row-block indexed attention kernel |
| `DS4_METAL_DISABLE_V41_PREFILL_LEAN` | the lean variant of that kernel |
| `DS4_METAL_DISABLE_V41_ENCODER_RESIDENCY` | SSD streaming: one read of each encoder layer per wide sweep |

Two-host tensor parallelism:

| Switch | Turns off |
| --- | --- |
| `DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE` | queuing the decode layers in one command stream |
| `DS4_METAL_DISABLE_V41_TP_BALANCED_ROUTE` | each rank taking half of the ordered expert selection |
| `DS4_METAL_DISABLE_V41_TP_GATE_EXPAND` | crossing the gate with the partial, summed in the consumer's expansion |
| `DS4_METAL_DISABLE_V41_TP_SHARED_OWNER` | the shared expert owned by one rank per layer, alternating |
| `DS4_METAL_DISABLE_V41_TP_SHARED_SPLIT` | splitting the shared expert's intermediate dimension across both ranks |
| `DS4_METAL_DISABLE_V41_TP_SLAB_WRITE` | producers writing their partials straight into the gate slot |
| `DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL` | batched prefill from 32 rows (256 instead) |
| `DS4_METAL_DISABLE_V41_PREFILL_Q8_ROWS8` | the eight-row Q8 kernel for TP attention partials on the M3 Ultra |

Other opt-outs:

| Switch | Turns off |
| --- | --- |
| `DS4_QWEN4_Q8_PREFILL_UNPACK=0` | unpack-then-F16 GEMM for wide BF16 Q8 prefill matmuls |
| `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY` | aligned tensor copies in the open compute encoder |
| `DS4_METAL_MODEL_TRACKED` | untracked V4.1 model views (`DS4_METAL_MODEL_UNTRACKED=1` still forces untracked views for any model) |

`DS4_METAL_V41_GPU_BRIDGE_ITERS=N` overrides the bridge length (default
17,591, tuned on the M3 Ultra; 0 turns it off).

### Experiments, off by default

| Switch | What it is | Status |
| --- | --- | --- |
| `DS4_V41_ENGRAM_DECODE=shift` | Ivan Fioravanti's F16 bit-pattern decode for the non-blend Engram reader | exact; superseded by the blend |
| `DS4_V41_SUFFIX_ALL=1` | exact decoder-suffix pruning for short sweeps and appends (Ivan Fioravanti) | awaiting a digest comparison |
| `DS4_METAL_V41_PREFILL_MMA=1` | simdgroup-matrix prefill attention (Ivan Fioravanti) | reassociated sums, not byte-identical; awaiting quality evidence |

### Diagnostics

| Switch | What it does |
| --- | --- |
| `DS4_METAL_V41_READER_COUNTS=1` | one exit line of decode reader dispatches per family and source |
| `DS4_V41_ROUTER_BF16_SIDECAR`, `DS4_V41_ENGRAM_FP8_SIDECAR`, `DS4_V41_INDEXER_FP8_SIDECAR` | name a sidecar for a family no `--v41-sidecar` named |
| `DS4_METAL_*_SOURCE` | load one Metal source file from another path |
