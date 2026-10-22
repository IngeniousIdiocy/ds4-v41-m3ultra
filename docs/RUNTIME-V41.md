# DeepSeek V4.1 on Metal: options and switches

This page lists what this branch adds to DwarfStar main's command line and
environment for DeepSeek V4.1 Flash on Metal: the options, the 44 group
switches and the few other variables. Everything else is main's and is
documented in main's own guides ([serving](SERVER.md),
[speculative decoding](SPECULATIVE_DECODING.md), [models](MODELS.md),
[performance](PERFORMANCE.md)).

A normal launch sets no environment variable. Every group is on by default,
and its switch exists to turn it off for comparison or bisection.

## A serving launch

```sh
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --metal \
  --host 127.0.0.1 --port 8000 -c 393216 --prefill-chunk 8192 \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark \
  --kv-disk-dir kvcache --kv-disk-space-mb 98304 \
  --kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096 \
  --default-top-p 0.95 --default-min-p 0 \
  --utility-ctx 32768 --serial-reasoning
```

Run it from the repository root (or pass `--chdir`), because the engine compiles
its Metal sources from `metal/` at startup. The server has no authentication;
keep it on 127.0.0.1 unless the network is private.

## Options this branch adds

### ds4-server

| Option | Default | What it does |
| --- | --- | --- |
| `--utility-ctx N` | 0 (off) | Keeps a second session of N tokens for requests that set `"ds4_utility": true`, such as titles and summaries. They never think, never draft and never touch the disk KV cache, and they leave the main session alone. A request that does not fit in N tokens is refused. Needs non-batched mode. |
| `--serial-reasoning` | off | V4.1 decodes every reasoning span serially; DSpark drafts only answers and tool calls. No effect on other models or in batched mode. |
| `--default-reasoning-effort E` | `high` | Effort for a request that names none: `low`, `medium`, `high` or `max`. A request's own values still win. V4.1 renders `low` as effort level 50, as DeepSeek's encoder does. |
| `--default-top-p F`, `--default-min-p F` | main's defaults | Sampler values for a request that sets none. |

The server also changes two behaviours without an option. An idle single slot
prefills in the engine's prefill chunk instead of 2,048 tokens. And the disk KV
cache uses the version 2 format described below.

### ds4, ds4-server, ds4-agent, ds4-bench, ds4-eval

| Option | Default | What it does |
| --- | --- | --- |
| `--v41-sidecar FILE` | none | Loads an optional V4.1 decode sidecar: the BF16 router, the FP8 Engram projections or the FP8 indexer query projection, as DeepSeek released them. Repeat it once per family; the engine tells them apart by `general.architecture`. Each file is checked against the model's source revision and audited tensor by tensor; a mismatch refuses to start. A sidecar takes the place of the narrow copy that NARROW_WEIGHTS otherwise builds at load. Metal only. |

The sidecars are written by `gguf-tools/deepseek41_router_bf16_sidecar.py`,
`deepseek41_engram_fp8_sidecar.py` and `deepseek41_indexer_fp8_sidecar.py`;
see [the gguf-tools guide](../gguf-tools/README.md#v41-decode-sidecars).

### ds4-bench

| Option | What it does |
| --- | --- |
| `--temp F` | Samples instead of taking the greedy token. Refused with `--teacher-forced-decode` and with DSpark. |
| `--top-p F`, `--top-k N`, `--min-p F` | Sampler settings for `--temp` (defaults 1, 0, 0). |
| `--seed N` | The sampler seed, reset at each frontier. |
| `--token-csv PATH` | Per-token timings to PATH, and one row per frontier to `PATH.runs.csv` with the sampler settings, the speed and hashes of the prefix and of the final logits. |
| `--jobs FILE` | With `--token-csv`: keeps one engine and session loaded and runs job lines as they are appended (`save`, `load`, `run`, `quit`; see the comment in `ds4_bench.c`). `PATH.jobs.csv` gets one row per job. |

Saved caches use snapshot format 4 (`ds4_snapstore.h`): the payload and a `.meta`
file, written atomically, recording the producer, the payload's size and hash, the
lineage of an extended frontier and optionally a numerics canary. A load checks
all of it before the engine sees a byte. Format 3 caches still load. A save
refuses to run while any `DS4_METAL_DISABLE_V41_*` variable is set, so every saved
cache comes from the default numerics.

### gguf-tools

| Command | What it does |
| --- | --- |
| `deepseek41_quantize.py --quant mxfp4` | Keeps the checkpoint's own MXFP4 expert values, repacked into GGUF blocks with no calibration. |
| `deepseek41_quantize.py --dspark-out FILE` | Also writes the checkpoint's DSpark drafter to its own GGUF (Q8_0 dense weights, MXFP4 experts), with the Markov tables under the names DwarfStar's DSpark loader expects. Without `--out`, only that file is written. |
| `deepseek41_prepare.py` | From the public 1017 files to the two GGUFs in one command, with pinned digests. `--verify`, `--dry-run` and `--digest FILE...` as described in [the gguf-tools guide](../gguf-tools/README.md#prepare-deepseek-v41-flash-1017). |

## The disk KV cache (version 2)

`ds4-server` and `ds4-agent` keep main's disk KV options (`--kv-disk-dir`,
`--kv-disk-space-mb`, `--kv-cache-cold-max-tokens`,
`--kv-cache-continued-interval-tokens` and the rest; `./ds4-server --help`). What
changes is what a checkpoint is bound to:

- Each checkpoint records which model and drafter files wrote it (a hash of each
  GGUF's header and tensor table), the weight form prefill reads, and a hash of
  its payload. A file from other weights, an older format or with a damaged
  payload is refused before it touches the session, and is evicted under the
  space budget like any other entry. Version 1 files, including those written by
  main's server, are refused rather than loaded.
- After it starts listening, the server prefills a fixed 4,096-token prompt once
  in a side session and records the hash of its logits (the numerics canary).
  Requests are accepted meanwhile and wait for it. Checkpoints written under
  other numerics, for example with a group switched off, are refused. The log
  line `KV disk cache on: canary` marks the verdict. If the canary fails, the
  cache stays off for the life of the process and a warning says why.
- A request that restores a prefix and extends it by at least one continued
  interval stores its own deeper checkpoint (log line `kv cache extends
  restored prefix`).
- A prefill interrupted by shutdown or by the client keeps the entry it started
  from; during shutdown such a request is answered with HTTP 503.
- With the DSpark drafter loaded, a V4.1 payload also carries the drafter's
  window (three feature rings of 128 rows and their positions, 7.5 MiB), so a
  restored session drafts at once instead of first decoding 128 tokens
  serially. Payloads without the window, including main's, still load and warm
  up as before. Two-Mac sessions keep the old form.

`make ds4-server-testhooks` builds a test-only server in which
`DS4_KV_CANARY_FAIL_TEST=1` forces the canary to fail. The normal `ds4-server`
does not read that variable.

## The group switches

Each group has one off switch. The switches are read once, at the first use, into
one mask with one bit per group (`ds4_gpu_v41_off()` in `ds4_metal.m`, the bit
names in `ds4_gpu.h`). A switch counts as set when it is present, non-empty and
not `0`. With a switch set, that group runs DwarfStar main's code.

The column "Exact" says whether turning the group on keeps the output bit for
bit. "Yes" means the same bits as the code it replaces. Where it says otherwise,
the generated text stayed the same in the checks that were run, and the note says
what can differ.

### Feeding the GPU

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 0 | `DS4_METAL_MODEL_TRACKED` | Untracked model weight views (V4.1's default). Main's `DS4_METAL_MODEL_UNTRACKED=1` still forces untracked views for any model. Applied at load. | Yes |
| 1 | `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY` | Aligned tensor copies in the open compute encoder (every model). | Yes |
| 2 | `DS4_METAL_DISABLE_V41_DECODE_QUEUE` | One command stream per decode step, committed every two layers, with the head in the stream; the same two-layer commits for verify and short prefill batches. | Yes for serial decoding. With DSpark, the previous release's form of this group changed DSpark's verify bits but not its tokens; not rechecked on this branch. |
| 3 | `DS4_METAL_DISABLE_V41_ROUTER_SELECT` | Expert selection for one token (or each verify row) in one dispatch. | Yes |
| 4 | `DS4_METAL_DISABLE_V41_ENGRAM_ASYNC` | Engram reads on background threads during the first layers, for decode and batch steps; one reader per row on Apple. Off reads synchronously, without the reader. | Yes |
| 5 | `DS4_METAL_DISABLE_V41_FUSED_ROUNDING` | bf16 rounding inside the producing kernels. | Yes |

### Decode kernels

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 6 | `DS4_METAL_DISABLE_V41_NARROW_WEIGHTS` | Narrow router, Engram and indexer copies built at load (and the sidecars). The copies are built only when the switch is unset at load; turning it on later does not build them. | Yes |
| 7 | `DS4_METAL_DISABLE_V41_QKV_FUSION` | q_a and kv in one pass, their norms in one dispatch, RoPE in the query projection. | Yes |
| 8 | `DS4_METAL_DISABLE_V41_ROPE` | The per-step RoPE table and RoPE fused into the cache store, the index query and the attention reduce. | Yes |
| 9 | `DS4_METAL_DISABLE_V41_FFN_JOIN` | A layer's FFN outputs joined and expanded into the streams in one dispatch. | Yes |
| 10 | `DS4_METAL_DISABLE_V41_HC_FUSION` | The fused hyper-connection block input, projections expanded into the streams, the deferred mix. | Yes |
| 11 | `DS4_METAL_DISABLE_V41_SPLIT_REDUCE` | The Engram gate and the HC collapse spread over more threadgroups. | Yes |
| 12 | `DS4_METAL_DISABLE_V41_SHARED_EXPERT` | The shared expert's gate, up and SwiGLU in one dispatch. | Yes |
| 13 | `DS4_METAL_DISABLE_V41_CONCURRENT_FFN` | The shared expert beside the routed experts. | Yes |
| 18 | `DS4_METAL_DISABLE_V41_DECODE_ATTENTION` | Decode attention staging its selected rows straight from the cache. | Yes |
| 19 | `DS4_METAL_DISABLE_V41_FIXED_SHAPES` | Kernels built for V4.1's exact shapes. | Yes |
| 20 | `DS4_METAL_DISABLE_V41_SAMPLER_SCAN` | The sampler's first pass in 16 NEON lanes (every model, sampled decoding only). | Yes |

### Decode at depth

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 14 | `DS4_METAL_DISABLE_V41_INDEX_SCORE_WIDE` | One thread per key for index rows of 8,192 keys or more, and its masked form (shared with GLM). | No: the heads are added in a different order, so scores differ in the last bits and a key near the cut-off can change. In the checks the selections and logits did not change. |
| 15 | `DS4_METAL_DISABLE_V41_INDEX_BF16` | The bf16 copy of the index keys. | Yes |
| 16 | `DS4_METAL_DISABLE_V41_TOPK` | The radix select and the shorter sort and merge for the index top-k. | Yes by rank; keys with exactly equal scores can trade places. |
| 17 | `DS4_METAL_DISABLE_V41_INDEX_CANDIDATES` | Later layers scoring only layer 20's kept blocks. Needs INDEX_SCORE_WIDE and TOPK. | Yes |

### DSpark verify

These never run in one-token decoding, so serial output never depends on them.

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 21 | `DS4_METAL_DISABLE_V41_ROW_MATMULS` | Q8_0 and F16 products of 2 to 8 rows sharing one weight pass (every model). | Yes |
| 22 | `DS4_METAL_DISABLE_V41_ROWS_MMA` | Q8_0 products of 3 to 8 rows of a DSpark block (the drafter's rows and the verify block) on simdgroup matrices. Prefills never take it. | No: verify rows can differ from serial decoding in the last bits. |
| 23 | `DS4_METAL_DISABLE_V41_VERIFY_KERNELS` | 6 to 8 V4.1 MXFP4 verify rows through the fused decode MoE kernels (DSpark blocks only; prefills keep main's MoE kernels). | No, compared with main's DSpark: each verify row now gives the bits of that row decoded alone, not the batch kernels' bits, so DSpark's tokens can change. |
| 24 | `DS4_METAL_DISABLE_V41_PUBLIC_CONTROLLER` | Drafting by measured cost (`ds4_deepseek41_admission.h`). Off brings back main's confidence cut-off and scheduler. One Mac only. | A different policy by design: it changes where DSpark drafts, so DSpark's tokens can change. |
| 28 | `DS4_METAL_DISABLE_V41_MOE_DEDUP` | 2 to 8 MXFP4 rows reading each selected expert once (any MXFP4 model). | Yes |
| 29 | `DS4_METAL_DISABLE_V41_VERIFY_BATCH` | A session's 2 to 8 consecutive rows attending together and expanding their outputs into the streams. | Yes per row, except where ROWS_MMA applies. |
| 30 | `DS4_METAL_DISABLE_V41_FLASH_HEAD_SHARE` | Eight heads per threadgroup in multi-row flash attention (every model). | Yes |
| 41 | `DS4_METAL_DISABLE_V41_VERIFY_INDEXED` | The batched verify: a draft block attends like a short prefill. Off verifies row by row, and its tokens are serial decoding's. | No: verified logits are not serial decoding's bits, so greedy DSpark output can differ where two tokens are nearly tied. |

### Prefill

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 25 | `DS4_METAL_DISABLE_V41_PREFILL_INDEX` | Prefill index scores of 8 or more rows on simdgroup matrices. | Yes |
| 26 | `DS4_METAL_DISABLE_V41_PREFILL_ATTENTION` | Prefill attention staging 16 rows per barrier. | Yes |
| 27 | `DS4_METAL_DISABLE_V41_PREFILL_FOLDS` | bf16 rounding and glue folded into the producers of large prefills. | Yes |
| 31 | `DS4_METAL_DISABLE_V41_PREFILL_MATMUL` | Unpacking Q8 weights to F16 once for products of 32 or more rows on the M3 Ultra. Main's `DS4_QWEN4_Q8_PREFILL_UNPACK=0` also turns it off. | Yes |
| 32 | `DS4_METAL_DISABLE_V41_PREFILL_INDEX_BATCH` | Batch top-k from 513 visible keys, select-all for short rows, batched candidates, ratio-1 batch publishing. | Yes |
| 33 | `DS4_METAL_DISABLE_V41_SHORT_SWEEP` | One sweep for 3,072 to 8,191 tokens with the decoder suffix pruned. | No: a short prompt is split into chunks differently, so prefill rounding can differ in the last bits (seen in the previous release; not rechecked on this branch). |
| 34 | `DS4_METAL_DISABLE_V41_ENGRAM_READAHEAD` | Reading the next sweep's Engram rows during the current sweep. | Yes |
| 42 | `DS4_METAL_DISABLE_V41_ENGRAM_READERS` | 32 concurrent Engram readers for a prefill sweep (16 when off). | Yes |
| 43 | `DS4_METAL_DISABLE_V41_SHORT_MMA` | An eight-row prefill batch's Q8_0 products on simdgroup matrices in the row kernels' order of sums, outside DSpark blocks. Needs ROW_MATMULS, one Mac and an M1 to M4 GPU. Off runs main's row kernels. | Yes: bit for bit main's row kernels (unit test on every V4.1 Q8_0 shape, and an eight-token prompt's logits). |

### The decode handoff and sampling

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 35 | `DS4_METAL_DISABLE_V41_PREENCODE` | Encoding the next token's first command buffer during the current token's wait, and the short GPU bridge kernel. | Yes |
| 36 | `DS4_METAL_DISABLE_V41_SAMPLER_GPU` | The GPU top-p summary at temperature 1, top-p 0.95, no top-k and no min-p. | Yes: the token and the random state are the CPU sampler's. |

### Two Macs

Built and unit-tested, not measured (no second Mac was available). One-Mac
decode never reaches them.

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 37 | `DS4_METAL_DISABLE_V41_TP_BALANCED_ROUTE` | Each Mac taking half of every token's ordered expert list. Read at load. | Not measured |
| 38 | `DS4_METAL_DISABLE_V41_TP_SHARED_SPLIT` | Splitting the shared expert's hidden width across both Macs. | Not measured |
| 39 | `DS4_METAL_DISABLE_V41_TP_GATES` | Partials written straight into the gate slots and added by the HC expansion. | Not measured |
| 40 | `DS4_METAL_DISABLE_V41_TP_PREFILL_ROWS8` | The attention output's K-slice rows in one dispatch, eight output rows per group on the M3 Ultra. | Yes (unit test) |

## Other variables this branch adds

| Variable | What it does |
| --- | --- |
| `DS4_V41_ROUTER_BF16_SIDECAR`, `DS4_V41_ENGRAM_FP8_SIDECAR`, `DS4_V41_INDEXER_FP8_SIDECAR` | Name a sidecar file for a family that no `--v41-sidecar` named. Diagnostics; the option is the normal way. |
| `DS4_METAL_SAMPLER_SELECTOR_SOURCE` | Loads the GPU top-p selector's Metal source (`metal/sampler_selector.metal`) from another path. If the source is missing, sampling stays on the CPU. |
| `DS4_V41_SUFFIX_ALL=1` | An opt-in experiment: builds the exact decoder suffix for any prefill sweep of 384 tokens or more, without the tile rounding. Off by default. |
| `DS4_KV_CANARY_FAIL_TEST=1` | Forces the numerics canary to fail. Read only by `ds4-server-testhooks`. |
| `DS4_BENCH_SHA_DIR` | A directory of recorded `*.sha256` files that `ds4-bench --jobs` uses for the weight identities in snapshot format 4. |

## Main's own V4.1 switches

DwarfStar main has its own V4.1 Metal switches, and the branch keeps them as they
are. They turn off main's batched prefill, its sweeps and its two-Mac decode
queue, among others: `DS4_METAL_DISABLE_V41_` followed by `8K_CHUNK`,
`BATCH_ATTN`, `BATCH_COMPRESS`, `BATCH_CORE`, `BATCH_ENGRAM`, `BATCH_HC`,
`BATCH_HEAD`, `BATCH_INDEX`, `BATCH_INDEX_PROJ`, `BATCH_MOE`, `BATCH_PUBLISH`,
`BATCH_TOPK`, `COMPACT_CARRY`, `DECODER_SUFFIX`, `DEFER_DECODER`,
`ENCODER_RESIDENCY`, `ENGRAM_PIPELINE`, `ENGRAM_PREFETCH`, `LAYER_PREFILL`,
`LINEAR_BF16`, `PACKED_INDEX`, `PACKED_M32N128`, `PREFILL_ALIAS`,
`ROUTER_BATCH`, `ROUTER_MM`, `SESSION_BATCH`, `TOPK_PREFIX`, `TOPK_SHUFFLE`,
`TP_DECODE_QUEUE`, `TP_SHARED_OWNER`, `TP_SMALL_PREFILL`, `WIDE_CHUNK` and
`WIDE_PREFILL`. They are read straight from the environment, not through the group mask. The
timing variables that help find idle GPU time are main's too, notably
`DS4_METAL_ENCODER_TIMELINE`.

## Tests

| Command | What it checks | Needs the model |
| --- | --- | --- |
| `make tests/test_deepseek41_metal && ./tests/test_deepseek41_metal --v41-groups` | One check per group: each fast path against the code it replaces, mostly bit for bit, with the switch off and on. | No |
| `make test-deepseek41-preencode` | Plain steps, DSpark cycles, a snapshot and its restore with PREENCODE off and on; the same logits after every operation. | Yes, and the drafter (`DS4_TEST_MODEL`, `DS4_TEST_DSPARK`) |
| `make test-deepseek41-dspark-payload` | A payload with the drafter window restores and repeats eight DSpark cycles exactly; an old payload still loads. | Yes, and the drafter |
| `make test-deepseek41-verify-indexed` | Batched verify blocks rewound to their first, second and third rows give the next four decode steps bit for bit. | Yes, and the drafter |
| `make tests/test_snapstore && ./tests/test_snapstore` | Snapshot format 4: hashes, atomic writes, compatibility verdicts. Also runs in `make test`. | No |
| `make tests/test_sampling && ./tests/test_sampling` | The GPU top-p summary's consumer against a reference on 30,000 summaries, some corrupted. | No |

Main's own full Metal unit test (`./tests/test_deepseek41_metal` with no
argument) fails on the M3 Ultra at main itself, in its FP4 E4M3 round-trip check,
where one value lands on a rounding tie. That is why the branch's checks also run
on their own with `--v41-groups`.
