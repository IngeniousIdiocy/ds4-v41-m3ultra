# DeepSeek V4.1 on Metal: options and switches

This page lists what this branch adds to DwarfStar main's command line and
environment for DeepSeek V4.1 Flash on Metal: the options, the 97 group
switches, the image input switches and the few other variables. Everything else is main's and is
documented in main's own guides ([serving](SERVER.md),
[speculative decoding](SPECULATIVE_DECODING.md), [models](MODELS.md),
[performance](PERFORMANCE.md)).

A normal launch sets no environment variable. Every group is on by default,
and its switch exists to turn it off for comparison or bisection. `--exact`
turns off, in one go, every group that is not bit for bit.

## A serving launch

```sh
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --metal \
  --host 127.0.0.1 --port 8000 -c 1048576 --prefill-chunk 8192 --batched-session 4 \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark --mtp-exact-sampling \
  --kv-disk-dir kvcache --kv-disk-space-mb 122880 \
  --kv-cache-cold-max-tokens 1048576 --kv-cache-continued-interval-tokens 4096 \
  --default-top-p 1.0 --default-min-p 0 --default-reasoning-effort low \
  --utility-ctx 32768 --serial-reasoning
```

This is the configuration the README's server figures for several sessions
were measured with (plus `--vision`). `-c` is per session, so four sessions of up
to 1,048,576 tokens and the utility session hold about 380 GiB wired on a
512 GB Mac; lower `-c` or `--batched-session` on a smaller machine.

Run it from the repository root (or pass `--chdir`), because the engine compiles
its Metal sources from `metal/` at startup. The server has no authentication;
keep it on 127.0.0.1 unless the network is private.

## Options this branch adds

### ds4-server

| Option | Default | What it does |
| --- | --- | --- |
| `--utility-ctx N` | 0 (off) | Keeps a second session of N tokens for requests that set `"ds4_utility": true`, such as titles and summaries. They never think, never draft and never touch the disk KV cache, and they leave the main session alone. A request that does not fit in N tokens is refused. With `--batched-session`, its row joins the decode step beside the other sessions and takes prefill turns with them, so `--batched-session` can be at most 7 when it is set (the server refuses more). |
| `--serial-reasoning` | off | V4.1 decodes every reasoning span serially; DSpark drafts only answers and tool calls. In batched mode a stream inside a reasoning span is a plain row of the shared step and never holds a verify block. No effect on other models. |
| `--default-reasoning-effort E` | `high` | Effort for a request that names none: `low`, `medium`, `high` or `max`. A request's own values still win. V4.1 renders `low` as effort level 50, as DeepSeek's encoder does. |
| `--default-top-p F`, `--default-min-p F` | main's defaults | Sampler values for a request that sets none. |
| `--exact` | off | Output bit for bit DwarfStar main's serial decoding, the same as `DS4_V41_EXACT=1` (below). Also on `ds4`. |

**Batched V4.1 sessions.** With `--batched-session`, V4.1 sessions that are
ready decode together in one step, each row at the bits of that session
decoded alone. With the DSpark drafter loaded they also draft: while at most
three sessions decode together (`DS4_SERVER_MIXED_ROWS`), one of them verifies
a draft block each step, beside the other sessions' rows, rotating among the
sessions whose drafting controllers admit a block. A lone stream keeps the
one-session DSpark path. Before this release, `--dspark` with
`--batched-session` was accepted but ran every step one session at a time
without drafting.

**Prefill beside decoding sessions.** A V4.1 prompt that prefills while other
sessions decode runs in 2,048-token chunks (unless `--mixed-prefill-quantum`
is given), and after each chunk the decoders keep the model for as long as the
chunk took (`DS4_SERVER_MIXED_DECODE_SHARE`). In batched mode a cold V4.1
prompt runs as one sweep in 32,768-token syncs that parks between its chunks,
so decoding sessions take a step at every park. A parked sweep holds until
threads queued on the inference lock have had it, and a request no wider than
one chunk does not take turns with sweeps. Only the end of a sync stores a
checkpoint. `DS4_SERVER_DISABLE_PREFILL_YIELD=1` returns to the chunks and the
time share alone.

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
| `--jobs FILE` | With `--token-csv`: keeps one engine and session loaded and runs job lines as they are appended (see the job list below and the comment in `ds4_bench.c`). `PATH.jobs.csv` gets one row per job. |

The job runner's jobs. `save`, `load`, `run` and `quit` are as before; `save`
also takes `quantum=Q` (prefill in Q-token chunks, as the server does beside
generations) and `fresh=1`. Every job that sets switches takes `off=HEX` for
the first word of V4.1 switches and `off2=HEX` for the second, on top of the
process's own, plus `set=NAME=VALUE` and `unset=NAME`.

| Job | What it does |
|

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

Each group has one off switch. The switches are read once, at the first use,
into two masks with one bit per group: `ds4_gpu_v41_off()` for the first word
(bits 0 to 63) and `ds4_gpu_v41_off2()` for the second (bits 0 to 32), both in
`ds4_metal.m`, with the bit names in `ds4_gpu.h`. ds4-bench takes the first as
`off=HEX` and the second as `off2=HEX`. A switch counts as set when it is
present, non-empty and not `0`. With a switch set, that group runs the code it
replaces.

`DS4_V41_EXACT=1`, which `--exact` sets, is read with the switches. It sets
those of PREFILL_MMA, CED, IMAGE_CED, INDEX_SCORE_WIDE, TOPK, ROWS_MMA,
VERIFY_INDEXED, KEYED_SAMPLING, KEYED_GPU, FULL_PROBS, PQ_GPU and, in the
second word, ATTN_TILES (INDEX_CANDIDATES needs INDEX_SCORE_WIDE and TOPK, so
it stops too), turns off the vision encoder's flash attention, and keeps
DSpark to target-only decoding, as `--dspark-strict` does. Those are the defaults whose
output can differ from main's serial decoding. VERIFY_KERNELS and
PUBLIC_CONTROLLER differ only from main's DSpark, and with drafting off they
never run. The engine logs one line at load when it is on.

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
| 22 | `DS4_METAL_DISABLE_V41_ROWS_MMA` | Q8_0 products of 3 to 8 rows of a DSpark block (the drafter's rows and the verify block) on simdgroup matrices. Prefills never take it. | Yes with the defaults: a verify block's logits are serial decoding's bits (VERIFY_STACK and VERIFY_DECODE, below). No with either of those off: verify rows can differ from serial decoding in the last bits. |
| 23 | `DS4_METAL_DISABLE_V41_VERIFY_KERNELS` | 6 to 8 V4.1 MXFP4 verify rows through the fused decode MoE kernels (DSpark blocks only; prefills keep main's MoE kernels). | No, compared with main's DSpark: each verify row now gives the bits of that row decoded alone, not the batch kernels' bits, so DSpark's tokens can change. |
| 24 | `DS4_METAL_DISABLE_V41_PUBLIC_CONTROLLER` | Drafting by measured cost (`ds4_deepseek41_admission.h`). Off brings back main's confidence cut-off and scheduler. One Mac only. | A different policy by design: it changes where DSpark drafts, so DSpark's tokens can change. |
| 28 | `DS4_METAL_DISABLE_V41_MOE_DEDUP` | 2 to 8 MXFP4 rows reading each selected expert once (any MXFP4 model). | Yes |
| 29 | `DS4_METAL_DISABLE_V41_VERIFY_BATCH` | A session's 2 to 8 consecutive rows attending together and expanding their outputs into the streams. | Yes per row, except where ROWS_MMA applies. |
| 30 | `DS4_METAL_DISABLE_V41_FLASH_HEAD_SHARE` | Eight heads per threadgroup in multi-row flash attention (every model). | Yes |
| 41 | `DS4_METAL_DISABLE_V41_VERIFY_INDEXED` | The batched verify: a draft block attends like a short prefill. Off verifies row by row, and its tokens are serial decoding's. | Yes with VERIFY_DECODE on (the default): the block attends with the decode kernels. With VERIFY_DECODE off, verified logits are not serial decoding's bits, so greedy DSpark output can differ where two tokens are nearly tied. |

### Prefill

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 25 | `DS4_METAL_DISABLE_V41_PREFILL_INDEX` | Prefill index scores of 8 or more rows on simdgroup matrices. | Yes |
| 26 | `DS4_METAL_DISABLE_V41_PREFILL_ATTENTION` | Prefill attention staging 16 rows per barrier. | Yes |
| 27 | `DS4_METAL_DISABLE_V41_PREFILL_FOLDS` | bf16 rounding and glue folded into the producers of large prefills. | Yes |
| 31 | `DS4_METAL_DISABLE_V41_PREFILL_MATMUL` | Unpacking Q8 weights to F16 once for products of 32 or more rows on the M3 Ultra. Main's `DS4_QWEN4_Q8_PREFILL_UNPACK=0` also turns it off. | Yes |
| 32 | `DS4_METAL_DISABLE_V41_PREFILL_INDEX_BATCH` | Batch top-k from 513 visible keys, select-all for short rows, batched candidates, ratio-1 batch publishing. | Yes |
| 33 | `DS4_METAL_DISABLE_V41_SHORT_SWEEP` | One sweep for 3,072 to 8,191 tokens with the decoder suffix pruned. | Yes: with `--exact`, cold prompts of 3,072, 4,096, 6,144 and 8,000 tokens give main's logits. (The previous release's form of this group changed prefill rounding.) |
| 34 | `DS4_METAL_DISABLE_V41_ENGRAM_READAHEAD` | Reading the next sweep's Engram rows during the current sweep. | Yes |
| 42 | `DS4_METAL_DISABLE_V41_ENGRAM_READERS` | 32 concurrent Engram readers for a prefill sweep (16 when off). | Yes |
| 43 | `DS4_METAL_DISABLE_V41_SHORT_MMA` | An eight-row prefill batch's Q8_0 products on simdgroup matrices in the row kernels' order of sums, outside DSpark blocks. Needs ROW_MATMULS, one Mac and an M1 to M4 GPU. Off runs main's row kernels. | Yes: bit for bit main's row kernels (unit test on every V4.1 Q8_0 shape, and an eight-token prompt's logits). |
| 44 | `DS4_METAL_DISABLE_V41_SUFFIX_TRIM` | Inside the chunks of a 1,024- to 8,191-row sweep, the decoder layers skip the rows no later row reads, cut at a 128-row boundary with at least 256 rows kept. One Mac, not `--quality`, not SSD streaming; prompts with images only through IMAGE_TRIM. | Yes: main's logits and next 16 greedy steps on follow-ups of 512 to 8,000 tokens and cold prompts of 512 to 8,000. |
| 45 | `DS4_METAL_DISABLE_V41_PREFILL_MMA` | Prefill attention on simdgroup matrices, 32 heads of a token per threadgroup, for the rows PREFILL_ATTENTION takes. M1 to M4 GPUs. | No: the sums run in another order. Mean KL after 29 follow-ups of 1K tokens (4K to 32K): 0.0102 prose, 0.0007 code, the same next token at every point; NLL over 21 long cases +0.0004 ± 0.0036. |
| 46 | `DS4_METAL_DISABLE_V41_CED` | The decoder layers (20 to 39) replay only the last 256 rows of a sweep of 512 rows or more, as in the V4.1 report's bounded replay. Not `--quality`, not SSD streaming. | No: earlier rows are not replayed through the decoder layers. Mean KL 0.0064 prose, 0.0004 code, the same next token at every point; NLL −0.0001 ± 0.0009. With PREFILL_MMA: 0.0143 and 0.0009. |
| 47 | `DS4_METAL_DISABLE_V41_IMAGE_CED` | CED for sweeps that hold images too. | No: as CED. First-token KL against the full replay 1.2e-4 and 1.4e-5 on two screenshots, the same top token. |
| 48 | `DS4_METAL_DISABLE_V41_IMAGE_TRIM` | SUFFIX_TRIM for sweeps that hold images too, where CED does not run (`--exact`, or CED off). | Yes |
| 49 | `DS4_METAL_DISABLE_V41_MOE_SHORT_TILES` | The routed experts' five exact large-prefill MXFP4 tile variants from 9 rows instead of 2,048 (V4.1 only). | Yes |
| 50 | `DS4_METAL_DISABLE_V41_PREFILL_PIPELINE` | One-chunk layers committing their command buffers without waiting; the host still waits at the Engram layers, the decoder start and the last layer. SSD streaming and the prefill profilers keep every wait. | Yes |
| 51 | `DS4_METAL_DISABLE_V41_HC_NARROW` | The hyper-connection mix projection of a prefill chunk on one SIMD group per 8 rows. | Yes |
| 52 | `DS4_METAL_DISABLE_V41_CTX_RESIDENCY` | The engine's own allocations (context, KV, scratch) in a residency set on the queue. | Yes |
| 53 | `DS4_METAL_DISABLE_V41_MOE_HALF_X` | The compact expert pair tiles reading rows narrowed to half once per layer. | Yes |
| 54 | `DS4_METAL_DISABLE_V41_CARRY_KEEP` | A one-chunk layer keeping the previous layer's rows in the batch buffers instead of loading them from the carry. | Yes |

### The decode handoff and sampling

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 35 | `DS4_METAL_DISABLE_V41_PREENCODE` | Encoding the next token's first command buffer during the current token's wait, and the short GPU bridge kernel. | Yes |
| 36 | `DS4_METAL_DISABLE_V41_SAMPLER_GPU` | The GPU top-p summary at temperature 1, top-p 0.95, no top-k and no min-p. | Yes: the token and the random state are the CPU sampler's. |
| 55 | `DS4_METAL_DISABLE_V41_KEYED_SAMPLING` | Position-keyed Gumbel-max draws for samples of the full distribution (temperature above 0, top_p 1, no top-k, no min-p); main's sampler draws instead. | No: the same distribution, a different draw. Off with `--exact`. |
| 56 | `DS4_METAL_DISABLE_V41_KEYED_GPU` | The GPU finding the keyed draw's candidates; the CPU scores the whole row instead. | Yes, against KEYED_SAMPLING on the CPU: the same token. Off with `--exact`. |
| 57 | `DS4_METAL_DISABLE_V41_FULL_PROBS` | The one-pass softmax for an untruncated probability build; the heap and full sort run instead. | No: another order of sums, within 2.3e-6. Off with `--exact`. |
| 58 | `DS4_METAL_DISABLE_V41_PQ_GPU` | DSpark's p/q verify at top_p 1 building q, p and the draws' candidates on the GPU; the CPU builds them. | No: the same test, keyed draws. Off with `--exact`. |

### Several sessions at once: first word

These run only when two or more sessions decode in one step (`--batched-session`,
or ds4-bench's `nrun`), except where a row says otherwise. Each gives every row
the bits of that session decoded alone.

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 59 | `DS4_METAL_DISABLE_V41_HEAD_ROWS` | Cross-session rows' vocabulary head on the Q8_0 row kernel, each row at its one-row head's bits. A verify block keeps its own head. | Yes. Off, rows' logits differ from one-row decode in the last bits (the tokens stayed the same in the checks). |
| 60 | `DS4_METAL_DISABLE_V41_DECODE_ROWS` | Six to eight cross-session rows through the fused decode MoE kernels, as a DSpark verify block. Prefill keeps its kernels. | Yes. Off, six to eight rows take the grouped MoE kernels and can change tokens. |
| 61 | `DS4_METAL_DISABLE_V41_BATCH_FLUSH` | Cross-session rows committing their first command buffer after layer 0, as one-row decode does. | Yes |
| 62 | `DS4_METAL_DISABLE_V41_ROWS_REG` | Q8_0 products of two to four cross-session rows in registers: one weight pass and one reduction, each row at its one-row kernel's bits. | Yes |
| 63 | `DS4_METAL_DISABLE_V41_BATCH_ROPE` | Cross-session rows keeping one-row decode's per-step RoPE tables; the attention reduce and q_b apply the rotations. | Yes |

### Several sessions at once: second word

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 0 | `DS4_METAL_DISABLE_V41_BATCH_KV` | Cross-session rows roping, quantizing and storing their KV rows in one dispatch. | Yes |
| 1 | `DS4_METAL_DISABLE_V41_BATCH_TAIL` | Two to four rows' output and shared projections expanding straight into the hyper-connection streams, as one-row decode's do. | Yes |
| 2 | `DS4_METAL_DISABLE_V41_BATCH_LOW` | The low projection of two to four rows in registers, one weight pass. | Yes |
| 3 | `DS4_METAL_DISABLE_V41_BATCH_NARROW` | Two to four rows reading the BF16 router copy. | Yes |
| 4 | `DS4_METAL_DISABLE_V41_BATCH_HIER` | Two to four rows selecting experts with the hierarchical router. | Yes |
| 5 | `DS4_METAL_DISABLE_V41_BATCH_OVERLAP` | Two to four rows' hyper-connection mix and shared gate and up beside the routed pair kernel. | Yes |
| 6 | `DS4_METAL_DISABLE_V41_BATCH_DOWN` | With BATCH_OVERLAP, the shared down beside the routed sum, then a plain add and expand. | Yes |
| 7 | `DS4_METAL_DISABLE_V41_BATCH_WAVE` | The rows' attention stages in waves, every row's k-th dispatch in one concurrent level, each row with its own copy of the attention scratch. | Yes |
| 8 | `DS4_METAL_DISABLE_V41_BATCH_MIX` | Two to four rows' hyper-connection mix beside q_b. | Yes |
| 9 | `DS4_METAL_DISABLE_V41_BATCH_STATIC` | Two to eight rows' routed experts on one-row decode's fixed-shape kernels with a row count, without a plan dispatch. | Yes |
| 10 | `DS4_METAL_DISABLE_V41_BATCH_RBLOCK` | With BATCH_STATIC, the routed grids visiting every pair at one row block before the next (oMLX's row-block-major order). | Yes: the same reads in another order. |
| 11 | `DS4_METAL_DISABLE_V41_BATCH_EXPAND` | The rows' FFN join with its shape as literals. | Yes |
| 12 | `DS4_METAL_DISABLE_V41_BATCH_QKV` | With BATCH_MIX, q_a and kv side by side at the head of that section. | Yes |
| 13 | `DS4_METAL_DISABLE_V41_BATCH_MMA` | Five to seven rows on SHORT_MMA's exact eight-row kernels, the row count an argument. | Yes |
| 14 | `DS4_METAL_DISABLE_V41_BATCH_TAIL8` | BATCH_TAIL's and BATCH_LOW's fused tail through eight rows. | Yes |
| 15 | `DS4_METAL_DISABLE_V41_BATCH_DEFER8` | BATCH_OVERLAP and what rides on it through eight rows. | Yes |
| 16 | `DS4_METAL_DISABLE_V41_BATCH_ROUTER8` | BATCH_NARROW and BATCH_HIER through eight rows. | Yes |
| 17 | `DS4_METAL_DISABLE_V41_BATCH_QROPE` | With BATCH_MMA, q_b of five to eight rows rotating as it stores. | Yes |
| 18 | `DS4_METAL_DISABLE_V41_BATCH_HEAD8` | With SHORT_MMA, the head of five to eight rows on sixteen SIMD groups. | Yes |
| 19 | `DS4_METAL_DISABLE_V41_BATCH_LOWBF` | With BATCH_LOW, the low projection rounding to bf16 as it stores, so output-B need not as it loads. | Yes |
| 20 | `DS4_METAL_DISABLE_V41_BATCH_PE` | With PREENCODE, the rows' next first command buffer encoded during the final wait. | Yes |
| 21 | `DS4_METAL_DISABLE_V41_BATCH_NARROW_ENGRAM` | Two to eight rows reading the narrow Engram copy in one pass. | Yes |
| 22 | `DS4_METAL_DISABLE_V41_BATCH_READS` | The rows' second Engram table read once the first has joined, instead of beside it. | Yes |
| 23 | `DS4_METAL_DISABLE_V41_VERIFY_STACK` | A DSpark verify block on the cross-session rows' FFN, Engram, router and head kernels. Also for one session's blocks. | Yes |
| 24 | `DS4_METAL_DISABLE_V41_VERIFY_COPIES` | A verify block's rollback snapshot and commit copies as one concurrent level. | Yes |
| 25 | `DS4_METAL_DISABLE_V41_VERIFY_DECODE` | With VERIFY_INDEXED, a verify block attending with the decode kernels: batched keys and selection, rows with at most top_k visible keys in their own pick's order, tied index keys past 16,384 ordered by id. | Yes: verify rows are serial decoding's bits. Off, VERIFY_INDEXED's batched attention runs, which is not. |
| 26 | `DS4_METAL_DISABLE_V41_MIXED_STEP` | A verify block sharing its step with other sessions' rows (one Mac). Off, sessions draft only in steps where they decode alone; with others they are plain rows. | Yes |
| 27 | `DS4_METAL_DISABLE_V41_MIXED_CARRY` | With MIXED_STEP, a rejected block's correction as its stream's row in the next shared step, instead of a one-row step. | Yes |
| 28 | `DS4_METAL_DISABLE_V41_READ_ONCE` | Rows that picked the same expert reading its gate and up weights once, as a group in one launch: every step with a verify block, plain steps from four rows. | Yes |
| 29 | `DS4_METAL_DISABLE_V41_FP4_NO_TABLE` | V4.1's decode expert kernels turning FP4 codes into floats with shifts, without a table in threadgroup memory. One-row decode too. | Yes |
| 30 | `DS4_METAL_DISABLE_V41_LIVE_SPLITS` | Decode attention launching only the splits that have keys, and its reduce reading only those. One-row decode too. | Yes |
| 31 | `DS4_METAL_DISABLE_V41_FINAL_SPIN` | A decode step's final wait spinning on an event the head buffer signals. One-row decode too. Off while `DS4_METAL_CB_TIMES` is set. | Yes |
| 32 | `DS4_METAL_DISABLE_V41_ATTN_TILES` | Decode attention's scores and PV on simdgroup matrices, 8 heads by 64 keys a threadgroup, at every row count, one-row decode and verify blocks included. M3 and M5 GPUs, one Mac, not `--quality`; otherwise the vec kernel runs. Off with `--exact`. | No: the sums run in another order than the vec kernel's. Rows still equal the same session decoded alone. Over 21 long scoring cases, mean NLL 0.2993 against 0.3003; 16 greedy runs to their first different token, 735 steps, mean KL 0.00175, at most 0.035; 14 of the 16 reached a different token. See [METAL-V41.md](METAL-V41.md#attn_tiles-live_splits-decode-attention). |

### Two Macs

Built and unit-tested, not measured (no second Mac was available). One-Mac
decode never reaches them.

| Bit | Switch | Turns off | Exact |
| ---: | --- | --- | --- |
| 37 | `DS4_METAL_DISABLE_V41_TP_BALANCED_ROUTE` | Each Mac taking half of every token's ordered expert list. Read at load. | Not measured |
| 38 | `DS4_METAL_DISABLE_V41_TP_SHARED_SPLIT` | Splitting the shared expert's hidden width across both Macs. | Not measured |
| 39 | `DS4_METAL_DISABLE_V41_TP_GATES` | Partials written straight into the gate slots and added by the HC expansion. | Not measured |
| 40 | `DS4_METAL_DISABLE_V41_TP_PREFILL_ROWS8` | The attention output's K-slice rows in one dispatch, eight output rows per group on the M3 Ultra. | Yes (unit test) |

### Image input

Read straight from the environment, not through the group mask. They change
only requests with images; text prompts never run the encoder.

| Switch | Turns off | Exact |
| --- | --- | --- |
| `DS4_METAL_DISABLE_V41_VISION_HOST_PREP` | Image preparation split by rows across the CPU cores, and the system SHA-256 for the fingerprint. | Yes: the same patches and fingerprints. |
| `DS4_DISABLE_V41_VISION_PREP_REUSE` | The encoder reusing the patches the server built when the request arrived. | Yes |
| `DS4_DISABLE_PNG_SYSTEM_INFLATE` | The system DEFLATE decoder for PNG images (the faster checksums stay). | Yes: the same pixels. |
| `DS4_METAL_DISABLE_V41_VISION_ROUND_FOLD` | The encoder's bf16 rounding inside its kernels. | Yes |
| `DS4_METAL_DISABLE_V41_VISION_ATTN_EXACT` | The exact attention kernel's 32 keys per step; off runs the one-key kernel. | Yes |
| `DS4_METAL_DISABLE_V41_VISION_ATTN_FLASH` | The flash attention kernel on simdgroup matrices; off runs the exact kernels. `--exact` turns it off too. | No: float sums in another order. With every image default on, 176 of 216 scored questions on rendered images against 177 with `--exact`; see [METAL-V41.md](METAL-V41.md#vision_attn_flash-the-encoders-attention-on-simdgroup-matrices). |
| `DS4_METAL_DISABLE_V41_VISION_ATTN_LAZY` | Flash attention's lazy rescale. | No, as flash: rounded differently from the eager rescale. |
| `DS4_METAL_DISABLE_V41_VISION_ATTN_KT16` | Flash attention's 16-key tiles (64 when off). | No, as flash: the lazy maximum moves per 16 keys. |

The disk KV cache stores the encoder's flash setting with every image chat, so
an image checkpoint written with one setting is refused with another.

## Other variables this branch adds

| Variable | What it does |
| --- | --- |
| `DS4_V41_EXACT=1` | The exact shortcut, for every tool; see [the group switches](#the-group-switches). `--exact` sets it. |
| `DS4_V41_ROUTER_BF16_SIDECAR`, `DS4_V41_ENGRAM_FP8_SIDECAR`, `DS4_V41_INDEXER_FP8_SIDECAR` | Name a sidecar file for a family that no `--v41-sidecar` named. Diagnostics; the option is the normal way. |
| `DS4_METAL_SAMPLER_SELECTOR_SOURCE` | Loads the GPU top-p selector's Metal source (`metal/sampler_selector.metal`) from another path. If the source is missing, sampling stays on the CPU. |
| `DS4_V41_SUFFIX_ALL=1` | An opt-in experiment: builds the exact decoder suffix for any prefill sweep of 384 tokens or more, without the tile rounding. Off by default. |
| `DS4_KV_CANARY_FAIL_TEST=1` | Forces the numerics canary to fail. Read only by `ds4-server-testhooks`. |
| `DS4_SERVER_MIXED_DECODE_SHARE=P` | V4.1 only: after each prefill chunk run beside decoding sessions, the decoders keep the model for P% of the time the chunk took (0 to 1,000; default 100; 0 gives one decode step per chunk). |
| `DS4_SERVER_MIXED_ROWS=K` | The most sessions decoding together, not counting the utility row, for one of them to verify a DSpark block in the shared step (default 3). Above it, every session decodes plain rows. |
| `DS4_SERVER_DISABLE_PREFILL_YIELD=1` | Batched V4.1 prefills no longer park between chunks in 32,768-token syncs; they run in mixed chunks with the time share, as above. |
| `DS4_SNAPSTORE_ANY_DRAFTER=1` | A ds4-bench saved cache loads whatever drafter it was saved with or without, for `mbits` rows that need no drafter features. Diagnostics. |
| `DS4_SCORE_TOK=FILE` | `score_official` writes one line per scored token to FILE: case, index, target id, greedy id and the target's log-probability. |
| `DS4_BENCH_SHA_DIR` | A directory of recorded `*.sha256` files that `ds4-bench --jobs` uses for the weight identities in snapshot format 4. |

Main's own variables, which this branch extends:

| Variable | What changes |
|

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

Main's `DS4_METAL_DISABLE_V41_BATCH_ENGRAM` turns off main's batched Engram
reads for prefill (each row then reads on its own), and with them
ENGRAM_READAHEAD's reuse of the next sweep's read-ahead rows. It does not touch
batched decode rows, whose Engram group has its own switch, BATCH_NARROW_ENGRAM
(bit 21 of the second word). `TOPK_SHUFFLE` is main's; this branch's
tie-ordered verify sort (VERIFY_DECODE) is chosen before it.

## Tests

| Command | What it checks | Needs the model |
| --- | --- | --- |
| `make tests/test_deepseek41_metal && ./tests/test_deepseek41_metal --v41-groups` | One check per group: each fast path against the code it replaces, mostly bit for bit, with the switch off and on. It now also covers HEAD_ROWS and BATCH_HEAD8's rows, ROWS_REG, BATCH_ROPE with BATCH_KV and BATCH_QROPE, BATCH_TAIL with BATCH_LOW, BATCH_NARROW with BATCH_HIER, BATCH_OVERLAP, BATCH_WAVE (its rows' own attention scratch included), BATCH_STATIC in both grid orders (BATCH_RBLOCK) inside the MOE_DEDUP check, and ATTN_TILES inside the DECODE_ATTENTION check (its kernel runs, and its distance from the vec kernel is printed). | No |
| `./tests/test_deepseek41_metal --batch-wave` | Rows attending in a wave each keep their own attention scratch after a verify block regrows it. Run alone, so the process's scratch starts empty and the growth is real. | No |
| `./tests/test_deepseek41_metal --head-rows`, `--rows-reg`, `--batch-rope`, `--batch-tail`, `--batch-router`, `--batch-overlap`, `--moe-dedup` | One of the checks above on its own. | No |
| `make test-deepseek41-preencode` | Plain steps, DSpark cycles, a snapshot and its restore with PREENCODE off and on; the same logits after every operation. | Yes, and the drafter (`DS4_TEST_MODEL`, `DS4_TEST_DSPARK`) |
| `make test-deepseek41-dspark-payload` | A payload with the drafter window restores and repeats eight DSpark cycles exactly; an old payload still loads. | Yes, and the drafter |
| `make test-deepseek41-verify-indexed` | Batched verify blocks rewound to their first, second and third rows give the next four decode steps bit for bit. | Yes, and the drafter |
| `make tests/test_snapstore && ./tests/test_snapstore` | Snapshot format 4: hashes, atomic writes, compatibility verdicts. Also runs in `make test`. | No |
| `python3 tests/test_server_vision_disk_cache.py --output DIR -- -m MODEL --vision VISION` | An image chat restored from the disk KV cache after a restart, with no full re-read; the same text with another image misses. | Yes, and the vision file |
| `make tests/test_sampling && ./tests/test_sampling` | The GPU top-p summary's consumer against a reference on 30,000 summaries, some corrupted. | No |

Main's own full Metal unit test (`./tests/test_deepseek41_metal` with no
argument) fails on the M3 Ultra at main itself, in its FP4 E4M3 round-trip check,
where one value lands on a rounding tie. That is why the branch's checks also run
on their own with `--v41-groups`.
