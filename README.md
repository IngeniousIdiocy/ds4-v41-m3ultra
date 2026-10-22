## DeepSeek V4.1 Flash on Metal, built on DwarfStar main

This branch is [antirez's DwarfStar](https://github.com/antirez/ds4) (main at
`5379c1c9`) plus a set of Metal speed-ups for DeepSeek V4.1 Flash. On a 512 GB
M3 Ultra with the MXFP4 build of the October 17 checkpoint, it decodes about
twice as fast as main: **41.4 tok/s at 8K context and 40.5 at 131K**, against
main's 19.9 and 17.9. With DSpark drafting, code answers run at **59.0 tok/s**.

The speed-ups come in 43 groups. Each group does one thing, is added on top of
main in its own commits, and has one switch that turns it off and runs main's
code instead.
Serial decoding gives main's output bit for bit.

Thanks to Ivan Fioravanti and Tarjei Mandt for their public V4.1 optimizations and for the ideas this work borrowed; [THIRD_PARTY-V41.md](THIRD_PARTY-V41.md) has the details.
Everything below the line further down is DwarfStar's own README, which
applies here unchanged.

### Speed

M3 Ultra (80 GPU cores, 512 GB), the MXFP4 1017 weights, a 393,216-token
context, on a quiet machine, one process at a time. Decode is 128 greedy tokens
after restoring an 8K or 131K prompt from a saved cache, so prefill is not timed,
with the drafter loaded; each figure is the mean of two timed passes. Prefill is
a cold prompt in an empty session, without the drafter.

| | DwarfStar main | this branch |
| --- | ---: | ---: |
| Serial decode, 8K | 19.9 t/s | **41.4 t/s** |
| Serial decode, 131K | 17.9 t/s | **40.5 t/s** |
| DSpark, prose, 8K | 18.6 t/s | **40.5 t/s** |
| DSpark, code, 8K | 26.5 t/s | **59.0 t/s** |
| Cold prefill, 32K | 45.0 s | **37.9 s** |
| Cold prefill, 131K | 189.5 s | **151.6 s** |

The prose prompt is `speed-bench/promessi_sposi.txt` (an Italian novel); the
code prompt is DwarfStar's own C and Python source. The model continues each
one. Main's serial and 32K figures come from the same session as the branch's.
Main's DSpark figures and its 131K prefill come from a later quiet session;
main cannot restore the branch's saved drafter state, so its DSpark runs start
with a warm-up from main's own caches. The previous release, our
earlier build of this work on an older DwarfStar, stays available as the tag
`v41-oct-release` of this repository.
[docs/REPRODUCE-V41.md](docs/REPRODUCE-V41.md) says how each number was taken.

A token takes about 24 ms on this branch at 8K, against about 50 ms on main.
Most of the difference is not arithmetic: it is time the GPU spent waiting for
the host, kernel launches that did very little work, and bytes read that were
wider than they needed to be. [docs/METAL-V41.md](docs/METAL-V41.md) explains
each group in those terms.

### Hardware and memory

- **512 GB Mac:** tested. The MXFP4 model and its drafter stay resident. All
  numbers on this page come from an M3 Ultra with 512 GB.
- **256 GB Mac:** main's Q2 build (152 GiB of main weights) fits resident. The
  Q4 and MXFP4 builds (278 to 294 GiB of main weights) do not, and need main's
  SSD streaming. We have not measured either on 256 GB.
- **Every size:** the Engram tables (189 GiB) are never loaded into memory.
  Every mode reads their rows from the GGUF on disk as needed, so keep the model
  on a fast local SSD.

Our numbers are for MXFP4. Most groups apply to any V4.1 weights, but Q2's and
Q4's routed experts run on main's kernels, and we have not measured those
builds.

### Weights

Main's `download_model.sh` offers V4.1 Flash as Q2 and Q4, not MXFP4, and no
MXFP4 drafter for DSpark. Prepare the MXFP4 model and its drafter from the
public files in one command:

```sh
hf download antirez/deepseek-v4.1-flash-gguf --revision 9dcee51692e6540f697141fc98b097e3c96492a3 \
    --include 'DeepSeek-V4.1-Flash-1017-MXFP4-experts-*' --local-dir shards
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --include config.json tokenizer.json 'model-0004[456]-of-00048.safetensors' --local-dir dspark-source
python3 gguf-tools/deepseek41_prepare.py --shards shards --dspark-source dspark-source --out gguf
python3 gguf-tools/deepseek41_prepare.py --verify --out gguf
```

This merges the nine published MXFP4 files into
`gguf/DeepSeek-V4.1-Flash-MXFP4.gguf` (467.2 GiB) and converts the drafter into
`gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf` (7.4 GiB). Every tensor byte of the
model is the published file's; `--verify` compares every tensor with pinned
digests. It needs about 544 GB free beside `--out`, Python 3.10 or later with
numpy, tokenizers and sympy, and `cc` and `make`.

To convert from the original checkpoint instead, use main's converter with the
two options this branch adds:

```sh
make -C gguf-tools libds4quants.dylib
python3 gguf-tools/deepseek41_quantize.py \
  --hf models/DeepSeek-V4.1-Flash-1017 \
  --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --quant mxfp4 \
  --out gguf/DeepSeek-V4.1-Flash-MXFP4.gguf \
  --dspark-out gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
```

`--quant mxfp4` keeps the checkpoint's own MXFP4 expert values, with no
calibration. `--dspark-out` writes the drafter with the tensor names main's
DSpark loader expects. A dry run of this conversion gives exactly the tensor
table and size of the prepared files. Main's own `deepseek41_dspark.py` writes a
different, wider drafter (8.04 GiB), which is not the one measured here.

### Upgrading from the previous release

After `git pull`, run `make` again. Then:

- **The model GGUF:** keep it. Its pinned digests have not changed.
- **The drafter:** convert it again. This build refuses the previous release's
  drafter at load (`DSpark support model and target checkpoint families differ`,
  or `invalid V4.1 DSpark weights`), because DwarfStar's DSpark loader expects
  other tensor names and a `dspark.target_architecture` key. The tensor data is
  the same. From the three drafter shards of the original checkpoint:

  ```sh
  hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
      --include config.json tokenizer.json model.safetensors.index.json \
      'model-0004[456]-of-00048.safetensors' --local-dir dspark-source
  make -C gguf-tools libds4quants.dylib
  mv gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.old.gguf
  python3 gguf-tools/deepseek41_quantize.py --hf dspark-source \
    --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --quant mxfp4 \
    --dspark-out gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
  python3 gguf-tools/deepseek41_prepare.py --verify --out gguf
  ```

  `--verify` checks both files against the pinned digests.
- **Decode sidecars:** keep them. The scripts that write them have not changed,
  and the engine audits them at load.
- **The disk KV cache:** point `--kv-disk-dir` at a new, empty directory. Every
  old checkpoint is refused, because each is bound to the old drafter file and to
  the old numerics (main changed the router's softplus).
- **ds4-bench saved caches:** save them again. The old ones record the old
  drafter and the old numerics.
- **Options and variables:** every command-line option is kept. The release's
  DSpark tuning variables (`DS4_DS41_DSPARK_*`,
  `DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING`), `DS4_METAL_V41_PREFILL_MMA`,
  `DS4_V41_ENGRAM_DECODE`, `DS4_METAL_ENABLE_V41_ROWS_ATTENTION`,
  `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` and its fine-grained
  `DS4_METAL_DISABLE_V41_*` switches are no longer read; setting one does
  nothing. The group switches in [docs/RUNTIME-V41.md](docs/RUNTIME-V41.md)
  replace them: `DS4_METAL_DISABLE_FLASH_HEAD_SHARE` is now
  `DS4_METAL_DISABLE_V41_FLASH_HEAD_SHARE`, and
  `DS4_DISABLE_V41_ENGRAM_READERS32` is now `DS4_METAL_DISABLE_V41_ENGRAM_READERS`.

### Quick start

From the repository root:

```sh
make
./ds4 -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --ctx 32768 \
  -p "Explain Redis streams in one paragraph."
./ds4-agent -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf
```

With DSpark, add the drafter:

```sh
./ds4 -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --ctx 32768 \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark
```

As a server, with the disk KV cache and the options this branch adds:

```sh
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --metal \
  --host 127.0.0.1 --port 8000 -c 393216 --prefill-chunk 8192 \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark \
  --kv-disk-dir kvcache --kv-disk-space-mb 98304 \
  --kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096 \
  --default-top-p 0.95 --default-min-p 0 \
  --utility-ctx 32768 --serial-reasoning
```

Run the binaries from the repository root (or pass `--chdir`): the engine builds
its Metal kernels from `metal/` at startup. No environment variable is needed.
[docs/RUNTIME-V41.md](docs/RUNTIME-V41.md) lists every option and switch.

### The groups

Each group's switch is `DS4_METAL_DISABLE_V41_<NAME>=1` unless the row says
otherwise. "Decode" means one token at a time; "DSpark" means the six-row step
that checks a draft; "TP" means two Macs.

| Group | What it speeds up |
| --- | --- |
| `DS4_METAL_MODEL_TRACKED=1` | Decode, prefill: skips Metal's hazard tracking on the read-only weights. |
| `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1` | DSpark: small copies stay in the open compute encoder instead of a blit (every model). |
| DECODE_QUEUE | Decode: a step's layers queue into one command stream, so the GPU starts while the host still encodes. |
| ROUTER_SELECT | Decode, DSpark: the six experts are picked in one dispatch. |
| ENGRAM_ASYNC | Decode, DSpark: the Engram disk reads overlap the first layers instead of stalling the step. |
| FUSED_ROUNDING | Decode: bf16 rounding inside the kernels that produce the values. |
| NARROW_WEIGHTS | Decode: reads BF16 and FP8 copies of three weight families, half the bytes. |
| QKV_FUSION | Decode: the attention input projections and their norms in fewer passes. |
| ROPE | Decode: RoPE computed once per step and fused into its neighbours. |
| FFN_JOIN | Decode: the FFN outputs join the residual streams in one dispatch. |
| HC_FUSION | Decode: hyper-connection mixing fused into the kernels next to it. |
| SPLIT_REDUCE | Decode: two single-core reductions spread over the GPU. |
| SHARED_EXPERT | Decode, DSpark: the shared expert's gate, up and SwiGLU in one dispatch. |
| CONCURRENT_FFN | Decode: the shared expert runs beside the routed experts. |
| INDEX_SCORE_WIDE | Decode at depth: long index rows scored one key per thread (shared with GLM). |
| INDEX_BF16 | Decode at depth: index scoring reads a bf16 copy of the keys, half the bytes. |
| TOPK | Decode at depth: the top 512 by radix select across the GPU. |
| INDEX_CANDIDATES | Decode at depth: later layers score only the blocks layer 20 kept. |
| DECODE_ATTENTION | Decode: attention stages its rows straight from the cache, no copy. |
| FIXED_SHAPES | Decode: kernels built for V4.1's exact sizes. |
| SAMPLER_SCAN | Sampled decode: the sampler's first pass in NEON (every model). |
| ROW_MATMULS | DSpark: a few rows share one pass over the weights (every model). |
| ROWS_MMA | DSpark: a draft or verify block's three to eight rows on the matrix units. |
| VERIFY_KERNELS | DSpark: six to eight verify rows through the fused decode MoE kernels. |
| PUBLIC_CONTROLLER | DSpark: drafts only where drafting beats serial decoding. |
| MOE_DEDUP | DSpark: each expert read once for the rows that picked it (any MXFP4 model). |
| VERIFY_BATCH | DSpark, short prefill: rows attend and project as one batch. |
| FLASH_HEAD_SHARE | DSpark: eight attention heads share their key reads (every model). |
| VERIFY_INDEXED | DSpark: a draft block attends like a short prefill (the batched verify). |
| PREFILL_INDEX | Prefill: index scores on the matrix units, no barriers. |
| PREFILL_ATTENTION | Prefill: attention stages sixteen rows per barrier. |
| PREFILL_FOLDS | Prefill: rounding and glue kernels folded into their producers. |
| PREFILL_MATMUL | Prefill: Q8 weights unpacked to F16 once on the M3 Ultra. |
| PREFILL_INDEX_BATCH | Prefill: index candidates and top-k picked as batches. |
| SHORT_SWEEP | Prefill: 3,072 to 8,191 tokens in one sweep, with less decoder work. |
| ENGRAM_READAHEAD | Prefill: the next sweep's Engram reads start during this one. |
| ENGRAM_READERS | Prefill: 32 concurrent Engram readers instead of 16. |
| PREENCODE | Decode: the next token's first command buffer is encoded during this token's wait. |
| SAMPLER_GPU | Sampled decode: the top-p summary is computed on the GPU. |
| TP_BALANCED_ROUTE | TP: each Mac takes half of every token's experts. |
| TP_SHARED_SPLIT | TP: the shared expert is split across both Macs. |
| TP_GATES | TP: partials go straight into the exchange slots. |
| TP_PREFILL_ROWS8 | TP prefill: the attention output rows in one dispatch. |

The two-Mac groups are built and unit-tested but not measured: no second Mac was
available.

### Exactness

Every group has an off switch. Serial decoding gives DwarfStar main's output
bit for bit. DSpark's verify groups are very accurate, but where two tokens are
nearly tied their output can differ slightly from strict decoding.

With every group on, serial decoding gives DwarfStar main's logits bit for bit:
every one of 128 greedy steps from main's own saved caches at 8K and 131K prose,
at every step of the progression below, and the logits after a cold 32K
prefill. Short prompts are exact too: after prompts of three to eight tokens the
logits and the next 32 greedy tokens are main's. Strict DSpark gives the serial
tokens.

One switch exists only for exactness. An eight-token prompt is prefilled as one
batch of eight rows, and SHORT_MMA runs that batch on the matrix units in main's
order of sums, so it gives main's bits at the speed of the faster kernels it
replaced. `DS4_METAL_DISABLE_V41_SHORT_MMA=1` runs main's own kernels instead,
with the same output.

DSpark's default mode is different, on main as here. It keeps the state of the
batched verify, and the batched verify is not bit for bit equal to decoding one
token at a time, so greedy DSpark output can differ from serial output where two
tokens are nearly tied. This branch verifies in a batch by default because it is
faster; `DS4_METAL_DISABLE_V41_VERIFY_INDEXED=1` verifies row by row instead,
which gives serial decoding's tokens.

The groups that are not bit for bit:

- **VERIFY_INDEXED** (the batched verify, above).
- **ROWS_MMA**: three to eight rows of a draft or verify block on the matrix
  units add in a different order, so verify logits can differ in the last bits.
  It runs only inside DSpark; prefills never take it.
- **VERIFY_KERNELS**: verify rows now carry the bits of each row decoded alone,
  instead of main's batch kernels' bits, so DSpark's tokens can differ from
  main's DSpark.
- **PUBLIC_CONTROLLER**: a different drafting policy by design; it changes
  where DSpark drafts.
- **DECODE_QUEUE with DSpark** and **SHORT_SWEEP**: in the previous release,
  turning these off changed DSpark's verify bits and the prefill rounding of a
  3,072- to 8,191-token prompt, while the tokens stayed the same.
- **INDEX_SCORE_WIDE**: the wide scorer adds the index heads in a different
  order, so a key near the top-512 cut-off could change. In every check, the
  selections and logits stayed the same.

[docs/RUNTIME-V41.md](docs/RUNTIME-V41.md) gives the exactness of every switch.

### Also carried from the previous release

- **ds4-server:** `--utility-ctx N` keeps a small second session for
  `"ds4_utility": true` requests (titles, summaries) that never disturb the main
  session. `--serial-reasoning` decodes reasoning serially and drafts only
  answers and tool calls. `--default-top-p`, `--default-min-p` and
  `--default-reasoning-effort` set request defaults. See
  [serving](docs/SERVER.md#request-defaults-utility-requests-and-serial-reasoning).
- **The disk KV cache, version 2:** a checkpoint is bound to the model and
  drafter files that wrote it, to a hash of its payload and to a numerics canary
  run at startup, so a checkpoint from other weights, other kernels or a damaged
  file is refused instead of loaded. A request that extends a restored prefix
  saves the deeper checkpoint. With the drafter loaded, a saved session also
  carries the drafter's window, so a restored session drafts at once. Caches
  written by main's version 1 store are refused.
- **Decode sidecars:** `--v41-sidecar FILE` loads DeepSeek's released BF16
  router, FP8 Engram or FP8 indexer values in place of the copies built at load.
  See [the gguf-tools guide](gguf-tools/README.md#v41-decode-sidecars).
- **ds4-bench:** sampled decoding (`--temp`, `--top-p`, `--top-k`, `--min-p`,
  `--seed`), per-token timings and digests (`--token-csv`), a resident job
  runner (`--jobs`) and saved caches that are checked on every load. See
  [performance](docs/PERFORMANCE.md#sampled-runs-digests-and-the-job-runner).

### Not included

- **The previous release's own DSpark runtime.** Main now has its own V4.1
  DSpark, and this branch keeps it. Only the release's drafting policy is
  ported, as PUBLIC_CONTROLLER. Its tuning variables, fault injection and
  deferred drafter upkeep went with the old runtime.
- **The prefill attention on matrix units** (`DS4_METAL_V41_PREFILL_MMA`) and
  **the shift decode of Engram codes**: experiments that were off by default
  and did not earn a place.
- **Dense attention up to 1,024 compressed rows.** Main deliberately selects the
  model's top 512 as soon as there are more than 512 index rows, because
  attending to more changes the output. This branch keeps main's rule.
- **A two-Mac fix to main's own batch MoE call**, which can only be checked on
  two Macs; it is left for a separate fix to main.
- **The release's rows attention for short prefills**: no speedup was measured,
  and it changes prefill logits.

[CHANGES-V41.md](CHANGES-V41.md) lists every commit and everything that moved
since the previous release.

### Progression

Each step turns on more groups, in commit order, on top of the step before. All
steps were measured as in the speed table, in one quiet session (prefill step 9
in a short second session right after it).
[docs/METAL-V41.md](docs/METAL-V41.md) explains each step.

Decode, in tokens per second:

| Step | Groups added | Serial 8K | Serial 131K | DSpark code 8K | DSpark prose 8K |
| --- | --- | ---: | ---: | ---: | ---: |
| main | | 19.9 | 17.9 | 26.5 | 18.6 |
| 1 | MODEL_TRACKED | 20.2 | 18.2 | 30.1 | 19.2 |
| 2 | COMPUTE_TENSOR_COPY | 20.6 | 18.6 | 31.3 | 19.6 |
| 3 | DECODE_QUEUE | 27.6 | 24.3 | 32.0 | 23.1 |
| 4 | ROUTER_SELECT | 29.2 | 25.5 | 32.6 | 23.9 |
| 5 | ENGRAM_ASYNC | 29.2 | 25.1 | 39.8 | 25.4 |
| 6 | FUSED_ROUNDING | 30.6 | 26.5 | 39.9 | 26.3 |
| 7 | NARROW_WEIGHTS | 31.5 | 27.2 | 39.9 | 26.6 |
| 8 | QKV_FUSION, ROPE, FFN_JOIN | 32.8 | 28.1 | 40.6 | 27.3 |
| 9 | HC_FUSION | 36.2 | 30.6 | 41.1 | 28.9 |
| 10 | SPLIT_REDUCE, SHARED_EXPERT, CONCURRENT_FFN | 37.5 | 31.5 | 41.6 | 29.5 |
| 11 | INDEX_SCORE_WIDE, INDEX_BF16 | 37.6 | 34.1 | 41.8 | 29.6 |
| 12 | TOPK | 37.9 | 36.6 | 42.2 | 29.9 |
| 13 | INDEX_CANDIDATES, DECODE_ATTENTION | 38.9 | 38.1 | 43.3 | 30.4 |
| 14 | FIXED_SHAPES | 41.1 | 40.3 | 43.3 | 31.2 |
| 15 | SAMPLER_SCAN, ROW_MATMULS | 41.2 | 40.3 | 48.2 | 32.4 |
| 16 | ROWS_MMA, VERIFY_KERNELS | 41.2 | 40.3 | 53.9 | 33.7 |
| 17 | PUBLIC_CONTROLLER | 41.2 | 40.3 | 52.5 | 40.3 |
| 18 | PREFILL_INDEX, PREFILL_ATTENTION, PREFILL_FOLDS, MOE_DEDUP | 41.2 | 40.3 | 52.6 | 40.3 |
| 19 | VERIFY_BATCH, FLASH_HEAD_SHARE | 41.2 | 40.3 | 54.0 | 40.2 |
| 20 | PREFILL_MATMUL, PREFILL_INDEX_BATCH, SHORT_SWEEP, ENGRAM_READAHEAD, PREENCODE | 41.4 | 40.5 | 54.0 | 40.5 |
| 21, all on | SAMPLER_GPU, the four TP groups, VERIFY_INDEXED, ENGRAM_READERS | 41.4 | 40.5 | 59.0 | 40.5 |

Cold prefill of 32K tokens, in seconds:

| Step | Groups added | 32K |
| --- | --- | ---: |
| main | | 45.0 |
| all off | this branch with every group off | 45.0 |
| 1 | the decode and DSpark groups through PUBLIC_CONTROLLER | 44.9 |
| 2 | PREFILL_INDEX | 43.5 |
| 3 | PREFILL_ATTENTION | 42.2 |
| 4 | PREFILL_FOLDS | 40.9 |
| 5 | MOE_DEDUP, VERIFY_BATCH, FLASH_HEAD_SHARE | 40.9 |
| 6 | PREFILL_MATMUL | 40.7 |
| 7 | PREFILL_INDEX_BATCH | 40.4 |
| 8 | SHORT_SWEEP, ENGRAM_READAHEAD and the decode groups PREENCODE through VERIFY_INDEXED | 40.4 |
| 9, all on | ENGRAM_READERS | 37.9 |

A cold 131K prefill takes 189.5 s on main (a later quiet session), 189.4 s with
every group off and 151.6 s with every group on.

After the progression session, ROWS_MMA and VERIFY_KERNELS were limited to
DSpark's rows; before, they also took an eight-token prefill, which was fast but
not main's bits. That change and the SHORT_MMA exactness switch (see Exactness)
were measured against the previous head with every group on, in alternating
processes: an eight-token prefill takes 0.3 s, as before, and now gives main's
logits; a cold 32K prefill takes 37.8 s, under 0.1 s from before; DSpark code at
8K runs at 59.2 t/s, within 0.1 t/s of before and within run-to-run noise.

Groups that share a row were measured together: they have little or no effect on
that row's measurements alone, and the TP groups and SAMPLER_GPU cannot change
greedy decoding on one Mac. Every serial step gave main's digest of 128 steps of
tokens and full logits at both depths, and every prefill step gave main's logits.
Main's DSpark row comes from main's own caches with a warm-up, in a later quiet
session, while the steps run from caches that carry the drafter window
with no warm-up, so the jump from main to step 1 is partly a change of method.

### Tests

```sh
make tests/test_deepseek41_metal
./tests/test_deepseek41_metal --v41-groups      # one check per group, no model
make test-deepseek41-verify-indexed DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-MXFP4.gguf \
  DS4_TEST_DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
```

[docs/RUNTIME-V41.md](docs/RUNTIME-V41.md#tests) lists the other V4.1 tests.

### V4.1 documents

- [docs/METAL-V41.md](docs/METAL-V41.md): what each group does and why it is faster.
- [docs/RUNTIME-V41.md](docs/RUNTIME-V41.md): every option and switch.
- [docs/REPRODUCE-V41.md](docs/REPRODUCE-V41.md): how the numbers were measured.
- [CHANGES-V41.md](CHANGES-V41.md): the commits, and what changed since the previous release.
- [THIRD_PARTY-V41.md](THIRD_PARTY-V41.md): where the pieces came from.
- [.claude/skills](.claude/skills): three skills for Claude Code and Codex users optimizing Metal inference: decode, prefill and speculative decoding.
- [history/previous-release](history/previous-release/README.md): the previous release's experiments, receipts and release notes, kept unchanged as the record.

---

<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 64 GB (for Qwen Flash next) and 96 GB or more (for DeepSeek and GLM). Smaller machines can use SSD streaming. SSD streaming is also needed in order to run very large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/IngeniousIdiocy/ds4-v41-m3ultra.git
cd ds4-v41-m3ultra
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
