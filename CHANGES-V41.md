# Changes for DeepSeek V4.1 Flash on Metal

This branch is DwarfStar main at `5379c1c9` plus the commits below, in order.
Each speed group is one commit with one off switch, and each group's off path
is main's code. [docs/METAL-V41.md](docs/METAL-V41.md) explains why each group
is faster, and [docs/RUNTIME-V41.md](docs/RUNTIME-V41.md) lists every switch.

## The commits

| # | Change | Switch |
| ---: | --- | --- |
| 1 | Skip hazard tracking on V4.1's weight views by default. | `DS4_METAL_MODEL_TRACKED=1` |
| 2 | Convert V4.1 Flash to MXFP4, with its DSpark drafter (`--quant mxfp4`, `--dspark-out`). | (converter) |
| 3 | Keep aligned tensor copies in the open compute encoder. | `COMPUTE_TENSOR_COPY` |
| 4 | Queue a decode step's layers into one command stream. | `DECODE_QUEUE` |
| 5 | Pick one token's experts in a single dispatch. | `ROUTER_SELECT` |
| 6 | Read the Engram rows during the first decode layers. | `ENGRAM_ASYNC` |
| 7 | Round decode activations to bf16 inside the producing kernels. | `FUSED_ROUNDING` |
| 8 | Read narrow copies of the router, Engram and indexer weights. | `NARROW_WEIGHTS` |
| 9 | Fuse the attention input projections. | `QKV_FUSION` |
| 10 | Fuse RoPE into the kernels around it. | `ROPE` |
| 11 | Join the FFN outputs into the hyper-connection streams in one dispatch. | `FFN_JOIN` |
| 12 | Fuse hyper-connection mixing into the kernels next to it. | `HC_FUSION` |
| 13 | Spread two serial reductions over more of the GPU. | `SPLIT_REDUCE` |
| 14 | Run the shared expert's gate, up and SwiGLU in one dispatch. | `SHARED_EXPERT` |
| 15 | Run the shared expert beside the routed experts. | `CONCURRENT_FFN` |
| 16 | Score long index rows one key per thread. | `INDEX_SCORE_WIDE` |
| 17 | Score the index from a bf16 copy of the keys. | `INDEX_BF16` |
| 18 | Pick the index top-k with a radix select and a shorter merge. | `TOPK` |
| 19 | Score later index rows among layer 20's blocks only. | `INDEX_CANDIDATES` |
| 20 | Stage the selected attention rows straight from the cache. | `DECODE_ATTENTION` |
| 21 | Run decode products on kernels built for V4.1's exact shapes. | `FIXED_SHAPES` |
| 22 | Find the sampler's maximum logit with NEON (every model). | `SAMPLER_SCAN` |
| 23 | Read the weights once for a few rows in Q8_0 and F16 products (every model). | `ROW_MATMULS` |
| 24 | Multiply a DSpark block's three to eight rows on simdgroup matrices. | `ROWS_MMA` |
| 25 | Read batch steps' Engram rows while their first layers run. | `ENGRAM_ASYNC` |
| 26 | Run DSpark's six- to eight-row verify blocks through the decode MoE kernels. | `VERIFY_KERNELS` |
| 27 | Let DSpark draft only where drafting beats serial decoding. | `PUBLIC_CONTROLLER` |
| 28 | Read each decode step's Engram rows on a worker per row on Apple. | `ENGRAM_ASYNC` |
| 29 | Score prefill index rows on simdgroup matrices. | `PREFILL_INDEX` |
| 30 | Stage sixteen rows per barrier in prefill attention. | `PREFILL_ATTENTION` |
| 31 | Fold the bf16 rounding and glue kernels of large prefills into their producers. | `PREFILL_FOLDS` |
| 32 | Read each selected expert once for a few MXFP4 rows (any MXFP4 model). | `MOE_DEDUP` |
| 33 | Run verify blocks and short prefills as one batch through attention and the output projections. | `VERIFY_BATCH` |
| 34 | Share key reads across eight heads in multi-row flash attention (every model). | `FLASH_HEAD_SHARE` |
| 35 | Unpack the Q8 weights of wide prefill products to F16 once on the M3 Ultra. | `PREFILL_MATMUL` |
| 36 | Pick prefill index candidates, short rows and top-k as batches. | `PREFILL_INDEX_BATCH` |
| 37 | Run prefills of 3,072 to 8,191 tokens as one layer sweep. | `SHORT_SWEEP` |
| 38 | Read the next prefill sweep's Engram rows while the current one finishes. | `ENGRAM_READAHEAD` |
| 39 | Encode the next decode token's first command buffer while the current one finishes. | `PREENCODE` |
| 40 | Summarize the logits for top-p sampling on the GPU. | `SAMPLER_GPU` |
| 41 | Check the narrow-weights switch on every use, not only at load. | `NARROW_WEIGHTS` |
| 42 | Two Macs: split each token's ordered expert list in half. | `TP_BALANCED_ROUTE` |
| 43 | Two Macs: split the shared expert instead of alternating it by layer. | `TP_SHARED_SPLIT` |
| 44 | Two Macs: write partials straight into the gate slots. | `TP_GATES` |
| 45 | Two Macs: run the attention output rows in one dispatch. | `TP_PREFILL_ROWS8` |
| 46 | ds4-server: utility session, serial reasoning and request defaults. | options |
| 47 | Bind disk KV checkpoints to the weights and numerics that wrote them (KV store v2). | (format) |
| 48 | Load the optional decode sidecars with `--v41-sidecar`. | option |
| 49 | ds4-bench: sampled decoding, per-token digests, a resident job runner and verifiable saved caches. | options |
| 50 | `gguf-tools/deepseek41_prepare.py`: from the public 1017 files to the two GGUFs. | (tool) |
| 51 | Document the server options, the disk KV cache, ds4-bench, the prepare script and the sidecars. | (docs) |
| 52 | Save the drafter window with the session, so a restored session drafts at once. | (format) |
| 53 | Encode the top-p summary only while the session samples with it. | `SAMPLER_GPU` |
| 54 | Encode the Engram layers while their rows are still being read. | `ENGRAM_ASYNC` |
| 55 | Verify DSpark blocks through the batched prefill attention. | `VERIFY_INDEXED` |
| 56 | Encode batch steps' Engram layers while their rows are still being read. | `ENGRAM_ASYNC` |
| 57 | Read a prefill sweep's Engram rows with 32 concurrent readers. | `ENGRAM_READERS` |
| 58 | Multiply eight-row prefill batches on simdgroup matrices at the row kernels' bits. | `SHORT_MMA` |

Switch names without a prefix are `DS4_METAL_DISABLE_V41_<NAME>=1`, except
`COMPUTE_TENSOR_COPY`, which is `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1`.

## Compared with the previous release

The previous release (published as `ds4-v41-m3ultra`) was built on an older
DwarfStar. This branch rebuilds the same work on current main as logical
groups. What moved:

- **DSpark runtime.** Main now has its own V4.1 DSpark: capture at the target
  layers into a 128-slot window, five drafts, verify through the batched layer
  step, commit or rewind. The branch keeps main's runtime and drops the
  release's. Only the release's admission policy is ported onto it, as
  PUBLIC_CONTROLLER, and its off switch gives main's scheduler. The release's
  tuning variables for its own runtime (`DS4_DS41_DSPARK_*`) and its fault
  injection are gone with it.
- **The drafter file.** Main's DSpark loader expects the drafter's two Markov
  tables under the names `markov_w1` and `markov_w2` and a
  `dspark.target_architecture` key. The converter now writes those names. The
  tensor data is byte for byte the release's drafter.
- **Switches.** The release had about 60 fine-grained switches. They are folded
  into the groups above, one switch per group. `DS4_METAL_DISABLE_FLASH_HEAD_SHARE`
  is now `DS4_METAL_DISABLE_V41_FLASH_HEAD_SHARE`, and the release's
  `DS4_DISABLE_V41_ENGRAM_READERS32` is now the ENGRAM_READERS group.
- **Deferred drafter upkeep** is not ported. It deferred the release runtime's
  per-step drafter upkeep through serial reasoning spans; main's runtime keeps
  no such upkeep (its per-step capture is a weighted sum, a quantize and a copy),
  so there is nothing to defer. `--serial-reasoning` is carried.
- **The disk KV cache** is the release's version 2, replacing main's version 1.
  Caches written by main's server are refused rather than loaded.
- **Logits.** Main changed the router's softplus after the release. The branch
  uses main's softplus, so its serial output is main's, bit for bit, and differs
  from the release's.

## Not carried

| Release feature | Why not |
| --- | --- |
| The release's own DSpark runtime and its tests | Main has its own V4.1 DSpark runtime; the branch keeps it. |
| `DS4_DS41_DSPARK_ADAPTIVE`, `..._MIN_SERIAL_TOKENS`, `..._P_MIN`, `..._MIN_DRAFT`, `..._LOSS_BUDGET`, `..._REASONING_SERIAL` | Tuning variables of that runtime. The ported controller runs with the release's tuned values; `--serial-reasoning` covers the reasoning case. |
| `DS4_DS41_DSPARK_FAIL` | Fault injection for that runtime. |
| `DS4_METAL_DISABLE_V41_DSPARK_DEFER` | Nothing to defer on main's runtime (above). |
| `DS4_METAL_V41_PREFILL_MMA` | An opt-in in the release, off by default and not bit-identical. |
| `DS4_V41_ENGRAM_DECODE=shift` | An experiment that did not pay. |
| `DS4_METAL_DISABLE_V41_TP_BATCH_POLL` | Dead code in the release (nothing reached it). |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (dense attention up to 1,024 compressed rows) | Main changed this on purpose: it selects the model's top-k as soon as there are more than 512 index rows, because attending to more rows changes the output. The branch keeps main's rule. It matters only between about 2K and 4K tokens. |
| The first-expert offset fix in the two-Mac batch MoE call | A fix to main's own code that can only be checked on two Macs; left for a separate fix to main. |
| `DS4_METAL_ENABLE_V41_ROWS_ATTENTION` | Its verify half is the row-by-row verify with VERIFY_INDEXED off; its short-prefill half showed no speedup and changes prefill logits. |
| The release's history, receipts and reproduction scripts, as working docs | They describe the old runtime on the old base, and this branch's commit messages and these docs replace them. They are kept unchanged as the record in [history/previous-release](history/previous-release/README.md). |
