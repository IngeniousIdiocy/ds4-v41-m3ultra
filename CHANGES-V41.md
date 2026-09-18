> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

# Optimizations on the V4.1 Flash M3 Ultra branch

> **Latest update:** [September 16 decode consolidation](docs/DECODE-CONSOLIDATION-20260916.md).
> The measurements and release identity below describe the earlier release stage.

## September 16 update

The [current release log](docs/RELEASE-V41-20260916.md) covers all validated gains since launch, enabled by default. The [chronological experiment records](docs/experiments/v41-20260915/README.md) include adopted, rejected, invalid and unrun experiments. Rejected implementations were culled from the source; their evidence is retained.

## Original launch record

The sections below describe the original release and preserve its measurements.

Adopted changes only, one line each. Every entry is on by default on the release
head `03eb931` unless the line says otherwise, and every entry keeps an `=0` kill
switch so any arm can be reproduced from one binary. Rows M1-M16 were measured on
`b977a33`, where the MTP decode levers were still opt-in and named explicitly by
every arm; M17-M20 on `f9dc5f9`; M21-M23 and K1-K5 on the release head itself.
Mechanisms that were built, measured and rejected are in section 4 with the number
that killed them; their kernels and levers are **not** in this branch.

The A/B gate throughout is an 8,192-token restored prefix and 512 greedy tokens on
a quiet machine, interleaved off/on in one resident server, with decoded output
compared byte for byte against the upstream `bd66c40` reference for that fixture.
Decode utilisation is `13.77 GB / (ms per token x 705 GB/s)`.

Environment names are the `DS4_DS41_*` / `DS4_METAL_*` forms; the short names in
parentheses are the `/debug/levers` keys.

---

## 1. Decode

| # | change | what changed | result |
|---|---|---|---|
| L1 | One command buffer for all layers (`DS4_DS41_QUEUE_LAYERS`) | Per-layer drains were gated to TP only; the single-host path now queues layers, and layer 14 gets its own Engram input buffer so nothing drains mid-stack | 41 -> 2 command buffers per token; +26 % (16.7 -> 21.1 t/s on a contended machine) |
| L2 | Asynchronous Engram row fetch (`DS4_DS41_ENGRAM_ASYNC`) | 48 synchronous 264-byte preads per token, 5-8 ms, moved to a worker pool issuing both modules at the top of the step into double-buffered results; layer 1 waits ~0.4 ms, layer 14 is hidden | +12.8 % (21.1 -> 23.8); L1+L2 together 16.6 -> 23.8 on a quiet machine |
| D1 | BF16 re-rounding folded into producers | Weighted RMS norm, HC weighted-sum and HC expand-split write BF16-rounded outputs directly instead of via ~770 tiny re-round dispatches | -327 dispatches/token; 23.85 -> 24.42 t/s |
| W1 | Grouped Q4_K expert tables, `group6` default | Upstream shipped four env-gated groupings for 384 experts and none had been benchmarked here; routed through the lever table and A/B'd in one server | group6 +0.47 t/s (24.69 -> 25.16); group8 +0.30; group24 0 |
| W2a | Wide Q4_K quant loads on routed gate/up and down (`q4_wide`) | Each Q4_K row's eight quant `uint16`s read as two `ushort4` loads, same lanes and same order, behind a new `FC_q4k_wide` function constant | +0.04 t/s, 3/3 pairs positive, not disjoint; kept on |
| W2 | 384-expert router chain in one dispatch (`router_fused`) | New `kernel_dsv41_router_select_one_384`: softplus, sqrt, bias, the same 512-lane bitonic network and the weight tail, 9 dispatches -> 1 | **+1.16 t/s (25.07 -> 26.23)**, disjoint, byte-identical; 48.1 % -> 51.2 % of the wall |
| R9 | Shared expert gate+up+SwiGLU in one dispatch (`shared_swiglu`) | `kernel_dsv41_shared_mid_swiglu_q8_0`: V4's fused geometry with V4.1's four rounding boundaries kept in the owning lane, 6 dispatches -> 1 | **+0.46 t/s (26.14 -> 26.59)** |
| R1a | HC pre-sublayer in two dispatches (`hc_norm_mix`, `hc_tail`) | V4's `hc_rms_norm_mix` host gate relaxed from `n == 16384` to the kernel's real invariants, plus a new Sinkhorn-collapse-norm kernel carrying the one-lane tail | adopted, byte-identical |
| R1b | Block BF16 round folded into the HC expansion (`hc_expand_fold`) | `kernel_dsv41_hc_round_expand4` rounds the block row on load, 2 dispatches -> 1 at both expansions | +0.09 t/s, -80 dispatches/token, disjoint |
| R2 | BF16 re-round on the matvec store (`mv_round`) | `helper_mv_reduce_and_write` gains a `ROUND` template parameter (a function constant would break the no-constants pipelines sharing the helper); three `_bf16` twins | adopted; `dsv41_bf16_linear` was the largest remaining glue family at 259.5 dispatches/token |
| R8' | The three glue dispatches R2 could not reach (`producer_round`, `ffn_add_fold`) | `out_a` low takes R2's template parameter; the heads round moves onto the flash-attention reduce kernel with an arm/consume handshake | adopted |
| R3a | Selected KV rows gathered as F16 (`kv_stage_f16`) | `kernel_get_rows_f32_f16` does the gather and the conversion the staging copy did a pass later, which makes the F16 KV stage kernel eligible | adopted; removed 118 conversion dispatches/token |
| D | Decode split into three submitted command buffers (`decode_chunks`) | Chunks cut after layers 1 and 13, head folded into the last, Engram joined before each owning commit | adopted against 2.31 ms of host encode and 3.01 ms of inter-CB gap fully exposed |
| A | HC prediction moved into the projection stream (`hc_stream`) | 12 HC mixer tasks at the head of the `q_b` and shared-GU grids, relaxed-atomic publication with device fences, Sinkhorn run by the twelfth completed group | adopted; 80 HC dispatches/token had been sitting idle between the collapse and the large grids |
| Flat | `q_a` and KV in one 896-threadgroup grid (`qa_kv_flat`) | 640 `q_a` row-pair tasks plus 256 KV tasks, same Q8 body, same BF16 store | +0.05 t/s (neutral), -40 dispatches/token; retained for the dispatch reduction, not counted as a speed lever |
| C2 | Selected-KV gather folded into the contiguous staging dispatch (`stage_gather`) | The gather and the staging copy were the last decode families at 111 and 201 GB/s, 923.5 us/token against a 106 us floor | part of the decode pass-2 set below |
| C1 | Wide selected-KV gather (`gather_wide`, counted, default 2) | `kernel_dsv41_gather_kv_f32_f16_w<EPT>`: `packed_float4` loads and `packed_half4` stores, against one 4 B load and one 2 B store per thread | part of the decode pass-2 set below |
| D2 | HC-expansion threadgroup width (`hc_expand_nth`, counted, default 64) | Host-side width only; the element map, the five-term accumulate and the store order are untouched. The kernel is elementwise and had been running 5,120 threads on 20 threadgroups of an 80-core GPU | part of the decode pass-2 set below |
| — | **Decode pass 2 production** | C2 + C1 + D2 together, control = the same binary with all three off | **8k 30.849 -> 31.360 t/s (60.3 % -> 61.3 %); 32k 29.952 -> 30.433 t/s** — byte-identical and disjoint at both depths |
| X1 | Skip excluded block scores in the serial and verifier rows (`031c8ee`) | Layers above 20 score only the compressed rows their block mask admitted, instead of scoring all of them and filtering afterwards | part of the depth set below |
| X2 | Compact admitted index blocks per verifier row (`057db5c`) | The admitted blocks are compacted once per row rather than re-derived per scoring pass | part of the depth set below |
| X3 | Bounded token selection with an exact GPU fallback (`44825cd`) | A bounded radix top-k replaces the general indexer top-k when the width is exactly 512, with the original kernel as the fallback whenever the bound is not met | part of the depth set below |
| X | **Selection chain default-on (`9f432d8`)** | `index_compact_score` and `index_topk_radix` become defaults; `DS4_DS41_INDEX_COMPACT_SCORE=0` and `DS4_DS41_INDEX_TOPK_RADIX=0` restore the original path. The masked-scorer early exit stays opt-in: it is worth 2.89 ms/token at 300k on its own, but the compact scorer supersedes it and the pair measures the same either way | **300k serial 43.29 -> 35.35 ms/token (23.10 -> 28.29 t/s); 131k 36.85 -> 33.40; 62k 33.62 -> 32.48; 8k inside a 0.29 ms band.** Each mechanism alone takes more than 0.5 ms/token off a 300k step — the compact scorer 4.03 ms, the radix selector 3.96, together 7.92. Byte-identical to the upstream reference on every arm at 8k, 62k, 131k and 300k, serial and six-row |

### Why depth costs what it costs

V4.1's compressed sparse attention attends a fixed 512 selected compressed rows
plus a 128-row raw window per token, so the attention itself is flat with
context — and it is: across a 36x context range, 8,192 to 300,000 tokens, flash
attention moves 1.813 -> 1.810 ms/token at a constant 40 dispatches, and the
gather, the KV stage, the routed gate-up/down, the shared and Q8 projections, the
HC/norm glue and Engram are all flat to within 1 %. What grows is the chain that
*chooses* which 512 rows to attend to.

Measured with the paged kernel ledger on the original selection path, before X1
to X3: the instrumented step grew 12.198 ms from 8k to 300k, of which indexer
scoring was 7.162 ms, the top-k sort/merge chain 4.362 ms and candidate/mask
construction 0.458 ms — **11.982 ms, 98.2 % of the whole increase**, with every
other family moving by less than 0.03 ms. The same arithmetic gives 93.8 % at
131k and 92.2 % at 62k. That attribution is what X1 to X3 were built against, and
it is why they buy nothing at 8k and 7.92 ms/token at 300k.

Two caveats on the instrument. Ledger mode 2 disables encoder batching and the
parallel-FFN overlap, so its base step reads 33.065 ms against production's
31.760 at 8k; what transfers is the relative structure, not the absolute
milliseconds. It tracks the depth penalty it is being used to attribute to within
about 0.6 ms everywhere: production's serial step grew +1.19 / +1.78 / +4.97 /
+11.63 ms with depth against the instrumented step's +1.11 / +2.09 / +5.44 /
+12.20. Every arm certified 100.000 % coverage; the sampler refuses to print a
timing field otherwise.

## 2. Prefill

| # | change | what changed | result |
|---|---|---|---|
| P2 | Q4_K routed live-tile cull (`routed_tail_cull`) | `CULL_TAIL_SIMDGROUPS` existed but was instantiated for MXFP4 only; V4.1's experts are Q4_K. Three new instantiations selected by the lever in the three routed pipeline choosers | adopted; +1.40 % on the 16k cold prefill and 872.8 -> 737.0 ms (-15.6 %) on 111-token agent-turn appends |
| P3 | Engram batch delivery concurrency 16 -> 32 (`engram_readers`, counted, default 32) | Phase 1 measured 5,625 ms of exposed Engram wait at 62k, 5,519 of it on layer 1, arriving as 0.8-1.1 s at *each* 8,192-row batch — sustained delivery falling behind the consumer, not lead time | adopted |
| P4 | Sixteen-row K/V staging in the prefill attention core (`prefill_attn_rb16`) | Batch prefill dispatches the `heads8_rb16` kernel that was already the decode path's production kernel, plus its 16 KiB threadgroup allocation | adopted |
| P7 | Six batch-glue folds (`prefill_f16_rows2`, `prefill_embed_init`, `prefill_hc_sum_round`, `prefill_hc_norm_round`, `prefill_hc_expand_round`, `prefill_ffn_add_round`) | Rounding and init work folded into the producing dispatches across the batch path | all six adopted as production defaults; the raw 129,280-float frontier row is byte-identical to the wave-2 configuration at 16,384 and 32,768 over two interleaved reps each |
| P9 | Register-lean selected-id form of the prefill attention core (`prefill_attn_lean_rows`) | The rb16 kernel held sixteen selected row ids in a `uint rows[16]` private array — sixteen live registers in a kernel measured to be register-bound. A sixteen-bit mask replaces the array and the id is re-read from `row_topk` in the staging loop | adopted; the mask is identical by construction |
| P7' | Bounded cross-sweep Engram prefetch (`prefill_engram_xsweep`) | Exposed Engram wait had *grown* 926 -> 1,646 ms because the consumer got faster while the readers sat at the SSD knee; the later-sweep component is the recoverable one | adopted; exposed Engram wait at 62k is now 0.86 s, first-sweep only |
| — | **Prefill production, cumulative** | all of the above | cold 62,000 tokens **697.24 -> 812.36 t/s**; **54.2 %** of the 27.4 TFLOPS ideal, frontier logits byte-identical at every freeze |

## 3. DSpark / MTP verify

The k-row verify step is the decode path when drafting. `k=1` is the non-drafting
path and is untouched by everything here.

| # | change | what changed | result |
|---|---|---|---|
| M5 | Capture undo, truthful snapshot metadata, seed generation (`mtp_state_fix`) | The verify context lazily saves `rows * 3 * 5120 * 4` B of target-hidden slots before writing them, and snapshot metadata stops advertising `min(pos,128)` valid rows unconditionally; position alone is not cache identity | adopted; prerequisite for every arm below |
| M10 | Capture completed from the decoder warmup residuals (`mtp_capture_warmup`) | The prefill decoder suffix computes layer 39 for one position only, so the capture ring held one valid row, not 128. At layer 39 the decoder warmup already restores the other 127 | adopted |
| E1 | Six-row Q8 projections as three shared-weight pairs (`mtp_q8_pair6`) | A pair grid dimension over the two-row helper: three explicit weight streams rather than one six-row stream, each grid-y pair offsetting input and output | see E1+E2 |
| E2 | Six-row Engram overlap (`mtp_engram_rows6`) | Six histories copied and advanced before fetch; twelve fetch objects own private id copies | **E1+E2: verify 176.98 -> 111.60 ms (-36.9 %), ledger 212.8 -> 337.4 GB/s (+59 %)** — the largest kernel gain of the campaign, ~65 ms |
| F2 | Independent verify chunk submission (`mtp_async_chunks`) | The verify's structural chunk cuts are submitted independently, so encode of the next chunk overlaps execution of the previous | adopted |
| H1 | Six-row Q8 projections on one weight stream, family mask 2 (`mtp_q8_stream6`, `mtp_q8_stream6_mask=2`) | The full mask costs +5.96 ms on `code` and +6.43 on `agent`: six of seven weight families lose to register pressure and only `q_b` pays, at -1.63 ms. Mask 2 is `q_b` alone | adopted at mask 2; per-family deltas are additive (sum +5.57 against a measured +5.72), which is what made the mask defensible |
| — | Prefill folds relaxed at six rows (`prefill_decoder_suffix`, `verify_wide_prefill`) | Five `count > 8` prefill folds relaxed so the six-row verify reaches them | adopted |
| M8 | EOS symmetry (`dspark_excl_eos`) | The serial control calls `argmax_excluding(eos)` while the DSpark cycle took a plain argmax and broke its loop on EOS. The cycle's row argmax now skips the EOS id and the generate loop no longer breaks on it | adopted; without it the speculative and serial paths are not comparing the same thing |
| M16 | Width and EOS boundaries made honest | `dspark_verify_rows` accepts 0 (full block) or 2..6 and refuses 1 and anything above 6, keeping the previous value; the cycle refuses rather than silently running the full block | adopted; see the note below |
| M17 | Windowed cost-feedback admission control (`dspark_adaptive`, **default 1**) | GLM-5.3-Flash's DFlash2 controller, ported: propose the trained block at every eligible position, judge three attempts against the median of the last nine measured serial tokens, back a losing window off for 16/32/64/128 consumed serial tokens | **1.22-1.43x serial over five real chat lengths; 1.104x on a held-out real-agent screen; 0.959x on a Spec-Bench subset where a fixed block reads 0.741x.** See [DSPARK-V41.md](docs/DSPARK-V41.md) section 4 |
| M18 | The five winning MTP decode levers become defaults | `mtp_engram_rows6`, `mtp_async_chunks`, `mtp_q8_pair6`, `mtp_q8_stream6` default 1 and `mtp_q8_stream6_mask` defaults 2 — the values every winning arm set explicitly | a fresh boot ran a 188 ms six-row cycle against the bench's 121 ms, so a fixed block lost to serial at every depth. Fixed six at an 8,745-token prompt went 26.46 -> 41.10 t/s once the levers were the defaults. `=0` is the kill switch for each |
| M19 | Exact speculative sampling at temperature above zero | accept the drafted token with probability `p(x)` under the target's filtered distribution, sample the residual on rejection, sample the bonus row outright; the opportunistic mode is deleted for V4.1 | makes `--dspark` legal at the published temperature 1.0 / `top_p` 0.95 settings; 0.893-1.257x serial over five lengths. Certified by chi-square against the target's filtered distribution, driving the shipped function. See [DSPARK-V41.md](docs/DSPARK-V41.md) section 5 |
| M20 | Published sampling defaults on the command line | `--default-top-p`, `--default-min-p` apply to requests that do not name the knob; a request that names it wins | a deployment's benchmark settings live in the deployment rather than in every client |
| M21 | Confidence admission (`35ed0f0`, defaults `p_min` 0.75 / `min_draft` 3) | GLM's `dflash_adaptive_prefix`, ported: the admitted prefix is the longest run from the anchor at or above `p_min` on the post-Markov proposal row, already on the host, and a run shorter than `min_draft` declines the cycle before the verify. The admitted prefix gates the cycle, never shortens it | 8k `code` controller 37.32 t/s against fixed six's 36.23 — it now beats six on the fixture it used to trail; `agent2` 32.65 against serial's 31.37 (+4.1 %); controller against fixed six at 8,745 / 57,976 / 62,537 = 0.982 / 1.053 / 0.971; all ten content fixtures byte-identical across their three arms |
| M22 | Declines priced inside the window (`afe84d1`) | A decline is a chosen call, so it takes a window slot at its drafter cost and three of them close a losing window; it is not a rejection, so `losing_cycles` counts verified calls only. The first build of the port kept declines out of the window entirely | prose 0.816 -> **0.988** and middle-chat 0.893 -> **0.988** over serial, with 218 declines becoming 18; the draftable fixtures give a little back (middle-code 1.369 -> 1.263, middle-math 1.208 -> 1.115, json 1.375 -> 1.328, sql unchanged at 1.576). Every arm byte-identical to serial |
| M23 | Drafter fault latch (`35ed0f0`, `f1ba77e`) | Session-scoped `{disabled, unsafe, requests, attempts, failures, skips}`, ported from the GLM branch. A drained failure falls back to serial for the life of the session; an undrained one has no correct fallback and the session now refuses with a reason instead of returning an empty 200 | unreachable without a drafter fault, so it touches no measured arm. Verified by injection: `DS4_DS41_DSPARK_FAIL=after_verify` now returns `dspark: unsafe drafter failure requires a new session`; `drafter_once` leaves the faulted request and the next one byte-identical to serial |

**The width guard is a lesson, not a feature.** `dspark_verify_rows: 1` was
rejected by the lever setter but the debug HTTP paths ignored the rejection, so an
arm asking for width 1 measured the full six-row block and reported it as k=1.
Those arms were voided and re-run. Both paths now refuse with HTTP 400 and the
message `dspark: verify rows must be 0 (full block) or 2..6; use serial for
width 1`.

## 3a. Prefill checkpoints and the disk KV cache

These do not change decode and are not in the A/B gate above; they change what a
long prefill leaves behind and what survives eviction, so their result column is
restore time and store behaviour. Full mechanism in
[docs/V41_M3ULTRA.md](docs/V41_M3ULTRA.md) sections 3 and 5.

| # | change | what changed | result |
|---|---|---|---|
| K1 | Continuable checkpoint at every chunk boundary (`0f0a367`) | V4.1 defers its output head to the end of a prefill, so the chunk epilogue cleared `checkpoint_valid` immediately before the progress callback and `ds4_session_stage_payload()` refused every mid-prefill store — upstream behaviour since `bd66c40`. The chunk's last row now takes the output head when no decoder work is pending; a deferred decoder still leaves the checkpoint invalid, which is the case that must stay refused | one output-head matmul per chunk, ~2-3 ms. `tokens=94208 trimmed=0 reason=continued size=593.70 MiB save=101.9 ms` where there had been `no valid checkpoint to stage`; a 96,591-token strict prefix restores 94,208 from disk in 50.9 ms and finishes in **10.3 s** against a cold ~140 s. 62k prefill 790.79 / 792.68 t/s against the 798.25 receipt, -0.82 % on the mean |
| K2 | Continued stores scheduled on anchored chunk boundaries (`9402d39`) | The target fired only when the live token count *was* a multiple of the interval, while V4.1 sweeps in `resume + k x 2048` chunks anchored wherever prefill resumed, so on an agent chat resuming from a ~2.8k anchor no continued store was ever scheduled. Read as "the next multiple has been reached or passed" and store the boundary's own length; lookup is by text prefix and accepts any stored length | a 278,169-token prefill that had written nothing now cold-stores at 2,912 and writes `reason=continued` at 35,680 and 37,728 with zero skips; a 38,690-token strict prefix restores 37,728 from disk in 32.5 ms and finishes in **4.11 s** against ~50 s from the anchor. 62k prefill 791.99 t/s against 791.7, +0.04 % |
| K3 | Eviction-score baseline decays (`e26d830`) | `(hits x 2^(-t/6h) + 1.0) x density` aged the hit bonus but not the `1.0`, so dense anchor-classed entries were effectively immortal and passes culled the live session's minutes-old waypoints instead. The baseline now decays on `DS4_KVSTORE_BASE_HALF_LIFE_SECONDS`, default 24 h, floored at `1e-6` | unused entries yield to fresh ones within a day; intra-day behaviour unchanged. Ported from this project's GLM-5.3-Flash branch |
| K4 | Waypoint ladder retention (`bdbe6cf`) | An un-thinned 300k chain stores ~67 GiB of full-prefix snapshots at ~6.3 KB/token, and under a 98,304 MB budget the cheap-victim policy let one chat evict its own shallow rungs. Thin at store time: dense within `KV_CACHE_LADDER_DENSE_WINDOW` (32,768) of the frontier, then a 5/7 geometric skeleton; dropped rungs unlink immediately, kept rungs are touched on every chain store. Interval-catchup stores get a real reason code instead of `UNKNOWN` | kept set 4.6x the frontier against 14.5x un-thinned. `DS4_KV_LADDER=0` restores the old behaviour. Ported from the same branch |
| K5 | Deepest-first shedding under pressure (`16e979b`) | With thinning in place the evict pass still tie-broke by age and ate the shallow rungs first, so a shallow divergence fell to token zero anyway. Same-chain superseded rungs are weighted by inverse depth: deep rungs free the most bytes and lose the least coverage, since the live slot serves near-frontier divergences | the 3 GB-budget end-to-end stops evicting 10240/20480 by age tie-break. Ported from the same branch |
| — | Two ported test assertions corrected (`e389ddd`), one new test (`03eb931`) | The superseded-prefix test encodes the pre-ladder policy and is guarded with `DS4_KV_LADDER=0`; the ladder budget bound asserted as `< 3x` the frontier is unsatisfiable, since the three dense rungs alone sum to 2.8x, and becomes `< 5x` plus "twice the kept total below the un-thinned total", checked against a standalone copy of the selector | both corrections are owed back to the branch this was ported from, where the same two assertions fail under its own default. The new test reproduces the eviction failure at the store layer with no prefill: the aged anchor is the victim, both live rungs survive, and a fresh-anchor contrast shows age drives it |

## 4. Rejected, with the number that killed it

None of these ship. Their kernels, levers and plumbing were deleted from this
branch — 2,152 lines against the experimental head — rather than left gated off.
Verify times are per cycle at six rows, 8k. Where a fixture is named, `sql` is the
clean read: it commits 6.000 tokens per cycle on every cycle, so its throughput
tracks verify cost alone and no acceptance artefact can contaminate it.

| mechanism | result | why |
|---|---|---|
| Exact routed gate/up union with shared quant tiles (`mtp_gu_union6`) | rejected | bit-exact but **+7.51 ms on `code`, +7.93 on `agent`**; the ledger rate *fell* 345 -> 323 GB/s. Mask 0 alone costs +1.8 ms, so the plan-and-stage structure has a fixed price |
| Two-row routed gate/up union (`mtp_gu_union2`) | rejected, not exact | first difference at layer 0 `gate` element 99, 1 ULP, on *every* route including singletons; crosses a BF16 carry ULP by layer 7 and expert selection differs by layer 12 |
| FP32 register-blocked skinny GEMM with a GPU planner (`mtp_gu_mk6`) | rejected | **verify 108.5 -> 228 ms**. The seven kernels share one pass per layer costing 145.51 ms against the parent's 27.75, ~5.2x. Not submission-bound: the command-buffer trace reads 98.1 % GPU busy |
| FP16 accumulators, routed gate/up (`mtp_gu_f16acc6`) | rejected | **+10.4 ms** (`sql` 109.78 -> 119.86, -7.9 %); `pair_swiglu` +32.6 %. The float/half conversions cost more than the narrower accumulators save |
| FP16 accumulators, dense Q8 (`mtp_q8_f16acc6`) | rejected, neutral | 0.0 % on `sql`; own kernel +0.04 %. Not a loss, nothing to retain |
| M8/N32/K32 simdgroup MMA, routed gate/up (`mtp_gu_mma6`) | rejected | **+11.6 ms** (`sql` -8.7 %). Executed matrix-row utilisation is 36/(8U) ~ 19-20 %: 22.6-23.9 distinct experts per layer, about 1.5 rows in an 8-row tile |
| Same tile, dense Q8 (`mtp_q8_mma6`) | rejected | **+22.1 ms**, own kernel +151 % — and dense has six of eight rows useful, so padding does **not** explain this one. The more interesting failure of the two |
| Lower the 32-row `use_mm_id` threshold (`mtp_force_mm`) | rejected | **+40 ms**; routed 42.87 -> 82.62 ms/cycle. `static_assert(NR1 == 32)` means every occupied expert pays a full 32-row tile |
| Prefill-style batched attention core for the verify (`verify_batch_core`) | rejected | **+1.7 ms** — and diagnostically the most useful failure: it removed 1,749 dispatches per cycle, 35 % of them, and GPU pass time *rose* 126.4 -> 127.4 ms |
| Mixed HC cohorts at k rows (`mtp_hc_mixed`) | rejected | +0.80 ms on `code`, costs on both fixtures |
| Flat QA/KV staging at k rows (`mtp_qa_kv_flat`) | rejected, inside noise | -0.35 ms on `code`, -0.22 on `agent` |
| Four-row verify as its own shape | rejected | 0.796x against six rows' 1.164x: verify barely falls (124.6 vs 108.5 ms) while committed/cycle drops 4.37 -> 3.38 |
| Integer MMQ (int8 x Q4_K nibbles) | closed on paper | M3 and MSL 3.2 expose no packed integer dot product. Said plainly before drawing a kernel |
| Dispatch reduction by batching HC/norm/quantize/RoPE | closed on paper | a no-model microbenchmark prices a dispatch at 1.65-1.84 us barrier-separated, 0.209-0.269 us unbarriered. At 4,993 dispatches per cycle that is at most 8.9 ms fully serialised, ~1.0 ms unbarriered — and the batched-core probe showed a 35 % dispatch cut buying nothing |
| Dense Q8 / routed Q4 operand reuse and staging in prefill | priced, declined, no kernel written | production -> no-dequant is the whole ceiling (+4.0-4.3 % dense, +5.9 % routed) and both families already run at ~100 % of their own measured wall |
| Agent-trained controller calibration | built, measured, rejected | it reproduced the code-trained controller's decisions on the held-out `agent2` fixture **to the cycle** — same 17 cycles, 39 attempts, 22 declines, 5.588 committed per cycle — while being slightly worse in sample. A nine-variant policy sweep spanned 32.37-33.01 t/s against a 32.86 default and the nominal best did not survive repetition. Superseded entirely by the row below |
| Calibrated-window admission control (controller D) | built, measured, **deleted** | its Wilson acceptance bins covered three 512-position windows, and real chat prompts arrive at 8,745 / 23,457 / 57,976 / 62,537 / 128,679 tokens, so it made **zero attempts on every one of them** — and zero attempts beyond 33,279 tokens by construction. Generalizing it with a donor prior admitted, won on the cycles it ran, then stopped the request on its own predeclared loss allowance: **31.9 t/s at 8,745 tokens against a fixed six at 41.1 on the same kernels.** A controller that can latch off has neither fixed six nor serial as a floor. Deleted rather than tuned, along with its calibration file, its loader and the `DS4_DS41_DSPARK_CALIBRATION` variable |
| Confidence admission on V4.1 | available, not consumed | the drafter's Stage 4 rank-256 Markov row logits are the equivalent of GLM's selector confidence, but scoring them costs a GPU readback inside the proposal, and the window already dials drafting back. The signal is left available rather than wired in |
| `--prefill-chunk` above 8,192 | accepted then clamped | `DS41_PREFILL_CAP` is a compile-time 8,192 the prefill path allocates its batch buffers against; raising it would multiply ~9 GiB of prefill buffers by eight. Reported rather than patched: it is a memory-plan decision |

**What this does and does not establish.** Six independent reformulations of the
six-row routed gate/up matmul lost to the parent GEMV family for a different
measured reason each time. The consistent observation is that `pair_swiglu` at six
rows is doing close to the minimum work its shape contains. It is **not**
established that the routed step is at a physical floor: unique-expert counts do
not establish DRAM traffic, and the instrumented routed figure is not an
independently measured production floor. This is a bounded search.

One ratio is unexplained and is the most defensible place to restart: the dense Q8
projections and attention move **1x the bytes of a serial step** with six times
trivially more compute, and take **4.4x and 4.9x the time**.

## 5. Instrumentation, kept

Default-off, no kernels, no arithmetic changed. These observe the engine rather
than alter it, and they are what caught two errors that had already been written
down as results.

- **Paged per-dispatch kernel ledger** with a `#COVERAGE` certificate. It
  **refuses to print any timing field when coverage is incomplete**. An earlier
  sampler silently dropped 173,369 of 319,558 passes — 45.7 % — at six rows while
  still printing totals, and every figure taken from it was retracted. Every cost
  map in this branch's record carries 100.000 % coverage.
- **Command-buffer trace** (`DS4_METAL_CB_TRACE`), hooked on buffer *creation*.
  Hooked on the drain path instead it saw 4.53 of 15.5 buffers per cycle and left
  68 % unattributed, which had supported a claim that ~44 % of the cycle was host
  cost. Hooked on creation, GPU busy is 96.1 % of the cycle.
- **`/debug/levers`** and `/debug/bench` with `ledger_dump`, for reproducing any
  arm from one binary.

## 6. Testing, common to every entry

1. Machine quiet before each arm: GPU utilisation 0 %, no stray model process, no
   hot daemons.
2. Speed: interleaved off/on through one resident server at the 8k gate, two arms,
   four when the delta is under 1 t/s; utilisation reported alongside.
3. Fidelity: decoded text byte-identical to the upstream `bd66c40` reference for
   that fixture, for **every** arm — never to a same-binary control.
4. Every lever keeps an `=0` kill switch.
5. One fresh-prefill confirmation in `ds4-bench` shape at the end, on the release
   head, which is the source of the numbers in [docs/V41_M3ULTRA.md](docs/V41_M3ULTRA.md)
   section 1.
