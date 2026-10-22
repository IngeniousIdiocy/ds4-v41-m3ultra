---
name: metal-prefill
description: Questions that find LLM prefill (prompt processing) speed on Apple Silicon GPUs with Metal, each tied to the measured win it found, from DeepSeek V4.1 Flash on DwarfStar (ds4) and GLM-5.3 on an M3 Ultra. Covers per-depth measurement, disk-backed inputs and reader counts, barriers, glue folding, dequantization, row pruning, token tiling and expert banks, an exact eight-row MMA kernel for small batches, row-count gating traps, prefill tier-2 evidence and the prefill ideas that did not pay. Use when prompt processing or time to first token on a Mac is slow, when writing or reviewing a prefill or batched-matmul Metal kernel, or when measuring prefill across context depths.
---

# Metal prefill

These are the questions that found prefill speed, each with the win it found. GPU basics are assumed.

Numbers are from a Mac Studio M3 Ultra (80 GPU cores, 512 GB) running DeepSeek V4.1 Flash MXFP4 on DwarfStar, and GLM-5.3-Flash where labelled. V4.1 cold 32K prefill went from 45.0 s on main to 37.9 s, and 131K from 189 s to 152 s.

## How to measure

- **Per depth, from one run.** Run one cold prefill of the deepest prompt and record the cumulative time at each chunk boundary. Use prompts that are prefixes of one document.
- **Why per depth.** A single 131K figure showed a 1% gap between two builds. The per-depth rates showed 6% at 32K closing with depth, which pointed to a fixed per-sweep cost.
- **In-run rates are not standalone short-prompt rates.** A standalone short prompt can prune work that the same depth inside a long prompt cannot.
- **Cache regimes.** Name the regime of every number: empty KV with warm weights, cold weights, a cached prefix plus an append, or a disk restore. A 0.23 s restore is not prefill throughput.
- **Appends.** Measure cached and short appends separately; fixed costs dominate short ones. 2,048 tokens after an 8K prefix went from 374 to 409 t/s, and after 63K from 302 to 326.
- **Drives and baselines.** Keep the model file on the same drive across arms, because the first sweep may wait on disk. Compare against the matrix units, not a scalar FMA probe.
- **FLOPs.** 2 × the decode-active matrix weights (16.1 B on V4.1, so 32 GFLOP per token) is only an upper bound. Count from the execution plan.

## Questions to ask

**What does the first sweep wait on?**
- Disk-backed rows are IOPS-bound: 1 reader gave 8,805 rows/s, 16 gave 121,763, 32 gave 202,015 and 64 gave 202,443, at only 53 MB/s. One uncached 264-byte read takes 113 µs. Price the reader count off the GPU, on the real table.
- Use 32 readers for prefill sweeps; decode keeps 16. That cut a cold 32K prefill by 2.5 s, the largest prefill step. Earlier, 16 to 32 readers gave +5.45% at 16K and cut the exposed wait at 62K from 5,625 to 926 ms.
- Read the next sweep's rows ahead, bounded, during this one. Adopt them only when their token ids match, and join the workers on reset, free and failure.

**Where do lanes sit at barriers?**
- Index scoring synchronized per head. Keeping each SIMD group's keys in registers across all 32 heads, on simdgroup matrices, cut 1.35 s at 32K with the same bits.
- Attention staged one key row per barrier. Staging 16 rows, with selected rows held as a 16-bit mask to free registers, cut 1.34 s.

**What glue runs as its own pass?**
- Fold the bf16 rounding, norms, expert add and embedding into their producers.
- Let F16 projections take two rows per threadgroup, sharing each weight read.
- Together these cut 1.29 s. Running index candidates and top-k as batches cut another 0.31 s.

**What is dequantized more than once?** Unpack Q8 weights to F16 once for products of 32 or more rows (M3 Ultra): 0.17 s.

**Which rows does the output never read?**
- V4.1's upper 20 layers need only the rows the last token depends on, so prefill is 18.2 GFLOP per token, not 32.
- Sweeps now start at 3,072 tokens instead of 8,192. That changes the chunking, so a short prompt's rounding can differ; say so.
- At 907 t/s, prefill runs at about 60% of the measured MMA peak (27.4 TFLOP/s, operands in registers). It runs at about 70% of what its quantized matmuls reach (22.7 to 24.1).
- Scalar FP32, FP16 and BF16 FMA all run at about 26 TFLOP/s, and there is no FP8 arithmetic.

**What is the minimum batch, and can it stay exact?**
- With the model resident, ds4's minimum prefill batch on one Mac is 8. Prompts of 3 to 7 tokens decode token by token.
- An exact eight-row Q8_0 kernel on simdgroup matrices took seven sites from 885 to 326 µs per layer, against 404 for the inexact MMA. End to end, the 8-token prefill was a tie within noise.
- The recipe:
  - Each (SIMD group, lane) pair stands in for one pair of the row kernel and keeps one accumulator tile per (block, quarter).
  - `simdgroup_multiply` from zero gives each `sumq`; scale and add it as the row kernel does.
  - Add lane sums in registers with simd_sum's adjacent-pairs tree, meet the halves in 2 KiB of threadgroup memory, then run a real `simd_sum` for the second level.
  - The epilogues stay the row kernel's code, applied per lane.
- `simdgroup_float8x8` thread layout: `qid = l/4; m = (qid&4) + ((l/2)%4); n = (qid&2)*2 + (l%2)*2`. Each thread holds (m, n) and (m, n+1).

**Earlier V4.1 prefill work.** Some changes were prefill-only:
- two-row F16 staging that keeps each row's reduction order;
- lean attention staging for large batches;
- the rounding folds;
- 32 readers with read-ahead.

A base without them ran 658 t/s at 8K and 741 at 63K; with them, 737 and 846.

**GLM.**
- GLM prefilled 550.27 t/s at 62,174 tokens and 473.64 at 300,000.
- **Token tiling.** A token-tiled router and low-rank projections stream weights once per block of tokens.
- **Expert bank.** A long-prompt expert bank expands a layer's experts once and reuses them across token tiles, with fused command buffers and bounded layer pipelining.
  - It is admitted only for long prompts: at 33,148 tokens it measured 556.73 against 540.64 t/s. That supported a conservative threshold, not the crossover.
  - Count its memory, preparation and fallback cost, and do not copy its guard to another model.
- **Other wins:**
  - split-K on very narrow BF16 shapes, under tier-2 evidence;
  - blocked online softmax with lean gather staging;
  - a faster prepare and recurrence for the linear-attention layers;
  - half-precision staging for dense Q8 where reuse repays the conversion.

## Traps

- **Row-count gates leak into prefill.**
  - Exactly 8 rows (an 8-token prompt or tail) is the batch that reaches 2-to-8-row paths.
  - Verify kernels gated on 3 to 8 rows caught it, and an 8-token prompt changed its first generated token (513 against 201).
  - Gate by purpose, and test every short prompt length against upstream: only 8 tokens differed. The same gates catch a short follow-up turn appended to a session.
- **Aliases.** Two-row F16 staging needs disjoint scratch where no trailing barrier protects reuse.
- **Read-ahead ownership.** Check that prefetched rows belong to the tokens that will consume them.
- **Two-die GPU.** Ordinary device stores read by an "elected last threadgroup" were unsafe despite fences. Use tested atomic publication or a dispatch boundary.
- **Function constants.** A function constant on a shared helper breaks the pipelines that do not set it. Use template parameters.

## Didn't pay

- **Pre-expanding five small Q8 matrices** (GLM). They were already hidden under large GEMMs, so it saved about 0.5 µs per token. Removing instructions from an overlapped side branch does not shorten the critical path.
- **Restricting the batched verify kernels to verify, so 8-token prompts match upstream.** Exact, but +17 ms (+6.5%) on the 8-token prefill. The exact batched kernel removed the trade.
- **A row-batched short-prefill attention path.** No speedup, and it changed prefill logits at 8K and 32K, because a sweep's last layer runs few rows.
- **The first exact eight-row MMA** (transpose through threadgroup memory, then real simd_sums). It was 3.8 ms per prefill slower than the inexact MMA. Rebuilding simd_sum's tree in registers made it faster than both.
- **Sixteen SIMD groups for the exact kernel.** It helped only the two smallest shapes (about 0.26 ms per prefill) and hurt the largest.
- **Fused verify expert kernels for eight-row prefill batches.** Not faster (0.265 against 0.272 s).
- **Large expert banks as a default** (GLM). They paid only above about 33K tokens.
- **A merged candidate compared without its prefill-only work.** It kept 741 t/s at 63K against 846. Inventory every phase before claiming a whole-build result.

## Exactness while fast

- **simd_sum.** It is a fixed adjacent-pairs tree (0 of 262,144 cases differ). An MMA started from zero matches per-lane sums; one with a nonzero C does not.
- **Fast math.** It reassociates written add trees (every result differed, by up to 4.9%). Use `#pragma clang fp reassociate(off)` where the order matters. Moving code also moves FMA contraction.
- **Logits.** Compare cold-prefill final logits and the first 32 greedy tokens against upstream at every short prompt length (3 to 8) and at long depths. Add test-only kernel-run counters, and keep an immutable copy of the old kernel in unit tests.
- **Tier 2** (reordered arithmetic) needs evidence agreed with the user before seeing the speedup. Perturbations propagate, so greedy identity alone is too brittle.
  - Use an FP64 reference at production shapes: at least 50,000 comparisons, with the same input staging.
  - Run a same-build teacher-forced set of at least 1,000 prompts, with a drift ledger.
  - Require at least +1% long-prefill speed in four interleaved repeats, with no short-bucket regression.
  - The prompts must reach the kernel's active regime.

## Procedure, switches and measurement

```
- [ ] Expected number written from the costs
- [ ] CPU gates per change: build, bitwise test vs the old kernel, kernel-run counter, switch restores old path
- [ ] One GPU window at the end, wall time estimated first (load ~2 min, prefill ~900 t/s)
- [ ] Project's existing lock/window scripts; other model processes stopped; any job over ~2 cores waits
- [ ] kill -STOP WallpaperAerialsExtension VTDecoderXPCService mediaanalysisd photoanalysisd (-CONT in an EXIT trap)
- [ ] pmset displaysleepnow, with display events logged; IOAccelerator "Device Utilization %" <= 5 for 10 s
- [ ] vm_stat anonymous and compressor watched; the model is never mapped through a CPU MAP_PRIVATE path
```

- **Switches.** Give each switch one idea, with prefill grouped as finely as decode. Check switches on every use. Measure upstream, all off (equal to upstream), each group on, then all on.
- **Display.** An awake display cost about 2 t/s on decode (39.2 against 41.1), and a quantizer at 1,200% CPU once ran during a prefill timing.
- **One process.** Run each prefill-changing step as a cold prefill in one process, flipping the switch mask, with the first run of each process labelled.
- **References.** A number that disagrees with a known reference stops the work. Check quiet, cache regime, path, binary and weights, and first-use compilation.
- **ds4 builds.** `make cpu` relinks the same binary names as CPU builds; rebuild with `make -B` before GPU runs.

Built on antirez's DwarfStar (ds4); thanks to Ivan Fioravanti and Tarjei Mandt for ideas this work borrowed.
