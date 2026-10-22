---
name: metal-mtp
description: Questions that make speculative decoding (MTP, draft models, DFlash, DSpark) pay on Apple Silicon GPUs with Metal, each tied to the measured win it found, from GLM-5.3 and DeepSeek V4.1 Flash on DwarfStar (ds4) on an M3 Ultra. Covers cost per committed token, rollback by prefix snapshot, drafter state in saved sessions, verify row sharing and its cost curve, the drafting controller, exact versus batched verify, sampled speculation correctness and the ideas that did not pay. Use when a drafter or MTP head is slower than expected or slower than serial, when tuning verify kernels or the draft controller, or when checking speculative output against serial decoding.
---

# Metal MTP (speculative decoding)

These are the questions that made speculation pay, each with the win it found. GPU and speculative-decoding basics are assumed.

Numbers are from a Mac Studio M3 Ultra (80 GPU cores, 512 GB):
- **DFlash on GLM-5.3-Flash:** an anchor plus seven proposals.
- **DSpark on DeepSeek V4.1 Flash on DwarfStar:** up to five drafts and a six-row verify. On an 8K code prompt it reaches 59.0 t/s with every group on, against about 27 to 30 on main (two methods). Serial decode is 41.4.

## Questions to ask

**What does a committed token cost, all in?**
- Break-even is cycle cost over measured serial token cost. For example, a 100 ms cycle against a 28 ms serial token must commit more than 3.57 tokens.
- Content changes acceptance, never break-even.
- Count declined attempts, calibration tokens, readback and the drafter's upkeep. A faster serial path raises the bar automatically, so do not hard-code a handicap.

**What does rejection pay for?**
- A DFlash run accepted 78% of drafts and still ran 35.35 t/s against 40.44 serial. It replayed 158 accepted tokens, 3.95 s.
- Snapshotting the complete accepted-prefix state (sparse-attention key and gate tails, linear-attention recurrent and convolution state) gave a matched pair of 47.40 against 39.77 t/s. Tail work fell from 3.97 s to 0.069 s.

**What stalls the verify block?**
- A verify block read every row's disk rows synchronously before encoding, which left the GPU idle 30 ms per block. Moving the reads onto workers that join just before the consuming commit gave DSpark code +7.1 t/s (35.99 to 45.86 in one measurement).
- An encoder timeline (each command buffer's encode, GPU start and GPU end) showed it; no kernel profile did.

**What state does a restored session need?** It needs more than attention KV. Save the drafter's feature window with the session (three rings of 128 rows, 7.5 MiB on V4.1). Without it, every restore decodes 128 serial tokens before it can draft.

**What can the rows share?**
- Sharing the vocabulary-head weights across four rows took GLM's eight-row head from 7.7 to 2.0 ms, exact over 26 million logits. A combined head experiment reached 49.26 against 39.86 t/s on that fixture.
- Reading weights once for a few rows, with each row keeping the one-row kernel's order, gave DSpark code +4.8 t/s.
- Six- to eight-row verify blocks on the fused one-token expert kernels make each row bit-identical to decode.
- A verify block attending as one batch, like a short prefill, gave +5.0.
- Row reuse has a cost curve: 2, 3, 6 and 8 rows are competing designs, because accumulators, registers and staging grow with rows.
- Test no, partial and full expert overlap; full overlap is a capability case, not an average.
- A short-row MMA verify design beat row-sharing kernels: disabling its matrix units dropped verify from 77.5 to 65.4 rows/s (16%).

**When should it draft at all?** The controller is part of the economics. On V4.1 prose, drafting everywhere ran at 31.2 t/s, and the controller alone brought it to 39.1, about serial. On code it costs 1.5 t/s. The policy:
- decode 16 tokens serially, and take the median of the last nine serial tokens as the cost;
- verify only when the first three drafts are each at least 75% confident;
- otherwise decline, which costs the drafter and one serial token, not a verify;
- judge three cycles together, carry losses up to half a serial token, then back off 16, 32, 64 and 128 tokens.

Keep the policy fixed while tuning kernels. A trained confidence head's sigmoid is not the probability the threshold was tuned on.

**Which verify contract?** Decide, and state it at startup.
- Batched MMA rows sum in another order and flip near-ties against serial. Two groups were measured together for DSpark code +5.8: 3 to 8 rows on simdgroup matrices (inexact) and the fused verify expert kernels (exact). The batched attention verify is inexact too.
- Keep a strict mode equal to serial decode. Never let one-row decode take a batched kernel.
- Gate verify kernels by purpose, never by row count. Kernels gated on 3 to 8 rows also caught the minimum prefill batch of 8, and an 8-token prompt changed its first generated token.
- Causal-state correctness (rollback) is a separate contract from numerical equality.
- An exact MMA is possible. `simd_sum` is a fixed adjacent-pairs tree, and `simdgroup_multiply` from zero matches per-lane FMA chains, so a batched kernel can reproduce the row kernels' bits. Turn off fast-math reassociation there with `#pragma clang fp reassociate(off)`.

**Is sampled decoding still correct?** Greedy tests do not prove it. GLM's rejection path reapplied top-k, top-p and min-p to modified logits, which could change the residual's support, so GLM's release stayed greedy-only. A faster approximate sampler is a product decision.

**What does the drafter's correction cost?** Run the Markov correction on the GPU, not as a CPU full-vocabulary structure.

## Traps

- **Complete prefix state** includes incomplete pool or ring tails of compressed caches, linear-attention recurrent and convolution state, absolute positions and cache visibility.
- **Saved payloads.** A payload without the drafter section must still load. A payload with it must still load without a drafter, and a test restores one and repeats several cycles with the same tokens and logits.
- **Rollback tests.**
  - Change the rejected suffix, including the first rejected token, and confirm the accepted frontier does not move.
  - Cover ring boundaries, partial and full acceptance, end of sequence, stop and cancellation.
  - Compare full state, frontier logits and the continuation against serial replay.
- **Drafter faults.** Disable speculation only after the GPU has finished and the state is known safe. Never reuse uncertain mutated state.
- **Fixed drafting mode.** Exactness runs need one; adaptive drafting is not repeatable. Hash every step's logits and compare against the control run twice.
- **Metrics.** Report rows/s, ms per verify, committed t/s and wall time under their own names. Never relabel rows/s as committed t/s.
- **Two instruments.** Fixed-work fixtures (fixed rows, inputs and rejection positions) isolate kernel and state cost. Real requests through the controller measure user benefit.
- **Different tokens.** If the tokens differ, it is a workload result, not an identical-work kernel attribution. A valid free-running best run still counts as capability; explain what changed.
- **Path engagement.** A benchmark labelled speculative ran fully serial. Count drafts and kernel runs.
- **Method changes.** Numbers taken with a warm-up and 256 tokens are not comparable with numbers from saved drafter-state caches. Label such jumps.
- **Favourable fixtures.** Across 32 randomized real agent requests, GLM's windowed controller with a Q8 drafter gained 4.3% (39.73 against 38.09 t/s), and a matched story workload lost 1.8%. Report both.
- **Reasoning tokens.** They are a large share of agent tokens and draft poorly, so serial speed still matters.

## Didn't pay

- **Universal six-row weight reuse** lost on some families despite fewer nominal reads.
- **A three-row Engram projection** was about 11% slower than the two-row form it replaced.
- **Stable expert sorting and union planning** lost even before counting the planning cost.
- **Early six-row MMA and split-K verify prototypes** were slower and changed rounding. A later short-row MMA design won by 16%; reject the implementation, not the approach.
- **A hot-copy microbenchmark win for row reuse** disappeared with rotating weights in the model.
- **A 64-slot GPU drafter correction cache.** +0.03 t/s for 32 MiB per session, after an upstream redesign removed most of the cost.
- **An early per-attempt controller** priced one poor narrow verify and kept real code writes nearly serial. A windowed policy fixed it.
- **A controller threshold tuned to content.** A controller paying 5 to 7% on chat met the bar written for it, but derived from costs it would have been rejected before the run. For example, a 121 ms cycle against a 31 ms serial token breaks even at 3.9 rows. A losing verified probe costs 1.3 steps; a drafter-only decline costs 0.3.

## Procedure, switches and measurement

```
- [ ] Expected committed t/s written from cycle and serial costs
- [ ] CPU gates per change: build, bitwise test vs the old kernel, kernel-run counter, switch restores old path
- [ ] One GPU window at the end, wall time estimated first (load ~2 min, prefill ~900 t/s, decode ~40 t/s)
- [ ] Project's existing lock/window scripts; other model processes stopped; any job over ~2 cores waits
- [ ] kill -STOP WallpaperAerialsExtension VTDecoderXPCService mediaanalysisd photoanalysisd (-CONT in an EXIT trap)
- [ ] pmset displaysleepnow, with display events logged; IOAccelerator "Device Utilization %" <= 5 for 10 s
- [ ] vm_stat anonymous and compressor watched; the model is never mapped through a CPU MAP_PRIVATE path
```

- **Switches.** Give each switch one idea, with speculation grouped as finely as decode. Check switches on every use. Measure upstream, all off (equal to upstream), each group on, then all on.
- **Display.** An awake display cost about 2 t/s on serial decode (39.2 against 41.1 quiet). Never mix display states.
- **One process.** Load the model once and flip a switch mask in that process. Restore saved caches that include the drafter window, time in serpentine order and take the median of three.
- **References.** A number that disagrees with a known reference stops the work. Check quiet, missed cache or missing drafter state, path, binary and weights, and compilation.

Built on antirez's DwarfStar (ds4); thanks to Ivan Fioravanti and Tarjei Mandt for ideas this work borrowed.
