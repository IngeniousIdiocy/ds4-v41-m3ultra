---
name: metal-decode
description: Questions that find serial decode speed for LLMs on Apple Silicon GPUs with Metal, each tied to the measured win it found, from taking DeepSeek V4.1 Flash on DwarfStar (ds4) from 19.9 to 41.4 t/s on an M3 Ultra bit-exactly. Covers the bandwidth floor and where the gap goes (idle, launches, bytes, work), command-buffer idle time, serial threadgroup tails, fixed shapes, narrow weight copies, fusion, long-context index and top-k, bit-exact kernels, Mac measurement traps, Metal hazards and the decode ideas that did not pay. Use when decode tokens/s on a Mac is too low or seems stuck at a ceiling, when writing or reviewing a decode Metal kernel, or when A/B-testing a decode change or proving it bit-exact.
---

# Metal decode

These are the questions that found decode speed, each with the win it found. GPU basics are assumed.

Numbers are from a Mac Studio M3 Ultra (80 GPU cores, 512 GB) running DeepSeek V4.1 Flash MXFP4 on DwarfStar. They are serial decode at 8K, measured one group at a time on DwarfStar main, unless stated. GLM-5.3-Flash numbers are labelled.

## Floor and attribution

**The floor.**
- Bytes per token = dense weights + routed experts × (used / total) + the rows actually read from lookup tables + the KV and index keys read at that depth.
- Floor = bytes per token / the measured streaming rate: about 705 GB/s on the M3 Ultra, against the spec's 819.
- Read tensor sizes from the GGUF header's tensor table. Never map the weights to do it.
- Leave out drafter and vision tensors.
- Count tables read by row (V4.1's two 101 GB Engram tables) by the rows read, not their size.
- A measured speed above the floor means the ledger is wrong.

**V4.1 numbers.**
- 8.74 GB of dense weights plus 4.51 GB of experts (6 of 384) is 13.25 GB per token: 18.8 ms, or 53 t/s.
- With exact narrow copies it is 12.74 GB: 18.1 ms, or 55 t/s.
- Main ran 19.9 t/s (50 ms); the release runs 41.4 (24.2 ms, 78% of the stored-weights floor).
- GLM: 13.83 GB per token, about 51 t/s; it shipped at 40.9 (80%).

**Attribution.** Split the gap into idle, launches, bytes and work.
- On untuned main at 8K, idle cost about 16 ms per token and launches about 11 ms, against about 2 ms of extra bytes and under 1 ms of work. Bytes are the last lever, not the first.
- Write down the expected number from the costs before measuring, and explain any gap over a point afterwards.
- Price the goal in ms: 35 to 37 t/s needs 1.54 ms per token.

## Questions to ask

**What is the GPU doing between kernels?** Kernel profiles do not show it. An encoder timeline does: each command buffer's encode time, GPU start and GPU end (ds4: `DS4_METAL_ENCODER_TIMELINE`).
- **One command stream per token.** The host committed and waited per layer. Commit after layer 1, then every two layers, with the output head in the same stream. Worth +7.1 t/s (19.9 to 27.6), the largest step.
- **Disk reads that start early and join late.**
  - Start per-token disk reads at the top of the step. Join each table just before committing the command buffer that holds its layer, not before encoding it.
  - One worker per row instead of four workers of six rows removed a 0.4 ms wait per token (39.68 to 40.02 t/s).
  - Decode keeps 16 batch readers.
- **Pre-encode the next token's first command buffer** while the host waits. The embedding reads the token id from a device word. Drop the buffer on any mismatch or reset. Worth +0.28 on main and +0.14 on the release branch.
- **Small copies in the open compute encoder**, because a blit closes it: +0.41. **Untracked read-only weight buffers**: +0.25.
- **Host work between tokens.**
  - A NEON scan of the sampler's first pass cut it from 674 to 555 µs per call on sampled decoding.
  - Moving the drafter capture into the head's command buffer saved 0.24 ms per token.

**How many threadgroups are busy in each phase?** A phase that ends on a few threadgroups is a serial tail, however few bytes it moves.
- **One-core reductions.** The Engram gate ran one 32-lane group per stream, and the hyper-connection collapse ran on one 1,024-thread group. Spread the loads but keep each lane's sum chain and order: +0.66 t/s, bit-exact. The gate went from 122.5 to 7.5 µs.
- **The top-k merge ladder** ended its last rounds on two or three threadgroups. A radix select finds only the 512th score's cutoff across the whole GPU, for rows of 16,384 or more: +2.56 t/s at 131K.

**What is known before the token runs?**
- **Fixed shapes.** Compile the model's constant decode shapes into the kernels: +0.98 t/s on the release branch and +2.3 on main.
  - Use one literal-shape wrapper per call site, with no function constants, gated on the exact shape, and keep the general kernel as the off path.
  - Admit shapes to existing fixed-shape kernels before writing new ones. The routed experts reached the engine's fixed-route MXFP4 kernels with a one-line admission.
  - For short inner sizes, give one output row to each SIMD group.
- **Position-only work, once per step.** One RoPE table per token: +0.62 t/s. Its bigger value is that neighbours absorb RoPE:
  - The attention reduce writes heads already inverse-rotated.
  - One dispatch ropes, quantizes and stores a cache row.
  - The table must copy the in-kernel expression and its FMA contraction.

**What precision does each tensor actually carry?** Compare the checkpoint's format with the file's storage format, tensor by tensor. V4.1 stores BF16 router gates as F32 (every low half zero) and FP8 E4M3 projections as F16.
- Build the exact narrow copy at load and check that every value converts. If any tensor in a family fails, keep the whole family wide. Worth +0.82 t/s, bit-exact.
- Keep the baseline's arithmetic. The reader that shipped drops E4M3 bits into an F16 bit pattern (exactly value/256, subnormals included), then multiplies each lane's partial sum by the block's power-of-two scale.
  - A generic E4M3 decoder streamed 9% slower than Q8_0, and a 256-entry lookup table 2% slower.
  - There is no FP8 arithmetic, and FP16 has no 2x over FP32 (scalar FMA is about 26 TFLOP/s for all three).
- Measure related narrow readers together. One alone showed nothing; three held at −0.22 and −0.19 ms per token.
- Check caches too. V4.1's index keys are written already rounded to bf16 but stored as f32.
  - A bf16 copy, read past 8,192 keys and scored one key per thread, gave +2.6 t/s at 131K.
  - It costs 64 floats per compressed row, and a restore rebuilds it and checks every value.

**Which dispatches move only a few kilobytes?** On GLM, 600 of about 1,000 dispatches per token had fewer than 100 threadgroups. They took a fifth of the time for 4% of the bytes. A dependent dispatch boundary cost about 1 µs, not the 6 µs a ledger intercept suggested.
- **Hyper-connections.** The mix took six passes per block. Fusing them, with producers writing the four streams directly, gave +3.5 t/s.
- **Router.** Probabilities, biased top-k and normalized weights in one dispatch: +1.6.
- **Rounding.** Round to bf16 in the producer's store or the consumer's load, not in separate passes: +1.4. Select it with a template parameter, and put every consumer of the rounded value under the same switch.
- **Fusion limits.** Fewer launches is not the goal (see "Didn't pay").

**What small independent work can ride beside a big stream?**
- The shared expert runs beside the routed experts in a concurrent encoder with explicit dependencies: about +0.5 t/s.
- The attention block's hyper-connection mix runs inside the query projection's dispatch.
- On GLM, the router was folded into the shared gate/up grid.
- Moving work that is already hidden beside a big stream saves nothing.

**What grows with context, and what has an earlier layer already ruled out?**
- Past layer 20, V4.1 can only select blocks that layer 20 kept. Build that candidate list once per token, and score and select only over it.
- Stage the selected attention rows straight from the cache by id, without a gather copy.
- Together these gave +1.4 t/s at 131K.

## Traps

- **Joins.** A host input joined before encoding blocks the host; joined just before the consuming command buffer's commit, it overlaps. Give each joined table its own storage, or the GPU drains before the second consumer.
- **Pre-encoded buffers.** Drop the buffer, and restore the state it advanced, on any key mismatch (graph, position, passes, drafter state), reset or other command buffer. Test plain steps, speculative cycles, a snapshot and its restore with the feature off and on.
- **Untracked buffers** skip hazard tracking: right for weights only the GPU reads, and anywhere else only under tested ownership and ordering.
- **Load-time builders.** Parallel narrow-copy builders share state. A plain bool "not exact" flag across threads was a data race; make it atomic, and stop the others once one fails.
- **Shared kernels.** V4.1's top-k shortcuts also ran in V4 Flash decode, which shares the kernel. Gate model-specific paths on the model and test the other family.
- **Tied keys.** A top-k select can swap keys with exactly equal scores. Say so; real scores almost never tie.
- **Allocations.** Allocate only where a path runs. A bf16 view was allocated on a backend that never read it.
- **Unused GPU work.** A top-p summary encoded on greedy steps that never read it cost about 40 µs per token (0.2%). Encode optional work only when it will be read.
- **Overlap.** Overlap exposes aliases: overlapped router and shared-expert work needed a separate intermediate tensor.
- **Two-die GPU.** Ordinary device stores read by an "elected last threadgroup" were unsafe despite fences. Use tested atomic publication or a dispatch boundary. A completion counter does not make its payload safe, and spin-waits on unscheduled producers can deadlock.
- **Function constants.** A function constant on a helper shared by several pipelines breaks the ones that do not set it. Use template parameters.
- **The path that runs.** Trace shape, format and mode selection to the pipeline that executes. A fast fallback contributes nothing, and a fallback that compiles is not proven to run.

## Didn't pay

- **Persistent whole-MoE kernel** (GLM; 240 resident threadgroups, counter hand-offs). It was bit-exact over 7.3 billion words and had 168 fewer dispatches per token, but it was slower than an ordinary wide schedule.
- **One-dispatch hyper-connection tail.** Slower than the fused pair: the single finishing threadgroup became the bottleneck.
- **Splitting the first command buffer, plus a completion-handler latch.** The gap was about 10 µs exposed, against a 40 µs bar.
- **Host pipeline-cache activation.** −0.006 t/s (SE 0.02). The host work was off the critical path.
- **The bf16 index scorer merged into the split scorer as a one-split case.** Exact, but −1.68 t/s at 131K (−4.2%). Time hot kernels before merging them for tidiness.
- **Expert-down widened from 2,560 to 15,360 threadgroups**, with 3.75 MiB of scratch and 640 combine groups. It was exact over 256 million comparisons and measured −0.031 ± 0.035 t/s. Threadgroup count is available work, not throughput.
- **Expert sums reassociated for parallelism** (GLM's tier-2 win). On V4.1 it improved mean NLL but broke the drift and prefix limits, and was rejected.
- **A four-accumulator Q8 layout on large projections.** It won on short projections (32.23 to 32.96 t/s on an earlier build) and lost on large ones (66 to 78 µs; 21.55 to 26.14 µs). Shape guards are part of the optimization.
- **A cooperative expert-down kernel** was exact and faster than the original, but 0.171 t/s slower than the current best. Compare against the best, not the oldest.
- **A matrix-unit index scorer carried over from prefill.** Exact, but scoring rose from 1,638 to 1,752 µs per token at 300K.
- **A reordered gate reduction** (float4 lanes, then simd_sum, then an eight-way threadgroup sum). It changed one bf16 value in 30,000 and moved NLL by +0.00256 nats per token. A later reordered form, 1.8 µs faster than the exact one, was also left out.
- **The first FP8 reader.** It had the right values, but a different scale placement and reduction order, and it failed the quality screen.
- **An Engram row cache.** 30.7% of rows recur, but a token stops waiting only when all 24 rows hit, which happened on 8.7 to 14.1% of tokens. Price the hit that removes a wait, not the hit rate.
- **A narrow Engram reader measured alone.** It showed no gain, because neighbouring work absorbed it.
- **Optimizing a shared-expert wrapper the default path did not call.** Speed barely moved; the default path ran a concurrent wrapper.

## Exactness while fast

- `simd_sum` is a fixed adjacent-pairs tree (strides 1, 2, 4, 8, 16; 0 of 262,144 random cases differ). Rebuild it in registers to merge lane sums with the same bits.
- `simdgroup_multiply` started from zero gives exactly the scalar FMA chain over k = 0..7. `simdgroup_multiply_accumulate` with a nonzero C does not.
- The Metal library builds with fast math, which reassociates written add trees: every result differed, by up to 4.9%. Put `#pragma clang fp reassociate(off)` inside the function where the order matters.
- Fast math also moves FMA contraction when code moves. Hoisting a norm changed 302 of 71.68 million values. Compare the compiled output at production shapes.
- Spread the loads, never the sum. Keep redundant ops whose removal changes rounding: eliding a duplicated norm moved one value by one ULP and changed tokens.
- **Logits hashes.** Hash every step's full logits (FNV-1a over the raw float bits, outside the timed region; ds4-bench builds that carry it print these with `DS4_BENCH_DECODE_LOGITS_HASH=1`).
  - Run 128 greedy steps at a short and a long depth.
  - Run the control twice first, and repeat with each switch off on its own.
  - Report the first step that differs.
- **Kernel-run counters.** Add a test-only counter per fast path, and require the expected count with the switch on and zero with it off. A fallback compared with itself passes.
- **Unit tests.**
  - Keep an immutable copy of the old kernel; a reference built from the edited template shared its bug.
  - Use production shapes with at least 50,000 fresh draws, and repeat identical inputs about 1,000 times.
  - Poison the outputs; stale NaNs made "zero mismatches" vacuous.
  - Plant a strong control; a single-bit change vanished in rounding.
- **Tier 2** (reordered arithmetic) needs evidence agreed with the user before seeing the speedup.
  - Run 100 teacher-forced prompts in the same build.
  - The paired mean NLL change must be at most twice its standard error and at most 1e-4.
  - Wins against losses: p > 0.05.
  - First mismatch within ±1 and mean common prefix within ±0.1.
  - Drift from a pinned reference at most 3e-4.
  - Add a kill switch.

## Procedure, switches and measurement

```
- [ ] Expected number written from the floor and costs
- [ ] CPU gates per change: build, bitwise test vs the old kernel, kernel-run counter, switch restores old path
- [ ] One GPU window at the end (a window per group for ~30 groups took 8 h), wall time estimated first
- [ ] Project's existing lock/window scripts (each newly written one forgot the display)
- [ ] Other model processes stopped; any job over ~2 cores waits
- [ ] kill -STOP WallpaperAerialsExtension VTDecoderXPCService mediaanalysisd photoanalysisd (-CONT in an EXIT trap)
- [ ] pmset displaysleepnow, with display events logged for the run
- [ ] ioreg -r -d 1 -c IOAccelerator: "Device Utilization %" <= 5 in 10 one-second samples
- [ ] vm_stat anonymous and compressor watched; the model is never mapped through a CPU MAP_PRIVATE path
- [ ] Logits hashes equal the control before any speed number counts
```

**Switches and records.**
- Give each switch one idea a person can explain in a sentence; never aim for a count.
- Check a switch on every use. Load-time switches (untracked buffers, narrow copies) cannot flip in process, so measure them in separate processes.
- Measure the progression: upstream, all off (equal to upstream), each group on in dependency order, then all on.
- Ship exact wins on by default. A failed idea loses its code and gains a "Didn't pay" line.

**Measurement traps.**
- **Display.** An awake display cost about 2 t/s: an overnight session measured 39.2 against 41.1 quiet. The compositor and an animated wallpaper took 13 to 14% of the GPU, which idle checks miss.
- **Contention and drift.** An upstream gate drifted from 19.3 to 16.6 t/s in one morning. Record the quiet check next to every run.
- **Model load dominates per-process A/B.** Load once, keep every switch in one bitmask with a test setter, and restore a saved cache from memory per configuration. Time in serpentine order (A B B A A B), median of three; single runs vary by 0.1 to 0.2 t/s.
- **Saved caches.** Save caches per branch and depth. A 131K decode from a restore takes about 7 s, against about 170 s cold.
  - Store each with its commit, state format, model and drafter digests, context and logits hashes.
  - Check that a restore gives the cold run's hash.
  - Caches belong to one engine lineage.
  - Keep them, and write the restore command into the results.
- **Wall time.** A six-hour validation (a restart per cell, cold prefills to 300K, a broken cache) should have stopped in its first hour. A full validation fits in about 40 minutes.
- **References.** A number that disagrees with a known reference stops the work. Check, in order:
  1. Was the machine quiet?
  2. Was a cache missed?
  3. Did the wrong path run?
  4. Do the binary or weights differ?
  5. Was first-use compilation timed?
- **Profilers.** Instrumented modes can disable batching, concurrency or fusion. Find gaps in traces; decide with uninstrumented runs.
- **Deltas.** Per-commit deltas do not sum to base-to-final (up to 0.1 t/s of drift). Re-test old wins after the baseline changes.
- **ds4 builds.** `make cpu` relinks `ds4`, `ds4-server` and `ds4-bench` as CPU binaries in the same tree; rebuild with `make -B` before GPU runs.
- **Fast runs.** Do not drop a run because it is fast. Exclude it only for a demonstrated invalid condition, and keep its record. Pin speed targets to the method that produced the reference.
- **Memory.** A token-dump job mapped the 500 GB model through the CPU path beside Metal processes. It left about 360 GiB of unowned anonymous and compressed memory and caused five Jetsam storms. Abort above 64 GiB anonymous or 16 GiB compressor beyond the measured process, and check for `JetsamEvent-*.ips` afterwards.

Built on antirez's DwarfStar (ds4); thanks to Ivan Fioravanti and Tarjei Mandt for ideas this work borrowed.
