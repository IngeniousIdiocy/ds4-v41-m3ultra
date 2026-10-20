# V4.1 Flash 1017: how serial decode reached 40 tok/s

This is a condensed history of the work behind the October update: what was
tried, what was kept or dropped and why, and the key numbers. The
[receipts](RECEIPTS-V41-1017.md) hold the evidence for everything this
branch ships; this page gives the route that led there.

## The goal and the rules

The goal was at least 40 tokens/s of single-stream ("serial") decode for
DeepSeek V4.1 Flash 1017 on one M3 Ultra (512 GB), measured at 8K context
with sampling on. The campaign ran from October 17 to 20, 2026.

Rules from the start:

- **No less model work and no loss of quality.** Every gain had to come from
  executing the same model more efficiently. A storage change was allowed
  only where it was lossless, or where it returned a tensor to the
  checkpoint's own FP8 values.
- **Exact or explicitly numerical.** An exact change must give the same
  decoded tokens, final logits and whole session state as its control, in
  every run. Anything else was a numerical change, and a quality check
  decided it, not its speed.
- **Measured, not argued.** Projections from micro-benchmarks earned no
  credit. Only a resident A/B in a protected window could move the basis.

## How it was measured

- **Protected windows.** The production server was stopped for each window,
  so the measured process was the only one mapping the model. Background
  media and wallpaper processes were paused, the display was asleep, and a
  memory admission gate refused to start when the host was short. Display
  state alone moved both arms by 0.35 to 0.39 ms/token, so display-awake and
  display-asleep numbers were never mixed.
- **A resident runner.** `ds4-bench --jobs` restored a saved 8K prefix,
  checked against an identity record, applied per-run switches and recorded
  dispatch counters and digests. Arms were seed-matched and run in mirrored
  order (ABBA), with a warm-up pair excluded.
- **Fixtures.** Two 8,192-token prompts, prose and code. Each run sampled
  512 tokens at temperature 1 and top_p 0.95.
- **Rate.** Tokens divided by the time from the first sampling to the last
  evaluation. Ivan Fioravanti's steady metric (wall time of tokens 2..N)
  reads the same within about 0.04 tok/s at 8K.
- **Quality reference.** The original BF16/FP8 checkpoint was scored on
  cloud GPUs (two B300) at concurrency 1. Numerical changes were judged
  against those records, as well as against the unchanged local engine.
- **Review.** AI agents implemented and measured each lever, and a separate
  reviewing model ruled on each window against rules declared before it ran.
  Every ruling, including the negative ones, went into a decision ledger.

The windows were not pooled across machine states. A host watchdog panic
after the seventh window led to a reboot; from then on each batch carried its
own matched controls, and the harness gained fail-closed memory admission,
owned-process cleanup and identity-checked pausing of background processes.

## Starting point

The production stack (the September 18 release with later serving changes)
decoded at **35.12 / 35.05 tok/s** (prose / code, drafter armed, display
awake). A prose token took 28.48 ms. Of that:

- 27.02 ms was GPU command buffers, about 494 GB/s over the 13.3 GB read
  per token;
- 0.87 ms was the CPU sampler, with the GPU idle;
- 0.30 ms was the gap after the drain before the draft capture.

So the budget was about 3.5 ms per token, and most of it had to come from
the GPU side.

## The route to 40

Deltas are ms/token, treatment minus control in the same window, prose /
code. They come from different windows and are not additive.

### 1. Host side and lossless narrowing

- **BF16 router** (exact, lossless): the GGUF's F32 router is the source BF16
  widened, so storing it as BF16 loses nothing. −0.21.
- **Draft capture in the head's command buffer** (exact): 22 command buffers
  per token became 21, and the capture gap fell from 0.30 to 0.02 ms.
  −0.24 / −0.23.
- **SIMD sampler scan** (exact): a NEON finite/max/count scan, with everything
  else unchanged. −0.41 / −0.41.

### 2. FP8 readers

- **Engram FP8.** The first fast kernel failed the quality screen. The cause
  was its reduction and scale placement, not the FP8 values. A reader with
  the baseline's F16 arithmetic over the checkpoint's FP8 codes replaced it
  (−0.41 / −0.41).
- **On this branch:** the narrow copies (BF16 router, E4M3 Engram and indexer
  q_b) are built at load from the public weights, so no sidecar files are
  needed. Their readers ship as one blended unit: −0.22 / −0.19 ms against
  the wide reads, bit-identical. Alone, the Engram saving is eaten by its
  neighbours; with the router and indexer readers, it holds.
- **Output projection pair in FP8:** fast (−0.18), but it stayed on quality
  hold and never entered the basis.

### 3. Indexer and token selection

- **GPU top-512 selector** with the CPU keeping the decision and RNG:
  −0.54 / −0.39.
- **Indexer:** an ordered prefix merge and an FP8 indexer q_b consumer.
  Neither paid alone; together −0.06 / −0.08.
- **SIMD-shuffle initial block sort:** −0.12 / −0.07.
- **A faster exact CPU selection consumer** (late in the campaign):
  −0.015 / −0.036.

### 4. Literal shapes and once-per-step work

The engine's kernels are generic over shapes. V4.1's shapes are fixed, and
the work that depends only on the position can be done once per step:

- **RoPE coefficients once per step** in the Q8 output-low projection:
  −0.16 / −0.16.
- **Fixed-shape output-low with the RoPE table,** adapted from Ivan
  Fioravanti's fixed-shape decode mechanism: −0.13 / −0.14.
- **Attention heads transformed once:** the reducer's final writer rounds to
  BF16 and applies the inverse rotation, so the consumers do not repeat it.
  −0.08 / −0.13.
- **Literal-shape entries** for the HC output consumer, the routed gate/up
  FFN entry (−0.19 / −0.19) and the HC join (−0.26 / −0.25).
- **Literal q_a/KV pair:** −0.13 / −0.15.

After this phase the exact basis stood at **39.40 / 39.33 tok/s** (25.38 /
25.43 ms). No exact lever in hand closed the remaining 0.4 ms.

### 5. EADD, the Engram gate

The Engram residual gate ran as 32 lanes per stream and was latency-bound.
A 256-thread form (EADD) saved −0.24 / −0.33, but its summation order
differed, changing about 1 BF16 value in 30,000. It was handled as a
numerical change: on the general set, NLL moved +0.00256 nats/token
(SE 0.00137), and it was cleared for continued experimental work. With EADD the basis reached 39.81 / 39.83.

On this branch, EADD has since been replaced by a staged kernel that gives
the 32-lane kernel's bits exactly. It keeps about 90% of the gain (0 of
3,072,000 fresh streams differ).

### 6. The host chain

The last 0.1 to 0.2 ms came from the time between tokens:

- **Pre-encoding the next token's first command buffer** in the CPU slack
  before the wait, with the Engram rows joined late: −0.05 / −0.04.
- **A GPU bridge:** a short, fixed piece of GPU work queued behind the
  head, so the GPU is still running when the next token's first buffer is
  committed. −0.018 / −0.023.
- **The indexer RoPE and FP4 quantize in one batched dispatch** (IRQ):
  −0.034 / −0.049.

**Result (October 20, 00:08 UTC):** 24.920 / 24.942 ms per token,
**40.129 / 40.093 tok/s**, every treatment run above 40. Every lever on top
of the pair + EADD composition was bit-exact against its control. Release
quality qualification of the whole stack is still open.

## What did not work

All of these were measured. Exact ones that did not pay earned zero credit;
numerical ones that failed a screen were not timed further.

- **Routed gate/up lookup through `simd_shuffle`:** exact, but +66 µs per
  layer (slower in 48 of 48 batches).
- **Integer reconstruction of the E2M1 FP4 values** in the gate/up and down
  paths: three forms failed the correctness screen before timing; the fourth
  was exact and neutral.
- **Paired-nibble FP4 bit decoder:** exact, +0.062 ms slower.
- **Planes byte layout for the MXFP4 experts,** built once to files and
  attached file-backed: exact and loadable, but neutral in the decoder. The
  full layout also put the loader against the host's memory limit.
- **FFN reordering** (shared and routed work side by side): +0.055 to
  +0.96 ms slower.
- **Router selection folded into the router's completion:** one form slower,
  one neutral. A later fused form was exact only with a relaxed device-atomic
  handoff and cost +95 µs.
- **Two-row Q8 query, query q_b in FP8, query RoPE from the table,
  head-argument literals, KV address reuse, round-once at the producer, a
  64-thread HC join:** each slower, neutral, or too small to keep.
- **Down projection with four lanes per block:** neutral, and on quality hold.
- **Splitting the first command buffer, a final-handler callback latch, an
  event-first terminal wait:** the latency each targeted was not there to
  recover (about 10 µs exposed after the handler, against a 40 µs bar).
- **An Engram row cache:** 30.7% of rows recur, but a token stops waiting
  only when all 24 of its rows hit, which happens on 8.7 to 14.1% of tokens.
  It was not built.
- **A hoisted paired norm** (one letter of the single-group latency
  kernels): not exact under fast math, 302 of 71.68M values. Removed.
- **A guarded MMA index scorer at depth:** exact, but index scoring rose
  from 1,638 to 1,752 µs/token at 300K.

## From the campaign tree to this branch

After the 8K milestone, the work moved to depth, serving and a public build:

- **Depth.** Tarjei Mandt's candidate-block index scoring (kernelpool), a
  512-wide winners' sort extending his radix top-k select, and building the
  candidate list once. Against the earlier depth curve (38.60 / 37.95 /
  36.53 tok/s at 64K / 131K / 300K), this gives **39.70 / 39.07 / 38.02**,
  +2.8% / +3.0% / +4.1%.
- **Prefill.** Ivan Fioravanti's exact prefill indexer MMA: 889 against 841
  tok/s cold at 64K (+5.7%, bit-exact).
- **Public weights.** The narrow copies are re-encoded at load from the
  public MXFP4 GGUF. On the pinned fixtures the logits and state are the same
  as with the original-checkpoint sidecars. `gguf-tools/deepseek41_prepare.py`
  goes from the nine public split files to the engine's GGUF and drafter,
  pinned by digest.
- **Disk KV cache v2.** Entries are bound to the engine identity, a numerics
  canary and a payload hash. A 300K conversation that took 391.6 s to first
  token cold resumes in 11.2 s after a restart. A restart test found an
  older bug that deleted a checkpoint on shutdown; it is fixed.
- **Against DwarfStar main as servers** (same GGUF and flags, no DSpark):
  decode is 2.06 to 2.16 times faster from 2K to 64K, and cold TTFT is 15 to
  35% lower.
- **DSpark:** 57.70 tok/s on a code answer, against 52.61 on the production
  build.

## Credits

- **Ivan Fioravanti:**
  - the E4M3 shift decode that the blended readers adapt;
  - the fixed-shape (literal-shape) decode mechanism;
  - the narrow decode copies built at load;
  - the exact prefill indexer MMA.
- **Tarjei Mandt (kernelpool):**
  - the V4.1 work this stack builds on (native MXFP4 conversion and
    experts, verification kernels, the GPU Markov chain; see
    THIRD_PARTY-V41.md);
  - candidate-block index scoring;
  - the radix top-k select.
- **antirez:** DwarfStar, the engine all of this is built on.

Their commits on this branch keep their authorship.
[THIRD_PARTY-V41.md](../THIRD_PARTY-V41.md) lists the other attributed work.

## Where the evidence is

- [RECEIPTS-V41-1017.md](RECEIPTS-V41-1017.md): every measurement behind a
  commit or a README number on this branch, by window.
- Each commit message gives its own what, why and measured effect.
- The September work: [V41_M3ULTRA.md](V41_M3ULTRA.md) and
  [RELEASE-MXFP4-20260918.md](RELEASE-MXFP4-20260918.md).
