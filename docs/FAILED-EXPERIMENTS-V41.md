# Failed experiments: batched decode for V4.1 Flash

This log keeps the ideas that lost, were withdrawn or were measured wrong while
we built batched decode for DeepSeek V4.1 Flash: several sessions decoding
together in one step, DSpark verify blocks sharing that step with other
sessions, expert weights read once for the rows that share them, attention tiles
derived from oMLX's kernels, and a layered prefill that lets decoders run
between its chunks. Each entry says what was tried, what it was expected to gain
and why, what it measured, and why it was dropped. The ideas that won are
described in [METAL-V41.md](METAL-V41.md). Several kernels here were ported from
[oMLX](https://github.com/jundot/omlx), which ships batched V4.1 decode on Apple
Silicon; where a port lost, the entry says what it was measured against.

## How to read it

**Regime.** Unless an entry says otherwise, every number is milliseconds per
decode step, measured in ds4-bench's resident job runner (one process loads the
model once and flips switches between jobs), with sessions restored from saved
caches at short depth (2,000 to 3,000 tokens), in serpentine order (A B B A A
B), median of three, on a quiet machine at thermal pressure 0, with DSpark off.
Lower is better. A delta such as "+0.45 ms" is the idea against the build
without it, so a plus sign means the idea was slower. An **arm** is one
configuration in such an A/B run.

**N** is the number of sessions decoding together in one step. N=3 is listed
first everywhere, because three streams is the main use case: a chat and two
agents. A **row** is one session's token in a step. A **block** is a DSpark
verify block, six rows of one session. A step with a block and other sessions'
rows is a **mixed step**.

**Prompt sets.** "Short" is eight saved conversations of 2,000 to 3,000 tokens.
"Band" is 34,000 to 48,000 tokens, "131K" and "300K" are saved sessions of that
depth. "oMLX-method" prompts are built the way oMLX's admin benchmark builds
them: a unique short prefix, then the start of one shared text, cut to 1,024
tokens. "Distinct" prompts are 1,024 tokens of different texts. "Same text" puts
one prompt in every row, the best case for rows sharing experts.

**Verdicts.**
- **dead:** do not retry without a new reason.
- **superseded:** something else took its gain.
- **enabler:** no gain alone, kept for what it allows.
- **partial:** won at some N and is gated to them.
- **invalid:** the measurement was wrong, not the idea.
- **withdrawn:** dropped by a decision, not by a number.
- **open:** the result is still owed.

"Retest" notes say when an entry deserves another look, because a kernel that
loses at four rows may win at eight.

## Rows on the matrix units and in registers

The batched rows' dense matmuls run on register kernels through four rows
(ROWS_REG) and on exact eight-row simdgroup-matrix tiles from five (BATCH_MMA).
These entries are the alternatives.

**Exact eight-row tiles at two to four rows.** *Dead below five rows.* One
kernel for every row count would have been simpler, and the exact eight-row MMA
kernel was already written. On a microbenchmark of the 1,280 to 32,768
projection at four rows, the tiles took 118 µs, the register kernel 84 and the
older four-row kernel 147: tiles lost to registers at every row count and to the
older kernel at two and three. The stage times said the same with the fused tail
live: output-B 4.08 ms at five rows and 4.29 at eight on the tiles against 3.85
at four on registers, the query 4.90 against 4.62, the low projection 3.17
against 2.76. One full-step A/B settled it: +1.8 ms at N=3, +2.1 at N=2, +0.50
at N=4, outputs identical. The 8x8 tile is the hardware's shape, so a tile with
rows unused does the same multiplies; the gap narrows toward four rows and
crosses by five, which is where BATCH_MMA starts.

**A row-count test inside the tile loop.** *Fixed.* The first tiles for five to
eight rows guarded each load with `!ra ? 0 : xa[k]`. That one test in the hot
loop cost 4 ms a step at N=8 (99.4 against 95.4). Choosing the pointer once,
outside the loop, removed it: a tile row past the row count repeats row 0 and is
never read.

**Outputs per threadgroup by row count.** *Dead past the chosen values.* Fewer
threadgroups means fewer launches, but each lane then keeps more sums. Taken:
two outputs per threadgroup at two rows, three at three and four. Lost: six at
two rows, +0.20 ms; two at four rows, +0.14 ms (sampled, 128 tokens). Two at
three rows and three at two rows tied with the chosen values.

**Row-block tiles of 4 and 16 per launch.** *Dead at N=2-4. Retest at 6-8.* The
routed expert grids visit every pair at one row block before the next
(BATCH_RBLOCK). Tiling that order in groups of 4 or 16 blocks might have kept
more in cache. It measured -0.05 / -0.09 ms at N=3, -0.08 / -0.03 at N=2 and
-0.10 / -0.13 at N=4: noise.

## Engram reads for rows

V4.1 reads 24 rows from each of its two Engram tables per token, from disk.
Batched rows read the narrow (FP8) copy with one register kernel at every row
count (BATCH_NARROW_ENGRAM), and start the second table's reads as soon as the first
has joined (BATCH_READS).

**One-row narrow Engram kernel, run once per row.** *Superseded.* One-row decode
already had a fast narrow reader. Run once per row it loses the shared weight
pass: the Engram stage went from 1.16 to 1.74 ms at N=4.

**Narrow Engram rows in two passes.** *Superseded.* A rows kernel over the
narrow copy paid at four rows (-0.21 ms) and lost above: +0.32 ms at N=5, +0.69
at N=8. A threadgroup holds four rows at eight simdgroups, so a fifth row starts
a second pass that reads the weights again. One register kernel that takes every
row in one pass replaced it (sampled at temperature 1): -0.11 ms at N=3, -0.08
at N=2, -0.27 at N=4, -1.1 at N=5, -1.0 at N=6, -1.0 at N=7, -0.61 at N=8, each
row equal to its one-row run.

**Rows' Engram reads on each session's parked readers.** *Dead.* One-row decode
reads its Engram rows on a session's own parked reader threads; batched rows
read them on one worker. Handing each row's reads to its session's parked
readers was expected to win back 0.5 to 0.7 ms at N=2-3. It lost at every N:
+0.81 ms at N=3, +0.40 at N=2, +0.88 at N=4, +1.3 at N=5, +1.7 at N=6, +2.0 at
N=7, +2.6 at N=8, with idle time at N=8 rising from 2.3 to 3.6 ms. Waking a set
of parked threads per session is slower than one worker doing the reads, and the
cost grows with rows.

**A row's Engram reads spread over workers.** *Dead; the premise was wrong.* At
N=8 the command buffer before layer 1 waited about 0.67 ms on Engram reads.
Spreading each row's reads over workers measured -0.05 ms at N=8 and nothing
elsewhere. Each row's 24 reads were already above the parallel threshold and
fanned out. Host timing then showed the join takes 2.3 ms at N=8 and 0.63 to
0.97 at N=3, about 0.33 ms a row.

**One read batch per table for all sessions.** *Dead.* Fewer, larger read
batches at N=8 instead of one per session. With 16 readers: +0.52 to +0.55 ms at
N=6 and N=8; with 32: +0.08 to +0.32; within 0.1 at N=3 and N=4. Outputs equal.
*Open:* at N=8 the buffer before layer 1 still waits on the Engram reads (0.54
ms median, 0.87 mean, on the timeline), and what serializes them is not known. It shows only
at N=8.

## Fusion and overlap that did nothing

**The hyper-connection mix beside q_a and kv.** *Dead.* The mix is short and
could hide beside any short dense stage. Beside q_a and kv it saved about 0 at
N=2-4. Beside q_b it pays (BATCH_MIX: 1.1 ms at N=3, 0.79 at N=2, 2.0 at N=4),
because q_b is the stage that leaves the GPU idle.

**Replaying the tail in waves.** *Dead.* The tail of each layer (copies, sum,
rounding, norm) runs about six small passes per row. Running every row's k-th
pass together saved 0.02 ms at N=3, 0.00 at N=2 and 0.05 at N=4. The passes are
too small for their launches to matter.

**Expert selection inside the routed pair kernel.** *Dead on paper.* Once the
fixed-shape row kernels (BATCH_STATIC) removed the plan dispatch, what was left
of the router was its own pass, 0.37 ms for the top-k plus 0.73 for the weights,
flat with N. Selection inside the pair kernel could only remove a barrier. Not
built.

## Attention

Attention per row is the one cost rows cannot share: each session has its own
keys. Batched rows run attention in waves (BATCH_WAVE: every row's k-th dispatch
together), and the default kernel is now the tiled attention ported from oMLX
(ATTN_TILES, not bit for bit; `--exact` restores the vec kernel).

**A row axis in the attention kernels.** *Dead on paper.* One dispatch per stage
across all sessions, instead of one per row in waves. One session's vec stage
already fills the GPU: overlapped in waves it keeps 78% of its time at N=4, and
vec costs 0.93 to 1.12 ms a row against 1.39 for one-row decode. The small
stages are launch-bound at 0.1 to 0.2 ms each and already overlap. A row axis
would remove launches only, under 0.15 ms at N=3, for a new kernel per stage and
per-row buffers. Not built.

**Rows out of layer lockstep.** *Dead on paper.* Staggering rows so one row's
attention hides under another row's weight stream. Rows that leave lockstep lose
the shared weight passes, and the saturated stages (vec and the routed experts)
are the bulk of the step, so little could sit beside them. Not built.

**Sixteen partials instead of 32 in the attention port.** *Invalid.* Fewer
partials looked like -7.8 ms at N=8 on distinct prompts, -0.43 on oMLX-method
prompts, +0.25 at N=1. The reduce's 32 lanes read past the 16 partials, so
garbage reached every row, routing collapsed, and the FFN ran under its byte
floor. The 32-slot layout stayed; only the splits that have keys are launched
and reduced (LIVE_SPLITS).

## Expert reads shared across rows (read-once)

Rows that pick the same expert now read its gate and up weights once, with one
load feeding one accumulator per row (READ_ONCE): on DSpark blocks at every N,
and on plain rows from N=4. The road there:

**The dedup kernel on cross-session rows.** *Superseded by read-once; its first
verdict's reason was withdrawn.* MOE_DEDUP reads each expert once for the rows that picked it, inside
a DSpark block. On cross-session rows it lost to the fixed-shape kernels even on
its best case, 1,024 tokens at N=4: +2.8 ms on distinct rows, +1.5 on same-text
rows. The first explanation, that rows share too few experts, was wrong. Each
simdgroup issued its own loads, so a shared expert was read once only in name,
through the cache. Forcing every row onto row 0's six experts saved just 1.9 ms
at N=3, 1.9 at N=4 and 2.6 at N=8, and at N=8 the FFN section went from 49.6 to
46.2 ms when every pair read a single expert, 1/48 of the bytes. Memory
bandwidth was not the limit; every row's loads still arrived. One load and one
decode feeding K accumulators, on the same forced overlap, gave the real figure:
38.7 to 32.7 ms at N=3, 31.0 to 28.2 at N=2, 48.0 to 40.9 at N=4, 79.6 to 66.0
at N=8. Same-text rows had been the slowest case at N=8 (88.4 against 82.4 on
short), so overlap read through the cache costs, it does not pay.

**How much there is to share.** Repeated experts per layer, of 6N slots: real
131K rows repeat 3.9% at N=3, 5.7% at N=4, 14.0% at N=8; short prompts 28% at
N=8; oMLX-method prompts on one shared code text 59% at N=8. Rows drawn from
replayed real agent turns repeat 6.2% at N=3 and 4.8% at N=2. A six-row DSpark
block repeats 31 to 34%. So blocks are read-once's main use, and plain rows at
N=2-3 stay off it: their repeats sit below the plan's break-even (10.3% at N=3,
13% at N=2). On the final build, plain rows at N=3 measured -0.06 ms on
oMLX-method prompts and +0.06 on distinct, and N=2 +0.02 and +0.16. *Partial:*
on plain rows from N=4.

**Sizing from real repeats.** *Invalid as a sizing.* The base build against
pair-only sharing on prompts with real repeats gave 5.7 ms at N=3, 7.0 at N=4,
14.6 at N=8. That includes cache sharing the base already had on real repeats.
Forced overlap with nothing shared in cache gives 3.8, 5.2 and 11.9.

**A one-thread plan.** *Replaced.* After the router, one GPU thread walks the 6N
(row, expert) pairs and writes the groups, expected to cost about 0.2 ms. It
cost +0.69 ms at N=3, +0.45 at N=2, +1.0 at N=4, +1.7 at N=6, +2.7 at N=8: a
lone GPU thread spends about 60 ns a compare, and stores to device memory one by
one. A thread per pair, counting its own company, costs +0.20 at N=3, -0.04 at
N=2, +0.26 at N=4, +0.18 at N=6, +0.33 at N=8 on distinct rows. That is what the
saving has to clear on a text with no repeats.

**`reassociate(off)` to make the group kernel exact.** *Dead; the real fix was
load order.* The group kernel gave every row a rounding difference from step 1,
tokens still equal at N=2-3. `#pragma clang fp reassociate(off)` did not help
and was far slower (same text at N=8: 109 against 86 ms). The library builds
with fast math, which orders each sum by where its terms are defined; with the
weights decoded before the inputs were loaded, the order changed. Loading the
inputs first gives the one-row kernel's bits: 0 of 1,600 logits differ, same
text, N=2-8.

**Groups of up to eight rows.** *Capped.* A group of any size was expected to
save at the same rate per shared read. On forced overlap at N=8, four groups of
two saved 6.9 ms, two groups of four 7.9, one group of eight 2.9; same text at
N=8 saved 1.5 to 2.6 ms with groups of eight against 7.1 to 7.7 in groups of
four. Exactness needs the inputs defined before the weights, so a group of eight
holds 128 input floats live. A non-exact form with late inputs saved only 3.8,
so register pressure is not the whole story. Groups are capped.

**One pipeline per group size.** *Replaced by one launch.* No size switch inside
the kernel, one launch per group size present. Launches cost more than the
switch: splitting only the lone pairs into 2, 4 or 8 launches cost +1.1, +3.0
and +3.8 ms at N=8, and +1.1 and +1.6 at 2 and 4 launches at N=4. One list in
one launch replaced it.

**Lone pairs writing their own list entries.** *Plan replaced.* Fewer reads for
pairs that share nothing. Lone pairs got cheaper (+0.36 against +1.04 ms at N=8
when nothing is shared), but groups got worse: -1.7 against -2.6 ms on short
prompts at N=8, -2.8 against -4.0 on a block beside two other sessions. With the
plan fixed, the kernel forms tied (-2.40, -2.37, -2.33). Lone pairs raced to the
front of the list and the groups ended last: the order inside one launch
matters.

**Sharing the expert down projection too.** *Not built.* On top of the pair
stage, priced from expert ids alone before any kernel cost. Two forms: (a)
exact, sharing only the repeats that keep every row's order of slots, which is
68 to 84% of them; (b) reassociated, sharing all. Net saving: a block beside two
other sessions (a) 0.04 to 1.3 ms, (b) 0.64 to 2.2; real rows at N=4 (a) 0.19 to
0.37, (b) 0.28 to 0.48; real rows at N=8 (a) 0 to 1.0, (b) 0.42 to 1.6. Exact
stays the default, so (b) was out of scope, and (a) needs a merge-order solver
on the GPU in every layer for at most about 1 ms.

## Ports of oMLX's kernels

oMLX's batched V4.1 decode was the reference. These are the ports that lost; the
FP4 decoding without a table (next section) and the attention tiles won.

**Inexact simdgroup-matrix dense stages, our own form.** *Withdrawn as
evidence.* oMLX runs its dense stages on matrix tiles from four rows up. Our
first inexact tile path at all four dense sites was slower at every N and not
exact: +1.8 ms at N=3, +1.7 at N=4, +1.0 at N=5, +3.2 at N=8 (N=2 not run). That
measured our implementation, not a limit, so the literal port below replaced it
as the evidence.

**oMLX's dense tiles, ported literally on Q8_0.** *Lost.* The expectation was
oMLX's own cost on the same machine. At N=8 on oMLX-method prompts, per stage
against our exact kernels: q_a and kv +0.89 ms, q_b -0.10, o_a +0.53, o_b +0.97,
shared expert -0.08, all of them +1.6. At N=4: +0.82, -0.22, +0.90, +1.1, -0.25,
all +1.8. N=3 was not measured. Kernel time at N=8, tile against our exact MMA:
q 5.27 against 5.01 ms, expand 4.34 against 3.75, low 3.14 against 3.27. Beside
the attention port it added nothing. Our N=8 kernels are already exact MMA
tiles, and at N=4 register kernels beat padding-free tiles. oMLX's own 10.1 ms
for these stages came from a profile that runs one stage twice, and the second
run finds the weights still in the system cache, so it is not a clean figure.

**Two-byte-aligned weight loads in those tiles.** *No change that matters.* Q8_0
blocks sit two bytes past a word, so the tiles used byte loads; perhaps that was
why they lost. With aligned loads they still lost by 0.66 to 0.97 ms at N=4 and
0.39 to 1.2 at N=8.

**oMLX's routed FFN, ported literally.** *Lost.* The FFN is the largest stage,
and oMLX's profile could not rule it out. The port: up and down kernels, the
shared expert inside the up dispatch, and top-k inside every up threadgroup.
Deltas in ms against our kernels, oMLX-method / distinct prompts, DSpark off,
each column adding one piece:

| N | up | down | up + down | + shared | + router |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | +0.64 / +0.52 | +0.15 / -0.24 | +0.67 / +0.35 | +0.68 / +0.34 | +0.53 / +0.16 |
| 1 | +0.45 / +0.44 | -0.24 / -0.24 | +0.19 / +0.19 | | |
| 4 | +0.49 / +0.66 | +0.13 / -0.08 | +0.63 / +0.45 | +0.78 / +1.6 | +1.6 / +1.1 |
| 8 | +2.4 / +1.1 | +0.40 / -0.27 | +3.0 / +0.76 | +3.7 / +2.0 | +3.8 / +1.2 |

Its up kernel gives our bits exactly, so its loss is pure shape: four rows a
simdgroup against our two, and at N=8 no read-once, where oMLX's axis order only
gets cache hits. The shared expert inside the up dispatch loses the concurrent
overlap ours has. Top-k in every threadgroup reads 384 scores each time against
one 0.37 ms pass. Down alone is mixed by prompt set and not exact (mean KL
0.0035, no top-1 flip in 32 steps), so it was dropped too.

**oMLX's thread shape on our routed kernels.** *Dead.* On top of FP4 without a
table: four rows a simdgroup instead of two, eight simdgroups a threadgroup
instead of one, and its 32-row block. Four rows a simdgroup: +0.24 ms at N=3,
-0.08 at N=4, +4.0 at N=8. Eight simdgroups: +0.22 at N=3, +0.32 at N=2, +0.19
at N=4, -0.19 at N=8. Both: +1.1 at N=3, +0.66 at N=4, +4.3 at N=8. Outputs
identical, no gain. Its shape was not its edge; the FP4 table and the barrier it
does without were.

## FP4 decode without a table

The routed experts' kernels used to turn FP4 codes into floats through a table
in threadgroup memory, with a barrier per threadgroup to fill it. Decoding with
shifts, as oMLX does, and no table (FP4_NO_TABLE) gave -1.5 ms at N=3, -0.62 at
N=2, -2.4 at N=4 and -6.0 at N=8 on oMLX-method prompts, exact. Getting it exact
took these:

**The decoded dot left to fast math.** *Fixed.* The first port gave every row a
difference from step 2, tokens split at steps 5 to 8, and rows against one-row
decode failed 1,086 of 1,536 checks. Fast math flattened the table form's sum
into one chain of 16 terms ordered by where they are defined. On synthetic
weights, that flat chain matched 256 of 256 and 14 other trees none. Written out
with `fma` and `reassociate(off)`, the chains are equal.

**FP4 words from three aligned loads and shifts.** *Dead.* Fewer loads than
eight byte loads: +0.62 ms at N=3, +0.71 at N=4, +5.1 at N=8. The byte loads
already merge.

**Raw unaligned word loads.** *Invalid.* Two word loads a lane looked like -4.4
ms at N=3 and -5.6 at N=4. Every step repeated the same token: unaligned device
loads returned wrong data here, and every row routing alike is fast.

**Where the 2^-14 scale goes.** *Fixed.* Deferring the scale to the dot passed
13 of 13 checks on the real model but failed the unit test: the sum kernel had
19 of 5,120 floats one ulp off, on test activations that are not bf16 and reach
far smaller values than the model's, near the flush to zero. Scaling the
activations at load gave 0 off, but in the pair kernels the read-once groups
then scaled each row's input in every lane, 3.1 ms at N=8. The final form scales
the dot in the pair kernels and keeps the activations scaled in the sum kernels.
The 19 values were not traced lane by lane.

## DSpark verify blocks inside a batch

A session's verify block now leads a shared step and the other sessions follow
as plain rows (MIXED_STEP), one block a step, rotating. Verify runs on the
batched rows' kernels (VERIFY_STACK) with decode's attention (VERIFY_DECODE), so
its rows carry serial decoding's bits, and a rejected block's correction is its
stream's row in the next step (MIXED_CARRY).

**The correction as its own one-row step.** *Replaced.* Keeping the mixed step
simple: after a rejection, the correction runs as a one-row step beside the
others. Every other session then waits a whole step for one token: by arithmetic
about 7 ms of a 10.6 ms margin at 131K and N=3. Carrying the correction into the
next shared step took rejected blocks from 115.6 to 91.1 ms (-24.5) with
accepted blocks unchanged, an implied +1.4% at three streams (replayed real
turns, DSpark on; the replay's warm start was flawed, but block rows were not
affected by it).

**Verify attention on decode's kernels, without per-row picks.** *Fixed.* Exact
at 4,096 to 150K tokens, but 96 of 96 rows differed at 256 and 1,024. Rows with
at most top_k visible keys get their key ids in ascending order, which suits the
batch's later sort, while decode sums in the row's pick order. Below (top_k + 1)
× ratio keys, which is 2,052 at the largest ratio of 4, a verify block now takes
the per-row pick, and it is exact from short prompts through 150K.

**Exact verify, proven on one text.** *Fixed.* Exact verify was bit-identical at
eight depths of one text. At 131K and N=3, one row on another text (repeated
code) differed, in solo verify too. Past 16,384 visible keys, decode's radix
select orders exactly tied scores by id; the batched block sort left ties in its
network's order, so a block through repeated text summed its keys in another
order. Ordering ties by id in the block sort moved a 131K block step from 70.9
to 71.1 ms, within noise. Exactness needs repeated text, not more depths on one text: the rerun
covered 42 saved sessions, from short to 300K and repeated code at two seeds,
with 0 differing rows.

**Two blocks in one step.** *Dead on paper.* At N=3 and an acceptance of 0.9,
one block computes to 82 t/s, three blocks of 3, 3 and 2 rows to 68, and plain
rows run 74 (DSpark on). So at most one block a step.

**Drafting beside three other sessions.** *Dead at N=4 on paper.* From the
measured step costs and real acceptance (93% of drafted tokens accepted, 20% of
drafts declined before verify), exact verify at N=4 computes to -1.1% on short
contexts, +1.8% at 131K and +0.9% at 300K against plain rows (DSpark on). Blocks
do not join steps of more than three sessions.

**Verify on the rows' kernels with batched attention.** *Not the default.*
Keeping the batched indexed attention inside the verify block is faster than
decode's attention: 70.1 against 72.5 ms per verify on replayed real turns with
one stream, an implied +1.1% on three-stream stretches. It is not bit for bit,
and exact verify stays the default.

## Prefill beside decoders

A cold prompt arriving while other sessions decode used to freeze them. It now
runs as a layered prefill that parks between chunks, so the decoders take a step
at each park. The worst stall for three decoders beside a cold 9,240-token
prompt went from 2.7 s to about 0.35 s, and it is about 0.8 s beside a 126K
prompt.

**Larger mixed-prefill chunks alone.** *Dead.* Bigger chunks finish the prompt
sooner, and the expectation was that decoders would still get a token every
couple of seconds. Server level, three decoders at about 2,700 tokens beside a
cold 9,240-token prompt, no drafting: decoders ran 1.37 t/s at 128-token chunks,
and 0.79, 0.40 and 0.27 at 512, 2,048 and 4,096, with median gaps between tokens
of 1.3, 2.7 and 4.8 s against 0.73 s. A chunk's wall time is 0.55 s plus 1.02 ms
a token, and the 0.55 s is one sweep of all the experts, paid per chunk however
short. So chunk size alone freezes decoders harder as it grows.

**A time share with smaller chunks.** *Superseded.* Decoders get as much wall
time as each chunk took. With 512- and 1,024-token chunks the prefill took 46.7
and 30.8 s against 22.8 s at 2,048, decoders ran 8.7 and 8.6 t/s against 8.1,
and the worst gap was 1.4 and 1.8 s against 2.7. 2,048 with the time share
became the fallback (`DS4_SERVER_DISABLE_PREFILL_YIELD=1`); the layered prefill
replaced it as the default.

**Taking turns at chunk level with short requests.** *Withdrawn.* Counting
waiters on the inference lock fixed a real fault: a parked sweep could not see a
thread queued on the lock and retook it ahead of the waiter, so a title request
beside two cold prefills took 129 to 154 s. With waiters counted it took 8.4 to
17.8 s, but a 15-token prompt still needed 7.7 to 14.5 s: a short prompt
prefills layer by layer, and it took turns with the sweep's wide chunks at every
layer. Letting a short prefill skip the turn-taking brought it to 0.6 to 0.9 s.

**Sync width.** *Lost; 32,768 kept.* One cold 125,915-token prefill alone, at
the deployment's settings: 8,192-token syncs, the old idle path's width, ran 791
t/s; 32,768 ran 868; the whole prompt in one sync ran 883, 1.7% faster, but
stored no checkpoint during the prefill, so a later request that diverged
partway through that prompt had nothing to restore from inside it.

## Stop, then resend

*Withdrawn.* When a user presses Stop and resends, the server restores the last
stored checkpoint and prefills from there, where a normal continuation keeps the
live session. Time to first token, server, DSpark on, temperature 1, top_p 1: a
normal continuation (n=6) against Stop then resend (n=4) was 0.3 against 1.4 s
at 4K, 0.4 against 2.5 s at 130K and 0.6 against 5.0 s at 368K. No checkpoint
sat at the earlier prompt's end, so 122 to 1,808 tokens were replayed each time,
and the resend always landed on another slot.

The fix built: V4.1's compressed rows and index keys are append-only, so a log
of the raw and drafter-feature ring rows written during a generation is enough
to rewind a cancelled slot to exactly what the client received. A rewound
session matched one that never ran past that point, with 0 differing logits and
0 differing state bytes, over odd and even positions, ring wrap, log wrap, 1 to
3 rows and 150K tokens; a refused rewind left the session intact. Starting the
log cost 1 to 3 ms and a rewind about 1.5 ms, and decode with the log on stayed
within 0.1 t/s of off (DSpark on, short and deep; DSpark off, short). Not
measured: the log at N>1, the server table after the fix, a Stop during
reasoning, and a Stop beside another stream. A few seconds after a Stop was
judged not worth the code, and it was not merged.

## Host-side gaps

**A 0.5 to 1.5 ms gap in the server between steps.** *Dead.* Sampled at four
streams, the server's step log read 54.61 ms against 55.0 for the whole cycle: a
gap of about 0.36 ms, not worth a design.

**The length of the GPU bridge after the final buffer.** *Dead.* A short spin
queued behind each step's final command buffer keeps the GPU busy while the host
hands off to the pre-encoded next step. Full, half or no bridge: within 0.15 ms
at every N, and a later try at three lengths within 0.13. The gap before the
first buffer is the host's hand-off; the bridge never holds back the next step.

**Pre-encoding the next step for batched rows.** *Partial: kept, small.* Batched
rows sat 0.5 to 0.7 ms idler than one-row decode at the first command buffer,
which one-row decode encodes during the previous token's wait (PREENCODE). The
same for rows (BATCH_PE) was expected to close that gap. Alone it gave 0 at N=3,
+0.19 at N=2 and +0.05 at N=4-8: the idle just moved to the second buffer, which
waited on the Engram join. With the second table's reads started earlier
(BATCH_READS) it gave 0.13 ms at N=3, 0.10 at N=2, 0.07 at N=4 and 0.09 at N=8
greedy, 0.09 and 0.06 at N=3 and N=2 sampled. Rows already commit their first
buffer about as fast as one-row decode does with the slot; what the slot removes
for one row is its own layer-0 encoding, which rows had not been waiting on.

## Measurement mistakes worth knowing

- **A profile with a sync per stage.** The first stage table for batched decode
  committed and waited after every stage. It added 48 ms a step, hid all
  overlap, and said attention cost 3 to 5 ms a row; the real figure was 2.09,
  the same as one-row decode. oMLX's per-stage profile with an evaluation at
  every stage had the same fault: its stages summed to 85.3 ms for a 47.3 ms
  step at N=3 and 102.8 for 70.8 at N=8. Take stage shares from an encoder
  timeline or from running one stage twice at full duty, and decide with
  wall-clock A/B runs.
- **An encoder timeline as a measure of savings.** The timeline pays a pass per
  dispatch, so it overstates what removing dispatches saves: 7.0 ms at N=4 on
  the timeline, 3.6 ms of wall time. Use it for shares, not savings.
- **A section that measured zero.** A five-to-eight-row arm measured nothing,
  because a `rows > 4u` test and a buffer sized for four rows kept the section
  shut; the first search had looked for `<= 4`. A section that measures zero now
  gets a timeline before a verdict.
- **Wrong data that looks fast.** Unaligned loads, the 16-partial reduce, and a
  group-size switch with no case for one pair (that pair's output was never
  written) each produced the fastest arm of their run. When every row routes the
  same way the step gets faster. Hash every row of every arm before reading a
  number.
- **A value stored unrounded.** Rows 5 to 8 differed from one-row decode because
  an exact kernel stored a mid value unrounded and the overlapped consumer read
  it as stored. The timing was valid; the result was not until the kernel
  rounded at store.
- **A counter read mid-step.** An expert counter that committed in the middle of
  the step ran layer 1's buffer before the Engram rows joined, so same-text
  rows, whose tokens are identical, differed from layer 1 on. Copy counters in
  the stream and read them at step end.
- **Restored source mtimes.** Copying sources with `rsync -a` or `cp -p` kept
  their old modification times, so `make` saw them older than the objects and
  skipped the rebuild. A test then ran new Metal against a stale binary. Touch
  restored files, or build with `make -B`.
- **A context change at N=8 in one process.** Changing the context size frees
  eight sessions and allocates eight new ones, and the two sets overlap:
  anonymous memory went from 37.6 to 105.0 GiB and back within 4 s, and the
  compressor from 7.0 to 16.2 GiB. It looked like the model being mapped through
  a private copy, but the only model mapping was shared. Keep one context size
  per process.
- **Saved caches are bound to their context size.** A 300K session saved at one
  context cannot be extended at another; the 400K sessions needed a fresh
  prefill, 492 s each.
- **Thermal pressure.** One stage was marked at thermal pressure 2, and in
  another one repetition ran about 1.5 ms slow in every arm; both were rerun.
  The runner now waits for pressure 0 before each cell.
- **A scratch buffer that hid from warm servers.** Batched rows each get their
  own copy of the process-wide attention scratch. The first form of the waves
  shared the top-k scratch and failed the hash, which was the test working.
  Later, copies were chosen by buffer label, and a DSpark verify block, the
  first call large enough to regrow one buffer, gave it another label. From then
  on every row's attention partials landed in one buffer, and streams took each
  other's tokens: 29 bad tool calls in 259 across the arms that drafted, none in
  the two that did not. Six arms in three configurations that shared one warm
  cache directory reproduced nothing (0 errors in about 350 calls), most likely
  because a warm server's short prefills had already grown the buffer under its
  old label. A bisect at about 10% errors per call read 0 on two switches, one
  by chance. Copies are now chosen by buffer identity as a row binds it; 270
  calls after the fix gave 0 errors.
- **Replays through the disk KV cache.** Three replayed clients drawn from one
  conversation, each prompt a prefix of the next, ran no three-row step (155
  two-row steps), so the test said nothing about three streams; clients now
  carry distinct first lines. Large re-prefills in that run were first read as
  one client taking another's slot. They were a small disk budget evicting
  checkpoints, one of them 40 s before it was needed. A live slot can be taken
  only when its whole content is a prefix of the new prompt, as in a fork, and
  then the loser restores from its own checkpoint in 3.4 to 4.7 s. Seed
  checkpoints copied in before the server started were evicted at startup under
  that small budget: copy them after the server is ready, and give each arm a
  new directory, because a shared one hands later arms the earlier arms'
  checkpoints.
- **A pressure test the cache could not see.** Filler checkpoints for an
  eviction test had names the cache's scan skips, so they applied no pressure.
- **A noise yardstick for accuracy.** One-bf16-step nudges of serial decoding
  give a yardstick for how far a non-exact kernel moves the output. Three nudges
  were too few: their range was narrower than each arm's own paired standard
  error, and 16 nudges spread 2.5 times wider. Nudges of 1/8 and 1/64 of a step,
  meant as a control that rounds away, did not stay bit-identical: the values
  feed the next pass as floats, so even tiny changes cross a later rounding.
