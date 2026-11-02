# DeepSeek V4.1 Flash on Metal: why each group is faster

This page explains the speed groups of this branch, one section per group,
grouped by phase and in the order they were added on top of DwarfStar main. Each section says what the
group changes, why that saves time, and what it measured when it was turned on.

Every group has one off switch, `DS4_METAL_DISABLE_V41_<GROUP>=1` (two groups
use `DS4_METAL_MODEL_TRACKED=1` and `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1`). With a switch set, that part of the
engine runs main's code again. [RUNTIME-V41.md](RUNTIME-V41.md) lists every
switch; [REPRODUCE-V41.md](REPRODUCE-V41.md) says how to measure them.

After the one-session groups, [Several sessions at once](#several-sessions-at-once)
explains the groups that decode several sessions in one step. The page ends
with the image input groups and one speed-up that was identified and sized but
not built.

Some of the code here started in forks by Ivan Fioravanti and Tarjei Mandt,
and three pieces of the batched decode started in oMLX;
[THIRD_PARTY-V41.md](../THIRD_PARTY-V41.md) has the details.

## Where the time goes

A decode step turns one token into the next. For each layer it reads that
layer's weights, runs attention over the context and runs the mixture of
experts, which picks six of 384 routed experts plus one shared expert. The work
per token is small, so most of the time is not arithmetic. It is one of four
things, and every group below attacks at least one of them:

- **Bytes moved.** Reading weights and the KV cache from memory. Fewer or
  narrower bytes mean a shorter step.
- **Kernel launches.** Every Metal dispatch has a fixed cost, and a dispatch
  that only rounds or copies a few kilobytes is almost all fixed cost. Fusing
  two kernels into one also removes the round trip of their intermediate
  result through memory.
- **GPU occupancy.** A kernel that runs on one threadgroup uses one of the M3
  Ultra's 80 GPU cores. Spreading the same work over more threadgroups, or
  running independent work side by side, keeps the cores busy.
- **Idle time.** Moments when the GPU waits for the host (encoding, a disk
  read, a decision) or the host waits for the GPU.

At 8K context on the M3 Ultra, main decodes a token in about 50.2 ms (19.9 t/s)
and this branch in about 24.2 ms (41.4 t/s). At 131K the step goes from about
55.9 ms (17.9 t/s) to about 24.7 ms (40.5 t/s).

## How the steps were measured

The numbers in the sections are the steps of the README's progression table,
measured in one quiet session. One process loads the model once, and for each
step turns on more groups, in commit order, then restores a saved prompt cache
from memory and decodes 128 greedy tokens twice; a step's speed is the mean of
the two passes, which agreed to within 0.1 t/s except where a section says
otherwise. Serial decode restores main's own 8K and 131K prose caches; DSpark
restores an 8K prose and an 8K code cache that carry the drafter window. The
drafter is loaded in every run. Groups with little or no effect alone were
measured together with their neighbours, and such a section gives the shared
step and names the groups in it. MODEL_TRACKED is decided at load, so its step
is measured against main's own binary.

Each serial step also checked that the output was unchanged: a digest of the
tokens and the full logits of every one of 128 greedy steps, at both depths,
matched main's from the same caches in the same session.

Prefill steps are cold 32K prefills of the prose prompt in a fresh session,
without the drafter, one run per step, starting from this branch with every
group off (45.0 s, the same as main). The logits after every prefill step
matched main's.

---

## Feeding the GPU

### MODEL_TRACKED: skip hazard tracking on the weights

`DS4_METAL_MODEL_TRACKED=1` turns it off.

Metal can track hazards on every buffer: before a kernel runs, the driver checks
whether an earlier kernel still writes to anything it reads. The model's weights
are only ever read by the GPU, so that check protects nothing and costs driver
time on every dispatch. Main already had `DS4_METAL_MODEL_UNTRACKED=1` to skip
it for any model; V4.1 now skips it by default.

Kind: launch overhead. Step, against main's own binary: 8K 19.9 to 20.2 t/s,
131K 17.9 to 18.2. Its prefill step is measured together with the decode and
DSpark groups through PUBLIC_CONTROLLER: 45.0 to 44.9 s.

### COMPUTE_TENSOR_COPY: keep small copies in the compute encoder

`DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1` turns it off. It applies to every model.

A tensor copy used to be a blit, and a blit has to close the open compute
encoder and open a blit encoder, then a new compute encoder after it. DSpark's
verify and rollback issue many small copies per cycle, so the GPU kept switching
encoders for a few kilobytes each time. A copy whose addresses and length are
16-byte aligned, outside a concurrent section and without overlap, now runs as
a small compute kernel in the encoder that is already open. The bytes moved are
the same.

Kind: launch overhead. Step: 8K 20.2 to 20.6 t/s, 131K 18.2 to 18.6;
DSpark prose 19.2 to 19.6, code 30.1 to 31.3.

### DECODE_QUEUE: one command stream per decode step

`DS4_METAL_DISABLE_V41_DECODE_QUEUE=1` turns it off.

On one Mac, main encoded a layer, committed it, waited for the GPU to finish it,
and only then encoded the next. So the GPU sat idle while the host encoded, and
the host sat idle while the GPU ran. Now a step's layers queue into one command
stream that is committed after layer 0 and then every two layers. The GPU starts
on the first layers while the host is still encoding the later ones, and the
output head follows the last layer in the same stream. DSpark's verify blocks
and short prefill batches commit every two layers the same way.

This is the largest single step on the branch: at 8K it took a token from about
48.6 ms to 36.2 ms, almost all of it idle time.

Kind: idle time. Step: 8K 20.6 to 27.6 t/s, 131K 18.6 to 24.3; DSpark prose
19.6 to 23.1, code 31.3 to 32.0.

### ROUTER_SELECT: pick the experts in one dispatch

`DS4_METAL_DISABLE_V41_ROUTER_SELECT=1` turns it off.

Each layer's router turns 384 logits into six chosen experts and their weights.
Main did that in several small dispatches: probabilities, the biased top six,
then the normalised weights. One kernel now does all three. A DSpark verify
block of up to eight rows gets the same kernel per row, after one matrix-vector
product for all rows' logits. The results are the same bits.

Kind: kernel launches. Step: 8K 27.6 to 29.2 t/s, 131K 24.3 to 25.5; DSpark
prose 23.1 to 23.9, code 32.0 to 32.6.

### ENGRAM_ASYNC: read the Engram rows while the GPU works

`DS4_METAL_DISABLE_V41_ENGRAM_ASYNC=1` turns it off.

V4.1's Engram tables stay on disk. Each decode step needs 2 x 24 rows of them,
one set at layer 1 and one at layer 14. On Metal, main read them before the
step's GPU work could start, so every token began with the GPU waiting on the
SSD. DwarfStar already had a reader that runs these reads on background threads
and waits for each table only at the layer that uses it, but only CUDA used it.
This group grew in four steps:

1. Decode uses that reader on Metal too.
2. Batch steps (a DSpark verify block of up to six rows, or a short prefill)
   read every row's Engram rows one after another before encoding any GPU work.
   Now each row starts its reads on a worker, the first layers are encoded
   meanwhile, and each table joins just before its layer. On a code prompt this
   removed about 30 ms of idle GPU time in front of each six-row verify block,
   which made it the largest DSpark step.
3. Main's reader keeps four workers, each reading six of a table's 24 rows one
   after another, so the GPU waited for the last of six disk reads about 0.4 ms
   into each token. On Apple the reader now keeps one worker per row, and a
   table is ready after a single read.
4. With the queued command stream, the rows now join right before the command
   buffer that holds their layer, not before the host encodes that layer, so
   the host encodes layer 1 while the reads finish. Batch steps do the same,
   and the second table's rows get their own storage so the GPU no longer
   drains before layer 14.

Steps 2 and 3 were found with main's encoder timeline
(`DS4_METAL_ENCODER_TIMELINE`), which shows each command buffer's GPU start and
end, so the gaps were visible as gaps.

Kind: idle time. Step, all four parts together: DSpark code 32.6 to 39.8 t/s,
prose 23.9 to 25.4; serial 8K 29.2 to 29.2, 131K 25.5 to 25.1 (the two
131K passes of this step read 25.5 and 24.7, so the serial change is within
the spread).

### FUSED_ROUNDING: round to bf16 where the value is produced

`DS4_METAL_DISABLE_V41_FUSED_ROUNDING=1` turns it off.

V4.1 keeps its activations at bf16 precision between operators. On Metal every
producer wrote float results and a separate kernel then rounded them, so most
steps of a decode layer paid an extra dispatch and an extra trip of the
activation through memory. Now the producers round as they store: the Q8 and
F16 matrix-vector products, the weighted RMS norm, the hyper-connection sums
and the hyper-connection expand. DwarfStar's CUDA backend already folded these
roundings into its kernels; this does the same on Metal. The results are the
same bits.

Kind: kernel launches. Step: 8K 29.2 to 30.6 t/s, 131K 25.1 to 26.5; DSpark
prose 25.4 to 26.3, code 39.8 to 39.9.

---

## Decode kernels

### NARROW_WEIGHTS: read narrow copies of three weight families

`DS4_METAL_DISABLE_V41_NARROW_WEIGHTS=1` turns it off.

Three decode products read weights stored wider than their values need. The
router gates are F32 tensors whose low halves are all zero, so they are really
BF16 values. The Engram projections and the indexer query projection are F16
tensors whose values are FP8 E4M3 codes at a power-of-two scale per 32 x 32
block. When the model loads on one Mac, the engine builds a BF16 copy of each
router and an E4M3 copy (one byte per value, one scale byte per block) of each
Engram and indexer projection, after checking that every value converts
exactly. Decode then reads the copies: half the bytes for the router and the
F16 tensors. A tensor that does not convert exactly keeps its wide read.
Nothing new is stored on disk. The optional `--v41-sidecar` files take the
place of these copies when they are given.

Kind: bytes moved. Step: 8K 30.6 to 31.5 t/s, 131K 26.5 to 27.2; DSpark
prose 26.3 to 26.6, code 39.9 to 39.9.

### QKV_FUSION: the attention input projections together

`DS4_METAL_DISABLE_V41_QKV_FUSION=1` turns it off.

Each layer's attention starts with two projections of the same normalised row
(q_a and kv), two RMS norms, the query projection and a RoPE pass over the query
heads. The two input projections now run as one product that reads the row
once, both norms run in one dispatch, and the query projection applies RoPE to
its outputs as it stores them.

Kind: kernel launches. Step, measured together with ROPE and FFN_JOIN: 8K
31.5 to 32.8 t/s, 131K 27.2 to 28.1; DSpark prose 26.6 to 27.3, code 39.9
to 40.6.

### ROPE: compute the rotation once, apply it inside its neighbours

`DS4_METAL_DISABLE_V41_ROPE=1` turns it off.

Each layer applied RoPE in separate passes: on the new cache rows before
quantizing and storing them, on the indexer query, and in reverse on the
attention heads before the output projection. Each step now computes its RoPE
sines and cosines once into a small table. A cache row is roped, quantized and
stored in one dispatch; the indexer query is roped and quantized in one
dispatch from the table; and the attention reduce writes the heads already
rounded and inverse-roped. That removes three passes per layer and recomputes
no sine or cosine.

Kind: kernel launches. Step: measured together with QKV_FUSION and FFN_JOIN
(see QKV_FUSION).

### FFN_JOIN: finish the layer in one dispatch

`DS4_METAL_DISABLE_V41_FFN_JOIN=1` turns it off.

At the end of each layer, the routed experts' sum and the shared expert's
output were added, rounded, expanded into the four hyper-connection streams and
rounded again, in separate passes, and the next layer's stream weights were then
copied out of that result. Now the add, both roundings and the expand run in one
dispatch, and the next layer reads the weights where the expand left them. Only
the last layer still copies them, for the output head.

Kind: kernel launches. Step: measured together with QKV_FUSION and ROPE (see
QKV_FUSION).

### HC_FUSION: hyper-connection mixing inside the kernels next to it

`DS4_METAL_DISABLE_V41_HC_FUSION=1` turns it off.

V4.1 carries four residual streams (hyper-connections). Each block's input took
six passes: a norm of the four streams, the mix projection, its split, the
weighted sum, its rounding and the block norm. One kernel now does all six. The
attention output projection and the shared expert's down projection write their
results straight into the four streams, so the separate expand passes are gone.
On a decode step the attention block's mix also moves into the query
projection's dispatch, where it runs beside the query tiles instead of before
them.

This was the largest kernel step on the branch: at 8K a token went from about
30.5 ms to 27.6 ms, mostly dispatches that each did very little work.

Kind: kernel launches. Step: 8K 32.8 to 36.2 t/s, 131K 28.1 to 30.6; DSpark
prose 27.3 to 28.9, code 40.6 to 41.1.

### SPLIT_REDUCE: two serial reductions over more of the GPU

`DS4_METAL_DISABLE_V41_SPLIT_REDUCE=1` turns it off.

The Engram gate gave each of the four streams a single 32-lane group, and the
hyper-connection collapse ahead of a deferred mix used a single 1,024-thread
group, so each pulled its data through one GPU core while the others waited.
Now the gate takes one 256-thread group per stream, and the collapse runs as 32
groups that each sum a slice, followed by 10 groups that apply the norm. Both
keep the exact order of the original sums.

Kind: occupancy. Step, measured together with SHARED_EXPERT and CONCURRENT_FFN:
8K 36.2 to 37.5 t/s, 131K 30.6 to 31.5; DSpark prose 28.9 to 29.5, code
41.1 to 41.6.

### SHARED_EXPERT: gate, up and SwiGLU in one dispatch

`DS4_METAL_DISABLE_V41_SHARED_EXPERT=1` turns it off.

The shared expert ran its gate and up projections as two products that each
rounded, then a separate SwiGLU pass. One kernel now computes both projections
over the same activation reads, rounds them and applies the SwiGLU, for one
token and for up to eight verify rows.

Kind: kernel launches and bytes (the activation is read once instead of twice).
Step: measured together with SPLIT_REDUCE and CONCURRENT_FFN (see SPLIT_REDUCE).

### CONCURRENT_FFN: the shared expert beside the routed experts

`DS4_METAL_DISABLE_V41_CONCURRENT_FFN=1` turns it off.

The shared expert ran between the router and the routed experts, one after the
other. Main already runs it beside the routed experts for other models; V4.1
now does too. Its gate, up and SwiGLU run beside the routed gate and up kernel,
and its down projection beside their sum, in one concurrent encoder. The routed
kernels leave cores free, so the shared expert fills them instead of taking its
own turn.

Kind: occupancy. Step: measured together with SPLIT_REDUCE and SHARED_EXPERT
(see SPLIT_REDUCE).

---

## Decode at depth

At long context, attention is preceded by the sparse indexer: it scores every
compressed key and keeps the top 512. That work grows with the context, so it is
what separates the 8K and 131K speeds. These four groups barely move 8K and do
most of their work at 131K.

### INDEX_SCORE_WIDE: one key per thread for long rows

`DS4_METAL_DISABLE_V41_INDEX_SCORE_WIDE=1` turns it off. GLM uses the same
kernel.

Main's decode scorer spent one threadgroup on every compressed key, each loading
the query again. Rows of 8,192 keys or more now use a wide scorer: each
threadgroup loads the query once and each thread scores one key. Past layer 20,
a masked form reads only the blocks layer 20 kept.

This group is not bit-exact: the wide scorer adds the heads in a different
order, so its scores differ from the direct kernel's in the last bits and a key
near the cut-off could change. In the measured runs the selections did not
change and the logits matched main.

Kind: bytes moved and occupancy. Step, measured together with INDEX_BF16: 8K
37.5 to 37.6 t/s (8K has 2,048 keys, below the wide scorer's threshold), 131K
31.5 to 34.1; DSpark prose 29.5 to 29.6, code 41.6 to 41.8.

### INDEX_BF16: score from a bf16 copy of the keys

`DS4_METAL_DISABLE_V41_INDEX_BF16=1` turns it off.

The index keys are rounded to bf16 when they are made but stored as f32, so the
scorer read twice the bytes it needed. Every key is now also written to a bf16
copy as it is stored, and the scorers read the copy. The values are the same, so
the scores are too. The copy costs 64 floats of memory per compressed row. A
restored session rebuilds it and checks every key is exact in bf16.

Kind: bytes moved. Step: measured together with INDEX_SCORE_WIDE (see
INDEX_SCORE_WIDE).

### TOPK: a radix select for the top 512

`DS4_METAL_DISABLE_V41_TOPK=1` turns it off.

Main picked the top 512 by sorting blocks of scores and merging them in a
ladder whose last rounds run on two or three threadgroups, so at long context
most of the GPU waited for those few. A row of 16,384 scores or more now uses a
radix select: three histogram passes over the score bits find the 512th score
across the whole GPU, the winners are written out and sorted. Shorter rows keep
the ladder with a faster block sort and a shorter merge.

The selected scores are the same, rank by rank. Keys with exactly equal scores
can trade places, which real index scores almost never have.

Kind: occupancy. Step: 8K 37.6 to 37.9 t/s, 131K 34.1 to 36.6, about 2.1 ms
per token at 131K; DSpark prose 29.6 to 29.9, code 41.8 to 42.2.

### INDEX_CANDIDATES: later layers score only layer 20's blocks

`DS4_METAL_DISABLE_V41_INDEX_CANDIDATES=1` turns it off. It needs
INDEX_SCORE_WIDE and TOPK on.

Past layer 20, once the context has more than 2,048 blocks of 8 keys, only the
2,048 blocks layer 20 kept can be selected, but decode still scored every key
and picked from the whole row. Now the list of kept blocks is built once per
token, the scorer reads only those keys, and one threadgroup runs the radix
select over those 16,384 scores. The picks are the same bits.

Kind: bytes moved. Step, measured together with DECODE_ATTENTION: 8K 37.9 to
38.9 t/s (INDEX_CANDIDATES has no filtered rows at 8K), 131K 36.6 to 38.1;
DSpark prose 29.9 to 30.4, code 42.2 to 43.3.

### DECODE_ATTENTION: stage the selected rows straight from the cache

`DS4_METAL_DISABLE_V41_DECODE_ATTENTION=1` turns it off.

Decode attention copied the 512 selected compressed rows out of the cache into
a separate buffer, and the attention kernel's staging pass then converted that
copy to f16. The staging kernel now reads the selected rows from the cache by
their ids and converts them as it goes, so the copy, its buffer traffic and its
dispatch are gone. The staged values are the same.

Kind: bytes moved and kernel launches. Step: measured together with
INDEX_CANDIDATES (see INDEX_CANDIDATES).

### FIXED_SHAPES: kernels built for V4.1's exact sizes

`DS4_METAL_DISABLE_V41_FIXED_SHAPES=1` turns it off.

Main ran V4.1's products on general kernels that take every size as an
argument, so loop bounds, strides and tails are computed at run time. V4.1
Flash's sizes never change, so the products with fixed sizes now use kernels
with those sizes written in: the q_a and kv pair, the attention output's low
projection, the projection into the hyper-connection streams and the FFN join.
Products with a short inner size (1280 into 32768 for the queries, 2304 into
5120 for the expert down projections) use a kernel where each SIMD group
computes one output row, which keeps more of the GPU busy on a short sum. The
routed experts now also reach main's fixed-route MXFP4 decode kernels with
V4.1's trip counts written in. Each kernel walks the same inputs in the same
order, so the results are the same bits.

It saved about 1.4 ms per token at 8K.

Kind: occupancy and per-dispatch overhead. Step: 8K 38.9 to 41.1 t/s, 131K
38.1 to 40.3; DSpark prose 30.4 to 31.2, code 43.3 to 43.3.

### SAMPLER_SCAN: the sampler's first pass in NEON

`DS4_METAL_DISABLE_V41_SAMPLER_SCAN=1` turns it off. It applies to every model.

Before sampling from the 129,280-entry vocabulary, the sampler walks every
logit once to count finite values and find the largest. That was a scalar loop
on the CPU, run while the GPU waits for the next token. It now runs in 16 NEON
lanes with the same result, signed zeros and ties included. Greedy decoding
never runs this loop. On a peaked vocabulary the call went from 674 to 555 µs.

Kind: idle time (host work on the critical path), sampled decoding only. Its
step was measured together with ROW_MATMULS (see ROW_MATMULS); greedy decoding
never runs this loop.

---

## DSpark verify

DSpark drafts up to five tokens with the checkpoint's own drafter, then the
target checks them in one batched step of six rows. These groups make that
six-row step cheaper. One-token decoding never takes these paths, so serial
output does not change.

### ROW_MATMULS: read the weights once for a few rows

`DS4_METAL_DISABLE_V41_ROW_MATMULS=1` turns it off. It applies to every model.

When two to eight rows go through the same Q8_0 or F16 matrix, each row ran in
its own threadgroups and every threadgroup read the weights again, so a six-row
verify read each weight six times. One threadgroup now handles all the rows over
weight lines it keeps in cache. Each row keeps the single-row kernel's order of
operations, so the results are the same bits.

Kind: bytes moved. Step, measured together with SAMPLER_SCAN: DSpark prose
31.2 to 32.4 t/s, code 43.3 to 48.2; serial unchanged.

### ROWS_MMA: three to eight DSpark rows on simdgroup matrices

`DS4_METAL_DISABLE_V41_ROWS_MMA=1` turns it off.

With the row kernels, a few rows read each weight once but still multiply one
row at a time. Now the three to eight rows of a DSpark block, the drafter's
seed and stages and the verify block, go through the GPU's simdgroup matrix
units: each SIMD group dequantizes a 32-wide block once and multiplies it
against all rows at once. The shared expert's gate, up and SwiGLU get the same
form.

This group is not bit-exact: the sums run in a different order, so verify rows
can differ from serial decoding in the last bits of their logits. It runs only
inside DSpark: a prefill batch keeps the row kernels, or takes SHORT_MMA at
eight rows, so serial output never depends on it.

Kind: arithmetic throughput. Step, measured together with VERIFY_KERNELS:
DSpark prose 32.4 to 33.7 t/s, code 48.2 to 53.9; serial unchanged.

### VERIFY_KERNELS: verify rows through the fused decode MoE kernels

`DS4_METAL_DISABLE_V41_VERIFY_KERNELS=1` turns it off.

Main sends routed-expert batches of up to five rows through the fused decode
kernels. A six-row verify block on MXFP4 experts fell through to the general
batch kernels instead. MXFP4 experts are now admitted to the fused decode
kernels for six to eight rows, so each verify row runs the experts as one-token
decode does.

This changes DSpark's output compared with main: every verify row now gives the
same bits as that row decoded on its own, instead of the batch kernels' bits.
Only verify blocks take this path; a prefill batch of six to eight rows keeps
main's MoE kernels.

Kind: kernel efficiency. Step: measured together with ROWS_MMA (see ROWS_MMA).

### PUBLIC_CONTROLLER: draft only where drafting pays

`DS4_METAL_DISABLE_V41_PUBLIC_CONTROLLER=1` brings back main's scheduler.

Main's DSpark drafts at every position and cuts each block at the first draft
position below 60% confidence. On prose, where few drafts are accepted, that
makes generation slower than plain decoding. This group measures what drafting
costs. It times serial tokens and takes the median of the last nine, and it
judges every three drafted cycles together against what the same tokens would
have cost serially. A window that saves time keeps drafting. Losses carry over
until they exceed half a serial token; then drafting stops for 16 serial tokens,
then 32, 64 and 128. A request decodes its first 16 tokens serially, so the
first window is judged against a measured cost. A block is verified only when
its first three draft positions are each at least 75% confident; otherwise it is
declined, which costs the drafter and one serial token instead of a six-row
pass. The policy is in `ds4_deepseek41_admission.h`.

This changes when DSpark drafts, not what greedy decoding produces. It applies
on one Mac; across two Macs main's cut-off decides.

Kind: wasted work. Step: DSpark prose 33.7 to 40.3 t/s (close to serial
decoding's 41.2: it backs off where drafting loses), code 53.9 to 52.5 (the
cost of the declined and judged cycles on a prompt where drafting pays).

### MOE_DEDUP: read each selected expert once for a few rows

`DS4_METAL_DISABLE_V41_MOE_DEDUP=1` turns it off. It applies to any MXFP4 model.

Verify rows often pick some of the same experts. Metal's pair kernel walks each
row's six experts on their own, so an expert that two rows picked is read twice.
For two to eight MXFP4 rows, a small planning dispatch now lists the distinct
experts, and one threadgroup per distinct expert serves every row that picked
it. Each expert's gate and up weights are read once. Each row keeps the same
arithmetic, so the results are the same bits.

Kind: bytes moved. Step, measured together with PREFILL_INDEX, PREFILL_ATTENTION
and PREFILL_FOLDS: DSpark prose 40.3 to 40.3 t/s, code 52.5 to 52.6.

### VERIFY_BATCH: a verify block attends as one batch

`DS4_METAL_DISABLE_V41_VERIFY_BATCH=1` brings back the per-row route.

Main verifies a draft block through the batched layer step, but each row still
attended on its own: a rotary pass, a staged flash attention, an inverse rotary
pass and a rounding pass per row and per layer, and the attention output
projections went through the general kernels. Now a session's consecutive rows
attend together. A verify block's rows store their keys and pick their index
selection in order, exactly as their own decode steps would, then attend in one
dispatch per group of rows that see the same number of selected rows. A short
prefill of two to eight tokens takes DwarfStar's batch attention. For both, the
attention low projection runs every row in one pass over its weights, and the
output and shared-expert down projections write straight into the
hyper-connection streams, two to eight rows per weight pass. From three or four
rows a verify block's projections use the matrix kernels when ROWS_MMA is on,
and an eight-row prefill uses SHORT_MMA's.

Kind: kernel launches and bytes moved. Step, measured together with
FLASH_HEAD_SHARE: DSpark prose 40.3 to 40.2 t/s, code 52.6 to 54.0; serial
unchanged. Its prefill step, together with MOE_DEDUP and FLASH_HEAD_SHARE: cold
32K 40.9 to 40.9 s.

### FLASH_HEAD_SHARE: eight heads share their key reads

`DS4_METAL_DISABLE_V41_FLASH_HEAD_SHARE=1` turns it off. It applies to any
model's flash attention of several rows.

When several rows attend over their keys, the flash attention kernel gave each
head its own threadgroup, and all the heads of a row read the same keys
separately. For two or more rows, each head now takes one SIMD group over the
whole key walk, and eight heads share a threadgroup, so they share the cache
lines of the keys. Each head keeps its own order of operations.

Kind: bytes moved. Step: measured together with VERIFY_BATCH (see VERIFY_BATCH).

### VERIFY_INDEXED: verify like a short prefill

`DS4_METAL_DISABLE_V41_VERIFY_INDEXED=1` brings back the row-by-row verify.

With VERIFY_BATCH, each verify row still picked its own index selection and
attended with the decode kernels. Now the block attends as a short prefill does:
one batched index top-k for all rows and one indexed attention pass over all of
them. The index scores come from the bf16 key rows on simdgroup matrices, and
blocks of two to seven rows use the tiled scorer that larger batches already
use. So that a commit can still rewind to any row, each row keeps the pooling
state its own decode step would leave.

This is the batched verify, and it is on by default. Its verified logits are no
longer the bits of serial decoding, so where two tokens are nearly tied, greedy
DSpark output can differ from serial output. Strict DSpark (`--dspark-strict`)
does not draft and stays equal to serial decoding. With this switch set, the
verify is row by row again and its tokens are serial decoding's.

Kind: kernel launches. Step, the last step, measured together with SAMPLER_GPU,
the TP groups and ENGRAM_READERS, none of which change one-Mac greedy decoding:
DSpark code 54.0 to 59.0 t/s, prose 40.5 to 40.5.

---

## Prefill

Prefill processes the whole prompt layer by layer, thousands of rows at a time,
so its kernels are large and the costs are different: arithmetic, barriers, the
glue kernels between big products, and the Engram disk reads at the start of
each sweep. The steps below are cold 32K prefills with no drafter loaded.

### PREFILL_INDEX: index scores on simdgroup matrices

`DS4_METAL_DISABLE_V41_PREFILL_INDEX=1` turns it off.

During prefill each compressed layer scores every new row against all its index
keys. DwarfStar's tiled kernel stages the queries and keys in threadgroup memory
and synchronizes the threadgroup for every head. The new kernel keeps each SIMD
group's slice of the keys in registers for all 32 heads and reads the queries
straight from memory, so the head loop runs with no barriers. It does the same
products in the same order, so the scores are the same bits. It runs for eight
or more rows.

Kind: barriers (idle lanes). Step: cold 32K prefill 44.9 to 43.5 s.

### PREFILL_ATTENTION: sixteen rows per barrier

`DS4_METAL_DISABLE_V41_PREFILL_ATTENTION=1` turns it off.

DwarfStar's prefill attention for V4.1 staged one key row at a time in
threadgroup memory and synchronized for each, while its decode kernel staged
sixteen rows per barrier. Prefill rows of the ratio-1 and ratio-2 layers now use
a copy of the decode kernel that keeps each block's selected rows as a 16-bit
mask instead of an array of row ids, which leaves registers free for more
threadgroups. The rows, their order and the arithmetic are unchanged.

Kind: barriers and occupancy. Step: cold 32K prefill 43.5 to 42.2 s.

### PREFILL_FOLDS: fold the glue into the producers

`DS4_METAL_DISABLE_V41_PREFILL_FOLDS=1` turns it off.

On Metal, DwarfStar rounded prefill activations in a separate pass after each
producer: after the hyper-connection sums and expands, the RMS norms and the
add of the shared and routed experts. It also embedded the prompt one token at a
time and ran its F16 projections one row at a time. For prefills of more than
eight rows, the sums, expands, norms and add now write bf16 directly; the
prompt's embeddings are gathered in one pass and repeated into the four streams
by one kernel; and the F16 projections take two rows per threadgroup, sharing
each weight read. Each folded kernel keeps the arithmetic of the kernels it
replaces.

Kind: kernel launches and bytes moved. Step: cold 32K prefill 42.2 to 40.9 s.

### PREFILL_MATMUL: unpack the Q8 weights once

`DS4_METAL_DISABLE_V41_PREFILL_MATMUL=1` turns it off.

DwarfStar's Q8 tiled product can first unpack a weight matrix to F16 and then
run the F16 tiles, which pays the dequantization once instead of once per tile
of tokens. It did that by default only for Qwen on the M3 Ultra. V4.1's prefill
products of 32 or more rows on the M3 Ultra now do it too. The F16 weights are
exactly the values the Q8 tiles would dequantize, so the results are the same
bits. Main's `DS4_QWEN4_Q8_PREFILL_UNPACK` still overrides it.

Kind: arithmetic (repeated dequantization). Step: cold 32K prefill 40.9 to
40.7 s.

### PREFILL_INDEX_BATCH: pick candidates and top-k as batches

`DS4_METAL_DISABLE_V41_PREFILL_INDEX_BATCH=1` turns it off.

During prefill, only rows that could see 1,024 or more compressed keys sorted
their index scores as a batch; the rest sorted one row at a time. Now every
batch of two or more rows sorts as one batch from 513 visible keys. Rows that
see at most 512 keys take all of them in one dispatch. Layer 20's candidate
blocks and the later layers' filters run once for the whole batch; rows that see
at most 2,048 blocks keep every block, so their candidate selection is skipped.
Ratio-1 keys are published in batches, as ratio-2 keys already were.

Kind: kernel launches. Step: cold 32K prefill 40.7 to 40.4 s.

### SHORT_SWEEP: one sweep for 3,072 to 8,191 tokens

`DS4_METAL_DISABLE_V41_SHORT_SWEEP=1` turns it off.

Prefills of 4,096 tokens or more run as layer sweeps: each layer processes every
token before the next starts, and the decoder layers from 20 on rebuild only the
rows the last token depends on. Below 8,192 tokens that pruning was skipped, and
the sweep was cut to whole 2,048-row tiles, so the remainder ran through a
second pass. Now sweeps start at 3,072 tokens, a sweep below 8,192 keeps its
final partial tile, and the pruning starts as soon as the prompt is longer than
the rows it keeps (2,541, or 2,668 with DSpark). Fewer rows go through the
decoder layers and there is no second pass.

This is a scheduling change. Each row's arithmetic is unchanged: cold prompts of
3,072, 4,096, 6,144 and 8,000 tokens give main's logits bit for bit (checked
with `--exact`, so that CED and MMA attention do not take part).

Kind: wasted work. Step: measured together with ENGRAM_READAHEAD and the decode
groups after it: cold 32K prefill 40.4 to 40.4 s. A 32K prompt is longer than
the prompts this group changes.

### ENGRAM_READAHEAD: read the next sweep's rows during this one

`DS4_METAL_DISABLE_V41_ENGRAM_READAHEAD=1` turns it off.

A long prefill runs several sweeps, and each sweep reads its Engram rows from
disk before layer 1 and layer 14 can use them. Once layer 14 has consumed a
sweep's last rows, the reader is free while the remaining layers run. The sweep
now hashes the next sweep's tokens and starts reading the next sweep's first
table into the prefetch buffer. The next sweep uses those reads when its token
ids match and drops them otherwise. Only the timing of the reads changes.

Kind: idle time. Step: measured together with SHORT_SWEEP (see SHORT_SWEEP).

### ENGRAM_READERS: 32 readers for a sweep's rows

`DS4_METAL_DISABLE_V41_ENGRAM_READERS=1` goes back to 16 readers.

A prefill sweep needs its 4,096 tokens' Engram rows from disk before layer 1,
and the first sweep has nothing to hide that read behind. With 16 readers the
first sweep of an 8K prefill waited about 0.9 s for its rows; 32 concurrent
readers halve that wait. Decode keeps 16. The rows read are the same.

Kind: idle time. Step: cold 32K prefill 40.4 to 37.9 s, with the same logits.
With every group on, a cold 131K prefill takes 151.6 s, against 189.4 s with
every group off.

### SHORT_MMA: eight prefill rows on simdgroup matrices, in main's order

`DS4_METAL_DISABLE_V41_SHORT_MMA=1` turns it off.

On one Mac the smallest prefill batch is eight rows: an eight-token prompt, or
an eight-token tail of a longer one (shorter prompts are decoded one token at a
time). Main runs each of that batch's Q8_0 products row by row, so each row
reads every weight again. Outside a DSpark block those eight rows now go
through simdgroup matrices at seven sites per layer: the q_a, kv and q_b
projections, the attention output's low projection, output-B and the shared
expert's down projection with their hyper-connection expansion, and the shared
expert's gate, up and SwiGLU.

It is faster for three reasons. A threadgroup of eight SIMD groups covers eight
outputs of all eight rows, so each weight block is read once instead of eight
times. Each 8-wide piece of a row's sum becomes one matrix multiply of an 8x8
tile, started from zero, which gives exactly the bits of the row kernel's own
chain of eight multiply-adds; the row kernel's scaling, its pairwise lane sum
and its second sum across SIMD groups then run in the same order, with
reassociation turned off for that tree. And an eight-row batch fills the 8x8
tile, so no padding rows are computed. The result is main's bits: the unit test
checks eight rows of every V4.1 Q8_0 shape against main's kernels, and an
eight-token prompt gives main's logits and main's next 32 tokens.

Kind: bytes moved and arithmetic throughput. In isolation the seven sites take
13.0 ms per eight-row prefill, against 35.4 ms on main's
row kernels. Measured after the progression, against the previous head with
every group on (which ran these rows through ROWS_MMA and VERIFY_KERNELS, not at
main's bits): an eight-token prefill 0.3 s as before, now with main's logits;
cold 32K 37.8 s, under 0.1 s from before; DSpark code 59.2 t/s, within 0.1 t/s
of before and within noise.

---

## Prefill after the progression

These groups came after the progression table. Each was measured on its own,
against the build before it, on the M3 Ultra with the MXFP4 1017 weights. Text
figures are cold prompts or follow-ups (a prompt appended to a saved 8K prose
cache) without the drafter; image figures are the language model's prefill of
an image prompt in one reused session, with the drafter loaded.

### SUFFIX_TRIM: skip the rows no later row reads, inside a chunk

`DS4_METAL_DISABLE_V41_SUFFIX_TRIM=1` turns it off.

Layer `l` of a prefill sweep is read later only through the last
1 + (39 − l) × 127 rows (plus the drafter's window). SHORT_SWEEP already skips
whole 2,048-row tiles outside that range, but a shorter sweep, and every
follow-up turn, ran all its rows through all 40 layers. Sweeps of 1,024 to
8,191 rows now also cut inside the chunks: chunks before the cut are skipped,
and the cut chunk starts at a 128-row boundary of the sequence (the compressor's
blocks) and keeps at least 256 rows. Each kept row's arithmetic is unchanged. The
router's F32 product still picks the kernel the whole chunk would have used,
because its kernels give each row the same sums for any row count but switch
kernels by row count.

Kind: wasted work. Measured: follow-ups of 1,024, 2,048, 4,096 and 8,000 tokens
5.5%, 14.6%, 15.6% and 9.9% faster; cold prompts of 2,048 to 8,000 tokens 9 to
15% faster. Every case gave main's logits and main's next 16 greedy steps. At
512 rows the trim was 1 to 3% slower, so it starts at 1,024.

### PREFILL_MMA: prefill attention on simdgroup matrices

`DS4_METAL_DISABLE_V41_PREFILL_MMA=1` turns it off.

Every head of a prefill token reads the same window and top-k rows, one latent
row serving as both key and value. The new kernel gives a threadgroup 32 heads
of one token: sixteen SIMD groups, each eight heads by a quarter of the 512
dimensions, over 16-row blocks staged once as half, with one softmax rescale
per block instead of one per row. Q, K and V keep the scalar kernel's half
rounding and the sums stay in float, but they run in another order, so the
output is not bit for bit the scalar kernel's. It takes the rows
PREFILL_ATTENTION's kernel would (more than eight rows, ratio 1 and 2, one Mac)
on M1 to M4 GPUs; decode and DSpark's verify rows (eight or fewer) never take it.

Kind: arithmetic throughput. Measured: cold 8K 10.95 to 10.29 s, 32K 37.86 to
35.40 s, 131K 151.6 to 142.0 s. Accuracy: after each of 29 follow-ups of 1K
tokens (4K to 32K), the first decode step's full-vocabulary KL divergence
against the scalar kernel averaged 0.0102 on prose and 0.0007 on code, with the
same top token at all 29 points. Over 21 long scoring cases (DeepSeek's own
V4.1 continuations, teacher-forced), the mean negative log-likelihood per token
moved by +0.0004 ± 0.0036, within noise.

### CED: replay only a prompt's last rows through the decoder layers

`DS4_METAL_DISABLE_V41_CED=1` turns it off.

DeepSeek V4.1's decoder layers (20 to 39) read the global keys the encoder
layers publish, and only a 128-row sliding window of their own. Section 3.2.2
of the V4.1 report describes a causal encoder-decoder with bounded replay: the
prompt runs through the encoder layers in full, and only its last rows through
the decoder layers, their windows cut to the replayed rows. DeepSeek trained the
model for this mode. A sweep of 512 rows or more now replays its last 256 rows
through layers 20 to 39 (the report's recipe is 128; with 256 the last 128 see
full windows), instead of the exact dependency suffix of up to 2,414 rows. The
encoder's work is unchanged. A long prompt already runs only the encoder in its
earlier sweeps, so the saving comes once per prompt, in its last sweep.

Kind: wasted work. Measured: follow-ups of 1,024 and 2,048 tokens 2.997 to
2.166 s and 4.845 to 3.149 s; cold 8K 11.5% faster, 32K 3.5%, 131K 1.2%. With
PREFILL_MMA too, cold 32K takes 34.25 s and 131K 140.3 s. Accuracy, measured
as for PREFILL_MMA: mean KL 0.0064 on prose and 0.0004 on code, the same top
token at all 29 points; with PREFILL_MMA too, 0.0143 and 0.0009. NLL over the
21 long cases moved by −0.0001 ± 0.0009.

### IMAGE_CED: CED for prompts that hold images

`DS4_METAL_DISABLE_V41_IMAGE_CED=1` turns it off.

An image's rows enter the sweep through the embedding at layer 0 and travel
through the layers like text rows, and the decoder layers read the whole image
through the encoder layers' global keys. So a prompt with images now replays
through the decoder like any other: from 512 rows, its last 256 rows.

Kind: wasted work. Measured: a 3840x2160 screenshot (978 prompt tokens) 2.688
to 2.007 s, a 1400x900 screenshot (782) 2.319 to 1.817 s; a 512x507 photo (194)
is under 512 rows and unchanged. The first token's full-vocabulary KL
divergence against the full replay was 1.2e-4 and 1.4e-5, with the same top
token.

### IMAGE_TRIM: SUFFIX_TRIM for prompts that hold images

`DS4_METAL_DISABLE_V41_IMAGE_TRIM=1` turns it off.

The trim's cut applies to image rows unchanged. It runs only where CED does
not, that is with `--exact` or with CED switched off, on sweeps of 1,024 to
8,191 rows; one image is at most 1,024 rows, so it takes an image chat with some
text.

Kind: wasted work. Measured with `--exact`, image prompts after about 1,250
text tokens: a 512x507 photo 3.44 to 3.20 s, a 1400x900 screenshot 4.61 to
4.03 s, a 3840x2160 screenshot 5.39 to 4.74 s, each with the same logits, bit
for bit.

### MOE_SHORT_TILES: the large-prefill expert tiles for short prefills

`DS4_METAL_DISABLE_V41_MOE_SHORT_TILES=1` turns it off.

Five exact variants of the routed experts' grouped products (a compact 32x32
pair tile, a token-centric scatter, two tail culls and a half-result table for
the down projection) were the default only from 2,048 rows. A short prefill
gives each of the 384 experts only a few rows, so most of its 32-row tiles are
short, and the culls and the compact tile pay most there: a 978-row image prompt
gives an expert about 15 rows. V4.1's experts now take them from 9 rows.

Kind: wasted work. Measured: a 194-row image prompt 1.063 to 0.888 s, 978 rows
2.688 to 2.397 s; text of 218 tokens 1.201 to 1.008 s, 2,048 tokens 2.889 to
2.779 s; every logit the same.

### PREFILL_PIPELINE: commit one-chunk layers without waiting

`DS4_METAL_DISABLE_V41_PREFILL_PIPELINE=1` turns it off.

A prefill sweep committed every layer's command buffer and waited for it, so
the GPU sat idle while the host encoded the next layer. When a layer's rows run
as one chunk, its buffers now commit without waiting. The host still waits
before its own work on GPU memory: the Engram reads at layers 1 and 14 and
their read-ahead, the decoder start at layer 20 and the last layer. The same
kernels run in the same order.

Kind: idle time. Measured: 20 to 40 ms per prompt (a 3840x2160 screenshot
1.623 to 1.591 s, text of 1,002 tokens 1.915 to 1.884 s), every logit the same.

### HC_NARROW: the hyper-connection mix projection on narrow tiles

`DS4_METAL_DISABLE_V41_HC_NARROW=1` turns it off.

Each layer projects its four 5,120-wide hyper-connection streams onto 24 mix
weights, twice. A chunk of a few hundred rows made only 8 to 31 threadgroups of
the 64x32 tiled kernel, each walking the 20,480-wide input in 32-wide steps
with a barrier per step: about 0.75 ms per dispatch whatever the row count. One
SIMD group per 8 rows now owns all 24 outputs and keeps 128 inputs in flight
per step. Each output adds the same 8x8 half products in the same order.

Kind: occupancy and barriers. Measured: a 512x507 photo 0.776 to 0.738 s, text
of 4,096 tokens 5.032 to 4.981 s; every logit and decode digest the same.

### CTX_RESIDENCY: keep the engine's own buffers resident

`DS4_METAL_DISABLE_V41_CTX_RESIDENCY=1` turns it off.

The model's weights sit in residency sets on the command queue, but the
context, KV and prefill scratch buffers (12 GB at a 393,216-token context) did
not, so on every request the driver made them resident again when the first
command buffers used them: about 75 ms with the GPU idle. Every buffer the
engine allocates now joins one residency set on the queue. The arithmetic does
not change.

Kind: idle time. Measured through the server, first token of a one-token reply:
a 512x507 photo 1.105 to 1.050 s, a 3840x2160 screenshot 3.446 to 3.366 s, text
of about 800 tokens 1.508 to 1.425 s.

### MOE_HALF_X: the expert tiles read half rows

`DS4_METAL_DISABLE_V41_MOE_HALF_X=1` turns it off.

The compact expert tiles (gate and up with SwiGLU) narrowed each routed row
from float to half in every one of the 72 column blocks that read it. Each
layer now narrows the rows once, and the tiles read half rows: half the bytes,
the same half values.

Kind: bytes moved. Measured: a 3840x2160 screenshot 1.590 to 1.570 s, text of
4,096 tokens 4.947 to 4.892 s; every logit the same.

### CARRY_KEEP: keep a one-chunk layer's rows in place

`DS4_METAL_DISABLE_V41_CARRY_KEEP=1` turns it off.

A sweep stores each layer's rows (the residual, the hyper-connection weights,
the index selection and its mask) to the carry and loads them back when the
next layer starts. When both layers run their rows as one chunk at the same
offset, the batch buffers still hold those rows, so the next layer no longer
loads them. The stores stay, because the decoder and the next sweep read the
carry.

Kind: bytes moved. Measured: a 3840x2160 screenshot 1.590 to 1.573 s, text of
1,002 tokens 1.880 to 1.855 s; every logit the same.

---

## The decode handoff

### PREENCODE: encode the next token while this one finishes

`DS4_METAL_DISABLE_V41_PREENCODE=1` turns it off.

A decode step ends by committing its last command buffer, waiting for the GPU,
reading the logits and choosing the next token. Only then did the next step
start encoding its embedding and layer 0, so the GPU idled through that handoff.
Now, after committing the last buffer, the host uses the wait to encode the next
token's first buffer (the embedding, the RoPE table and layer 0). The embedding
reads the token id from a small device word. When the next step starts, it
writes the chosen token into that word and commits the held buffer at once. A
short GPU kernel committed behind the last buffer keeps the GPU busy across the
handoff. The kernels and their order are the same as without it. Any mismatch
(another graph, another position, a reset) drops the held buffer.

It applies to one-Mac resident decode with the queued head and the fused router.

Kind: idle time. Step, measured together with the prefill groups PREFILL_MATMUL
through ENGRAM_READAHEAD, which do not run in decode: 8K 41.2 to 41.4 t/s,
131K 40.3 to 40.5; DSpark prose 40.2 to 40.5, code 54.0 to 54.0.

### SAMPLER_GPU: summarize the logits for top-p on the GPU

`DS4_METAL_DISABLE_V41_SAMPLER_GPU=1` keeps sampling on the CPU.

Sampling at temperature 1 with top-p 0.95 starts by scanning all 129,280
logits on the CPU after the step finishes, while the GPU waits. Now the step
appends three small kernels after the output head that produce a summary: the
finite count, the maximum and its index, partial sums of exp(v - max) per 256
logits, and every candidate near the top 512, up to 2,048. The CPU checks the
summary, sorts the candidates and decides the nucleus from them. It decides only
where the summary's error bound cannot change the nucleus; otherwise the
unchanged CPU sampler decides. Either way the token and the random state are
the CPU sampler's.

A session encodes the summary only while it samples with exactly those settings
(temperature 1, top-p 0.95, no top-k, no min-p). Greedy decoding never encodes
it; encoding it on every token cost greedy decoding about 0.2% at 8K.

Kind: idle time, sampled decoding only. Not measured on its own: the
progression decodes greedily, where this group never runs.

---

## Several sessions at once

`ds4-server --batched-session N` keeps N sessions resident and decodes the ones
that are ready in one step: one token per session, one pass over the dense
weights for all of them. The groups in this part make such a step cost what it
should. "N" is the number of sessions in a step, and a **row** is one session's
token in it. N=3 comes first throughout, because three streams (a chat and two
agents) is the main use.

At the start, a step at N=4 took 71.3 ms against a byte floor of 38.0 ms, and
almost all of the 33 ms gap grew with the rows: each added row cost 14.0 ms,
where its own experts' bytes cost 6.4. A row paid for about 330 extra dispatches
(RoPE, rounding, expands), dense products that multiplied one row at a time,
and kernels that had no fixed shapes. The attention itself was not the
problem: 2.1 ms a row, the same as one-row decode.

**How these numbers were measured.** They were measured while building each
group, not on the release build; the release's own figures are in the README.
Unless a section says otherwise: ds4-bench's resident job runner (`nrun`),
sessions restored from saved caches of 2,000 to 3,000 tokens, greedy decoding,
DSpark off, milliseconds per step, serpentine order, median of three, on a quiet
machine. Each group was measured on top of the groups before it. Lower is
better, so "−1.3 ms" is a saving.

Over the whole part, measured the same way:

| N | before, ms per step | after, ms per step | after, t/s for all rows |
| ---: | ---: | ---: | ---: |
| 3 | 57.4 | 36.6 | 81.9 |
| 1 | 24.0 | 22.7 | 44.0 |
| 2 | 43.8 | 30.0 | 66.7 |
| 4 | 71.4 | 43.7 | 91.6 |
| 8 | 112.6 | 68.5 | 116.9 |

(Greedy, DSpark off. The N=8 "before" is after HEAD_ROWS and DECODE_ROWS, the
first groups below.)

**Exactness.** Every group here gives each row the bits of the same session
decoded alone, except ATTN_TILES. Every group was checked by hashing each row's
full logits at every step against that session's own one-row run, at N=2 to 8,
at short depth and at 34K to 300K. ATTN_TILES changes the attention sums for
one-row decode as well, so a row still equals its session decoded alone, but
neither equals main's vec kernel; `--exact` keeps the vec kernel (see
ATTN_TILES).

### HEAD_ROWS, DECODE_ROWS: each row at its one-row bits

`DS4_METAL_DISABLE_V41_HEAD_ROWS=1` and `DS4_METAL_DISABLE_V41_DECODE_ROWS=1`
turn them off.

Before these, a row's logits differed from its one-row run from the first step.
A trace of every stage of every layer showed that at two to four rows the whole
body already matched one-row decode bit for bit; only the vocabulary head
differed, because the batch took a head kernel that sums in another order.
HEAD_ROWS runs the rows' head on the Q8_0 row kernel, which gives each row its
one-row head's bits. From six rows a batch took the grouped MoE kernels, and the
first difference appeared in layer 0's routed experts; at eight rows that
changed tokens within 64 steps. DECODE_ROWS sends six to eight rows through the
fused decode MoE kernels that a DSpark verify block already uses. Prefill keeps
its kernels, and a verify block keeps its own head.

Kind: exactness. Cost: HEAD_ROWS +0.19 ms at N=3, +0.07 at N=2, +0.16 at N=4;
DECODE_ROWS −0.28 ms at N=6, +0.67 at N=8. These are the reference every later
group is checked against.

### BATCH_FLUSH, ROWS_REG: commit early, multiply in registers

`DS4_METAL_DISABLE_V41_BATCH_FLUSH=1` and `DS4_METAL_DISABLE_V41_ROWS_REG=1`
turn them off.

One-row decode commits its first command buffer after layer 0, so the GPU
starts while the host encodes the rest; the rows' step committed after two
layers. BATCH_FLUSH commits after layer 0 too.

ROWS_REG is the largest group for two to four rows. The dense Q8_0 products
(projections, the head, the shared expert) ran each row through the weights in
turn. Now one kernel keeps a sum per output and row in registers, reads each
weight once for all rows, and merges the lane sums along `simd_sum`'s own tree,
so each row keeps its one-row bits. On a microbenchmark of the 1,280 to 32,768
projection at four rows it took 84 µs, against 118 for the exact eight-row tiles
and 147 for the older four-row kernel. The rows' Q8 cost fell from 2.1 to about
0.6 ms a row. The number of outputs per threadgroup fits the row count: two at
two rows, three at three and four.

Kind: idle time (BATCH_FLUSH), arithmetic (ROWS_REG). Measured while building
it: BATCH_FLUSH −1.3 ms at N=3, −0.72 at N=2, −1.6 at N=4; ROWS_REG −4.5 ms at
N=3, −1.9 at N=2, −6.0 at N=4. Fitting outputs per threadgroup later took another
0.28 ms at N=3 and 0.23 at N=4.

### BATCH_ROPE, BATCH_KV: RoPE and the KV row as one-row decode does them

`DS4_METAL_DISABLE_V41_BATCH_ROPE=1` and `DS4_METAL_DISABLE_V41_BATCH_KV=1`
turn them off. BATCH_KV is the first switch of the second word (below).

Rows roped, stored and unroped in separate passes: about 133 dispatches a row,
1.2 ms a row, nearly all launch cost. With BATCH_ROPE the rows keep one-row
decode's per-step RoPE tables, one per session at its own position: the
attention reduce writes the heads already rotated back, and q_b rotates its rows
as it stores. With BATCH_KV the rows rope, quantize and store their KV rows in
one dispatch.

Kind: kernel launches. Measured while building it: BATCH_ROPE −1.4 ms at N=3,
−1.0 at N=2, −1.7 at N=4; BATCH_KV −0.54 at N=3, −0.21 at N=2, −0.83 at N=4.

### BATCH_TAIL, BATCH_LOW, BATCH_LOWBF: the layer's tail

`DS4_METAL_DISABLE_V41_BATCH_TAIL=1`, `…_BATCH_LOW=1` and `…_BATCH_LOWBF=1`
turn them off.

One-row decode's fused tail was gated off for rows, so a step at N=4 ran 246
separate bf16 rounding passes and 80 expands. With BATCH_TAIL the output and
shared-expert projections of two to four rows expand straight into the
hyper-connection streams, with no separate add, expand or rounding pass.
BATCH_LOW runs the attention low projection in registers, one weight pass for
all rows. BATCH_LOWBF has that projection store bf16, so output-B no longer
rounds every row again in each threadgroup.

Kind: kernel launches. Measured while building it: BATCH_TAIL −1.7 ms at N=3,
−1.5 at N=2, −1.8 at N=4; BATCH_LOW, measured after the tail had taken part of
its share, −0.38 at N=3, −0.21 at N=2, −0.95 at N=4; BATCH_LOWBF −0.18 at N=3,
−0.45 at N=4 and above, not distinguishable from noise at N=2.

### BATCH_NARROW, BATCH_HIER: the router for rows

`DS4_METAL_DISABLE_V41_BATCH_NARROW=1` and `…_BATCH_HIER=1` turn them off.

BATCH_NARROW lets two to four rows read the BF16 router copy that NARROW_WEIGHTS
builds for one-row decode, half the bytes of the stored F32 gate. BATCH_HIER has
them pick the top six with one-row decode's hierarchical router.

Kind: bytes moved, kernel launches. Measured while building it: BATCH_HIER −0.39
ms at N=3, −0.33 at N=2, −0.26 at N=4. BATCH_NARROW measured +0.05 at N=3,
−0.26 at N=2 and +0.14 at N=4, in an arm that also carried a narrow Engram form
that was later removed, so its own effect was not separated; it is within
0.3 ms either way. It stays as the rows' form of NARROW_WEIGHTS and is the path
BATCH_ROUTER8 extends to eight rows: with BATCH_NARROW off, five to eight rows
read the F32 gate too. Its own gain is therefore not known. It has not been
measured on its own, at two to four rows or, through BATCH_ROUTER8, at five to
eight.

### BATCH_OVERLAP, BATCH_DOWN, BATCH_WAVE, BATCH_MIX, BATCH_QKV: work side by side

`DS4_METAL_DISABLE_V41_<NAME>=1` turns each off.

These run independent work in concurrent encoder sections, as one-row decode
does with its shared expert.

- **BATCH_OVERLAP**: the hyper-connection mix and the shared expert's gate and
  up run beside the routed pair kernel. **BATCH_DOWN** adds the shared down
  beside the routed sum.
- **BATCH_WAVE**: each row's attention section (RoPE, key store, index, KV
  stage, vec, reduce) runs in waves: every row's k-th dispatch together in one
  concurrent level. Each row gets its own copy of the process-wide attention
  scratch. One session's vec stage nearly fills the GPU (overlapped, it keeps
  78% of its time at N=4), but the small stages are launch-bound and overlap
  almost completely. The section recorder first cost more host time than the
  waves saved; a table of the objects it holds, instead of a map lookup and an
  array append per bind, took the idle time at N=4 from 1.7 to 1.3 ms.
- **BATCH_MIX**: the mix the next block needs runs beside q_b, the short stage
  that otherwise leaves the GPU idle. Beside q_a and kv the same mix saved
  nothing.
- **BATCH_QKV**: q_a and kv run side by side at the head of that section, as
  one-row decode fuses them.

Kind: occupancy. Measured while building it: BATCH_OVERLAP −1.1 ms at N=3, −1.0
at N=2, −0.99 at N=4; BATCH_DOWN −0.57, −0.27, −0.72; BATCH_WAVE, after the
recorder fix, −1.1, −0.40, −2.0; BATCH_MIX −1.1, −0.79, −2.0; BATCH_QKV −0.20,
−0.21, −0.28. BATCH_QKV is small but removes all of the 0.35 ms that separate
q_a and kv passes cost against one-row decode's fused pair.

### BATCH_STATIC, BATCH_RBLOCK, BATCH_EXPAND: fixed-shape experts

`DS4_METAL_DISABLE_V41_<NAME>=1` turns each off.

BATCH_STATIC gives two to eight rows one-row decode's fixed-shape routed pair
and sum kernels, with every stride a literal and the row count an argument. They
need no plan dispatch, so MOE_DEDUP's plan (0.18 ms plus 0.24 a row) leaves the
rows' path; it stays for verify rows. After it, a line through the step times at
N=2 to 4 met one-row decode at N=1 within 0.31 ms: the fixed cost that rows had
carried was gone.

BATCH_RBLOCK reorders the routed grids so every pair visits one row block before
the next, oMLX's row-block-major order. The reads and the bits are the same;
only their order changes, and the routed section then streamed at about 790
GB/s against the 705 GB/s measured in the old order. oMLX reports about 4% at
four rows from this order, and it measured 3.3% at N=4 here.

BATCH_EXPAND writes the FFN join with its shape as literals, as one-row decode
does.

Kind: kernel efficiency. Measured while building it: BATCH_STATIC −1.1 ms at
N=3, −1.3 at N=2, −1.0 at N=4; BATCH_RBLOCK −0.86, −0.29, −1.7; BATCH_EXPAND
−0.21, −0.22, −0.25 (small and steady: the generic join cost 0.23 ms more than
one-row decode's).

### BATCH_MMA, BATCH_TAIL8, BATCH_DEFER8, BATCH_ROUTER8, BATCH_QROPE, BATCH_HEAD8: five to eight rows

`DS4_METAL_DISABLE_V41_<NAME>=1` turns each off.

The register kernels hold four rows; past that, rows fell back to slower
general paths, and the step jumped by a cliff at five rows (N=7 was even slower
than N=8). From five rows, the dense products run on SHORT_MMA's exact eight-row
simdgroup-matrix tiles (BATCH_MMA), which now take a row count: a tile row past
the count repeats row 0, and its outputs are not stored, so each row has the
same bits as in a full tile. BATCH_TAIL8, BATCH_DEFER8 and BATCH_ROUTER8 carry
the fused tail, the overlap section and the router through eight rows.
BATCH_QROPE rotates q_b as it stores on the tiles; an output's partner sits in
the next SIMD group, so the two meet in threadgroup memory. BATCH_HEAD8 runs the
head of five to eight rows on sixteen SIMD groups.

The tiles lose below five rows: an 8x8 tile is the hardware's shape, so unused
rows cost the same multiplies. At N=3 the tiles cost 1.8 ms more than the
register kernels.

Kind: arithmetic, kernel launches. Measured while building it, all six together:
N=5 75.0 to 59.7 ms, N=6 86.6 to 67.3, N=7 98.9 to 75.6, N=8 95.4 to 83.6, about
8 ms a row with no cliff. The six were not measured one by one, so how the gain
splits among them is not known. All but BATCH_MMA carry a two-to-four-row group
(BATCH_TAIL and BATCH_LOW, BATCH_OVERLAP, BATCH_NARROW and BATCH_HIER, the
rotation of BATCH_ROPE, HEAD_ROWS) on to eight rows.

### BATCH_NARROW_ENGRAM, BATCH_READS, BATCH_PE: Engram and the handoff for rows

`DS4_METAL_DISABLE_V41_<NAME>=1` turns each off.

BATCH_NARROW_ENGRAM has two to eight rows read the narrow (FP8) Engram copy in
one pass: a lane decodes each weight word once and keeps a sum per output and row.
An earlier form ran a SIMD group per row, so above four rows it started a second
pass that read the weights again.

Layer 1's command buffer waits for the first Engram table. The second table's
24 reads per row queued beside the first, though nothing needs them for
thirteen layers. BATCH_READS starts them once the first table has joined, which
took the idle time before layer 1's buffer from 0.76 ms to 0 at N=4 and from
1.5 ms to 0 at N=8. The cost had been the 48N disk reads, not the kernels.

BATCH_PE is PREENCODE for rows: the next step's first command buffer is
encoded during the final wait, with each row's token read from a device word at
commit; any change of rows or positions drops it.

Cross-session rows also find their keyed draws on the GPU now (KEYED_GPU, at
top_p 1), as one row does.

Kind: idle time, bytes moved. Measured while building it:
- BATCH_NARROW_ENGRAM, one pass against the form before it (sampled at
  temperature 1): −0.11 ms at N=3, −0.08 at N=2, −0.27 at N=4, −1.1 at N=5,
  −1.0 at N=6 and N=7, −0.61 at N=8. It pays most from five rows, where the
  form before it read the weights twice; below that it is small.
- BATCH_READS: about 0 at N=2 and N=3, −0.40 at N=4, −0.25 at N=5, −0.49 at
  N=6, −0.69 at N=7, −0.97 at N=8. It pays from four rows and costs nothing
  below.
- BATCH_PE: small. Alone it moved the idle time to the second buffer, which
  waited on the Engram join (0 at N=3, +0.19 at N=2). With BATCH_READS: −0.13 ms
  at N=3, −0.10 at N=2, −0.07 at N=4, −0.09 at N=8 greedy; −0.09 and −0.06 at
  N=3 and N=2 sampled. Rows already committed their first buffer about as fast
  as one row does with the slot, so there was little left for it to remove. With
  BATCH_READS it pays a small, steady 0.06 to 0.13 ms at every N measured, and
  it stays on.
- KEYED_GPU for rows, sampled at temperature 1 and top_p 1: −0.28 ms at N=3; at
  the server, three streams went from 71.8 to 72.7 t/s.

### MIXED_STEP, MIXED_CARRY, VERIFY_STACK, VERIFY_COPIES, VERIFY_DECODE: DSpark in a batch

`DS4_METAL_DISABLE_V41_<NAME>=1` turns each off.

Before this release, `--dspark` with `--batched-session` was accepted but ran
every step one session at a time, with no drafting: three streams took 73.1 ms a
step, 41 t/s in all. Now rows capture the drafter's features, so a batched
session stays ready to draft.

- **MIXED_STEP**: one session's verify block leads a step and the other
  sessions follow as plain rows. One block a step, rotating among the sessions
  whose controllers would draft, and only while at most three sessions decode
  together (`DS4_SERVER_MIXED_ROWS`). Two blocks in a step lose: at N=3, one
  block computes to 82 t/s, three blocks of 3, 3 and 2 rows to 68, plain rows
  to 74. A lone stream keeps the one-row DSpark path.
- **MIXED_CARRY**: a rejected block's correction token becomes that stream's
  row in the next shared step, instead of a one-row step that the other
  sessions wait for.
- **VERIFY_STACK**: the verify block runs the rows' FFN, Engram, router and head
  kernels from this part. On one timeline, a six-row verify block had cost 18.9
  ms more than six rows from different sessions: 1,756 passes a step against
  684 (598 of them one small copy each), about 7 µs of idle per extra pass, and
  an FFN without the rows' fused input.
- **VERIFY_COPIES**: the block's rollback snapshot and commit copies run as one
  concurrent level, since their destinations are distinct.
- **VERIFY_DECODE**: with VERIFY_INDEXED, the block attends with the decode
  kernels while keys and selection stay batched. Rows with at most top_k
  visible keys keep their own pick's order, and past 16,384 keys tied index
  scores are ordered by id, as decode's radix select orders them. Every verify
  row's logits then equal serial decoding's.

Kind: wasted work, kernel launches. Measured while building it, DSpark on:
- Verify time on 60 replayed real agent requests (one stream, 134K to 278K
  context, temperature 1, top_p 1), median per block: 79.8 ms before, 70.1 with
  VERIFY_STACK, 72.5 with VERIFY_DECODE too, which makes it exact. Decode speed
  over the same requests: +3.7% (95% interval +1.8% to +5.9%).
- VERIFY_COPIES: −0.51 ms per verify at short depth and at 300K, +0.2% and
  +0.6% t/s.
- MIXED_STEP: a step with a block beside two other sessions takes 84.5 ms at
  short depth and 88.5 ms at 131K, against 40.8 and 43.4 ms for three plain
  rows. On three real streams at 134K, 190K and 263K, replayed together, the
  stretches with all three decoding ran 3.8% faster than plain rows (95%
  interval +2.5% to +5.0%); with two streams, 8.1% (+5.7% to +10.3%).
- MIXED_CARRY: rejected blocks went from 115.6 to 91.1 ms with accepted blocks
  unchanged, an implied +1.4% at three streams. (That replay's warm start was
  flawed; the blocks' rows were not affected by it.)

Every row and every draw of a mixed step was checked against each stream's
serial chain: greedy and sampled, rejections, streams joining and leaving, and
the server's real step replayed serially, with 0 differing logits and 0
differing state bytes; and exact verify against serial on 44 saved sessions
from short to 400K.

### READ_ONCE: a shared expert read once

`DS4_METAL_DISABLE_V41_READ_ONCE=1` turns it off.

Rows that pick the same expert ran their pairs separately. MOE_DEDUP had shared
those reads in name only: every SIMD group still issued its own loads, so the
rows shared the cache, not the load. With READ_ONCE, rows that picked the same
expert run as a group in one launch: one load and one decode of the gate and up
weights feed one sum per row. Each row adds in its one-row order, with its
inputs loaded before the weights, because library fast math orders a sum by
where its terms are defined. A thread per (row, expert) pair counts its own
company to build the plan.

How much there is to share decides where it runs. Real sessions at 131K repeat
3.9% of their experts at N=3 and 14% at N=8. A six-row verify block repeats 31
to 34%. So a step with a verify block always takes READ_ONCE, and plain rows
take it from four rows.

Kind: bytes moved. Measured while building it: a block beside two other
sessions at short depth 84.7 to 80.9 ms (DSpark on), at 131K 88.6 to 86.0; a
block alone 68.4 to 66.1; plain rows at N=8 81.9 to 79.5 at short depth, 86.8
to 86.0 at 131K. Plain rows at N=4: −0.29 ms on the short caches, −0.17 on
oMLX-style prompts (one shared text with a unique prefix per row, 1,024
tokens), +0.02 on 1,024-token distinct texts, and +0.17 at 131K. Plain rows at N=2 and N=3 do
not take it: they repeat too few experts for the plan's cost. At four plain rows
it is about even, a small saving where rows repeat experts and a small cost at
131K, where they repeat fewer; it pays in every step with a verify block and at
eight rows.

### FP4_NO_TABLE: FP4 codes without a table

`DS4_METAL_DISABLE_V41_FP4_NO_TABLE=1` turns it off.

V4.1's decode expert kernels turned FP4 codes into floats through a table in
threadgroup memory, which every threadgroup filled behind a barrier. They now
decode with shifts on the packed bytes, after oMLX's FP4 decode. The dot keeps
the table form's chain of sixteen terms, spelled out with `fma`, so every row
keeps its bits: left to library fast math, the chain was flattened into another
order. The pair kernels scale the dot once; the down kernels, whose activations
can be small enough for the scale to matter, keep it on their activations. It
applies to one-row decode too.

Kind: kernel efficiency. Measured while building it, 128 tokens (64 on the
short caches), ms per step:

| N | oMLX-style prompts | distinct prompts | short caches |
| ---: | ---: | ---: | ---: |
| 3 | 40.3 to 38.8 | 41.1 to 40.1 | 40.6 to 39.2 |
| 1 | 24.1 to 23.9 | 24.1 to 23.9 | |
| 2 | 32.7 to 32.1 | 32.9 to 32.4 | 32.6 to 32.0 |
| 4 | 49.2 to 46.8 | 50.3 to 48.8 | 49.4 to 46.9 |
| 8 | 76.8 to 70.8 | 80.2 to 76.5 | 79.1 to 74.1 |

(Greedy, DSpark off.) A block beside two other sessions at short depth went
from 80.6 to 73.9 ms (DSpark on).

### ATTN_TILES, LIVE_SPLITS: decode attention

`DS4_METAL_DISABLE_V41_ATTN_TILES=1` and `…_LIVE_SPLITS=1` turn them off.

ATTN_TILES runs decode attention's scores and PV on simdgroup matrices: 8 heads
by 64 keys a threadgroup, 1,024 threads, 8x8 tiles, the keys read straight from
the cache by index. It is adapted from oMLX's decode attention (notice in
`licenses/omlx.txt`) and runs at every row count: one-row decode, batched rows
and verify blocks. V4.1 decode attends to at most 640 keys (128 raw and up to
512 selected), so the kernel's limits are never reached; it falls back to the
vec kernel under `--quality`, on GPUs other than the M3 and M5 families, with
DECODE_ATTENTION off, and across two Macs.

This group is not bit for bit with the vec kernel. Rows stay equal to the same
session decoded alone, since one-row decode takes the tiles too, but not to
main's serial output; `--exact` turns it off. Its accuracy against the vec
kernel: over 21 long DeepSeek V4.1 scoring cases, mean NLL 0.2993 against
0.3003; over 129 further cases, the token-weighted distance to the original
checkpoint's output grows by 0.0011 (+4%), NLL moves by less than 0.0002, and
top-1 agreement with the original is 0.9680 against 0.9673. Greedy from 16
prompts to the first differing token, 735 steps: mean KL 0.00175, at most 0.035,
14 differing tokens (1.9%).

LIVE_SPLITS: attention splits its keys over 32 partials, and with at most 640
keys 12 of them had none. Only the splits with keys launch, and the reduce reads
only those. The logits are the same bits.

Kind: arithmetic (ATTN_TILES), kernel launches (LIVE_SPLITS). Measured while
building it, 128 tokens, ATTN_TILES off to on, oMLX-style / distinct prompts:
N=3 38.3 to 36.2 ms / 39.4 to 36.9; N=1 23.6 to 22.7 / 23.6 to 22.7; N=2 31.4
to 29.7 / 31.9 to 30.3; N=4 46.1 to 43.2 / 48.2 to 45.0; N=8 69.7 to 65.4 /
75.3 to 70.2 (greedy, DSpark off). LIVE_SPLITS: −0.60 and −0.43 ms at N=8 on the
two prompt sets, −0.13 and −0.04 at N=4, 0 at one row: it pays from four rows.

### FINAL_SPIN: wake on the GPU's signal

`DS4_METAL_DISABLE_V41_FINAL_SPIN=1` turns it off.

A step ends with the host waiting for its last command buffer. The buffer's
completion reached the host about 0.33 ms after the GPU had finished. Now the
head buffer signals a shared event, and the host spins on that event for the
final wait. One CPU core spins through the last command buffer, 3 to 5 ms a
step, and only for that wait. It is off while `DS4_METAL_CB_TIMES` is set.

Kind: idle time. Measured while building it: −0.33 ms at every N, one row
included; serial decode at depth 40.2 to 40.7 t/s; one-stream DSpark at short
depth 50.4 to 50.7 t/s sampled and 65.4 to 65.5 greedy. The logits are the same
bits.

### The server: prefill beside decoding sessions

A cold prompt that arrives while other sessions decode used to stop them. The
server changes below are V4.1-only and need `--batched-session`.

**A time share, 2,048-token chunks.** Any V4.1 prefill chunk sweeps all the
routed experts, about 0.55 s, plus about 1.0 ms a token, however short it is.
At the old 128-token chunk, decoders got one step per chunk: three decoders at
about 2,700 tokens beside a cold 9,240-token prompt ran 1.37 t/s, and the
prefill took 53.1 s. Larger chunks alone froze them harder (0.40 t/s at 2,048
tokens). Now the decoders get as much wall time after each chunk as the chunk
took (`DS4_SERVER_MIXED_DECODE_SHARE`, 100% by default), and a chunk is 2,048
tokens: decoders 8.1 t/s, the prefill 22.8 s, everything done in 41.8 s against
78.2. Its worst stall between two tokens was 2.7 s.

**Parked sweeps.** The prefill now runs as one sweep in 32,768-token syncs and
parks between its chunks, keeping its place on its own stack and its own graph,
so the decoding sessions take a step at every park. A parked sweep that is
cancelled still comes back to unwind under the lock. Only a sync's end is a
checkpoint: 32,768-token syncs ran 868 t/s on a cold 125,915-token prompt,
against 883 for one sync of the whole prompt, which stored no checkpoint during
the prefill, and 791 for 8,192-token syncs.

**Waiters and short requests.** A parked sweep could not see a thread queued on
the inference lock and retook the lock first, so a title request beside two cold
prefills took 129 to 154 s. A sweep now holds until queued threads have had the
lock. A request no wider than one chunk also no longer takes turns with sweeps:
layer-major chunks run per layer however few the rows, so a 15-token prompt had
alternated with wide chunks at every layer, 7.7 to 14.5 s.

Measured while building it, server level, three sessions decoding greedily
beside a cold 9,240-token prompt:

| | prefill s | decoders t/s | gap p99 ms | gap max ms |
| --- | ---: | ---: | ---: | ---: |
| time share, no drafter | 22.7 | 12.5 | 2,663 | 2,675 |
| parked sweeps, no drafter | 22.1 | 14.2 | 278 | 554 |
| time share, DSpark on, disk KV cache | 22.8 | 11.6 to 12.8 | 2,686 | 2,725 |
| parked sweeps, DSpark on, disk KV cache | 22.7 | 13.6 to 16.0 | 276 | 337 |

Beside a cold 125,916-token prompt the worst stall is 0.8 s, where a sweep's
encoder chunk is 8,192 rows. A prefill alone still runs at 917 to 925 t/s. The
prefill's logits and state, and every decoder's, equal a plain prefill and
serial decoding at every park (3,500 to 40,000 tokens).

A replay of real agent traffic at a deployment configuration (four batched
sessions and a utility session, a 1,048,576-token context, DSpark on), against
the build before the waiter, short-request and sync-width changes: utility requests
(titles, summaries) median 15.1 to 2.3 s, worst 234 to 23.3 s, with 117 served
instead of 29; first tokens of short appends median 13.9 to 5.8 s, worst 66.3 to
9.7 s; checkpoints refused during prefill 15 to 0. Cold prefills there finish
later than before, since short work now runs beside them and decoders keep half
the wall clock. `DS4_SERVER_DISABLE_PREFILL_YIELD=1` returns to the time share
alone.

**The utility session in a batch.** With `--batched-session` and
`--utility-ctx` together, the utility session's row joins the decode step beside
the other sessions; before, the utility session needed non-batched mode. During three decoding streams a title took 0.57 s and an image request
2.6 s; the streams' four-row steps cost the normal four-row time.

## Two Macs

These four groups apply only to two-Mac tensor parallelism. They were built and
unit-tested here but not measured, because no second Mac was available. On one
Mac they never run.

### TP_BALANCED_ROUTE: half of every token's experts on each Mac

`DS4_METAL_DISABLE_V41_TP_BALANCED_ROUTE=1` turns it off. It is read at load.

Each Mac owned half of the routed experts by id, so a token whose six experts
fell unevenly between the halves made one Mac wait for the other. Now each Mac
takes half of the token's ordered selection: the first three experts on one,
the last three on the other. Both Macs map every expert.

Kind: idle time (load balance).

### TP_SHARED_SPLIT: split the shared expert

`DS4_METAL_DISABLE_V41_TP_SHARED_SPLIT=1` turns it off.

The shared expert ran whole on one Mac, alternating by layer, so in every layer
one Mac worked while the other waited. Now each Mac computes half of its hidden
width and the matching slice of the down projection. The two partial outputs
join the routed sums that already cross between the Macs, so no exchange is
added.

Kind: idle time (load balance).

### TP_GATES: partials go straight into the gate slots

`DS4_METAL_DISABLE_V41_TP_GATES=1` turns it off.

Each exchange copied this Mac's partial into its outgoing slot, waited for the
peer, added both slots, rounded the sum, and then expanded it into the four
streams. Now the producers write their partial straight into the slot, and in a
decode step the expansion reads both Macs' slots and adds them itself, with the
same rounding.

Kind: kernel launches and bytes moved.

### TP_PREFILL_ROWS8: the attention output rows in one dispatch

`DS4_METAL_DISABLE_V41_TP_PREFILL_ROWS8=1` turns it off.

With two Macs, prefill rows that the matrix path does not take were encoded as
one product per token. They now go through one dispatch over all rows, and on
the M3 Ultra, with 32 or more rows, a kernel that covers eight output rows per
threadgroup reuses each input row across more outputs. The results are the
same bits.

Kind: kernel launches and bytes moved.

---

## Image input

An image request runs in three parts before its first token: the host decodes
and prepares the image, the vision encoder turns its patches into rows of
embeddings, and the language model prefills the prompt that holds them. The
groups below speed up the first two; IMAGE_CED, IMAGE_TRIM and the other
prefill groups above speed up the third. Encoder times are for the 1017 vision
encoder on the M3 Ultra, in one process. A 512x507 photo is 169 encoder rows, a
1400x900 screenshot 748 and a 3840x2160 screenshot 943. Their switches are read
straight from the environment, not through the group mask.

### VISION_HOST_PREP: prepare the image on every core

`DS4_METAL_DISABLE_V41_VISION_HOST_PREP=1` turns it off.

Before the encoder starts, the host orients, resizes, normalizes and cuts the
image into patches, on one thread. Output rows are independent, so each of
those steps now splits by rows across the cores, each row with the same
per-pixel code as before. The image's fingerprint uses the system's SHA-256,
which gives the same digest.

Kind: idle time. Measured: decode and preparation of a 3840x2160 screenshot 340
to 114 ms, a 4032x3024 photo 510 to 174 ms; patches and fingerprints the same,
bit for bit, on 29 images.

### VISION_PREP_REUSE: decode each image once

`DS4_DISABLE_V41_VISION_PREP_REUSE=1` turns it off.

The server decodes and prepares every new image when the request arrives, to
learn its token count and fingerprint, and the encoder then decoded and
prepared the same bytes again. The encoder now takes the patches the first
pass built, keyed by a full copy of the image's bytes.

Kind: wasted work. Measured: a 4K PNG 1,322 to 1,279 ms, a 4K camera JPEG
1,511 to 1,349 ms, with the same embeddings.

### PNG decoding

`DS4_DISABLE_PNG_SYSTEM_INFLATE=1` uses the bundled decoder alone.

The bundled PNG reader spent most of its time on checksums. Adler-32 now
reduces once every 5,552 bytes (zlib's bound) instead of twice per byte, CRC-32
uses a table, and on Apple the compressed stream goes through the system's
DEFLATE decoder, whose output is kept only when it has exactly the expected
length and checksum; anything else falls back to the bundled decoder.

Kind: arithmetic. Measured: a 3840x2160 PNG 97 to 35 ms, 1400x900 14.3 to
4.8 ms, with the same pixels on 83 PNGs.

### VISION_ROUND_FOLD: round inside the encoder's kernels

`DS4_METAL_DISABLE_V41_VISION_ROUND_FOLD=1` turns it off.

The encoder rounded its activations to bf16 in 195 separate passes per image,
after each bias, norm, attention output, residual and MLP product. Each rounding
now happens in the kernel that makes the value or first reads it, with the same
arithmetic.

Kind: kernel launches and bytes moved. Measured: 7, 21 and 42 ms less for the
photo, the 1400x900 and the 3840x2160 screenshot, with the same embeddings.

### VISION_ATTN_EXACT: 32 keys per step

`DS4_METAL_DISABLE_V41_VISION_ATTN_EXACT=1` turns it off.

The encoder's attention is global over all patches, and its kernel gave one
SIMD group to each query row and head and walked the keys one at a time, each
with a sum across the SIMD group: 89% of the encoder's time on a large image.
The new kernel takes 32 keys per step, shared by eight query rows through
threadgroup memory. Each lane forms one key's score and adds its products in
the same tree as the old sum, and the softmax then runs in key order exactly as
before, so the output is the same bits.

Kind: occupancy and arithmetic. Measured: the encoder takes 216 to 176 ms on
the photo, 2,701 to 2,079 ms on the 1400x900 screenshot and 4,375 to 3,466 ms
on the 3840x2160 one, with the same embeddings and the same 200 greedy tokens
after each image.

### VISION_ATTN_FLASH: the encoder's attention on simdgroup matrices

`DS4_METAL_DISABLE_V41_VISION_ATTN_FLASH=1`, or `--exact`, turns it off.

The exact kernel still forms every score and output with scalar arithmetic.
The flash kernel runs them on 8x8 simdgroup matrices. Eight SIMD groups share
each tile of keys and values; each row's scores sit in four lanes of the
matrix, so the running maximum, the probabilities and the output rescale stay
in registers; and K and V are written as half once by the projection kernel
and read straight from memory, with no staging and no barriers. Every product
and sum stays in float, but the order differs, so it is not bit for bit.

Kind: arithmetic throughput. Measured on the 3840x2160 screenshot: the first
flash kernel took the encoder from 3,466 ms (exact) to 2,537 ms; keeping the
softmax in registers, in a later session, from 2.71 to 1.60 s; and reading K
and V as half from 1,267 to 1,185 ms.

Accuracy, with every image default on against `--exact`: on 216 exactly scored
questions about 42 rendered images (charts, tables, dark interfaces, layout and
small text), 176 against 177 correct answers. Five answers differed, all
readings of characters 9 to 11 pixels tall, and in four of them both modes were
wrong. On 20 images the first token's top choice was the same, with a median
KL divergence of 1.3e-3 and at most 0.035 on dense text. The set has rendered
images only, short answers and one image per request.

### VISION_ATTN_LAZY: rescale only when a maximum grows

`DS4_METAL_DISABLE_V41_VISION_ATTN_LAZY=1` turns it off.

The online softmax rescaled all 64 output values of every row on every tile of
keys. It now keeps each row's running maximum until some row would grow it by
more than ln 256, so the probabilities stay below 256 and the rescale is rare
after the first tile. It is the same softmax after the final division, rounded
differently. The key mask moves to the last, partial tile.

Kind: arithmetic. Measured: the encoder takes 1,467 to 1,305 ms on the
3840x2160 screenshot and 949 to 852 ms on the 1400x900 one.

### VISION_ATTN_KT16: 16 keys per tile

`DS4_METAL_DISABLE_V41_VISION_ATTN_KT16=1` turns it off.

With 64 keys per tile each SIMD group held eight score fragments beside the
query and the output, and the register load, not the arithmetic, set the pace.
Sixteen keys per tile hold two. The reductions and the rescale test then come
four times as often, and the kernel is still faster; 32 keys gained less and
8 keys was twice as slow.

Kind: occupancy (register pressure). Measured: the encoder takes 1,236 to
1,123 ms on the 3840x2160 screenshot and 814 to 738 ms on the 1400x900 one.

### Image state in the disk KV cache

A checkpoint of a chat with images carries the identity of each image and of
the encoder's numerics: the vision file's fingerprint, tagged with the flash
kernel's version while it runs, and a tag while IMAGE_CED runs. An image chat
stored under one setting misses under another and is read again. Text
checkpoints are guarded by the numerics canary, a fresh prefill whose logits
change with PREFILL_MMA and CED, so they too never cross between the default
and `--exact`.

---

## Identified, not built

### One routed-expert tile per expert for short prefills

Expected gain: 0.10 to 0.16 s on a 4K image prompt, and some on text prompts
longer than about 500 tokens. Decode and the decoder layers under CED gain
nothing. Not built: the speed targets for this release were met first.

**Where the time goes.** A 4K image prompt (978 rows) takes about 1.55 s of
language-model prefill, and the routed experts are about 0.75 s of that. The
decoder layers (20 to 39, 256 rows under CED) take about 11 ms each, already at
the floor set by streaming the expert weights. The encoder layers (0 to 19, all
978 rows) take about 27 ms each against a compute floor of about 14 ms: 18.2 ms
for gate and up, 8.7 ms for down. At 978 rows each selected expert gets about
15 rows, and the MXFP4 tile kernels cover an expert's rows in 32-row tiles,
dequantizing each weight tile once per tile. With its matrix products skipped,
the gate and up kernel still takes 10.2 ms; with only the dequantization
skipped, 16.3 ms. So staging, barriers and memory latency cost about as much as
the arithmetic, and dequantization is a small part.

**The change.** A prefill kernel for about 500 to a few thousand rows, a few to
about 64 rows per expert:

1. One threadgroup per expert and 64-column block covers all of that expert's
   rows, up to 64, so each weight tile is dequantized once per expert.
2. The next step's weights and activations are staged through threadgroup
   memory while the SIMD groups multiply the current one. Double buffering in
   registers was tried and was 40% slower from register pressure; staging
   through threadgroup memory is the open design problem.
3. The same structure for the down projection.
4. Its own switch, for those row counts only; above them the large-prefill
   tiles win.

**Exactness.** It can be bit for bit: the same half-precision dequantized
weights and the same ascending order of 8x8 multiply-adds for every output.
The gates are the usual ones: the logits after cold prefills of 8, 8K, 32K and
131K tokens, an image prompt's first-token logits, 256 greedy steps from main's
8K cache and the DSpark digest.

**Expected gain.** 20 to 30% off the encoder layers' experts: about 5 to 8 ms
per layer over 20 layers, 0.10 to 0.16 s per 4K image prompt. Text prompts over
about 500 tokens gain in proportion to their encoder layers' expert time.

**Already tried, slower:**

- Double buffering in registers: 40% slower.
- Skipping the products of empty 8-row blocks: the same bits, but 3 to 4%
  slower everywhere (a 4K image prompt 1.553 to 1.606 s).
- The other existing expert tile variants: no better than the defaults at these
  row counts.

**Effort and risk.** About a day of kernel work, with a measurement at each
step. Two earlier attempts at this kernel were exact but slower.
