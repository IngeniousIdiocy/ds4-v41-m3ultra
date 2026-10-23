# DeepSeek V4.1 Flash on Metal: why each group is faster

This page explains the speed groups of this branch, one section per group,
grouped by phase and in the order they were added on top of DwarfStar main. Each section says what the
group changes, why that saves time, and what it measured when it was turned on.

Every group has one off switch, `DS4_METAL_DISABLE_V41_<GROUP>=1` (two groups
use `DS4_METAL_MODEL_TRACKED=1` and `DS4_METAL_DISABLE_COMPUTE_TENSOR_COPY=1`). With a switch set, that part of the
engine runs main's code again. [RUNTIME-V41.md](RUNTIME-V41.md) lists every
switch; [REPRODUCE-V41.md](REPRODUCE-V41.md) says how to measure them.

The page ends with the image input groups and one speed-up that was identified
and sized but not built.

Some of the code here started in forks by Ivan Fioravanti and Tarjei Mandt;
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
