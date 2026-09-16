# Serial fusion campaign — 2026-09-16

> Historical experiment log, published with the consolidated decode release.
> Entries retain their original stage and verdict, including failures and later
> corrections. Current defaults and results are in the
> [release notes](../../docs/DECODE-CONSOLIDATION-20260916.md).
> `LOCAL_ARTIFACTS` and private commit IDs identify original provenance; those
> machine-local archives are not included. Numeric receipts are alongside this log.
> Rejected implementation code, private request bodies and conversation output
> are not distributed. The public integration retains all nine accepted paths.


Base: private UAT `bbaebc1`. Worktree: `v41-serial-fusion-20260916`.
Goal: attention-output fusion, complete down schedule, router/shared/HC
scheduling, and down-tail specialization. Use the GLM Tier 1/Tier 2 gates.
No quantization changes in this campaign.

## Current status

**Deployed to private UAT**, implementation `723e156`. All nine exact features
are on by default. Prior UAT is preserved for rollback; original service
plists and model weights are unchanged. Colima, all workers, both games,
vision, switch and bridge are restored. The full Trench canary passes.

The scalar bundle gains +2.213/+2.138/+2.090 t/s at8k/62k/300k against
bbaebc1, with exact full outputs. Six-row code gains +0.65/+0.75/+0.67 t/s;
the fixed-six agent gains +0.61 t/s. All kernel, native/depth, forced-rollback,
existing regression and bounded serving gates pass. Normal code serving
improves47.115->48.730 t/s. Controller timing adapts to the faster serial
path on each request; no policy patch was needed. The extended14-request
answer aggregate is deferred under the owner's deploy-now instruction.

Rejected implementations are culled. Their measurements, verdicts and archive
locations remain in the chronological ledger below. The initial KV exactness
failure is repaired with the original dynamic loops. The wider validated KV
fallback replaces the redundant narrower implementation.

## A1 — serial attention output + HC

Candidate retains the Q8 K/lane assignment and both SIMD reduction trees.
The owning output lane rounds the block to BF16, writes it, then computes
four ordered HC FMAs and rounds each residual result to BF16. The block is
still materialized for diagnostics. Scope is scalar, single-device,
non-streaming Metal DS41, Q8 8192→5120, HC=4, NSG=4. Other paths retain their
existing calls. `attn_out_hc` is a diagnostic A/B lever, initially off while
validation is pending; accepted exact optimizations will default on.

Reference kernel source is extracted from immutable commit `bbaebc1`,
not from a modified template. The harness regenerates all weights, input,
residual, and mixer values on every draw; NaN-poisons both output buffers;
compares all 5,120 block and 20,480 HC words; verifies guard words; and plants
a changed-post-gate control on every draw. A separate 1,000-run test poisons
the output before each identical-input run. Production geometry is used.

Initial screen: 64 draws, 1,638,400 words, 100% finite, zero differing words,
zero poison survivors, intact guards, 64/64 controls detected, 1000/1000
identical-input runs. The first 50k run compared 1,280,000,000 words: zero mismatches, 100% finite,
50,000/50,000 controls detected, 1000/1000 determinism. Audit found that the
standalone HC reference used 32 threads whereas current production uses 64.
The harness has been corrected; its final production-geometry rerun is pending
until native timing finishes, so the GPU workloads do not overlap.

8k native result: 4 interleaved 512-token pairs, first token excluded,
33.883325 → 34.262775 t/s; paired +0.379450 ± 0.035263 t/s SE. All paired
outputs byte-identical. Colima and omni were stopped before measurements.
62k/300k are still running. These are quiet-campaign comparisons, not a
replacement for a production-service-on UAT comparison.

## Experiments and failures

Initial entry (superseded by the verdicts below): no candidate had yet been
accepted or rejected. Small kernel screens alone do not establish adoption.

## Cache discipline and rollback

The diagnostic server gains `DS4_BENCH_REQUIRE_DISK_KV`: a missing or
incompatible fixture returns an error instead of silently prefilling.
Use isolated ordinary copies of the existing DSpark-compatible snapshots;
do not symlink a writable cache to a production file. Original GGUF files,
production binary, and LaunchAgent configurations remain unchanged.
No services have been stopped at this point. Campaign lifecycle receipts
will record any service changes and restoration commands before isolation.

## D1 — complete down schedule, exact consumer prototype

First isolate scheduling from arithmetic: retain the current routed-down
accumulator, postpone shared-down until routed-down finishes, and put the HC
consumer in shared-down's owning lane. This preserves all intermediate
rounding and makes it possible to assess moving the consumer independently
of the rejected R4 association change. `shared-down-candidate.metal` is a
standalone prototype; it is not linked into the current resident server and
has no performance or fidelity verdict yet.

An expert-parallel follow-up must pass a fresh GLM Tier 2 gate. The old R4
failure remains a failure; adding consumer fusion would not erase it.

## Isolation update

The user's explicit 2026-09-16 request authorized stopping Colima and llama.cpp.
Both were stopped gracefully before native A/B. The original model, bridge,
and switch LaunchAgents were already stopped for the isolated server.
State and restore commands are in
`LOCAL_ARTIFACTS/serial-fusion-20260916/lifecycle.json` and
`background-services.json`. The latter records the three running containers
and exact argv/cwd for the two ephemeral game servers (8809 and 8810), which
must be restarted after Colima. No plists or weight files were edited.

## R1 — router/shared/HC mixed grid prototype

The first arm combines independent router matvec, shared gate/up, and
already-deferred HC coefficient work in one grid. It retains the standalone
384-expert selector after the grid. Router keeps production NSG=8/NR0=2;
shared GU keeps the existing two NSG=4 cohorts in each 256-thread group;
HC keeps its existing 12-group coherent publication protocol unchanged.
Total grid: 12 HC + 576 shared GU + 192 router = 780 groups. Keeping selection
as the next dispatch avoids a new cross-die router publication protocol.
This is `router-shared-candidate.metal`, not yet wired or validated.

Both D1 and R1 are prototypes awaiting standalone and full-model gates.
Source reasoning alone is not evidence of bit exactness or speed.

## A1 completed evidence and capability reporting

The production-geometry rerun passed 50,000 draws / 1,280,000,000 words,
100% finite, zero mismatches or poison survivors, 50k/50k controls, and
1000/1000 deterministic runs. Native tests used one resident server and
only restored KV, no prefills. At each depth all eight measured arms decoded
the identical 512-token continuation, not just identical pairs.

Per the owner's measurement clarification, the best valid clean run is a
measured capability, not a result to discard. Report best control and best
candidate alongside paired/mean throughput. A fast observation alone does not
identify the cause of slower observations. The 62k 34.5377 t/s observation
is valid: same cached prefix, 512 tokens, identical continuation and timed
work, no prefill, GPU idle beforehand, CPU idle 98.3%. It is retained.

| Context | Best control → candidate | Mean control → candidate |
|---|---:|---:|
| 8k | 33.9302 → 34.3587 | 33.8833 → 34.2628 |
| 62k | 33.5883 → 34.5377 | 33.5550 → 34.0660 |
| 300k | 32.1385 → 32.4753 | 32.1294 → 32.4568 |

No post-hoc outlier exclusions. Means retain all four pairs. These are
campaign baseline vs A1 results, not public-branch vs final UAT statistics.
`attention-results.json` records the workload identity and best-run requests.

## D1 and R1 current state

D1's full production-geometry harness passed: 50,000 draws, 1,536,200,000
words (shared, block, HC, carried pre coefficients), zero differing bits,
100% finite, 50k/50k controls, 1000/1000 deterministic runs. Full-model
integration and speed remain untested.

R1's initial screen passed: 64 draws, 175,168 words (router logits, shared
mid, HC mix/split and completion counter), 100% finite, zero differing bits,
64/64 controls detected; 99.9884% of the controlled route/shared words moved;
1000/1000 deterministic runs. The 50k campaign is now running.

D1/R1 host paths are wired behind initially-off diagnostic levers
`shared_down_hc` and `router_shared_hc`. Host Metal build, CPU object build,
and full Metal library / all three PSO compile checks pass. The running
server still contains A1 only; its original source/binary manifest remains
in the raw artifact directory. Do not infer live deployment from source files.

No candidate has failed a fidelity gate so far. Setup corrections are also
recorded: HC reference geometry changed 32→64 and full gate rerun; a
maintenance script was first invoked from the wrong cwd, failed before editing,
and was rerun with the correct path. No benchmark was affected.

## Bundle 1 native result and validation update

R1 full Tier 1 kernel gate passed: 50,000 fresh draws, 136,850,000 words,
zero mismatches, 100% finite, 50k/50k controls (99.9885% controlled words
moved), 1000/1000 deterministic runs. A1/D1/R1 are now in the isolated
resident server; production UAT remains unchanged. Source/binary identity
is recorded in raw `server-manifest.json`; the A1-only predecessor manifest
is preserved as `server-a1-manifest.json`.

Four interleaved 512-token pairs per arm at 8k; every arm and pair produced
identical text, disk prefix restored, zero new prefill. A separate 128-token
warmup is excluded because it has a different workload. All rates exclude
the first decoded token. GPU <=5% and CPU idle >=90% before every request;
Colima/omni stayed stopped. No valid measured observations were excluded.

| Arm | Best t/s | Mean t/s | Paired mean change ± SE |
|---|---:|---:|---:|
| Control | 33.8725 | 33.8393 | — |
| A1 attention | 34.2285 | 34.1825 | +0.3431 ± 0.0198 |
| D1 shared-down consumer | 33.7839 | 33.7528 | −0.0865 ± 0.0133 |
| R1 router/shared/HC | 34.2221 | 34.1007 | +0.2613 ± 0.0492 |
| A1+D1+R1 | 34.4999 | 34.4041 | +0.5648 ± 0.0355 |

D1 alone was slower in all four pairs and has not earned inclusion.
The next combination should exclude it to isolate A1+R1. This does not
reject the broader complete-down schedule hypothesis; this arm retained
the serial routed-down accumulator and only moved its consumer.
The aggregate is still below the requested materiality target.

## T1 — fixed nine-block expert-down loop

Two variants preserve the six-slot accumulator and per-lane block order,
with the K=9 schedule written as three fixed iterations plus its existing
lane-tail predicate. Both plain routed and folded HC producer paths are
covered. The initial compile failed because QK_K/N_R0_Q4_K were out of
scope in the destination Metal file; fixed local constants (256 and 2)
resolved it. Full library and all five PSOs compile. This was a setup
failure, not a performance result.

The folded 64-draw screen passed 1,966,336 words, 100% finite, no mismatches
or poison survivors, intact guards, 64/64 changed-input controls and
1000/1000 determinism. References come from immutable bbaebc1 sources;
fresh active Q4 blocks, activations, residual, shared BF16 and HC coefficients
are regenerated every draw using production expert group strides and
rotating group order. Plain screen/full gates and native timing remain
pending. T1 has never been enabled on a model request or in UAT.

T1 harness setup correction: the plain reference uses function constant 608
as `short`, not `bool`. The first plain screen aborted during PSO creation
before executing any draws. Corrected the host type and added an explicit
nil-function check; the rerun passed 64 draws / 327,680 words, zero differing
bits, 100% finite, 64/64 controls, and 1000/1000 deterministic repeats. The
folded reference hardcodes its wide mode and was unaffected. Both full
gates are queued sequentially. No model timings overlapped the harness.

G1 is a separate scheduling hypothesis prepared for screening: use NSG=1,
NR0=2 instead of NSG=2, NR0=2. It doubles threadgroups while preserving
2,560 total SIMD groups and the same per-lane continuous six-expert sum.
The older rejected NR0=1 experiment doubled SIMD work as well; it is not
this candidate. G1 currently has a standalone harness only, no host lever
and no speed or fidelity verdict. Its reference remains the immutable
production NSG=2 kernel. Native full-model timing must decide adoption;
fixed-input microkernel timing is diagnostic only.

T1 folded full gate passed: 50,000 draws / 1,536,200,000 output words, zero
mismatches/poisons, intact guards, 100% finite, 50k/50k controls and
1000/1000 deterministic repeats. The plain full gate is still running.

G2 standalone prototype: four output rows per SIMD instead of two, keeping
the original per-row K traversal and six-slot accumulator. This trades
activation reuse against register pressure without exposing expert slots
as separately reduced partials. Source and a geometry harness are prepared;
no GPU screen, host integration or performance verdict yet. The immutable
reference retains two rows/SIMD. G2 shaders are appended only to the
standalone harness library; the resident model library is unchanged.

T1 plain full gate also passed: 50,000 fresh draws / 256,000,000 words,
zero differing bits/poisons, intact guards, 100% finite, 50k/50k controls,
1000/1000 deterministic repeats. Both T1 kernel paths are now eligible
for native A/B performance testing, not yet for deployment.

Subsequent geometry harnesses use one 256-thread comparison group instead
of one lane scanning all words. Coverage is unchanged. An independent
checker self-test plants first/middle/last mismatches, matching NaNs and
infinities, and two damaged guard words, then checks all five expected
counters before any numerical gate. The completed T1 50k results used the
original serial checker. No production kernel was altered for this change.

## Geometry screens completed; Bundle 2 running

All six geometry screens passed the checker self-test, 64 fresh draws,
complete output/guard comparisons, all planted controls and 1000/1000
identical-input repeats. G1/G2 have not passed full 50k gates or native
model tests. Fixed-input GPU times are diagnostic, not full-model claims.

| Folded arm | Best reference → candidate, μs | Mean reference → candidate, μs |
|---|---:|---:|
| T1 fixed nine-block loop | 64.700 → 62.842 | 65.487 → 63.102 |
| G1 NSG=1, NR0=2 | 64.637 → 64.104 | 65.327 → 64.539 |
| G2 NSG=1, NR0=4 | 64.278 → 64.616 | 65.136 → 64.839 |
| G2 NSG=2, NR0=4 | 64.838 → 62.837 | 65.382 → 63.625 |

Plain-path NSG=1 and NR0=4/NSG=1 screens also passed, with small kernel
time improvements. No native G1/G2 integration exists. The full record is
`geometry-screen-summary.json`, including all observations. The G2 NSG=1
best rate did not improve, despite its slightly improved mean; retain both.

Bundle 2 is running on the same resident server, comparing control, T1,
A1+R1, and A1+R1+T1 at 8k (128 warmup, four 512-token pairs). D1 is off
throughout. Initial warmup rates suggest T1 may be slower in the real
model despite its fixed-input kernel result; the measured pairs will
establish the native verdict. Do not promote from the microkernel screen.

## Bundle 2 verdicts and runtime cull

Four measured 512-token pairs at 8k, with disk prefix restore and no prefill;
all greedy token continuations were identical across arms. All quiet checks
passed. Best and mean rates retain all valid measured observations.

| Arm | Best t/s | Mean t/s | Paired mean delta ± SE |
|---|---:|---:|---:|
| Control | 33.9320 | 33.8638 | — |
| T1 fixed loop | 33.4760 | 33.4515 | −0.4123 ± 0.0371 |
| A1+R1 | 34.5023 | 34.4449 | +0.5812 ± 0.0435 |
| A1+R1+T1 | 34.1827 | 34.1171 | +0.2533 ± 0.0469 |

T1 is rejected: slower in every pair, alone and against A1+R1. Its runtime
lever, host hook and two shader kernels have been removed. The standalone
D1 consumer arm and old rejected R4 scalar-partial implementation have also
been removed from the runtime. The new P1 below is a separate hypothesis.
Pre-cull source is archived outside the repository under raw
`rejected-runtime-preimage/`, including a SHA256 manifest. No production
UAT binary, model file or LaunchAgent has been modified.

## Hot-weight versus rotating-weight diagnostic

The first geometry microbenchmarks reused one expert selection. A follow-up
rotates through 40 selections initialized by the 64-draw screen, traversing
well over a GB of expert weights. Both arms use the same 40 selections and
same dispatch counts. A matched hot control uses the same inline-ID binding
mechanism while repeating one selection. Activations and shared output stay
fixed; this isolates expert-weight reuse, not the full model dependency graph.

T1 best kernel times: matched hot 64.018 → 61.807 μs; rotating 79.495 →
79.384 μs. Mean savings shrink from 2.340 to 0.567 μs. Thus much of the
isolated gain depends on weight reuse; this does not by itself fully explain
the native regression. The failed T1 unroll retained the original 3/2/2/2
lane mapping and did not eliminate the ninth-block imbalance.

G2 NR0=4/NSG=2 retains a small rotating-weight gain (best 79.126 → 77.183 μs).
G1 NSG=1 is nearly neutral there (best 79.744 → 79.375 μs). Full gates and
native integration remain pending for these standalone geometry candidates.

## P1 — complete down schedule with per-lane expert partials

Producer: independent expert slots, NR0=2/NSG=2, 1280×6 groups. Store six
32-lane partials per output row. Consumer: original Q8 shared-down arithmetic,
combine the six partials within each logical lane, then perform the original
SIMD reduction, shared BF16 boundary, block BF16 and ordered HC expansion.
This changes cross-expert accumulation association and is Tier 2. It differs
from old R4, which reduced lanes inside each expert and later added six scalars.
P1 adds 3,932,160 bytes of scratch writes and reads per layer. The graph has
one reusable scratch buffer. Unsupported routed paths retain ordinary routed
output and use separate shared-down fallback; no unproduced partials are read.

Kernel gate: 50,000 fresh draws, 256,200,000 untouched shared/pre words exact,
100% finite, no poison survivors or damaged guards, 50k/50k controls against
the saved candidate pre-control output, 99.7249% of touched control words moved,
1000/1000 deterministic repeats. Changed-output distributions:

| Output | Different words | Mean absolute difference | Maximum absolute difference |
|---|---:|---:|---:|
| Routed F32 | 76.5893% | 7.4003e-7 | 7.6294e-6 |
| Block BF16 | 0.0098098% | 7.4708e-7 | 0.25 |
| HC BF16 | 0.0025946% | 4.5565e-7 | 0.375 |

ULP histograms are in `lane-split-tier2-kernel.log`; BF16 histograms use
BF16 steps. Rare rounding-boundary crossings explain large BF16 absolute
steps; this kernel evidence is not a whole-model quality gate.

P1 complete-pipeline best times: matched hot 78.907 → 71.071 μs; rotating
expert weights 93.197 → 85.278 μs. Shared weights remain fixed in this test.
The larger-workset improvement survives, unlike most of T1's isolated gain.
Native speed, same-build 100-prompt scorer, depth, upstream drift and MTP
contract remain unproven. P1 currently affects scalar decode only: six-row
verification consistency must be addressed before adoption. It defaults off.

Host/CPU builds and the four retained/new Metal PSOs compile. The isolated
server is being replaced once for these runtime changes; all subsequent
arms share the same loaded model. Its predecessor manifest is preserved as
`server-bundle2-manifest.json`. The new `/debug/bench` replay option uses saved
token IDs so a Tier 2 speed test need not compare different generated text.
Replay responses explicitly identify forced replay and report greedy top-1
agreement; replayed text identity must never be called a fidelity pass.

## Tpack — actual ninth-block work packing (prepared, not yet screened)

NR0=4 groups retain the two full K-loop iterations. On the ninth block,
four 8-lane subgroups work on four different output rows simultaneously.
SIMD shuffles fetch each row's original logical-lane accumulator before the
update and return it to the same logical lane afterward. All six expert-slot
updates and the final reduction retain their reference order. This is Tier 1
intent, unlike P1; the shuffles and changed lane placement require fresh gates.
It addresses the lane imbalance that the rejected T1 unroll left intact.
No runtime integration or performance claim exists yet.

Rejected T1/D1 source and harness files have now been archived outside the
repository under raw `rejected-test-artifacts/`. The reusable geometry harness
is direct source, backed by `reference_source.py`; it no longer depends on a
rejected experiment's generator. P1 reference assembly also stands alone.
Reports and failure logs remain tracked. Native A/B resets P1 off explicitly
when comparing the exact A1/R1 arms, avoiding inherited diagnostic lever state.

## P1 native replay screen complete

Four interleaved 256-token pairs, one resident server, quiet checks before
every arm, cached 8k prefix. All requests replay the identical token IDs from
one 512-token greedy control capture. All 256 greedy predictions agreed with
the replay in every arm. This is an initial speed/greedy agreement screen,
not the 100-prompt quality gate or a final UAT comparison.

| Arm | Best t/s | Mean t/s | Paired mean delta ± SE |
|---|---:|---:|---:|
| Control | 33.9053 | 33.8508 | — |
| A1+R1 | 34.6042 | 34.5221 | +0.6713 ± 0.0318 |
| P1 | 34.5560 | 34.3564 | +0.5055 ± 0.0916 |
| A1+R1+P1 | 35.1929 | 34.9842 | +1.1333 ± 0.0778 |

The 35.1929 result is retained as demonstrated capability. No exclusions.
The aggregate is still below the materiality target. P1 remains off by default.

Tpack folded and plain 64-draw screens passed: checker self-tests, all output
words exact/finite, intact guards, all controls, 1000/1000 determinism. The
folded rotating-weight kernel improves by roughly 9 μs, substantially more
than the rejected unroll. Full 50k gates are now running sequentially, with
the resident server idle. Tpack is still a standalone candidate.

## Tpack full gates and controlled attribution

Both Tpack paths passed 50,000 fresh draws against immutable bbaebc1:
folded 1,536,200,000 words, plain 256,000,000 words; 100% finite, zero
mismatches/poisons, intact guards, 50k/50k controls, 1000/1000 determinism.
These are kernel gates; native Tpack integration/timing is still pending.

To isolate packing from the NR0=4 geometry change, `packed-vs-nr4.m` compares
the two four-row kernels on the same rotating expert selections. This is
an attribution diagnostic, not a replacement for the immutable fidelity gate.
Best unspecialized NR4 → packed NR4: 76.978 → 69.653 μs. All 64 screen
draws were identical. This establishes a kernel-level cost of the old
ninth-block work mapping; it does not assign the full native GLM gap to it.

Debug endpoint checks: missing replay file fails before fixture construction;
the retired T1 lever is rejected. Results are in `replay-validation.json`.
The native replay screen demonstrated correct token replay and unchanged
control/AR greedy predictions. Do not interpret forced text identity itself
as evidence of numerical fidelity.

Checkpoint status: experimental private branch only. Nothing is promoted to
UAT. A1/R1 native depth gates, P1 model quality/MTP contract, native Tpack
integration, aggregate materiality, default-on adoption, private promotion
and restoration of all stopped services remain outstanding. P1's scalar-only
coverage is an explicit adoption blocker until its verification contract is
resolved; it is not a reason to weaken that contract.


## Tpack native 8k: exact and useful, below materiality alone

Private experiment checkpoint: `51042f6`; no production/public promotion.
Tpack's folded kernel was copied token-for-token from the certified candidate
(only whitespace changed). The host restricts it to the existing scalar,
non-streaming fold hint, 2304 -> 5120, wide Q4, NSG=2; NR0 becomes 4 and the
grid is 640 groups. A one-time execution witness confirmed that selected path.
P1 takes precedence and is not silently combined with Tpack.

Four interleaved 512-token pairs plus separate 128-token warmups, one resident
server, all cached 8k prefixes, all quiet checks passed, all generated outputs
identical across all four arms. No observations excluded.

| Arm | Best t/s | Mean t/s | Paired mean delta ± SE |
|---|---:|---:|---:|
| Control | 33.7939 | 33.7637 | — |
| A1+R1 | 34.5651 | 34.4753 | +0.7115 ± 0.0375 |
| Tpack | 34.1978 | 34.1269 | +0.3631 ± 0.0337 |
| A1+R1+Tpack | 34.8848 | 34.7962 | +1.0325 ± 0.0274 |

The bundle's best-to-best gain is +1.0909 t/s. Tpack beats control in 4/4
pairs and adds to A1+R1 in 4/4. This does not erase the earlier valid P1
35.1929 result, which belongs to a different schedule. Depth gates pending.
Host build and all five current PSOs passed. Current server provenance is
raw `server-manifest.json`, predecessor `server-p1-manifest.json`;
transition `server-transition-packed.json`, raw log `server-packed.log`.

Ppack is now prepared as a separate candidate: Tpack's per-row arithmetic
inside P1's independent expert producer. Its harness compares all 983,040
expert/lane partials against P1 at immutable `51042f6`, before any consumer
reduction. This isolates packing fidelity from P1's Tier 2 association.
The first preparation command lacked DEVELOPER_DIR on the Python process;
reference extraction failed before GPU work. Retried with the correct toolchain.


## Ppack contraction diagnosis and full per-lane gate

The initial direct Tpack-to-P1 port failed the per-lane screen: 3,463,098
mismatches in 50 draws, all in lanes 0..7 (the ninth-block owners), no poison
survivors or guard damage, all 50 controls detected. Recomputing both arms
on the same post-control input confirmed the divergence independently of the
planted control. The first mismatches were one/two F32 ULPs.

Six explicit contraction variants were screened against unchanged P1 at
`51042f6`. Variants 2 and 4 passed; 0/1/3/5 failed. Variant 2 makes the tail
update's outer FMA explicit, preserving the P1 reference contraction. Raw
rejected candidate/variant sources are outside the repository; logs are retained.
The correction is specific to the expert-partial producer; the existing Tpack
six-expert folded kernel is unchanged and retains its previous certificates.

Corrected Ppack full gate: 50,000 draws, 49,152,000,000 expert/lane words,
100% finite, zero differing words/poisons, intact guards, 50k controls detected,
1000/1000 deterministic repeats. Best rotating-weight times are nearly neutral;
packing must not be assumed to add its scalar gain to expert parallelism.
The host now selects Ppack when both `down_lane_split` and `down_tail_pack`
are on; it preserves the partial-buffer layout and shared-down consumer.
All six PSOs compile. Native Ppack measurement is pending.

## Resident quality instrument

`/debug/score` runs score_official's chat rendering (THINK_NONE), tokenization,
FP64 teacher-forced NLL, first-match and greedy-prefix metrics on the same
resident server. `cache_create:true` explicitly creates the control prompt's
prefix once; every candidate must restore that disk snapshot. Token identity
is checked after restore. The endpoint is debug-only and uses the bench mutex.
No repeated cold model load or repeated per-arm prefill is needed.

An initial server build exposed that its -ffast-math flags would weaken the
scorer's finite checks. The score arithmetic is now in `ds4_score.c`, compiled
without -ffast-math, including the accumulated NLL. Against verbatim metric
functions from immutable bbaebc1 score_official: 10,000 finite draws have exactly
the same accumulated NLL/argmax; NaN/Inf and invalid target controls are rejected.
Metal server, CPU server object and server-unit-test builds passed; server
parser/render/cache unit tests passed. CPU/CUDA Makefile server/test link paths
include the portable scorer object. No CUDA execution is claimed.

Initial three resident control cases reproduce the frozen upstream reference
at its TSV precision (NLL differences <5e-10); first_match and greedy_lcp match.
P1 changes those NLLs, confirming that the instrument is exercising the candidate.
The three-case screen is not the 100-case gate. Full scoring is in progress;
`resident_score.py` reports all S1-S5 quantities using per-prompt avg_nll.


## P1/Ppack REJECTED: complete 100-prompt quality gate

All 100 control cases reproduce the frozen bd66c40-numerics reference at its
TSV precision, with identical prompt/target counts, first_match and greedy_lcp.
Every candidate restored its control prefix from disk; no candidate prefill.
The same resident binary produced both arms. P1 changes decode arithmetic;
prefill rendering and prefixes are identical. Results:

| GLM criterion | Result | Verdict |
|---|---|---|
| S1 paired mean / SE | −0.0008284494 / 0.0020081804; ratio 0.4125 | PASS |
| S2 absolute paired mean <= 0.0001 | 0.0008284494 | FAIL |
| S3 wins/losses/ties, binomial p | 56/42/2, p=0.1888467 | PASS |
| S4 first_match, avg_lcp | 88->88; 12.14->12.28 (allowed ±0.1) | FAIL |
| S5 absolute per-prompt upstream drift <=0.0003 | 0.0008284494 | FAIL |

NLL improves on average; rejection is because the numerical change exceeds
the stipulated limits, not because the quality score got worse. The gate is
not relaxed to retain the speedup. Ppack is per-lane identical to P1 and shares
its rejection. Runtime levers, scratch allocation, host hooks, GPU APIs and all
three P1/Ppack kernels are removed. Source/harness pre-images are archived
outside the repo in raw `rejected-p1-preimage/`; measurements and verdicts remain
here. `resident_score.py` is now a reusable per-lever instrument rather than a
hard-coded rejected implementation.

The live test server was explicitly reset to P1 off before subsequent tests;
receipt `p1-disabled-after-rejection.json`. It retains its pre-cull binary for
the exact depth comparisons so no model reload is needed. Its source/binary
manifest is preserved independently of the current culled source. The culled
Metal server and CPU objects build, and the three retained PSOs compile.
An initial PSO check accidentally used an older checker executable that still
requested deleted P1 kernels and failed; the rebuilt retained-PSO checker passed.

C2 next hypothesis: compute expert Q4 operands concurrently inside one threadgroup,
then apply them in the original expert/block order using on-chip staging. This
avoids P1's reassociated expert sums and off-chip scratch. `prepare_cooperative_down.py`
prepares four contraction variants against immutable bbaebc1. No compile,
fidelity, or performance claim exists yet; native depth measurements have priority.


## Exact bundle: completed native depth gates

Four interleaved 512-token pairs per depth, separate 128-token warmups, one
resident server. All arms restored the same disk prefix, never prefilling.
All quiet checks passed. Every continuation is identical between arms at each
depth, including all warmups. No observation excluded.

| Context | Control best / mean t/s | A1+R1 best / mean | A1+R1+Tpack best / mean | Bundle paired mean gain ± SE |
|---|---:|---:|---:|---:|
| 62,000 | 33.5312 / 33.4950 | 34.3522 / 34.2209 | 34.7578 / 34.6081 | +1.1131 ± 0.0537 |
| 300,000 | 32.1404 / 32.1105 | 32.7675 / 32.7239 | 33.1272 / 33.0820 | +0.9715 ± 0.0241 |

The exact bundle remains below the approximately 2 t/s target. Its MTP
regression checks and eventual default-on adoption/deployment are outstanding.
Cached agent fixtures were located; their paths/metadata are recorded in raw
`agent-fixture-sources.json` for reuse without a new prefill.

## C2 cooperative down: screens and storage refinement

The initial four-word-per-block staging version found one exact contraction
(mode 2) among four variants; modes 0/1/3 failed the 64-draw gate. Exact mode 2
was slower in the rotating-weight screen (best reference 80.264 -> 81.207 μs).
Its operands were stored as four floats, using 13,824 bytes of threadgroup memory.

The retained form packs the two existing half-precision Q4 scales into one
32-bit word, preserving their bits; the two FP32 operands remain FP32. This is
storage packing, not quantization. Three operand planes keep adjacent logical
lanes in adjacent words. The two-row group uses 10,368 bytes of shared storage.
NR1/NR2/NR4 screens all passed 64 fresh draws, all outputs/guards, all controls,
100% finite and 1000/1000 determinism. Best rotating-weight timings:

| Rows/group | Reference -> candidate μs |
|---|---:|
| 1 | 78.924 -> 74.075 |
| 2 | 78.890 -> 72.815 |
| 4 | 78.858 -> 95.266 |

Only the selected two-row form remains in the candidate generator; prior
variants are archived in raw `cooperative-preimage/`, with failure logs retained
here. The full 50k gate is running; no full-gate or native gain is claimed yet.
C2 and Tpack are alternative down kernels, so these gains must not be added.


## C2 full kernel gate completed

Selected NR2/NSG6, packed half-scale storage: 50,000 fresh poisoned draws,
1,536,200,000 output words, 100% finite, zero differing words/poison survivors,
intact guards, 50,000 controls detected and 1000/1000 deterministic repeats.
This is against immutable bbaebc1, including routed F32, block BF16, all HC
outputs and next-pre coefficients. No native C2 speed claim yet. The next
comparison is C2 versus Tpack and control on the resident fixtures.

## C2 native verdict — rejected in favor of Tpack

Four interleaved 512-token runs per arm, cached 8k prefix, quiet checks, all
outputs identical. Separate 128-token warmups are retained in raw JSON.

| arm | best t/s | mean t/s | paired gain vs control ± SE |
|---|---:|---:|---:|
| control | 33.8333 | 33.796225 | — |
| A1+R1 | 34.5108 | 34.460725 | +0.664500 ± 0.023422 |
| A1+R1+Tpack | 34.8498 | 34.813275 | +1.017050 ± 0.021980 |
| A1+R1+C2 | 34.7240 | 34.642200 | +0.845975 ± 0.028947 |

C2 loses to Tpack in all four pairs, mean −0.171075 t/s. It is an alternative
to Tpack, not additive. Culled C2 runtime/harness source is archived outside
the repository in `rejected-c2-preimage/` with hashes; results remain tracked.
The running server may still contain C2 until the next planned warm restart.

## P2 — three consecutive expert pairs

New hypothesis after P1 rejection: compute expert pairs (0,1), (2,3), (4,5)
independently, then combine three per-lane partials in shared down before the
original SIMD reduction. Scratch is 1,966,080 bytes (half P1). This still
changes floating-point association and must pass the full Tier 2 gate.
It is not a revival of P1's unchanged six-way reduction.

50k fresh poisoned draws against immutable bbaebc1: 256,200,000 untouched
words exact, 100% finite, controls 50,000/50,000, determinism 1,000/1,000.
Touched mismatch fractions: routed F32 0.755887863, block BF16 0.000094250,
HC BF16 0.000024893. Full absolute/ULP distributions in pair-down-tier2-kernel.log.
Rotating-weight pipeline screen: control best 94.458 us, candidate 88.053 us.
These are kernel-screen observations, not native speed or quality acceptance.
Review caught and corrected an initial scratch declaration in the prefill
macro and duplicate header declarations before building or running the model.

## P2 native verdict — rejected, no full quality claim

Four interleaved 512-token runs, all cached, quiet checks retained.
Control best/mean 33.8251/33.767025; exact A1+R1+Tpack 34.8587/34.806225;
A1+R1+P2 34.9262/34.847550 t/s. P2 minus Tpack mean +0.041325 t/s
(three positive pairs, one negative), best-to-best +0.0675 t/s.
P2 changes the continuation, so this is free-running workload performance,
not an identical-token kernel comparison. No full quality score was run: the
small gain over the retained exact alternative does not justify adoption.
Rejected P2 runtime/harness source is archived outside the repo under
`rejected-p2-preimage/` with hashes. All results stay in the ledger.

## E1 — verifier residual epilogues

At two/six verify rows, the current attention path rounds the output block,
expands HC, then rounds residuals. The FFN path additionally adds routed and
shared outputs. E1 combines those operations into one dispatch per sublayer,
retaining block materialization, BF16 boundaries, and the shipped HC expression.
It changes neither weight projections nor MTP admission.
Reference harness extracts add, BF16 rounding and HC expansion kernels from
immutable bbaebc1. Initial six-row add screen is exact and finite, controls64/64,
determinism1000/1000; complete sequence ~15 us -> ~6.7 us. Full kernel gates passed: 50,000 draws for each of rows2/6 × add0/1,
20,480,000,000 words exact, 100% finite, 200,000/200,000 input controls,
1,000 deterministic repeats per case. Host build passes; native MTP gate next.

## E1 cached code MTP screen

All four 512-token pairs and separate128-token warmups restore disk KV.
Output text,117cycles,702verified rows,511committed tokens and the full
acceptance histogram agree between all three arms on every measured pair.
All quiet checks pass; no controller policy changes.

| arm | best t/s | mean t/s | paired gain vs control ± SE |
|---|---:|---:|---:|
| control | 44.7309 | 44.458275 | — |
| A1+R1+Tpack | 44.5155 | 44.403850 | −0.054425 ± 0.108532 |
| A1+R1+Tpack+E1 | 44.7076 | 44.594475 | +0.136200 ± 0.105440 |

E1 versus the exact bundle is +0.190625 t/s mean, positive in4/4 pairs;
full paired details in verify-expand-native-contract.json. The control's
fastest observation44.7309 remains valid and exceeds the full arm44.7076.
This is a small candidate, not a material gain or a deployment gate. Cached
agent MTP testing follows on the same binary. CPU object build also passed.

## Q8 NR1 — rejected screening geometry

The prior campaign tested NR4/8/16. This distinct screen instead changes
two output rows/group to one, preserving NSG4, logical K assignment, and
immutable bbaebc1 reduction/template bodies.64 activation draws per shape
are exact; this is only an initial screen, not a Tier1 certification.
Eight rotating weight copies are used for timing.

OutB~67->66.5us is a tiny difference; shared-down~21.5->21.4us and shared
GU~20.8->20.8us are neutral. Query-B loses decisively~64.5->76.9us.
No runtime installation; no full fidelity/adoption claim. Source and harness
archived in rejected-q8-nr1-preimage; all raw screen timings retained.

### Agent fixture warmup coverage failure

The first full-arm512-token agent run is retained at31.3609 t/s, versus
38.2699 control. Its proposal phase is4195.192ms versus1231.965ms; verify
is11877.712ms versus11945.229ms (candidate faster). Outputs,135cycles and
acceptance histogram match. The128-token warmup has no serial fallback;
the512-token run has one. The first Tpack dispatch witness appears during
this fixture, consistent with first-use scalar pipeline compilation inside
the proposal phase. The phase attribution is observed; compile attribution
is an inference, not an instrumented compile-time measurement.

Do not discard or relabel that measured run. Finish the four pairs, then
perform a new comparison with explicit512-token warmups for each arm.
The driver now exposes --warmup-tokens and --verify-rows for this purpose
and subsequent two-row coverage. No model reload or manual prefill is needed.

## M1 — ninth-block packing in the six-row verifier down kernel

Prepared a new standalone candidate by applying the certified scalar Tpack
logical-lane mapping to the immutable bbaebc1 six-row down-wide kernel.
Four output rows per SIMD group; the interleaved tile becomes128 groups so
it still spans1024 output channels. Expert and K accumulation order and the
final SIMD reduction are intended unchanged. This is not runtime-integrated
and has no fidelity or performance verdict yet. Screen no/partial/full expert
overlap only after the active native timings finish.

### E1 agent follow-up completed

The original agent comparison is retained in full: control best/mean
38.3086/38.26155, candidate38.4486/36.64895 t/s, including the slow first
candidate observation. All outputs and acceptance data agree.

New comparison with512-token warmups per arm and four512-token measured
pairs: control best38.2438,mean38.23305; full bundle best38.4608,mean38.42020.
Paired gain +0.18715±0.015688 t/s SE,4/4 positive. Full outputs,135cycles,
acceptance histogram, verified and committed counts match on all pairs.
This supports a small E1 gain, not materiality. Full-depth/two-row gates
and deployment remain outstanding. Both protocols and every observation
are retained separately; no result was deleted as an outlier.

M1 initial compilation failed because QK_K is undefined after the immutable
library's final include. Fixed the candidate's local block-width constant
to256; retained the failed compilation log. No model process was changed.

### M1 screen passed; full gate running

64 activation draws each for no/partial/full overlap, complete outputs exact
and finite. Native microbenchmarks (not full-model claims): no overlap~375→365us;
partial overlap~338→318us; full overlap~313→286us. The existing1024-channel
interleave is retained with128 NR4 groups rather than256 NR2 groups.

Full gate now regenerates all36 experts' weights and all36 activation rows
on the GPU for each draw; poisons both outputs/guards before each execution;
checks every word against immutable bbaebc1; plants whole-input sign-flip
controls every draw; poisons deterministic repeats.50,001 draws total over
three overlap cases,1000 identical-input runs per case. No runtime M1 hooks
are present yet; wait for this gate before integration and native A/B.

### M1 full gate passed; runtime diagnostic integrated

50,001 fresh poisoned draws across three overlap cases:1,536,030,720 exact
output words,100%finite,50,001 whole-input controls detected,1000/1000
determinism percase. Log mtp-tail-tier1.log; exec45393 confirmed exit0.
Only whitespace cleanup followed the tested candidate body.

Diagnostic `mtp_down_tail_pack` defaults off; guarded to existing six-row
Q4 down-wide selection,384experts,5120→2304→5120,nonstreaming,single GPU,
nonquality and original NR2 metadata. It sets a local copy of dispatch
metadata to NR4,640×6 groups,2SIMDgroups. Other routes retain original args.
Server build, CPU object, all five retained Metal PSOs pass.
One warm restart enables resident A/B; all arms share the same model process
and disk-prefix restores, with512-token warmups per arm before4×512 trials.
No UAT or public changes have been made. Materiality is not established.

### M1 cached native code comparison completed

Four interleaved measured 512-token trials per arm, after separate 512-token
warmups, same resident process and disk-restored 8k prefix. Every arm matches
full token IDs, text hash, cycle/committed/verified counts, acceptance histogram
and serial-row count. No performance prefill. Quiet samples retained per trial.

| Arm | Best t/s | Mean t/s | Paired mean gain vs control |
|---|---:|---:|---:|
| Prior UAT control | 44.3410 | 44.32085 | — |
| A1/R1/Tpack/E1 | 44.6324 | 44.541475 | +0.220625 |
| A1/R1/Tpack/E1/M1 | 45.0791 | 45.02455 | +0.703700 |

Bundle best-to-best gain +0.7381 t/s; paired mean SE 0.017755 t/s.
M1 incremental mean gain over the other candidates is +0.483075 t/s,
positive in all four pairs. Retain for further agent/depth validation.
This does not establish the approximately 2 t/s goal. No deployment occurred.
See mtp-tail-native-summary-8192.json and mtp-tail-native-contract.json.

### M1 cached agent comparison completed

Same full 512-token warmups and four 512-token measured pairs, three arms.
Control best/mean 38.2643/38.20950 t/s; prior candidate bundle
38.4449/38.363275; full bundle with M1 38.8594/38.74415.
Best-to-best gain +0.5951 t/s; paired mean +0.53465 ±0.027543 SE.
M1 increment +0.380875 t/s, positive in every pair. Full tokens, acceptance
histogram, cycles, committed/verified counts and serial rows are identical.
All performance prefixes restored from disk; no prefill. Depth validation
and materiality remain outstanding. See mtp-tail-agent-{summary-8192,contract}.json.

## N1 — shared-down / routed-GU overlap (diagnostic, unvalidated)

The previous S15 screen approximated shared GU with a 5120→4608 projection.
This experiment uses the actual 2304→5120 BF16-rounded Q8 shared DOWN and
immutable production Q4 group6 GU kernels. Both inputs are ready after
router/shared/HC and selection. Dispatch shared down and routed GU in one
concurrent encoder, then close it before routed down consumes both outputs.
No shader math, rounding, weights or expert ordering changes.

Runtime prototype is restricted to scalar resident single-GPU DS41; only
the fused group6 GU route can remain concurrent. Other routes close the
scope first; the graph caller also closes it on any routed-call failure.
The existing generic parallel FFN and profiling routes are not admitted.
New diagnostic shared_down_overlap defaults off pending validation.
Screen, runtime correctness and native throughput verdicts are pending.

### N1 screens passed; native comparison running

Immutable production shader screen: 64 draws, 2,981,888 exact finite output
words including BF16 shared-down, guards intact. The original serial schedule
measured 131.815–136.681 us and concurrent 129.820–132.304 us over alternating
rotating-arm runs (weights cache-hot in this screen).
Production-wrapper test then exercised shared-down, routed GU/down and folded
HC, with changing inputs/routes/weights, poisoned outputs and changed-input
controls: 64 draws, 2,294,016 exact finite chain words; 64/64 controls detected.
This is an initial scheduling screen, not a completed Tier 1 certification.
Metal/server and CPU-object builds passed.

The resident cached 8k native test has started after one warm mmap restart.
Its first measured pair loses ~0.42 t/s versus ARP. Finish all paired trials
before disposition; do not infer a throughput gain from the kernel screen.

## Q1 — four logical Q8 K groups in one physical SIMD group

New hypothesis: the 72-block shared-down and 40-block query-B dimensions
leave only one of four physical SIMD groups active on the final K tile.
Keep all four original accumulators per output row, including each lane's
ordered K walk, but compute them in one physical SIMD group. Assign the
four physical groups to four independent output rows. Reduce each logical
accumulator with the original SIMD sum, then place the four results into
lanes 0..3 (zeros elsewhere) and perform the original second SIMD sum.
This removes cross-SIMD scratch/barriers and packs tails across output rows.
It differs from rejected Q8 NR1, which retained the original physical K map.

A standalone candidate and immutable-reference harness are prepared. The
first screen will compare all output bits, poison/finite/guard checks and
forty rotating weight sets at both production shapes. No runtime Q1 code
is installed and no numerical or speed result is claimed yet.

### N1 rejected and culled

Four measured pairs all lose versus the retained ARP bundle. ARP best/mean
34.8301/34.775225 t/s; ARP+overlap 34.4124/34.380125. Increment
-0.395100 ±0.026220 t/s SE; best-to-best -0.4177.
All 512-token outputs exact, all prefixes restored, quiet checks passed.
The extra encoder boundaries are a plausible explanation, not an isolated
measured attribution. No full 50k gate was run after the negative speed verdict.

Removed runtime lever/APIs/hooks and standalone implementation/harness sources.
Preserved source pre-images plus SHA256 manifest in raw rejected-n1-preimage/.
Measurements, build and runtime-screen logs remain tracked. The resident
experimental binary still contains N1, explicitly disabled and verified through
/debug/levers; its server-manifest identifies that exact binary/source build.
Source now omits N1. Do not conflate the live manifest with later source files.

### Q1 initial screen passed

64 random activation draws at each production shape: every output exact and
finite; poisoned buffers/guards checked. Forty rotating weight sets.
Shared down's final observed pair: 21.941→20.412 us; query-B
65.438→64.088 us. Earlier slower observations retained in full; see
q8-virtual-screen.log. These are microbenchmarks, not native throughput gains.
Further full numerical gates and integration are pending.

### Q1 broader shape screen: retain short shapes, reject large shapes

All four shapes passed 64 exact finite draws. Rotating forty weight sets:
short shapes repeated the gain (shared down ~22.55→20.93 us, query-B
~66.14→64.02 us). Attention-high 8192→5120 regressed ~66→78 us;
5120→2304 projection regressed ~21.55→26.14 us once warm.
Large-shape implementations were removed and archived with a manifest under
raw rejected-q1-large-preimage; the full log remains tracked. No native
claim from these screens. Only 2304/1280 input widths advance to the gate.

### Q1 gate infrastructure correction and runtime preparation

The first GPU comparison helper reported only one finite output per draw
instead of counting every word (25 versus 128,000 in the first batch), while
bit comparisons/poison/guards passed and sampled outputs were finite.
The failed log and diagnostic rerun are retained. Replaced the per-word
float-isfinite/device-counter updates with integer exponent-bit checks, local
counts and atomic totals, matching the established gate pattern. Added a
comparator self-test that deliberately plants infinity, poison, an altered
finite word and a damaged guard; it must detect the exact expected counts.
All four shape/round-mode self-tests and 64-draw gates then passed, with
1,000 deterministic poisoned repeats each. Full 50k/case gate is running.

Prepared narrow q8_virtual_down and q8_virtual_query diagnostic hooks, both
default off. Shared down calls the same validated range/binding helper with
the candidate pipeline, NR4 and no scratch. Query-B retains the baseline
HC body verbatim and uses eight independent query rows per 256-thread group
(4,108 total groups). A separate complete query/HC gate is prepared; it will
run after the standalone gate, without overlapping GPU jobs. Existing
resident process still has N1 disabled and no Q1; preserve its build manifest.
No Q1 throughput or complete Tier 1 verdict is claimed yet.

### Q1 full arithmetic gates passed

Standalone gate exited 0: 50,000 fresh-weight/input draws for each of the
two shapes in plain and BF16 modes; 3,788,800,000 exact finite words,
200,000 changed-input controls detected, 1,000 deterministic repeats/case.
Complete query/HC gate also exited 0: 50,000 draws, 1,640,850,000 exact
finite words including query outputs, HC mix/split and publication counter;
50,000 controls detected and 1,000 deterministic repeats. Poison and guards
passed throughout. Full logs accompany this entry. Native integration and
depth validation remain required before adoption.

Resident N1 process 61690 exited 0. Preserved its manifest as raw
server-shared-overlap-manifest.json. One warm mmap start now uses the Q1
binary, unchanged model weights and isolated disk fixtures. Colima VM is
confirmed stopped; its pre-existing foreground supervisor remains idle.
No llama.cpp process is present. No public/production deployment changed.

## K1 — scalar KV preparation fusion (screen prepared, not runtime)

The flat QA/KV projection makes KV available before query-B completes.
Its current path then executes weighted RMS/BF16, RoPE/BF16, FP8
quantization/BF16 and ring copy separately. Initial candidate fuses these
four KV operations in one 128-thread group, preserving the original four
SIMD RMS partials and second reduction, explicit BF16 boundaries, precise
RoPE expressions and per-32-value FP8 reduction. A later mixed query-grid
placement is only a hypothesis; it has not been implemented.

Prepared an immutable-production-reference screen, changing input, norm
weights and positions through 1M context, comparing every KV output word
and full ring/guard contents. The ring-copy reference is a screen-only copy
kernel, not a production-wrapper timing claim. Host harness compiles; GPU
execution waits until Q1 native timing ends. No full gate or runtime code yet.

### Q1 first native batch: shared down ran; query did not

Execution witness audit found hc_stream_layout=2 in all production-default
arms, while the query prototype required layout0. The query flag was set
but no query candidate executed. Preserve original arm names/raw JSON;
interpret this batch as ARP plus Q1 shared-down only. This is an integration
coverage failure, not query performance evidence. The 50k query/HC gate
used layout0; a new comparison against actual layout2 is required.

Four 512-token measured pairs: control best33.7526, ARP34.8713,
ARP+shared-down34.9323 t/s. Shared-down mean increment
0.085750 +/-0.008458 SE, positive4/4.
Bundle mean delta vs control+1.1743; best delta+1.1797. Full token IDs
and text exact, all cached prefixes and quiet checks. Materiality unmet.

### Q1 corrected layout coverage

Compared candidate against immutable production layout2 query/HC:
64-draw screen plus50,000 fresh GPU draws pass, 1,640,850,000 exact finite
words,50,000 controls,1,000 deterministic repeats. All guards/poison pass.
Cache-hot mixed-kernel screen includes equal output reset overhead:
layout2 ~47.0us vs candidate ~44.6us; not a native speed claim.
Runtime guard now admits exactly layouts0 and2. The prior resident binary
still has the old guard; no corrected-layout native measurement yet.

### K1 initial screen and integration preparation

128 fresh input/weight draws,131,072 exact finite KV/ring words; untouched
ring and guards passed. Micro separate chain ~14.7–17.5us vs fused
~9.2–10.5us. Reference copy is screen-only, no native inference yet.
Prepared scalar-flat-only hook: requires single GPU, no streaming/quality/
imatrix/batched-pointwise, production fused norm and F32 512-wide norm
weights. Preserves original RoPE frequency construction in a shared helper
under precise host floating-point controls. Other RoPE callers retain the
same validation and expressions. Candidate uses the RMS argument ABI prefix
because norm.metal is assembled later. Full server/CPU builds and all eight
runtime Metal pipelines compile. Full K1 numerical and depth gates pending.

### K1 native exactness failure; diagnostic investigation

All four warmups completed with witnesses. Control, Q1shared-down and
Q1both produce identical512-token output. Adding K1 diverges first at
token245; native driver aborts before measured pairs. Its35.6266t/s
warmup is not credited as an exact-path improvement. Raw kv-prepare files
and failure log retained. K1 remains diagnostic/off by default.

Expanded fresh-GPU-input gate (real raw/YaRN frequency formulas and
input/output aliasing) passes128draws plus128controls and1000determinism,
then the50k run fails draw1102, position799391, word510:
reference bf400000 vs candidate bf300000. This disproves exactness of
the current K1 candidate. Investigate stage arithmetic before deciding
whether to repair or reject; do not promote or hide the counterexample.

### Q1 layout2 native results

Four512-token measured pairs after512-token warmups, current production
layout2 throughout. Execution witnesses confirm both Q1 kernels.
Control best/mean33.7433/33.707675; ARP+Q1down34.8831/34.859175;
ARP+both35.1165/35.05315 t/s. Query increment mean+0.193975, positive4/4.
Bundle vs control pairedmean+1.345475 +/-0.033525 SE; best+1.3732t/s.
All fulltokenIDs/text exact, diskrestore/quiet checks passed, K1off in every
arm. No observed fast result discarded. Depth checks pending; materiality
still unmet. This is an experimental result, not UAT deployment.

### K1 stage localization and normalization-loop repair candidate

Stage diagnostic finds first intermediate mismatch at draw907, position
925575, norm word10: reference3eef0000 vs candidate3eee0000. Therefore
normalization already changes before RoPE/FP8. Simplifying the dynamic
production dot loop to a straight-line dot is a compiler-association
hypothesis; no disassembly attribution is claimed.

Restored both original runtime-bounded normalization loops, including the
threads-per-threadgroup stride, in the standalone candidate. Stage-by-stage
2,000-draw alias test now passes norm, RoPE and final outputs (2,048,000
final exact finite words),2,000 controls and1,000 deterministic repeats,
including both recorded counterexamples. Full50k repaired-candidate gate
now running. Runtime shader and resident binary STILL contain the original
failed K1; keep its lever off until the repaired source is integrated, built
and the native exactness test rerun. No repaired K1 speed claim yet.

Repaired normalization loops are now copied into the runtime shader source
on disk. The already-running server still uses the old compiled shader,
with K1 disabled. Full runtime library will be checked after the standalone
GPU gate finishes, then one warm restart will exercise the repaired path.

### K1 repaired kernel full gate passed

50,000 fresh GPU input/weight draws with production input/output aliasing,
51,200,000 exact finite KV/ring words;50,000 changed-input controls detected;
1,000 poisoned deterministic repeats; untouched ring/guards exact. Both raw
and YaRN frequency formulas, positions0 through1,048,575 sampled.
Full runtime Metal library and all eight pipelines compile. Original failing
straight-line normalization is no longer in candidate/runtime source; its
pre-image and failure evidence are retained. Native repair validation pending.
Stopped resident63710 cleanly and preserved its manifest. Warm mmap start
now loads the repaired runtime shader using unchanged original weights and
isolated disk KV; production/public branches untouched.

### K1 repaired native result

Four512-token measured pairs, separate512-token warmups. Full token IDs
and text exact in every arm and pair, all prefixes restored and all quiet
checks passed. Control best/mean33.7834/33.716175; ARP+Q1both
35.0364/34.97205; adding repaired K1 gives35.5945/35.5558 t/s.
K1 increment pairedmean+0.583750, positive4/4. Full bundle pairedmean
+1.839625 +/-0.031633 SE over bbaebc1-control; measured best-to-best
+1.8111t/s. Warmups retained separately (control33.9282, Q135.0059,
K135.6109); none discarded as unusually fast. Do not claim the2t/s goal
complete. Depth validation has started at62k/300k in the same resident
server, two arms, four pairs/depth, full warmups and existing disk caches.
Controller and original weights remain unchanged. No UAT promotion.

### Repaired K1/Q1 bundle: long-context native validation

Four measured pairs per depth, 512 tokens per arm and separate 512-token
warmups, same resident server. All full token IDs and text match; disk
restores and quiet checks pass throughout.

| Context | Control best / mean | Bundle best / mean | Paired gain (SE) |
|---|---:|---:|---:|
| 62k | 33.6646 / 33.573350 | 35.4128 / 35.350625 | +1.777275 (0.029352) |
| 300k | 32.2513 / 32.146375 | 33.8670 / 33.819800 | +1.673425 (0.049567) |

Every measured pair improves. Preserve all trials and separate warmups.
Together with the 8k results, scalar depth validation passes; materiality
remains below 2 t/s and MTP/release validation is outstanding.

## K2 — wider scalar KV preparation (screen prepared)

Use 256 threads instead of 128. The first four SIMD groups retain the
original RMS partials; the additional four contribute zeros. Keep the
dynamic normalization loops that fixed K1. Distribute 16 independent FP8
blocks over eight SIMD groups with stride eight, avoiding duplicate writers.
Prepared immutable-reference stage gate and a direct K1-versus-K2 timing
screen. No runtime hook or speed claim yet.

## K3 — KV preparation inside the query/HC grid (shader prototype only)

Add one independent KV task after the 4,108 existing query/HC groups.
Reuse dynamic scratch (2,176 bytes) instead of adding a static scratch
allocation to every query group. This could hide KV preparation under query
weight reads and remove its standalone dispatch. Prototype shader generated;
complete combined gate and runtime integration are not yet implemented.

### K2 screen and full gate passed

2,000 stage-by-stage draws compare normalization, RoPE and final outputs
against immutable kernels, including input/output aliasing. Full 50,000-draw
gate then passes 51,200,000 exact finite KV/ring words, 50,000 controls,
1,000 deterministic repeats and all untouched ring/guard checks. Direct
cache-hot K1-versus-K2 timing screen (no alias-copy overhead) measures
K1 ~9.84–10.55 us and K2 ~7.65–8.83 us. Native increment remains unmeasured.

### K3 combined screen and checker throughput correction

Combined query/HC/KV screen passes 64 draws, all controls and 1,000
deterministic repeats. Position8192, with the entire 128-slot ring guarded;
KV aliases input/output in both arms. Reference uses the current Q1 query
plus certified K1; candidate adds KV as one independent mixed-grid task.
Micro screen including equal reset/copy overhead: separate ~76.9–77.6 us,
mixed ~66.9–68.4 us. These are not native throughput gains.

The first full gate against immutable layout2 query/HC plus certified K1
passed at least11,000 draws, then was deliberately stopped: the one-thread
checker scanning the full ring made validation unnecessarily slow. Preserved
partial log; this was not a numerical failure. Replaced the checker with
256-way integer comparison and exact integer reductions, retaining every
output and untouched ring word. Added self-tests injecting a changed word,
infinity, poison, a damaged guard and a KV/ring mismatch; exact expected
counts pass, including a single control count per draw. The 64-draw screen
and 1,000 repeats pass with this checker. Restarted the full 50k gate from
zero; no tested model arithmetic changed for the checker optimization.

Prepared diagnostic runtime flags kv_prepare_wide and kv_query_mix, both
off. K3 is restricted to K1 eligibility plus deferred HC and the validated
Q1 query layouts0/2. It skips the standalone KV kernel only after successful
mixed-grid encoding. Existing router/shared and other attention paths pass
null optional KV state. Server and CPU-object builds pass; runtime shader
compilation and native tests wait for the full numerical gate.

### K3 full combined gate passed

Parallel checker self-tests pass. Full 50,000 fresh GPU weight/input draws:
1,692,050,000 exact finite words spanning query, HC mix/split/counter, KV
and written ring row; all other ring words and guards preserved. All 50,000
controls detected, alternating query and KV input changes; 1,000 poisoned
deterministic repeats pass. Reference is immutable layout2 query/HC plus
certified K1. The preceding K2 full gate separately validates the 256-thread
KV arithmetic at varying raw/YaRN positions through 1M context.
No native throughput claim until the new resident comparison completes.

### K2/K3 runtime ready for native comparison

Server and CPU-object builds, full runtime Metal library and all ten PSOs
pass. Preserved the repaired-K1 resident manifest, stopped PID64493
cleanly, and performed one warm mmap start of the combined build. No model
weights, original launch plists or disk snapshots changed. Native 8k A/B
now compares control, the K1 bundle, standalone K2 and mixed K3, with four
measured pairs and separate full warmups. Every arm resets all ten switches.

### K2/K3 native 8k comparison completed

Four measured 512-token pairs plus separate full 512-token warmups. All
full token IDs and text agree across every arm and pair; all quiet checks
pass. Runtime witnesses confirm 128-thread K1, 256-thread K2 and the
4,109-group query/HC/KV mixed path. Native driver exited successfully.

| Arm | Best t/s | Mean t/s | Paired gain over control (SE) |
|---|---:|---:|---:|
| Prior-UAT control | 33.7219 | 33.693775 | — |
| ARP + Q1 + K1 | 35.6311 | 35.540425 | +1.846650 (0.022485) |
| Standalone wider KV (K2) | 35.6597 | 35.637350 | +1.943575 (0.004971) |
| Mixed query/HC/KV (K3) | 35.9347 | 35.906750 | +2.212975 (0.024198) |

K3 improves all four measured pairs and exceeds the additional 2 t/s target
at 8k. Best measured trial improves by 2.2128 t/s. Separate warmups remain
in the raw log (control 33.7892, K3 35.9274); none discarded for being fast.
K2 is positive but superseded by K3 on the same eligible path. Long-context,
MTP, release-controller and deployment validation remain outstanding. This
is an experimental result, not a UAT promotion or completed goal.
Raw evidence: kv-query-native.log and corresponding native JSON artifacts.

### K3 scalar depth results and release preparation

Four512-token measured pairs per depth plus full512-token warmups. The full
contract audit compares all token IDs/text, requested levers, disk-restored
prefixes and quiet receipts:20/20 cases pass.

| Context | Control best / mean | K3 bundle best / mean | Paired gain (SE) |
|---|---:|---:|---:|
| 62k | 33.6046 / 33.575700 | 35.7874 / 35.714050 | +2.138350 (0.026589) |
| 300k | 32.1208 / 32.085250 | 34.4069 / 34.175075 | +2.089825 (0.065449) |

All four pairs improve at both depths. The300k warmup reached34.8126 t/s;
it remains in the record as a valid observation, separately from the
predetermined measured-pair summary. It is not discarded for being fast.

Prepared release defaults on for all nine exact feature switches. The K2
standalone kernel beat K1 by +0.1415,+0.1281,+0.0216,+0.0965 t/s in the
four8k pairs, so it becomes the fused-KV fallback when query mixing is not
eligible. Removed the redundant narrow runtime kernel and width selector;
retained both harnesses and measurements. K3's shader arithmetic is unchanged.
No production deployment has occurred; remaining tests gate adoption.

The fourteen previous agent API cache files had been evicted from the live
cache. All were found in the preserved user-speed replay cache and APFS-cloned
into the isolated cache (4,456,535,501 bytes logical size), with source metadata
and header checks recorded in api-cache-clones.json. No manual prefill or
original-cache writes. Prepared a diagnostic-only API guard to refuse an
incompletely cached long prompt before prefill; short ordinary API prompts
retain the prior serving protocol. Normal-controller replay scripts preserve
prior UAT Q8/SwiGLU rounding, use the same14 requests and predetermined first-
arm repeats, and never execute emitted tools.

### Combined six-row code/depth release matrix passed

Four512-token measured pairs plus512-token warmups at each depth, same resident
build. Full token IDs/text, acceptance histograms, cycles, committed/verified
and serial work counts agree across arms. All30 cache/quiet/lever contracts pass.

| Context | Control best / mean | Full bundle best / mean | Paired gain (SE) |
|---|---:|---:|---:|
| 8k | 44.3745 / 44.312350 | 45.0159 / 44.959550 | +0.647200 (0.033339) |
| 62k | 48.4129 / 48.397875 | 49.1804 / 49.144075 | +0.746200 (0.010348) |
| 300k | 51.8142 / 51.744700 | 52.4337 / 52.410250 | +0.665550 (0.028543) |

These are different cached prompt texts at each depth. Throughput across rows
of this table therefore also reflects acceptance:8k commits4.3675 tokens/cycle,
62k4.8667. Compare matched arms within a depth. Do not replace the published
normal-controller code/agent metrics with this fixed-six diagnostic.

At8k, pooled verifier time changes88.4244->86.9866 ms/cycle. With the unchanged
37.657 GB logical-weight ledger this is425.87->432.91 GB/s. These are the same
weight-normalized calculation used previously, not hardware DRAM counters.

### Owner closeout instruction

The owner superseded further optimization work with: finish the current
experiment, consolidate gains, resolve any MTP timing issues and deploy UAT.
The queued native matrix is the final experiment. The extended14-request
normal-controller replay is deferred in favor of a bounded code/prose and
107k-context agent serving check, forced-reject contracts, existing regression
tests and deployment. No new optimization candidates will be introduced.
The controller already begins each request with fresh serial timing evidence
and a rolling-nine-sample median; no stale fixed timing calibration was found.
Any timing-policy edit requires an observed serving issue, not an assumption.

### Final queued native matrix and build checks passed

Two-row coverage:18/18 full-output, acceptance-work, cache/quiet contracts
pass across8k/62k/300k (two measured pairs and512-token warmups per depth).
Six-row agent:10/10 contracts pass. Control best/mean38.1854/38.1521;
full bundle38.8127/38.7631, paired+0.6110(SE0.021210), positive4/4.
All queued native experiments are complete; no additional candidates started.

Final default-on source builds all Metal executables and CPU objects.
Existing decode-bandwidth, controller, adaptive-admission and Markov-cache
regression tests pass. Source comparison confirms retained shader bodies are
unchanged from the numerical/native gates; only the redundant narrow KV body
was removed. The release resident starts successfully with all nine new exact
features on and original UAT features/controller enabled. Forced rollback and
the bounded normal-serving check are the remaining predeployment checks.

### Release rollback and normal serving checks passed

Final default-on executable: forced-reject rows2/6 agree with serial for all
128 tokens at8k/62k/300k (nine cases). The diagnostic long-API cache guard
refuses an uncached prefix before prefill, as intended.

Bounded normal serving:11 requests (code warmups +ABBA, cached107k agent ABA,
and prose AB) match prior UAT text, phase hashes and completion counts. All
agent prefixes fully restored. Zero drafter faults. Code measured server decode
47.10/47.13 t/s control versus48.74/48.72 candidate (means47.115->48.730).
The controller records the faster serial timing itself: code warmup median
28.862->27.349 ms. Code retains17 cycles/6 declines; the agent retains47 cycles/
12 declines; prose retains0 cycles/18 declines and decode time15.163->14.224s.
No timing-policy change was necessary. The107k agent's decode wall time is
10.322/10.306s control versus9.935s candidate. This is a bounded serving check,
not a new14-request answer-phase aggregate.

All requested closeout gates pass. Private promotion and service restoration
are the next actions; the implementation will default exact gains on.

### UAT deployment and restoration verified

Implementation723e156 is committed and pushed to private/v41-mtp-prod. The
production executable matches release-runtime.json byte for byte; the original
LaunchAgent starts it from the production worktree. Diagnostic routes return404.
Model/switch/bridge/UI, vision wrapper/backend and both game HTTP endpoints
return200. All three worker containers are running. A real bridge->worker->DS4
request returned READY, with zero tool calls/errors and zero workspace changes.
Original service plist bytes match their saved copies. Preserved API cache
source headers/inodes/mtimes remain unchanged. Rollback instructions and the
original executable are retained.

One restoration checker initially tried HTTP on the NDJSON worker TCP ports;
it exited before the canary. Corrected the check to TCP connectivity and the
real bridge request. This was a verification-script error; no worker/model
change was needed. The corrected restoration receipt records this explicitly.
