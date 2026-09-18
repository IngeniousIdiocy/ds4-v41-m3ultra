> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Second material decode campaign — 2026-09-15

Baseline: current UAT `6f4aff1a38c9cd3c26f6cf4d756a646f0a81db6b`, resident
PID 84032, binary SHA256 `03cec517859f6a3423dbabe9e9f1e5071f3a4ee47cca555fe342e239d64f7b29`.
Goal: another approximately 2 t/s over this build, in serial and/or MTP.
Prior-round gains do not count toward this round. No candidate adopted yet.

## Experiments (retain failures)

| ID | Change | Result | Disposition |
|---|---|---|---|
| S1 | Merge the serial Q4 gate/up K loops to share activation loads; preserve each projection's dot and reduction order. | 64 full-size randomized activation draws, all 2,654,208 gate/up/mid words finite and exact. After initial warmup, reference roughly 110–114 us; candidate 132–136 us. | Rejected: slower. Never installed in resident server. |
| S2 | Widen serial Q8 output channels per group from 2 to 4/8/16, preserving lane/K order. Four shapes, 8 rotated weight copies for timing. | 64 draws per shape: initial exact-output screen passes all seven variants. outB medians 66.88 us baseline versus 67.82/70.11/73.12 us for NR4/8/16. Shared-down 22.27 versus 22.22/23.29/27.65; query 64.91 versus 65.52/66.01/81.71; shared gate/up 17.44 versus 17.86/18.60/31.19. | Rejected as a speed candidate. NR4 shared-down is neutral, not a win. Never installed. |
| S3 | Replace reduction scratch zeroing/barrier/store/barrier with disjoint zero/partial writes and one barrier, retaining SIMD reduction inputs/order. | Same initial Q8 screen passes. NR2 times 66.89/22.12/64.76/17.50 us versus 66.88/22.27/64.91/17.44 across four shapes. | No useful standalone gain. Not adopted; combined/fused reductions remain a separate possible experiment. |

These are screening checks, not 50,000-draw adoption gates. Timing is native
microbenchmark evidence only; it does not establish a full-model t/s result.
All individual timing repetitions, including warmup variation, are retained in
`bench/decode-material2-20260915/` and the external artifact directory below.

## Tooling failures

- First S1 harness compile omitted the SDK root and failed to find Foundation.
  No benchmark ran. Corrected invocation explicitly uses Command Line Tools and
  `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`; no global Xcode/license change.
- First S2 Metal compile declared Q8 payload as `char` instead of `int8_t`.
  Pointer types disagreed. No benchmark ran; corrected signed-byte declaration.
  Complete compiler output retained in `q8-screen-compile-failed.log`.
- Initial live ledger setup found no static PSO symbols through `nm` on the
  stripped release binary. Its empty generated expression consumed the next
  command and failed. Mode/page initialization had completed, but encoder hooks
  had not. The process detached and stayed alive. Completed setup with preserved
  release dSYM and explicit/cache PSO bindings before any profiled request.
  `ledger-enable.log` retains the failure; completion log retained externally.

## Isolation and restoration

Saved current binary, shaders, plist and all 136 current `.kv` files using APFS
clones. Stopped the three previously running worker containers through Docker;
the Colima VM remains running. Stopped four recorded native front-end services;
paused 23 identified wallpaper/video/Claude background processes. Unlike the
previous campaign, no VM process is suspended. Exact process identities and
service/container states are recorded for restoration.

Three existing disk-prefix fixtures were cloned into the cache; no manual
prefill. Temporarily enabled the resident server's existing file-gated debug
routes. Model weights have not been reloaded. Ledger profiling is temporary,
with original encoder method implementations retained for rollback.

Artifacts and restoration ledgers:
`LOCAL_ARTIFACT/`.
Do not reuse previous-campaign pipeline pointers: this round binds PID 84032.
Before restoring UAT, disable/revert temporary pipelines and ledger hooks,
restore the captured service/container/process states, remove the debug gate,
and verify the full bridge → worker → DS4 path.


## Further screens

- S4: MTP Q4 down-projection ushort4 loads. 64 draws at each of three expert
  overlap patterns pass finite exact-word output checks. Native means:
  400.94→369.71, 377.95→348.19, 353.76→326.43 us. Constant shape / K-loop
  unrolling alone is neutral; combining them with wide loads adds no clear gain.
  Resident 8k fixed-six screen: candidate 40.8729/40.7889 versus restored
  baseline 40.5643/40.6058 t/s, same 512-token hash and 117 acceptance cycles.
  Retain as a small candidate (~0.25 t/s), not a material result. No adoption.
  Original PSO restored. The initial restore succeeded, but its receipt parser
  expected two pointer prints and saw a third from the global assignment.
  Verified the log's identical original/current pointers and detach, repaired
  receipt, and changed assignments to void so future receipts are unambiguous.
- S5: packed, correctly aligned eight-byte Q8 weight loads using packed_short4.
  All four initial 64-draw screens exact and finite. Slower: settled outB
  ~96→102 us, shared down ~31.6→35.0, query ~119→144, shared GU ~32→36.
  Rejected, never installed. Initial warmups remain in logs.
- S6a: direct M8×N32×K32 matrix kernel for six rows, local Q8 dequantization
  into FP16 and FP32 matrix accumulators. This changes rounding (Tier 2).
  Four finite randomized draws per shape show relative RMS difference ~3.05e-4
  against the current kernel (not a quality gate). Settled native timings:
  outB ~96→716 us, shared down ~31.7→257, query ~119→177, shared GU ~32.5→407.
  Rejected as a speed candidate; no full-model installation or fidelity claim.
  Follow-up S6b splits the long K loop into independent partial tiles to test
  whether the poor small-output-grid occupancy explains the loss.

Fresh instrumented resident profiles cover 100% of dispatches with no counter
pool exhaustion or resolution failure. They confirm routed GU and down remain
largest verifier families. Their absolute t/s values are NOT adoption timings.
Profiling hooks have been restored: all four method pointer comparisons true,
mode=0 and by_tg=0. No server reload.

## Additional screens S6b–S10

- S6b: split-K six-row MMA, 4/8/16 Q8 blocks per partial. All slower;
  eight-block settled reference/candidate us: outB 96.325/159.203,
  shared down 32.725/61.150, query 119.530/160.643, shared GU 32.150/61.091.
  Relative RMS difference remains about 3e-4. Rejected before quality gate.
- S7: shorten activation-register lifetimes by loading inputs per output/token.
  Exact 64-draw screens. Direct shared-GU variant ~32.01→31.32 us is a tiny
  candidate; outB ~95.57→95.12 is near neutral, shared down neutral, query
  ~118.56→163.75 loses. Volatile-load variant loses substantially. No installation.
- S8: Q8 elements per lane 2/4/16 changes the reduction tree. Four numeric
  screens are finite, relative RMS ~1e-7; all geometries lose or are neutral.
  Rejected. Both input-layout combinations and isolated original-layout runs
  retained; no quality/adoption gate.
- S9: six-row F16 Engram sharing loses ~901→1364 us; altered roundoff (~61%
  words differ, relative RMS 9.5e-8). Dot/sum-temporary variants ~2031/2052 us
  lose further. Six-row compact grid ~900→1302 us also loses. A separate
  **three-row compact grid** removes empty groups while retaining current
  arithmetic: four screened draws exact (zero differing words), ~898.39→852.68 us.
  Retain the latter as a small candidate only, requiring a host grid change.
  `rows6-contractoff` failed compilation because the pragma was not at the
  compound-statement start; it has no timing result.
- S10: pack exactly representable activation values into BF16/FP16 registers,
  with exact-F32 reload fallback. 64 draws per shape (including full-F32
  fallback inputs) exact and finite. Both slower: BF16 outB ~96→207 us,
  query ~119→1044; FP16 outB ~96→179, query ~120→815. Rejected.

No new candidate is adopted; resident pipelines and profiler hooks remain at
baseline between explicit tests. Small candidate screens are not added together
as a throughput claim: only the final combined full-model A/B can establish that.

## S11–S17: scheduling and host boundaries

- S11: route sort in two SIMD registers per lane, avoiding shared-memory
  cross-SIMD exchange. 64 draws at no/random/full overlap exact. Partial-overlap
  native ~507→501 us, no-overlap ~689→685, full-overlap ~436→435. Small candidate.
  Alternative stable rank-sort is slower (~508→515 us); rejected. Initial source
  extraction accidentally included an unrelated MXFP4 helper and failed compile;
  corrected by balanced-brace extraction. Both compiler failures retained.
- S12: exact float32 router projection, six scalar calls → one independent-row
  grid using the same existing kernel and reductions. 64 draws exact. Rotating
  weights ~53.5→23.5 us. Two/three-row register-sharing alternatives were also
  exact but offer no clear advantage over this simpler grid. Six-row sharing
  changes roundoff and is slower (~59→38 us) than independent-row batching;
  do not adopt it. Candidate `mtp_f32_rows`, default off.
- S13: exact GPU-generated sin/cos table for RoPE. 2,048 random position/input
  draws exact across six rows. Full table generation 5.61 ms; head64/row1
  ~2.95→2.74 us, head64/row6 ~3.10→2.87 us. Rejected as a priority: too small
  for 512 MiB plus new cache/binding machinery. No resident install.
- S14: compile existing GU/down pipelines with explicit max64 threads and
  subgroup-multiple hint. Exact screens; neutral timings for both families.
  Rejected. Compiler descriptor changed maxthreads1024→64 as requested.
- S15: native overlap of independent serial routed GU and a shared Q8 projection
  approximation, using concurrent encoder. 64 draws exact on routed outputs
  and Q8 output. ~157→148 us for combined pair. Candidate hypothesis only:
  production HC/shared SwiGLU pipeline and dependency boundaries differ, so
  this is not yet a host implementation or a throughput claim.
- S16: vocabulary head chunk sizes 1/2/4/8, and six-row sharing instead of two
  three-row tiles. Eight numeric draws each exact. All alternatives lose:
  baseline ~2040 us; chunk1/2/8 at three rows ~2120/2176/2450; six-row
  variants ~2138/2253/3175/3593 us. Rejected. Both cache-hot and rotating
  repetitions retained, including warmup.
- S17: selective six-row RoPE and KV quantization batching. Keep the original
  attention kernels and chronological per-row KV ring/index publication. Move
  only independent query/KV transforms before that loop and inverse-head RoPE
  after it. New default-off `mtp_pointwise_batch` is restricted to contiguous
  six-row verification with existing paired output projections. This differs
  from the historical failed full batched-attention core. Full-model screen pending.

Five independently switchable candidates are being assembled into one resident
experimental build: S4 down wide loads, S9 compact Engram grid, S11 SIMD sort,
S12 F32 rows, S17 pointwise batching. Every new switch defaults off. No adoption
or second-round material improvement is claimed yet. One new resident process
will allow all combinations to reuse weights and disk-prefix fixtures.

Tooling note: first integration script expected a nonexistent environment helper
and stopped at its assertion after adding one header field; corrected to the
existing explicit default-off initialization form before compilation.

### First combined resident results (screen only)

One experimental resident replacement: PID95855, binary
`1dc29edfdf3203f7d8fac33aa4099a31e45e5481c2756896e83247e26482d1b7`.
All subsequent A/B arms reuse this process and its weights. Production-host
checks pass F32 rows1..8 with offset/guard checks (64 draws), compact F16
rows1..8 (18,432,000 finite exact words), and pointwise transforms at short,
wrap and deep positions in both directions/frequency families. First Metal
integration failed because the appended Q4 kernels followed macro undefinitions;
no model process was changed then. Moved the kernels into the macro scope and
retained the compiler failures. Corrected host checks pass.

Fixed-six, cached8k,512tokens: all outputs `c21119215ff8…`,117cycles.
Warmed pairs40.4749→42.8661 and40.4786→42.8678 t/s. Candidate repeat42.8450.
Verification approximately97.7→91.8ms/cycle. This is preliminary, not adoption.
The first control and another control repeat were34.9729/34.8415 because proposal
cost jumped to3004/3113ms (normally~1060), while verification stayed11478/11429ms.
Two ablations show the same preexisting startup effect. All are retained; do not
attribute that extra two seconds to these verifier changes or silently discard it.

Ablation verifier ms/cycle, full bundle~91.8: without F32 batching93.244,
without pointwise batching95.436, without wide down92.794, without SIMD sort92.053.
Without compact F16 was91.588, so that candidate has not demonstrated a model
benefit despite the native screen. A focused interleaved comparison follows.

Compact F16 follow-up: off42.8820/43.0172 t/s, on42.8233/42.9278;
mean verifier91.5813ms off versus91.7514ms on. **Not retained**: the native win
fails to transfer to the model. It remains a default-off experimental switch.
Final candidate excludes compact F16: F32 row batching, selective pointwise
batching, wide routed-down loads, SIMD-local route sort. Controller unchanged.
Final full numerical gates and held-out/depth replay pending.

GU SIMD sort gate passed50,001draws across no/random/full overlap, all gate/up/mid
words finite and bit-exact, poisoned output/tail checks, input-sign perturbation
controls on every draw, and1,000 independent deterministic submissions per
route-overlap class. Elapsed133.4s. Full log retained. No Tier2 arithmetic is
being proposed for adoption; Tier2 numeric alternatives above lost their screens.

Final numerical gates complete: GU50,001draws (133.4s), down50,001draws
(101.3s), F32 rows50,000draws/115,200,000finite exact words (36.9s),
pointwise50,000draws across heads1/64 at short/wrap/deep positions (46.9s).
Every gate checks exact outputs, poisoned guards, known input perturbations,
and1,000 separately submitted deterministic repetitions per shape/overlap class.
The pointwise harness initially omitted the SDK root and failed to find stdio.h;
corrected invocation and both logs retained. No numerical failure occurred.

### Controlled depth matrix

All six cases passed output-hash equality, using existing disk-prefix snapshots
and quiet-machine gates, with warmups followed by ABBA timing. Serial decode is
unchanged within measurement noise: 33.2864 → 33.2787 t/s at 8k and
31.6360 → 31.6325 at 300k. Fixed-six MTP improves 40.4038 → 42.7043 t/s at 8k
(+2.3006), and 44.2470 → 46.7334 at 62k (+2.4864). Acceptance counts match.
The 8k verifier drops 97.9157 → 92.0774 ms per block.

Keep the timing anomalies visible: the adaptive8 final control has a roughly
2-second delay outside its measured proposal/verify totals; fixed300 first
control spends 3,095 ms proposing versus about 812 ms in the other arms.
Consequently their raw ABBA throughput means overstate the optimization gain.
The fixed300 verifier itself falls from 108.40/108.73 to 102.73/102.77 ms per
block; the unaffected final control is 47.6092 t/s versus 50.1507/50.1277 for
candidate. These raw records are retained, not silently filtered. Materiality
is supported independently by the clean 8k and 62k comparisons above.

Normal-controller content and held-out agent replay are still pending at this
checkpoint. Deployment is not approved until those and forced rollback pass.

### Normal-controller replay checkpoint

Timed ABBA content means (t/s): code 44.550 → 46.115; SQL 55.525 → 58.635;
prose 33.335 → 33.340; chat 33.275 → 33.080. Outputs match their retained
reference exactly. The small chat regression is reported rather than removed.
The adaptive controller is unchanged; faster verification can alter its timed
attempt decisions (chat uses 2 control cycles versus 3 candidate cycles).

All 14 initial held-out pairs pass: 14,198 completion tokens per arm, identical
phase/output hashes, complete disk-prefix cache hits, no DSpark faults. The
predeclared first-arm repeat pass is in progress. Two 4,096-token cases are
reasoning-only and deliberately retained; they contribute no answer-time gain.

Harness review: the original pointwise gate's one-element negative control was
weaker than the GLM whole-row hygiene standard. A stronger gate now replaces
an entire activation row, reports the minimum moved-word count, checks that
other rows are unchanged, and runs 50,000 draws per head geometry. This is a
validation-harness correction, not a change to the candidate arithmetic. Both
the initial and stronger harness/results are retained; final adoption requires
the stronger run to pass.

### Final held-out result

All 14 predetermined first-arm repeats pass. Agent answer throughput is
46.96410 → 49.33632 t/s, **+2.37222 t/s**,
using the same estimated answer-token numerator as prior UAT. Answer duration
falls 70.80457 → 67.40011 s.
Reasoning duration is 328.43845 → 328.22269 s;
whole-decode throughput gains 0.915%, not +2 t/s. The materiality
claim is specifically MTP/answer decode. All raw runs and the predeclared
selection are retained. No output or phase-hash differences, cache misses, or
DSpark faults occurred.

Forced-reject rollback at two and six rows passes at cached 8k, 62k, and 300k
prefixes: all six candidate outputs match their serial reference. Fixtures
were released and their source paths restored unchanged.

### Final release checks

The stronger pointwise gate passes 50,000 draws at each of heads1 and heads64,
100% finite outputs, exact output words/guards, whole-row controls detected on
every draw, and 1,000 deterministic submissions per geometry. The minimum
changed control-word counts are retained in pointwise-strong-gate.log.
All production-host decode tests, controller/adaptive tests, and the Markov
50,000-draw cache test pass. All frontends build.

The final binary-hash guard stopped promotion after make relinked ds4-server:
benchmark SHA1dc29edf… versus rebuilt SHAcb18dc06…. Inspection found all30
Mach-O sections, including code and constants, identical; the differences are
outside those sections. Both binaries and the audit are retained. We restored
the exact benchmarked executable as the release artifact, verified its full
SHA256, and did not rerun successful numerical tests or substitute a new build.
This is a resolved release-artifact mismatch, not a numerical failure.

Private promotion and final restoration completed at 2026-09-15T20:33:24.331247-04:00. The implementation
commit is 2b9b59136105677a2239fa7c2b30affe531d007a. Full user-facing UAT canary passes;
all workers/services/cache/background processes restored, debug endpoints closed.
No new performance claim is inferred from the short health-check prompt.
