> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Decode experiment ledger — 2026-09-15

Campaign baseline: private UAT `0d24e54`. This is a chronological ledger;
intermediate candidate/pending dispositions describe their state at that point.
Final selected changes are deployed in private `v41-mtp-prod` at `38e8c9d`;
see the final results below and `DECODE-MATERIAL-20260915.md`.
Artifacts: `LOCAL_ARTIFACT/decode-material-20260915`.
Microbenchmarks do not establish a full-model speedup. Failed and superseded
records are retained; never silently replace them with a favorable repeat.

| ID | Hypothesis / change | Evidence so far | Disposition |
|---|---|---|---|
| R1 | One SIMD group selects six unique router maxima; retain the original sort for ties/nonfinite inputs. Initial dispatch still 512 threads. | 50,000 poisoned draws: 0 mismatches, 19,498,547/19,500,000 finite FP words, 0 poison survivors; 1,000 deterministic groups; 50,000/50,000 planted controls detected. About 15.5 -> 11.9 us in isolated sequential dispatches. `router-512.log`. | Superseded by R2, retained as evidence. Too small alone. |
| R2 | R1 with a 32-thread dispatch and the original 512-position fallback sort virtualized onto those lanes. | Same fidelity counts, exact. About 15.4 -> 11.6 us. `router-test.log`. | Candidate, default off. Extended to independent small-batch rows for upcoming in-model checks; that integration has not yet been tested. |
| C1 | Spread the collapse/RMS work over groups while keeping all 1,024 virtual lanes, each accumulation, and both reduction trees. Two dispatches. | 50,000 draws, 512,000,000 compared finite words, 0 mismatches/poison survivors, 50,000 planted controls detected; 1,000 separate deterministic runs. Roughly 7.3 -> 6–7 us, with run variation. `collapse-test.log`. | Candidate, default off. Small potential saving; no full-model gain established. |
| Q1 | Three independent token rows share Q8 loads, between deployed pair reuse and high-register-pressure six-row reuse. Preserve each row's arithmetic. | 50,004 randomized input draws across all six production dense shapes: 2,355,388,416 finite output words exact, no poison survivors/tail changes; 50,004 planted mismatches detected; 1,000 independent deterministic submissions per shape. Production host path also matches 1,271,808 words across 1/2/4/6/8 rows. `trio-gate.log`, `trio-host.log`. | Kernel fidelity passed. Full-model gates remain. |
| Q1-q_a | Three-row reuse at K5120/N1280. | Rotating eight weight copies: 20.281 -> 20.240 us, +0.2%; hot copy +11.1%. | Neutral under rotating weights. Do not infer a general gain from the hot-copy result. |
| Q1-q_b | Three-row reuse at K1280/N32768 against deployed six-row stream. | Rotating copies: 233.377 -> 125.092 us; hot copy 220.341 -> 122.789 us. | Largest preliminary lead; needs full-model confirmation before any t/s claim. |
| Q1-KV | K5120/N512. | Rotating copies: 10.711 -> 10.651 us, +0.6%. | Effectively neutral; exclude unless an in-model interaction supports it. |
| Q1-out_b | K8192/N5120. | Rotating copies: 103.868 -> 94.816 us, +9.5%; hot copy +17.4%. | Candidate. |
| Q1-shared gate/up | K5120/N2304. | Rotating copies: 31.810 -> 31.896 us, -0.27%; hot copy +8.6%. | No demonstrated win under rotating weights. Retain the slight regression. |
| Q1-shared down | K2304/N5120. | Rotating copies: 36.176 -> 32.307 us, +12.0%; hot copy +9.2%. | Candidate. |

## Markov follow-up

M1: oMLX keeps the existing previous-token correction on the GPU. Our earlier
sample estimates ~2 ms/proposal, so the ceiling appears modest. Before the
single planned replacement, add direct optional timing and an exact CPU bias
cache as a bounded alternative: the correction depends only on predecessor and
immutable drafter weights. A 64-entry per-drafter FIFO stores the literal old
matvec output (~32 MiB at production vocabulary), reusing it when that token
recurs. No proposal selection, confidence calculation, controller threshold,
weights or quantization changes. Default off, allocation failure falls back.
50,000 synthetic chains at production rank 256 and reduced vocabulary 257
match all 64,250,000 post-correction logit words and proposal IDs, with alternating
independent models, eviction, hits and bypass. 50,000 planted differences
detected; 1,000 repeated chains deterministic. `markov-gate.log`. Production
vocabulary and full-model A/B checks remain; this is not adopted.

## Invalid runs and build failures

- The first router harness did not compile because its scalar/vector Metal
  position attributes disagreed. No performance result exists for that run.
- While changing router dispatch width, an Objective-C build failed because
  `MTLComputePipelineState.label` is read-only. A shell without fail-fast then
  ran the old harness at the wrong geometry. `router-stale-harness-invalid.log`
  is **invalid**, not a candidate regression. The harness was corrected,
  rebuilt, and rerun at the intended 32-thread geometry. Subsequent build/run
  chains use fail-fast or checked subprocesses.
- The Q8 harness initially used a symbol named `pipe`, colliding with POSIX;
  compilation failed and no benchmark ran. It was renamed.
- The selected Xcode command wrapper began refusing compilation pending its
  license agreement. No agreement was accepted and no system developer path
  was changed. Explicit Command Line Tools compiler and SDK paths are used
  per invocation. The first such invocation lacked an SDK root and failed;
  the corrected invocation specifies `-isysroot` and compiles successfully.

- The new host test initially used an implicit Make rule without the core
  objects; linking failed. An explicit target now uses the production objects.
- Router host integration test first requested unsupported group gating. That
  was a harness argument error; the production model does not request it. The
  failed log is retained as `host-paths.log`, and the corrected run is separate. A second harness invocation also requested
  unsupported four/eight-row grouped attention; the corrected test covers the
  actual two/six-row API. `host-paths3.log` passes all host integration paths.

## Historical negatives used to prune this search

The existing `OPTIMIZATIONS.md`/`DECODE-REMAINING.md` records remain authoritative
for their old experiments: routed-down split failed its quality gate; NR0=1
routed down and widened variants lost; attention shared staging lost; six-row
forced GEMM and eight-row MMA lost; FP16 routed accumulation lost; dense FP16
accumulation was neutral. None is being relabeled as a new improvement here.

## Resident screening outcomes

All comparisons below restore the 8k prefix, generate 512 committed tokens,
keep all continuations exact (`c21119215ff8…`), and require three quiet GPU
samples plus >=90% CPU idle before each arm. Initial screen, not adoption.

- Warm fixed-six control: 38.2510 and closing 38.2168 t/s.
- Q1-q_b: 37.8687 / 38.2932; Q1-mask82: 38.3138 / 38.2559.
  **No material full-model improvement** despite the standalone q_b lead.
- Q1-mask127: 38.5325 / 38.4263; small candidate, insufficient alone.
- R2: 38.6485 / 38.7614 fixed-six; scalar **32.6397 / 32.7416** versus
  scalar control 32.9756 / 32.8696 / closing32.8760. A batch-only candidate;
  reject enabling it for serial based on this screen.
- C1: fixed-six38.2242 /38.1869, scalar32.8252 /32.8671. No win established.
- M1:38.3794 /38.4007. Direct 59-proposal diagnostic per arm:
  readback62.3 /62.6 us, uncached chain1639.7 us, cached996.8 us,
  53.90% hit rate. Profile run throughput excluded from adoption.
- Combined mask82 + R2 + C1 + M1:38.8278 /38.7702. About+0.6 t/s,
  **below the user's materiality requirement**. Not promoted.
- First control32.3607 had3512.8ms proposal vs1128.6ms on its repeat, with
  verify12145.5 vs12102.8ms. Retained as a startup/warmup outlier; do not turn
  its extra proposal time into a claimed candidate gain.

### Resident shader replacement diagnostic

To preserve loaded weights, a debugger attachment between requests can compile
an individual experimental Metal pipeline and replace only its dictionary
entry. PID81068 remains the same. `pipeline-transaction.json` retains the old
pipeline and exact key/pointer for rollback; every attachment is logged and
ends with detach. No model buffers, controller state, or OS setting is changed.
The first attempts failed Objective-C expression type checking before any
pipeline replacement; all logs are retained (`inject-trio*.log`).

A standalone-compiled copy of Q1 was installed to check compile-context effects.
It also failed to reproduce the isolated q_b advantage: control38.1292/38.2183,
q_b38.0990/38.3478; mask12738.4667/38.5967. This confirms a reversible resident
iteration path, not a new speedup. The original pipeline must be restored before
final deployment unless a separately gated replacement is adopted.

G1: direct adjacent-row Q4 gate/up reuse, with duplicate expert lookup inside
the existing kernel and no planner or shared-weight staging pass. **Rejected**:
no-overlap 693→956 us; half-overlap first mismatch was one ULP in gate output
word18433. Not installed in the model. `gu-pair-test.log` retains the failure.

### Corrected Q8 compilation and subsequent negative experiments

The first Q8 microbenchmark used literal NSG/tile constants; production uses
Metal function constants600/610. This materially slowed the reference stream6
kernel in the standalone compilation. The original claimed q_b micro speedup
is **nonrepresentative**, not an optimization result. Matching production
specialization (`trio-specialized.*`) gives q_b118.90→125.28us (regression),
q_a19.99→19.92, KV10.70→10.59, outB102.55→95.26,
sharedGU31.70→31.33, sharedDN35.25→31.50. Fidelity results for the initial
build do not substitute for a final gate using production specialization.

- Q8 NR4/NR8 widening: preliminary output equality passed, all shapes slower.
  q_b119.49→162.21us (NR4),118.61→180.18 (NR8); outB and DN also regress.
  Logs `trio-nr4*`, `trio-nr8*`. Not installed.
- G2 wider Q4 gate/up ushort4 loads: preliminary full-output equality passed;
  half-overlap589.8→591.6us, full516→517.7. No win. `gu-wide*`.
- GU NR4/8/16: preliminary equality passed; no-overlap688.9→720.8,
  688.3→784.4,689.5→910.6us, respectively. Overlap cases also slower.
  `gu-nr{4,8,16}*`. Not installed.
- GU fixed K20, full known-shape specialization, with/without loop unrolling:
  all preliminary equality checks passed; all neutral within noise or slower.
  `gu-k20*`, `gu-shape*`. No adoption.
- G3 expert-sorted job order: 64-key SIMD bitonic sorting permutes the 36
  independent row/expert jobs, preserving the original floating-point body and
  output locations. Preliminary all-output equality passed at full K5120/N2304
  for 0/3/6 adjacent-row expert overlap. Native medians approximately
  688→687us,593→554us,513→458us. **Screening only**, not an adopted win;
  real-model A/B and full fidelity gate pending. `gu-sort*`.

Pipeline rollback authority is now `pipeline-registry.json` (same PID81068),
with each retained original PSO and every replacement/restore logged. The earlier
`pipeline-transaction.json` records the first trio transaction only. The original
trio entry has been restored. No additional model reload occurred.

G3 resident fixed-six code512 screen: control38.1577/38.1375 versus
expert-sorted38.4749/38.4205 t/s, all exact (`c21119215ff8…`). About+0.30,
not material. Original GU PSO restored after the screen. Existing UAT already
interleaves output tiles across rows; G3 adds sorting by selected expert IDs.
It does not introduce cross-row reuse for the first time. Tiles1/2/4/8/32/64
were screened natively; tile8 is similar to16, others neutral/slower.

The pipeline replacement script initially misclassified a successful attach as
failed because its error detector matched Objective-C selector `error:` echoed
inside a command. The log proved install and detach completed; receipt recovered
from exact pointer values, then original restored. Detection now matches actual
error lines. Retained as a harness error, not a model or kernel failure.

D2: independent output-tile permutation in fused Q4 down/sum6, keeping expert
accumulation order fixed. Native full-output equality passed for tiles1/4/16/64/256
under 0/3/6 adjacent-row overlap. No-overlap regresses about1–3%; partial/full
overlap sometimes gains up to3.7%. No clear general win; not installed in the
model. `dn-tile*-test.*` retains all results.

R3: hierarchical SIMD selection in the existing512-thread serial-router launch.
Each SIMD group nominates6 candidates; one SIMD group chooses the global6.
Ties/nonfinite inputs fall back to the literal original bitonic body.
`router-hier-test.log`:50,000 poisoned draws,19,498,547/19,500,000 finite words,
zero mismatches/poison survivors,1000 determinism groups,50,000 planted controls.
Native sequential-dispatch latency reference16.55us versus5.36us.
Resident serial code8192/512: candidate33.4419/33.3732 versus restored-original
32.9446/33.0110 t/s; all complete text hashes match. About+0.43t/s, screening
only. Original router PSO restored. This preserves controller behavior and
uses no model reload. New shader and generator: `router-hier-*`,
`make-router-hier.py`. Candidate not yet incorporated into production source.

C2 register-resident collapse: replaced the shared activation staging row with
per-thread registers and removed redundant partial-sum initialization/barrier
for the known1024-thread shape. Initial version changed a handful of BF16
normalized outputs (collapsed outputs exact). Forcing the accumulator volatile
did not fix it. Explicitly rounding `cached * rscale` before weight multiply
restored equality in1000draws/10,240,000words plus1000separate deterministic
submissions and1000planted controls. Reversed multiply order still failed.
Corrected micro6.73→6.52us is too small to prioritize; no live install and no
50k adoption gate claimed. `collapse-reg-*` retains every variant/failure.

G4: map paired rows onto separate SIMD groups with the same output channels,
keeping the original accumulator/register count per thread. Preliminary full
outputs exact. Adjacent-overlap0/3/6 native times~688→688,590→559,513→483us.
Adding G3 job sorting yields~687→687,594→560,515→479us, not additive to G3's
best isolated result. `gu-simd-pair*`.
G5: sequentially evaluate paired rows in separate local scopes to reduce the
activation-register lifetime of rejectedG1. Equality passed but performance
regressed in every overlap case:~691→851,591→672,515→576us. Rejected; not
installed. `gu-sequential-pair*`.
G6: pair matching experts across all6rows using SIMD rank/neighbor lookup,
rather than adjacent row pairs only. Preliminary equality passed; with sorting
~688→688,588→559,514→479us on adjacent-overlap harness. Broader cross-row
route patterns and model screening pending. `gu-global-pair*`.

G6 broader-route result: on randomized unique-within-row routes across36experts,
G3 sorting alone524→490us; global SIMD pairing+sorting523→526us. When all6rows
share the same6experts, G3 is neutral437→437us and G6 regresses438→445us.
Both pass96full-shape draws including these cross-row patterns. G6 is rejected
as a performance candidate; no live model installation. `*-random-test.log`.

## Fresh resident attribution

`resident-profile-{serial,mtp}.ledger` uses128cached-prefix decode tokens, same
PID81068, all candidate levers off. Both certify100% coverage, no counter-pool
exhaustion or resolution failures. Instrumentation changes batching/concurrency;
its26.73/28.40tps are **not adoption timings**. Serial/routedGU5.08ms/token,
routedDN+expansion3.60, outB2.87, query/HC2.87, sharedGU/HC1.93.
For28MTPcycles: routedGU21.87ms/cycle, routedDN15.11, F16 large6144-input
projections5.14, Q8query4.99, attentionoutA4.59, outB4.31.

Ledger was enabled without a model reload by initializing its existing sampler
and binding existing pipeline names, then installing its4encoder hooks. No
device pipeline-construction hooks or environment changes. `ledger-disable.log`
verifies all4original Objective-C method implementations restored exactly and
mode0/by_tg0. Three inactive counter pages remain allocated (small diagnostic
state, removable at final cleanup); `ledger-transaction.json` records this.
Initial read/enable attempts hit debugger symbol/type issues; all logs retained.
The partially initialized enable attempt was completed before any request.

F1 under test: the two F16 Engram projections use the ordinary independent-row
matvec, K6144/N25600/sixrows. Share half4 weight loads/conversions across2/3/6
rows while retaining each row's original per-lane dot sequence and reduction.
Fallback is literal existing kernel for every other shape. `f16-rows*`.

F1 screen: fullK6144/N25600, sixrows, literal reference NSG8/NR0=2,
8draws/1,228,800 finite exact outputs for pair2 and trio3. Rotating8weight
copies gives reference2563us → pair1298us or trio897us. Six-row sharing
fails immediately by1F32ULP (`f16-rows6-test.log`), so it is excluded.
Resident trio3 fixed-six8k code512:39.3347/39.3942tps; warm restored control
38.3392. A preceding control32.8628 had3308msproposal versus1127ms on repeat,
while verify12114vs12076ms; retained as another first-request proposal outlier.
Do not count it as candidate gain. Trio saved~3.25ms/cycle in verify, matching
the isolated2projection expectation~3.33ms/cycle. For final comparisons, run
an explicit full-request warmup after every pipeline replacement before timed
repeats, in both arms; retain those warmups separately.

F1 full gate passed:50,000 fullK6144/N25600/six-row draws, all
7,680,000,000 output words finite and bit-exact, poisoned outputs/tails checked,
50,000 input-sign-inversion planted controls detected,1000separate submitted
determinism runs. `f16-rows3-gate.log`, `f16-rows-gate.m`.
Q1 final selected-mask80 gate passed with real function constants600/610:
25,000draws each at8192→5120 and2304→5120, total1,536,000,000finite exact
words,50,000planted controls,1000independent determinism runs per shape.
`trio-final-gate.log`. No q_b speed claim or default adoption.

Combined F1+G3+Q1mask80+R2+M1 screen: warm40.4594, repeat40.4547tps versus
warm control38.3392; identical512-token output and117cycles. Another retained
repeat34.8218 had3105msproposal versus1057ms on the other two, while verify
11443/11447ms was stable. A warmup alone does not eliminate this proposal-time
variance; final protocol must report it honestly, with interleaved repeats and
separate verify/proposal attribution. No materiality/deployment claim yet.

Integrated F1, G3 and R3 behind new default-off levers. R2 is now batch-only
(n_tokens2..8); serial routing uses the original or separately gatedR3 path.
No controller thresholds, phase policy, proposal generation or acceptance rules
changed. `make all` passed with explicitCommandLineTools environment; the first
invocation omitted that environment and hit the unaccepted-Xcode shim before
building anything (`build-integrated-license-shim.log`). No license/OS change.

G3 full gate passed: 50,001 draws across three routing distributions,
12,441,848,832 finite exact gate/up/mid output words; poisoned outputs and tails,
50,001 planted controls and 1,000 independent deterministic submissions per
routing distribution. `gu-sort-gate.log` retains the complete report.

Integrated production host tests passed: F16 trio (18,432,000 exact words,
rows 1..8, tensor offsets and poisoned guards), material paths (1,425,600 router
words plus Q8/collapse), Q8 trio (1,271,808 words), and GU expert ordering
including expert ID383. Census-only reruns prove candidate dispatch engagement:
20 F16 trio, 700 batch router, 100 hierarchical router and one sorted-GU dispatch.
These are coverage checks, not performance measurements. The rebuilt Markov
cache gate also passes 50,000 draws/64,250,000 words, 1,000 deterministic
submissions and 50,000 planted controls.

Rebuilding the experimental objects invalidated the old resident executable's
LLDB debug map. The first frozen confirmation aborted before measurements;
no model or weights were reloaded. Recovered the pipeline-cache and device
object addresses from live `ds4_gpu_get_pipeline` machine-code references and
verified three PSOs against retained originals/candidates and the device name.
`pipeline-address-{recovery,verify}.log` and PID-bound `pipeline-bindings.json`
record proof. The first repair needed an explicit NSMutableDictionary type;
that failed setter did not mutate the cache. The typed repair then restored
F16 successfully. New integrated executable debug symbols are preserved in
`integrated-ds4-server.dSYM` before any further rebuild. Pointer bindings are
strictly PID-specific and must never survive a server restart. Both tooling
failures and subsequent frozen measurement runs are retained separately.

Frozen fixed-six code8k confirmation (ABBABAAB), all eight timed runs retained:
control 38.2872, 38.2060, 38.2294, 38.2101 t/s; candidate 40.5103, 40.4589,
40.4301, 40.3442. Means 38.233175 → 40.435875 (+2.2027 t/s, +5.76%).
Every output hash and all117 acceptance cycles match. Proposal times control
1124–1132ms vs candidate1058–1064ms; verify12095–12119ms vs11429–11475ms.
An intervening pre-request CPU check saw83.63% idle and aborted before the
next request. Resume retained the first three timed runs and completed the
original fixed order; no timing was discarded. Quiet checking now waits for
CPU>=90% idle as well as three GPU<=5% samples (bounded30attempts), without
relaxing either threshold. `material-confirm*.log`, raw JSON and summary retain
all evidence. Broader adaptive/long-context integration validation remains.

Integrated release-candidate build and production host/controller tests passed
with selected defaults enabled: Q8mask80, F16trio, GUexpert-sort, batchSIMDrouter,
serialhierarchicalrouter and Markovcache. Profile/collapse remain off. All six
have independent disable switches. Full source/debug symbols and binary hashes
are recorded before installation. PID84032 replaced the old resident experiment
once for final integration validation; it is the second load in this campaign,
not another per-experiment model load. The old process's retainedPSOs and small
counter pages expired. Old address bindings were archived and cannot be reused.
All remaining A/Bs use the integrated binary's levers. Fourteen complete agent
prefixes were APFS-cloned from the prior campaign; no manual prefill was run.

Integrated cached-prefix validation so far: serial8k32.74915→33.19025,
serial300k31.29605→31.71250, fixed-six8k38.02605→40.34575,
adaptive code8k35.57265→36.60555, adaptive agent8k35.86310→36.94705.
All four timed ABBA outputs match per fixture. At32k, the adaptive agent fixture
has matching outputs but the existing timing-dependent controller makes3cycles
in each control and27 in each candidate:31.93705→34.66120t/s. Candidate kernels
cross the existing cost threshold; no controller-policy change. A preceding
control warmup made23cycles, so retain the timing-dependent admission caveat.
Do not compare absolute throughput across distinct fixture prompts: acceptance
rates differ. The62k code fixture, for example, has105cycles for512tokens versus
117cycles on8k code. Held-out normal-server replays remain pending.

Integrated matrix complete: fixed-six62k41.79625→44.25500 (+2.45875t/s),
all105cycles and complete outputs identical. All seven cached-prefix cases pass;
no request prefills. All fixture slots released and their temporarily hidden
source files restored with matching hashes.
Normal API code benchmark (same132tokens,17cycles,6declines in all arms):
server-final averages42.37/42.34 versus44.37/44.44t/s, mean42.355→44.405
(+2.05t/s). This uses the ordinary adaptive controller and the same public-table
prompt, not client first-to-last token arithmetic. Both outputs match the prior
UAT hash. Other content/agent replays still running; no promotion claim yet.

Final normal-controller validation passed: four content ABBA comparisons and
all14held-out agent pairs plus every prepared-first-arm repeat. All outputs and
phase hashes match prior UAT;14,198tokens per selected arm, all full-prefix cache
hits, zero speculative faults. Every held-out request improves full decode time.
Pooled reasoning334.303616→329.037322s;
answer75.065010→70.477681s;
full decode409.368626→399.515003s.
Answer estimate44.298579→47.181934t/s (+2.883355); normal API SQL52.120→55.425
(+3.305). Materiality and exact-output gates pass. No new controller policy or
Tier2 arithmetic is adopted. See release-summary.json and agent-summary.json.

Promoted and pushed private implementation `38e8c9d1287090e766c0d2ecd2a14c49fca91aae`.
Validated PID84032 retained. Original plist/cache restored, diagnostic routes
closed, four services restored,25processes resumed. VM PID51394 had exited;
no reused PID signalled or replacement VM started. Ordinary API smoke READY
passed. Final receipt records hashes, private remote, table and restoration.


Post-promotion recovery: the initial restoration missed the dead Colima VM,
leaving UAT's worker unavailable. The existing VM and containers were recovered
and the full agent path passed at 17:56 EDT. See
recovery record (`DECODE-MATERIAL-UAT-RECOVERY-20260915.md`, local historical artifact).
