> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Agent-answer decode: 49.336 → at least 50.0 t/s

Baseline UAT `59c1576`, implementation `2b9b591`. Preserve the controller and all four previously adopted opt-ins. The established held-out 14-request phase calculation is unchanged; no new asterisk on the comparison table. User-observed 54.3 t/s is a separate workload, not an adoption result for this campaign.

Artifacts: `LOCAL_ARTIFACT/`.

## Setup and rejected directions

- Wider gate/up weight loads: skipped after reading prior campaign G2. That experiment was exact but neutral/slower. No new performance claim or repeat experiment.
- Isolation tooling failure: initial script searched incorrect snapshot paths and stopped after reversible service isolation/cache backup. Recovered by cloning the existing `user-speed-20260915/kv` snapshots; did not repeat isolation, prefill, or reload weights. Journal: `RESTORE.md`, `isolate.log`.

## A1: precompute stable expert-job ordering

Hypothesis: the same stable 36-job sort is repeated inside every gate/up output tile. Screen an upper bound by precomputing the exact sort in the synthetic harness and having the candidate read its job mapping. The floating-point body is unchanged. This first screen omits the cost of a GPU planning dispatch and is not a production proposal or adoption gate. A promising screen requires a GPU planner-inclusive follow-up, strengthened fidelity gate, and resident cached full-model A/B before adoption.

Status: rejected at native screening; result below.

A1 result: rejected. 64 complete-output exact/finite draws per routing case, but even the planner-free upper bound was slower: no overlap ~686→693 µs; random overlap ~502→516 µs; full overlap ~437→441 µs. No production code adopted. `sort-plan.log` retains all samples.

## A2: Q8 batch BF16 store folding

Reuse the existing pair/trio reduction helpers with `ROUND=true`; retain dot-product order, job grid, function constants and rounding point. The old path materializes F32 then performs the same BF16 integer rounding in a separate dispatch. New opt-in `mtp_q8_round` defaults off. Native screen covers the six projection families, independent 2–6-row fallback plus production pair/trio six-row dispatch, poisoned bounds, finite exact word comparisons, input-sign negative controls and determinism.

Initial 64-draw screen exact. Most families save ~2 µs per dispatch, while query-B was ~1.5 µs slower. Query-A had a clock transition in the first screen; retain its raw samples and do not claim that apparent gain. The subsequent full50,004-draw gate passed; full-model results are recorded below.

A2 first gate passed 50,004 draws across six shapes, with exact finite outputs, poisoned guards, sign-flip controls and 1,000 repeatability runs per shape. The strengthened50,000-draw gate for each new pair and trio kernel also passed before resident testing. Existing scalar BF16 kernel also covered in the six-shape gate. Stream-six selection is covered separately through the linked host test.

## A3: shared SwiGLU BF16 store folding

Copy the production activation expression unchanged and apply the existing BF16 integer rounding to the final store; remove only the separate rounding dispatch. Opt-in `mtp_swiglu_round`, decode batches 2–8 rows only. Native 50,000 draws covering all supported row counts, all finite/word-exact outputs, poisoned tails, 50,000 sign-flip controls and 1,000 repeatability runs passed. Native six-row time ~5.3→2.8 µs. The model gain is evaluated in the combined A2+A3 screen below.

The earlier `verify_wide_prefill` experiment was located in the global optimization ledger (S5c: +0.12%, 9.105→9.116 t/s) and is not being repeated or enabled.

A2+A3 resident fixed-six 8k screen: warm A42.5253/B42.7122; timed ABBA A42.8074/B43.0149/B43.0137/A42.7344 t/s. Mean **42.7709→43.0143** (+0.2434). All 512-token hashes identical, 117 cycles, cached snapshot only. Small retained candidate, not goal completion. Original four adopted opt-ins remain enabled; serial/controller unchanged.

## A4: combine independent Q8 row reductions

Distinct from prior S3's scalar single-barrier reduction. The pair/trio kernels currently invoke the reduction helper separately for each token row. New helpers perform each row's same SIMD sums, initialize all disjoint scratch regions, synchronize once, write the same partials, synchronize once, then finish the same per-row reductions and BF16 stores. No across-token sum, arithmetic-order or dispatch-geometry change.

Native screen compares against A2 folded stores. Two-barrier version is faster than the one-barrier variation for query-B (~155→136 µs vs ~155→141); shared-down ~33→30.6 µs; other families ~0–2 µs improvement. Every 64-draw shape screen is finite and bit-identical, with input controls/poisoned bounds. Query-A first one-barrier run had a clock transition and is not counted as its benefit. The full pair/trio50k-per-kernel gate subsequently passed, including finite exact outputs, input controls and1,000 deterministic repeats.

Resident testing replaced only the two BF16 pipeline-cache entries, retaining original PSOs for rollback and logging exact PID/pointers/source hashes. Baseline uses the untouched non-folded pipelines. This avoids another model load.

A4 resident fixed-six 8k ABBA means **42.79765→43.10475 t/s** (+0.3071 for A2+A3+A4 together). All 512-token hashes and 117-cycle counts identical. The native query-B advantage mostly did not carry through; this is not a material result by itself.

Replay harness assertion failure: an added all-content-cache assertion stopped at the short public code warmup (205 prompt tokens, 1.603 s normal API prefill, cached=0). The previous campaign's identical warmup also had cached=0; the preserved content protocol does not claim disk-prefix reuse for these tiny API prompts. No manual prefill endpoint was called. Retained the record and failed log; continuing with the 14 cached long agent prefixes first, for which full cache restoration remains mandatory. No weight reload.

A5 hypothesis: integer GU sorting using a 32-key network plus insertion ranks for the remaining four keys; intended to reduce duplicated sort work without the extra mapping load that lost in A1. A6 down-row interleaving on the newly adopted wide-load kernel. Prior D2 tested the older narrow-load kernel and never reached real routed model fixtures; a new test would evaluate the interaction with the wide loads and use the actual agent workload to decide, rather than assuming the synthetic overlap win applies. These are hypotheses only; no performance or fidelity claims yet.

A6 screening narrowed to tiles64/256 before execution: the previous narrow-load D2 ledger has tile256 as the strongest full-overlap case (+3.89%, with -0.91% no-overlap) and tile64 as the strongest partial-overlap case (+2.93%, with -2.37% no-overlap). Tiles4/16 were generated but will not be rerun without another reason. No GPU test overlaps the current agent replay.

A2+A3+A4 complete 14-request prepared-first-arm replay: **49.34265932→49.56435724 t/s** answer phase (+0.22169791, +0.4493%). Common answer-token estimate3325.27329279; answer seconds67.39144865→67.09001142. All14198 completion tokens per arm, exact text/phase hashes; every long prompt fully restored from cache. Whole decode +0.0462%; reasoning -0.0361% (timing variation on an unchanged serial path). Goal **not met**. Full initial and prepared records archived under `a2-a4-agent/`; nothing discarded or substituted. The prescribed prepared A02 control has119 cycles versus123 in the initial runs, reflecting controller timing/state variation; policy is unchanged. Retain that discrepancy rather than claiming every repeat took the same path.

A5 rejected at native screening: 64 exact full-output draws for each of three routing cases. Shorter 32+4 sort is neutral/noisy without overlap, slower on random routes (~503.5→507.8 µs), and slightly faster only for all six rows sharing experts (~433.4→429.9 µs). This does not justify installing it in the model. `sort36.log` retains all samples.

A6 native screening on the wide-load down kernel is exact across all outputs. Tile64 improves partial overlap (~348→340 µs) but regresses adjacent-row full overlap (~327→329). Tile256 improves partial (~349→337) and adjacent-row full (~328→314) overlap. It is the selected full-model screen; this is not an adoption decision. Original per-token expert addition order remains unchanged.

A6 resident screen (additive to A2+A3+A4): incumbent **43.03085**, tile256 **43.37290 t/s** (+0.34205), fixed-six 8k512 ABBA; all outputs/cycles exact. The original down PSO was restored afterward. The optional CPU fast-pipeline lookup index is invalidated on each down PSO change so it cannot retain a stale pipeline; original/candidate objects are retained explicitly. Candidate PSOs are reused on subsequent switches, rather than recompiled. The full50k fidelity gate passed (below); complete combined workload validation is running.

A6 full gate passed: 50,001 draws split evenly across no/partial/full overlap, complete finite bit-identical outputs, poisoned guards, input controls every draw, and 1,000 deterministic repeats per case. Source integration changes only the threadgroup-to-row mapping for six rows at width5120/NSG2. The next complete agent replay switches the down PSO back to the original for every A arm and installs the retained candidate for every B arm; round-store flags also follow the arm.

A7 CPU argmax screening: the initial compile failed because the explicit command-line SDK sysroot was omitted; no benchmark ran from that attempt. Corrected locally without changing Xcode settings. A NEON prototype retains earliest-index ties and falls back to the original scalar scan for nonfinite inputs. 1,000 initial draws/controls and determinism passed; native no-exclusion scan ~64→18.7 µs, exclusion scan ~50→20 µs. This is a CPU microbenchmark only, not adopted or installed in the server.

A8 preparation only: identified that the six-row Q8 query-B stream path still has separate per-row reduction barriers. Generated a native comparison for combining those independent barriers, analogous to A4. No screen, full gate, model installation, or performance result yet. This is a possible follow-up if A2+A3+A4+A6 misses the goal; generation does not change the release candidate.

A2+A3+A4+A6 complete held-out result: **49.27335019→50.04017320 t/s** answer phase, +0.76682301 t/s (+1.5563%). The unrounded candidate exceeds50.0; rounding was not used to cross the threshold. Common estimated numerator3325.27329279; answer seconds67.48624317→66.45207400. All14,198 completion tokens per arm and all phase/text hashes exact; every long prompt fully cached. Reasoning throughput -0.0326%, whole decode +0.2349%. The controller policy is unchanged; its time-sensitive attempt counts can vary (A02 initial119/125 cycles, prepared A119; A18 control24→25 on its predetermined repeat). All initial and prepared records retained. Full depth/content/release gates remain before promotion.

A7 remains an unadopted CPU microbenchmark and A8 remains an unrun generated hypothesis: the existing candidate reached the explicit follow-up target, so no further tuning or weight reload is justified for this release. Both are recorded for future work, without a full-model performance claim.

Complete cached matrix (ABBA means, t/s): serial8 33.24235→33.28310; serial300 31.66910→31.65155; fixed8 42.82230→43.26885; adaptive8 37.25525→37.44405; fixed62 46.68095→47.40360; fixed300 50.13980→50.65590. All text hashes exact; fixed-six acceptance counts match. The first fixed8 control warmup was36.5363 t/s and is retained as a warmup anomaly, not used to inflate the timed gain. All fixtures restored existing snapshots; no prefill occurred.

Normal-controller API ABBA results (t/s): middle-code512 46.095→46.410; middle-chat512 33.105→33.140; prose512 33.350→33.345; sql512 58.630→59.560. All output hashes and completion counts match the prior reference, with no DSpark fault latches/failures. The tiny negative prose result is retained. Same short-prompt warmup protocol as prior/public; no claim that its first tiny prompt was disk-restored.

Release validation passed: forced rejection at2/6 rows and8k/62k/300k matches serial; full build and all decode bandwidth/controller/adaptive/Markov tests passed. The rebuilt executable's Mach-O sections match the measured executable; the exact measured artifact was deployed. Original PSOs were restored before final cutover. UAT's full bridge→worker→DS4 READY canary passed after restoration. Private promotion and final operational receipts are committed together with this ledger.
