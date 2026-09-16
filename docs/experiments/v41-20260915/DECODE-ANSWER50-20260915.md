> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# V4.1 agent-answer decode follow-up

The established 14-request agent-answer replay improves **49.273 → 50.040 t/s**. This meets the explicit follow-up goal of at least 50.0 t/s from prior UAT 49.3. Baseline private UAT: `59c15768b3ff5e7a6c3c63804e24894ff186f53d`. These are incremental gains over that UAT, not gains counted again from an earlier campaign.

## Public branch → new UAT

| Decode workload | Public branch | New UAT |
|---|---:|---:|
| Serial, 8k context | 31.3 t/s | 33.3 t/s |
| Serial, 300k context | 28.3 t/s | 31.7 t/s |
| DSpark on code | 40.5 t/s | 46.4 t/s |
| DSpark on agent turns, answer phase | 41.3 t/s | 50.0 t/s |

The agent-answer calculation is unchanged: the same common estimated answer-token numerator divided by measured streaming phase durations. The label applies consistently to both columns; no new asterisk is introduced. Prefill, TTFT and disk-cache restore latency were not remeasured in this decode-only campaign. The user-observed 54.3 t/s run is a separate workload, not substituted for this comparison.

## Same-build paired results

| Workload | Control | Candidate | Change |
|---|---:|---:|---:|
| serial8 | 33.242 | 33.283 | +0.041 t/s |
| serial300 | 31.669 | 31.652 | -0.018 t/s |
| fixed8 | 42.822 | 43.269 | +0.447 t/s |
| adaptive8 | 37.255 | 37.444 | +0.189 t/s |
| fixed62 | 46.681 | 47.404 | +0.723 t/s |
| fixed300 | 50.140 | 50.656 | +0.516 t/s |
| middle-code512 | 46.095 | 46.410 | +0.315 t/s |
| middle-chat512 | 33.105 | 33.140 | +0.035 t/s |
| prose512 | 33.350 | 33.345 | -0.005 t/s |
| sql512 | 58.630 | 59.560 | +0.930 t/s |
| Agent answer phase | 49.273 | 50.040 | +0.767 t/s |

Held-out reasoning time: 328.259 → 328.366s. Answer time: 67.486 → 66.452s. Whole decode: 395.745 → 394.818s (+0.235% throughput). The serial path and controller policy are unchanged; their timing variations and any workload regressions remain in the table.

## Retained changes

- A2: fold the existing BF16 rounding into small-batch Q8 output stores, preserving dot products and reduction order.
- A3: fold the same rounding into shared SwiGLU stores, preserving the activation expression.
- A4: synchronize independent pair/trio reductions together; each row retains its original SIMD reduction tree.
- A6: interleave six-row routed-down threadgroups in 256-output tiles. Per-token expert accumulation order is unchanged.

All four are Tier 1, same-arithmetic changes. The two new store-fold switches default off and are explicitly enabled in UAT: `DS4_DS41_MTP_Q8_ROUND=1` and `DS4_DS41_MTP_SWIGLU_ROUND=1`. The existing four opt-ins remain enabled. A6 operates inside the existing `MTP_DOWN_WIDE` path. No controller retuning, quantization change, Trench source change, or model-weight change.

## Verification and evidence

- Native Q8 folding and combined-reduction gates: 50,000 draws each for pair/trio kernels, finite bit-identical complete outputs, poisoned bounds, input controls and 1,000 deterministic repeats per kernel. Additional six-shape and linked-host row/stream fallback coverage.
- Shared SwiGLU gate: 50,000 draws across supported row counts, finite word equality, poisoned tails, input controls, 1,000 deterministic repeats.
- Down interleaving gate: 50,001 draws over no/partial/full overlap, finite complete-output equality, guards, per-draw input controls, 1,000 deterministic repeats per overlap case.
- Cached 8k/62k/300k fixtures, warmups followed by ABBA, all observations retained. CPU idle ≥90%, three GPU-utilization samples ≤5% before every request. No manual prefill; no model reload between candidate trials.
- Four normal-API content fixtures plus 14 held-out agent requests. Agent outputs, phase hashes and completion counts exactly match; 14,198 completion tokens per arm and all long prompts restored fully from cache. Tiny normal-API content warmups use the historical protocol.
- Agent requests alternate AB/BA, then repeat the first arm of every request. Analysis uses this predetermined selection, never best-of selection. Controller attempt counts can differ with timing; policy is unchanged.
- Forced rejection at 2/6 rows and 8k/62k/300k matches serial output. Production-host decode bandwidth, adaptive/controller/Markov tests and build pass.
- Final kernel bodies match the live experimental PSO sources after removing whitespace/comments and normalizing the down entry-point name. The exact benchmarked executable is preserved; any relink is audited by every Mach-O section.

## Verifier timing and historical byte ledger

Six-row verification: **91.86 → 90.78 ms/block** in paired controls. Public launch was 112 ms, before launch 177 ms. Using the same 37.657 GB/block ledger gives **415 GB/s**, versus 337 at public launch and 213 before launch. This is the same normalized weight-traffic calculation throughout, not a physical DRAM counter measurement.

## Reproducibility and rollback

Full results and scripts: `../bench/decode-answer50-20260915/`. Positive, rejected and inconclusive experiments are recorded in [the experiment ledger](DECODE-ANSWER50-EXPERIMENTS-20260915.md). See rollback instructions (`DECODE-ANSWER50-ROLLBACK-20260915.md`, local historical artifact). Model caches, executables, original machine configuration and paused-process identities remain local only. Deployment/restoration/canary receipts are appended after the guarded private-branch promotion.

## Deployment receipt

Private implementation commit: `1c0d363bd11497b31ca53b736930e7106a6fd4a6`. UAT runs the exact measured executable with all six expected opt-ins. Original caches, services, workers and matching paused processes were restored; diagnostic routes are closed. The full bridge→worker→DS4 canary returned READY with zero tools, errors, unmatched results or workspace changes. See `deployment.json`, `restoration.json`, `uat-health.json` and `uat-canary.json` in the evidence directory.
