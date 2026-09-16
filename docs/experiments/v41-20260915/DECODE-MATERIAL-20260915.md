> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Exact decode improvements — 2026-09-15

Status: deployed to UAT and pushed to private `v41-mtp-prod`. Implementation
commit `38e8c9d1287090e766c0d2ecd2a14c49fca91aae`. Deployment and restoration are recorded
in `DECODE-MATERIAL-DEPLOYMENT-20260915.json`.
Prior UAT: `0d24e543e69c162a4363671fe73052d6517b1d84`.
Scope: the private DS4 fork, DeepSeek V4.1 Flash Q4 on this M3 Ultra.

## Public branch → new UAT

| Decode workload | Public branch | Validated UAT |
|---|---:|---:|
| Serial, 8k context | 31.3 t/s | 33.2 t/s |
| Serial, 300k context | 28.3 t/s | 31.7 t/s |
| DSpark code, normal API/controller | 40.5 t/s | 44.4 t/s |
| DSpark agent turns, answer phase* | 41.3 t/s | 47.2 t/s |

*Agent answer throughput uses the same common estimated answer-token numerator
and pooled phase timings as the previous public/UAT table. The numerator is
3325.2733, derived from the public serial reference; it is not an exact tokenizer
count of streamed tool-call phases. Public values are the user's retained public
branch measurements. Prefill813t/s, TTFT31.1s and disk-KV restore0.23s were not
remeasured in this decode-only campaign.

## Materiality versus concurrent prior-UAT controls

Both arms use the same integrated resident process. Control selects the prior
kernel paths and disables the new memoization; candidate enables the six retained
changes. Controller policy, precision and per-output accumulation order are
unchanged. Quiet gates require three GPU utilization samples<=5% and CPU>=90%
idle. Every timed ABBA/BAAB result is retained; warmups are identified separately.

| Normal API workload | Prior-UAT control | Candidate | Gain |
|---|---:|---:|---:|
| Code | 42.36 | 44.41 | +2.05 t/s |
| Chat | 32.59 | 33.00 | +0.41 t/s |
| Prose | 32.67 | 33.23 | +0.56 t/s |
| SQL | 52.12 | 55.42 | +3.30 t/s |
| Agent answer phase* | 44.30 | 47.18 | +2.88 t/s |

The14 held-out agent requests produce14,198 completion tokens per arm, including
two4096-token reasoning-only cases. Every request is faster overall in the
candidate after the predeclared prepared-first-arm repeat protocol. Every output,
phase hash and completion count matches prior UAT. All long prompts restore
entirely from disk KV; emitted tool calls are never executed.

Pooled decode time: 409.369→399.515s (+2.47% throughput). Reasoning throughput improves1.60%; answer throughput improves6.51%. The material gain is in MTP answers; the serial improvement is smaller.

| Cached-prefix fixture | Control | Candidate |
|---|---:|---:|
| serial8 | 32.749 | 33.190 |
| serial300 | 31.296 | 31.712 |
| fixed8 | 38.026 | 40.346 |
| adaptive8 | 35.573 | 36.606 |
| agent8 | 35.863 | 36.947 |
| agent32 | 31.937 | 34.661 |
| fixed62 | 41.796 | 44.255 |

Different fixtures have different prompts and acceptance rates; compare arms
within a row. At32k, faster verifier cycles pass the existing controller's cost
threshold more often (3control versus27candidate cycles in the timed runs).
The controller itself was not retuned. An independent eight-run fixed-six8k
confirmation also measured38.233→40.436t/s with117identical acceptance cycles.

## Retained implementation

- F16 Engram projections reuse each weight load across three verifier rows,
  preserving each row's original dot products and reduction sequence.
- Q8 trio reuse is enabled only for outB and shared-down families (mask80).
- Six-row routed GU jobs are sorted by expert ID to improve locality. Existing
  cross-row interleaving already shared cache traffic; this is a refinement.
- Batch routing uses exact SIMD top-six selection; serial routing uses a
  hierarchical selection with literal bitonic fallback for ties/nonfinite data.
- A64-entry FIFO cache stores exact CPU-computed Markov biases per drafter.
  It changes neither proposal arithmetic nor acceptance/controller policy.

All six levers have independent zero-valued disable switches. Experimental
collapse splitting and profiling remain disabled. No quantization or Tier2 math
change is adopted. The public GLM fidelity rules were read from
`LOCAL_ARTIFACT/FIDELITY.md` and `EXACT-MODE-PLAN.md`.

## Validation and evidence

The selected kernels pass50,000-draw exact-output campaigns, poisoned outputs,
planted controls and1000 independently submitted determinism checks. Q8's50,000
draws are split evenly over its two enabled shapes. F16 alone compares7.68billion
finite exact words; GU compares12.44billion. Production host tests cover actual
row guards, tensor offsets and expert IDs through383; census runs prove the new
paths engage. Existing controller tests pass. All seven full-model fixture
comparisons and four normal API content comparisons are exact.

The six-row F16 variant failed by one F32 ULP and was rejected. Other failures,
neutral results, invalid initial timing setups, tooling interruptions and their
corrections are retained in the [experiment ledger](DECODE-MATERIAL-EXPERIMENTS-20260915.md).
Standalone harnesses, selected logs and all timed/warmup measurement metadata are
in the evidence package (`../bench/decode-material-20260915/README.md`, local historical artifact).
Raw artifacts remain at `LOCAL_ARTIFACT/decode-material-20260915`.

## Deployment and rollback

Only private `v41-mtp-prod` is promoted. No public branch or Project Trench source
is changed. Runtime experimentation reused one loaded model; one final integrated
build installation followed, with all subsequent A/Bs in that same process.
No manual long prefill was performed. The previous binary, three Metal sources,
plist, original133KV files and temporary-process/service ledger are retained.
Rollback instructions (`DECODE-MATERIAL-ROLLBACK-20260915.md`, local historical artifact) list every switch
and how to restore the prior runtime and machine state.


Restoration complete for the original plist,133KV files, four stopped services
and25 surviving paused processes. Debug endpoints return404; ordinary API smoke
returns READY from the validated resident PID84032. One paused VM process
(PID51394) exited during the session and could not be resumed; no replacement VM
was started and no reused PID was signalled. This exception is retained in the
receipt. The prior runtime backups and exact rollback instructions remain.


**Restoration correction (17:56 EDT):** The missing VM was Colima, which hosts
the UAT worker. Leaving it stopped broke the agent path despite healthy DS4.
The existing VM and containers have now been restored, and a bridge-to-worker-to-DS4
canary returned READY. See incident and recovery (`DECODE-MATERIAL-UAT-RECOVERY-20260915.md`, local historical artifact).


## Completion audit, 18:30 EDT

Rechecked the live PID and binary/Metal hashes against the tested release,
private remote `v41-mtp-prod`, Open WebUI/DS4 health and worker container state.
All match the deployed release; the restored workers remain running.
Rehashed all 23 archived result files with zero mismatches. Read the kernel,
production-host and controller gate results. Independently checked the selected
14 raw agent pairs for identical request/content/phase hashes and token counts,
complete cached prefixes, no speculative faults and positive total-decode gains;
recomputed all four normal-API ABBA means from individual records. Materiality
is met in MTP code/SQL and agent answer phases, not in serial alone. The public
comparison remains the table above; public values are historical user-supplied
measurements. No additional GPU run or model reload was needed for this audit.
