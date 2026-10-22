> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Second material decode improvement — 2026-09-15

## Result

Four exact verifier changes improve cached 8k fixed-six decode from **40.40 to 42.70 t/s** (+2.30) over the UAT build already deployed at the start of this campaign. This is a second increment; the prior campaign is not counted.

Normal-controller SQL improves 55.52 → 58.64 t/s; short code 44.55 → 46.11. Held-out agent answer-phase throughput is 46.96 → 49.34 t/s using the historical estimated token numerator. Serial reasoning is unchanged. The controller policy is unchanged.

## Public launch → new UAT

| Decode workload | Public launch | New UAT |
|---|---:|---:|
| Serial, 8k context | 31.3 t/s | 33.3 t/s |
| Serial, 300k context | 28.3 t/s | 31.6 t/s |
| DSpark code, normal controller | 40.5 t/s | 46.1 t/s |
| DSpark agent turns, answer phase* | 41.3 t/s | 49.3 t/s |

*Agent answer throughput uses the same common estimated answer-token numerator and streaming phase-timing protocol as the prior/public comparison. It is not a tokenizer-exact answer-token rate or whole-turn throughput. Raw durations and selection rules are retained. Prefill, TTFT, and disk-restore latency were not remeasured in this decode-only campaign.

## Same-build A/B against prior UAT

| Workload | Control | Candidate | Change |
|---|---:|---:|---:|
| Serial 8k | 33.286 | 33.279 | -0.008 t/s |
| Serial 300k | 31.636 | 31.633 | -0.003 t/s |
| Fixed-six MTP 8k | 40.404 | 42.704 | +2.301 t/s |
| Fixed-six MTP 62k | 44.247 | 46.733 | +2.486 t/s |
| Normal-controller code | 44.550 | 46.115 | +1.565 t/s |
| Normal-controller chat | 33.275 | 33.080 | -0.195 t/s |
| Normal-controller prose | 33.335 | 33.340 | +0.005 t/s |
| Normal-controller SQL | 55.525 | 58.635 | +3.110 t/s |
| Agent answer phase* | 46.964 | 49.336 | +2.372 t/s |

The small chat regression is retained, not hidden. The unchanged time-sensitive controller can choose different attempts when verification becomes cheaper. Its policy, loss budget, reasoning behavior, and safety/fault handling were not retuned.

Adaptive8 and fixed300 raw means contain approximately two-second delays in control runs and are not used for the materiality claim. All observations remain in the raw evidence. At 300k the verifier itself improves from about 108.6 to 102.7 ms/block; its clean final control is 47.61 t/s versus candidate 50.15/50.13. Different prompt acceptance makes these cross-depth MTP rates unsuitable for attributing context cost.

## Verifier timing and the Reddit bandwidth statement

Six-row verification is **92.08 ms/block**, versus 97.92 in the same-build control, approximately 112 at public launch, and 177 before the public optimizations. The launch-to-current latency reduction is 17.8%.

Using the old fixed ledger numerator of 37.657 GB/block gives **409 GB/s** (public: 337; prior UAT: approximately 384). This is a historical weight-byte equivalent, **not measured physical DRAM bandwidth**. Weight reuse and caching make it inappropriate to call this a fraction of the machine's 700 GB/s DRAM limit.

Suggested wording: “Six-row verification is now about 92 ms per block, down from 112 ms at public launch and 177 ms before optimization. The old fixed-byte ledger would correspond to about 409 GB/s; that is a normalized weight-traffic metric, not a DRAM counter measurement.”

## Whole-decode impact on the held-out set

Reasoning consumes 328.44 → 328.22 s; answers 70.80 → 67.40 s. Total measured decode is 399.24 → 395.62 s, a 0.92% throughput gain across this reasoning-heavy mix. The approximately +2 t/s criterion is met in MTP answer decode, not in aggregate serial-plus-answer throughput. These durations exclude prompt restore and tool execution.

## Retained implementation

- F32 router rows: use the existing scalar kernel and reduction tree in a single independent-row grid.
- Pointwise batching: batch query/KV RoPE and KV quantization before the chronological row loop, then inverse-head RoPE afterward. Keep attention, KV ring publication, and rollback ordering intact.
- Routed-down Q4: use wider quantized loads with the same dequantization and FP expression order.
- Expert sorting: build the same stable job order inside each SIMD group, avoiding the cross-SIMD exchange.

All are Tier 1 scheduling/load changes. The compact Engram grid failed its full-model comparison and remains off. Tier 2 alternatives lost their performance screens and were not adopted. Each retained change has an independent kill switch; UAT explicitly enables the four switches through its LaunchAgent.

## Validation and attribution

- Existing cached prefixes at 8k, 62k, and 300k; warmups then ABBA timing, with CPU idle ≥90% and three GPU-utilization samples ≤5% before every request. One resident build for all combinations; no repeated model reload tuning loop.
- Byte-identical short/deep outputs, with matching fixed-six acceptance counts. Four normal-controller content fixtures, then 14 held-out agent requests, 14,198 completion tokens per arm. Exact output/phase hashes and completion counts; all held-out prompt tokens restored from disk cache; no DSpark faults.
- Agent requests alternate AB/BA, then repeat the first arm of every request regardless of timing or outcome. Analysis uses that predetermined prepared-first-arm selection, matching prior UAT methodology. All initial runs remain archived.
- GU/down randomized gates: 50,001 draws each, no/random/full expert overlap, all finite exact output words, poisoned guards, detected controls, 1,000 separate deterministic submissions per overlap class.
- F32 rows: 50,000 draws, 115,200,000 finite exact output words, poisoned guards, controls, 1,000 deterministic submissions.
- Pointwise: strengthened gate has 50,000 draws at each of head counts 1 and 64, both RoPE frequency families/directions and short/wrap/deep positions, full-word exactness and guards, whole-row known-different controls with moved-word counts, and 1,000 deterministic submissions per geometry. The earlier weaker one-element control gate is retained as historical evidence.
- Forced-reject rollback at 2 and 6 verify rows, at 8k/62k/300k, matches serial bytes. No emitted agent tools are executed in replay.
- Build and production-host decode/controller/Markov tests pass; see release-checks-passed.json. No further numerical changes after these comparisons.

## Provenance and operations

Baseline private UAT: `6f4aff1a38c9cd3c26f6cf4d756a646f0a81db6b`. Experimental binary: `1dc29edfdf3203f7d8fac33aa4099a31e45e5481c2756896e83247e26482d1b7`. Runtime and full raw evidence are in `../bench/decode-material2-20260915/`. Only trailing EOF whitespace in moe.metal changed after the benchmarked shader was loaded; the final deployment receipt distinguishes both hashes.

The [experiment ledger](DECODE-MATERIAL2-EXPERIMENTS-20260915.md) records rejected screens and tooling failures. The rollback document (`DECODE-MATERIAL2-ROLLBACK-20260915.md`, local historical artifact) records all temporary machine changes and the exact prior build. Promotion is private-only; final service cutover, restoration, health, and canary receipts are recorded separately. Trench source, model weights, and controller policy are unchanged.

## Deployment verified

At 2026-09-15T20:33:24.331247-04:00, private UAT implementation `2b9b59136105677a2239fa7c2b30affe531d007a` is running as PID98498 with the exact benchmarked executable and all four opt-ins verified in the loaded launchd job. The final reload persisted the environment and rebuilt the restored disk-cache index; all tuning comparisons used the earlier single resident build.

All136 original KV files, three workers, four services, and all23 paused processes were restored. Colima was never paused. Diagnostic endpoints return404. The real `v41-agent` bridge → worker → DS4 canary returned READY/HTTP200 in 3.88s with no tools or workspace mutations; Open WebUI, bridge, and DS4 health checks passed. This is a functional check, not a speed benchmark. Receipts: `deployment.json`, `restoration.json`, `uat-health.json`, and `uat-canary.json` in the evidence directory.
