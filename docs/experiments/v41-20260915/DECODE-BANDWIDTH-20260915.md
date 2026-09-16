> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# V4.1 decode bandwidth results — 2026-09-15

The candidate improves fixed six-row native MTP by **6.7–14.7%** across the measured fixtures. Adaptive MTP improves **1.6–9.6%**. Short-context serial decode is unchanged; the 40 tokens/sec serial target remains unresolved. These are elapsed-time gains for identical output, not physical DRAM-counter utilization.

## Final uninstrumented ABBA results

Values are mean steady committed tokens/sec, two measurements per arm. Warmups are excluded. All 36 timed requests restored cached prefixes. All nine comparisons produced identical text; fixed-MTP acceptance histories also matched exactly.

| Mode | Fixture / context | Baseline | Candidate | Gain |
|---|---|---:|---:|---:|
| fixed | agent.txt / 8,192 | 31.51 | 33.63 | +6.72% |
| fixed | code.txt / 8,192 | 36.49 | 38.98 | +6.81% |
| fixed | depth-mix.txt / 131,072 | 45.13 | 49.12 | +8.84% |
| fixed | depth-mix.txt / 300,000 | 39.83 | 45.70 | +14.75% |
| adaptive | agent.txt / 8,192 | 34.52 | 35.09 | +1.64% |
| adaptive | code.txt / 8,192 | 32.67 | 34.68 | +6.15% |
| adaptive | depth-mix.txt / 131,072 | 44.30 | 48.53 | +9.56% |
| serial | code.txt / 8,192 | 31.67 | 31.64 | -0.08% |
| serial | depth-mix.txt / 300,000 | 28.46 | 30.28 | +6.38% |

For fixed six rows, mean complete draft + verify + commit cycle cost:

| Fixture / context | Baseline ms | Candidate ms | Saved ms |
|---|---:|---:|---:|
| agent.txt / 8,192 | 119.88 | 112.32 | 7.56 |
| code.txt / 8,192 | 119.69 | 112.05 | 7.64 |
| depth-mix.txt / 131,072 | 127.22 | 116.89 | 10.33 |
| depth-mix.txt / 300,000 | 142.15 | 123.85 | 18.29 |

Two repeats per arm establish local repeatability, not a broad statistical confidence interval. Before timed arms, the last three GPU utilization samples were all <=1%; minimum CPU idle was 91.39%. VM counters recorded no swap-outs and twelve total swap-in pages across one arm.


## Interpretation

The optimization objective is committed tokens divided by draft + verify + commit time. A six-row verification is useful only to the extent that its draft tokens are accepted. For example, this code fixture commits 511 tokens in 117 cycles: 4.3675 tokens/cycle. Faster six-row kernels improve this denominator directly, while faster scalar-only kernels mainly benefit adaptive fallback. Index scoring runs for each verifier row, so improving a nominally single-row kernel can also improve the entire six-row cycle.

The GLM comparison remains a useful diagnostic, but active parameters alone do not predict elapsed time. V4.1 has routing, mHC, normalization, index selection and thousands of dependent dispatches. The historical single-row byte ledger is approximately 13.77 GB/token; 40 tokens/sec requires about 551 GB/s of logical work and a 25 ms step. This is plausible against the measured 705 GB/s memory-copy ceiling, but assumes away non-streaming work. It does not establish that the existing decoder can sustain that rate.

The old six-row 37.657 GB/cycle ledger in the public description is not a physical DRAM counter. Pair6 loads some Q8 weights as three streams, stream6 loads other families as one, and repeated loads can hit caches. Therefore neither the old 337 GB/s estimate nor a scaled version is a valid measured fraction of hardware bandwidth. This campaign demonstrates improved elapsed time for identical work; it does not claim a measured DRAM utilization percentage.

Existing production work already overlaps Engram disk fetches with GPU execution. A prior six-row timeline placed about 114.45 ms of a 119.08 ms cycle inside GPU command-buffer intervals, leaving about 4.63 ms outside. That is not shader occupancy or a precise SSD-wait measurement, but it argues against assigning the whole gap to disk. This campaign changes no Engram I/O code and still produces the measured gains.

## Changes

1. Q4 routed gate/up: reorder independent output tiles across the six verifier rows and routes, using tiles of 16 original threadgroups. Keep existing SIMD width, arithmetic, and accumulators. This improves the opportunity to reuse nearby weight tiles in cache.
2. Pair6 Q8 projections: reorder independent jobs in tiles of eight output threadgroups across the three row pairs. Keep the current arithmetic and per-family pair6/stream6 selection. The earlier attempt to collapse every family into six accumulators lost performance; this change avoids increasing accumulator pressure.
3. V4.1 index scoring: stage the query once for 32 keys, retain four keys per SIMD group, and replace the repeated per-head threadgroup barriers with one staging barrier. Preserve float4 dots, SIMD reductions, head accumulation order and an explicitly rounded F32 product. Direct, masked and compact paths share the body. Selection masks, candidates and overflow semantics remain unchanged.

The GLM scorer API retains its original path. The new V4.1 scorer API aliases the original implementation on non-Apple builds. The swizzles apply only to the supported V4.1 six-row shapes; scalar and other row-count controls retain the existing mapping.

## Measurement protocol

M3 Ultra 80-core GPU, 512 GiB RAM; original production baseline `03eb931`. Candidate source `6cd068e6ca944869a7053a933294ae079187dc08`, server SHA256 `6d67ed630403c82f70491edf67c92e86003fddea84bad974d5c9e9e40ec19018`. The later Makefile cleanup only removes unrelated test-flag edits; it changes no model or kernel binary.

One resident model server, loopback port 8199, isolated APFS-cloned KV snapshots. All three candidate levers change together between 0 and 1. Other production optimizations remain enabled. Fixed MTP uses six verifier rows, five drafts, adaptive disabled. Adaptive tests enable the existing controller. Each native comparison generates 512 tokens, with two baseline and two candidate runs in ABBA order after explicit specialization warmups. Serial comparisons generate 256 tokens. The fixture harness uses `dspark_excl_eos=1` to complete its fixed token budget; production defaults to 0. Both arms use the same EOS policy, and raw request levers are archived. The chat smoke test also held this lever at 1; it is not a default-policy end-to-end throughput claim.

Before every arm, require three consecutive GPU utilization samples <=5% and CPU idle >=90%. No compilation or other GPU tests overlap final timing. One transient 85.55% CPU-idle gate stopped the matrix before the 300K warmup; the remainder resumed once quiet. This stop produced no timed sample. Every record stores quiet samples, VM statistics, complete output, acceptance statistics, fixture hash, levers, source commit and binary hash. All final fixture runs restore a cached prefix without prefill.

## Correctness and scope

`make -j8 all` passed. `make test-ds41-decode-bandwidth` passed with strict host floating-point checks: 2,450,010 exact score words; 1,271,808 exact Q8 output words; real-shape Q4 routed MoE results exactly matching the existing scalar oracle for rows 1..8. Includes score masks, empty masks, compact overflow, odd tails, tensor offsets, row-count controls and unchanged GLM entry behavior.

The generic `make test` passed its early unit tests, then reached the live-model suite and correctly refused a second huge model while the benchmark server held the instance lock. That suite is not claimed passing. CUDA and tensor-parallel hardware were not tested. Measurements establish local fixture improvements, not a universal gain for every prompt or Mac.

## Rollback

Candidate branch: `v41-bandwidth-20260915`, isolated worktree `LOCAL_ARTIFACT/ds4-v41-wt-bandwidth`. No push or production deployment. Each change has an independent runtime rollback: `DS4_DS41_MTP_GU_SWIZZLE=0`, `DS4_DS41_MTP_Q8_SWIZZLE=0`, `DS4_DS41_INDEX_SCORE_STREAM=0`; the equivalent debug levers omit the prefix and use lowercase names.

Original production binary, source and launch plist remain untouched. `restore.py` stops the benchmark, bootstraps the original launchd service on port 8195, confirms health, and resumes only the processes paused by this campaign after matching their commands. `ROLLBACK.md` records the initial service-stop correction and the complete change ledger. No power, OS or global environment settings changed. Original disk KV cache is retained; original live RAM cache is rebuilt by service restart.


## Artifacts

Raw records, complete output, scripts, hashes and logs: `LOCAL_ARTIFACT/bandwidth-20260915`.

`final-summary.json` links each result to its four raw records. `bench.py` and `final-matrix.py` reproduce the protocol on a resident isolated server; `final-matrix-resume.py` records the quiet-gate continuation. The unprofiled command is saved in `benchmark-command-unprofiled.json` once instrumentation begins. Focused test output: `tests-strict-host.log`; build: `build-pinned.log`; full-suite limitation: `regression.log`.

## Serving API and kernel attribution

The normal `/v1/chat/completions` endpoint passed five 512-token requests (one warmup plus ABBA), with byte-identical choices and no drafter faults. This 8,759-token review prompt failed the existing drafter confidence admission and used serial fallback in every arm. It is a serving correctness check, not evidence of native-MTP acceleration. Complete responses, cache accounting and server logs are in `chat-*.json`; summary in `chat-summary.json`. End-to-end wall times include prefix restore/prefill and are not used for the decode headline.

A separate 128-token profile per arm used the same binary with `DS4_KERNEL_LEDGER=2`, `DS4_KERNEL_LEDGER_PAGES=32`, and `DS4_KERNEL_LEDGER_BY_TG=1`. All four ledgers report **100% coverage, complete=YES**, with no counter-pool exhaustion or resolution failures. The selected families occupy solo timed passes, so no shared-pass attribution is included in this table.

| Context | Kernel family | Baseline ms/cycle | Candidate ms/cycle | Saved ms/cycle |
|---|---|---:|---:|---:|
| 8,192 | Routed Q4 gate/up | 26.66 | 21.98 | 4.68 |
| 8,192 | Paired Q8 projections | 15.37 | 12.97 | 2.40 |
| 8,192 | Index scoring | 1.66 | 0.97 | 0.69 |
| 300,000 | Routed Q4 gate/up | 27.11 | 23.00 | 4.11 |
| 300,000 | Paired Q8 projections | 13.95 | 12.76 | 1.19 |
| 300,000 | Index scoring | 19.00 | 6.39 | 12.61 |

Instrumentation disables encoder batching, the concurrent KV-normalization task and parallel FFN. Its absolute cycle times and deltas must not replace the uninstrumented results or be summed to predict them exactly. It confirms that the changed kernel families improve; it does not measure physical DRAM traffic. Raw ledgers and exact per-family numbers are in `profile-final-*.ledger` and `profile-summary.json`. The first profiling readiness probe used unsupported `/health`; it was corrected to `/v1/models` and the already-running server continued without another model load.

## Restored machine and review state

Restoration completed at **2026-09-15 14:46:48 UTC**. Original production service is ready on port **8195**, PID **41046**, running the original `LOCAL_ARTIFACT/ds4-server`. All **24** paused background processes resumed after command identity checks. A follow-up check confirmed none remained stopped, the launch plist is byte-identical to its backup, the production checkout is clean, and only the original model server is running. Evidence: `restoration.json`, `restoration-verification.json`, `restore.log`.

Candidate code remains isolated on `v41-bandwidth-20260915`. Implementation commit: `6cd068e6ca944869a7053a933294ae079187dc08`; test-Makefile scope cleanup: `14cd2b8a21518c7d3167328f05ae1e104d9f6f8c`. Nothing was pushed or deployed. Measurements, cloned benchmark snapshots and this report are retained as session artifacts; original production KV files were not modified.

## Prior evidence consulted

- `LOCAL_ARTIFACT/MTP.md`, especially H1 and its corrected byte-ledger discussion.
- `LOCAL_ARTIFACT/CYCLE-COSTMAP-V2.md` and `DEPTH-STAGES.md`.
- `LOCAL_ARTIFACT/REPORT.md`.
- User-provided public description: https://www.reddit.com/r/LocalLLaMA/s/m02CYvb6oZ.

## Subsequent UAT deployment

At Mark’s request, these changes were subsequently pushed to the private `v41-mtp-prod` branch and deployed to UAT on 2026-09-15. The preceding restoration section records the end of the original experiment; current deployment status and rollback are in UAT-BANDWIDTH-20260915.md (`UAT-BANDWIDTH-20260915.md`, local historical artifact).
