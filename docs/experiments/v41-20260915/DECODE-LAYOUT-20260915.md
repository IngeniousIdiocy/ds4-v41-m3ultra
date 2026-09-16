> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Scalar query projection layout — 2026-09-15

This campaign starts from private UAT `674bc4d` and improves scalar decode
with an exact query-projection layout change. Final installation is recorded
in `UAT-DECODE-LAYOUT-20260915.md`.

## Change

The scalar V4.1 query projection processes four independent output channels per
four-SIMD-group cohort instead of two. Each physical group still has two cohorts
and 256 threads. The fused grid now contains 4,108 groups instead of 8,204:
4,096 query groups plus the same twelve hyperconnection groups. Query weights
and arithmetic are unchanged. The lane-to-K map, ordered accumulation, both
SIMD reductions, BF16 store, hyperconnection completion counter and Sinkhorn
calculation retain their original order.

The shared expert keeps the original kernel. Its measured variants did not
improve elapsed time. The target six-row verifier, draft model, reasoning gate,
confidence threshold and adaptive controller are unchanged. This change targets
serial reasoning and the serial steps inside adaptive answer decoding.

`DS4_DS41_HC_STREAM_LAYOUT=0` restores the original mixed grid. The default is
2 (four channels). Diagnostic values 1 and 3 select specialized two- and
eight-channel query cohorts. Values outside 0–3 are rejected by the live setter;
invalid numeric environment values fall back to the default.

## Validation

The focused kernel test compares complete query output, HC mix, Sinkhorn state
and the completion count against layout 0. It passes 1,266,048 exact words,
including the unchanged shared-expert control. The first two exploratory logs
accidentally exercised layout 1 for requested values 2 and 3 because the new
lever had not been registered as numeric. They are superseded by the corrected
runs, which check the getter after every setter.

A short full-model screen selected four channels over eight: 32.94 versus
32.88 tokens/s, against 32.26 for the original. The larger microbenchmark gain
for eight channels did not carry through to full decode.

The independent 512-token cached-prefix ABBA comparisons currently show:

| Serial fixture | Original layout | Four channels | Gain |
|---|---:|---:|---:|
| Code, 8,192 tokens | 32.2307 t/s | 32.9604 t/s | 2.26% |
| Mixed, 300,000 tokens | 30.8093 t/s | 31.4292 t/s | 2.01% |

Both comparisons produce identical output. Absolute speeds across different
campaigns include run-to-run variation; use each campaign's interleaved control
to attribute an incremental gain. These are elapsed decode rates, not a claim
of measured physical DRAM utilization.

The original fourteen captured agent requests replayed with identical full
output and phase hashes: **14,198 completion tokens per arm**. All fourteen
prompts were completely restored from disk KV. The first arm of every request
was repeated after its continued cache checkpoints were prepared; this protocol
was frozen before the full set completed, and all original results are retained.

| Prepared agent set | Control seconds | Candidate seconds | Throughput gain |
|---|---:|---:|---:|
| All decode | 414.559 | 409.340 | 1.27% |
| Reasoning | 339.530 | 334.422 | 1.53% |
| Answer/tool emission | 75.029 | 74.919 | 0.15% |

The two 4,096-token reasoning-only cases improved from 127.851 to 125.412 seconds
and from 128.276 to 125.532 seconds. Matching serving-path content ABBA tests
improved prose by 2.15%, chat by 2.09%, code by 0.59%, and SQL by 0.02%.
Answer-phase gains from this scalar-only change are effectively neutral.

The controller still uses measured timing, so identical output need not imply
identical admission histories. A10 initially favored the control by about 0.2
answer seconds, but the balanced confirmation selected 45 cycles in both arms
and averaged 5.676 versus 5.663 answer seconds. A30 initially lost about 0.10 answer seconds with 18 versus 19 cycles. Its
balanced confirmation selected 18 cycles in both arms and averaged 3.236 versus
3.207 answer seconds, also favoring the candidate. Per-request variation is
retained rather than replaced selectively in the headline prepared-cache set.

### Intermittent benchmark outliers

Some auxiliary `/debug/bench` runs have about two additional seconds: the first
adaptive agent control, both adaptive prose candidates, and one fixed-code
candidate. They are not valid evidence of an MTP speedup or universal regression
freedom. The complete serving-path ABBA comparisons above do not reproduce those
large prose/code losses.

A separate negative control disables `hc_stream` in **both** arms, making the
changed kernel unreachable. It reproduces the fixed-code slowdown even in the
layout-0 arm: 33.11 t/s versus about 38.2 normally, drafting 3,189 ms versus about
1,128 ms, with verification still about 12,112 ms and the same 117 cycles. Thus
these stalls can occur independently of the new layout. The underlying host or
runtime cause is unresolved; no claim about CPU scheduling or thermal cause is
made. These diagnostic runs are excluded from all headline speed claims.

`make test-ds41-decode-bandwidth`, `make -j8 all`, and `git diff --check` passed.
The focused suite includes exact layout output and the pre-existing decode
kernel checks. No controller threshold, loss budget, verification depth, or
reasoning policy was changed.

## Public branch -> UAT

Same Mac, Q4 weights. Public values are the right-hand values in the original
Reddit comparison, used here as the left-hand baseline.

| Measurement | Public branch | New UAT |
|---|---:|---:|
| Serial decode, 8k context | 31.3 t/s | 33.0 t/s |
| Serial decode, 300k context | 28.3 t/s | 31.4 t/s |
| Prefill, 62k prompt | 813 t/s | Not remeasured |
| TTFT, 23k system prompt | 31.1 s | Not remeasured |
| Same prompt restored with disk KV | 0.23 s | Not remeasured |
| DSpark on code | 40.5 t/s | 42.5 t/s |
| DSpark on agent turns, answer phase | 41.3 t/s | 44.4 t/s |

The code figure uses the same original 132-token completion and server final
average measurement: candidate runs 42.51 and 42.44 t/s. The agent answer figure
uses the public table's phase-token estimation method: a common estimated
3,325.273 answer tokens divided by 74.918739 candidate answer seconds. It is not
an independent tokenizer count of answer tokens. Different campaign absolute
speeds fluctuate; the interleaved control above measures this change's increment.

## Resident validation and deployment

All full-model arms use one process on port 8195, PID 68058. Prepared fixture
snapshots and fourteen complete agent prefixes are APFS clones of earlier
verified caches. No long prompt is manually prefilled. Generated tool calls are
captured as output and never executed.

To keep this process through UAT promotion, an optional `DS4_DEBUG_LEVERS_FILE`
marker gates explicitly enabled diagnostic routes. It never enables diagnostics
on its own. Removing the marker closes `/debug/bench`, `/debug/prefill` and
`/debug/levers` without reloading weights; normal serving does not check the
file. The original launch plist is restored at the end. The currently loaded
job retains its marker-gated diagnostic environment until re-bootstrap; without
the marker those endpoints remain closed, including after a KeepAlive restart.

The resident executable was built with an initial layout-3 default; every timed
arm explicitly selects its layout. The final executable defaults to measured
layout 2. Promotion keeps the resident process on live layout 2 and installs the
layout-2 executable for subsequent starts. The default-selection constants are
the only runtime code difference; the GPU source differs only by the documented comment correction. Receipts must
record both executable hashes rather than conflate the running image and the
installed file.

Artifacts and machine change ledger:
`LOCAL_ARTIFACT/decode-layout-20260915`.
`ROLLBACK.md` records all temporary machine changes. `deployment.json` records
restoration and installed/resident provenance; `rollback-uat.py` restores the
baseline runtime and reverts the implementation commits on an unchanged head.
