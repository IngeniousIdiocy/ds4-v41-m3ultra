# September 16 consolidation: faster serial and speculative decode

The public branch now includes the nine exact optimizations deployed in UAT
`e3eaff1` (implementation `723e156`), on top of the
[earlier post-launch update](RELEASE-V41-20260916.md). **All accepted paths default
on.** DSpark uses its existing support weights and serving setup. There are no
quantization changes or new admission-controller policy changes in this bundle.

## Current results versus the original public launch

Same 512 GB M3 Ultra and Q4 weights. Serial rates below are best valid measured
runs; code is the mean of the bounded normal-controller serving ABBA check.
The latest code check emits 132 tokens; the launch code fixture used a 512-token
budget. The matched incremental code comparison is 47.115 → 48.730 t/s; the
launch-to-current code row is a record of reported results, not a matched A/B.

| Workload | Original public launch | Consolidated UAT paths now published |
|---|---:|---:|
| Serial decode, 8k context | 31.3 t/s | **35.9 t/s** |
| Serial decode, 300k context | 28.3 t/s | **34.4 t/s** |
| DSpark on code | 40.5 t/s | **48.7 t/s** |
| Six-row verifier block | 112 ms | **87.0 ms** |
| Normalized verifier weight traffic | 337 GB/s | **433 GB/s** |

The last full agent-answer measurement remains **50.0 t/s**, versus **41.3 t/s**
at launch; it belongs to the earlier update. This bundle passed a bounded
107k-context agent replay, but the full fourteen-request answer aggregate was
not rerun. Prefill **813 t/s**, cold 23k-system TTFT **31.1 s**, and disk-restored
TTFT **0.23 s** also remain earlier measurements.

Verifier bandwidth uses the unchanged **37.657 GB/block** logical-weight ledger
and pooled block time, not hardware DRAM counters. The latest paired verifier
comparison is **88.4244 → 86.9866 ms**, or **425.9 → 432.9 GB/s**.
These are validated UAT measurements, not a fresh benchmark of the public build.

### Incremental comparison against the immediately preceding UAT

Four alternating 512-token measured pairs at each depth, with separate 512-token
warmups, one resident server, restored disk KV and quiet-machine checks. Every
serial token ID and complete output matched. Best runs show demonstrated
capability; paired means show repeatability. Fast valid runs are retained.

| Context | Prior UAT best / mean t/s | New best / mean t/s | Paired gain ± SE |
|---|---:|---:|---:|
| 8k | 33.7219 / 33.6938 | 35.9347 / 35.9068 | +2.2130 ± 0.0242 |
| 62k | 33.6046 / 33.5757 | 35.7874 / 35.7141 | +2.1384 ± 0.0266 |
| 300k | 32.1208 / 32.0853 | 34.4069 / 34.1751 | +2.0898 ± 0.0654 |

Fixed-six code gains **+0.65/+0.75/+0.67 t/s** at 8k/62k/300k, with identical
outputs, acceptance histograms and work counts. The fixed-six agent fixture
gains **+0.61 t/s**. Different-depth fixtures have different continuations and
acceptance; compare matched arms within each depth. These fixed-row rates do
not substitute for the normal-controller answer metric.

Normal-controller code ABBA improves **47.115 → 48.730 t/s**. Eleven ordinary
API checks reproduce the prior outputs/phase hashes and completion counts,
with zero drafter faults. The fully cached 107k agent case takes **9.935 s**
versus **10.322/10.306 s** controls. Prose still declines drafting.

## What changed and why it helps

HC below is the model's hyper-connection residual mixing/expansion. Each fusion
retains the original logical accumulators, reduction order and BF16 rounding
boundaries; work is moved between GPU lanes or dispatches without changing
those operations. Unsupported shapes retain their existing fallbacks.

| Experiment | Accepted implementation | Engineering rollback |
|---|---|---|
| A1 | Fuse scalar Q8 attention output and the existing HC residual expansion; remove a dependent dispatch | `DS4_DS41_ATTN_OUT_HC=0` |
| R1 | Put independent router, shared gate/up and deferred HC work in one 780-group grid; keep selection afterward | `DS4_DS41_ROUTER_SHARED_HC=0` |
| Tpack | Pack the ninth Q4 expert-down block across independent output rows while preserving each accumulator | `DS4_DS41_DOWN_TAIL_PACK=0` |
| E1 | Fold the existing residual add/round/HC/round epilogues for two- and six-row verification | `DS4_DS41_MTP_HC_EPILOGUE=0` |
| M1 | Apply exact ninth-block tail packing to six-row routed expert down | `DS4_DS41_MTP_DOWN_TAIL_PACK=0` |
| Q1 | Put four logical Q8 K accumulators in one physical SIMD group for short shared-down/query shapes; preserve both reduction trees | `DS4_DS41_Q8_VIRTUAL_DOWN=0`, `DS4_DS41_Q8_VIRTUAL_QUERY=0` |
| K1/K2 | Fuse scalar KV normalization, RoPE, FP8 rounding and ring publication; use the validated 256-thread standalone fallback | `DS4_DS41_KV_PREPARE=0` |
| K3 | Run independent KV preparation alongside query/HC work in a 4,109-group grid | `DS4_DS41_KV_QUERY_MIX=0` |

K3 requires the eligible query and KV paths. Disabling it keeps standalone KV
fusion; disable both KV switches to restore the unfused KV chain. Disable all
nine switches above to isolate this bundle while preserving the earlier public
optimizations. To restore the previous public tree, build a separate checkout
at `d6ac8fa`; do not overwrite a running executable or reset unrelated work.

These gains aggregate several small wins. They do not establish that the
remaining GLM efficiency gap is inherent to DeepSeek or to higher-precision
backbone weights. In particular, exact tail packing addresses uneven work in
nine Q4 blocks; short Q8 remapping fills underused physical SIMD groups; the
mixed grids reduce exposed dependencies and launch overhead.

### Controller behavior

The earlier bounded-loss controller remains unchanged. Every served request
starts with fresh cost evidence and uses the median of its last nine serial
timings. In the code warmups, measured serial cost adjusts **28.862 → 27.349 ms**
automatically; no hard-coded timing adjustment is needed. Code retains seventeen
verified cycles and six declines; the 107k agent retains forty-seven cycles and
twelve declines. Prose retains zero cycles and eighteen declines. Reasoning
continues to decode serially by default, so the serial gains matter directly.

## Failed experiments and what we learned

Rejected implementations and switches are removed. Measurements, numerical
failures and corrections remain in the [complete chronological ledger](../bench/serial-fusion-20260916/EXPERIMENTS.md).

| Experiment | Outcome and decision |
|---|---|
| D1: move the down consumer into shared-down | Exact but slower; rejected. |
| T1: unroll nine Q4 blocks | Retains the 3/2/2/2 imbalance; slower. Actual tail packing replaces it. |
| NR4/NSG geometry screens | Useful screens did not beat the retained packed-tail schedule; measurements retained. |
| P1/Ppack: GLM-style expert partials | Faster, but the changed sum association fails Tier 2 S2/S4/S5. Mean NLL drift magnitude 0.00082845 exceeds 0.0001; average greedy prefix changes 12.14→12.28 beyond ±0.1. Rejected despite improving mean NLL. |
| C2: exact cooperative expert staging | Exact full gate passes, but loses 0.1711 t/s to Tpack across all four pairs; removed. |
| P2: three expert pairs | Changes output; only +0.0413 t/s over the exact alternative. Rejected without claiming a completed quality gate. |
| N1: concurrent shared-down/routed gate-up | Kernel screen wins fail to transfer: −0.3951 t/s in native decode. Removed. |
| Q8 NR1 and large Q1 shapes | Slower query/large projections; only the measured short shapes are retained. |
| Initial K1 normalization rewrite | Fails exactness at draw 1102 and native token 245. Restore original dynamic normalization loops; repaired kernel passes full gates. |
| Narrow standalone KV fallback | Superseded by the faster validated 256-thread form; redundant kernel/switch removed. |
| Harness and measurement mistakes | Wrong initial HC reference width, ineffective query-layout selection, inadequate agent warmup, stale PSO checker, finite-input/checker issues and compile failures are recorded with corrections. |

No failed numerical experiment enters this release. Faster microbenchmarks are
not treated as native gains, and individual screen gains are not added together
to manufacture the final result.

## Validation and reproduction

Every retained kernel passed at least 50,000 fresh randomized draws, finite
output comparison, poison/guard checks, differing-input controls and 1,000
identical-input repeats. The combined query/HC/KV gate compares **1.69205 billion**
finite output words. Full serial/native, two/six-row, depth and forced-reject
contracts pass. Exactness refers to the tested supported paths and gates.

The public integration preserves UAT's retained shader arithmetic and the
public tree's previous rejected-path removals. It is built and checked without
loading a second model or restarting UAT. The [evidence bundle](../bench/serial-fusion-20260916/README.md)
contains original numerical receipts plus fresh public build/default/rollback,
production-path regression and nine-pipeline compilation checks. It also binds
source and executable hashes. CUDA/distributed performance is not newly certified.

The debug-only resident scorer (`/debug/score`) uses strict FP64 quality metrics
and cached teacher-forced prefixes. `DS4_BENCH_REQUIRE_DISK_KV=1` makes benchmark
cache misses fail; `DS4_BENCH_REQUIRE_CACHED_API=1` rejects uncached long prompts
in diagnostic serving. These support experiments without repeated weight loads
or repeated long prefills; do not enable diagnostic routes in a public service.
The existing [reproduction guide](REPRODUCE-V41.md) covers fixtures and A/B use.
