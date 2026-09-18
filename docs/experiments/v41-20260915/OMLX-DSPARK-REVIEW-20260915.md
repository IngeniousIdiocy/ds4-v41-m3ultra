> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# oMLX DSpark implementation review — 2026-09-15

Scope: ideas for faster DS4 V4.1 decode. No oMLX server or model was installed;
third-party code was downloaded for inspection only. The DS4 candidate under
validation remains frozen while this review runs.

## Sources inspected

- [V4.1 support, merged PR 3574](https://github.com/jundot/omlx/pull/3574),
  head `9968f20124cf0c5168e0ef69eefc03752acfc375`.
- [Earlier V4 DSpark, merged PR 2460](https://github.com/jundot/omlx/pull/2460),
  head `80ccd97171b6382f08359a14440c4fa52afe7c31`.
- [Alternative V4 implementation, open PR 2468](https://github.com/jundot/omlx/pull/2468),
  head `2d82b8729211894a9fd147c3ff290d04015c26c3`.
- [Scheduler-native PR 2441](https://github.com/jundot/omlx/pull/2441) was screened;
  its generic provider integration is outside this DS4 decode task.

Pinned source copies and PR metadata are under
`LOCAL_ARTIFACT/omlx-reference`.

## Findings mapped to our engine

| Mechanism | What the inspected code does | DS4 decision |
|---|---|---|
| GPU Markov chain | The shared DSpark loop keeps the previous-token projection, logit addition and token selection in an MLX GPU graph, then submits drafts asynchronously. | **Profile next.** Our V4.1 `ds41_dspark_markov_greedy` reads all five base-logit rows and runs five rank-256 CPU matvecs plus argmaxes. This is a concrete difference. |
| Specialized short-row projection | Earlier V4 `dspark_exact_mxfp8_qmv_pair` specializes 2–6 rows, reuses weight loads across rows, and computes paired projections in one dispatch. | Our deployed six-row Q8 pairing/swizzle and routed gate/up swizzle already pursue this. Their MXFP8 layout is not a drop-in replacement for Q8_0/Q4_K. |
| Grouped short expert pipeline | V4.1 `GroupedExpert` submits existing MLX gate, up, activation and down primitives through one native encoder. The short-block path retains separate per-token quantized inputs. | Useful confirmation of reducing scheduling overhead, but it does **not** add cross-row expert-union weight reuse. Their expert sorting path starts at 32 tokens; six-row verification does not use it. DS4 already directly encodes its GPU graph and fuses gate/up. |
| Accepted-boundary restoration | V4.1 commits the verified causal prefix and restores window, compressor and Engram history to the accepted boundary without a target replay. | Already present in `ds41_verify_commit`; preserve this invariant. |
| Acceptance-only depth policy | V4.1 sets the next depth to accepted drafts plus one and never parks for poor wall-clock throughput. Its comment attributes this to different numerical rounding across row counts. | **Do not transplant.** It would abandon our measured-cost fallback and does not establish a benefit on long serial reasoning or hostile prose. |
| Removing rollback snapshots | The alternative V4 PR elides a fallback snapshot based on it never having been needed in its observed runs. | Do not remove our recovery guarantees on that evidence. Only optimize a measured copy cost while retaining correct restoration. |
| Engram lookahead reads | V4.1 submits the next Engram layer's selected pages while GPU work progresses. | DS4 already has asynchronous Engram loading. Review is not evidence that this Mac's disk latency is the main remaining bottleneck. |

## Why the Markov lead needs a measurement

The original local Stage 4 note dismissed the CPU chain by comparing roughly
176 MB of chain reads with 22–23 GB of target verification reads. Those byte
counts use different execution resources and do not establish elapsed cost:
the CPU has a different attainable bandwidth, and five dependent matvecs also
pay thread dispatch, reduction and synchronization costs. Measure milliseconds
per proposal and per committed output token before deciding on a GPU port.

A correct port must preserve the existing CPU Q8 activation quantization,
ordered NEON FMA/reduction arithmetic, tie-breaking and post-Markov confidence
inputs. Plain floating-point GPU matvecs would change that contract. Confidence
thresholds and six-row verification remain fixed in such an experiment.

## Comparing published performance

The V4.1 PR reports M3 Ultra 512 GiB results using oQ4e, temperature 1, 128-token
samples and different prompts. Our measurements use Q4_K/Q8_0 and matched greedy
fixtures or captured agent requests. Its advertised percentage gain over its
own serial baseline cannot be applied to our already faster baseline. Its RAM
versus SSD table also mixes different sampled continuations; it is not an
isolated latency measurement of Engram on our engine.

No new performance claim or adoption follows from this source review alone.


## Measured follow-up

A separate 1,024-token fixed-six code diagnostic sampled the decode thread for
10 seconds. Of 6,856 thread samples, 118 were inside the Markov chain, including
its logits readback: **about 1.7%**. Applying that fraction to the diagnostic's
113.86 ms average cycle estimates **about 2 ms per proposal**. This is a sampled
estimate, not an exact section timer; confidence admission was disabled in this
fixed-six control. Raw sample and summary are retained in the campaign directory.

Even a zero-cost replacement would therefore improve this observed fixed-six
case by only about 1.8%; a real GPU port would save less. The result supports a
small follow-up experiment, not a large explanation for the remaining bandwidth
gap. No Markov implementation or controller policy from oMLX is shipped in this
release. The profile run's throughput is excluded from adoption measurements.
