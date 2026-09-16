> Historical research record, published with the September 16 update.
> Status/default/deployment statements below describe that experiment's stage.
> For shipped defaults, removed candidates, and current results, see
> [the release update](../../RELEASE-V41-20260916.md).
> Local artifact names are provenance references; machine-control scripts,
> private request bodies, cache files, and full conversation outputs are not
> distributed. Numerical summaries are published in the release evidence bundle.

# Faster DS4 decode — 2026-09-15

Baseline: deployed private `v41-mtp-prod` at `7c7a5e4`. This is a second campaign;
all gains below are incremental to the previously deployed bandwidth changes.
Scope is DS4 decode, with serial reasoning and native answer-phase MTP measured
separately. No prefill, first-token latency, Trench or UI changes.

## Implementation

1. `dae13bf`: preserve the standalone expansion's ordered FP32 FMAs inside the
   existing scalar routed-FFN producer epilogue, then enable that producer by
   default. Removes one standalone HC expansion dispatch per layer on eligible
   single-row Metal steps. Supports standard and wide Q4 variants. Existing
   streaming/TP guards remain in place; the six-row verifier is unchanged.
2. `89dbb49` controller: carry measured net losses across three-attempt windows
   until unrepaid loss exceeds half a measured median serial step. Wins repay
   debt without accumulating credit. Exceeding the threshold invokes the existing
   16/32/64/128 cooldown. Clear debt on backoff, evidence reset and reasoning exit.
   Actual loss accounting remains intact. Reasoning stays serial; confidence
   floor .75, minimum useful draft prefix 3, six verifier rows and entry delay
   are unchanged. Diagnostic `DS4_DS41_DSPARK_LOSS_BUDGET=0` restores the old
   immediate-backoff decision.

The controller adjustment is needed because cheaper serial tokens move the
break-even point. The kernel-only build repeatedly took 29 verified cycles
instead of 41 on a code fixture, losing about 1.6% to the deployed build. The
initial startup-only explanation did not survive an explicit warmup and full
ABBA repeat. Those unfavorable results remain archived. The revised candidate
was frozen before replaying the historical agent/prose gate.

## Serial measurements already complete

512-token cached-prefix ABBA runs, one resident model, quiet GPU and CPU:

| Fixture | Baseline committed tok/s | Fused producer | Gain |
|---|---:|---:|---:|
| Code, 8,192-token depth | 31.360 | 32.466 | 3.53% |
| Mixed fixture, 300,000-token depth | 29.985 | 31.047 | 3.54% |

Both output sequences match exactly. The earlier independent sweep also gained
3.3–3.6% on code, agent and deep-context serial cases.

## Native adaptive screen already complete

Three warmups followed by A/B/C/C/B/A, 512 tokens per arm. A is deployed behavior,
B is kernel-only, C is kernel plus bounded-loss controller. All output hashes
match. Fixed-six verifier performance is unchanged; these gains result from
cheaper serial work and better admission near break-even.

| Fixture | A tok/s | B tok/s | C tok/s | C vs A |
|---|---:|---:|---:|---:|
| Code, 8K | 34.754 | 34.128 | 35.532 | +2.24% |
| Agent, 8K | 34.921 | 35.768 | 35.764 | +2.41% |
| Second agent, 32K | 30.674 | 31.663 | 31.649 | +3.18% |
| English prose, 8K | 30.665 | 31.865 | 31.806 | +3.72% |

Small C/B differences outside code are below 0.2% in this screen. They are not
claimed as controller wins. Real answer-phase regressions still block adoption.

## Quality and regression checks already complete

- `make -j8 all` and controller unit tests pass.
- `make test-ds41-decode-bandwidth`: 3,932,672 exact FFN producer-chain words,
  2,450,010 score words, 1,271,808 Q8 words, plus Q4 routed batches for rows 1–8.
- The same FFN test against the old fused kernel fails, so it detects the original
  rounding defect rather than merely reproducing implementation assumptions.
- Full model: 114 quality cases and 3,018 scalar-decode-produced full vocabulary
  vectors (129,280 FP32 logits each) match exactly; score fields also match.
- Metal SSD-expert and SSD-cache regression tests pass. No CUDA or distributed
  implementation changes. The scalar producer retains its original applicability
  guards, and the adaptive controller is confined to the Apple V4.1 path.

## Final replay protocol — passed

Direct DS4 chat API replays, with generated tool calls captured but never
executed. Four historical prose/chat/code/SQL stress prompts use balanced
A/B/C/C/B/A runs. The 14 existing agent request bodies retain a 4,096-token cap
and natural EOS. Two reasoning-only requests compare A/C; the other twelve
compare A/B/C. First visible to last visible output measures decode; the first
answer/tool delta splits reasoning and answer time. Preparation and first-token
latency are excluded.

The historical cloned cache contained obsolete entries. Before later outcomes
were known, the protocol was corrected to repeat the first arm of every agent
request on its prepared cache, preserving initial arms as preparation records.
The initial process also retained an earlier 1,024-token cap on the two
reasoning-only requests; both A/C arms are repeated at 4,096. Thus 16 prepared
repeats are required, along with exact request, text, phase and completion-count
agreement. The final protocol checker enforces this and the tested source/binary
hashes. No deployment is authorized by incomplete summary files.

All 16 prepared repeats completed, with all 40 selected arm records matching
request, output, phase and completion-count checks. The two long reasoning
requests reached 4,096 tokens in both arms, with no MTP attempts. The complete
14-request set produced 14,198 tokens per A/C arm. No generated tools were run.

| Phase, pooled across matched requests | Baseline seconds | Candidate seconds | Throughput gain |
|---|---:|---:|---:|
| All decode | 423.858 | 410.793 | **3.18%** |
| Reasoning | 348.861 | 336.625 | **3.63%** |
| Answer and tool emission | 74.997 | 74.169 | **1.12%** |

The two 4,096-token reasoning-only cases improved by 3.73% and 3.81%. Against
kernel-only B, pooled answer speed improved 0.70%; the worst individual answer
speed difference was -0.39% (roughly 12 ms on a 3.2-second answer). Most controller
choices were identical. Two requests had better MTP decisions; these account
for much of the small pooled answer gain. This is evidence of preserved agent
answer performance, not a claim that the controller improves every workload.

The original prose/chat/code/SQL stress replays also matched output in all 24
timed arms. Candidate mean decode time beat deployed behavior in all four.
Middle-chat had a 0.37% mean cost versus kernel-only B: one candidate arm made
an extra verification attempt, while the other matched B's cycle count. This
small observed exploration tradeoff is retained rather than hidden by pooling.
There was no material answer-phase regression in the real-agent gate.

## Limits

These are measured decode gains, not a claim of physical DRAM bandwidth
utilization. The fixture suite is finite and the controller can still respond
to content and timing. Very short answer phases have timing noise. Existing
reasoning and answer behavior must be assessed separately.

See `docs/OMLX-DSPARK-REVIEW-20260915.md` for the parallel source review requested
by the user. No oMLX implementation has been adopted into this candidate.


## Adoption, provenance and rollback

Adopt the fused scalar producer and bounded-loss controller together. The
kernel-only candidate is rejected because of its repeatable adaptive code
regression. Implementation commits are `dae13bfe05b2a63ef9ecdfc86c6ae3f1861e3f4d`
and `89dbb49cf06907b983c2efe771b339e118f5fcd7`.

The tested server SHA-256 is
`1167e29419bfce85fa44772e0ba00d3de2567c1acb5234e0760611ad2ddca9f2`.
The final agent/prose replays and profiling used this exact binary. Earlier
controller-screen code differs only in comments; Metal hashes match. The scalar
full-vocabulary gate uses the same `metal/dsv41.metal` SHA-256
`f65d391ccc5eb904693ce05722044639bae93937d0eb95cdcb6168725d86e3fb`.

Artifacts: `LOCAL_ARTIFACT/user-speed-20260915`.
Key files: `serial-confirmation-summary.json`, `controller-screen-summary.json`,
`replay-agent-summary.json`, `replay-protocol-pass.json`, `score-gate-pass.json`,
`tested-source-sha256.json`, and the raw request/boot records. Superseded initial
and unfavorable sweeps remain available. The instrumented Markov profile is
explicitly excluded from throughput comparisons.

UAT promotion copies the exact tested executable and fast-forwards the clean
private production branch. The original launch plist is unchanged. The machine
ledger is `ROLLBACK.md` in the artifact directory; `restoration.json` and
`deployment.json` record actual restoration and deployment status. The service
and 24 background processes paused for isolated benchmarking are restored by
`restore.py`, matching command lines before resuming any process.

To undo this release on an otherwise unchanged UAT checkout:

```sh
python3 LOCAL_ARTIFACT/rollback-uat.py
```

The helper refuses later source/plist changes, reverts the two implementation
commits in reverse order, restores the backed-up baseline executable, and checks
readiness. It retains audit documents and does not push the local revert.
`DS4_DS41_FFN_PRODUCER=0` and `DS4_DS41_DSPARK_LOSS_BUDGET=0` are diagnostic
runtime controls; disabling only the controller is not the adopted combination.
