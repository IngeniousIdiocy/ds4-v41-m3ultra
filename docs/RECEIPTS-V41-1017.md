# V4.1 1017 decode update: receipts

The measurements and exactness checks behind the 1017 decode commits on this
branch. Each commit message carries its own what/why and ledger entry; this
page collects the results that arrived after a commit landed, or that span
several commits.

Glossary for the labels in this page and in the commit messages:

- **Worker A to worker D**: the parallel implementation agents of the serial
  40 tok/s campaign that produced this branch.
- **Codex**: the reviewing architect model.
- **Window labels** such as `fp8pf-a5`, `irqres-a1`, `w1-a2`, `W5` or `W72`:
  measurement windows, each a protected run on the M3 Ultra. This page
  summarises their evidence.
- **P numbers** (P11 to P27) are the results recorded for those windows.
  **Ledger ids** (`S40-*`, `D0xx`) are entries in the campaign's decision
  record. That record is not published.

Unless stated otherwise: one M3 Ultra (512 GB), the 1017 MXFP4 target
weights, decode at concurrency 1, and the campaign's prose and code fixtures.

## Composed run of this branch at 8K (fp8pf-a5 d01, P25)

Setup:

- The tree: a clean export of 77eea27c, the default build, with no
  environment set.
- Each fixture is prefilled cold to 8K. The W72 fixtures (prose and code) are
  used.
- Each run samples 512 tokens at temp 1 and top_p 0.95, with seed 1017. There
  are two runs per fixture.
- The rate uses our metric: first sampling to last evaluation.

| Fixture | Run 1 | Run 2 | prefix | logits | state |
| --- | ---: | ---: | --- | --- | --- |
| prose | 40.437 | 40.482 | `3cd4d4bba0ab9986` | `ed3ca952608e6274` | `6d2d529ba8c0e5c0` |
| code | 40.508 | 40.418 | `166bd265feac6270` | `c1c571c0a9ade5be` | `428ef17df51250d8` |

The digests are those of the campaign's irqres-a1 arms. The T arm and its
control give the same three digests per fixture (runs p01–p08).
So the branch reproduces irqres-a1 T bit for bit from a cold prefill.

The same window ran the `DS4_V41_S1TG` arms. Their result is in the
single-group kernels section below.

The depth rows of the README (64K, 131K, 300K) are still W5 on the
state-depth tree. A composed depth run on this branch is pending.

## Candidate-block index scoring (9b9dfee)

*Score only the candidate blocks of later DeepSeek V4.1 index layers*:
Tarjei Mandt (kernelpool), adapted from 2d5ef0d3.

It is on by default. It runs on index layers 24 to 36 once a layer has more
than 16,384 compressed keys, so it never runs at the 8K anchor. Opt-out:
`DS4_METAL_DISABLE_V41_INDEX_CANDIDATES=1`.

Window W3 (resident, real data):

| Context | Exact | Decode speed |
| --- | --- | ---: |
| 64K | yes | +0.79% |
| 131K | yes | +0.91% |
| 300K | yes | +2.03% |

The commit message predates these numbers ("Not measured yet").

## Session create/free footprint (76a23b2)

*Give C callers an autorelease pool around the engine entry points*.

Window W3, repeated session create/sync/free cycles:

| | Footprint growth per cycle |
| --- | ---: |
| Before | +1.14 GiB |
| After | +0.001 GiB (flat) |

There is no change to the numbers the model produces: the pools only release
autoreleased objects earlier.

## Ivan Fioravanti's decode candidates (0f8972d, 3c298c0)

Window ivdec-a1 checked both candidates on top of the irqres T arm. Result:
VALID 12/12, every row bitwise equal to irqres-a1 T.

- **Shift decode of the Engram E4M3 codes (3c298c0, opt-in):** −26.7 µs/token
  on prose and −33.4 µs/token on code. On each fixture both S runs were below
  both control runs. It stays opt-in: the blended Engram reader (1e4efb2,
  default since f7ee976) carries the shift-decode idea and supersedes it on
  the decode path. Window attr-a1 also explains the small in-stack gain: the
  shift decode cuts the Engram kernel by −164.9 µs/token, but its neighbours
  lose most of that (wall +42.5 µs/token under the encoder timeline).
- **Aligned tensor copies in the open encoder (0f8972d):** −5.4 / −37.3
  µs/token. This is inconclusive (one code run drives it). The commit is kept
  because it is correctness-neutral, with no claimed gain.

## Narrow decode copies built at load (4230d9d)

A CPU re-encode check (no model) builds the router, Engram and indexer copies
the way the loader does and compares them with the source encoding. Result: 6
of 6 passes, 0 mismatches.

Re-encoding can recover different bytes for small codes. Where it does,
`table_ok` is cleared and that table falls back to the shift decode.

## Heads transformed once (5c3c02f)

S40-HEADS (D079, D080, D087, D093, D094):

- Segment: −0.125 ms/token (512/512).
- Resident window 28r: −0.079 / −0.132 ms/token (prose / code), output
  identical in 4 of 4 runs.

CPU proof:

- bf16 rounding is idempotent over all 2^32 patterns.
- 0 non-NaN differences over 87,031,808 words of real reduce output.

## Blended narrow decode readers (1e4efb2, f7ee976, 0f1095d)

The blend kernels read the Engram projections, the router and the
index-source indexer q_b from the narrow copies. The shift decode of the
E4M3 codes is co-credited to Ivan Fioravanti. They are the default since
f7ee976; `DS4_METAL_DISABLE_V41_BLEND=1` opts out.

Window blend-a2 (8K, order C T T C, VALID 8/8; every measured row bitwise
equal to the irqres T arm, and T equal to C):

| Fixture | C ms/token | T ms/token | T − C | T tok/s |
| --- | ---: | ---: | ---: | ---: |
| Prose | 24.898785 | 24.677890 | −220.9 µs | 40.522 |
| Code | 24.870855 | 24.681204 | −189.7 µs | 40.517 |

Both T runs were below both C runs on both fixtures, by at least 175 µs.

Window attr-a1 attributed the gain with the encoder timeline. It used 12
fresh prose processes, mirrored C T E R X S S X R E T C, all with the same
logits, prefix and state digests. The timeline serializes encoders, so the
absolute speed reads about 38.0 tok/s; the deltas are in µs/token against C:

| Arm | Wall | Kernel time |
| --- | ---: | --- |
| T (all three) | −116.5 | Engram −178.3, router −93.6, indexer −38.8 |
| E (Engram alone) | +31.7 | Engram −176.6; router matvec +27.2, other +41.6 |
| R (router alone) | −72.8 | router −119 |
| X (indexer alone) | −18.2 | indexer −38.6 |

The Engram saving alone is eaten by its neighbours, so the three readers
ship as one unit. That the router slowdown comes from SLC eviction is
inferred, not measured.

## Narrow copies built at load: decode and startup (cddebae, P17/P18)

Window fp8pf-a4 first showed a 16-token decode average of 31.48 tok/s with
the builder on against 37.66 off. A per-token split traced that to
the first token: 127.7 ms with the builder on against 34.5 ms off. Over
tokens 2 to 16 the builder is faster: 39.46 against 38.46 tok/s (25.34
against 26.00 ms/token).

The build takes 102.9 ms for 490.3 MiB (router BF16, Engram E4M3, indexer
q_b E4M3), at session creation before prefill. Peak RSS is 1.5 GB higher
with the builder on.

cddebae removes the first-token cost. It adds the built buffers to the
residency set right after the build and compiles their readers' pipelines
with one warm-up dispatch each. Its cold-start A/B (startup, prefill, first
token, steady decode and memory) is pending and not yet
recorded here.

## Shared FP8 prefill and public-derived values (P11, P17)

These measurements describe the narrow copies themselves; the shared
prefill path is not ported to this branch.

- **Shared FP8 prefill** (Engram and indexer prefill read the decode's narrow
  copies, router BF16): bit-exact to the control in frontier logits, final
  logits and teacher-forced NLL. The counters show zero wide GGUF reads
  through prefill and decode at 512, 8K and 64K. Prefill costs +0.59% at 8K
  (717.3 against 721.5 tok/s) and +0.20% at 64K. Decode is unchanged.
- **Public-derived values** (the at-load builder's re-encode of the public
  F16 tensors) against the source sidecars, window fp8pf-a3, 8K, 512
  tokens: identical logits and state digests on both fixtures, with no speed
  difference.
- At the Engram output itself the two are not value-identical (window
  fp8pf-a4: 92 of 409,600 BF16 values differ, mostly by 1 ulp and at most
  by 11, almost all in layer 1). The differences are absorbed before the
  logits on these fixtures. This is evidence on the pinned fixtures, not a
  proof for every input.

## Top-k winners' sort and candidate list once (0761f41, e7ab260, 82923a2)

The state-depth windows used greedy 128 on saved snapshots, mirrored
arms, and the same tokens and state hash in every run.

Window W4 (2 runs per arm; tok/s, deltas against A = candidate scoring):

| Depth | A | S (sort512) | O (list once) |
| --- | ---: | ---: | ---: |
| 64K | 39.09 | 39.60 (+1.30%) | 39.43 (+0.86%) |
| 131K | 38.43 | 38.91 (+1.24%) | 38.84 (+1.06%) |
| 300K | 37.44 | 37.92 (+1.30%) | 37.72 (+0.77%) |

At 131K, sort512 cuts selection from 877 to 551 µs/token and the list-once
path cuts candidate work from 131 to 73. Widening the radix select per
threadgroup (T4/T8/T16) lost −0.02 to −1.16% and was removed.

Window W5 (4 runs per arm; tok/s):

| Depth | A (sort512 off) | S | S + O |
| --- | ---: | ---: | ---: |
| 64K | 39.148 | 39.648 (+1.28%) | 39.702 (+1.42%) |
| 131K | 38.515 | 39.018 (+1.31%) | 39.073 (+1.45%) |
| 300K | 37.481 | 37.936 (+1.22%) | 38.021 (+1.44%) |

The list-once increment over S is +0.14 / +0.14 / +0.22% by the eval-interval
metric. By the more conservative wall-boundary metric, A to S is −0.30 to
−0.33 ms/token. Both become the default; `DS4_METAL_DISABLE_V41_TOPK_SORT512=1`
and `DS4_METAL_DISABLE_V41_INDEX_CAND_ONCE=1` opt out. Against W2's
pre-candidate depth curve (38.60 / 37.95 / 36.53) the S + O stack is +2.8% /
+3.0% / +4.1%.

## Rejected designs

These were measured and are not in the branch:

- **Guarded MMA index scorer (W5):** exact, with the fast path on 99.93 to
  99.98% of keys and no guard fallbacks. But index scoring rose from 1638 to
  1752 µs/token at 300K; S + O + MMA was −0.35 to −0.40% against S. The W4
  guard capture (100% of keys inside the exactness bound at 64K to 300K) is
  kept as evidence for a future key-cache layout.
- **Router selection fused into the matvec (fusion micro):** exact only with
  a relaxed device-atomic handoff. Plain stores plus a fence left about 26%
  stale probabilities and experts. Fused it was 16.85 µs per layer against
  14.48 separate, about +95 µs/token.
- **EADD fused into the Engram tail (fusion micro):** exact, but 236.3 µs
  (value rows first) or 271.7 µs (in order) against 227.8 separate.

The micros ran with production up, so their absolute times carry
contention; the A/B order repeated.

## Single-group latency kernels (f45f4e8, 77eea27, 913aa61; opt-in, experimental)

`DS4_V41_S1TG`:

- `cn`: the split HC collapse and the hoisted paired norm. In a model-free
  micro at the decode shapes it is bit-identical: 0 of 16.4M collapse values
  and 0 of 1.4M paired-norm values differ. The micro saves −125 to
  −200 µs/token on the collapse.
- `k` and `r`: the fused top-k merge ladder and the ballot router.
- `h`: eight heads per threadgroup in single-row decode attention.

`k`, `r` and `h` are also bit-identical in their micros.

On the stack, `cn` is **not exact**. In window fp8pf-a5 (d01) its arm's
digests differed from the control:

- The prefix digests are equal.
- The logits and state digests differ. On prose they are `2bbe429e4417fcb6` /
  `4feb7562d5a596e7`, against `ed3ca952608e6274` / `6d2d529ba8c0e5c0`. On
  code they are `afdf72f24377f997` / `e3de3dfb45ec2ab7`, against
  `c1c571c0a9ade5be` / `428ef17df51250d8`.
- `cn` and `cnkr` give the same digests as each other in both repeats. The cause is under
investigation. Until it is found and fixed, every letter (`cnkrh`) stays opt-in and
experimental, and none of them enters a measured basis.

## Clean-checkout check (25b45ea)

A fresh `git clone --branch pub/v41-1017-update` with nothing else in the
directory, on the M3 Ultra (macOS, Xcode command-line tools):

- `make` (docs/METAL.md) builds ds4, ds4-server, ds4-bench, ds4-agent and
  ds4-eval with no warnings.
- Every prerequisite of `make test` builds, as does
  `tests/test_deepseek41_metal`.
- The Metal library assembled from the clone's `metal/` (26 files) compiles to
  548 functions. The selector library (`metal/sampler_selector.metal`)
  compiles with its three kernels. Both resolve from the checkout root; see
  [RUNTIME-V41.md](RUNTIME-V41.md) section 4.
- The model-free part of `make test` passes, run from the clone root: ds4-eval
  case validation and extractor self-tests, `ds4_agent_test`,
  `ds4_test --server`, test_layer_pack, test_engine_mgpu_placement, test_gpu_args
  and its CLI script (90/90), test_prompt_prefix, test_sampling,
  test_deepseek4_vision_image, test_engram, test_linux_memory, the TP command and
  TCP tests, the web-recovery test, and the Q4_K and MXFP4 dot tests.
- Not run here: `./ds4_test` without arguments (model-backed and Metal tests)
  and `tests/test_session_state` (Metal). They need a model or a GPU window, as
  [TESTING.md](TESTING.md) says.

## GPU tests

These tests are built with each commit and run by the coordinator in GPU
windows:

| Commit | `tests/test_deepseek41_metal` flag |
| --- | --- |
| 9b9dfee | `--index-candidates` |
| 3c298c0, 4230d9d | `--engram-decode` |
| 21c0519 | `--decode-topk` |
| dbc6c18, a05c8be | `--rope-tab` |
| 5c3c02f | `--heads-once` |
| 1e4efb2, f7ee976, 0f1095d | `--blend` |
| 0761f41 | `--topk-select` |
| e7ab260, 82923a2 | `--index-candidates` |
| 80c86de | no flag (`check_engram` in the full run) |
| 9ab435c | `--rope-tab` |
