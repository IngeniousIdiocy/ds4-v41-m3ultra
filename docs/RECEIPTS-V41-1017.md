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
- **P numbers** (P11 to P29) are the results recorded for those windows.
  **Ledger ids** (`S40-*`, `D0xx`) are entries in the campaign's decision
  record. That record is not published.

Unless stated otherwise: one M3 Ultra (512 GB), the 1017 MXFP4 target
weights, decode at concurrency 1, and the campaign's prose and code fixtures.

## Composed run of this branch at 8K (fp8pf-a5 d01, P25)

Setup:

- The tree: a clean export of 77eea27c, the default build. d01 ran the
  window's builder-on arm, which sets a `DS4_*` variable. Only b13, the
  window's plain launch, ran with no `DS4_*` variable. It was bit-identical to
  the builder-on arm (P26).
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

Caveats:

- The timing is provisional: 24.716182 / 24.714071 ms/token (prose / code),
  leaving 0.283818 / 0.285929 ms to the 25 ms (40 tok/s) line. The run-to-run
  ranges were 0.027327 / 0.054748 ms. Two harness dry runs started during d01.
  The contention check aborted both at once, and no bits were affected, but
  they may have added CPU noise. A clean repeat is planned.
- The window did not pass as a whole. In the same d01 process, the optional
  `DS4_V41_S1TG` arm (`cn`) failed the exactness gate, which stopped the
  window. Rows d02 and d03 (the sidecar and builder-off decode arms) and a01
  did not run. The window's verdict was INVALID, with its rows kept for
  adjudication and no speed verdict. P25 accepted the d01 default-arm rows on
  their digests.

The same window ran the `DS4_V41_S1TG` arms. Their result is in the
single-group kernels section below.

The depth rows of the README (64K, 131K, 300K) are still W5 on the
state-depth tree. A composed depth run on this branch is pending.

## Serving comparison with DwarfStar main (W8, P29)

Arms:

- U: antirez/ds4 main `5379c1c9ac315f1540371b06a3db88491d4059ff`, server
  binary sha256 `887992a8…`.
- P: this branch at `0a72af29a080a136d12fe816d2da4f0c64582838`, server
  binary sha256 `d808adbf…`, with `--serial-reasoning`.

Setup:

- Same MXFP4 GGUF and the same flags on both arms: `-c 393216
  --prefill-chunk 8192 --kv-disk-space-mb 98304 --kv-cache-cold-max-tokens
  393216 --kv-cache-continued-interval-tokens 4096`, with a fresh kv
  directory for each lifetime.
- No DSpark on either arm. DwarfStar main refuses this branch's DSpark GGUF,
  which lacks `dspark.target_architecture`.
- Cycles ran in the order U P P U U P, with three runs per arm and cell.
- Every cell is VALID.

Medians [ranges] over the three runs.

Decode, server log, 512-token answers:

| Context | U tok/s | P tok/s | P/U |
| ---: | ---: | ---: | ---: |
| 2K | 19.56 [19.41–19.58] | 41.32 [41.31–41.37] | 2.112 |
| 8K (8,215 tokens) | 19.84 [19.48–19.87] | 40.78 [40.73–40.78] | 2.055 |
| 32K | 19.14 [19.11–19.28] | 40.43 [40.43–40.46] | 2.112 |
| 64K (66,752 tokens) | 18.64 [18.58–18.68] | 40.21 [40.20–40.21] | 2.157 |

Cold TTFT, thinking off:

| Context | U s | P s | P/U |
| ---: | ---: | ---: | ---: |
| 2K | 6.275 [6.244–6.319] | 5.110 [5.100–5.112] | 0.814 |
| 8K | 23.413 [23.403–23.479] | 15.300 [15.282–15.302] | 0.653 |
| 32K | 49.030 [48.973–49.224] | 41.576 [41.511–41.593] | 0.848 |
| 64K | 95.724 [95.686–95.964] | 79.477 [79.252–79.483] | 0.830 |

Other cells:

- **Reasoning decode** (effort high, max_tokens 1024, server log):
  - U: 19.84 / 19.65 / 18.59 tok/s at 2K / 8K / 64K;
  - P: 41.41 / 40.77 / 40.22 tok/s;
  - P/U ×2.07 to ×2.16.
- **Warm agent turn:** both arms resume from a cached start of 29,748
  tokens and append 437. TTFT is 2.389 [2.382–2.401] s on U and 2.079
  [2.058–2.092] s on P.
- **Restart and restore:** one run per arm. The server gets SIGTERM, is
  relaunched on the same kv directory, and a follow-up turn is resent. These
  are follow-up turns, not fresh-versus-restored quality checks.

  | Context | U TTFT | P TTFT | U load | P load |
  | ---: | ---: | ---: | ---: | ---: |
  | 8K | 2.484 s | 2.485 s | 10.7 ms | 23.4 ms |
  | 64K | 2.897 s | 2.518 s | 54.6 ms | 172.3 ms |
  | 128K | 3.546 s | 3.131 s | 80.6 ms | 303.9 ms |

  The restored prefixes are 8,201 / 67,635 / 131,172 tokens. P's longer
  file load is the v2 payload digest.
- **Startup restore window:** P's resend at 8K sent before cache-on (M4R2now)
  was refused ("the numerics canary has not completed yet") and prefilled
  cold, in 14.580 s. P's cache came on 14.35 to 29.06 s after launch, with
  canary `e8c0517e0b2d6638`. fbd9ff30 replaces that policy. Requests now
  queue behind a one-shot canary; that change is not measured yet.
- **Load** (launch to ready, page cache warm): U 2.30 [2.28–2.31] s, P 2.29
  [2.28–2.30] s.
- **Memory:**
  - Planned memory is 290.12 GiB on both arms.
  - The 64K wired peak is 295.4 to 296.3 GiB on U and 296.6 to 297.2 GiB on
    P.
  - Idle RSS is 1.66 GB on U and 3.19 GB on P (the at-load builder, P26).

Answer lengths:

- Every 512-token decode answer ran to its limit, except P's at 2K, which
  stopped at 494 tokens in all three runs.
- Reasoning answers had medians of 695 completion tokens on U and 958 on P,
  with different samples. That difference is not treated as a finding.

Corrections to the window's first summary (relabels only; the raw data are
unchanged):

- `stop_s` is the server's lifetime from launch, not its shutdown time.
  The exit instant is not logged, so shutdown is bounded instead. The
  shutdown checkpoint store took 31 to 291 ms, and the upper bounds are 1.30
  to 1.71 s.
- The window's launcher is a Perl shim (`/usr/bin/perl`) that execs the
  driver, which spawns the server. The server identity is the hash of
  `ds4-server` above, not the launcher's.
- The contention warnings name the window's own registered server children.
  They are ownership false positives, not outside contention.
- The client-side decode figure is an SSE throughput estimate. The tables
  above use the server log.
- Prompt tokens / TTFT is a user-visible rate, not the prefill kernel's rate.

Scope: this result does not isolate individual optimizations or establish
quality equivalence. It does not replace the composed 131K/300K checks or
the pinned-profile checks.

## Disk KV cache v2 and saved checkpoints (W6, W7; P24, P27)

KV v2 entries bind to the engine identity, a numerics canary and a payload
hash (d49d1158).

Window W6 (P24) ran the state-depth tree's server with the production
flags. Each run was greedy 64 tokens, then SIGTERM, a restart and a resend.
Every restored answer's text and token count equalled the cold one:

| Depth | Cold request | Restored request | Load | Largest entry |
| ---: | ---: | ---: | ---: | ---: |
| 8K (8,238) | 13.9 s | 2.55 s | 49 ms | 64 MiB |
| 64K (67,310) | 87.2 s | 8.09 s | 190 ms | 423 MiB |
| 300K (303,908) | 461.2 s | 8.69 s | 825 ms | 1,869 MiB |

- An entry was refused, with nothing restored, when its payload byte was
  flipped, its canary bytes were changed, or its identity bytes were changed.
- Its one-shot canary cost 7.28 s at startup.
- Those entries carry that tree's identity (`028036d5c520a07d`) and canary
  (`772938c1f52ab041`), so this branch, whose identity differs, does not
  index them.

Window W7 (P27) ran on this branch at 2357478c, with the stepped canary
that ran between requests (04d4db75):

- **Server:** the canary was `e8c0517e0b2d6638` in all six lifetimes, idle
  and under load. Under a continuous request stream the cache came on 42 to
  51 s after start; idle, after 12 s. No entry was stored or loaded early.
  A request sent while the canary was pending (8,238 tokens) was prefilled
  again rather than restored. That is the gap fbd9ff30 closes: requests now
  queue behind a one-shot canary, which is still to be measured. Entries
  under `e8c0517e` are refused by the new canary and evicted normally.
- **ds4-agent:** store, hit, then a payload flip and an identity flip. Each
  was refused and re-saved, and the identity refusal was logged. ds4-agent
  binds no numerics canary yet.
- **ds4-bench snapshot format 4:** a c8k save carries its canary, and the
  c16k lineage names c8k's hash at depth 8,192. Five damaged copies were each
  rejected with the right verdict: canary, prefill representation, short
  payload, missing end marker, payload hash. Runs were equal across save,
  load, extension and a new process.

These are checkpoint checks, not whole-session qualification.

## Public 1017 weights: download and header check

- Nine GGUF shards, `DeepSeek-V4.1-Flash-1017-MXFP4-experts-0000N-of-00009`,
  501,602,535,360 bytes in all. Every size and SHA-256 matches the pinned
  manifest (`425d8d0a…`).
- The download log (sha256 `5a106dde…`) ends `ALL VERIFIED`. Six transfer
  attempts failed and were retried. Each shard was verified after its final
  attempt.
- Header comparison with the local GGUF: all 1,046 tensor names, shapes,
  types and payload lengths match. Metadata differs in calibration and
  imatrix provenance, alignment, source URL and split keys.
- This proves no payload equality. A streamed per-tensor comparison is
  designed but not yet run.


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
with one warm-up dispatch each.

Window fp8pf-a5's builder rows (P26) measured the at-load builder (BON)
against the original-checkpoint sidecars (SC) and against neither (BOFF,
wide reads). Cold 8K prompt, 512 tokens; all thirteen rows are VALID.

| Arm | Steady tok/s (runs) | Mean | 4K / 8K, two frontiers |
| --- | --- | ---: | --- |
| BON | 40.88, 38.78 | 39.83 | 41.62 / 40.85 |
| SC | 40.78, 40.70 | 40.74 | 41.35 / 40.73 |
| BOFF | 39.37, 39.40 | 39.39 | 39.91 / 39.43 |

- BON's 38.78 (b06) is a single outlier that breached no gate. The repeat is
  still to be checked. Built or sidecar, the narrow readers are about +1.4
  tok/s over the wide reads.
- **Plain launch** (b13, no `DS4_*` variable): 40.79 tok/s, bit-identical to
  BON (b01).
- **Cold first token** (exec to prefill start + prefill + token 1):
  - BON 2.39 + 11.04 + 0.036 = 13.46 s;
  - SC 9.01 + 11.02 + 0.036 = 20.07 s (mapping the sidecars costs 6.6 s);
  - BOFF 13.39 s.
- The build takes 105 ms (490.3 MiB), and the readers warm in 7 ms.
- **Peak RSS:** BON 3.48 GB, SC 2.30 GB, BOFF 1.94 GB. BON's extra 1.2 GB
  over SC is the built copies while the wide views stay mapped. Unmapping
  the wide views is designed, not done.
- The at-load builder stays the default. Sidecars remain an option
  (`--v41-sidecar`).
- Caveat: BON's single slow run (b06) is unexplained, and the BON mean
  carries it.

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

- `c`: the split HC collapse.
- `k` and `r`: the fused top-k merge ladder and the ballot router.
- `h`: eight heads per threadgroup in single-row decode attention.

Model-free exactness, counted over distinct inputs. An earlier version of
these receipts counted repeats of the same inputs: they show determinism, not
coverage.

- `c`: 0 of 20.48M distinct collapsed and 0 of 20.48M distinct normed values
  differ, on fresh data for every call, with and without the engine's macros.
  An engine-linked check adds 0 of 819,200. The earlier "0 of 16.4M" was 20
  repeats of 819,200 values.
- `k`: 140 crafted rows (seven widths; Gaussian, many-tie, `-inf`-tail, NaN
  and all-equal rows), plus 71,680 distinct selected indices per fused form.
  The merge only compares and moves pairs, with no float arithmetic.
- `r`: 15,840 distinct values (the printed 316,800 is 20 repeats). That is
  small coverage for its float part.
- `h`: 6,316,032 distinct partial values per key count (the printed 18.9M is
  3 repeats), with one key set per shape.

None of the four gets speed credit. Their micro timings are not added to any
figure on this page or in the README.

On the stack, `cn` is **not exact**. In window fp8pf-a5 (d01) its arm's
digests differed from the control:

- The prefix digests are equal.
- The logits and state digests differ. On prose they are `2bbe429e4417fcb6` /
  `4feb7562d5a596e7`, against `ed3ca952608e6274` / `6d2d529ba8c0e5c0`. On
  code they are `afdf72f24377f997` / `e3de3dfb45ec2ab7`, against
  `c1c571c0a9ade5be` / `428ef17df51250d8`.
- `cn` and `cnkr` give the same digests as each other in both repeats.

The cause is letter `n`, the hoisted paired norm, and it is removed from the
branch (P28). Its micro had reported "0 of 1.4M", but that was 20 repeats of
the same 71,680 values. Over 71.68M distinct values, 302 differ. Fast math
reassociates its `(x * scale) * w` into `(x * w) * scale` once the weight
loads move ahead of the reduction. No exact form of the hoist gained anything. `c`, `k`, `r` and
`h` stay opt-in and experimental until their per-letter stack results are in.
None of them enters a measured basis.

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
