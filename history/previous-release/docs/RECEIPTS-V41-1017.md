# V4.1 1017 decode update: receipts

The measurements and exactness checks behind the 1017 decode commits on this
branch. Each commit message carries its own what/why and ledger entry; this
page collects the results that arrived after a commit landed, or that span
several commits.

Glossary for the labels in this page and in the commit messages:

- **Window labels** such as `fp8pf-a5`, `irqres-a1`, `w1-a2`, `W5` or `W72`:
  measurement windows, each a protected run on the M3 Ultra. This page
  summarises their evidence.
- **P numbers** (P11 to P29) are the results recorded for those windows.
  **Ledger ids** (`S40-*`, `D0xx`) are entries in the campaign's decision
  record. That record is not published.

Unless stated otherwise: one M3 Ultra (512 GB), the 1017 MXFP4 target
weights, decode at concurrency 1, and the campaign's prose and code fixtures.

The October release is e8d23bb6. Its numbers come from four windows,
summarised first: r1bench-a3 (decode at every context), rcv-a1 (the release
itself: decode, the DwarfStar comparison and restart), codeds-a1 (DSpark on
code) and r1cc-a5 (the README's steps on a clean clone). The sections after
them are the earlier evidence for individual commits.

## README numbers: one-method comparison against DwarfStar (w192-onemethod)

Window w192-onemethod, Oct 20 2026, 22:19–23:03 EDT. It replaces the README's mixed
sources (server single runs, resident-benchmark medians, and a prefill row from two
windows) with one method. Every cell is the median of three runs.

Engines:

- U: DwarfStar, antirez/ds4 main at 5379c1c9, the tree rcv-a1 used
  `ds4-server` sha256 `3bc56f9b…`.
- P: this branch, 9e6012b3. Its `ds4-server` was built
  from 8664da6d, which differs from 9e6012b3 only in README.md. sha256 `f1b8e62a…`.
- Each binary ran with its own tree as the working directory.

Model: both engines loaded the GGUF that the README's Prepare step wrote (window
prep-a1).
That is the file rcv-a1 gave U. DSpark was off for both.

Server launch: the README's Run line without `--mtp-model`, `--dspark` and
`--mtp-exact-sampling`, on port 8297, with this window's cache directory (one per
engine). U does not have `--default-top-p`, `--default-min-p`, `--utility-ctx` or
`--serial-reasoning`, so its line drops those four, as in rcv-a1. Every request is
temperature 0 with thinking off, so none of the four changes the work.

Prompts: every prompt is a prefix of one document, *I promessi sposi*
(`promessi_sposi.txt`, 419,509 tokens). Each one starts with "Summarize the following
text in a few paragraphs." and then holds the first 8K, 64K, 131K or 300K tokens of the
book: 8,242, 67,314, 132,966 and 303,912 prompt tokens.

**Seeding** (not reported). Each engine filled its cache once, in its own lifetime:

- DwarfStar stores a checkpoint only for a prompt it prefilled from zero. U therefore
  prefilled each depth cold, longest first: 300K, 131K, 64K, 8K.
- P walked 8K → 64K → 131K → 300K. Each step restored the previous checkpoint and stored
  its own depth ("kv cache extends restored prefix").
- Both engines ended with checkpoints at 8,192, 65,536, 131,072 and 303,104 tokens.

**Measurement.** Six server lifetimes in the order U P U P U P. Each lifetime ran:

1. One untimed warm-up: a 2K prose request, 16 tokens.
2. The cold prefills, 8K then 64K. Each opens with a tag line naming its lifetime, so
   nothing is restored (cached 0 in every run).
3. The decode runs at 8K, 64K, 131K and 300K, in that order: 256 tokens each. Every run
   restored its depth's checkpoint from disk (cached 8,192 / 65,536 / 131,072 / 303,104).

Rates:

- Decode rate is the client's: tokens 2..256 over their streaming wall time. The
  server's own average agrees to within 0.05 tok/s.
- Prefill rate is prompt tokens divided by the client's time to first token. That time
  includes, for both engines, the cold-checkpoint store and, before 64K, storing the
  previous 8K session (61 MiB).

**Decode, 256 tokens after the prompt** (tok/s, median of three; min–max):

| Context | DwarfStar main | this branch |
| ---: | ---: | ---: |
| 8K | 20.00 (19.98–21.20) | **41.51** (41.46–41.61) |
| 64K | 19.07 (19.03–20.17) | **40.90** (40.89–41.01) |
| 131K | 18.14 (18.10–19.10) | **40.63** (40.60–40.71) |
| 300K | 16.21 (16.18–17.04) | **40.03** (40.01–40.09) |

DwarfStar's first measurement lifetime decoded about 5% faster than its other two at
every depth. The other two agree within 0.05 tok/s. The median is from the agreeing
pair, and the maximum is the first lifetime.

**Cold prefill** (prompt tokens / time to first token, median of three; min–max):

| Prompt | DwarfStar main | this branch |
| ---: | ---: | ---: |
| 8K (8,251 tokens) | 13.51 s, 611 t/s (607–611) | **11.70 s, 705 t/s** (704–706) |
| 64K (67,323 tokens) | 96.11 s, 700 t/s (691–701) | **80.59 s, 835 t/s** (832–836) |

Not measured in this window: 2K and 32K decode, and DSpark (the window had no room for
them in its 45-minute budget).

Memory guard (memwatch v3, 64/16 GiB) never tripped. Jetsam events before and after: 1
and 1. Raw rows: `rows.jsonl`. Server logs: `life1-server.log` … `life8-server.log`.
Driver: `tools/w192_driver.py`.

## DSpark on the same method (w193-dspark)

Window w193-dspark, Oct 21 2026, 05:27–05:45 EDT. It puts the code-answer and DSpark rows on
the method of w192-onemethod: same harness, model file, port and server lines.

Engines:

- U: DwarfStar, antirez/ds4 main at 5379c1c9, `ds4-server` sha256 `3bc56f9b…`.
- P: this branch at 9e6012b3, `ds4-server` sha256 `f1b8e62a…`.

Model: both loaded the GGUF and drafter that the README's Prepare step wrote (window
prep-a1). DwarfStar's DSpark arm loaded a copy of that drafter with only its two
Markov-head tensors renamed (`markov_w1` / `markov_w2`). It cannot load the drafter
under the checkpoint's names.

Server launch:

- P-on is the README's Run line exactly (`--mtp-model … --dspark --mtp-exact-sampling`),
  on a test port with a per-window cache directory.
- P-off is that line without the three DSpark flags.
- U-off and U-on drop the four flags DwarfStar does not have (`--default-top-p`,
  `--default-min-p`, `--utility-ctx`, `--serial-reasoning`). U-on adds the same three
  DSpark flags with the renamed drafter.
- The DSpark arms ran with `DS4_DSPARK_STATS=1`, which only counts, so the accept rate
  could be read at shutdown.

Lifetimes, in this order: U-off, P-off, U-on, P-on code, P-on prose. Each began with an
untimed 2K prose warm-up (16 tokens).

- **Code answers:** in each arm's lifetime, the three code prompts (an LRU cache in C,
  an asyncio token-bucket rate limiter in Python, a trie in Rust) ran three times,
  round-robin. Each run was 512 tokens, temperature 0, thinking off.
- **Prose cost:** P-off, and a second P-on lifetime, decoded the w192 prose prefixes at
  8K and 64K (8,242 and 67,314 prompt tokens) three times each, 256 tokens. Every run
  restored 8,192 or 65,536 tokens from the disk cache.
- **Why P-on prose had its own lifetime:** the DSpark counters print once, at shutdown,
  so this kept the code lifetime's accept rate free of prose.

The rate is the client's: tokens 2..N over their streaming wall time, as in w192. The
server's own average agrees to within 0.2 tok/s.

**Code answers, 512 tokens** (tok/s, median of three; min–max):

| Prompt | DwarfStar | DwarfStar + DSpark | this branch | this branch + DSpark |
| --- | ---: | ---: | ---: | ---: |
| LRU cache, C | 20.03 (19.99–20.11) | 30.77 (30.74–30.79) | 41.86 (41.86–41.86) | **50.41** (50.40–50.43) |
| rate limiter, Python | 20.04 (19.97–20.41) | 31.35 (31.35–31.41) | 41.86 (41.84–41.87) | **49.38** (49.34–49.44) |
| trie, Rust | 20.07 (19.99–20.26) | 28.95 (28.93–28.99) | 41.84 (41.72–41.86) | **47.09** (46.92–53.73) |

DSpark accept rate (drafted tokens accepted, per lifetime, all nine code runs):

- DwarfStar: 86.4% (3,527 of 4,082).
- This branch: 90.9% (1,760 of 1,936).

The counters cover a whole lifetime. Neither engine reports them per request, and
DwarfStar has no per-cycle log, so a per-run rate would have needed a lifetime per run.

Answers at temperature 0 were the same text in all three runs of every cell but one. In
the trie prompt with DSpark on, this branch's third run gave a different answer: the
same text as its serial runs, at 53.73 tok/s. Its first two runs agreed, at 47.09 and
46.92. DSpark's batched verify is not bitwise equal to one-token decode, which the server
states at startup, so the answer can change.

**Prose, 256 tokens after a restored prompt** (this branch, tok/s, median of three;
min–max):

| Context | without DSpark | with DSpark |
| ---: | ---: | ---: |
| 8K | 41.31 (41.30–41.32) | 39.57 (39.52–39.57) |
| 64K | 40.75 (40.74–40.75) | 39.30 (39.29–39.31) |

On prose the admission controller drafted rarely (45 tokens proposed over 1,546 cycles,
6 accepted). DSpark costs 4.2% at 8K and 3.6% at 64K.

The DSpark-on server did not accept w192's cache entries for the prose prompts. Its
cache identity includes the drafter: `19b0e9b3…` with DSpark against `b0998615…`
without, same canary. The P-on prose lifetime therefore restored copies of the DSpark-on
checkpoints of the same two prompts, stored in window rcv-a1 under the same identity and
canary. Its 64K answers have the same digest as rcv-a1's (`fa6a74e0…`).

Memory guard never tripped. Raw rows: `rows.jsonl`. Per-lifetime counters: `lives.json`.
Server logs: `life1-server.log` … `life5-server.log`. Driver: `tools/w193_driver.py`.

## Exactness of the default build

With the exact Engram gate (3ea32856, in its register form since 0004c5cf),
the default build makes no numerical change to the engine's output. Every
change reproduces its control's bits. The evidence:

- **Kernel tests:** bit-for-bit comparisons over fresh random inputs at the
  production shapes, in `tests/test_deepseek41_metal` (see GPU tests below).
  The Engram gate's test, `--engram-add-exact 750`, found 0 of 3,072,000
  streams and 0 of 15,728,640,000 values differing.
- **End-to-end digests:** decoded tokens, frontier prefix, final logits and
  session state, equal to the previous October build at 8K to 300K, greedy
  and sampled (r1bench-a3 below).
- **Narrow copies:** lossless by construction. `ds41_fp8_copy_build` keeps
  an E4M3 copy only if every block is exact, and the BF16 copies are kept
  only if every value is exact BF16. Otherwise the wide path is used.

Limits:

- End-to-end identity is shown on the pinned fixtures. Beyond them it rests
  on the kernel tests.
- The statement covers decode without drafting. DSpark's batched verify is
  not bitwise equal to one-token decode, by the engine's design.

## Release decode numbers (r1bench-a3)

Setup:

- Control: ece7746a, the previous October build. Candidate: d35a9315. The
  release adds two commits on top of d35a9315, a server change (08d79dd0)
  and a CPU-build link fix (e8d23bb6); rcv-a1 B below reruns the candidate's
  greedy cells on the release.
- `ds4-bench`, one engine per process. Every context was prefilled once in
  window W10-a3 and saved; each process loads the same saves (8K prose, 8K
  code, and prose at 65,536, 131,072 and 300,000 tokens). The loaded frontier
  logits were identical in every process.
- G is greedy, 128 tokens, seed 1. S is sampled, 512 tokens, seed 1017, temp
  1, top_p 0.95.
- Four processes, mirrored: control, candidate, candidate, control. Each
  cell has three runs per arm; the tables give the median.

Every cell was IDENTICAL: the control and the candidate gave the same token
ids, logits and state digests in every run.

| Cell | Control ms/token | Candidate ms/token | Candidate tok/s | Δ ms |
| --- | ---: | ---: | ---: | ---: |
| G prose 8K | 24.630 | 24.472 | 40.862 | −0.158 |
| S prose 8K | 24.615 | 24.454 | 40.892 | −0.160 |
| G code 8K | 24.752 | 24.442 | 40.914 | −0.310 |
| S code 8K | 24.853 | 24.548 | 40.736 | −0.305 |
| G prose 64K | 25.182 | 24.718 | 40.457 | −0.464 |
| S prose 64K | 25.157 | 24.765 | 40.379 | −0.392 |
| G prose 131K | 25.493 | 24.953 | 40.075 | −0.541 |
| S prose 131K | 25.474 | 24.940 | 40.096 | −0.534 |
| G prose 300K | 26.242 | 25.335 | 39.471 | −0.906 |
| S prose 300K | 26.152 | 25.294 | 39.535 | −0.858 |

Depth slope, 64K to 300K (prose): greedy +1.060 ms on the control against
+0.618 on the candidate; sampled +0.995 against +0.529. The candidate's
processes never logged "not bf16-exact", so every index key took the BF16
view.

Opt-outs, each in its own process, against the control's numbers:

- po6 set the six switches new since the control
  (`DS4_METAL_DISABLE_V41_INDEX_BF16`, `_CAND_SPLIT`, `_CAND_SELECT_ONE`,
  `_TOPK_GATHER`, `_SCORE_SPLIT`, `_VERIFY_INDEX_BF16`). Its digests equal
  the control's at 8K and 300K, greedy and sampled, at the control's speed
  (+0.024 to +0.069 ms/token).
- po9 set those six plus `_SPLIT_COLLAPSE`, `_TOPK_FUSED_MERGE` and
  `_ENGRAM_ADD_TG`. Its digests also equal the control's, at +0.32 to +0.53
  ms/token.

Cold prefill at 64K (65,536 tokens) on the candidate, four processes: 890.2
and 890.5 tok/s by default, 845.0 and 845.0 with
`DS4_METAL_DISABLE_V41_INDEX_MMA=1` (×1.054).

`tests/test_deepseek41_graph` from the candidate tree: `MODEL --memory-plan`
and `--batch-admission` pass.

All processes exited 0 and were valid; teardown was proven.

## Release validation (rcv-a1)

Trees and binaries:

- P: the release, e8d23bb6, `ds4-server` sha256 `510c3c8e…`. The restart
  cells that force a canary failure used its test build
  (`ds4-server-testhooks`, `6600a9b4…`).
- U: DwarfStar, antirez/ds4 main at 5379c1c9, `ds4-server` sha256
  `3bc56f9b…`.
- The servers loaded the MXFP4 GGUF and drafter that the README's Prepare
  step wrote (window prep-a1). U's DSpark arm loaded a copy of that drafter
  with its two Markov-head tensors renamed to `markov_w1` / `markov_w2`;
  DwarfStar's loader reported `missing=0 invalid=0` for it. With the
  unrenamed drafter, DwarfStar refused to start ("invalid V4.1 DSpark
  weights", window r1srv-a4).
- Server flags: the README's Run launch, with DSpark on or off as named, on a
  test port and a per-window cache directory.

**B, the release's decode against d35a9315.** `ds4-bench`, greedy 128 tokens
from the W10-a3 saves, three runs. Every cell IDENTICAL to the r1bench-a3
candidate's runs (one token, logits and state digest per cell):

| Context | Release tok/s | d35a9315 tok/s (r1bench-a3) |
| ---: | ---: | ---: |
| 8K | 40.95 | 40.86 |
| 64K | 40.39 | 40.46 |
| 131K | 40.09 | 40.08 |
| 300K | 39.51 | 39.47 |

**Decode against DwarfStar.** One document's prefixes (2,054 / 8,252 /
33,575 / 67,324 tokens), thinking off, 256 tokens each, one request at a
time. Rates are the client's; the server log agrees to within 0.1 tok/s.

| Context | U | U + DSpark | P | P + DSpark |
| ---: | ---: | ---: | ---: | ---: |
| 2K | 19.68 | 19.90 | 41.45 | 39.99 |
| 8K | 19.54 | 20.75 | 41.23 | 39.94 |
| 32K | 19.23 | 19.97 | 40.91 | 39.74 |
| 64K | 18.78 | 19.75 | 40.74 | 39.56 |

- U, U + DSpark and P + DSpark restored each prompt from disk-cache entries
  saved in window r1srv-a4 (2,054 / 8,192 / 32,768 / 65,536 tokens cached).
  P without DSpark prefilled each prompt cold in this window, and its decode
  followed that prefill.
- One run per cell.

**Cold prefill against DwarfStar** (time to first token, thinking off):

| Context | U | P |
| ---: | ---: | ---: |
| 2K | 6.15 s | 4.90 s |
| 8K | 13.62 s | 11.64 s |
| 32K | 48.13 s | 41.45 s |
| 64K | 96.56 s | 80.73 s |

- P is rcv-a1's cold arm above, one run on the release.
- U is window r1srv-a4: the same build, model and prompts, three server
  lifetimes, median of six runs at 2K and 8K and of three at 32K and 64K.
  r1srv-a4 also ran d35a9315 the same way: 4.88 / 11.93 / 41.71 / 81.50 s.

**S, extending a restored prompt** (P with DSpark):

- Life 1: an 8K prompt (8,242 tokens) prefilled cold, 11.94 s, and stored
  its 8,192-token checkpoint. A 64K prompt (67,314 tokens) that extends it
  restored 8,192 tokens, prefilled the rest (72.17 s to first token), and
  logged "kv cache extends restored prefix cached=8192 store=65536
  prompt=67314". It stored a 65,536-token checkpoint.
- Life 2, after a restart: the same 64K prompt restored 65,536 of 67,314
  tokens, 5.61 s to first token, with the same output digest as life 1.
- A cold 8K prose request in life 2 gave the same digest as an earlier run of
  the release candidate.

**K, restart and the disk cache** (P with DSpark, nine server lifetimes on
one cache directory). Every cell passed:

| Cell | Result |
| --- | --- |
| k0 | Cold: a 303,912-token conversation, 388.49 s to first token; an 8K one, 16.39 s. Identity `19b0e9b36b25d909`, canary `772938c1f52ab041`; the canary took 7.08 s idle, and the cache came on 13.11 s after launch. |
| k1 | 8K resend right after a restart: 9.10 s to first token, 8,105 of 8,600 cached. A fresh 11-token prompt at the same point: 7.31 s. |
| k2 | 300K resend right after a restart: 303,944 of 304,463 cached, prompt done in 3.19 s, 11.04 s to first token. |
| k3 | 300K, 8K and an 11-token request sent together during the canary: answered in that order, at 10.88 / 14.53 / 15.28 s. |
| k4 | SIGTERM 2 s after listen, with the 300K request queued: exit 0 in 6.18 s, every cache entry kept (8 before and after), and the queued request answered 503. |
| k5 | Next restart: the 300K conversation restored 303,944 of 304,472 tokens, 10.82 s to first token. |
| k6 | Canary forced to fail: "KV disk cache stays off", no stores, entries unchanged; an 8K request prefilled cold, 19.97 s. |
| k7 | Relaunched with a 1,024 MiB budget over the cache plus 26 entries seeded under an earlier build's canary: 35 entries evicted as disk-cache-full, including all 26 seeded ones. |

The canary itself took 7.04 to 7.18 s in each lifetime of rcv-a1, codeds-a1
and r1cc-a5.

## DSpark on code (codeds-a1)

The release with the README's Run launch, against the same launch without
`--mtp-model … --dspark --mtp-exact-sampling`. Three code-writing prompts
(an LRU cache in C, an asyncio rate limiter in Python, a trie in Rust),
thinking off, temperature 0, 512 tokens each, one run per arm. Rates from
the server log:

| Prompt | DSpark on | DSpark off | On / off |
| --- | ---: | ---: | ---: |
| LRU cache, C | 50.10 | 41.64 | ×1.203 |
| Rate limiter, Python | 48.81 | 41.66 | ×1.172 |
| Trie, Rust | 46.65 | 41.72 | ×1.118 |

The DSpark arm's counters over its lifetime: 612 tokens proposed, 552
accepted (90.20%). All six answers ran to 512 tokens and were sane code.

## Clean-checkout run of the release (r1cc-a5)

A fresh clone of e8d23bb6, following the README's Prepare, Run and Test
steps. Deviations: the downloads and the prepare step were skipped because
window prep-a1 had already written the files, so the steps point at that
output; the server used a test port and its own cache directory.

- **v01:** `deepseek41_prepare.py --verify` on those files: main OK, DSpark
  OK.
- **s01:** `make`, then the Run launch. It listened, built the narrow copies
  at load (101.8 ms, 490.3 MiB), and turned the cache on after its canary
  (7.10 s). A chat request, a 600-token code answer (40.76 tok/s, DSpark
  77.0% accepted) and a short reasoning request all succeeded. SIGTERM exited
  in 1.0 s.
- **t01:** `./ds4_test --server`: pass.
- **t02:** `./tests/test_deepseek41_metal`: pass (150 PASS lines).
- **t03:** `./ds4_test` with `DS4_TEST_MODEL`: 19 OK, 0 errors. Skipped by
  design: the V4 Flash 0731 fixtures, `--dspark-verify-depth` (needs
  `DS4_TEST_DSPARK`), and tests for other models or opt-in modes.
- **t04:** `./ds4_test --dspark-verify-depth` with both models: pass. Its
  V4-only prefix-capture check is skipped on V4.1, which verifies through its
  own path.

Every process was valid, and teardown was proven.

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
  they may have added CPU noise. r1bench-a3 (above) is the clean repeat, on
  the release candidate.
- The window did not pass as a whole. In the same d01 process, the optional
  `DS4_V41_S1TG` arm (`cn`) failed the exactness gate, which stopped the
  window. (`DS4_V41_S1TG` was a letter list of single-group kernels; it was
  removed in ece7746a, see the single-group kernels section.) Rows d02 and d03 (the sidecar and builder-off decode arms) and a01
  did not run. The window's verdict was INVALID, with its rows kept for
  adjudication and no speed verdict. P25 accepted the d01 default-arm rows on
  their digests.

The same window ran the `DS4_V41_S1TG` arms. Their result, and the later
default for `c` and `k`, are in the single-group kernels section below.

The depth rows measured before the release (64K, 131K, 300K) were W5 on the
state-depth tree. The composed depth run of the release is r1bench-a3, above.

## DwarfStar main reference (W8, P29)

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
  queue behind a one-shot canary, which W9 measured (below).
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
quality equivalence. The release's own comparison with DwarfStar main is
rcv-a1, above.

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
- Those entries carry identity `028036d5c520a07d` and canary
  `772938c1f52ab041`. This branch at aeab950a reports the same pair (W9),
  so it indexes them. An earlier version of this page said the branch's
  identity differs; W9 showed that it does not.

Window W7 (P27) ran on this branch at 2357478c, with the stepped canary
that ran between requests (04d4db75):

- **Server:** the canary was `e8c0517e0b2d6638` in all six lifetimes, idle
  and under load. Under a continuous request stream the cache came on 42 to
  51 s after start; idle, after 12 s. No entry was stored or loaded early.
  A request sent while the canary was pending (8,238 tokens) was prefilled
  again rather than restored. That is the gap fbd9ff30 closes: requests now
  queue behind a one-shot canary, which W9 measured (below). Entries
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

## DSpark, serial reasoning and restart behaviour (W9)

Trees:

- P: this branch at `aeab950a7b28c56bacbab373ebc9848cc0b22b8b`, server
  binary sha256 `65c8eb01…`. Cell k6 used the same tree's test build that
  can force the canary to fail (`ds4-server-testhooks`, `114f4a8a…`).
- Production: our production server build of the September 18 stack with
  later serving changes (`872e03cf`, not published), binary `59876783…`.
- For the test cells only: Ivan Fioravanti's main at `a74d62a5` and this
  branch at `04e85391`.

Setup:

- Both servers used the same MXFP4 GGUF and drafter, and the README's Run
  flags: 393K context, prefill chunk 8192, `--dspark
  --mtp-exact-sampling`, a 98,304 MiB disk KV cache with a 4,096-token
  continued interval, `--utility-ctx 32768` and `--serial-reasoning`.
- The decode rates are from the server log. DSpark counters came from
  `DS4_DSPARK_STATS`.
- The window ran every cell once, in order. Its verdict was 10 PASS, 1 FAIL
  (k4).

DSpark and reasoning:

| Cell | Result |
| --- | --- |
| d2 | P, a 36-token code prompt, 600-token answers, three runs: 57.51 / 57.70 / 57.73 tok/s (median 57.70). 95.40% of drafted tokens accepted; 1.703 accepted per cycle. |
| d3 | Production, the same requests: 52.54 / 52.73 / 52.61 (median 52.61); 89.63% accepted; 1.650 per cycle. P/production ×1.097. All six answers were sane code. |
| d4 | P with reasoning on: the reasoning span (context 62 to 2,112) ran 0 DSpark cycles; the answer after it ran 139. 45.71 tok/s over all 2,902 tokens. |
| d5 | An 8K follow-up turn from the disk cache with DSpark on: 8,137 of 8,632 tokens cached, TTFT 2.134 s against 15.277 s cold. |
| d6 | `ds4_test --dspark-verify-depth` on Ivan's main fails on the V4-only prefix-capture assertion, as on this branch before 764ff1cb. |
| d7 | The same test passes at 04e85391, which skips that check on V4.1. |

Restart and the disk cache (each restart reuses the same cache directory):

| Cell | Result |
| --- | --- |
| k0 | Cold: a 303,912-token conversation, TTFT 391.612 s; an 8K one, 16.513 s. Identity `028036d5c520a07d`, canary `772938c1f52ab041`; the canary took 7.15 s idle, and the cache came on 13.76 s after launch. |
| k1 | 8K resend right after a restart: TTFT 9.226 s, 8,105 of 8,600 cached. A fresh 11-token prompt at the same point: 7.361 s. |
| k2 | 300K resend right after a restart: 303,944 of 304,463 cached, prompt done 3.205 s, TTFT 11.172 s. |
| k3 | 300K, 8K and an 11-token request sent together during the canary: answered in that order at 10.876 / 14.748 / 15.506 s. |
| k4 | **FAIL.** SIGTERM 2 s after listen, with the 300K request queued. The server exited 0 in 7.25 s, but the queued request's prefill was interrupted, its 303,944-token entry was deleted (8 entries to 7), and the client saw "Remote end closed connection without response". |
| k5 | Next restart: the 300K conversation restored from its continued checkpoint, 303,104 of 304,472 cached, prompt done 6.149 s, TTFT 14.078 s. |
| k6 | Canary forced to fail: "KV disk cache stays off", no stores, entries unchanged; an 8K request prefilled cold, TTFT 20.231 s. |
| k7 | The cache plus entries seeded from the W8 build (stepped canary `e8c0517e`), relaunched with a 1,024 MiB budget: 18 entries evicted as disk-cache-full, 19 to 1 (60.8 MiB left), including all 11 seeded entries. |

k4 had two causes, and both predate the V4.1 work:

- A failed prefill from a disk entry always deleted the entry, including a
  prefill that shutdown or a client disconnect had interrupted.
- Once shutdown was requested, every reply write failed, so the drain could
  not answer anyone.

e2136edd fixes both. An interrupted prefill keeps the entry and logs "kv cache
kept reason=prefill-interrupted". During shutdown, an interrupted prefill and
a job taken off the queue are both answered 503 "server is shutting down".
The server unit test `test_shutdown_prefill_keeps_kv_and_answers` covers this
(`ds4_test --server`). rcv-a1 re-ran the restart cells on the release, and
k4 passed there: every entry kept and the queued request answered 503.

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
- **The first, inexact EADD kernel:** the Engram gate's three reductions were
  summed as float4 lanes, then SIMD sums, then an 8-way threadgroup sum. That
  is a different order from the 32-lane kernel, and it changed about 1 BF16
  value in 30k. It measured −0.229 / −0.368 ms/token (prose / code). D144's
  general-set NLL was +0.00256 nats/token (SE 0.00137). It was replaced
  because it was not bit-exact. The staged form gives the 32-lane
  bits and keeps about 90% of the gain (micro: 18.6 against 8.25 µs per call,
  32 lanes 122.5). Its register form (0004c5cf) keeps the same bits and
  takes 7.5 to 7.6 µs per call at 1 to 6 rows. A reordered 320-thread form
  was about 1.8 µs faster per call and differed in 2,426 of 40,960 micro
  streams; it is not used.
- **EADD fused into the Engram tail (fusion micro):** exact, but 236.3 µs
  (value rows first) or 271.7 µs (in order) against 227.8 separate.

The micros ran with production up, so their absolute times carry
contention; the A/B order repeated.

## Single-group latency kernels (f45f4e8, 77eea27; default since 7f2ba58)

Two decode kernels, measured as letters of the former `DS4_V41_S1TG` list:

- `c`: the split HC collapse, two dispatches over 32 + 10 groups in place of
  one 1024-thread group (80 calls per token).
- `k`: the decode top-k merge ladder, staged in threadgroup memory and
  finished in one 1024-thread group (8 calls per token).

Both are on by default. Since ece7746a each has its own opt-out, read per
call: `DS4_METAL_DISABLE_V41_SPLIT_COLLAPSE=1` restores the single-group
collapse and `DS4_METAL_DISABLE_V41_TOPK_FUSED_MERGE=1` the per-level merge
ladder. `DS4_V41_S1TG` no longer exists.

Exactness of `c` and `k`:

- **Model-free, the opt-in micros.** These were counted over distinct
  inputs. An earlier version of these receipts counted repeats of the same
  inputs, which show determinism, not coverage.
  - `c`: 0 of 20.48M distinct collapsed values and 0 of 20.48M distinct
    normed values differ, on fresh data for every call, with and without the
    engine's macros. An engine-linked check adds 0 of 819,200.
  - `k`: 140 crafted rows (seven widths; Gaussian, many-tie, `-inf`-tail,
    NaN and all-equal rows), plus 71,680 distinct selected indices per fused
    form. The merge only compares and moves pairs; it does no float
    arithmetic.
- **Model-free, in the test suite.** `tests/test_deepseek41_metal --s1tg`
  ran each call with and without `DS4_V41_S1TG`; since ece7746a the same
  checks are `--split-collapse` and `--fused-merge`, each against its
  opt-out.
  - `c`: 200 fresh calls, 0 of 2,048,000 values differ.
  - `k`: widths 1024 to 65536, covering one level, fused only, and staged
    levels first. Random, tied, `-inf`-tailed, signed-zero, NaN and
    constant rows: 0 of 39,936 selections differ.
- **On the stack at 8K.**
  - Window a6, each letter alone against the same stack without it: both
    bitwise equal. `c` measured −0.069 / −0.061 ms/token and `k` −0.118 /
    −0.105 (prose / code). The base's digests equal the irqres-a1 T arm.
  - Window W10-a2, `c` and `k` together: bitwise equal to the base, with
    logits digests `ed3ca952` (prose) and `c1c571c0` (code).
- **At depth and with the Engram gate, on the release candidate:**
  r1bench-a3's po9 process set both opt-outs (with seven others) and gave
  the control's digests at 8K and 300K.

Removed letters:

- `r`, the ballot router: exact (15,840 distinct values in its micro), but
  slower on the stack in window a6, +0.052 / +0.056 ms/token. Removed in
  d116f7d.
- `h`, eight heads per threadgroup in single-row decode attention: exact
  (6,316,032 distinct partial values per key count), but within noise in a6,
  +0.018 / −0.036 ms/token. Removed in d116f7d.
- `n`, the hoisted paired norm: not exact. See below.

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
loads move ahead of the reduction. No exact form of the hoist gained anything.

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
| 3ea32856, 0004c5cf, 45a2152c | `--engram-add-exact` |
| ece7746a | `--split-collapse`, `--fused-merge` |
| 19da258b, 435249bd, e2cf602b | `--index-score-bf16` |
| e9a34b33 | `--index-candidates` |
| 197979ee | `--topk-select` |
| a2271493, d35a9315 | `--verify-index-bf16` |

The full run of `tests/test_deepseek41_metal` on the release passed in
r1cc-a5.
