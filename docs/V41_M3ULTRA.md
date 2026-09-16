# DeepSeek V4.1 Flash on one Mac Studio M3 Ultra

## 1. What this is

A build of `ds4` that serves **DeepSeek-V4.1-Flash** (Q4, 483.0 GiB on disk) from a
single 512 GB M3 Ultra Mac Studio, at roughly twice upstream's decode rate, with
speculative decoding that is safe to leave on and a disk KV store that turns a
minute-long prefill into a quarter of a second.

Upstream is **[`antirez/ds4`](https://github.com/antirez/ds4)** and the base is
commit `bd66c402070042bf0a79ad6ece8242de4c93680c`, "DeepSeek v4.1 Flash support
for Metal". This is a standalone repository carrying upstream's full history to
that commit plus this branch's work, not a GitHub fork object, so nothing above
the file list will tell you where it came from. `LICENSE` is upstream's,
unchanged (MIT). `README.md` retains upstream's introduction with a fork-specific navigation note.

What the branch adds:

* **Metal decode and prefill kernels** for V4.1 — a rewritten dispatch ladder,
  fused HC and norm producers, paired and streamed Q8 projections, a compact
  index scorer and a bounded radix top-k selector for the compressed-attention
  selection chain.
* **DSpark speculative decoding**, opt-in, with a windowed admission controller
  that prices itself from the wall time of the request it is serving. No
  calibration file, no acceptance table, nothing to fit.
* **A disk KV store with continuable checkpoints and an eviction ladder**, so a
  long conversation leaves restorable waypoints behind and keeps them under disk
  pressure.
* **Exact sampled speculative decode** at any positive temperature.

Every change is an exact reformulation. Under greedy decode this branch emits,
byte for byte, what upstream emits on the same prompt — see section 4.

* [CHANGES-V41.md](../CHANGES-V41.md) — the engineering log: every adopted change, and everything tried and rejected, with receipts.
* [REPRODUCE-V41.md](REPRODUCE-V41.md) — the benchmark invocations, the fixtures, the bound identities.
* [DSPARK-V41.md](DSPARK-V41.md) — the support GGUF, the drafter, the admission controller, exact sampled decode, the width guard, EOS.
* [PROMPTING-V41.md](PROMPTING-V41.md) — coding-agent guidance: draft in real files, test, and revise; includes a copyable prompt and observations from two game-building runs.
* [bench/RELEASE-EVIDENCE-V41.md](../bench/RELEASE-EVIDENCE-V41.md) — which receipt directory and which binary stands behind each table below.
* [THIRD_PARTY-V41.md](../THIRD_PARTY-V41.md) — upstream, DeepSeek's reference stack, weights licences.

## 2. Why it is useful

### Current decode results

All validated post-launch optimizations are included and enabled by default.

| Workload | Original public release | September 16 update |
| --- | ---: | ---: |
| Serial decode, 8k context | 31.3 t/s | **35.9 t/s** |
| Serial decode, 300k context | 28.3 t/s | **34.4 t/s** |
| DSpark on code | 40.5 t/s | **48.7 t/s** |

Six-row verification improved from 112 to **87.0 ms/block**; the same normalized
weight-traffic calculation improved from 337 to **433 GB/s**. The last full agent-answer result remains **41.3 → 50.0 t/s**, from the
earlier update; the full suite was not rerun for this bundle. Serial rates are
best valid runs and code is a normal-serving ABBA mean (132 output tokens;
the launch code fixture used a 512-token budget). These are the validated UAT measurements of the paths
now shipped here; see [the consolidation](DECODE-CONSOLIDATION-20260916.md) for exact scope,
controller behavior, controls, validation, and pruned experiments.

### Original release measurements

The table and discussion below preserve the original launch comparisons,
including prefill/TTFT/cache results that were not retimed for the decode update.


One machine, one weights file, one quiet GPU, the same fixtures on both sides.
The baseline is a clean build of stock upstream except on the DSpark rows, where
the baseline is this branch's own non-drafting path.

| what | baseline | this branch | ratio |
|---|---:|---:|---:|
| serial decode, 8,192-token prefix | 16.560 t/s (upstream) | **31.250 t/s** | **1.89x** |
| serial decode, 300,000-token prefix | 13.950 t/s (upstream) | **28.274 t/s** | **2.03x** |
| cold prefill, 62,000 tokens | 736.75 t/s, 84.15 s (upstream) | **813.47 t/s, 76.22 s** | **1.10x** |
| TTFT, cold, 23,446-token agent system prompt | 35,769 ms (upstream) | **31,083 ms** | **1.15x** |
| TTFT, same prompt restored from the disk KV store | 31,083 ms (this branch, cold) | **229 ms** | **135x** |
| TTFT, restored plus a 111-token append | 31,083 ms (this branch, cold) | **869 ms** | **36x** |
| DSpark, `code512` fixture | 32.09 t/s (this branch, serial) | **40.53 t/s** | **1.26x** |
| DSpark, real agent turns, whole turn | 30.96 t/s (this branch, serial) | **32.68 t/s** | **1.06x** |
| DSpark, real agent turns, answer phase | 31.32 t/s (this branch, serial) | **41.29 t/s** | **1.32x** |

**What that means for an agent session.** Agent turns are long, think-heavy and
arrive in tool loops, and each of those three is served by a different part of
the branch. They are *long* — a randomly drawn real turn from an agent archive is
about as likely to be 200k tokens as 20k — so depth is the whole game: decode
falls 9.5 % from 8k to 300k here against 15.7 % upstream, and the prefill is paid
once, because a long prefill leaves a restorable checkpoint at every chunk
boundary and the next turn restores it in milliseconds. They are *think-heavy* —
reasoning is the normal case in real agent traffic, not a minority one — and the
drafter cannot predict reasoning prose: inside a reasoning span it commits one
token per cycle, so every attempt pays a full verify to advance one row.
Speculation is therefore off inside reasoning spans by default, and stays on
through the answer phase — code, tool calls, structured output — which is where
it pays: 1.32x, against 1.00x across the span it skips. And they arrive in *tool
loops*, each re-sending a prefix of the last prompt; the store's ladder keeps
shallow waypoints alive under pressure, so a mid-history divergence restores from
a rung instead of re-prefilling from token zero.

## 3. How to use it

### Build

```
make -j8
```

### Weights

Nothing is redistributed here. The target is upstream's published calibrated Q4
GGUF, 518,596,067,328 bytes (483.0 GiB on disk), fetched with upstream's own
downloader, which pulls two parts and joins them:

```
./download_model.sh ds41f-q4
```

Every number in this document was measured on that file. The model licence is
in [THIRD_PARTY-V41.md](../THIRD_PARTY-V41.md).

DSpark needs a second, small GGUF built from the `mtp.*` tensors of the official
`deepseek-ai/DeepSeek-V4.1-Flash` checkpoint. They live in three of its 48
safetensors shards (44, 45 and 46), so the conversion downloads 7.4 GiB of
shards plus the config and index, not the whole checkpoint:

```
gguf-tools/deepseek41_dspark_quantize.py --quant q4 <checkpoint> DeepSeek-V4.1-Flash-DSpark-Q4.gguf
```

That produces 8,328,937,472 bytes (7.757 GiB). Details in
[DSPARK-V41.md](DSPARK-V41.md).

### Serial

```
./ds4-server -m DeepSeek-V4.1-Flash-Q4.gguf \
    -c 131072 --host 127.0.0.1 --port 8095 \
    --kv-disk-dir /path/to/kvcache --kv-disk-space-mb 98304
```

The model id the server advertises is `deepseek-v4.1-flash`.

`--debug-levers` must not be passed to a serving process: it holds a full session
per bench fixture and is worth roughly 24 GiB.

### DSpark speculative decoding

`--dspark` is opt-in, requires `--mtp-model`, and is admitted only when
`DS4_V41_DSPARK_DECODE_READY=1` is set in the environment:

```
DS4_V41_DSPARK_DECODE_READY=1 \
./ds4-server -m DeepSeek-V4.1-Flash-Q4.gguf \
    --mtp-model DeepSeek-V4.1-Flash-DSpark-Q4.gguf --dspark \
    -c 131072 --host 127.0.0.1 --port 8095 \
    --kv-disk-dir /path/to/kvcache --kv-disk-space-mb 98304
```

`--dspark` means *drafter loaded and the admission controller on*. There is
nothing else to install, no path to set and nothing to fit.

**How admission decides.** The drafter proposes the full trained block — one
committed row and five draft rows — at every eligible greedy position. Two
mechanisms then decide whether that proposal is worth a target pass:

* **Confidence.** Per-position confidence is the normalised top probability of
  the post-Markov proposal row, which the Markov pass has already left on the
  host, so reading it costs no GPU work. The admitted prefix is the longest run
  from the anchor at or above `p_min`; a proposal whose run is shorter than
  `min_draft` is **declined without verification**, paying the drafter and one
  serial token instead of a full target pass.
* **Measured cost.** A request decodes its first 16 tokens serially so the first
  judgement is made against a measured serial step rather than a guess. A window
  is three chosen calls — declines included — each contributing
  `net = wall_ms − consumed × serial_ms`, judged against the median of the last
  nine measured serial tokens. A window that does not beat serial backs off for
  16, then 32, 64, 128 consumed serial tokens and then asks again. It never
  latches off for the rest of a request.

Reasoning spans decode serially by default; leaving one clears the cooldown.

| lever | environment variable | default | what the non-default does |
|---|---|---:|---|
| `dspark_adaptive` | `DS4_DS41_DSPARK_ADAPTIVE` | 1 | `0`: propose everywhere and never back off — this is the fixed-block arm |
| `dspark_reasoning_serial` | `DS4_DS41_DSPARK_REASONING_SERIAL` | 1 | `0`: draft inside reasoning spans too |
| `dspark_serve` | `DS4_DS41_DSPARK_SERVE` | 1 | `0`: serial decode; the drafter stays loaded |
| — | `DS4_DS41_DSPARK_MIN_SERIAL_TOKENS` | 16 | the serial entry wait, clamped to 0..64; `0` removes it |
| — | `DS4_DS41_DSPARK_P_MIN` | 0.75 | the per-position confidence floor |
| — | `DS4_DS41_DSPARK_MIN_DRAFT` | 3 | an admitted prefix shorter than this declines the cycle, of 5 |
| — | `DS4_DS41_DSPARK_FAIL` | unset | test-only fault injection: `drafter_once`, `state_save`, `after_verify` |

Per-request telemetry rides in the `ds4_dspark` usage block: cycles, committed
rows, attempts, declines, skipped steps, windows, backoffs, losing cycles, serial
rows, the serial and cycle millisecond estimates, the fault ledger and
`dspark_net_ms` — the request's cumulative wall minus rows × measured serial, so
negative is the controller's measured win over serial on that request.
`dspark_committed / dspark_cycles` is committed tokens per verify from the real
consumed count.

### The disk KV store

```
    --kv-disk-dir /path/to/kvcache --kv-disk-space-mb 98304 \
    --kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096
```

A long prefill stages a checkpoint at every backend chunk boundary it crosses, so
a conversation that is interrupted, branched or re-sent as a prefix restores
instead of re-prefilling. Lookup is by text prefix and accepts any stored length.

Stored waypoints are thinned deliberately at store time rather than left for the
eviction pass to cull: every rung within `KV_CACHE_LADDER_DENSE_WINDOW` (32,768
tokens) of the frontier is kept, and below that a geometric skeleton in which each
deeper kept rung sits at or below **5/7** of the previous kept depth. The
eviction score's baseline decays on its own half-life,
`DS4_KVSTORE_BASE_HALF_LIFE_SECONDS`, default 24 h, so unused entries yield to
fresh ones within a day instead of staying immortal; and under disk pressure the
ladder sheds its **deep** end first, because the live slot already serves
near-frontier divergences and a deep rung frees the most bytes for the least
coverage.

`DS4_KV_LADDER=0` restores the previous cheap-victim behaviour in one step.

This is disk-cache-only. Eviction decides which KV snapshots survive on disk, not
what the model computes from one.

### Sampling

Sampled decode takes the exact speculative-sampling path described in section 4.
There is no separate flag and no opportunistic mode to avoid. To have clients
that send no sampling parameters inherit the deployment's published settings —
DeepSeek publishes temperature 1.0, `top_p` 0.95 for V4.1 Flash — add:

```
    --default-top-p 0.95 --default-min-p 0
```

A request that names `top_p` or `min_p` still wins.

### Memory: what fits alongside, and what does not

This model does not fit anything smaller than 512 GB. The resident set alone is
**294.14 GiB**; the whole process lands at **302.020 GiB of wired residency** at
`-c 65536` with `--dspark`, against upstream's 294.789 GiB without a drafter, and
a single slot at `-c 131072` with DSpark off peaks at **304.8 GiB**. The drafter
costs **7.231 GiB wired**, 93.2 % of its GGUF's own 7.757 GiB.

Context size is not the lever — context buffers fit
`7,709.9 MiB + 0.011110 MiB per context token` to better than one part in ten
thousand, so 65,536 and 131,072 differ by 728 MiB, 0.24 % of the resident set.
The slot count is. And **the boot line does not see the drafter**: it prints the
same `resident model 294.14 GiB … = 302.38 GiB planned` either way, so it is a
test for the wrong `-c`, for `--debug-levers` or for too many slots, not for a
drafter. Watch wired pages instead.

**What not to run alongside.** The GGUF is 483.0 GiB but only 294.14 GiB of it is
resident; the remaining **188.9 GiB is the Engram table, read from disk**, and it
wants page cache. Headroom from 512 GiB to a 304.8 GiB peak is 207.2 GiB, of
which ~188.9 GiB belongs to that table — leaving about **18.3 GiB genuinely
discretionary, or ~11.1 GiB with the drafter loaded**. A ~35 GB vision sidecar
and a fully cached Engram table do not both fit; one of them wins, and the cost
of losing the page cache is prefill latency, not a memory alarm. Do not respond
to memory pressure by dropping the Engram cache — that converts a memory alarm
into a prefill regression.

### Coding-agent prompting: draft in files

For tool-using coding tasks, explicitly ask V4.1 to write a small runnable
draft to a real file, test it, and revise from observed results. Settle the
design choices up front. This can reduce the time spent drafting and
reconsidering the implementation in reasoning before writing it.
See [the copyable instruction block and observed results](PROMPTING-V41.md).

## 4. What is guaranteed

**Every change that ships here is exact.** Each one performs the same arithmetic
in the same order as the upstream code it replaces, so no shipped change moves a
bit of the model's output. There is no alternative "safe" or "reference" mode to
enable, because the default build already computes what upstream computes: a
stock build is the configuration the numbers in this document describe.

**Byte identity with upstream, under greedy decode, on the published gates.**
Every greedy arm in this release emits, byte for byte, what upstream `bd66c40`
emits on the same prompt. That covers serial decode at four depths, every DSpark
arm including the fixed block and the controller, both TTFT arms, the
restored-from-disk TTFT arm, the held-out real-agent screen and all 252
Spec-Bench requests. The reference digests, the fixture digests and the binary
identities are bound by SHA-256 in
[bench/v41-manifest.json](../bench/v41-manifest.json), and
[bench/verify-v41-manifest.py](../bench/verify-v41-manifest.py) checks a checkout, a
fixture set or a finished run against them.

Two properties of that identity are worth naming. It holds with DSpark drafting
active — under greedy decode the controller changes how many tokens are committed
per cycle, not which tokens. And the restored-from-disk arm emits the same bytes
as the cold arm on both binaries, which is a direct check that a KV snapshot
reproduces the cold computation rather than approximating it.

**Sampled decode is exact speculative sampling.** The drafter is deterministic —
its proposal is an argmax, so `q(x) = 1` and `min(1, p/q)` reduces to accepting
the drafted token `x_j` with probability `p_j(x_j)` under the target's *filtered*
distribution for that row, and on rejection sampling the residual. Two details
make that exact rather than approximate: the filters are applied to the original
row logits and the support is fixed *before* the drafted token is masked, so
`top_p` and `min_p` are not re-derived from a distribution the rejection already
changed; and the engine hands the serving loop a one-hot row carrying the
decision the acceptance rule already made, so the loop's own re-sample cannot
discard the residual and redraw from `p_j`, which would give `P(x) = p(x)(2 −
p(x))` instead of `p(x)`.

Byte identity is **not** claimed at temperature above zero and is not expected:
the object reproduced there is a distribution, not a draw, and the two paths
consume the RNG in different orders. What is certified instead is **determinism**
— the same seed and configuration give the same bytes — and **distribution
correctness**: the shipped `ds41_accept_or_residual()` itself, not a copy, is
driven by `tests/test_sampling.c` through a chi-square test against the target's
filtered distribution for every drafted token index, at `top_p` 1.0 / 0.95 / 0.5
crossed with `min_p` 0 / 0.05, 200,000 draws each, passing at chi-square < 150.

**DSpark refuses a session it cannot answer correctly.** Up to the verify batch a
drafter failure is *drained* — the target's KV, position and checkpoint are
untouched — so the step falls through to serial, the session latches DSpark off
for its lifetime, and the request finishes normally. After the batch it is *not*
drained: rows have been consumed that the session cannot un-feed, so serial
decode from that point would answer from a state no serial decode would have
reached. There is no correct fallback, and the engine says so instead of
inventing one. The latch is session-scoped; only destroying the session clears
it.

| injected failure | kind | the faulted request | the next request on that session |
|---|---|---|---|
| `drafter_once` | safe | 192 tokens, `finish=length`, byte-identical to serial | byte-identical to serial, drafting disabled |
| `after_verify` | unsafe | 17 tokens, `finish=error` | refused — HTTP 500, `dspark: unsafe drafter failure requires a new session` |

## 5. Original release evidence

For the subsequent decode campaigns, see [September 16 evidence](../bench/v41-20260916/README.md) and the [full experiment history](experiments/v41-20260915/README.md).

Everything below is one 512 GB M3 Ultra Mac Studio, one model process at a time,
the GPU sampled idle before every timed arm. Nothing is projected, averaged
across machines or extrapolated from a neighbouring shape.
[bench/RELEASE-EVIDENCE-V41.md](../bench/RELEASE-EVIDENCE-V41.md) names the receipt
directory, the server digest and the commit behind each table.

### Serial decode against upstream, four depths

512 greedy tokens from a restored prefix, `gen_steady_tps` (tokens 2..N over the
sum of their individual eval spans). Held fixed: the weights file, the fixture,
the context size and the machine. Every arm on both binaries is byte-identical to
its depth's upstream reference.

| prefix depth | upstream | this branch | ratio |
|---:|---:|---:|---:|
| 8,192 | 16.560 | **31.250** | 1.89x |
| 62,000 | 16.230 | **30.683** | 1.89x |
| 131,072 | 15.290 | **29.898** | 1.96x |
| 300,000 | 13.950 | **28.274** | **2.03x** |

Decode is not flat with context on either binary — V4.1's compressed sparse
attention attends a fixed 512 selected compressed rows plus a 128-row raw window,
but the scoring and sorting that *choose* those 512 rows grow with depth. What
this branch changes is how much they cost: the fall from 8k to 300k is 9.5 %
here against 15.7 % upstream.

### Cold prefill, and what a restore costs instead

One cold prefill per binary per depth.

| tokens | upstream | wall | this branch | wall | ratio | restore from disk |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 626.94 | 13.07 s | **711.46** | 11.51 s | 1.14x | 5.9 ms |
| 32,768 | 740.59 | 44.25 s | **835.72** | 39.21 s | 1.13x | 15.2 ms |
| 62,000 | 736.75 | 84.15 s | **813.47** | 76.22 s | 1.10x | 16.1 ms |
| 131,072 | 692.06 | 189.39 s | **781.90** | 167.63 s | 1.13x | 46.6 ms |
| 300,000 | 570.30 | 526.04 s | **632.62** | 474.22 s | 1.11x | 104.0 ms |

A 300,000-token prefix comes back from disk about 4,560x faster than it prefills.
That, not the 1.11x, is what makes depth affordable.

### Time to first token on a 23,446-token agent system prompt

| arm | upstream | this branch |
|---|---:|---:|
| cold | 35,769.47 ms | **31,082.86 ms** |
| restored from the disk KV store, fresh session | not measured | **229.34 ms** |
| restored plus a 111-token append | not measured | **869.43 ms** |

The upstream restored cells are blank on purpose: upstream supports
`--kv-disk-dir` and would restore, and the restore path is shared machinery
rather than a production change, so two more model loads would have bought a
number that cannot move.

### What the disk store's checkpoints are worth

Held fixed: one conversation, one store, the same prompt re-sent as a strict
prefix.

| | before | after |
|---|---:|---:|
| mid-prefill store during a long cold prefill | none written | `tokens=94208 trimmed=0 reason=continued size=593.70 MiB save=101.9 ms` |
| a 96,591-token strict prefix of that conversation | cold, ~140 s | 94,208 restored (load 50.9 ms), 2,383 prefilled, **10.3 s** |
| a 38,690-token strict prefix of an agent chat | ~50 s from the anchor | 37,728 restored (load 32.5 ms), 962 prefilled, **4.11 s** |
| a 216,598-token summariser prompt on a live 299k conversation | 248,428 re-prefilled, 365.3 s | 2,393 re-prefilled — 98.9 % served from disk — **13.7 s** |

### DSpark per fixture: serial, fixed block, controller

One resident server, one boot, three arms per fixture, every arm byte-identical
to its serial counterpart. `serial` is the drafter loaded and not served, which
is also the k=1 non-regression control: it sits at 31.65–32.09 t/s across all
nine fixtures, inside the 8k serial figure above.

| fixture | generated | serial | fixed block | **controller** | block / serial | **ctl / serial** |
|---|---:|---:|---:|---:|---:|---:|
| `sql512` | 512 | 31.78 | 50.75 | **50.10** | 1.597x | **1.576x** |
| `mixed-sql512` | 512 | 31.74 | 50.52 | **49.97** | 1.592x | **1.574x** |
| `json512` | 512 | 31.78 | 44.46 | **42.21** | 1.399x | **1.328x** |
| `code512` | 132 | 32.09 | 47.17 | **40.53** | 1.470x | **1.263x** |
| `math512` | 263 | 31.88 | 36.66 | **35.54** | 1.150x | **1.115x** |
| `rag512` | 174 | 31.80 | 28.68 | **31.92** | 0.902x | **1.004x** |
| `prose512` | 512 | 31.77 | 18.67 | **31.38** | 0.588x | **0.988x** |
| `chat512` | 512 | 31.76 | 21.47 | **31.38** | 0.676x | **0.988x** |
| `summary512` | 276 | 31.65 | 23.33 | **30.97** | 0.737x | **0.979x** |

**This table is the argument for admission control in one picture.** A fixed
block gives up 41 % on prose and 32 % on chat; the controller gives up 1 % on
both, because after eighteen declines it stops paying the drafter at every
position to rediscover that prose does not draft. Where the block wins, the
controller keeps almost all of it.

The three release gates on the same boot, all byte-identical to their upstream
references:

| gate | serial | fixed block | controller | identity |
|---|---:|---:|---:|---|
| 8k `code` | 31.465 | 36.254 | **32.779** | IDENTICAL |
| 8k `sql` | 31.250 | 49.373 | **48.385** | IDENTICAL |
| 8k `agent2` | 31.378 | — | **34.281** | IDENTICAL |

`agent2` is a real agent transcript held out of every fit, and it is the fixture
a fixed block used to lose 19 % on. The controller is 1.09x serial there.

### The held-out real-agent screen

Not fixtures: real agent traffic. 226 archived agent sessions, every real
assistant turn reconstructed as the request that produced it, 32 drawn at random
with a fixed seed and held out — no constant in the controller was fitted on
them. Fourteen were measurable within the screen's cost cap, 6.7k to 107k
rendered tokens, three arms each, a 4,096-token cap with natural EOS. Phases are
separated on the stream, the first delta that is not reasoning content ending the
reasoning phase; all three arms are greedy and byte-identical, so a phase holds
the same tokens in every arm and its ratio is the ratio of its wall times. 14/14
byte-identical to serial in both controller arms.

| phase | serial | **controller** | ratio | controller, reasoning gate lifted | ratio |
|---|---:|---:|---:|---:|---:|
| whole turn | 30.96 | **32.68** | **1.056x** | 31.70 | 1.024x |
| reasoning | 30.85 | 30.72 | 0.996x | 29.64 | **0.961x** |
| answer | 31.32 | **41.29** | **1.318x** | 41.06 | 1.311x |

Every one of the twelve requests that opened an answer phase is faster in it,
from 1.13x to 1.46x. The two that never left reasoning land at 0.995 and 0.998 —
the gate doing exactly what it is for. Whole-turn ratios run 0.995 to 1.270, and
a request's whole-turn number is almost entirely a function of how much of the
turn was reasoning.

**The reasoning gate is measured, not inherited.** Lifting it costs 3.5 points on
the reasoning phase and 3.2 on the whole turn, wins the reasoning phase on 2
requests of 13, and loses 18 % on the worst. Inside a reasoning span the drafter
commits 1.0 token per cycle — the bonus row and nothing else — so the gate-off
arm spends an all-reasoning turn discovering, three declines at a time, that
reasoning does not draft: 34 backoff episodes and 3,978 skipped positions on each
of the two 4,096-token all-reasoning requests.

The screen is conservative twice over: the cost cap excluded the long draws,
which is where the controller does best, and nine further draws exceeded a
393,216-token context and were not measurable at all. Every one of the 32 drawn
turns is reasoning-enabled, which is why the gate-lifted arm is in the table — a
two-arm screen would have compared serial with serial.

### One long served turn at depth

Through the ordinary serving path on a post-compaction context: prompt 86,961
tokens of which 86,793 were cached (99.81 %), 168 prefilled in 1.2 s, then
**56,497 generated tokens, `finish=stop`, 2,018.4 s — 28.0 t/s for the whole
turn**. Of 1,130 decode samples, 974 carry the reasoning flag and 156 do not: the
reasoning run is 48,700 tokens at a **median 27.20 t/s** and the answer run that
follows is 7,797 tokens at a **median 36.75 t/s**, 1.35x, at roughly 135k depth.
One turn, both behaviours, nothing altered.

### Spec-Bench: the workload where drafting loses

A frozen public subset, so the release is not judged only on fixtures chosen by
the people who built it: 72 questions, 84 turn requests per mode, 252 requests,
greedy, natural EOS, 256-token cap, reasoning off, one resident server. Every one
of the 252 answers is byte-identical across the three modes. Method, per-category
tables and receipts: [bench/SPEC-BENCH-V41.md](../bench/SPEC-BENCH-V41.md).

| boundary | serial | fixed block | **controller** |
|---|---:|---:|---:|
| server end-to-end | 29.43 | 22.32 (0.758x) | **29.32 (0.996x)** |
| decode-only | 31.89 | 23.55 (0.739x) | **31.65 (0.993x)** |
| client wall | 29.34 | 22.26 (0.759x) | **29.23 (0.996x)** |

Per category, decode-only:

| category | fixed block | **controller** |
|---|---:|---:|
| `mt_bench` | 0.628x | **0.979x** |
| `qa` | 0.729x | **0.973x** |
| `summarization` | 0.756x | **0.968x** |
| `rag` | 0.839x | **0.982x** |
| `translation` | 0.864x | **0.950x** |
| `math_reasoning` | **1.192x** | **1.130x** |

**DSpark loses on this workload, badly, and the controller is why that does not
matter.** A fixed block gives up 26 % of serial; the controller gives up 0.7 %.
The cause is one number: a fixed block commits **2.71** tokens per verify here,
against 5.0–6.0 on the release fixtures, and a block verify needs about four to
break even. The controller attempts 439 cycles where the fixed arm attempts
4,503, commits **4.36** per verify on those, and spends 8,311 positions in
cooldown — and it still re-tests, 238 backoff episodes across 84 requests, rather
than latching off. Latching would have scored serial exactly here and would never
have found the 1.13x on `math_reasoning` or the 1.32x on real agent answers.
This is the clearest answer to whether a fixed block is safe as a production
default: on a workload nobody here chose, it costs a quarter of the throughput.

## 6. Original release limits

The later update's additional validation and remaining limits are documented in [the release update](RELEASE-V41-20260916.md).

* **Both of the original targets — 80 % of the DRAM wall in decode, 70 % of the
  matmul roof in prefill — were missed, and are recorded as missed.** Decode
  utilisation, `13.77 GB / (ms per token × 705 GB/s)` against this machine's
  measured 705 GB/s memory wall and its measured 13.77 GB of per-token traffic,
  reaches **61 % at 8k**. Cold 62k prefill reaches **54.2 %** of the 27.4 TFLOPS
  arithmetic ideal, and there is no known path from here to 70 % on this chip:
  the production kernels' own measured wall is 22.7–24.1 TFLOPS on wide shapes
  rather than 27.4, because the Metal 4 TensorOps path (`mpp_direct_rhs`) is
  hardware-disabled before M5. ~60 % is the number to plan against on this GEMM
  formulation.
* **Decode is not flat with context.** 31.3 t/s at 8k falls to 28.3 at 300k. The
  growth is in the selection chain that chooses which compressed rows to attend
  to; attention itself, the gather, the routed MoE and the dense projections are
  flat across a 36x context range.
* **DSpark's verify cost is the binding constraint, not acceptance alone.** A
  block cycle costs about 120 ms against a ~31.8 ms serial step, so it must commit
  about 3.8 tokens to break even — which is why "should I draft" is a per-request
  question and not a per-deployment one. Drafting loses outright on plenty of real
  workloads, and admission is not free: on the Spec-Bench subset the controller
  runs at 0.993x serial, the price of proposing, losing, backing off and
  re-testing rather than latching off.
* **Agent workloads sit near break-even whole-turn.** The whole-turn gain on real
  agent traffic is 1.06x; the useful number is the 1.32x on the answer phase, and
  how much of a turn is answer is a property of your traffic, not of this branch.
* **Admission has two signals and both are coarse.** Wall time is judged three
  attempts at a time and confidence one position at a time against a fixed floor,
  so a workload whose acceptance swings faster than a three-attempt window can
  follow is served worse than one that could see it coming. `p_min` is a constant,
  not something fitted per workload.
* **Byte identity is a greedy claim, and it is per gate.** It is the complete
  fidelity criterion for this branch, because every shipped change is exact — but
  it is established on the gates and fixtures that were run, not over a corpus,
  and there is no byte-identity claim at temperature above zero. Section 4 says
  what is certified there instead.
* **The disk KV store's eviction ladder is proven at the store layer and on one
  live conversation, not across days of traffic.** There is a store-layer
  reproducer of the failure it was written for, restore timings for the
  checkpoint work and one real compaction on a 299k-token conversation — but no
  published multi-week serving trace.
* **`--prefill-chunk` above 8,192 has no effect on V4.1.** `DS41_PREFILL_CAP` is a
  compile-time 8,192 that the prefill path allocates its batch buffers against, so
  a larger value is parsed and then clamped. Raising it is a memory-plan decision.
  Separately, `--dspark` does not imply `DS4_V41_DSPARK_DECODE_READY=1`; that
  environment gate has to be set by hand.
* **The Spec-Bench subset is a subset** — 72 questions, the first 12 rows per
  category across six, frozen and hash-bound before any result was seen, with its
  categories weighted by its own shape rather than by any workload. And the two
  agent fixtures are real transcripts and are not redistributed;
  [REPRODUCE-V41.md](REPRODUCE-V41.md) describes their shape so
  comparable ones can be built.
* **Per-arm JSON receipts are not published for the kernel arms.**
  [bench/v41-manifest.json](../bench/v41-manifest.json) binds the commits, binary
  digests, fixtures and reference outputs, and the Spec-Bench run ships full
  per-request receipts; a per-arm receipt directory for the decode and prefill
  sweeps does not exist.

## 7. Licence and attribution

`LICENSE` is upstream's, unchanged (MIT), and applies to this branch's work as
well. The model weights are not distributed here and carry their own licence.

The DSpark admission controller and the disk KV store's eviction ladder are ports
of this project's GLM-5.3-Flash branch, where both designs were developed and
shipped; the design credit is that branch's and what is new here is the port, its
V4.1 reproducers and the corrections found in porting.
[THIRD_PARTY-V41.md](../THIRD_PARTY-V41.md) records upstream, DeepSeek's reference
stack, the Spec-Bench checkout and every weights licence.
