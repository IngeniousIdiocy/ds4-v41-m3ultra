# Bounded Spec-Bench subset — V4.1 Flash, three DSpark modes

`reproduce-spec-bench-v41.py` runs a frozen, category-balanced subset of the
upstream Spec-Bench dataset through the real chat API, in three DSpark modes.
It sits between the six hand-written fixtures of the release matrix and a full
benchmark run. **It is not a full Spec-Bench score.**

It exists because the release's DSpark numbers come from fixtures we chose, and
a reader is entitled to ask what happens on questions we did not choose.

## Scope, and why it is fixed

The driver requires the upstream Spec-Bench repository at commit
`fd2c1cd7d2201ef71db4c5f4e455008f017967bf` and refuses to run unless
`data/spec_bench/question.jsonl` hashes to
`4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04`.

It selects the **first 12 question rows, in stable file order, from each of
`mt_bench`, `translation`, `summarization`, `qa`, `math_reasoning` and `rag`**.
The 12 MT-Bench rows have two turns each; the other 60 have one. So the scope is

- **72 questions**
- **84 turn requests per mode**
- **252 requests** across serial, six and controller

and the driver asserts those three numbers rather than trusting them. If the
dataset changes shape, the run refuses instead of silently measuring something
else.

Every turn is greedy, permits natural EOS, and caps generation at 256 tokens,
and asks for **no reasoning** (`reasoning_effort: "none"`). That last part is a
stated choice, not a silent one: this engine reasons by default, and a reasoning
span would spend the whole 256-token cap without producing an answer. Spec-Bench
measures answer generation, so the subset turns reasoning off and says so. What
the controller does *inside* a reasoning span is a separate question, answered by
the held-out agent screen in the release notes, not here.
Every declared turn gets **one attempt**. There are no retries and no choosing
of a favourable attempt. A failed first turn blocks its dependent second turn
and is recorded as blocked, not as missing.

Dataset text stays in the caller's checkout. `subset-manifest.json` binds every
selected row and every turn by index, question id, byte count and SHA-256
without copying the text into this repository.

## The three modes

| mode | DSpark | levers | what it is |
|---|---|---|---|
| `serial` | not served | `dspark_serve=0`, `dspark_reasoning_serial=1` | the non-drafting path, and the denominator |
| `six` | served | `dspark_serve=1`, `dspark_adaptive=0`, `dspark_verify_rows=0`, `dspark_reasoning_serial=1` | always attempt, no admission control. The trained block is six rows, so this *is* the fixed-six arm — the arm that loses on the held-out `agent2` fixture |
| `controller` | served | `dspark_serve=1`, `dspark_adaptive=1`, `dspark_verify_rows=0`, `dspark_reasoning_serial=1` | the shipped defaults: the windowed cost-feedback admission controller on, full trained block width. **The production default** |

`controller` is exactly what a stock checkout does: the levers named for it are
the binary's own defaults, set explicitly only so the receipt is unambiguous.

With one server process per mode (the upstream shape) the modes are selected by
the process's flags and environment: `serial` gets no `--dspark`, `six` gets
`--dspark` plus `DS4_DS41_DSPARK_ADAPTIVE=0` and `DS4_V41_DSPARK_VERIFY_ROWS=6`,
`controller` gets `--dspark` and nothing else. Inherited `DS4_*` and `MTL_*`
variables are scrubbed.

### The one declared deviation: a resident server

`--use-running-server` switches one already-listening server between the three
modes through `POST /debug/levers` instead of starting a process per mode. The
reason is model-load economy and nothing else: the target is 483 GiB and the
drafter sits beside it, so three loads cost most of an hour of exclusive GPU
time that buys no information. The run reported below used it.

It is a deviation, so it is proved rather than asserted. After each switch the
driver reads the endpoint's full lever map back and refuses the run unless every
lever it asked for reads back at the value it asked for. That map is kept per
mode in `<mode>-levers.json` and in `report.json` under `mode_levers`, so a
reader can check the lever state each request set actually ran under.

### The KV cache, and why the run is primed

The resident server was booted with `--kv-disk-dir`, and the binary has no
runtime cache bypass. Left alone that would make the comparison unfair rather
than merely warm: `summarization` and `rag` prompts are long enough to be
stored, so whichever mode ran first would pay a cold prefill on them and the
other two would restore them.

`--prime-prefixes` runs an untimed one-token pass over every first turn before
any timed request. All three modes then meet the same cache state — identical
for all three rather than zero for all three, which the upstream method gets by
leaving disk KV unconfigured. Every request records its own `cached_tokens`, so
the equality is checkable in the receipts rather than taken on trust, and the
**decode-only** boundary below is untouched by cache state either way.

## Running it

```sh
bench/reproduce-spec-bench-v41.py \
  --spec-bench-root /path/to/Spec-Bench \
  --server ./ds4-server \
  --model /path/to/DeepSeek-V4.1-Flash-Q4.gguf \
  --mtp-model /path/to/DeepSeek-V4.1-Flash-DSpark-Q4.gguf \
  --ctx 32768 \
  --output /new/evidence/directory
```

Use a new output directory; the driver refuses an existing one.

`--dry-run` validates every identity, selects the subset and writes
`subset-manifest.json` and `identity.json` **without starting a server or
touching the GPU**. Run it first; it is free and it catches a wrong dataset
checkout before you take the lock.

`--preflight-smoke` runs one two-turn MT-Bench row and one single-turn QA row —
3 requests per mode, 9 in total. It is always labelled smoke in the manifest,
and **no subset performance claim may be made from a smoke run.**

`--modes serial` runs the baseline alone and needs no `--mtp-model`.

To run against a server that is already up — the deviation described above —
add `--use-running-server`, point `--server-log` at that server's log so the
driver can read the per-request timer lines, and add `--prime-prefixes` if the
server was booted with `--kv-disk-dir`:

```sh
bench/reproduce-spec-bench-v41.py \
  --spec-bench-root /path/to/Spec-Bench \
  --server ./ds4-server \
  --model /path/to/DeepSeek-V4.1-Flash-Q4.gguf \
  --mtp-model /path/to/DeepSeek-V4.1-Flash-DSpark-Q4.gguf \
  --host 127.0.0.1 --port 8199 \
  --use-running-server --server-log /path/to/server.log --prime-prefixes \
  --output /new/evidence/directory
```

`--server` and `--model` are still required in that form: they are hashed into
`identity.json`, which is how a reader knows which binary produced the numbers.

Run the full subset only with exclusive GPU access and one model process on the
machine.

## Expected runtime

Arithmetic rather than a guess, so you can see which term dominates on your box.

**Generation.** 84 turns per mode at a 256-token cap. At a serial rate of
~31 t/s a turn that runs to the cap takes 8.2 s, so a mode in which every turn
hit the cap would spend 84 x 8.2 s ~ **11.5 minutes** generating. Natural EOS
makes most turns shorter, so treat that as the ceiling. `six` and `controller`
are faster where drafting pays and no slower where it does not.

**Prefill.** Spec-Bench prompts are short by this engine's standards — hundreds
of tokens for `mt_bench`, `qa`, `translation` and `math_reasoning`, one to three
thousand for `summarization` and `rag` — so well under a second per turn.

**Model load dominates, and with one server per mode it is paid three times.**
The 483 GiB target takes about 104 s cold and ~10-20 s warm. That is the whole
reason `--use-running-server` exists: with a resident server the load is paid
once, by whoever booted it, and the subset itself costs only its generation and
prefill terms.

**So: about 30-40 minutes for all three modes on a resident server, or an hour
with one server per mode.** `--dry-run` costs nothing and `--preflight-smoke`
costs a few minutes.

## What the report contains

`report.json` and the per-mode `*-records.jsonl` keep **three timing boundaries**
separate and never average them together, as the GLM subset report does:

- **server end-to-end** — the engine's own chat timer for that request, from
  before prefill through the last generated token.
- **decode-only** — that timer minus the engine's prefill timer (`prompt done`),
  i.e. the decode clock that starts immediately after prefill. This is the
  boundary that isolates the thing DSpark changes.
- **client wall** — the monotonic duration of the HTTP request.

For each boundary it reports pooled throughput (total tokens over total time),
mean per-question throughput, and matched elapsed and throughput ratios against
serial. The elapsed ratio and the throughput ratio are separate statements
because natural-EOS output lengths can differ between modes: a mode that emits
fewer tokens can look faster pooled and slower per question.

Every unit keeps its status, token counts, all three timings, its DSpark
consumed counts, the answer's byte count and SHA-256, and the request body's
SHA-256. **Neither the prompt text nor the full server trace is written into
this repository** — the dataset text stays external, as the method requires.
A second MT-Bench turn includes **that mode's own first answer**, so later
prompts legitimately differ across modes; each request's own SHA-256 is retained
rather than assumed equal.

### Committed tokens per verify, from real consumed counts

The chat path now emits the DSpark block in `usage.ds4_dspark`, so committed
tokens per verify is measured here rather than estimated: `dspark_committed` is
the number of tokens the frontend actually consumed out of verified blocks, a
terminal partial block included, and `dspark_cycles` is the number of verify
calls that produced them. The ratio is the real consumed count, not an
accepted-plus-anchor estimate. The same block carries the controller's own
ledger — `dspark_attempts`, `dspark_skipped_steps`, `dspark_windows`,
`dspark_backoffs`, `dspark_losing_cycles` — so the backoff behaviour is visible
per request.

## Results

One resident server per run, modes switched through `/debug/levers`; the lever
maps read back per mode are in the receipts. The run below is the shipped
admission controller: confidence admission in, and a confidence decline priced as
a chosen call inside the three-call window. It is the run whose receipts ship in
`bench/receipts/`, and it is the run docs/V41_M3ULTRA.md quotes. An earlier run of the same
frozen subset, before declines were priced, is kept below for comparison.

| | |
|---|---|
| server binary SHA-256 | `cfeb2c4d186d6000e2687de88df228272eaff7d86a28ef16405a9480f3f188b6` |
| target | `DeepSeek-V4.1-Flash-Q4.gguf`, 518596067328 bytes |
| drafter | `DeepSeek-V4.1-Flash-DSpark-Q4.gguf`, 8328937472 bytes |
| Spec-Bench commit | `fd2c1cd7d2201ef71db4c5f4e455008f017967bf` |
| question.jsonl SHA-256 | `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04` |
| questions / turns per mode / requests | 72 / 84 / 252 |
| byte identity vs serial | `six` 84/84, `controller` 84/84 |

| mode | ok | failed | blocked | completion tokens | cached prompt tokens |
|---|---:|---:|---:|---:|---:|
| `serial` | 84/84 | 0 | 0 | 12227 | 17882 |
| `six` | 84/84 | 0 | 0 | 12227 | 17882 |
| `controller` | 84/84 | 0 | 0 | 12227 | 17882 |

### What the numbers say, before the tables

**On this subset DSpark loses, and that is the finding.** A fixed block runs at
0.739x serial decode-only; the controller holds it to 0.993x. The cause is
visible in one column: committed tokens per verify is **2.71** for the fixed
block here, against 5.0-6.0 on the release fixtures. A block verify costs roughly
four serial steps, so it needs about four committed tokens to break even; at 2.7
it cannot.

That is not a defect in the controller, it is the question these prompts ask.
Spec-Bench is short open-ended prose — write a travel blog post, translate a
sentence, answer a trivia question — and the drafter is far less sure of
unconstrained prose than of code, tool calls or structured text. The one category
where it is sure, `math_reasoning`, is the one category where a fixed block wins:
4.36 committed per verify and **1.192x** serial.

So the honest summary is two sentences. Where drafting does not pay, the shipped
controller turns a 26 % loss into a 0.7 % one, which is what an admission
controller is for. Where drafting does pay it stays out of the way, and on
`math_reasoning` it keeps 1.130x of the block's 1.192x.

The residual is the price of continuing to ask: the controller re-tests after 16,
32, 64 and 128 consumed serial tokens rather than latching off, so on a workload
that loses almost everywhere it still pays for 238 backoff episodes' worth of
re-testing across 84 requests, 75 of which had at least one. That is a deliberate
choice — a controller that latches has serial as its floor wherever drafting
loses *and* never discovers when it starts winning again.

### Pooled throughput, three boundaries

| boundary | serial | `six` | **`controller`** | six ratio | **controller ratio** |
|---|---:|---:|---:|---:|---:|
| server end-to-end | 29.43 | 22.32 | **29.32** | 0.758x | **0.996x** |
| decode-only | 31.89 | 23.55 | **31.65** | 0.739x | **0.993x** |
| client wall | 29.34 | 22.26 | **29.23** | 0.759x | **0.996x** |

### Per category — decode-only, pooled t/s

| category | tokens | serial | `six` | **`controller`** | six ratio | **controller ratio** |
|---|---:|---:|---:|---:|---:|---:|
| `mt_bench` | 4747 | 31.91 | 20.06 | **31.23** | 0.628x | **0.979x** |
| `translation` | 269 | 31.97 | 27.62 | **30.38** | 0.864x | **0.950x** |
| `summarization` | 2251 | 31.78 | 24.04 | **30.76** | 0.756x | **0.968x** |
| `qa` | 2046 | 31.99 | 23.31 | **31.11** | 0.729x | **0.973x** |
| `math_reasoning` | 1609 | 31.95 | 38.09 | **36.10** | 1.192x | **1.130x** |
| `rag` | 1305 | 31.73 | 26.61 | **31.16** | 0.839x | **0.982x** |

### Per category — server end-to-end, pooled t/s

| category | serial | six ratio | **controller ratio** |
|---|---:|---:|---:|
| `mt_bench` | 29.74 | 0.648x | **0.981x** |
| `translation` | 19.14 | 0.942x | **0.998x** |
| `summarization` | 29.84 | 0.769x | **0.970x** |
| `qa` | 30.14 | 0.746x | **0.977x** |
| `math_reasoning` | 28.04 | 1.181x | **1.123x** |
| `rag` | 31.73 | 0.839x | **0.982x** |

### The controller's own ledger

| | `six` | **`controller`** |
|---|---:|---:|
| verify calls | 4503 | **439** |
| attempts | 4503 | **439** |
| committed tokens | 12222 | **1915** |
| committed per verify | 2.714 | **4.362** |
| serial rows | 5 | 2691 |
| skipped (cooldown) steps | 0 | 8311 |
| judged windows | 0 | 362 |
| backoff events | 0 | 238 |
| losing cycles | 0 | 140 |
| requests with >=1 backoff | 0 | 75 |

Committed tokens per verify is the real consumed count, terminal partial blocks
included. `declines` is a counter the earlier run's build did not have.

### The receipts

`bench/receipts/spec-bench-v41/2026-09-14-glmctl/` holds everything this section was computed from:

| file | what it is |
|---|---|
| `identity.json` | binary, target and drafter hashes and sizes, Spec-Bench commit and question-file hash |
| `subset-manifest.json` | every selected row and turn bound by index, question id, byte count and SHA-256 |
| `<mode>-levers.json` | what was asked of `/debug/levers` and what it read back, per mode |
| `<mode>-records.jsonl` | one receipt per request: status, token counts, all three timings, `cached_tokens`, the DSpark consumed counts, and the SHA-256 of the request body and of the answer |
| `prime-pass.json` | the untimed priming requests |
| `report.json` | the pooled, per-category and matched-ratio summary |
| `receipt-hashes.json` | SHA-256 of each of the above |

No prompt text and no server trace is in any of them: the dataset text stays in the caller's
checkout, and a request is identified by the hash of its body.

One edit was made to the receipts as produced: the run's `--spec-bench-root` was an absolute
path on the machine that ran it, and `identity.json` and `report.json` recorded it verbatim.
Both now carry the checkout's basename instead, and `receipt-hashes.json` is computed over the
edited files. The driver emits the basename from here on, so a fresh run reproduces these files
without the edit. Nothing else in either file was touched, and no measured value changed.

### Byte identity against serial, per question turn

| mode | compared | identical | mismatched |
|---|---:|---:|---:|
| `six` | 84 | 84 | 0 |
| `controller` | 84 | 84 | 0 |

Every answer in both DSpark modes is byte-identical to the serial answer for the same turn.

Cache state: 84 of 84 turns reported the same `cached_tokens` in all three modes.


## The earlier run, before confidence declines were priced

Same subset, same resident-server method, the same three modes. The only
difference in the engine is the controller's window accounting: a confidence
decline did not take a slot in the three-call window, so nothing throttled
probing on content where every proposal declines. Decode-only, the controller
reads 0.959x serial here against 0.993x above; end-to-end 0.966x against 0.996x.
Nothing about the drafter or the verify width differs between the two.

| | |
|---|---|
| server binary SHA-256 | `463b186a2b96a7813fb0391e370fd1ad6b28ac348579a4dd71bbada8b80504ad` |
| questions / turns per mode / requests | 72 / 84 / 252 |
| byte identity vs serial | `six` 84/84, `controller` 84/84 |

These receipts are not shipped in `bench/receipts/`; the tables below are the
record of them.

### What the numbers say, before the tables

**On this subset DSpark loses, and that is the finding.** Fixed six runs at 0.741x serial
decode-only; the controller holds it to 0.959x. The cause is visible in one column: committed
tokens per verify is **2.71** for fixed six here, against 5.0-5.6 on the release matrix's own
fixtures. A six-row verify costs roughly four serial steps, so it needs about four committed
tokens to break even; at 2.7 it cannot.

That is not a defect in the controller, it is the question these prompts ask. Spec-Bench is short
open-ended prose — write a travel blog post, translate a sentence, answer a trivia question — and
the MTP drafter is far less sure of unconstrained prose than of code, tool calls or structured
text. The one category where the drafter is sure, `math_reasoning`, is the one category where
fixed six wins: 4.36 committed per verify and **1.195x** serial.

So the honest summary is two sentences. Where drafting does not pay, this controller turns a 26 %
loss into a 4 % one, which is what an admission controller is for. Where drafting does pay it
stays out of the way, and on `math_reasoning` it keeps 1.160x of fixed six's 1.195x.

The 4 % residual is the price of continuing to ask, and pricing declines inside the window is
what reduced it to 0.7 %: the controller reaches the long cooldown sooner, because three cheap
declines now close a losing window where before only three full losing verifies could.


### Pooled throughput, three boundaries

| boundary | mode | pooled t/s | mean per-question t/s | elapsed s | pooled ratio | per-question ratio | elapsed ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| server end-to-end | `serial` | 29.45 | 27.88 | 415.2 | — | — | — |
| server end-to-end | `six` | 22.39 | 23.43 | 546.1 | 0.760x | 0.840x | 1.315x |
| server end-to-end | `controller` | 28.45 | 27.27 | 429.8 | 0.966x | 0.978x | 1.035x |
| decode-only | `serial` | 31.90 | 31.89 | 383.3 | — | — | — |
| decode-only | `six` | 23.63 | 26.53 | 517.4 | 0.741x | 0.832x | 1.350x |
| decode-only | `controller` | 30.58 | 30.87 | 399.9 | 0.959x | 0.968x | 1.043x |
| client wall | `serial` | 29.32 | 27.66 | 417.1 | — | — | — |
| client wall | `six` | 22.34 | 23.31 | 547.3 | 0.762x | 0.843x | 1.312x |
| client wall | `controller` | 28.37 | 27.14 | 431.0 | 0.968x | 0.981x | 1.033x |

### Per category — server end-to-end, pooled t/s

| category | serial | six | controller | six vs serial | controller vs serial | controller vs six |
|---|---:|---:|---:|---:|---:|---:|
| `mt_bench` | 29.75 | 19.36 | 27.86 | 0.651x | 0.937x | 1.439x |
| `translation` | 19.07 | 18.08 | 18.93 | 0.948x | 0.993x | 1.047x |
| `summarization` | 29.87 | 22.98 | 28.07 | 0.769x | 0.940x | 1.222x |
| `qa` | 30.19 | 22.58 | 28.24 | 0.748x | 0.935x | 1.251x |
| `math_reasoning` | 28.10 | 33.23 | 32.36 | 1.183x | 1.151x | 0.974x |
| `rag` | 31.74 | 26.65 | 30.47 | 0.840x | 0.960x | 1.143x |

### Per category — decode-only, pooled t/s

| category | serial | six | controller | six vs serial | controller vs serial | controller vs six |
|---|---:|---:|---:|---:|---:|---:|
| `mt_bench` | 31.90 | 20.13 | 29.67 | 0.631x | 0.930x | 1.474x |
| `translation` | 31.88 | 27.71 | 29.91 | 0.869x | 0.938x | 1.079x |
| `summarization` | 31.78 | 24.09 | 29.75 | 0.758x | 0.936x | 1.235x |
| `qa` | 32.05 | 23.40 | 29.63 | 0.730x | 0.925x | 1.266x |
| `math_reasoning` | 31.99 | 38.22 | 37.11 | 1.195x | 1.160x | 0.971x |
| `rag` | 31.74 | 26.65 | 30.47 | 0.840x | 0.960x | 1.143x |

### Per category — client wall, pooled t/s

| category | serial | six | controller | six vs serial | controller vs serial | controller vs six |
|---|---:|---:|---:|---:|---:|---:|
| `mt_bench` | 29.69 | 19.34 | 27.83 | 0.651x | 0.937x | 1.439x |
| `translation` | 18.85 | 17.90 | 18.73 | 0.950x | 0.994x | 1.046x |
| `summarization` | 29.67 | 22.92 | 27.99 | 0.772x | 0.943x | 1.221x |
| `qa` | 30.12 | 22.54 | 28.17 | 0.748x | 0.935x | 1.250x |
| `math_reasoning` | 28.04 | 33.14 | 32.27 | 1.182x | 1.151x | 0.974x |
| `rag` | 31.30 | 26.48 | 30.27 | 0.846x | 0.967x | 1.143x |

### Committed tokens per verify, measured

| category | six | controller |
|---|---:|---:|
| `mt_bench` | 2.305 | 2.575 |
| `translation` | 3.165 | 2.864 |
| `summarization` | 2.803 | 2.966 |
| `qa` | 2.660 | 2.744 |
| `math_reasoning` | 4.360 | 4.544 |
| `rag` | 3.115 | 3.352 |
| **pooled** | **2.714** | **3.244** |

### The controller's own ledger

| | six | controller |
|---|---:|---:|
| verify calls | 4503 | 1043 |
| committed tokens | 12222 | 3383 |
| attempts | 4503 | 1043 |
| serial rows | 5 | 2547 |
| skipped (cooldown) steps | 0 | 7513 |
| judged windows | 0 | 337 |
| backoff events | 0 | 220 |
| losing cycles | 0 | 627 |
| requests with >=1 backoff | 0 | 71 |

| category | controller backoff events |
|---|---:|
| `mt_bench` | 85 |
| `translation` | 3 |
| `summarization` | 48 |
| `qa` | 43 |
| `math_reasoning` | 16 |
| `rag` | 25 |

## Licensing of the dataset

The Spec-Bench checkout's `LICENSE` and its `Readme.md` are hash-bound into
`identity.json`. That repository's licence is **not** a claim that every
underlying source-dataset text carries the same terms: the dataset text stays in
the caller's checkout, and the upstream source attribution and terms of each
constituent dataset remain applicable. Nothing from it is copied into this
repository.
