# DSpark speculative decoding for DeepSeek V4.1 Flash

DSpark is the drafter DeepSeek ships inside the V4.1 Flash checkpoint as the
`mtp.*` tensors. This branch converts it to a ds4 support GGUF, runs its forward
on Metal, and verifies its proposals against the backbone in one batched step.

It is **opt-in**. Without `--dspark` no drafter file is opened and the boot line
reads exactly the same resident total. Every DSpark arm in this release is
byte-identical to upstream `bd66c40`: fixed six and the controller change how many
tokens are committed per cycle, not which tokens.

Read [V41_M3ULTRA.md](V41_M3ULTRA.md) section 2 first for whether it will help
your workload. The short version: it depends on the corpus, the break-even is
about 3.76 committed tokens per cycle, and on one of the two real agent
transcripts measured a fixed six-row draft **loses 19 %**. The controller exists
for exactly that case.

---

## 1. Build the support GGUF

No drafter weights are redistributed here. Build one from your own copy of the
checkpoint.

The `mtp.*` tensors live in exactly three safetensors shards of
`deepseek-ai/DeepSeek-V4.1-Flash` (shards 44, 45 and 46 of 48, holding 801 / 798 /
802 `mtp.*` tensors). No other tensor in those shards is `mtp.*`-prefixed and no
`mtp.*` tensor lives outside them, so ~7.4 GiB of shards plus `config.json`, the
tokenizer files and the index is all the converter needs — the 550B backbone is
never loaded.

```
python3 gguf-tools/deepseek41_dspark_quantize.py \
    --hf   /path/to/DeepSeek-V4.1-Flash \
    --out  DeepSeek-V4.1-Flash-DSpark-Q4.gguf \
    --quant q4 --threads 16
```

The default `q4` recipe is the one every number in this release was measured on:

| component | tensors | type | bytes | GiB |
|---|---:|---|---:|---:|
| routed experts (3 x 128 x w1/w3/w2) | 9 | Q4_K | 7,644,119,040 | 7.119 |
| attention projections (`q_a`, `q_b`, `kv`, `output_a`, `output_b`) | 15 | Q8_0 | 403,587,072 | 0.376 |
| shared experts (3 x w1/w3/w2) | 9 | Q8_0 | 112,803,840 | 0.105 |
| `main_proj` (stage 0 only, 5120x15360) | 1 | Q8_0 | 83,558,400 | 0.078 |
| Markov heads (`markov_w1`, `markov_w2`) | 2 | Q8_0 | 70,328,320 | 0.065 |
| router (`ffn_gate_inp`, `exp_probs_b`, `exp_probs_b_vl`) | 9 | F32 | 7,867,392 | 0.007 |
| mHC `fn` / `base` / `scale` | 18 | F16 + F32 | 5,898,888 | 0.005 |
| norms and attention sinks | 17 | F32 | 186,112 | 0.000 |
| confidence head projection | 1 | F32 | 21,504 | 0.000 |
| **payload** | **81** | | **8,328,370,568** | **7.756** |
| alignment padding + header | | | 566,904 | 0.001 |
| **file** | | | **8,328,937,472** | **7.757** |

Routed-expert arithmetic, exactly: 3 stages x 128 experts x
(2 x 2304 x 5120 + 5120 x 2304) = 13,589,544,960 parameters; Q4_K is 144 B per
256 values, so 13,589,544,960 x 144 / 256 = 7,644,119,040 B.

Other recipes exist (`--quant q8`, `--quant q2`, `--quant f16`) and were swept.
**None beat `q4` where it matters.** True F16 is unreachable anyway: ds4's binder
accepts only Q8_0/IQ2_XXS/Q2_K/Q4_K/Q5_K/Q6_K/MXFP4 for a routed expert tensor and
every routed MoE kernel dispatches on that set, so Q8_0 is the precision ceiling
for the 92 % of the file that is routed experts. The Markov head stays Q8_0 in
every recipe because the runtime refuses anything else.

Validate the file with `gguf-tools/deepseek41_validate_gguf.py`.

## 2. What the drafter is

Three V4.1 blocks with a private 128-slot sliding-window KV ring, a rank-256
factorised bigram (Markov) head, and a confidence head. It routes **3 of 128**
experts where the backbone routes 6 of 384, which is why the file carries explicit
`deepseek41.dspark.n_routed_experts` / `.num_experts_per_tok` /
`.expert_feed_forward_length` metadata: the loader must read the drafter's shape
from those and never from the backbone's.

It has **no token embedding and no output head of its own** — the reference ties
them to the backbone — so the support file must not carry `token_embd` or
`output`, and the runtime reads them from the main model. It also has no
`hc_head_*` tensors, unlike the V4 DSpark checkpoint: the final collapse uses the
`ffn_pre` carried out of stage 2. Tensor names otherwise mirror V4's DSpark
support file so ds4's binder is shared.

**The draft block is five positions and non-causal.** Running a narrower pass is a
different computation, not a cheaper one, so the drafter always proposes five.
What varies is how many rows the verifier admits.

The Markov chain runs on the CPU: Metal has no fused markov-argmax kernel, the
chain is 176 MB against a 22-23 GB verify, and it is serial by construction.

## 3. Run it

```
DS4_V41_DSPARK_DECODE_READY=1 \
./ds4-server -m DeepSeek-V4.1-Flash-Q4.gguf \
    --mtp-model DeepSeek-V4.1-Flash-DSpark-Q4.gguf --dspark \
    -c 131072 --host 127.0.0.1 --port 8095
```

- `--dspark` requires `--mtp-model`.
- `--dspark` is admitted only when `DS4_V41_DSPARK_DECODE_READY=1` is set. Without
  it the drafter can still be validated, bound and inspected — `--mtp-model` alone,
  or `--inspect` — but decode stays refused. Making `--dspark` imply the gate is
  queued as a production change with its own re-gate window; until then, set it.
- **The admission controller is on by default** under `--dspark`, and needs no
  configuration file: it prices itself from the wall time it measures on the
  request it is serving. Section 4 is the policy and its levers.
- For sampled traffic add `--default-top-p` / `--default-min-p` so clients that
  send no sampling parameters inherit the deployment's published settings.
  Section 5.
- `--ssd-streaming` is not compatible with `--mtp-model`.
- The drafter costs **7.231 GiB of wired residency**, measured. The boot line does
  not show it: both binaries print `resident model 294.14 GiB ... = 302.38 GiB
  planned` with and without `--dspark`. Watch wired pages instead.

Verify width is `dspark_verify_rows`: `0` means the full six-row block, and `2..6`
sets an explicit cap.

### The width guard

`dspark_verify_rows` **refuses 1 and anything above 6**, keeping the previous
value, and the cycle refuses rather than running a different width silently:

```
HTTP 400  dspark: verify rows must be 0 (full block) or 2..6; use serial for width 1
```

This is a scar. Width 1 was rejected by the lever setter but the debug HTTP paths
ignored the rejection, so arms asking for width 1 measured the full six-row block
and reported it as k=1. Those arms were voided and re-run, and both paths now
refuse loudly. Width 1 is the serial path; ask for serial.

Four-row verify was measured as its own shape rather than interpolated, and it
lost: 0.796x against six rows' 1.164x, because verify barely falls (124.6 vs
108.5 ms) while committed tokens per cycle drop 4.37 to 3.38. **Widths are jagged.
An unmeasured width is unavailable, not estimated.**

### EOS handling

`dspark_excl_eos` (on in the production set) makes the speculative path use the
same EOS policy as serial. The serial control calls `argmax_excluding(eos)`; the
DSpark cycle originally took a plain argmax and broke its loop on EOS. With the
lever on, the cycle's row argmax skips the EOS id and the generate loop does not
break on it. Without it the two paths are not generating the same text and a
throughput comparison between them is meaningless.

## 4. The admission controller

`--dspark` means *drafter loaded and this controller on*. It is the
**windowed cost-feedback controller from this project's GLM-5.3-Flash branch** —
`ds4_dflash_adaptive.h` on its DFlash2 lane — ported to V4.1 as
`ds4_ds41_dspark_adaptive.h`, with its confidence admission
(`dflash_adaptive_prefix`) and its drafter fault latch (`ds4_dflash_fault.h`,
ported as `ds4_ds41_dspark_fault.h`). The design credit is that branch's; the
port and its measurement on V4.1 are this branch's.

### The policy

**Propose by default.** Every eligible greedy position attempts the full trained
six-row block. There is no position gate and no per-cycle prediction. Two things
dial it back: a per-position confidence floor that can decline a cycle before it
reaches the target, and loss accounting, which backs off for a bounded number of
tokens.

- **Confidence admission.** Per-position confidence is the normalised top
  probability of the post-Markov proposal row. It is already on the host after
  the Markov pass, so scoring it costs no GPU work. The admitted prefix is the
  longest run of drafted positions from the anchor at or above `p_min`, default
  **0.75** — GLM's value verbatim. The admitted prefix *gates* the cycle; it
  never shortens it.
- **Decline.** A proposal whose admitted run is shorter than `min_draft` is
  declined before the verify: the cycle pays the drafter and one serial step
  instead of a six-row target pass. `min_draft` is **3**, which is GLM's
  threshold re-expressed for a different block — GLM declines below 4 of its 7
  draft positions (0.57), V4.1 has 5, so 3 of 5 (0.60) carries it across.
- **A decline is a chosen call.** It takes a slot in the three-attempt window:
  its wall is the drafter plus the serial token it still has to take, its
  consumed rows are 1, so its net is a straight loss of the drafter cost, and
  three of them close a losing window and arm the cooldown. What a decline is not
  is a rejection — it never asked the target to check anything — so
  `losing_cycles` counts verified calls only. That asymmetry is the whole of the
  accounting, and getting it wrong is expensive in both directions: an earlier
  build of this port kept declines out of the window entirely and measured
  0.816x serial on prose, because nothing then throttled probing on content where
  every proposal declines. With declines priced, prose reads 0.988x and 218
  declines become 18.
- **Entry wait.** A request decodes its first 16 consumed tokens serially, so the
  first window is judged against a *measured* serial step rather than a guess.
  `DS4_DS41_DSPARK_MIN_SERIAL_TOKENS` moves it, clamped to 0..64.
- **Measured serial cost.** The serial step is the **median of the last nine**
  measured serial tokens — a median, so one jittery token cannot move a decision.
  The full cycle also carries an EMA, but only for reporting.
- **A window is three complete attempts.** Each contributes
  `net = wall_ms - consumed * serial_ms`; negative is a win.
- **Judgement.** A window whose summed net is below zero engages and clears any
  cooldown. A window that is not backs off.
- **Backoff ladder.** 16, then 32, 64, 128 consumed serial tokens
  (`16 << (bad_run-1)`, `bad_run` capped at 4). The cooldown is spent in consumed
  serial tokens, so a losing request spends at most 128 serial tokens before it
  tries again. **It never latches off.**
- **Reasoning.** Reasoning spans decode serially by default; leaving one clears
  the cooldown, because evidence gathered inside a reasoning span does not price
  the span after it.
- **Evidence reset.** When the session's prefix stops being an extension of what
  it was, the serial window, the cycle EMA and the open window are dropped. The
  cooldown survives: it is request policy, not measured evidence.

### What was not ported from GLM, and why

1. **No width search.** GLM's DFlash2 searches a verify width and keeps a
   per-width cost table. V4.1 verifies the full trained block, so there is one
   width and the table would have one row. The width machinery — `n_min`,
   `n_max`, `n_start` and the per-width costs — is not ported rather than ported
   and pinned.
2. **No ACK guard.** GLM prices a window only once the caller confirms the rows
   it consumed. V4.1's step commits inside itself and returns the advance it just
   made, so the guard could never fire; it is documented in the header rather
   than carried as unreachable state.

**Confidence admission used to be on this list and is not any more.** An earlier
revision of this document said it was deliberately left out because reading the
signal back would cost a GPU readback inside the proposal. That was wrong about
where the signal lives — the post-Markov proposal row is on the host already —
and the port is now complete, `p_min` and the decline accounting included.

### Levers

| lever | environment variable | default | what `=0` does |
|---|---|---:|---|
| `dspark_adaptive` | `DS4_DS41_DSPARK_ADAPTIVE` | 1 | propose everywhere, never back off — **the fixed-six arm** |
| `dspark_reasoning_serial` | `DS4_DS41_DSPARK_REASONING_SERIAL` | 1 | draft inside reasoning spans too |
| `dspark_serve` | `DS4_DS41_DSPARK_SERVE` | 1 | serial decode; the drafter stays loaded |
| `dspark_verify_rows` | `DS4_V41_DSPARK_VERIFY_ROWS` | 0 | 0 is the full block; 2..6 caps it; 1 and >6 are refused |
| — | `DS4_DS41_DSPARK_MIN_SERIAL_TOKENS` | 16 | 0 removes the entry wait (0..64) |
| — | `DS4_DS41_DSPARK_P_MIN` | 0.75 | per-position confidence floor; 0 admits every position (0..1) |
| — | `DS4_DS41_DSPARK_MIN_DRAFT` | 3 | admitted prefix below this declines the cycle; 0 never declines |
| — | `DS4_DS41_DSPARK_FAIL` | unset | `drafter_once` / `state_save` / `after_verify` inject a drafter failure at that exit, for testing the latch |

The last four are documented `=` overrides read at the start of every request,
not levers on the `/debug/levers` surface. They change which cycles are worth
verifying, never what a verified cycle emits.

### There is no calibration file

The design this replaced was a calibrated-window controller: Wilson acceptance
bins and per-interval cost records, built offline and shipped as
`dspark/calib-ds41-code.txt`, with a `DS4_DS41_DSPARK_CALIBRATION` override and a
refusal to start when neither could be read. **All of it is deleted** — the
levers `dspark_controller`, `dspark_controller_confidence`,
`dspark_controller_widths` and `dspark_admit_generalize`, the calibration loader,
the environment variable and the file itself.

It is worth saying why, because the replacement looks less clever. A fixed
acceptance table cannot cover the positions real prompts land at: the shipped
file held three 512-position windows and production chat prompts arrived at
8,745, 23,457, 57,976, 62,537 and 128,679 tokens, so the controller made **zero
attempts on every one of them**. Generalizing it with a donor prior admitted, won
on the cycles it ran, and then stopped the request on its own predeclared loss
allowance — 31.9 t/s at 8,745 tokens against a fixed six at 41.1 on the same
kernels. A controller that may choose six every cycle has fixed six as its floor
where six wins and serial as its floor where six loses; one that can latch off
has neither floor. That is a design fault rather than a tuning error, so the
design went.

If you are upgrading from a build that had one, delete
`dspark/calib-ds41-code.txt` and unset `DS4_DS41_DSPARK_CALIBRATION`. Both are
ignored.

### The drafter fault latch

A drafter failure is either recoverable or it is not, and the latch
(`ds4_ds41_dspark_fault.h`, ported from the GLM branch's `ds4_dflash_fault.h`) is
where the difference is decided. It is session-scoped: it belongs to the
allocated session, not to its drafter state, its conditioning generation or its
request ledger, and only destroying the session clears it.

- **Drained** — the failure happened up to the verify batch, so the target's KV,
  position and checkpoint are untouched. The step falls through to the serial
  path, the session latches, the request finishes normally, and every later
  request on that session skips the drafter. The faulted request and the one
  after it are byte-identical to serial.
- **Undrained** — the failure happened after the batch, when the target has
  already consumed rows the session cannot un-feed. Serial decode from here would
  answer from a state no serial decode would have reached, so there is no correct
  fallback and the session refuses: `dspark: unsafe drafter failure requires a
  new session`. It used to return an empty 200 to every later request instead,
  which said nothing about why.

Counters — `{disabled, unsafe, requests, attempts, failures, skips}` — ride in
the `ds4_dspark` usage block, and a `ds4: dspark fault event=...` line mirrors
the GLM branch's. `DS4_DS41_DSPARK_FAIL=drafter_once|state_save|after_verify`
injects each exit; the hook is read from the environment at process start, so a
server not started with it cannot reach any of this.

### What it is worth

Full tables, with the serial and fixed-block controls beside them, are in
[V41_M3ULTRA.md](V41_M3ULTRA.md) section 5. In one paragraph: across nine
content fixtures it runs from 0.979x to 1.576x its own serial control where a
fixed block runs from 0.588x to 1.597x — the controller's floor is serial and the
block's is not; on a held-out screen of real agent traffic it runs at 1.056x
serial whole-turn and 1.318x through the answer phase, 14 of 14 arms
byte-identical; and on a Spec-Bench subset — a workload where drafting loses
outright — it gives up 0.7 % against the block's 26 %, while keeping 1.130x of
the block's 1.192x on `math_reasoning`. **Run the shipped default if you do not
know your workload.** Run `dspark_adaptive=0` only on traffic you have measured,
like `sql`.

### Telemetry

`usage.ds4_dspark` on the chat path, per request: `dspark_cycles`,
`dspark_committed`, `dspark_serial_rows`, `dspark_tokens_per_cycle`,
`dspark_attempts`, `dspark_skipped_steps`, `dspark_windows`, `dspark_backoffs`,
`dspark_losing_cycles`, `dspark_declines`, `dspark_serial_ms`, `dspark_cycle_ms`,
`dspark_net_ms`, and the latch's `dspark_fault_latched`,
`dspark_fault_requests`, `dspark_fault_attempts`, `dspark_fault_failures`,
`dspark_fault_skips`.

`dspark_declines` counts cycles refused by the confidence gate and
`dspark_losing_cycles` counts verified cycles that lost; the two are disjoint by
construction, and a request whose declines are high while its losing cycles are
zero is content the drafter cannot help with, not a controller that is mispricing
its attempts.

`dspark_committed / dspark_cycles` is committed tokens per verify from the real
consumed count, including a terminal partial block — a serving number, not a
bench-only one. `dspark_net_ms` is the request's cumulative wall minus rows times
the measured serial step: negative is the controller's measured win on that
request.

The ledger is **request-local**, opened at decode start and read at decode end.
Its predecessor's counters lived in rewindable session state and were reset by
prefix reconstruction, which is why two byte-identical runs used to report
different attempt and serial-row counts. `dspark_tokens_per_cycle` also used to
subtract serial rows from a count that already excluded them, underflowing the
unsigned to `1.84e19`. Both are fixed.

Extra host memory: policy state only, well under a kilobyte per session. No new
GPU buffers.

## 5. Sampled decode at temperature above zero

Positive-temperature V4.1 decode is **exact speculative sampling, always**. The
opportunistic mode — commit the drafter's greedy proposals and let the server
sample each boundary token — is deleted for this lane, because it does not
reproduce serial sampling's output distribution and so could never be a default.
V4's own path is untouched.

The drafter is deterministic: its proposal is an argmax, so `q(x) = 1` and
`min(1, p/q)` reduces to accepting the drafted token `x_j` with probability
`p_j(x_j)` under the target's **filtered** distribution for that row, and on
rejection sampling the residual — `p_j` with `x_j` removed and renormalised. The
last row of a block carries no proposal and is sampled outright: the bonus token.
A cooled-down serial step samples from the request's own distribution rather than
argmax.

Two details decide whether that is correct, and both are easy to get wrong:

1. **The filters — temperature, `top_k`, `top_p`, `min_p` — are applied to the
   original row logits, and the support is fixed BEFORE the drafted token is
   masked.** Masking first and filtering afterwards re-derives the support from a
   distribution the rejection already changed.
2. **The serving loop re-samples the frontier token from the row logits the step
   leaves in the session.** Under greedy that reproduces the same argmax. Under
   sampling it would discard the rejection residual and redraw from `p_j`, which
   biases the slot towards the drafted token: `P(x)` becomes `p(x)(2 - p(x))`
   instead of `p(x)`. The cycle therefore hands the caller a one-hot row carrying
   the decision the acceptance rule already made.

`ds41_accept_or_residual()` is the whole rule, and `tests/test_sampling.c` drives
**that shipped function**, not a copy of it, through a chi-square test against the
target's filtered distribution for every drafted token index, at `top_p`
1.0 / 0.95 / 0.5 crossed with `min_p` 0 / 0.05, 200,000 draws each.

**What is claimed.** Determinism: the same seed and the same configuration give
the same bytes. Distribution correctness: by the chi-square test above.
**Byte identity is NOT claimed at temperature above zero** and is not expected —
exact speculative sampling reproduces the target's distribution, not its
particular draw, and the two paths consume the RNG in different orders. Byte
identity is a greedy claim only.

Acceptance is lower under sampling than under greedy, which is what accepting
with probability `p(x)` instead of deterministically must cost: 4.5-5.1 committed
tokens per cycle against 5.0-5.6. Measured ratios against serial at temperature
1.0 / `top_p` 0.95 are in [V41_M3ULTRA.md](V41_M3ULTRA.md) section 4.

### Server-side sampling defaults

```
    --default-top-p 0.95 --default-min-p 0
```

`--default-top-p` takes a value in `(0, 1]` and `--default-min-p` a value in
`[0, 1)`. They supply a deployment's choice for requests that do not name the
knob; a request that names it still wins. They exist because a published
benchmark setting belongs in the deployment rather than in every client.


## 6. Tests

```
make -j4 all tests/test_ds41_controller tests/test_ds41_dspark_adaptive \
    tests/test_ds41_prefix tests/test_sampling
./tests/test_ds41_controller
./tests/test_ds41_dspark_adaptive
./tests/test_sampling
./tests/test_ds41_prefix --self-test-logits
./tests/test_ds41_prefix MAIN.gguf SUPPORT.gguf PROMPT.txt 8192 6 129 TOLERANCE
```

`test_ds41_dspark_adaptive` is ported alongside the controller from the GLM
branch's `tests/test_dflash_windowed.c` with V4.1 constants. It covers the
backoff ladder, the median's robustness to a jittery serial token, the entry
wait, the reasoning gate and its clear, an evidence reset keeping the cooldown,
the admitted-prefix rule — the longest run at or above `p_min`, and no admission
at all below `min_draft` — and the decline accounting: three declines take a
window's three slots, arm the first backoff and leave `losing_cycles` at zero.

`test_sampling` carries the chi-square block for section 5: it drives the shipped
`ds41_accept_or_residual()` through 200,000 draws per configuration and compares
the recovered distribution against the target's filtered one. It builds under
`-DDS4_NO_GPU`; that build had been broken since the V4.1 serving path landed,
because the serving helpers sat outside the Metal guard while the state types
they use sat inside it, and it is fixed.

`test_ds41_prefix` separates a final-logit tolerance from exact prefix state,
reports per-row argmax and a ULP count against a predeclared ceiling, and runs the
state checks regardless of the logit verdict. Its predecessor died at a bitwise
logit compare before any state assertion ran, which is why the separation exists.

Depth and compressed-boundary identity was checked at P = 768, 1020, 1024, 1025,
1026, 1027 and above: **10 of 10 identical, zero greedy divergences**, six rows
against the serial control for 128 tokens at each length with a fresh prefill.

## 7. Limits

- **Verify cost is the binding constraint.** A six-row cycle is ~120 ms against a
  ~31.8 ms serial step. Six reformulations of the six-row routed gate/up matmul
  were built and all lost; see
  [CHANGES-V41.md](../CHANGES-V41.md) section 4.
- Taking 11 ms off the verify would move break-even from ~3.76 to ~3.42, which
  would still leave `agent2` at about 0.89x. **No kernel result in reach makes
  drafting pay on that fixture. Only declining to draft does.**
- `--dspark` still needs `DS4_V41_DSPARK_DECODE_READY=1` in the environment.
  Making the flag imply the gate is queued as a production change.
- **Widths 2..5 are unmeasured and therefore unavailable, not estimated.** Only
  the full trained block and the four-row shape were ever measured, and four rows
  lost. `dspark_verify_rows` will cap the block if you ask it to; nothing selects
  a width for you.
- **`p_min` is a constant, not a fitted value.** 0.75 is carried over verbatim
  from the branch this was ported from; it was not tuned on V4.1 traffic, and
  `min_draft` is GLM's fraction re-expressed for a five-position block rather
  than a threshold measured here. Both are overridable and neither is claimed to
  be optimal.
- **Admission is still coarse in time.** Wall time is judged three attempts at a
  time. A workload whose acceptance swings faster than a three-attempt window can
  follow will be served worse than one that could see it coming, and the
  confidence gate does not fix that: it prices a position, not a trend.
- **Drafting loses outright on some real workloads.** On a Spec-Bench subset the
  controller runs at 0.993x serial and a fixed block at 0.739x. The controller
  bounds the loss; it does not turn it into a gain. See
  [SPEC-BENCH-V41.md](../bench/SPEC-BENCH-V41.md).
- **Byte identity is a greedy claim.** At temperature above zero what is
  certified is determinism at a fixed seed and the chi-square test of the
  acceptance rule, not identical bytes. Section 5.
