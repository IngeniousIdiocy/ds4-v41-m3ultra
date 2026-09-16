# Reproducing the V4.1 Flash M3 Ultra numbers

> This guide preserves the original release protocols. For the September 16
> code, default-on optimizations, new production-path tests, numerical results,
> and experiment records, start with [the update](RELEASE-V41-20260916.md).

Everything in [V41_M3ULTRA.md](V41_M3ULTRA.md) comes from one measurement session
on one machine. This file gives the bound identities, the fixtures, and the exact
invocations, so a reader can tell what was measured and run it again.

Machine: Mac Studio, M3 Ultra, 512 GB unified memory, macOS. One model process at
a time, GPU sampled idle before every arm, the server killed between builds and
between binaries.

---

## 1. Bound identities

Six binaries. Which one stands behind which table in
[V41_M3ULTRA.md](V41_M3ULTRA.md) is stated table by table in
[../bench/RELEASE-EVIDENCE-V41.md](../bench/RELEASE-EVIDENCE-V41.md) and bound in
[../bench/v41-manifest.json](../bench/v41-manifest.json) under `readme_tables`.

| | role | commit | `ds4-bench` sha256 | `ds4-server` sha256 |
|---|---|---|---|---|
| **UPSTREAM** | stock antirez/ds4 | `bd66c402070042bf0a79ad6ece8242de4c93680c` | `f956d4245e674abc...` | `0c9b9a944d86058a...` |
| **THIS BRANCH** | kernel matrix, TTFT, memory | `b977a33f062e98808373369e718a66b134e1f6ab` | `9a938e5319d0c4c5...` | `88a14ce25f41aa7e...` |
| **THIS BRANCH** | depth sweep, original selection path | `c050b75c324fc6f1f6907d240cd8c8a65495aae0` | not built | `5323ade0bdf1abeb...` |
| **THIS BRANCH** | depth sweep, shipped selection defaults | `9f432d87b9942b1e0069b2751b1a77636436c63e` | not built | `86da6c8fcd95ced2...` |
| **THIS BRANCH** | admission and agent screens, Spec-Bench | `afe84d1770a126b5784a615f3534f2b610831778` | not built | `cfeb2c4d186d6000...` |
| **THIS BRANCH** | release head, standing byte gates | `03eb931` | `03eb93182d0755fe22580b0165808caa89bd1739` | not built | `a671b6eff9c147fa...` |

The `ds4-server` digests in full:

```
c050b75  5323ade0bdf1abeba925e0e98b4d3e86c98a4030721ce6163c7a26bc35214c20
9f432d8  86da6c8fcd95ced20e794483337d8800bd3f919e4bee73a3807d633fb47232a1
afe84d1  cfeb2c4d186d6000e2687de88df228272eaff7d86a28ef16405a9480f3f188b6
f9dc5f9  463b186a2b96a7813fb0391e370fd1ad6b28ac348579a4dd71bbada8b80504ad
03eb931  a671b6eff9c147faf0413bd36a4955e47fdd8f7f2235a6d5815c49486c11a28c
```

`f9dc5f9` is listed because an earlier pass of the admission sweeps, the
sampled-decode sweep and the Spec-Bench subset ran on it, before confidence
admission and before declines were priced inside the admission window. Those
numbers are kept in the manifest under `admission.superseded_measurements` and
`spec_bench.superseded_run`; none of them is quoted in V41_M3ULTRA.md, and the
receipts shipped in `bench/receipts/` are the later run's.

The kernel matrix, the TTFT arms and the memory measurements were taken on the
kernel head, where every arm named its levers explicitly rather than relying on a
default, so a later lever-default change moves no number in them. The prefill and
matmul path is unchanged between that commit and the release head; the decode
selection chain is not, which is why the depth rows come from the build whose
selection defaults are the shipped ones. The release head's own build re-ran the
production 8k/512 and 32k/256 byte gates and passed them.

Both worktrees were clean at measurement time (`git status --porcelain` empty
apart from an untracked models directory in the upstream tree, which holds no
source). Upstream was clean-rebuilt for this matrix because no binary hash for
`bd66c40` had ever been recorded; reusing an unidentified binary was not an
option.

Model: a Q4 conversion of `deepseek-ai/DeepSeek-V4.1-Flash`,
**518,596,067,328 bytes**.

Drafter, DSpark arms only: the support GGUF built by
`gguf-tools/deepseek41_dspark_quantize.py --quant q4`, **8,328,937,472 bytes**.
See [DSPARK-V41.md](DSPARK-V41.md) section 1.

> **Partial.** The GLM branch ships compact JSON receipts under `bench/receipts/`
> with staging labels and source-receipt hashes. For V4.1 that directory now holds
> the Spec-Bench subset run — `bench/receipts/spec-bench-v41/2026-09-14-glmctl/`, one
> record per request with the SHA-256 of each request body and each answer — but
> not the section 1 and 1a arms. For those, the identities above and the fixture
> hashes below are the binding, and they are stated inline in every table rather
> than referenced.

## 2. Fixtures

| fixture | bytes | sha256 (first 16) | what it is |
|---|---:|---|---|
| `code.txt` | 30,480 | `2df8394d464b8fff` | C source plus the GGUF Python tooling from this repository, held at exactly the control's prompt length |
| `code-32k.txt` | 143,287 | `f7bff4f433bdd07e` | the same, extended to reach a 32,768-token prefix |
| `code-62k.txt` | 300,002 | `73ecc0e82eab2f7d` | the same, extended to reach a 62,000-token prefix |
| `agent.txt` | 37,064 | `70961a7ead99cb9c` | a coding-agent transcript: a real agent system-prompt capture followed by a synthetic tool-call / tool-result exchange in the model's own DSML tool-call format |
| `agent2.txt` | 162,902 | `57865c96a3138ce4` | a **different** real agent transcript from another project — tool calls, JSON tool inputs and results, short prose. Not a slice of `agent.txt`. Held out of every fit |
| `sql.txt` | 98,309 | `d993b7eea0268447` | a SQL-generation instruction extended into the natural continuation of the same task, 2,000 `INSERT` tuples |
| `prose-en.txt` | 32,616 | — | English technical prose: this repository's README and twelve files from `docs/` |
| 23.4k system prompt | 102,640 | `66a0dedf3774c749` | the agent system prompt used for the TTFT arms; the server renders it to 23,446 tokens (23,441 cached anchor plus 5 trailing turn markers) |
| 111-token append | 416 | `3b0457778d745912` | the user turn appended in the restored-plus-append TTFT arm |

**The three `code` fixtures are reproducible from this repository** — they are its
own C sources and Python tooling, truncated to the token length each gate wants.

**`agent.txt` and `agent2.txt` are not redistributed.** They are real transcripts.
Their shape is given above so a comparable fixture can be built: an agent system
prompt, then a sequence of assistant turns each containing a tool call in DSML
format followed by a tool result, with short prose between. The property that
matters for DSpark is that such text has *lower* draft acceptance than source
code, and `agent2` is the harder of the two — see section 4.

**A note on corpora.** An Italian-prose corpus was used for earlier gates and has
been withdrawn from the fixture set. It is not a control here and no number in
this release is measured on it. This matters because it is the reason an earlier
upstream 8k decode figure of 19.44 t/s, and the 1.62x paired with it, are struck:
those were measured on that corpus. On the fixtures above, upstream reads
16.570 / 16.600 / 16.630 t/s across three runs on two corpora.

## 3. How the references were generated

The identity references are decoded-text files produced by the **upstream
`bd66c40` binary** — not by a same-binary control — at each gate shape, on each
fixture. Every production arm is compared against the reference for its fixture,
byte for byte, on the decoded text.

```
ds4-bench -m <Q4 gguf> --metal --prompt-file <fixture> \
  --ctx-start <ctx> --ctx-max <ctx> --ctx-alloc <ctx+768> \
  --gen-tokens <gen> --show-output --csv <arm>.csv
```

run against the upstream binary, with `--show-output` capturing the decoded text.
`(ctx, gen)` is `(8192, 512)`, `(32768, 256)` or `(62000, 128)`.

For the 62k prefill arm the comparison is on logits, not text: the run generates
nothing (`--gen-tokens 0`) and dumps the frontier row, and the 129,280-float array
is compared as bytes.

```
ds4-bench ... --ctx-start 62000 --ctx-max 62000 --gen-tokens 0 \
  --dump-frontier-logits-dir <dir>
```

`--dump-frontier-logits-dir` does not create its directory. `mkdir -p` it first or
the run pays for the whole prefill and then dies.

The whole-file hash of a frontier dump differs between a `--gen-tokens 0` run and
a generating run because `--gen-tokens 0` allocates eight fewer context slots and
the `"ctx"` header records it. Compare the body, not the file.

## 4. Serial decode and cold prefill (V41_M3ULTRA.md sections 2 and 5)

One fresh process per arm, `DS4_METAL_MODEL_UNTRACKED=1`, quiet check before every
arm:

```
ds4-bench -m <Q4 gguf> --metal --prompt-file <fixture> \
  --ctx-start <ctx> --ctx-max <ctx> --ctx-alloc <ctx+768> \
  --gen-tokens <gen> --show-output --csv runs/<arm>.csv
```

with `(ctx, gen)` = `(8192, 512)`, `(32768, 256)`, `(62000, 128)`, and
`--gen-tokens 0` for the prefill-only arms.

`--debug-levers` is **not accepted by upstream `bd66c40` at all**. The first
attempt at giving both binaries identical server flags died on
`unknown option: --debug-levers`. Worth knowing before any future run tries it.

## 4a. The depth sweep (V41_M3ULTRA.md section 5)

Five depths — 8,192, 32,768, 62,000, 131,072 and 300,000 tokens — in three modes
each. A 300,000-token cold prefill costs 474 s on the production binary and 526 s
on upstream, so the whole point of this recipe is to pay each prefill **once**.

### The rule: one resident server, restore per depth

**Start one production server at the largest context the sweep needs and vary
`ctx_start` and `ctx_alloc` per request. Never boot per depth.**

```
DS4_V41_DSPARK_DECODE_READY=1 \
./ds4-server -m <Q4 gguf> --mtp-model <DSpark gguf> --dspark \
  --ctx 301056 --port 8199 --host 127.0.0.1 --debug-levers \
  --kv-disk-dir <kvdisk-depth> --kv-disk-space-mb 8192
```

301,056 = 300,000 + 1,056 of headroom. Then, per depth, one fixture-building
call that prefills cold and persists the post-prefill session to the KV disk —
**this call is the cold-prefill measurement** — followed by every other arm at
that depth restoring from that snapshot:

```
POST /debug/bench
{"path":"<fixture>","ctx_start":<depth>,"ctx_alloc":<depth+1056>,
 "gen_tokens":512,"fresh":true,"dspark":false,"levers":{ ... }}     # cold, timed

POST /debug/bench                                                    # restored
{"path":"<fixture>","ctx_start":<depth>,"ctx_alloc":<depth+1056>,
 "gen_tokens":512,"fresh":false,"dspark":<bool>,"levers":{ ... }}
```

Arm levers are the same three as section 5, and the same translation note
applies. Restore costs
5.9 / 15.2 / 16.1 / 46.6 / 104.0 ms at the five depths, so a restored arm is
free next to a prefill.

Two obstacles that make a literal single-server sweep impossible today, and both
are worth fixing before the next one:

- `/debug/bench` holds **four fixture slots** (`BENCH_FIXTURES = 4`) against five
  depths, so the fifth evicts one.
- `DS4_KERNEL_LEDGER`'s mode is **process-level**, so a ledger-instrumented arm
  cannot share a process with a production-shape arm.

The run behind the depth curve paid 13 model loads because of these, and an
earlier partial sweep paid 10 more. With both fixed the whole sweep is one
resident server per binary, five cold prefills, and everything else restored.

**Upstream has to be `ds4-bench`, one process per depth.** `bd66c40` has no
`/debug/bench` and does not recognise `--debug-levers`, so its column is five
separate processes, each paying its own cold prefill — which is also where the
upstream cold-prefill numbers come from.

### Fixtures for the two deep points

8k, 32k and 62k use `code.txt`, `code-32k.txt` and `code-62k.txt` as everywhere
else. 131k and 300k use `depth-mix.txt`, 1,731,960 bytes, sha256
`7f71336c946a8a30…`, which tokenises to 451,346 tokens on this binary. It is
**five repetitions** of the concatenation of five existing campaign fixtures —
an agent system prompt, `code-32k`, `code`, `agent` and `prose-en` — each
repetition separated by a distinct marker line. Nothing in it is generated, and
the retired Italian corpus appears nowhere in it.

**This is why the 131k and 300k acceptance numbers are not workload numbers.**
By the time the model reaches token 131,072 it has already seen the same text
once, and by 300,000 it has seen it four times, so drafts are accepted far more
often than on real traffic. Serial decode is corpus-insensitive, so the serial
curve stands; the DSpark arms at those two depths are quoted for **verify
milliseconds per cycle** and per-family cost only.

An independent reproduction exists on a different deep corpus,
`code-deep.txt` (1,737,938 bytes, sha256 `b3a3ece9d785e37b…`, pure C source):
every serial pair agrees with the `depth-mix` sweep to 0.5 % or better, on two
different days' processes, and its cold prefill at 300k agrees too (upstream
512.7 s, production 473.3 s against 526.04 and 474.22). The serial depth curve
is not a property of how `depth-mix.txt` was built.

### References and identity

The upstream arm at each depth **is** the reference:
`depth-{8k,32k,62k,131k,300k}-upstream-bd66c40.{text,csv,log}`, digests in
[../bench/v41-manifest.json](../bench/v41-manifest.json). Note that these are
512-token arms, so `depth-32k` and `depth-62k` are **not** the same references as
the `gate32k256` and `gate62k128` entries used in section 4 — different
generation lengths and, at 62k, a different fixture. `depth-8k` and
`gate8k512-code` are the same bytes, because that depth is the same fixture at
the same shape.

`bench/verify-v41-manifest.py --run-dir <dir>` matches every `*.text` a run
emits against all of these digests, so a depth arm landing on the wrong
reference is a failure rather than a silence.

### The costmap arms

The per-family attribution in CHANGES-V41.md comes from a second pass
with `DS4_KERNEL_LEDGER=2`, `DS4_KERNEL_LEDGER_BY_TG=1` and
`DS4_KERNEL_LEDGER_PAGES=192`, dumped through `/debug/bench`'s `ledger_dump`.
The ledger is reset after the prefix is restored and dumped immediately after
the decode loop, so the window is the decode and nothing else. Serial windows
are 16 generated tokens; six-row windows are 5-8 complete cycles.

Every arm must print a **100.000 % coverage certificate with `complete=YES`**.
The sampler refuses to print any timing field when coverage is incomplete, which
is the safeguard that exists because an earlier unpaged sampler silently dropped
45.7 % of a six-row cycle's passes while still printing totals, and every figure
taken from it had to be retracted.

Ledger mode 2 is **not** the production shape: it disables encoder batching, the
concurrent kv-norm task and the parallel-FFN overlap, and timing every pass costs
more again. Its numbers are for relative structure only and must never be mixed
with section 4's or section 5's.

## 5. DSpark committed throughput (V41_M3ULTRA.md section 5)

One resident server for all 24 arms. Arms interleaved, two reps each. Every lever
is named in every arm string; none is left to persist from the previous request.

```
DS4_V41_DSPARK_DECODE_READY=1 \
./ds4-server -m <Q4 gguf> --mtp-model <DSpark gguf> --dspark \
  --ctx 8960 --port 8199 --host 127.0.0.1 --debug-levers
```

```
POST /debug/bench
{"path":"<fixture>","ctx_start":8192,"ctx_alloc":8960,"gen_tokens":512,
 "fresh":<bool>,"dspark":<bool>,
 "levers":{"dspark_capture":1,"mtp_state_fix":1,"mtp_capture_warmup":1,
           "mtp_engram_rows6":1,"mtp_async_chunks":1,"prefill_decoder_suffix":1,
           "dspark_excl_eos":1,"dspark_force_reject":0,"dspark_draft_trace":0,
           "dspark_controller_widths":0,"verify_batch_core":0,
           "mtp_q8_pair6":1,"mtp_q8_stream6":1,"mtp_q8_stream6_mask":2,
           <arm levers>}}
```

`<arm levers>` is `"dspark_verify_rows":6,"dspark_controller":0` for fixed six,
`"dspark_verify_rows":0,"dspark_controller":1` for the controller, and the serial
control sets `"dspark":false`.

> **These lever names are the ones the measurement used, and two of them no longer
> exist.** The lever block above is reproduced as it was sent, because that is
> what makes the section auditable. The admission design was replaced in
> `cda2b9f` (see [DSPARK-V41.md](DSPARK-V41.md) section 4), and the arms
> translate like this:
>
> | arm | as measured | on the release head `03eb931` |
> |---|---|---|
> | serial | `"dspark":false` | `"dspark_serve":0` |
> | fixed six | `"dspark_verify_rows":6,"dspark_controller":0` | `"dspark_adaptive":0,"dspark_verify_rows":0` |
> | shipped default | `"dspark_verify_rows":0,"dspark_controller":1` | `"dspark_adaptive":1,"dspark_verify_rows":0` — the binary's own defaults |
>
> `dspark_controller`, `dspark_controller_widths`, `dspark_controller_confidence`
> and `dspark_admit_generalize` are gone, and so is the `dspark/calib-ds41-code.txt`
> calibration file and its `DS4_DS41_DSPARK_CALIBRATION` override. A request naming
> a lever that does not exist is refused rather than ignored, so an old lever block
> fails loudly instead of measuring the wrong thing. `dspark_reasoning_serial`
> (default 1) is new and was not a variable in these arms; every arm here is
> outside a reasoning span.

`--debug-levers` is required for `/debug/bench` and **must not be used on a
serving process**: each call holds its own full session, and a server driven this
way went from 389 MB at idle to 13 GB after 24 arms.

Harness numbers read about **1 t/s above** `ds4-bench` numbers on the same binary.
That is why every DSpark fixture carries its own serial control measured in the
same server, and why the DSpark table and the serial table are never mixed.

**Run one driver at a time.** An earlier pass of this section was corrupted when a
second copy of the driver was launched while the first was still alive: both
reached the same single server, the second server refused to start, and the two
lever sets interleaved, producing an arm labelled "six" that carried a controller
signature. Fifteen rows were purged and the section re-run from one verified
driver.

## 6. TTFT (V41_M3ULTRA.md sections 2 and 5)

One server per binary.

```
./ds4-server -m <Q4 gguf> [--mtp-model <DSpark gguf> --dspark] \
  -c 65536 --port 8199 --host 127.0.0.1 \
  --kv-disk-dir runs/ttft-kv-<binary> --kv-disk-space-mb 8192
```

The client is a stdlib socket that timestamps the moment the request body is on
the wire and the moment the first SSE `content_block_delta` carrying non-empty
text arrives. Greedy, thinking disabled, `max_tokens: 32`. The prompt goes in the
`system` field; the append arm adds the 111-token user turn.

`--kv-disk-dir` is emptied before the cold arm. **No `--debug-levers`** — it costs
about 24 GiB and is not the production shape.

**Forcing the restored path.** Each restored arm is preceded by an unrelated
four-token request, which leaves the live session holding 20 tokens. The next
request then logs `live kv cache miss live=20 prompt=23446 common=1` and takes the
disk path. Those displacement requests (327.24 ms and 186.79 ms) are recorded but
are not results.

What the server logs on each arm, which is the check that the right path ran:

| arm | binary | TTFT | server line |
|---|---|---:|---|
| cold | upstream | 35,769.47 ms | `chat ctx=0..23446:23446 prompt done` |
| cold | this branch | 31,082.86 ms | `prompt done 31.038s`, then `kv cache stored tokens=23441 ... size=161.26 MiB save=58.5 ms` |
| restored, fresh session | this branch | 229.34 ms | `kv cache hit ... load=22.5 ms`, `chat ctx=23441..23446:5 prompt done 0.161s` |
| restored + 111-token append | this branch | 869.43 ms | `kv cache hit ... load=16.6 ms`, `chat ctx=23441..23555:114 prompt done 0.805s` |

## 7. Memory measurement

Every server start was measured the same way: `vm_stat` immediately before the
process started, `vm_stat` plus `footprint -p` five seconds after it printed
`listening on`, and `footprint` again after the arms. Wired pages are 16,384 B
each.

| | upstream, ctx 65,536, no drafter | this branch, ctx 65,536, `--dspark` |
|---|---:|---:|
| boot line, planned | 302.38 GiB | 302.38 GiB, identical |
| static context buffers | 8,437.56 MiB | 8,437.70 MiB |
| wired pages before boot | 578,392 | 613,313 |
| wired pages at idle | 19,897,662 | 20,406,496 |
| **wired attributable to the process** | **294.789 GiB** | **302.020 GiB** |
| `footprint` at idle | 636 MB | 639 MB |

The difference, 473,913 wired pages, is the drafter: **7.231 GiB**, 93.2 % of the
drafter GGUF's own size. The boot line is identical with and without `--dspark`,
so it cannot be used to detect a drafter. Watch wired pages.

## 8. On the quiet check

A quiet check that samples GPU utilisation for ten seconds and fails on any sample
above 5 % will **false-negative on its own predecessor**: sampled immediately after
a `ds4-bench` exits, the first second catches that run's own work draining at
97 %, then reads 0 % on the other nine. That is a statement about the run just
finished, not the machine. Let 45 seconds pass before the check.

In this session the check also reported not-quiet before every server-bearing arm
for one reason only: it counts our own `ds4-server` as a model process. GPU
utilisation read 0 % on all five samples of every such check, with no other
inference process and no CPU consumer above 20 %. The `ds4-bench` arms read quiet
outright.

## 8a. Prefill checkpoints and KV eviction (V41_M3ULTRA.md sections 3 and 5)

Two of these need nothing but a checkout, and the third needs a server and a long
prompt.

**The eviction policy, at the store layer, no model and no GPU.** The ladder
selector and the eviction scoring are host code with their own tests:

```
make -j4 ds4_test
./ds4_test --server
```

`test_kv_ladder_survives_compaction_pressure` is the reproduction of the failure
the policy was written for: a live session's fresh ladder plus a stale idle-chat
anchor, a full disk, one eviction pass. It asserts the aged anchor is the victim
and both live rungs survive, and its fresh-anchor contrast shows age rather than
class is what drives that. It scores against real wall-clock time, so the
half-life is exercised rather than mocked. Two other assertions in the same suite
are guarded with `DS4_KV_LADDER=0` or restated — see
[CHANGES-V41.md](../CHANGES-V41.md) section 3a; run the suite with
`DS4_KV_LADDER=0` as well to exercise the pre-ladder policy.

**That a long prefill leaves a checkpoint behind.** Start a server with a fresh
KV disk directory and a continued interval well inside the prompt length, send a
prompt long enough to cross several chunk boundaries, and watch the server lines:

```
./ds4-server -m MODEL.gguf -c 393216 --port 8199 \
    --kv-disk-dir "$(mktemp -d)" --kv-disk-space-mb 98304 \
    --kv-cache-continued-interval-tokens 4096 --prefill-chunk 8192
```

What to look for, in order: a cold store at the anchor, then
`kv cache stored tokens=... reason=continued` at each crossed boundary, and no
`no valid checkpoint to stage`. Before `0f0a367` the last line is what you get
instead, on every boundary; before `9402d39` you get nothing at all when the
prefill resumed from an anchor that is not a multiple of the interval, which is
the case for any chat with a system-and-tools prefix.

**That the checkpoint is worth having.** Send a strict prefix of the same
conversation to a fresh session and read the prefill line: the restored token
count, the load milliseconds and the number actually prefilled. The measured
rows in V41_M3ULTRA.md section 5 are that line, against the cold time for the same
prefix on the same build.

Neither of the two checkpoint commits may move throughput, so both were gated on
a 62k cold prefill inside 1 % of its standing receipt and on the 8k byte gates;
`bench/reproduce-v41-native.sh --depth 62k-prefill` is that check.

## 9. The scripts

Every invocation above is also an argument-driven script. None of them hard-codes
a path, a host or a machine.

| script | what it runs |
|---|---|
| `bench/reproduce-v41-native.sh` | one `ds4-bench` arm: `--depth 8k\|32k\|62k\|62k-prefill`, `--model`, `--fixture`, optional `--fixture-sha256` to refuse a wrong prompt. Writes `identity.txt`, the CSV, the log and the decoded text |
| `bench/reproduce-v41-ttft.py` | the cold / restored / restored+append arms, including the displacement requests that force the disk path. Starts and stops its own server; prints the server lines that say which path each arm took |
| `bench/reproduce-v41-dspark.py` | the serial / fixed-six / controller arms over any set of `--fixture NAME=PATH`, interleaved and repeated, each checked against `--reference NAME=PATH`. Exits non-zero on an identity failure |
| `bench/reproduce-spec-bench-v41.py` | the bounded Spec-Bench subset in the same three modes — see [../bench/SPEC-BENCH-V41.md](../bench/SPEC-BENCH-V41.md) |
| `bench/reproduce-v41-dspark.py` (depth) | the same three modes at depth: pass `--ctx-start`/`--ctx-alloc` per depth against one resident server — see section 4a for the restore protocol |
| `bench/verify-v41-manifest.py` | checks local files against `bench/v41-manifest.json`: model and drafter byte counts, binary identity, fixture digests, and every `*.text` in a finished run directory against the reference digests |

All of them scrub inherited `DS4_*` and `MTL_*` variables, so an arm runs the
build's own defaults rather than whatever was in the shell.

`bench/v41-manifest.json` is the hash-bound index: upstream and release commits
with their binary digests, the model and drafter byte counts, all eight fixtures,
and the upstream reference digest for every gate. A reader with nothing but a
checkout can still run

```
bench/verify-v41-manifest.py
```

and confirm that the fixtures and reference outputs in hand are the ones the
measurements used; pointing it at a finished run directory also checks every
`*.text` in it against a reference digest, so an output matching nothing fails
rather than passing silently.

### Still open

- The compact per-arm JSON receipt directory with staging labels **for the
  section 1 and 1a arms**. The drivers above emit `identity.json` /
  `identity.txt` and `arms.jsonl` per run, which is the raw material for it. The
  Spec-Bench subset run already has one, at
  `bench/receipts/spec-bench-v41/2026-09-14-glmctl/`.
- An upstream comparator patch (counters only), disclosed as a diff in-repo.
- DSpark counters on the chat path, which is what the Spec-Bench subset needs to
  report committed tokens per verify. See
  [../bench/SPEC-BENCH-V41.md](../bench/SPEC-BENCH-V41.md).
