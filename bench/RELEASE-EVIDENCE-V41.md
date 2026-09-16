# Release evidence: what stands behind each table

This is the original release evidence record. The [September 16 evidence](v41-20260916/README.md) covers subsequent UAT decode improvements and their public integration.

[docs/V41_M3ULTRA.md](../docs/V41_M3ULTRA.md) states results and says nothing about where they
came from, deliberately: a product document should not be a history lesson. This
file is the other half. For every table in that document it names the receipt
directory, the binary digest, the commit and the instrument, and it records the
checks that were made before a number was carried forward.

Receipt directories named here live in this project's measurement tree, not in
this repository, except `bench/receipts/`, which is shipped and hash-bound by
[v41-manifest.json](v41-manifest.json) and checked by
[verify-v41-manifest.py](verify-v41-manifest.py).

## Builds

| role | commit | `ds4-server` sha256 | `ds4-bench` sha256 |
|---|---|---|---|
| upstream baseline | `bd66c402070042bf0a79ad6ece8242de4c93680c` | `0c9b9a944d86058a29b5e0d330fca7fae2d98dff8c6457b4cba9e3159427348b` | `f956d4245e674abcb20c3d27c27b4f64a811891f8845ea8888909462d5f9bc8b` |
| kernel matrix, TTFT, memory | `b977a33f062e98808373369e718a66b134e1f6ab` | prefix `88a14ce25f41aa7e` | prefix `9a938e5319d0c4c5` |
| depth sweep, original selection path | `c050b75c324fc6f1f6907d240cd8c8a65495aae0` | `5323ade0bdf1abeba925e0e98b4d3e86c98a4030721ce6163c7a26bc35214c20` | — |
| depth sweep, shipped selection defaults | `9f432d87b9942b1e0069b2751b1a77636436c63e` | `86da6c8fcd95ced20e794483337d8800bd3f919e4bee73a3807d633fb47232a1` | — |
| admission screen, agent screen, Spec-Bench | `afe84d1770a126b5784a615f3534f2b610831778` | `cfeb2c4d186d6000e2687de88df228272eaff7d86a28ef16405a9480f3f188b6` | — |
| release head, standing byte gates | `03eb93182d0755fe22580b0165808caa89bd1739` | `a671b6eff9c147faf0413bd36a4955e47fdd8f7f2235a6d5815c49486c11a28c` | — |

## Table by table

| docs/V41_M3ULTRA.md table | receipt directory | binary | commit | instrument |
|---|---|---|---|---|
| §2 headline, decode rows | `harness/runs/depth/W.log`, `W-server.log`, per-arm `.quiet`; upstream `baseline/references/depth-{8k,62k,131k,300k}-upstream-bd66c40.csv` | `86da6c8f…` / upstream `f956d424…` | `9f432d8` / `bd66c40` | `/debug/bench` `gen_steady_tps`, 512 greedy tokens from a restored prefix; upstream column `ds4-bench` `gen_steady_tps` |
| §2 headline, prefill row; §5 cold prefill | `harness/runs/depth/` (`A.log`, `A-*-r1.json`, `.quiet`); upstream `baseline/references/depth-*-upstream-bd66c40.csv` | `5323ade0…` / upstream `f956d424…` | `c050b75` / `bd66c40` | one cold prefill per binary per depth; `restore_ms` from the same arms |
| §2 headline, TTFT rows; §5 TTFT | `release/runs/` (`ttft-*.text`, `ttft-{PROD,UPSTREAM}-server.log`, `ttft-*.quiet`) | prefix `88a14ce2…` / upstream `0c9b9a94…` | `b977a33` / `bd66c40` | stdlib SSE client, first non-empty text delta; greedy, thinking off, `max_tokens 32` |
| §2 headline, DSpark rows; §5 per-fixture table; §5 release gates | `harness/runs/glmctl/` (`arms.jsonl`, `counters.json`, `content.v3.log`, `predictions3.json`, per-arm `log-*.txt` and `q-*.quiet`) | `cfeb2c4d…` | `afe84d1` | resident server, one boot, three arms per fixture; fixture t/s is the server's own final `avg` decode line, gate t/s from the arm record |
| §5 held-out real-agent screen | `harness/runs/glmctl/` (`screen3.jsonl`, `screen3.log`, `screen3-analysis.txt`, `screen3.py`, `analyze-screen3.py`, per-arm slices and quiet checks) | `cfeb2c4d…` | `afe84d1` | streamed, per-delta timestamps; the first non-reasoning delta ends the reasoning phase; token-weighted pooling by `analyze-screen3.py` |
| §5 Spec-Bench | `bench/receipts/spec-bench-v41/2026-09-14-glmctl/` (shipped in this repository) | `cfeb2c4d…` | `afe84d1` | `bench/reproduce-spec-bench-v41.py`, resident server, three boundaries, per-request records |
| §5 disk-store checkpoints, rows 1–3 | `harness/runs/ckfix/`, `harness/runs/ckfix2/` (`arms.jsonl`, `server.log`, per-arm logs and quiet checks) | — | `0f0a367`, `9402d39` | server log lines for the store and the restore, wall clock per request |
| §5 disk-store checkpoints, row 4; §5 one long served turn | `deploy/runs/uat/final/` | `b9064f69…` (long turn), `a671b6ef…` (byte gates) | `f1ba77e`, `03eb931` | the ordinary serving path on the deployed stack; decode-rate split from the server's own chunk samples |
| §4 fault latch | `harness/runs/faultlatch/` (`fault.sh`, `fault.py`, `result-drafter_once.json`, `result-after_verify.json`, per-point server logs) | not recorded | `f1ba77e` | dedicated boot, 8k code prompt, 192 greedy tokens, one injection point per boot |
| §4 sampled-decode certification | `tests/test_sampling.c` in this repository | — | release head | chi-square against the target's filtered distribution, 200,000 draws per configuration, driving the shipped `ds41_accept_or_residual()` |
| §3 memory numbers | `release/runs/` (`mem-mtp-*.txt`, `mem-ttft-*.txt`) | prefix `88a14ce2…` / upstream `0c9b9a94…` | `b977a33` / `bd66c40` | `vm_stat` before the process and five seconds after `listening on`, plus `footprint -p`; wired pages of 16,384 B |

## Checks made before a table was carried forward

**The prefill and matmul path is unchanged since the prefill and TTFT arms were
taken.** Between `b977a33` and the release head the only differences in
`metal/dense.metal`, `metal/moe.metal` and `metal/dsv41.metal` are comment text.
The two files that did change compute — `metal/dsv4_misc.metal` and the new
`metal/argsort.metal` — are reached only through `ds41_attention_select_published`
and `ds41_attention_pick`, the single-token decode path; no prefill dispatch
calls them. The cold-prefill and TTFT rows therefore stand.

**The decode selection chain did change, and the depth column was rebuilt rather
than carried.** A compact index scorer and a bounded radix top-k selector landed
after the original depth sweep and are on by default. The depth rows in
docs/V41_M3ULTRA.md come from the run whose defaults are the shipped ones
(`harness/runs/depth/W.log`), not from the earlier sweep. The two are anchored to
each other inside that same run: its control arm with both mechanisms off reads
23.101 t/s at 300,000 tokens against the earlier sweep's 23.047, and its shipped
defaults read 28.274. Every arm in it is byte-identical to its depth's upstream
reference and every arm was taken with the GPU sampled idle.

**Residual uncertainty in the decode comparison, stated rather than hidden.** The
upstream decode column is `ds4-bench` and this branch's is the resident
`/debug/bench` harness; the two wrappers compute `gen_steady_tps` identically but
differ in how the prefix is established. On this branch's own binary the two
agree at 8,192 tokens to 0.06 % — `ds4-bench` 31.415 t/s against the harness's
31.434 on a commit that changed no kernel between them — so the pairing is sound
to well inside the ratios quoted. No equivalent harness measurement of the upstream binary exists, and that
is the one thing the comparison rests on that was not directly checked.

**32k decode is not in docs/V41_M3ULTRA.md and this is why.** The matched 32,768-token
decode pair (upstream 16.310 t/s, this branch 30.500) predates the selection
stages, and no arm at that depth was taken with the shipped defaults. The stages
are worth nothing at 8k and 1.27 ms per token at 62k, so the shipped binary can
only be faster than 30.500 there — which makes the old figure a lower bound
rather than a result, and it was left out instead of being quoted as one. The
32k/256 byte gate on the deployed binary is unaffected and passes; it is an
identity gate, not a rate.

**The Spec-Bench receipt directory was chosen by binary, not by date.** Two runs
of the same frozen subset exist. The one shipped in `bench/receipts/` is the one
whose `identity.json` records `cfeb2c4d…`, the binary carrying the shipped
admission controller. The earlier run, on `463b186a…`, measured the controller
before confidence declines were priced into the window and is not shipped.
`identity.json` and `report.json` in the shipped directory carry the Spec-Bench
checkout's basename where the driver had recorded an absolute path; nothing else
was touched and no measured value changed. `receipt-hashes.json` is computed over
the files as shipped.

**The DSpark fixture and agent tables come from the third pass, not the second.**
Two screens of the same fixtures on the same day differ in one rule: whether a
confidence decline takes a slot in the three-call admission window. The shipped
controller prices declines, so the shipped receipts are `arms.jsonl`,
`counters.json` and `content.v3.log`. The second pass is kept beside them as
`arms.v2.jsonl`, `counters.v2.json` and `predictions.v2.json` and is not quoted
in docs/V41_M3ULTRA.md. The `boot.log` in that directory records the first boot of the
day; the third pass ran on the rebuilt binary recorded above.

**Tables that exist only for the controller that was deleted were dropped, not
caveated.** The five-length greedy sweep, the five-length sampled sweep, the
512-token agent screen, the earlier standing-gate table and the calibrated-window
depth column all measured a controller that is gone from the tree and from the
build. None of them appears in docs/V41_M3ULTRA.md. Their numbers remain in
[v41-manifest.json](v41-manifest.json) under keys that say what they measured.

**The per-kernel depth attribution is not in docs/V41_M3ULTRA.md.** The ledger run that
attributed 98.2 % of the depth penalty to the selection chain was taken on the
original selection path, before the mechanisms that address it shipped. Its
structural conclusion still holds — attention, gather, the routed MoE and the
dense projections are flat across a 36x context range — and that conclusion is
what docs/V41_M3ULTRA.md states, without the millisecond table. The table itself is in
[v41-manifest.json](v41-manifest.json) under `depth_sweep.depth_attribution` and
in [CHANGES-V41.md](../CHANGES-V41.md).
