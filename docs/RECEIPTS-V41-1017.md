# V4.1 1017 decode update: receipts

The measurements and exactness checks behind the 1017 decode commits on this
branch. Each commit message carries its own what/why and ledger entry; this
page collects the results that arrived after a commit landed, or that span
several commits. Ledger ids (S40-*, D0xx) refer to the serial 40 tok/s
campaign record.

Unless stated otherwise: one M3 Ultra (512 GB), the 1017 MXFP4 target
weights, decode at concurrency 1, and the campaign's prose and code fixtures.

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
  both control runs. It stays opt-in until the blended narrow readers land.
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
