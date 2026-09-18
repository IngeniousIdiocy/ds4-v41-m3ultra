## September 18 MXFP4 integration

This release incorporates [kernelpool/ds4, ds41-perf](https://github.com/kernelpool/ds4/tree/ds41-perf) at `af7c02b`, including native MXFP4 conversion/experts, short-row verification kernels, expert deduplication, fused epilogues and the GPU Markov chain. Original license notices remain. Further contributions are separated in the [decode](https://github.com/kernelpool/ds4/pull/1), [prefill](https://github.com/kernelpool/ds4/pull/2), and [DSpark](https://github.com/kernelpool/ds4/pull/3) PRs.

# Third-party code, weights and dependencies on the V4.1 Flash M3 Ultra branch

The repository licence (`LICENSE`, MIT) is unchanged. `third_party/iris/`,
`licenses/` and `cuda/` are upstream's and unchanged. This file records what this
branch adds on top of them.

## Upstream: antirez/ds4

This branch is a fork of **[antirez/ds4](https://github.com/antirez/ds4)** at
commit **`bd66c402070042bf0a79ad6ece8242de4c93680c`** ("DeepSeek v4.1 Flash
support for Metal", Salvatore Sanfilippo, 2026-09-12). It has not been rebased
onto later upstream commits.

Upstream is **MIT**. Its `LICENSE`
([antirez/ds4](https://github.com/antirez/ds4/blob/main/LICENSE), unchanged here)
carries three copyright lines verbatim:

```
Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek
```

Everything this branch
changes — `ds4.c`, `ds4.h`, `ds4_metal.m`, `ds4_server.c`, `ds4_engram.{c,h}`,
`ds4_gpu.h`, `metal/*.metal`, the `gguf-tools` additions and the `tests`
additions — is a modification of, or an addition to, that MIT source and carries
the same licence.

The DSpark runtime here is also built on upstream's existing V4 DSpark machinery:
the support-GGUF tensor names mirror V4's so upstream's binder is shared, and
`verify_batch_core` is pre-existing ds4 code, not part of this branch's work.

## DeepSeek's reference PyTorch stack

The DSpark drafter's forward pass on Metal was **derived from DeepSeek's own
released reference implementation**, not reverse-engineered from weights.

Source: the `inference/` directory of
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
at revision **`df42c109f1defefcbfcedbe7d905718a12266e40`** — in particular
`model.py`, `engram.py` and `kernel.py`, with `convert.py` for the tensor
conventions. DeepSeek describe it as "a readable reference implementation rather
than a production serving engine".

**Licence: MIT.** The repository's model-card metadata reads `License: mit` and
its card states, verbatim, *"This repository and the model weights are licensed
under the MIT License."* The repository's
[`LICENSE`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/LICENSE)
is the standard MIT text under `Copyright (c) 2023 DeepSeek` — the **same
copyright line upstream ds4 already carries in its own `LICENSE`**, which is
independent corroboration rather than our reading of one page. The card draws no
distinction between code and weights and names no separate licence for
`inference/`, for `encoding.py`, or for the DSpark components; `inference/README.md`
states no licence of its own, so the repository licence governs.

What was derived from it:

- **`ds41_dspark` (in `ds4.c`)** — the drafter forward. The four V4-specific sites
  were re-parameterised for V4.1: no per-head q RMSNorm, FP8-E8M0 KV, V4.1's BF16
  rounding boundaries, and a 3-of-128 MoE read from file metadata rather than the
  backbone's shape. The non-causal five-position block structure, the sliding-window
  KV ring, the rank-256 factorised Markov head and the confidence head all follow
  the reference.
- **`gguf-tools/deepseek41_dspark_quantize.py`** — the tensor inventory, shapes,
  dtypes and the tying of `token_embd` / `output` to the backbone come from
  `model.py` (`Transformer.__init__`) and `convert.py`, read together with the
  safetensors headers.
- **`gguf-tools/tests/dspark41_reference.py`** — a CPU fp32 oracle in which
  `sparse_attn`, `hc_split_sinkhorn` and `act_quant` are reimplemented from their
  released tilelang source (which is CUDA-only), driven by synthetic hidden states
  so the backbone is never loaded.

Architecture constants (`hidden_size`, `moe_intermediate_size`, head counts,
`dspark_block_size`, `dspark_markov_rank`, `dspark_target_layer_ids`, and the
rest) were read from the checkpoint's own `config.json`.

Verified 2026-09-13 against the live repository: model card, card metadata and
`LICENSE` blob. No contrary licence, no separate agreement and no
non-commercial or no-derivatives clause was found anywhere in the checkpoint
repository.

## Model weights

**No weights are distributed here** — not the backbone, not the drafter, not any
derivative tensor, and no golden weight data.

- **Backbone.** The validated weights are a Q4 conversion of the official
  [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  checkpoint (518,596,067,328 bytes as converted). **The weights are MIT**, on the
  same statement and the same `LICENSE` file quoted above: *"This repository and
  the model weights are licensed under the MIT License."* They are not part of this
  repository. Produce the GGUF yourself with the V4.1 conversion tooling in
  `gguf-tools/`.
- **Drafter.** There is no separate DSpark weights repository for V4.1. The
  drafter ships **inside the same checkpoint**, as the 2,401 `mtp.*` tensors in
  shards 44-46 of 48, and is therefore covered by the same MIT licence and the same
  `LICENSE` file. (`deepseek-ai/DeepSeek-V4-Flash-DSpark` is the *V4* drafter, a
  different checkpoint, and is not used here.) The support GGUF
  (8,328,937,472 bytes at the `q4` recipe) is built from those tensors by
  `gguf-tools/deepseek41_dspark_quantize.py`; it is a derivative of the checkpoint
  and carries the checkpoint's licence. It is not redistributed here.
  [docs/DSPARK-V41.md](docs/DSPARK-V41.md) section 1 documents how to build it.

  Note for anyone comparing against this project's GLM branch: that branch's
  DFlash2 drafter is **CC BY-NC-ND 4.0** and had to be handled as
  non-redistributable and non-commercial. V4.1's drafter is not in that position —
  it is MIT, like the backbone. We still do not redistribute it, because we
  redistribute no weights at all.

## Code carried over from this project's GLM-5.3-Flash branch

This project also maintains a **GLM-5.3-Flash / DFlash2** branch, in a separate
public repository of its own. Several mechanisms here are ports from it. Same
project and the same MIT terms as the rest of this repository; recorded because a
reader should know where a design came from.

- **`ds4_ds41_dspark_adaptive.h`**, the windowed cost-feedback admission
  controller, is a port of `ds4_dflash_adaptive.h` from that branch. The policy —
  propose by default, admit the confident prefix and decline a proposal shorter
  than `min_draft`, judge three attempts against the median of the last nine
  measured serial tokens, back a losing window off for 16/32/64/128 consumed
  serial tokens, decode reasoning spans serially — is that branch's design,
  including `dflash_adaptive_prefix`, the confidence floor `p_min` = 0.75 carried
  over verbatim and the decline-accounting rule. This branch contributes the port,
  the removal of the width search, `min_draft` re-expressed against V4.1's
  five draft positions, and the V4.1 measurements.
  `tests/test_ds41_dspark_adaptive.c` is likewise ported from that branch's
  `tests/test_dflash_windowed.c`.
- **`ds4_ds41_dspark_fault.h`**, the session-scoped drafter fault latch, is a port
  of that branch's `ds4_dflash_fault.h`, unchanged in shape and meaning: the
  drained/undrained distinction, the session-lifetime latch, the counters carried
  in the usage block and the `DS4_DS41_DSPARK_FAIL` injection points are all that
  branch's.
- **The disk KV cache's eviction policy** — the decaying score baseline
  (`DS4_KVSTORE_BASE_HALF_LIFE_SECONDS`), the store-time waypoint ladder with its
  32,768-token dense window and 5/7 geometric skeleton, and deepest-first
  shedding of same-chain superseded rungs under pressure — is a port of that
  branch's three eviction commits into `ds4_kvstore.c`. The design credit is that
  branch's; this branch contributes the port, the V4.1 store-layer reproducer and
  two corrections to the unit tests that came with it. **Both corrections are owed
  back to that branch, where the same two assertions fail under its own default:**
  the superseded-prefix test encodes the pre-ladder policy and needs the
  `DS4_KV_LADDER=0` guard, and the ladder budget bound asserted as *under 3x the
  frontier* is unsatisfiable — the three dense rungs alone sum to 2.8x.
- The "mask before filter" bug the sampled path avoids
  ([docs/DSPARK-V41.md](docs/DSPARK-V41.md) section 5) was first found and written
  down on that branch.

## Test fixtures

- **The `code`, `code-32k` and `code-62k` benchmark fixtures** are this
  repository's own C sources and Python tooling, truncated to each gate's token
  length.
- **The `prose-en` fixture** is this repository's README and twelve files from
  `docs/`.
- **The `agent` and `agent2` fixtures are real agent transcripts and are not
  redistributed.** [docs/REPRODUCE-V41.md](docs/REPRODUCE-V41.md) section 2
  describes their shape so comparable fixtures can be built.
- **`speed-bench/promessi_sposi.txt`** is inherited from upstream and is not used
  by any measurement in this release.
- **`tests/long_context_security_prompt.txt`** and
  `gguf-tools/quality-testing/data/*` are upstream's, inherited at `bd66c40`, with
  upstream's provenance. Paths and internal-looking strings inside those fixtures
  are upstream test text, not local data.
- **Spec-Bench** is not vendored here. The subset driver reads a caller-supplied
  checkout, binds it by commit and by the SHA-256 of its `question.jsonl`, and
  copies no dataset text into this repository — the receipts identify a request by
  the hash of its body. Its licence terms, and the separate terms of each
  constituent source dataset, stay with that checkout. See
  [bench/SPEC-BENCH-V41.md](bench/SPEC-BENCH-V41.md).

## Tooling dependencies added by this branch

- **numpy**, used by `gguf-tools/deepseek41_dspark_quantize.py`,
  `gguf-tools/deepseek41_validate_gguf.py` and
  `gguf-tools/tests/dspark41_reference.py`. Already a dependency of upstream's
  gguf tooling. Not needed to build or run `ds4`.
- The reproducers under `bench/` use only the Python standard library.
