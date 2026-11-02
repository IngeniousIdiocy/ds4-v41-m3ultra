# Third-party code, weights and dependencies for DeepSeek V4.1 on Metal

The repository licence (`LICENSE`, MIT) is DwarfStar's and is unchanged.
`third_party/` and `licenses/` are DwarfStar's, with three changes:

- `licenses/omlx.txt`, which this branch adds (below).
- `third_party/iris/png.h`, the bundled PNG decoder, is this branch's own
  change and decodes the same pixels faster (a 3840x2160 screenshot in 35 ms
  instead of 97): CRC-32 from a 256-entry table, Adler-32 reduced once every
  5,552 bytes as zlib does, and on Apple the image data inflated by the
  system's DEFLATE decoder, accepted only when it gives exactly the expected
  bytes with a matching Adler-32; otherwise the bundled decoder runs.
  `DS4_DISABLE_PNG_SYSTEM_INFLATE=1` always runs the bundled decoder.
- `third_party/iris/jpeg.h` carries Giulio Davide Carparelli's fix from
  DwarfStar (antirez/ds4 commit `2768d0ed`, issue #1292): progressive scans
  whose spectral selection or successive approximation values break the JPEG
  standard are rejected, since one with an end index past 63 could write
  outside a coefficient block.

This file records what the V4.1 Metal work adds on top of them, and where the
pieces came from.

## DwarfStar

This branch is built on **[antirez/ds4](https://github.com/antirez/ds4)**
(DwarfStar) at commit `5379c1c9`. Every change here is a modification of, or an
addition to, that MIT source and carries the same licence. DwarfStar's
`LICENSE` carries three copyright lines, kept verbatim:

```
Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek
```

The engine, the V4.1 model graph, the DSpark runtime, the server, the disk KV
cache, the tools and the tests are DwarfStar's. Each speed group on this branch
switches off back to DwarfStar's own code.

## Ivan Fioravanti and Tarjei Mandt

This branch is built on antirez's DwarfStar. Thanks to Tarjei Mandt ([kernelpool/ds4](https://github.com/kernelpool/ds4)) and Ivan Fioravanti ([ivanfioravanti/ds4-metal](https://github.com/ivanfioravanti/ds4-metal)): some of the code here started in their forks, including several of Tarjei's decode and verify kernels and Ivan's narrow decode weight copies, prefill index scorer, prefill attention on simdgroup matrices (PREFILL_MMA), bounded decoder replay (CED) and in-chunk suffix trim (SUFFIX_TRIM). The speed groups and how they fit together are this project's own.

## oMLX

Thanks to the [oMLX](https://github.com/jundot/omlx) contributors for their
V4.1 decode kernels and the ideas this work borrowed from them. Three pieces of
the batched decode started in oMLX's `omlx/patches/deepseek_v41/` (MIT,
`Copyright (c) 2023 DeepSeek`): decode attention's scores and PV on
simdgroup-matrix tiles (ATTN_TILES), rewritten in Metal for this engine's key
cache and reduce; the FP4 expert decode by shifts instead of a table
(FP4_NO_TABLE); and the row-block-major order of the routed expert grids
(BATCH_RBLOCK). [licenses/omlx.txt](licenses/omlx.txt) has the notice. oMLX's
other kernels were ported and measured too; they lost to this engine's own and
are not shipped ([docs/FAILED-EXPERIMENTS-V41.md](docs/FAILED-EXPERIMENTS-V41.md)).

## DwarfStar pull requests

Two image features come from open pull requests to DwarfStar, taken with their
authors' commits: #1225 by nazeshinjite, which pairs a V4.1 vision file with its
language checkpoint by revision so the 1017 encoder loads, and #1197 by joohooo,
which reads WebP images on macOS through ImageIO.

## The DSpark admission policy

PUBLIC_CONTROLLER (`ds4_deepseek41_admission.h`) is this project's own policy:
judge three drafting attempts together against the median cost of the last nine
serial tokens, back off for 16, 32, 64 and 128 serial tokens after a loss, and
verify a block only when its first three draft positions are each at least 75%
confident. It was first written for this project's GLM 5.3 Flash work and then
for the previous V4.1 release; here it runs on DwarfStar's DSpark runtime.

## DeepSeek

The model, its tokenizer and the DSpark drafter stages come from DeepSeek's
release of DeepSeek V4.1 Flash. DeepSeek's model card states that the repository
and the model weights are licensed under the MIT licence, the same terms and the
same `Copyright (c) 2023 DeepSeek` line DwarfStar's `LICENSE` already carries.

**No weights are distributed here.** Prepare them from the public files with
`gguf-tools/deepseek41_prepare.py` (the MXFP4 GGUF published at
`antirez/deepseek-v4.1-flash-gguf`, and three shards of
`deepseek-ai/DeepSeek-V4.1-Flash-1017` for the drafter), or convert them with
`gguf-tools/deepseek41_quantize.py`. The optional decode sidecars are copied
byte for byte out of DeepSeek's released safetensors by the scripts in
`gguf-tools/`.

## Tooling dependencies

- **numpy, tokenizers and sympy** for the V4.1 conversion and preparation
  scripts in `gguf-tools/`, as DwarfStar's own V4.1 converter already needs.
  None of them is needed to build or run the engine.
