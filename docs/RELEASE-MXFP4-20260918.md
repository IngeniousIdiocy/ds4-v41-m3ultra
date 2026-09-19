# MXFP4 release — September 18, 2026

Accepted in interactive UAT by the maintainer. This release combines kernelpool's native MXFP4 experts, short-row matrix kernels, expert deduplication, fused epilogues and GPU Markov chain with the accepted scalar, prefill and controller work described in the three component documents.

## Get the same implementation

```sh
git clone https://github.com/IngeniousIdiocy/ds4-v41-m3ultra.git
cd ds4-v41-m3ultra
make -j4
```

Use matching native DeepSeek V4.1 Flash MXFP4 target and DSpark GGUF files. The included converter accepts the original Hugging Face safetensors checkpoint; `--quant mxfp4` preserves the released expert codes/scales and packs them into GGUF. It does not convert Q4_K back to MXFP4. See [conversion instructions](../gguf-tools/README.md).

Install the converter dependencies (`numpy`, `tokenizers`, and `sympy`). Set `SOURCE_REVISION` to the full 40-character Hugging Face commit used to download the checkpoint; the converter requires it for source provenance.

```sh
make -C gguf-tools libds4quants.dylib
python3 gguf-tools/deepseek41_quantize.py \
  --hf models/DeepSeek-V4.1-Flash --quant mxfp4 \
  --source-revision "$SOURCE_REVISION" \
  --out gguf/DeepSeek-V4.1-Flash-MXFP4.gguf \
  --dspark-out gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf

./ds4-server --metal \
  -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf \
  --dspark --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf \
  --ctx 65536 --prefill-chunk 8192 \
  --default-top-p 0.95 --default-min-p 0 --mtp-exact-sampling \
  --kv-disk-dir ./kvcache-mxfp4
```

Choose context and cache limits to fit available memory/storage. The accepted deployment used context 393216; the example uses 65536. No optimization-enabling environment flags are required. Exact speculative sampling is selected explicitly in this example for nonzero temperature. Explicit request sampling values override server defaults.

The alternative cost-aware controller is the default on supported local Metal execution. Set `DS4_METAL_DISABLE_V41_PUBLIC_CONTROLLER=1` to retain kernelpool's original controller, including its original reasoning behavior and no post-Markov probability readback. The environment-variable name is historical; it is a policy choice, not a kernel enable switch.

Draft-enabled disk snapshots now include target and drafter state. Old target-only or differently formatted snapshots are not interchangeable; use a fresh cache directory. Conversion and cache restoration are separate from benchmark timing.

## Matched results

M3 Ultra, 80 GPU cores, 512 GB; native MXFP4 target and drafter loaded. Four repeats, 512 decoded tokens, cached identity-checked prefixes, quiet-machine checks. Serial disables drafting but retains drafter capture, matching reasoning in a DSpark-enabled session.

| Fixture | Previous public mean / best | Current mean / best |
|---|---:|---:|
| Serial 8k | 34.36 / 34.43 | 35.76 / 35.80 t/s |
| Serial 62k | 34.11 / 34.17 | 34.86 / 34.91 t/s |
| Directed code, 8609-token prefix | 41.17 / 41.58 | 45.02 / 45.11 t/s |
| Final answer A26, 8626-token prefix | 33.40 / 33.43 | 35.62 / 36.80 t/s |
| Final answer A30, 5441-token prefix | 39.08 / 39.13 | 42.14 / 42.19 t/s |

The baseline is b10cb40 with loader-only aliases for the native drafter's tensor names and shape metadata; kernels and controller are unchanged. Current runtime was qualified at c47d036. The publication separates its changes into reviewable commits; the inference engine and Metal kernels are unchanged. The server reasoning gate additionally follows the selected controller, preserving the original controller when explicitly selected.

A26/A30 are synthetic final-answer variants, not the historical full-agent UI benchmark. A26 has one faster valid continuation with more useful admitted drafts; retain that best while showing the mean. Within the current release, A26 MTP averages 35.62 versus serial 35.77 t/s. MTP does not win on every prose fixture. Code's four MTP runs have identical token streams; there is no cross-fork or serial/MTP byte-identity claim.

Empty-KV prefill with warmed weights, chunk 8192: 8k previous public mean/best 706.49/708.53, current 736.67/737.18 t/s; 63,488 tokens previous 840.40/840.58, current 845.75/846.05 t/s. Kernelpool-control prefill and append comparisons are in [the prefill notes](METAL-V41-PREFILL.md).

The standard no-drafter 128-token sweep is separate: 8k kernelpool 34.52/34.53, previous public 34.92/34.97, scalar blend 36.96/36.97 t/s (mean/best). Do not relabel that as the drafter-loaded result.

## Accepted and rejected work

- [Decode implementation, validation and failed experiments](METAL-V41-DECODE.md)
- [Prefill implementation, validation and failed experiments](METAL-V41-PREFILL.md)
- [DSpark state/controller integration and failed experiments](METAL-V41-DSPARK.md)
- [Prior Q4 experiment history](experiments/v41-20260915/README.md)
- [Prior Q4 consolidation](DECODE-CONSOLIDATION-20260916.md)

Rejected Q8 verify replacements, three-row Engram projection, expanded expert-slot scheduling, normalization elision, pipeline-cache activation and Markov bias caching are absent from the runtime. Kernelpool's established verify kernels remain selected. The prefill comparison was initially incomplete; the first candidate preserved roughly 741 t/s rather than restoring the faster producer/attention/read-ahead combination. That candidate was rejected, the comparison expanded, and the complete bundle above was qualified before acceptance.

Invalid measurements were excluded: repetitive raw-code continuations, forced continuation beyond a tool handoff, and a benchmark entry point that selected serial instead of speculative generation. No speedup is credited to those runs.

## Validation and limitations

Accepted integration: Metal build, CPU engine-object build, controller/server units, model-independent regressions, isolated Metal kernels, 50,000-case probability publication gate, 72 integrated rewind cases, controller lifecycle/fault checks, and live tool-call/continuation smoke. Interactive UAT accepted. Live UI rates are not substituted for controlled benchmarks.

The broad test target previously stopped at a Qwen checkpoint test because its expected `ds4flash.gguf` was absent. This is not a full-suite pass. Actual CUDA/distributed model execution was not tested. Publication checks separately rebuild/link the three independent branches and combined release, run server/controller units, and compile their complete Metal sources without reloading model weights or stopping the live deployment.

The previous Q4 release remains in Git history and is tagged `v41-pre-mxfp4-20260918`. Check out that tag for its original code, measurements and reproduction scripts. Older benchmark material retained under `bench/` is historical evidence; it is not a claim that its old harnesses target this MXFP4 release.

## Upstream review

- [Decode PR](https://github.com/kernelpool/ds4/pull/1) — independent branch against `kernelpool/ds4:ds41-perf`.
- [Prefill PR](https://github.com/kernelpool/ds4/pull/2) — independent branch against `kernelpool/ds4:ds41-perf`.
- [Dspark PR](https://github.com/kernelpool/ds4/pull/3) — independent branch against `kernelpool/ds4:ds41-perf`.

The DSpark policy and runtime integration are separate commits; the state improvements can be taken while retaining the original controller.
