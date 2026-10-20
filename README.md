## DeepSeek V4.1 Flash 1017 on M3 Ultra — October update

This branch is our own implementation of DeepSeek V4.1 Flash for Apple
Silicon, written on [antirez's DwarfStar](https://github.com/antirez/ds4)
engine. The October update targets the 1017 checkpoint. On one M3 Ultra it
takes serial decode past 40 tokens/s.

We incorporated specific pieces of others' work, and their commits keep their
authorship:

- **Ivan Fioravanti:**
  - the E4M3 shift decode that the blended readers adapt;
  - the literal-shape dispatch mechanism;
  - the narrow decode copies built at load;
  - the exact prefill indexer MMA.
- **Tarjei Mandt (kernelpool):**
  - candidate-block index scoring (9b9dfee);
  - the radix top-k select (af7c02b), which this branch's 512-wide winners'
    sort extends.
- **antirez:** DwarfStar itself.

[THIRD_PARTY-V41.md](THIRD_PARTY-V41.md) lists the other attributed work.

### Headline numbers

One M3 Ultra (512 GB), 1017 MXFP4 weights, concurrency 1. Each row was
measured in its own window, with the method named in that row. Labels such as
q2k-a1, fp8pf-a5 and W5 name those measurement windows; the
[receipts](docs/RECEIPTS-V41-1017.md) summarise each one's evidence.

| Context | Decode tok/s | Window and method |
| ---: | ---: | --- |
| 2K | 40.60 | q2k-a1: campaign stack, greedy 128, Ivan Fioravanti's steady metric (wall time of tokens 2..N); ours read the same |
| 8K | 40.44 / 40.48 (prose), 40.51 / 40.42 (code) | fp8pf-a5 d01: this branch (a clean export of 77eea27c, default build, the window's builder-on arm), cold prefill, W72 fixtures, 512 tokens at temp 1 / top_p 0.95, seed 1017, two runs each, our metric (first sampling to last evaluation); timing provisional, see below |
| 64K | 39.70 | W5 on the state-depth tree: candidate scoring + sort512 + candidate list once, greedy 128 on saved snapshots, eval-interval metric |
| 131K | 39.07 | W5 on the state-depth tree, as above |
| 300K | 38.02 | W5 on the state-depth tree, as above |

The 8K runs are bit-identical to the irqres-a1 T arm of the campaign: the
prefix, final logits and session state digests are the same.

| Fixture | prefix | logits | state |
| --- | --- | --- | --- |
| prose | `3cd4d4bba0ab9986` | `ed3ca952608e6274` | `6d2d529ba8c0e5c0` |
| code | `166bd265feac6270` | `c1c571c0a9ade5be` | `428ef17df51250d8` |

The d01 arm set a `DS4_*` variable for the window's builder comparison. The
one launch in that window with no `DS4_*` variable at all was b13 (40.79
tok/s steady), and it was bit-identical to the builder-on arm. The d01
timing, 24.716182 / 24.714071 ms/token (prose / code), is provisional: two
harness dry runs started during d01 and may have added CPU noise. A clean
repeat is planned.
The window as a whole did not pass. Its optional `DS4_V41_S1TG` arm (`cn`)
failed the exactness gate in d01, so the window stopped and its d02, d03 and
a01 rows did not run.

Window w1-a2 recorded both metrics for the same runs. At 8K, greedy 128, Ivan's
steady metric gave 40.130 and ours 40.087.

Comparison against Ivan Fioravanti's performance branch
(`v41-tp-decode-engram-warm`, 8d6a39f): pending window a8.

Cold prefill at 64K was 889 tok/s, against 841 without Ivan's indexer MMA
(+5.7%, bit-exact; window w1-a2). That was measured on the campaign tree with
his commit applied. The same path (feed660) is on by default on this branch;
`DS4_METAL_DISABLE_V41_INDEX_MMA=1` turns it off.

Exactness method:

- Every arm is compared with its control by FNV-1a digests of the decoded
  tokens, the frontier prefix, the final logits and the whole session state.
- Every accepted change matched in every run.

**EADD, the staged Engram gate.** The decode step's Engram gate runs as one
256-thread group per stream:
- all threads stage the inputs into threadgroup memory;
- one SIMD group runs the 32-lane kernel's reduction chains in the same order;
- all threads apply the residual update.

Its output is bit-identical to the 32-lane kernel. The model-free test
`tests/test_deepseek41_metal --engram-add-exact` found 0 of 3,072,000 fresh
streams (15.7G values) differing. In a micro at the decode shape, a call takes
18.6 µs, against 122.5 µs for the 32-lane kernel. EADD is on by default, and
`DS4_METAL_DISABLE_V41_ENGRAM_ADD_TG=1` turns it off.

**Single-group latency kernels.** Two decode kernels run in single-group
latency forms by default:
- the HC collapse, as two dispatches over 32 + 10 groups;
- the indexer top-k merge ladder, staged in threadgroup memory and finished
  in one group.

Both give the replaced kernels' bits. Window a6 checked each alone at 8K:
the collapse saved 0.069 / 0.061 ms/token and the ladder 0.118 / 0.105
(prose / code). Together they gave the base's digests at 8K (window W10-a2),
and `tests/test_deepseek41_metal --s1tg` compares both with the originals.
`DS4_V41_S1TG=none` restores the originals.

The rows and digests above were measured before the staged Engram gate and
these two kernels became the default. The composed run of this release is
pending.

The 8K row is a composed run of this branch. A composed run at depth (64K and
beyond) on this branch is still pending. Windows, arms and per-run numbers are
in [RECEIPTS-V41-1017.md](docs/RECEIPTS-V41-1017.md). How the campaign got
here, including what was tried and dropped, is in
[V41-1017-HISTORY.md](docs/V41-1017-HISTORY.md).

### Prepare the model

The engine loads one 16 KiB-aligned MXFP4 GGUF. The public release ships it as nine
split files; the DSpark drafter is converted from three shards of the original
checkpoint. Download both, then prepare once:

```
hf download antirez/deepseek-v4.1-flash-gguf --revision 9dcee51692e6540f697141fc98b097e3c96492a3 \
    --include 'DeepSeek-V4.1-Flash-1017-MXFP4-experts-*' --local-dir shards
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --include config.json tokenizer.json 'model-0004[456]-of-00048.safetensors' --local-dir dspark-source
pip install numpy tokenizers sympy
python3 gguf-tools/deepseek41_prepare.py --shards shards --dspark-source dspark-source --out gguf
python3 gguf-tools/deepseek41_prepare.py --verify --out gguf
```

This writes `gguf/DeepSeek-V4.1-Flash-MXFP4.gguf` (467.2 GiB) and
`gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf` (7.4 GiB). The shards are checked
against their published SHA-256 while they are merged; only `general.alignment`
changes and the `split.*` keys are dropped, so every tensor byte is the release's.
The drafter is the 1017 checkpoint's `mtp.*` stages (7.93 GB of source) in Q8_0 /
MXFP4, with `dspark.target_architecture=deepseek41`. `--verify` re-reads both
files and compares every tensor with the digests pinned in
`gguf-tools/deepseek41_prepare.json`; exit 0 means byte-identical to the reference
build. `--dry-run` checks inputs, free space and Python modules without writing.

Needs: 543.9 GB free beside `--out` (509.6 GB of output plus a 32 GiB reserve),
plus the 509.5 GB of inputs if they share the volume; delete the inputs after
`--verify` passes. Python 3.10+ with numpy, tokenizers and sympy, and `cc` and
`make` (the quantizer library builds on first use). On an M3 Ultra with inputs and
outputs on one Thunderbolt NVMe drive: merge 5.7 min, drafter 10 s, verify 2.3 min.

The engine needs only the MXFP4 GGUF. It builds the narrow decode copies
(BF16 router, E4M3 Engram and indexer q_b) at load. Optional sidecars from the
original checkpoint load with `--v41-sidecar FILE`.

### Run

The recommended serving launch, from the checkout, with the paths that
Prepare writes:

```
make
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --metal \
    --host 127.0.0.1 --port 8195 -c 393216 --prefill-chunk 8192 \
    --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --dspark --mtp-exact-sampling \
    --kv-disk-dir kvcache --kv-disk-space-mb 98304 \
    --kv-cache-cold-max-tokens 393216 --kv-cache-continued-interval-tokens 4096 \
    --default-top-p 0.95 --default-min-p 0 \
    --utility-ctx 32768 --serial-reasoning
```

- The server listens on 127.0.0.1 only. Pass `--host 0.0.0.0` (or an
  interface address) to serve other machines; it has no authentication.
- `--kv-disk-dir` keeps prompt checkpoints on disk (created if missing),
  within the `--kv-disk-space-mb` budget (96 GiB here). A 300K-token
  checkpoint is about 1.8 GiB. Requests
  that resume a saved prefix, including after a restart, skip its prefill.
- Right after start the server runs a one-shot numerics canary (about 7 s);
  requests sent in that time wait for it, then use the cache.
- `--serial-reasoning` decodes reasoning one token at a time and drafts
  answers and tool calls with DSpark. Without it, DSpark also drafts
  reasoning.

DSpark on this stack (window W9, this branch at aeab950a with the launch
above, a short code request answered in 600 tokens, three runs): 57.70 tok/s
median by the server log, against 52.61 on our production build of the
September 18 stack (×1.097). 95.4% of drafted tokens were accepted, against
89.6%. With `--serial-reasoning`, a 2,050-token reasoning span ran no DSpark
cycles and the answer that followed was drafted.

[RUNTIME-V41.md](docs/RUNTIME-V41.md) lists every option: the server policy
that Project Trench uses, the disk KV cache, and the diagnostic switches. No
environment variable is needed: b13 above was a plain launch.

### Test

```
make ds4_test tests/test_deepseek41_metal
./ds4_test --server                       # server unit tests, no model
./tests/test_deepseek41_metal             # V4.1 Metal kernels, no model
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-MXFP4.gguf ./ds4_test
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-MXFP4.gguf \
    DS4_TEST_DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf ./ds4_test --dspark-verify-depth
```

`./ds4_test --help` lists the tests. Some are skipped on V4.1 by design:
- `--logprob-vectors` and `--local-golden-vectors` default to the
  DeepSeek V4 Flash 0731 fixtures, which are invalid for another model
  ([tests/test-vectors/README.md](tests/test-vectors/README.md));
- `--dspark-verify-depth` skips its V4-only prefix-capture check, since V4.1
  verifies through its own path.

### DwarfStar main reference

For reference, window W8 served the same requests from DwarfStar main and
from this branch:

- U: antirez/ds4 main `5379c1c9ac315f1540371b06a3db88491d4059ff`.
- P: this branch at `0a72af29a080a136d12fe816d2da4f0c64582838`, with
  `--serial-reasoning`.

Both arms used the same MXFP4 GGUF, the same server flags (393K context, a
fresh disk-cache directory per server lifetime) and no DSpark. Each cell
was run three times, in the order U P P U U P; the numbers are medians.
Decode is the server log's rate over 512-token answers.

| Context | U decode tok/s | P decode tok/s | U cold TTFT | P cold TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 2K | 19.56 | 41.32 | 6.275 s | 5.110 s |
| 8K | 19.84 | 40.78 | 23.413 s | 15.300 s |
| 32K | 19.14 | 40.43 | 49.030 s | 41.576 s |
| 64K | 18.64 | 40.21 | 95.724 s | 79.477 s |

Other results from the same window:

- **Reasoning decode** (thinking on, effort high, server log): U 19.84 / 19.65 /
  18.59 tok/s at 2K / 8K / 64K; P 41.41 / 40.77 / 40.22.
- **Warm agent turn** (a 32K code conversation, next turn from the live
  cache): 2.389 s on U, 2.079 s on P.
- **Restart, then resend** (restored from the disk cache):
  - U: 2.484 / 2.897 / 3.546 s at 8K / 64K / 128K;
  - P: 2.485 / 2.518 / 3.131 s.
- **Load to ready:** 2.29 to 2.30 s on both arms.
- **Memory:** both arms plan 290.12 GiB, and both peak at 296 to 297 GiB
  wired at 64K. P's idle RSS is 1.5 GB higher: these are the narrow decode
  copies built at load while the wide views stay mapped.

Scope and caveats:

- These are serving measurements of the two builds. They do not isolate
  individual optimizations or establish quality equivalence.
- On P, the 2K answers stopped at 494 tokens; every other decode answer ran
  to 512.
- P's binary predates the queued canary. An 8K resend right after a
  restart, before its canary finished, was refused by the cache and
  prefilled cold: 14.580 s against 2.485 s after cache-on. Requests now wait
  for the canary instead ([RUNTIME-V41.md](docs/RUNTIME-V41.md)). Window W9
  measured that; see "Restart and the disk cache" below.

The receipts give the ranges, answer lengths and the corrections to the
window's first summary. Source: W8's summary (not published).

### Restart and the disk cache

Window W9 ran this branch at aeab950a with the launch under Run, restarting
the server on the same cache directory:

- **300K restore.** A 303,912-token conversation took 391.6 s to first
  token cold. Sent again right after a restart, it was restored from its
  1.8 GiB checkpoint: 11.17 s to first token, including the canary.
- **The canary's cost.** In the first seconds after a restart, every
  request waits for the one-shot canary (7.15 s). An 8K resend took 9.23 s
  to first token and a fresh 11-token prompt 7.36 s. Three requests sent
  together during the canary (300K, 8K, short) were answered in that
  order, at 10.9, 14.7 and 15.5 s.
- **Canary failure.** With the canary forced to fail (a test build), the
  cache stayed off and an 8K request prefilled cold (20.2 s).
- **Old entries.** Entries from earlier builds' canary were refused and
  evicted under the disk budget.
- **DSpark follow-up.** An 8K follow-up turn resumed from the cache in
  2.13 s, against 15.28 s cold.

One cell failed. A SIGTERM 2 s after listen, with the 300K request queued,
deleted that request's checkpoint, and the client saw a dropped
connection. Both causes predate the V4.1 work: a failed prefill from a
checkpoint always deleted it, even when shutdown had interrupted the
prefill, and replies were dropped once shutdown began. e2136edd fixes both.
An interrupted prefill keeps the checkpoint, and requests still queued at
shutdown get a 503 "server is shutting down". A server unit test covers the
fix; it has not been re-run in a window. In the same window, the next
restart restored the conversation from its continued checkpoint at 303,104
tokens (14.08 s to first token).

The receipts have each cell.

## DeepSeek V4.1 Flash on M3 Ultra — September 18 update

This branch now blends its prior optimizations with a MXFP4 combination of scalar decode,
prefill and native DSpark improvements, incorporating pieces of [kernelpool's V4.1 work](https://github.com/antirez/ds4/pull/1073).
Accepted optimizations and the cost-aware controller are enabled by default on
supported local Metal paths. Reasoning stays serial; DSpark handles eligible answers and tool output.

Same 512 GB M3 Ultra, same native MXFP4 weights and matching prompts; best valid runs:

| Workload | Previous public implementation | Current release |
|---|---:|---:|
| Serial decode, 8k, DSpark loaded | 34.43 | **35.80 t/s** |
| Serial decode, 62k, DSpark loaded | 34.17 | **34.91 t/s** |
| DSpark, directed code | 41.58 | **45.11 t/s** |
| DSpark, final answer A30 | 39.13 | **42.19 t/s** |
| Prefill, 63,488 tokens | 840.58 | **846.05 t/s** |

The earlier Q4 release used different weights, prompts and generation budgets;
its 35.9/48.7 t/s headlines are historical and are not an apples-to-apples baseline
for this table. Both implementations here use native MXFP4. See the
[release notes](docs/RELEASE-MXFP4-20260918.md) for means, best runs, methodology,
setup, controller selection and validation limits.

Implementation and failed experiments: [decode](docs/METAL-V41-DECODE.md),
[prefill](docs/METAL-V41-PREFILL.md), [DSpark](docs/METAL-V41-DSPARK.md).
Runtime options, server policy and the diagnostic switches: [RUNTIME-V41.md](docs/RUNTIME-V41.md).
The [coding-agent prompting guide](docs/PROMPTING-V41.md) still applies: draft
code in real files, then inspect and iterate, rather than writing the implementation in reasoning.

---

<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming. SSD streaming is also needed in order to run very
  large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/IngeniousIdiocy/ds4-v41-m3ultra.git
cd ds4-v41-m3ultra
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 and V4.1 Flash DSpark need a
matching support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
