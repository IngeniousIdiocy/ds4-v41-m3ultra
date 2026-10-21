## DeepSeek V4.1 Flash 1017 on the M3 Ultra — October release

This branch is our Metal implementation of DeepSeek V4.1 Flash, built on
[antirez's DwarfStar](https://github.com/antirez/ds4) engine and tuned for one
512 GB M3 Ultra. On the 1017 checkpoint it decodes a single stream at **41.5 tok/s**
at 8K context and still does **40.0 tok/s at 300K**, where upstream falls to 16.2.
That is more than twice upstream's speed at every depth, and every optimization in it
is bit-exact against the kernel it replaces. With DSpark
drafting, code answers run at **47–50 tok/s**.

Same machine, same prepared public 1017 MXFP4 weights, same prompts, one request
at a time. The upstream column is DwarfStar main at `5379c1c9`. Every number is the
median of three runs, with the two engines' server lifetimes interleaved in one window:

| measurement | DwarfStar main | this branch |
|---|---:|---:|
| serial decode, 8K context | 20.00 t/s | **41.51 t/s** |
| serial decode, 64K context | 19.07 t/s | **40.90 t/s** |
| serial decode, 131K context | 18.14 t/s | **40.63 t/s** |
| serial decode, 300K context | 16.21 t/s | **40.03 t/s** |
| prefill, 8K prompt (cold) | 611 t/s | **705 t/s** |
| prefill, 64K prompt (cold) | 700 t/s | **835 t/s** |

Decode rows: the 256 tokens after the prompt, thinking off, temperature 0, each
prompt restored from its disk checkpoint, so only decode is timed. Prefill rows:
prompt tokens divided by the cold time to first token. Every prompt is a prefix of
one long document. Going from 64K to 300K costs this branch 0.53 ms per token, and
DwarfStar main 9.25 ms.

### DSpark

DSpark drafts tokens with the checkpoint's own multi-token-prediction stages,
and the target verifies them in one batched step. On code, where the drafter
is usually right, it pays well. Three code-writing requests, 512 tokens each,
thinking off, temperature 0, median of three runs per cell, all four arms in
one window:

| request | DwarfStar main | DwarfStar + DSpark | this branch | this branch + DSpark |
|---|---:|---:|---:|---:|
| LRU cache in C | 20.03 t/s | 30.77 t/s | 41.86 t/s | **50.41 t/s** |
| asyncio rate limiter in Python | 20.04 t/s | 31.35 t/s | 41.86 t/s | **49.38 t/s** |
| trie in Rust | 20.07 t/s | 28.95 t/s | 41.84 t/s | **47.09 t/s** |

This branch accepted 90.9% of drafted tokens, DwarfStar 86.4%. DwarfStar's V4.1
loader expects the drafter's two Markov-head tensors under other names
(`markov_w1` / `markov_w2`) than the checkpoint's (`embed` / `head`), which our
prepare command keeps; its DSpark arm ran on a copy of our drafter with only
those two names changed.

On prose the drafter is usually wrong. The admission controller probes,
loses, and backs off, and DSpark costs about 4% there: 39.57 against 41.31 t/s
at 8K and 39.30 against 40.75 at 64K, in the same window. `--serial-reasoning`
keeps reasoning out of drafting altogether: reasoning is decoded one token at a
time, and answers and tool calls are drafted.

### What makes it fast

Every change executes the same model, the same work, with the same
arithmetic. A token on this branch takes 24.1 ms at 8K, against 50.0 ms on
upstream main. The largest pieces:

- **Fixed shapes and once-per-step work.** V4.1's shapes are fixed, so the
  hot kernels have literal-shape entries: the output projection with RoPE
  folded in, the routed gate/up entry, the hyper-connection join and the
  q_a/KV pair. Position-only work, such as RoPE coefficients and the inverse
  rotation of the attention heads, runs once per step instead of in every
  consumer.
- **Token selection on the GPU.** The sparse-attention indexer's top-512 is
  selected on the GPU with a 512-wide sort, and the candidate list is built
  once. The CPU keeps only the decision and the RNG.
- **The indexer at depth.** The indexer's keys are exact BF16 values stored as
  F32, so its scorers read a BF16 view of them: half the bytes, the same bits.
  Candidate rows are scored two lanes per key, and a one-row top-k finishes in
  a single gathered pass. One scorer call over 300K keys went from 494 µs to
  277 µs.
  DSpark's verify rows use the same scorer.
- **The exact Engram gate.** The Engram residual gate was latency-bound at
  about 120 µs per call. Ours loads each lane's 160 terms into registers in eight
  segments and hands the running sums from one SIMD group to the next, so the
  summation order never changes. It now takes 7.5 µs per call and is
  bit-identical: 0 of 3,072,000 fresh test streams differ.
- **Narrow copies built at load.** The BF16 router and the E4M3 Engram and
  indexer q_b copies are built from the public weights when the model loads,
  and only kept where they reproduce the wide values exactly. Their readers
  ship as one unit.
- **The host chain.** The next token's first command buffer is encoded in the
  CPU slack before the wait, a short GPU bridge keeps the GPU busy across the
  hand-off, the sampler's scan uses NEON, and the draft capture rides in the
  head's command buffer.

[V41-1017-HISTORY.md](docs/V41-1017-HISTORY.md) has the full route, with each
lever's measured gain and the many that did not pay.

**Exactness.** In the release window, every run gave the same token ids, final
logits and whole-session state digests as the previous build, at every context.
Each changed kernel is also tested bit for bit against the one it replaces, at
the production shapes (`tests/test_deepseek41_metal`). Every optimization has a
switch that turns it off (`DS4_METAL_DISABLE_V41_*`, listed in
[RUNTIME-V41.md](docs/RUNTIME-V41.md)); turning off the nine added in this
release gives the previous build's output bit for bit. DSpark's batched verify is, by the engine's design, not bitwise equal
to one-token decode, and the server says so at startup.

**Ideas we adopted.** This branch uses ideas and code from:

- **Ivan Fioravanti** ([ds4-metal](https://github.com/ivanfioravanti/ds4-metal)):
  - the fixed-shape decode mechanism, which our literal-shape kernels extend;
  - building the narrow decode copies at load;
  - the E4M3 shift decode;
  - the exact prefill indexer MMA (+5.4% prefill at 64K).
- **Tarjei Mandt** ([kernelpool](https://github.com/kernelpool/ds4)):
  - the V4.1 MXFP4 conversion, expert and verification kernels
    ([antirez/ds4 #1073](https://github.com/antirez/ds4/pull/1073));
  - candidate-block index scoring;
  - the radix top-k select.

Their commits keep their authorship, and
[THIRD_PARTY-V41.md](THIRD_PARTY-V41.md) lists every attributed piece.

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
`--verify` re-reads both files and compares every tensor with the digests pinned in
`gguf-tools/deepseek41_prepare.json`. `--dry-run` checks inputs, free space and
Python modules without writing.

Needs: 543.9 GB free beside `--out`, plus the 509.5 GB of inputs if they share the
volume; delete the inputs after `--verify` passes. Python 3.10+ with numpy,
tokenizers and sympy, and `cc` and `make`. On an M3 Ultra with inputs and outputs on
one external Thunderbolt NVMe drive: merge 5.7 min, drafter 10 s, verify 2.3 min.

### Run

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

- The server listens on 127.0.0.1 only. Pass `--host 0.0.0.0` to serve other
  machines; it has no authentication.
- `--kv-disk-dir` keeps prompt checkpoints on disk within the
  `--kv-disk-space-mb` budget. A 300K-token checkpoint is about 1.8 GiB.
- Right after start the server runs a one-shot numerics canary (about 7 s);
  requests sent in that time wait for it, then use the cache.

[RUNTIME-V41.md](docs/RUNTIME-V41.md) lists every option and switch. No
environment variable is needed.

### Restart and the disk cache

Measured on this release with the launch above, restarting the server on the
same cache directory:

- **300K restore.** A 303,912-token conversation took 388.5 s to first token
  cold. Sent again right after a restart, it restored from its checkpoint in
  11.0 s, including the canary.
- **Extending a restored prompt.** A request that restores part of its prompt
  from disk and prefills the rest now saves a checkpoint at its new length.
  A 64K prompt extending a restored 8K checkpoint saved its own 64K
  checkpoint; after a restart it restored 65,536 of its 67,314 tokens in
  5.6 s instead of 72.2 s, with identical output. Upstream's server saves a
  checkpoint only when nothing was restored.
- **Requests during the canary** wait for it, then run in order. Three sent
  together (300K, 8K, short) were answered at 10.9, 14.5 and 15.3 s.
- **Shutdown with a request queued** exits in 6.2 s, keeps every cache entry,
  and answers the queued request with a 503.
- **Old entries** from an earlier build's canary are refused and evicted under
  the disk budget.

### Test

```
make ds4_test tests/test_deepseek41_metal
./ds4_test --server                       # server unit tests, no model
./tests/test_deepseek41_metal             # V4.1 Metal kernels, no model
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-MXFP4.gguf ./ds4_test
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-MXFP4.gguf \
    DS4_TEST_DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf ./ds4_test --dspark-verify-depth
```

A fresh clone of this release passed every command above, built per Run, and
served the launch above (window r1cc-a5). The full evidence for every number
on this page is in [RECEIPTS-V41-1017.md](docs/RECEIPTS-V41-1017.md).

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
