# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md)

## Start locally

```sh
./ds4-server --ctx 32768
```

The default address is `http://127.0.0.1:8000`. Use `--host 0.0.0.0` to listen
on other interfaces. Restrict access to trusted clients; for an Internet-facing
deployment, put authentication and TLS in front of the server.

`--cors` enables browser cross-origin headers. It does not change the listening
address or provide access control.

The selected build chooses the backend. Pass `-m FILE` for an explicit model.
Use `--chdir /path/to/ds4` when starting outside the project directory so
relative runtime files such as Metal kernels can be found.

## APIs

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Loaded model information |
| `POST /v1/chat/completions` | OpenAI-style chat |
| `POST /v1/responses` | Responses-style requests and continuations |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-style messages |

The Flash and PRO names accepted by the model endpoints are compatibility
aliases, not separate loaded models. The GGUF passed at startup selects the model.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Explain Redis streams."}],"stream":true}'
```

Chat, Responses, and Anthropic support tools and SSE streaming. Reasoning is
returned separately from visible text in each API's native form. Standard
sampling and output-budget fields are supported; explicit request parameters
take precedence over defaults.

The default sampling settings are temperature 1, top-p 1, and min-p 0.05.
For DeepSeek, thinking is on by default. `reasoning_effort=max` selects Think
Max only with sufficient context; otherwise it falls back to normal thinking.
`xhigh` maps to normal thinking, not Think Max. Use `think:false`, a disabled
thinking object, or a non-thinking model alias for direct answers.

## Multiple sessions

```sh
./ds4-server --ctx 4096 --batched-session 4
```

Without `--batched-session`, there is one resident session. With it, the server
preallocates independent KV states and queues requests when all slots are busy.
Choose context and slot count together: a context that fits once may not fit
four times. Idle slots can be cached before reuse; active requests are not evicted.

Where the model supports it, the slots share one prefill workspace instead of
each keeping its own, so an extra slot costs only its caches. That matters most
for Qwen3.8 Flash Next, whose transients are sized by the prefill chunk rather
than by the context: at the default chunk they run to several GiB per session.
The startup line reports both figures.

| Backend/model | Decode execution |
| --- | --- |
| Metal, resident Flash | Native shared-expert/QKV batching where supported |
| Metal, resident V4.1 Flash | Native decoding for 2-8 sessions |
| Metal RDMA TP, V4.1 Flash | Native decoding for 3-8 sessions; ordered fallback for two |
| Metal SSD streaming, V4.1 Flash | Ordered fallback |
| Metal, GLM 5.2 | Ordered fallback |
| Metal, GLM 5.3 | Native batching through 2051 visible tokens; ordered fallback afterward |
| Metal, Qwen3.8 Flash Next | Native batching of the shared work; recurrent state, caches and PLE history stay per session |
| CUDA, supported multi-GPU Flash TP layout | Native grouped decode and mixed prefill/decode |
| Single-GPU CUDA, including Spark | Ordered fallback |

Fallback executes the rows separately. It provides concurrency and scheduling
fairness, not the aggregate speedup of native batching. Native grouping may
change floating-point reduction order slightly. V4.1 sessions containing images
use the ordered fallback.

Long prefills yield to active decoders in bounded intervals, normally 128
tokens. `--mixed-prefill-quantum N` changes that interval for testing.
Session-batched serving uses ordinary target decoding, except Qwen3.8 on
Metal, where `--mtp` also batches speculative decoding. Its
`--mtp-exact-sampling` mode uses ordinary batches for nonzero-temperature
requests. Other models do not use MTP/DSpark while session batching is active.
For the eight-L40S example, see [CUDA GPUs](CUDA_MULTI_GPU.md#serve-multiple-users).

## Request defaults, utility requests and serial reasoning

```sh
./ds4-server -m gguf/DeepSeek-V4.1-Flash-MXFP4.gguf --ctx 393216 \
  --default-top-p 0.95 --default-min-p 0 --default-reasoning-effort high \
  --utility-ctx 32768 --serial-reasoning
```

`--default-top-p F`, `--default-min-p F` and `--default-reasoning-effort E`
(`low`, `medium`, `high` or `max`; default `high`) set what a request gets
when it names none. A request's own values, thinking switches and model
aliases still win. DeepSeek V4.1 renders `low` as effort level 50, as
DeepSeek's encoder does.

`--utility-ctx N` (default 0, off) keeps a second session of N tokens for
requests that set `"ds4_utility": true`, such as titles, tags or follow-up
suggestions. They never think, never draft and never read or write the disk
KV cache, and they leave the main session's state alone. A utility request
whose prompt plus `max_tokens` does not fit in N tokens is refused. It needs
non-batched mode.

`--serial-reasoning` (default off) makes DeepSeek V4.1 decode every reasoning
span serially; DSpark then drafts only answers and tool calls. It changes
nothing for other models or in batched mode.

## Images

Start with the matching language GGUF and `--vision FILE`; see
[model-specific instructions](MODELS.md#vision).

OpenAI chat and Responses accept inline PNG/JPEG data URIs. Anthropic accepts
base64 image sources. Remote URLs and server-side file paths are rejected.
Image blocks preserve their order in the request. The limit is 16 images and
a 64 MiB HTTP body.

## Disk KV cache

Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.

TP cache loading rebuilds the saved token prefix on both ranks. It is not an
instantaneous restoration of both GPUs. Pipeline loading redistributes the
saved layer state over its route.

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Check `./ds4-server --help` for their defaults.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.

Each checkpoint records the weights that wrote it (a hash of the model and
drafter GGUF headers and tensor tables), the weight form prefill reads, and a
hash of its payload. A file from other weights, an older format or with a
damaged payload is refused before it touches the session, and is evicted under
the space budget like any other entry. After it starts listening, the server
prefills a fixed 4,096-token prompt once in a side session (the numerics
canary) and binds every checkpoint to the hash of its logits, so entries
written under other kernels or switches are refused too. Requests are accepted
meanwhile but wait until the canary has finished; `KV disk cache on: canary`
in the log marks it. If the canary fails, the cache stays off for the life of
the process and a warning says why. `make ds4-server-testhooks` builds a test
server in which `DS4_KV_CANARY_FAIL_TEST=1` forces that failure.

A request that restores a prefix from disk and extends it by at least one
continued interval saves its own deeper checkpoint (log line `kv cache extends
restored prefix`). A prefill interrupted by shutdown or by the client keeps the
disk entry it started from; during shutdown such a request gets HTTP 503.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Tool history and debugging

For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace /tmp/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../ds4_kvstore.h) and
[ds4_kvstore.c](../ds4_kvstore.c); model-specific payload handling is in
[ds4.c](../ds4.c).
