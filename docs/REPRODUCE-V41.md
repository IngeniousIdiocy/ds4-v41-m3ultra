# Measuring DeepSeek V4.1 on Metal

This page says how the numbers in the README were taken, so they can be taken
again. It covers the quiet machine, decode from saved caches, cold prefill at
several depths, comparing groups inside one process, follow-up turns, image
first tokens, the accuracy of the defaults that are not bit for bit, and
several sessions decoding at once.

The machine was a Mac Studio with an M3 Ultra (80 GPU cores) and 512 GB of
memory. The weights were the MXFP4 build of the October 17 checkpoint and its
DSpark drafter, prepared as in the README.

## 1. A quiet machine

A few percent of background GPU use moves decode speed by a few percent, which
is the size of most single steps. So no number is taken until the machine is
quiet, and every comparison runs inside one quiet session.

Before a session:

- Stop every other model process: servers, agents, benchmarks. Only one model
  process runs at a time, and the next starts only after the last has exited and
  its wired memory is released.
- Pause the background media analysis services (on macOS these include
  `mediaanalysisd`, `photoanalysisd` and the video decoder services) with
  `kill -STOP`, and resume them with `kill -CONT` afterwards.
- Put the displays to sleep (`pmset displaysleepnow`), and check afterwards that
  they stayed asleep for the whole session.
- Check quiet before starting: sample the GPU's utilization once a second for
  ten seconds (`ioreg -r -d 1 -c IOAccelerator`, the "Device Utilization %"
  field) and require every sample at 5% or less, with a one-minute load average
  of 6 or less and no other model process. Wait and check again if it is not.
- Keep the GPU utilization logging during the session, so a disturbance shows up
  next to the run it touched.

During a session:

- Interleave the configurations being compared in serpentine order (A B B A, or
  A B B A A B), so slow drift falls on both sides.
- Time each configuration more than once. The README's decode figures are the
  mean of two timed passes, which agreed to within 0.1 t/s at all but one step.
- Compare only numbers taken in the same session. Numbers from a session with
  background load are lower across the board and are only good for steps
  measured within it.

## 2. Decode from saved caches

Decode speed is measured after the prompt has been restored from a saved cache,
so that prefill and model loading are not timed.

1. For each prompt and depth (for example 8K and 131K tokens of the prose
   prompt, 8K of a code prompt), prefill once on the build under test and save
   the session payload. Keep the caches and write down how they were made: the
   commit, the model and drafter files, the context size and the prefill chunk,
   and the hash of the logits after prefill.
2. Restore the cache in a fresh session, decode once untimed so every pipeline
   is built, restore again, and time 128 greedy tokens. Report the steady rate:
   the wall time of tokens 2 to N. The README's decode figures were taken with
   the drafter loaded, as a server would load it.
3. For DSpark, decode the same way with speculation on. A payload that holds
   the drafter window starts drafting at once; an older payload decodes 128
   serial tokens first, so do not time those. The README's DSpark figures for
   this branch come from caches saved with the drafter window.
4. Caches are tied to their engine's payload format. This branch reads main's
   payloads and extends them with the drafter window; the previous release used
   a format of its own. Make each engine's caches with that engine.
5. A cache also holds its prompt's token ids. This branch pre-tokenizes text as
   `tokenizer.json` does and main does not, so the prose prompt is 417,059
   tokens here and 419,509 on main. Decoding from a restored cache does not
   tokenize anything, so main's caches still serve for decode; a prefill that
   continues a cache's text must use a cache made with the same tokenizer.

`ds4-bench --jobs FILE --token-csv PATH` does this in one resident process: a
`save` job prefills a fixture to a depth and writes a cache (snapshot format 4,
checked on every load), a `load` job restores one, and each `run` job restores
the frontier from memory and decodes. `PATH.runs.csv` and `PATH.jobs.csv` record
the speed and hashes of each run. See [performance](PERFORMANCE.md) and the
comment at the top of the job runner in `ds4_bench.c`.

## 3. Cold prefill at each depth

Prefill is timed from an empty session with the model already loaded and warm.
The README's prefill numbers were taken without the drafter.

- Use prompts that are prefixes of one long document, so each depth is the same
  text cut shorter. The README's prose prompt is `speed-bench/promessi_sposi.txt`.
- For a single depth (32K), time one cold prefill of that many tokens, in a fresh
  session each repetition, and compare the logits after prefill across builds.
- For several depths at once, run one cold prefill of the deepest prompt with the
  default 8,192-token chunk and record the time at the end of each chunk. One
  131K prefill gives the time to 32K, 64K, 98K and 131K. These are times to reach
  each depth inside one long prefill, not separate cold prefills of each prefix,
  because a long prefill plans its sweeps for the whole prompt.
- Run main and the build under test in the same session, as for decode.

The first sweep of a prefill waits for its Engram rows from disk, so prefill
times depend on the SSD. Keep the GGUF on a fast internal or Thunderbolt NVMe
drive, and do not compare numbers taken with the model on different drives.

## 4. Comparing groups in one process

There are two ways to turn groups off.

**Separate processes.** Set the group's switch in the environment of its own
process, `DS4_METAL_DISABLE_V41_<GROUP>=1`, and interleave those processes with
default ones. This is the simplest way and needs nothing but the binaries. It
pays the model load for every process, so it suits a few configurations.

**One process, the switch mask.** The switches are read once into one mask with
one bit per group; [RUNTIME-V41.md](RUNTIME-V41.md) gives each group's bit. The
engine exports `ds4_gpu_test_set_v41_off(uint64_t mask)`, which replaces the
mask; the Metal unit test uses it to run each check with a group off and on. A
small driver linked against the engine can load the model once, keep the saved
payload in memory, and for each configuration set the mask, restore the payload
and decode. Run every configuration once untimed first with a hash of each
step's logits: that is the exactness check, and it builds every pipeline before
anything is timed. Then time the configurations in serpentine order.

The mask setter changes which code runs from the next dispatch on. A few groups
decide something at load and are not fully switchable afterwards:

- MODEL_TRACKED chooses how the weight views are created at load.
- NARROW_WEIGHTS builds its copies at load only when it is on at load. Turning
  it off later works (the readers check the bit on every use); turning it on
  later does not build the copies.
- TP_BALANCED_ROUTE decides what each Mac maps, at load.

`ds4-bench --jobs` can set environment variables per `run` job (`set=NAME=VALUE`,
`unset=NAME`), but the group switches are read once, so for them that only works
through separate processes or the mask setter.

## 5. Finding idle time

Several groups came from looking at where the GPU waited, not at kernel times.
Main's `DS4_METAL_ENCODER_TIMELINE` prints each command buffer's encode time, GPU
start and GPU end. A gap between one buffer's end and the next one's start is
time the GPU spent waiting for the host. That is how the 30 ms wait in front of
each DSpark verify block (a synchronous Engram read) and the 0.4 ms mid-token
wait for the first Engram table were found.

## 6. Exactness checks

For every group that claims to keep the output, these were compared between
builds, or with the group off and on:

- Serial decoding: a hash of the tokens and the logits at each of 128 greedy
  steps, from main's own saved caches (8K and 131K prose), against DwarfStar
  main, at every step of the README's progression.
- Cold prefill: the logits after a 32K prefill, at every prefill step.
- Strict DSpark (`--dspark-strict`): the tokens, which must equal serial
  decoding's.
- Default DSpark: the tokens and the verify logits, which with the defaults
  equal serial decoding's (see the README's exactness section).

The unit checks in `tests/test_deepseek41_metal --v41-groups` compare each fast
kernel with the one it replaces at V4.1's shapes, without the model.

With `--exact` (or `DS4_V41_EXACT=1`), the same comparisons hold against main
with the faster defaults on: the logits after cold prefills of 8, 8K, 32K and
131K tokens, 256 greedy steps from main's 8K cache, and the same 256 steps with
the drafter loaded and DSpark on.

Main and this branch now tokenize the prose prompt differently (section 2), so
a cold prefill of the same text gives different ids, and its logits cannot be
compared across the two. For this update the `--exact` prefills were compared
with the previous update of this branch, which tokenizes the same way; decoding
from main's saved cache needs no tokenizer and is compared with main directly.

## 7. Follow-up turns

A follow-up turn is a prompt appended to a saved session, as a chat's next turn
is. Prefill 8K tokens of the prose prompt with the build under test and save
the session, then restore it and time the prefill of the next N tokens of the
same text (for example 1K, 2K and 4K), restoring again for each repetition.
Main's figure comes from main's own 8K cache. The README's figure is the 2K
follow-up, without the drafter.

## 8. Image first token

Image figures are timed through `ds4-server` with one session (no
`--batched-session`, `-c 393216`), the drafter, the disk KV cache and
`--utility-ctx 32768`, plus `--vision` with the 1017 vision file. Send each request to a server that
has already answered a short text request, with an empty disk KV cache
directory so nothing is restored: one image, the text "Describe this image."
and `max_tokens` 1. Time from sending the request to the end
of the response, which is the first token. Use four different images of each
size, none sent before, and take the median. The sizes are 3840x2160 and
1400x900 screenshots and a 512x507 image, all PNG; they give 943, 748 and 169
encoder rows and prompts of 1,002, 806 and 218 tokens.

To split a first token into its parts, the server's log gives the vision
embedding time and the prompt's prefill time for each request.

## 9. Accuracy of the non-exact defaults

PREFILL_MMA and CED are compared with the exact prefill in two ways, and
ATTN_TILES with main's decode attention by the long scoring cases and by greedy
runs.

- **Follow-up chains.** Prefill 3K tokens exactly and save the session. Then,
  for each configuration, restore it and append 1K tokens at a time up to 32K,
  keeping the logits after each append (29 points). Report the full-vocabulary
  KL divergence of each configuration's logits from the exact ones at every
  point, its mean, and whether the top token is the same. The prose prompt and
  a code prompt (the first 400 KB of `ds4.c`) were both used.
- **Long scoring cases.** `gguf-tools/quality-testing/score_official` scores
  DeepSeek's own continuations of 21 long V4.1 prompts, teacher-forced at a
  131,072-token context, and reports the mean negative log-likelihood per
  token. The 21 cases are the long ones of two suites in
  `gguf-tools/quality-testing`, listed in `deepseek-v4.1-flash-long21.tsv`:
  `make gguf-tools/quality-testing/score_official`, then
  `gguf-tools/quality-testing/score_official MODEL gguf-tools/quality-testing/deepseek-v4.1-flash-long21.tsv OUT.tsv 131072`
  from the repository root, once per configuration with the switches in the
  environment.
  The paired difference between two configurations, with its standard error
  over the 21 cases, is the figure the docs give.
- **Greedy runs to the first difference** (ATTN_TILES). Decode 16 prompts
  greedily with the group on and off, from the same caches, keeping the full
  logits of every step, and stop each prompt at the first step whose tokens
  differ. Report the KL divergence of the two logits at every step before it,
  its mean and maximum, and how many prompts reached a difference.

## 10. Several sessions at once

The server figures are taken through `ds4-server` with the launch the README
shows, plus `--vision` with the 1017 vision file: four batched sessions, the
drafter on, temperature 1 and top_p 1. Each conversation is a raw
`/v1/completions` prompt, a 2K- to 3K-token slice of a C source file behind a
header of its own, so no two share a prefix.

- **Warm, then decode together.** Send each conversation once with
  `max_tokens` 1, so its prompt is prefilled and resident. Check that the
  machine is quiet, then send the N measured requests at the same moment,
  streamed, 256 tokens each, each with its own seed, and record the arrival time
  of every streamed token. A session's rate is its tokens after the first over
  the time from its first token to its last; the figure for all sessions is the
  tokens that arrived in the window where all N were decoding, over that window.
  Host sampling and the server's own work between steps are inside these times.
- **One session.** The same, with one conversation, on the same server.
- **How much was drafted.** With `DS4_SERVER_BATCH_LOG=1` (main's variable)
  the server logs a line per step; this branch adds how many tokens the
  drafting session kept in that step (`block=`, 0 when no session drafted) and
  each session's controller state (`states=`, `b` for a back-off after a
  loss). The README's drafting counts come from these lines.
- **A cold prompt beside decoding sessions.** Three conversations decode greedily
  while, 3 s in, a fourth request arrives with a cold 9,240-token prompt. Record
  every gap between two tokens of the same decoding session while the cold
  prompt prefills, and report the 99th percentile and the longest.

The comparison with oMLX is decode only, greedy, DSpark off, in one resident
process per engine. ds4-bench restores N saved sessions (`session` jobs) and
decodes them together (`run nrun=N`), 128 tokens, and reports milliseconds per
step for all N rows; each N runs three times in serpentine order and the median
is taken. The prompts are the same 1,024 token ids on both engines, in two
sets of eight: prompts made the way oMLX's own benchmark makes them (its source
code behind a unique header, so rows share much of their text), and prompts of
distinct text. oMLX
v0.7.3 ran DeepSeek's released FP4 and FP8 weights of the same checkpoint
through a small harness over its `BatchGenerator`: one batched prefill of the N
rows, then 127 steps of N rows, the same shape as `nrun`, timed the same way,
in a quiet session of its own. oMLX's own
benchmark command was not used, because it uploads its results.

Batched rows are checked against each session decoded alone: decode the
sessions together with a hash of every row's full logits at every step, then
each session alone from the same cache, and compare the hashes, at two to
eight rows and at short depth and 34K to 300K.
