# V4.1 coding agents: draft in files, then test and revise

For coding tasks with filesystem tools, tell DeepSeek V4.1 Flash to use real
files as its drafting surface. Ask it to write a small runnable draft, execute
it, and make the next edit from observed results. Also settle the important
design choices in the prompt so it has fewer decisions to reopen.

In a local game-building experiment, the model spent substantial reasoning
output planning implementation details, reconsidering choices, and repeatedly
announcing that it would start writing. Explicitly asking it to draft directly
in files helped it get to implementation sooner and finish with much less
generated output. It still planned before writing; this instruction is useful
guidance, not a hard limit on reasoning.

## Copyable instruction block

Add this after a concrete description of the artifact you want:

```text
Use actual files as your drafting surface.

- Treat the requirements and design choices above as settled. Do not
  brainstorm alternatives unless a concrete incompatibility prevents the task.
- Keep initial planning brief: identify the next edit, then make it.
- Do not compose the full implementation in reasoning and then transcribe it.
  Write the first draft directly into the target file.
- Build incrementally. Start with the smallest runnable version, then add the
  required behavior in a few concrete edits.
- After each substantial edit, inspect or run the file. Use observed behavior
  and errors to decide the next edit.
- When unsure about code, write a small implementation and test it instead of
  repeatedly simulating it mentally.
- Avoid repeatedly announcing that you are about to write. Make the edit.
- Choose the simplest reasonable implementation. Revisit it only when testing
  reveals a problem or a requirement is unmet.
- Check existing tooling briefly. Browser or test-tool setup should not delay
  creating the first runnable draft when it is not needed for that edit.
- Fix concrete failures and stop when the requested scope works. We will
  iterate after I try it.
- Report what you tested, the artifact location, and any remaining limitations.
```

Give one consistent set of requirements. For example, choose either a fixed
canvas resolution or a viewport-derived height; do not append both and call
each final. Specify the genre, controls, first-level scope, asset constraints,
and target device rather than asking the model to invent all of them.

## Observed game-building runs

Two completed runs on the same 512 GB M3 Ultra Mac, using the September 16,
2026 V4.1 UAT deployment and high reasoning effort:

| Measurement | Open-ended game request | Directed request, draft in files |
| --- | ---: | ---: |
| First tool call | 16:58 | 0:08 |
| First HTML create call | 24:14 | 7:10 |
| Total turn elapsed | 1:12:31 | 30:51 |
| Generated tokens, including reasoning | 118,662 | 45,772 |
| Recorded reasoning characters | 236,711 | 51,184 |
| Measured aggregate decode | 34.4 t/s | 38.7 t/s |
| Completed tool calls / errors | 68 / 0 | 61 / 0 |

The first request asked for a self-contained retro game of the model's choice,
one level, with an iPhone interface. The directed request specified a portrait
space shooter, relative dragging and automatic firing, simple rectangle art,
standard monospace text, sound effects without music, and a bounded first
level. It explicitly asked for file drafts followed by inspection and tests.

These are observations from two different prompts, not a controlled A/B test
isolating one sentence. The scope became more specific, the generated games
were different, and tool availability changed: the second run made 13 vision
and four audio calls after the local multimodal service was restored. Browser
tooling/cache state also differed. The evidence does not establish identical
game quality or guarantee the same time reduction on another task.

The times above come from the saved turn and tool-call records. A create call
is recorded after the model finishes generating its arguments; it is not the
instant it starts emitting code. Reasoning characters are saved text length,
not a tokenizer count. Decode t/s uses total generated tokens divided by
measured server decode time, excluding prefill and tool execution. The changed
reasoning/code mix can change that aggregate even without a kernel change.

These UAT observations do not replace this public release's benchmark table.
The practical target here is less redundant generation and earlier runnable
artifacts. Track time to first file, time to a tested result, reasoning volume,
and correctness alongside tokens per second.
