# V4.1 DSpark state and admission

## Independent changes

Session snapshots include captured target rows and the drafter attention ring. The local draft payload uses format 2; target-only format 1 cannot restore into a drafter-enabled session. Incompatible dimensions/size are rejected, and local format 2 is rejected in TP replay. This is a cache-format compatibility change: regenerate older target-only caches for DSpark.

Interior speculative rewind retains the draft frontier and restores target/drafter state and next-token logits directly. The server uses this when ending inside a verified block and falls back to its existing rebuild path when unavailable. These changes do not require the alternative admission policy.

## Controller choice

The policy and its integration are separate commits after the state fixes. Omitting them retains the original controller. The final integration selects the cost-aware controller for local Metal V4.1; set `DS4_METAL_DISABLE_V41_PUBLIC_CONTROLLER=1` before process launch to select the original controller instead. The historical environment-variable spelling is retained for compatibility.

The original controller uses a moving-average comparison and records losing cycles below 90% of its serial baseline. The alternative uses 16 serial calibration steps, a median of nine samples, three-attempt cost windows, normalized draft top probability (0.75, at least three confident proposals), a half-serial-step loss budget, and 16/32/64/128-token backoff. It verifies the trained block once admitted and keeps reasoning serial. Request, prefix and answer transitions update its state; drafter faults latch disabled for that session, with serial recovery only when state is safe.

Post-Markov vocabulary publication is a separate optional GPU API. Only the alternative policy asks for those rows; selecting the original policy avoids their writes/readback and retains its trained confidence-head input. The GPU Markov arithmetic, existing short-row matrix kernels, expert deduplication and HC epilogues are retained.

## Validation

- 72 integrated rewind cases at 8k/62k compare complete persisted state, frontier logits and four-token continuations (37,232,640 continuation-logit words).
- Post-Markov publication: 50,000 fresh cases plus 1,000 repeats; original token/confidence-head/partial-reduction outputs preserved, complete rows finite/deterministic, guards intact.
- Controller units and integrated lifecycle checks cover calibration, confidence, cost accounting, reasoning/answer transitions, request/prefix reset and fault latching.
- Each extracted code commit builds Metal host code and CPU engine objects. The independent branches are not claimed to have newly measured native throughput. Actual CUDA/distributed model execution was not run.

```sh
make tests/test_ds41_dspark_adaptive tests/test_v41_markov_post tests/test_v41_spec_rewind
./tests/test_ds41_dspark_adaptive
./tests/test_v41_markov_post 50000
# The rewind gate needs matching format-2 target/drafter prefix snapshots:
./tests/test_v41_spec_rewind MODEL.gguf DRAFT.gguf speed-bench/promessi_sposi.txt CACHE8K CACHE62K
```

No isolated controller speedup is claimed. Full-integration throughput also includes scalar and prefill changes, so it must not be attributed to this PR alone. MTP is workload-dependent and can lose on low-acceptance prose even with backoff.

## Experiments not retained

- A 64-slot GPU Markov bias cache passed exactness but added only ~0.029 t/s on code and ~0.027 t/s on an answer fixture for about 32 MiB/session. Removed.
- Alternative Q8 verification projections changed logits, had no fixed-work speed gain and produced repetitive native continuations. Removed; apparent throughput from repetition was excluded.
- The trained confidence-head score was initially considered as input to the alternative controller. It is not the normalized post-Markov probability that its threshold expects; the prototype was discarded.
- Fixed-budget tool benchmarks that continued after the first handoff and an API that silently selected serial generation were invalid MTP measurements. Those timings are excluded.
