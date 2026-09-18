> Historical Q4 release record. For the current MXFP4 implementation and setup, see [the September 18 release](https://github.com/IngeniousIdiocy/ds4-v41-m3ultra/blob/v41-m3ultra/docs/RELEASE-MXFP4-20260918.md).

# Consolidated decode evidence — September 16, 2026

This bundle records the campaign from prior UAT `bbaebc1` to validated UAT
`e3eaff1` (implementation `723e156`). The public branch carries the same retained
arithmetic, with its previous rejected-path cleanup preserved.

- [Release notes](../../docs/DECODE-CONSOLIDATION-20260916.md): accepted changes,
  current performance, limitations and rollback switches.
- [EXPERIMENTS.md](EXPERIMENTS.md): full chronological research ledger, including
  every failed, superseded and inconclusive attempt.
- `kv-query-summary-8192.json` and `kv-query-depth-summary-{62000,300000}.json`:
  four measured serial pairs at each depth; every observation, best, mean and SE.
- `kv-query-contract.json` and `kv-query-depth-contract.json`: full native output,
  restored-prefix, quiet-machine and selected-path checks.
- `release-{two,six}-code-*` and `release-six-agent-*`: fixed-row MTP checks.
- [release-serving-contract.json](release-serving-contract.json): eleven normal
  serving checks, controller work, phase hashes and timing; no conversation text.
- [verifier-bandwidth-summary.json](verifier-bandwidth-summary.json): per-run
  verification time/cycle counts and the unchanged logical-weight normalization.
- [release-rollback-contract.json](release-rollback-contract.json): forced reject
  matches serial at two/six rows and all three depths.
- `*-tier1*.log`: retained arithmetic gates and initial failed gates.
- `p1-score-*`, `pair-native-summary.json`, `cooperative-native-summary.json`,
  `shared-overlap-native-summary-8192.json`, geometry and Q8 screens: negative
  results, including quality failures and microbenchmarks that did not transfer.
- [public-manifest.json](public-manifest.json): public source/binary hashes,
  UAT identity, source audit and fresh public integration checks.

The measurements were made before publication with one resident model and
cached prefixes. Warmups are separate; fast valid runs remain in the record.
Do not add incremental gains from different screens together. Native profiling
and synthetic timings are not substitutes for uninstrumented model throughput.
The full fourteen-request answer aggregate was not rerun for this bundle.

Recorded JSON/log/TSV files preserve numerical data. Trailing blank log lines are trimmed. Machine-local path prefixes
are replaced with `LOCAL_ARTIFACTS`, `LOCAL_CHECKOUTS` or `LOCAL_HOME`.
Private commit IDs and original source hashes are provenance, not references
that a public clone must resolve. Private request bodies, full conversations,
service-control scripts, cache snapshots and rejected implementations are omitted.
Historical candidate source names identify local archived experiments; they are
not a claim that a rejected kernel is still available in the current runtime.

## Public integration checks (no model loading)

From the repository root:

```sh
make -j8 all
make test-ds41-decode-bandwidth
make tests/test_ds41_controller tests/test_ds41_dspark_adaptive tests/test_ds41_markov_cache
./tests/test_ds41_controller
./tests/test_ds41_dspark_adaptive
./tests/test_ds41_markov_cache
python3 bench/serial-fusion-20260916/compile_full_metal.py
cd bench/serial-fusion-20260916
clang -O2 -fobjc-arc -framework Foundation -framework Metal compile-full.m -o compile-full
./compile-full
```

The final check compiles the complete current Metal library and all nine new
pipeline entry points. It does not run a throughput benchmark or load weights.
The default test checks all fifteen previously/latest promoted switches,
explicit `=0` rollback and rejection of removed experiment switches.
The original 50k-draw and native gates are recorded evidence from UAT;
publication checks do not claim to rerun that entire campaign.
See [REPRODUCE-V41.md](../../docs/REPRODUCE-V41.md) for resident A/B testing.
