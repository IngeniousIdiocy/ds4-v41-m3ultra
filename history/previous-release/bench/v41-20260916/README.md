# September 16 decode release evidence

This bundle accompanies [the release update](../../docs/RELEASE-V41-20260916.md).
The original launch evidence and manifest remain separately available in
[RELEASE-EVIDENCE-V41.md](../RELEASE-EVIDENCE-V41.md).

## Public integration

- [manifest.json](manifest.json) binds the validated UAT commit/binary, public
  implementation commit, built binary, changed source hashes, and checks.
- [production-path-checks.txt](production-path-checks.txt) records the fresh
  model-free exactness/default/rollback checks on the public integration.
- Before changing defaults and culling rejected alternatives, 129 source files
  matched UAT token-for-token after excluding comments and whitespace. After
  culling, the five changed Metal sources differ only by deletions. Retained
  kernel arithmetic was not rewritten during publication.
- Public defaults enable all six paths that UAT selected explicitly. Tests
  check default-on behavior, explicit `=0` rollback, removed switch rejection,
  and complete production-path output against controls.

## Measured UAT results

- [release-summary.json](release-summary.json): final cached-depth, normal API,
  held-out agent, and original-public-to-UAT comparisons.
- [integrated-matrix-summary.json](integrated-matrix-summary.json): both timed
  observations per arm for the final cached-prefix ABBA matrix.
- [agent-summary.json](agent-summary.json): all 14 requests, predetermined arm
  selections, exact completion counts, phase durations, and pooled calculations.
- [rollback-validation.json](rollback-validation.json): forced rejection at two
  and six rows across 8k/62k/300k, with matching continuation hashes.
- [adoption.json](adoption.json): final A2/A3/A4/A6 adoption decision and verifier
  timing, including the rejected and screening-only alternatives.

The A/B controls in these final-campaign files are the immediately preceding
UAT, not the original public release. The cumulative public comparison is
identified separately. Default-off labels in historical records describe the
old candidate binary; current public defaults are on.

Performance figures were measured in UAT. This release integration was built
and checked without loading another copy of the large model or disrupting the
serving instance. Its new binary was not separately benchmarked. Synthetic
test timings are not substituted for full-model measurements.

## Failed and superseded experiments

The [chronological ledgers](../../docs/experiments/v41-20260915/README.md) cover
all post-launch campaigns, including invalid runs and compiler failures. They
preserve the distinction between native screening, full fidelity gates,
resident model tests, and final adoption. Rejected implementations are not
shipped as selectable experiments.

Private agent requests, full generated responses, local KV caches, process
identities, and machine-restoration scripts are not distributed. Numeric
summaries here were copied from retained experiment records; the original raw
records remain archived locally. A filesystem name mentioned in an archived
ledger is not a promise that its corresponding private artifact is included.
