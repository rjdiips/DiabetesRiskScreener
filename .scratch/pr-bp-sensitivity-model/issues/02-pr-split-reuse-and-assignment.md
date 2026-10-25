# 02: PR split reuse and assignment

**What to build:** Give the PR runs a leakage-free cluster-grouped split assignment that reuses the primary split wherever a cluster overlaps. The assignment is saved as an artifact, records whether it was reused, PR-only, or mixed, and supports filtering to ages 15–49 without changing the partition. This is the only new modeling logic in the PR feature; everything after it is configuration plus reuse of the primary pipeline.

**Blocked by:** 01; primary-model-pipeline (external spec, not yet implemented).

**Status:** ready-for-agent

- [x] The PR split assignment reuses the primary cluster split and fold for overlapping clusters.
- [x] Clusters that do not appear in the primary assignment are assigned deterministically with the same seed and stratification rules.
- [x] Run metadata records the split source as `reused_primary`, `pr_only`, or `mixed`.
- [x] Every PR respondent is assigned exactly one split and one development fold.
- [x] No cluster appears in more than one split or development fold.
- [x] No household appears in more than one split or development fold.
- [x] The 15–49 assignment is the same partition filtered to 15–49 rows, with no re-splitting.
- [x] Fold prevalence and sex composition are reported.
- [x] The split assignment is saved as an artifact and is deterministic for a fixed seed.
- [x] Tests cover overlapping-cluster reuse, PR-only clusters, the leakage assertions, and the 15–49 filter.

## Comments

Implemented `src/screener/splits.py` as the shared split seam. `assign_splits` owns the
primary cluster-grouped 20% holdout and 5-fold rule; `assign_pr_splits` accepts an optional
primary assignment, reuses the primary split and fold for every overlapping cluster, and
assigns PR-only clusters with the same seed and stratification rule (cluster-level target
prevalence and female share). `assignment_source` records per-row provenance and the summary
records the overall `split_source` as `reused_primary`, `pr_only`, or `mixed`.

The test phase fills the global holdout quota across strata before folds are assigned, so
reused primary test clusters cannot drag the overall holdout away from 20%. Folds use a
combined stratum and global row deficit to stay balanced when the primary folds carry unequal
PR rows. Household identity is the `(cluster, household)` pair because DHS `hv002` household
numbers repeat across clusters.

Added `scripts/build_pr_split.py`, which writes `data/processed/pr_split_assignment.parquet`
(local only) and `data/pr_split_summary.json` (aggregate). The summary reports split sizes,
fold prevalence, sex composition, the 15-49 slice, leakage checks, and the split source.

Real-data PR-only run: 1,808,013 rows, 30,125 clusters, 614,792 households, 361,364 test rows
(19.99%), fold sizes 289,270 / 289,355 / 289,362 / 289,339 / 289,323, fold prevalence
0.1415 / 0.1395 / 0.1392 / 0.1408 / 0.1401, female share 0.531-0.533. The 15-49 slice has
1,291,831 rows and reuses the same slots. Leakage checks pass for clusters and households.
The run is `pr_only` because the primary pipeline has not produced a primary assignment yet;
synthetic tests exercise the `reused_primary` and `mixed` paths. Full suite: 58 passing.

Final validation rejects a partial or degenerate partition (no test rows, no development
rows, or an empty development fold) and a summary whose assignment does not cover exactly
the feature-table respondents. Two independent runs produced byte-identical artifacts, and
the 15-49 summary reports any fold that the younger slice does not cover.
