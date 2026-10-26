# 04: PR 15–49 age-matched run

**What to build:** Run the same PR BP model on the age-matched 15–49 subset so the effect of the reduced feature set can be separated from the effect of adding older adults. Use the same cluster partition and the same config, changing only the sample filter.

**Blocked by:** 03.

**Status:** resolved

- [x] The run filters to `age_15_49 == True` and otherwise uses the PR 15+ config.
- [x] The BP-valid 15–49 sample reconciles to 1,290,219 rows and 112,898 positives.
- [x] The run uses the same split assignment as the PR 15+ model, filtered to 15–49 rows, with no re-splitting.
- [x] Calibration and the screening threshold are fitted independently on 15–49 out-of-fold predictions.
- [x] The full artifact set is produced.
- [x] Run metadata and the results report distinguish this run from the PR 15+ run.
- [x] A test asserts the 15–49 sample filter and the reuse of the 15+ partition.

## Comments

The age-matched run is configuration on top of the shared pipeline; no second modeling path was added.

- `pr_15_49_config` (`src/screener/pr_model.py`) adds a single `SampleFilter(column="age_15_49",
  values=(True,))` on top of the PR 15+ filters and otherwise passes the same feature set, target,
  grouping, weight column, and non-age subgroups. The previous `age <= 49` filter was removed, so the
  configured sample is exactly the builder's documented flag and cannot drift from
  `build_pr_features`. Age-band reporting narrows to 15–24 … 45–49.
- `reconcile_pr_15_49_sample` and `PR_15_49_REFERENCE` count the same flag-based BP-valid sample and
  check it against 1,290,219 rows and 112,898 positives. `pr_sample_summary` gained an optional
  `sample_flag` so the summary and the run filter share one definition; the reconciliation uses the
  flag alone (no second age ceiling), so a drifted flag shows up as a mismatch instead of being
  masked. `scripts/run_pr_model.py --sample 15-49` now uses this helper instead of a local
  duplicate.
- The run loads the saved `data/processed/pr_split_assignment.parquet`, which is the PR 15+ partition.
  The pipeline filters it to the run's `case_id`s and validates one-to-one coverage, so the 15–49
  rows keep their parent split and fold and nothing is re-split. The CLI refuses
  `--no-split-assignment` for the 15-49 sample, because generating a fresh partition would confound
  the comparison.
- Calibration and the screening threshold come from `run_training_pipeline`'s own out-of-fold
  development predictions for the 15–49 sample; nothing is inherited from the 15+ run.
- Config metadata now carries `sample` (`age_15_49` / `age_15_plus`), `run_role`, and, for the
  comparison, `comparison_run: pr_15plus_bp`. The shared pipeline copies these to
  `run_metadata.json` and renders a `## Run role` section in `results_report.md`, so the two runs
  are distinguishable by name, sample, and role. The 15+ config does not set `comparison_run`.

Real-data smoke run (`--sample 15-49 --smoke --smoke-rows 3000`, 20 bootstrap replicates):
reconciliation `pass` with 1,290,219 BP-valid rows / 112,898 positives, 1,290,219 rows after filters,
3,000 used, 592 test rows, split source `pr_only`, and all 5 folds present. The saved split
assignment matched the parent `pr_split_assignment.parquet` exactly on split, fold, and cluster for
all 3,000 respondents. Output was written under the temp directory, not the repo. The full real run
is left to ticket 07.

Tests: `tests/test_pipeline.py` adds
`test_pr_15_49_run_filters_the_flag_and_reuses_the_15_plus_partition`, which asserts the used rows are
exactly the BP-valid `age_15_49` sample, the run assignment equals the 15+ assignment filtered to
those rows (no re-split), the threshold confusion counts cover the run's own development rows, every
artifact exists, the age-band subgroups stop at 45–49, and the metadata/report carry the run role and
comparison. `tests/test_pr_model.py` covers the flag-based summary with flag/age disagreement, the
reference counts, the required flag column, and the distinct run roles.
`tests/test_run_pr_model_cli.py` covers the `--no-split-assignment` guard. Full suite: 99 passing.
Ruff is clean on the changed files.

A review pass tightened four points: the reconciliation now filters on the `age_15_49` flag alone; the
role prose and README no longer claim the age-band reporting is unchanged; run names and sample labels
are named constants shared by the configs and metadata; and the CLI refuses to bypass the parent
partition for the age-matched run.

