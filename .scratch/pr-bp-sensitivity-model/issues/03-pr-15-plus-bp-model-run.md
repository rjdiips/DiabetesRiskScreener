# 03: PR 15+ BP model run

**What to build:** Run the PR 15+ BP model by reusing the primary training pipeline with a PR config. The default model uses the 12 BP/demographic features, produces its own calibration and threshold selection, and writes the same artifact set as the primary model. The fact-sheet target is reported alongside the primary target, and a smoke mode exercises the run on a small sample.

**Blocked by:** 02; primary-model-pipeline (external spec, not yet implemented).

**Status:** resolved

- [x] The PR config sets the PR feature table, the `age >= 15` sample, the 12-feature default set, the `target` column, cluster grouping, the survey weight column, and the PR subgroup columns.
- [x] The default feature set is exactly `female`, `age`, `urban`, `wealth`, `education`, `tobacco_any`, `alcohol_any`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, and `bp_medication`.
- [x] The default excludes `state`, `religion`, `social_group`, all anthropometry, diet, cooking fuel, marital status, children ever born, the smoking/smokeless split, and alcohol frequency.
- [x] The run reuses `run_training_pipeline`; there is no second modeling implementation.
- [x] Calibration and the screening threshold are fitted independently on PR out-of-fold predictions.
- [x] The BP-valid 15+ sample reconciles to 1,805,697 rows and 253,091 positives.
- [x] The run writes the split assignment, resolved config, run metadata, metrics, subgroup metrics, calibration curve, threshold analysis, feature importance, SHAP summary, results report, and model bundle.
- [x] Run metadata records the split source, config, seeds, package versions, dataset fingerprint, and row counts.
- [x] Fact-sheet target metrics are reported separately from the primary target.
- [x] A small-sample smoke mode runs without training on the full table.
- [x] The pipeline-reuse test uses a synthetic PR-shaped table, asserts the 12-feature default, asserts the required artifacts, and round-trips the model bundle.
- [x] No microdata or model binaries are committed.

## Comments

Ticket 03 depended on the external `primary-model-pipeline` spec, which was not implemented when this
ticket started. That spec was implemented first as the shared seam so the PR run could reuse it rather
than growing a second modeling path:

- `src/screener/pipeline.py` — `TrainingConfig`, `TrainingResult`, `run_training_pipeline`; owns
  filtering, split validation, out-of-fold prediction, model selection, calibration, threshold
  selection, metrics, subgroups, artifacts, report, model card, metadata, and bundle.
- `src/screener/modeling.py` — model factories and matrix preparation.
- `src/screener/calibration.py` — isotonic/Platt selection and sensitivity-first threshold selection,
  both fitted on out-of-fold predictions.
- `src/screener/metrics.py` — discrimination, calibration, sensitivity/specificity, and cluster
  bootstrap helpers.
- `src/screener/feature_policy.py` — the fixed primary 32-feature and PR 12-feature defaults plus
  forbidden-column validation.
- `src/screener/model_bundle.py` — schema-validated model bundle with save/load round-trip.

PR-specific work is configuration only, in `src/screener/pr_model.py`:

- `pr_15_plus_config` sets `data/processed/pr_features.parquet`, `age >= 15`, BP-valid filters,
  `target`, `cluster` grouping, `household`, `weight_scaled`, and the PR subgroup set (sex, age band,
  wealth, residence, education, state, religion, social group).
- The default feature list is exactly the 12 columns in the ticket; `state`, `religion`,
  `social_group`, anthropometry, diet, cooking fuel, marital status, children ever born,
  smoking/smokeless split, and alcohol frequency are excluded by policy and rejected if requested.
- `target_fact_sheet` is carried as the secondary target and reported separately in `metrics.json`
  and the results report.

`scripts/run_pr_model.py` wraps the shared pipeline for `--sample 15plus|15-49`, with `--smoke`,
`--smoke-rows`, `--bootstrap`, `--shap-sample`, `--no-split-assignment`, and `--feature-variant`.
Smoke mode performs a stratified sample *after* filtering, so it never trains on the full table; the
metadata records `row_counts.rows_after_filters` separately from `row_counts.rows_used`.

Real-data reconciliation (`pr_sample_summary` / `reconcile_pr_bp_sample`) passes all checks:
1,805,697 BP-valid 15+ rows and 253,091 positives; 15–49 = 1,290,219 / 112,898; 50+ = 515,478 /
140,193.

Real-data smoke runs completed for both samples with split source `pr_only`, reconciliation `pass`,
and every required artifact written. The 15+ run used 3,000 of 1,805,697 filtered rows (613 test,
94 test positives), selected isotonic calibration, and wrote `split_assignment.parquet`,
`resolved_config.json`, `run_metadata.json`, `metrics.json`, `subgroup_metrics.csv`,
`calibration_curve.csv`, `threshold_analysis.csv`, `feature_importance.csv`, `shap_summary.csv`,
`results_report.md`, `model_card.md`, and `model_bundle.joblib`. The selected model was logistic, so
SHAP uses the documented permutation-importance fallback; metadata records
`shap.method = "permutation_importance_fallback"`.

Tests: `tests/test_pipeline.py` builds a synthetic PR-shaped table, runs `pr_15_plus_config` through
`run_training_pipeline`, asserts the 12-feature default, asserts every required artifact exists, and
round-trips the bundle. It also loads the split assignment from `split_assignment_path` and asserts
the pipeline uses it rather than generating a new partition. `tests/test_pr_model.py` covers configs,
sample summary, reconciliation, subgroups, and the 15–49 filter. `tests/test_metrics.py` covers
weighted prevalence. Full suite: 92 passing. Ruff is clean on all new/changed files.

A review pass fixed four gaps: survey-weighted metrics now report a genuinely weighted prevalence;
cluster-bootstrap intervals keep cluster labels aligned when non-finite scores are dropped; the
`split_assignment_path` config field is now loaded instead of silently ignored; and the results
report/model card now include sensitivity at 80% specificity, 95% intervals, weighted metrics,
PR-sensitivity framing, ethical considerations, and the citation.

No microdata or model binaries are committed: real runs and smoke outputs are written outside the
repo (or under ignored `artifacts/`), and `.gitignore` ignores nested `*.joblib` under `artifacts/**/`.
The full real 15+ training run is deliberately left to ticket 07.
