# 07: Real-data PR runs and publishable aggregate outputs

**What to build:** Execute the PR 15+ and PR 15–49 runs on the real NFHS-5 PR table, verify the definition-of-done counts, and publish the aggregate results and comparison report. Microdata and model binaries stay local and gitignored; only aggregate metrics, plots, and reports are publishable.

**Blocked by:** 04, 05, 06.

**Status:** resolved

- [x] The PR feature table reconciles to 1,808,013 rows and 253,421 positives.
- [x] The BP-valid 15+ sample reconciles to 1,805,697 rows and 253,091 positives.
- [x] The BP-valid 15–49 sample reconciles to 1,290,219 rows and 112,898 positives.
- [x] The BP-valid 50+ sample reconciles to 515,478 rows and 140,193 positives.
- [x] The PR 15+ and PR 15–49 runs complete with the required artifacts.
- [x] The comparison report contains the sample, feature, overall, calibration, threshold, sex, age-band, and subgroup tables.
- [x] Aggregate metrics, plots, and reports are published; microdata and model binaries are not committed.
- [x] The project and data documentation record the PR build and run commands and the sample reconciliation.
- [x] A real-data smoke run completes before the full run.

## Comments

The full real-data runs were executed on the local NFHS-5 tables. The primary model run was
implemented first as a thin wrapper around the existing `run_training_pipeline`
(`scripts/run_primary_model.py`, `src/screener/primary_model.py`), because the published comparison
needs a real primary reference run and no primary CLI existed yet. The primary run generated the
cluster partition; the PR partition was then rebuilt with
`--primary-assignment` so the three runs share one cluster partition where the recodes overlap. Both
PR runs record `split_source: mixed`.

Published runs:

- Primary IR/MR 15–49: `artifacts/primary/primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e`
  — 774,861 rows / 64,526 positives; LightGBM with isotonic calibration; ROC-AUC 0.7099,
  PR-AUC 0.2035, Brier 0.0721, threshold 0.0549, sensitivity 0.8096, specificity 0.4558.
- PR 15–49: `artifacts/pr/runs/pr_15to49_bp_20261026T084051779902Z_42_96adb799`
  — 1,290,219 rows / 112,898 positives; LightGBM with isotonic calibration; ROC-AUC 0.6979,
  PR-AUC 0.1900, Brier 0.0761, threshold 0.0599, sensitivity 0.7958, specificity 0.4708.
- PR 15+: `artifacts/pr/runs/pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6`
  — 1,805,697 rows / 253,091 positives; LightGBM with isotonic calibration; ROC-AUC 0.7459,
  PR-AUC 0.3320, Brier 0.1077, threshold 0.1040, sensitivity 0.7925, specificity 0.5614.
- Comparison: `artifacts/pr/comparison/primary_vs_pr_comparison.{md,json}`.

Every run wrote `split_assignment.parquet`, `resolved_config.json`, `run_metadata.json`,
`metrics.json`, `subgroup_metrics.csv`, `calibration_curve.csv`, `threshold_analysis.csv`,
`feature_importance.csv`, `shap_summary.csv`, `shap_summary.png`, `results_report.md`,
`model_card.md`, and `model_bundle.joblib`. The comparison report contains the sample
(including age composition), feature-set, overall and survey-weighted metrics, reliability curves,
threshold behaviour, sex-specific, age-band-specific, and PR subgroup-gap tables. A review pass
fixed the comparison report's calibrator and selected-model rows, which previously rendered `n/a`
because the report read them from the metric block instead of the run-level fields.

Sample reconciliation is now a repeatable command: `python scripts/verify_pr_samples.py`. It checks
1,808,013 / 253,421 (feature table), 1,805,697 / 253,091 (BP-valid 15+), 1,290,219 / 112,898
(BP-valid 15–49), 515,478 / 140,193 (BP-valid 50+), and 774,861 / 64,526 (primary 15–49). All
checks pass. `data/pr_split_summary.json` records `split_source: mixed` and the leakage checks.

Aggregate artifacts are committed under `artifacts/`. Split assignments, per-respondent SHAP values,
model bundles, and all raw/processed microdata are local-only and covered by `.gitignore`
(`artifacts/**/split_assignment.parquet`, `artifacts/**/shap_values.csv`, `artifacts/**/*.joblib`).
README and `data/README.md` now record the primary/PR build and run commands, the split-reuse order,
the sample reconciliation, and the published run paths.

Tests: full suite 126 passing after the comparison-report fix; ruff is clean on the new/changed
files.
