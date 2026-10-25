# 06: Primary vs PR comparison report

**What to build:** Produce a side-by-side comparison of the primary IR/MR 15–49 model, the PR 15–49 age-matched model, and the PR 15+ model. The report makes the sample and feature-set differences explicit and compares discrimination, calibration, and threshold behaviour rather than raw probability levels. It is written as readable Markdown and as structured JSON.

**Blocked by:** 03, 04, 05; primary-model-pipeline (external spec, not yet implemented).

**Status:** ready-for-agent

- [x] A pure comparison helper accepts the primary result and the PR 15+ and PR 15–49 results and returns a structured comparison.
- [x] The report includes a sample table with row counts, sex composition, age composition, prevalence, and weighted prevalence.
- [x] The report includes a feature-set table showing the 32-feature primary model and the 12-feature PR model.
- [x] The report includes overall discrimination, calibration, and threshold metrics for all three runs.
- [x] The report includes reliability curves.
- [x] The report includes sex-specific and age-band-specific metrics.
- [x] The report includes subgroup gaps for the PR model.
- [x] The report includes a narrative explaining that the PR 15–49 run isolates the feature-set trade-off while the PR 15+ run adds older adults.
- [x] The report includes a limitations section and states that raw probabilities are not directly comparable across the three models.
- [x] The report is written as Markdown and as structured JSON.
- [x] The report does not claim that one model is universally better.
- [x] Tests cover the complete table for all three runs, correct differences, graceful handling of a missing metric, and the sample/feature-difference notes.

## Comments

The comparison lives in `src/screener/pr_comparison.py`:

- `compare_primary_vs_pr(primary, pr_15_49, pr_15_plus)` is pure and accepts the
  three finished `TrainingResult` objects. It returns run snapshots, sample and
  feature tables, overall and survey-weighted metric blocks, descriptive deltas
  in all three directions, reliability curves, sex and age-band rows, PR subgroup
  gaps, sample and feature-difference notes, the narrative, and limitations.
- Missing metrics do not raise: absent weighted values, subgroup blocks, or
  calibration curves become `null`/empty and render as `n/a`. This keeps a
  partial run comparable without inventing a value.
- `render_primary_vs_pr_report(comparison)` writes the Markdown report and
  `write_primary_vs_pr_report(...)` writes both the Markdown and JSON. Sample
  composition and prevalence are reported on the test set, because that is the
  set every run's metrics use; the filtered row count is shown separately.
- `load_run_result(run_dir)` reconstructs a result from a finished run's
  `resolved_config.json`, `run_metadata.json`, `metrics.json`,
  `subgroup_metrics.csv`, `calibration_curve.csv`, and optional
  `threshold_analysis.csv`.
- `scripts/run_pr_comparison.py --primary ... --pr-15-49 ... --pr-15-plus ...`
  loads the three run directories and writes
  `artifacts/pr/comparison/primary_vs_pr_comparison.{json,md}`.

The report makes the sample differences explicit: the primary run uses the IR/MR
recodes, the PR runs use the PR household-member recode, the PR sample extends to
adults aged 50+, and the primary and PR feature sets differ. It states that the
PR 15–49 run isolates the feature-set trade-off and the PR 15+ run adds older
adults, that no run is universally better, and that raw probabilities are not
directly comparable. Limitations also note that the PR 15–49 comparison is not a
pure feature ablation because the underlying instruments differ.

Tests: `tests/test_pr_comparison.py` covers the complete three-run table, feature
counts (32/12/12), deltas in all three directions (unweighted and weighted),
missing metrics, missing subgroup and reliability blocks, the
sample/feature-difference narrative, PR-only feature union, JSON/Markdown
writing, and artifact loading. `tests/test_run_pr_comparison_cli.py` covers the
CLI output and a missing run directory.

A review pass made the sample note data-driven, made a missing age or sex block
render an explicit `n/a` row instead of disappearing, derived PR-only features
from both PR runs, removed the unused `mandatory_subgroups`/threshold-analysis
fields, and moved the JSON/Markdown scalar formatters to
`src/screener/reporting.py` so `pr_variants.py` and `pr_comparison.py` share
them. Full suite: 119 passing. Ruff is clean on all new and changed files.

Real-data comparison runs are left to ticket 07; the helper and CLI are exercised
against synthetic results in tests.
