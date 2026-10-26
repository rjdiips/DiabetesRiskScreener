# 08: PR sensitivity model-card addendum

**What to build:** Write a model-card addendum for the PR BP sensitivity analysis that can be published without microdata. It makes clear that the PR model is a sensitivity analysis, not the deployed screener, and explains what changes when the sample extends to older adults and the feature set is reduced.

**Blocked by:** 07.

**Status:** resolved

- [x] The addendum covers intended use, sample and feature differences, the age gradient, calibration, threshold, subgroup gaps, label bias, limitations, and non-goals.
- [x] It states that the PR model is not the deployed model and not a clinical diagnostic tool.
- [x] It states that PR religion and social group describe the household head.
- [x] It states that 183 transgender respondents were excluded.
- [x] It states that the PR model does not use anthropometry or diet.
- [x] It quotes real metrics from the PR runs, not smoke-run placeholders.
- [x] It is publishable without redistributing microdata.
- [x] It is linked from the comparison report and the project README.

## Comments

The addendum is generated, not hand-written: `render_pr_model_card_addendum` in
`src/screener/pr_comparison.py` reads the same comparison structure as the comparison report, and
`write_primary_vs_pr_report` writes `pr_sensitivity_model_card_addendum.md` alongside the report. The
comparison report links to it, and the README links it in both the comparison-command section and the
published-aggregates list. A custom comparison stem keeps the back-link correct, because the writer
passes the actual report filename into the renderer.

The committed addendum was produced from the published runs listed in ticket 07:

```
python scripts/run_pr_comparison.py --primary artifacts/primary/primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e --pr-15-49 artifacts/pr/runs/pr_15to49_bp_20261026T084051779902Z_42_96adb799 --pr-15-plus artifacts/pr/runs/pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6
```

Its tables and prose use those runs' real values: primary 15–49 ROC-AUC 0.7099 / PR-AUC 0.2035, PR
15–49 0.6979 / 0.1900, and PR 15+ 0.7459 / 0.3320. The sample table is read from the comparison's
sample summary, so row counts, test sizes, positives, and prevalences cannot drift from the runs. The
only hard-coded sample fact is the required statement that 183 transgender respondents were excluded.

The addendum covers intended use, sample and feature differences, the age gradient, calibration,
threshold, subgroup gaps, label bias, limitations, non-goals, publication, and citation. It states that
the PR model is a sensitivity analysis, not the deployed model or screener and not a clinical diagnostic
tool; that PR religion and social group describe the household head; that 183 transgender respondents
were excluded; and that the PR model does not use anthropometry or diet. It claims no fairness-variant
result. The publication section states that only aggregate counts, metrics, and run identifiers are
included, with no respondent records, split assignments, per-respondent SHAP values, or model binaries.

Tests: `tests/test_pr_comparison.py` checks every required section statement, run-derived sample rows
and metrics, the absence of microdata paths, the forward and custom-stem back-links, and the
household-head/183-transgender/feature claims; `tests/test_run_pr_comparison_cli.py` checks that the
CLI writes and reports the addendum. Full suite 130 passing; ruff clean on the changed files.
