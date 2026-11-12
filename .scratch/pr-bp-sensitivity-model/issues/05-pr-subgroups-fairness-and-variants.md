# 05: PR subgroup, fairness, and feature-set variants

**What to build:** Report PR subgroup performance and run the feature-set variants that test the primary model's exclusion decisions. The default model is audited by sex, age band including 50+, wealth, residence, education, and state; religion and social group are audited through the fairness variant. State, fairness, combined, and cooking-fuel variants are compared with the default so the report shows what those features change rather than assuming they are irrelevant.

**Blocked by:** 03.

**Status:** resolved

- [x] Subgroup rows exist for sex, age band (15–24, 25–34, 35–44, 45–49, 50–59, 60–69, 70+), wealth quintile, residence, education, and every state with at least 500 test respondents.
- [x] Each subgroup row reports n, positives, prevalence, ROC-AUC, PR-AUC, sensitivity at the chosen threshold, calibration error, and the gap versus the overall test set.
- [x] Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged as unreliable.
- [x] Religion and social group are audited through the fairness variant.
- [x] The report states that PR religion and social group describe the household head, not necessarily the respondent.
- [x] The state variant adds `state`; the fairness variant adds `religion` and `social_group`; the combined variant adds all three; the cooking-fuel variant adds `cooking_fuel_clean`.
- [x] Variant metrics are compared with the default model.
- [x] The report does not claim fairness from excluding a protected attribute.
- [x] Sex-specific and age-band-specific metrics are mandatory.
- [x] Tests assert each variant's feature set and the small-group flagging behaviour.

## Comments

The subgroup and fairness work stays inside the shared pipeline seam. `_subgroup_metrics` now emits
n, positives, prevalence, ROC-AUC, PR-AUC, sensitivity, specificity, ECE, prevalence/ROC-AUC/PR-AUC/
sensitivity/ECE gaps versus the overall test set, a `reliable` flag, an `unreliable_reason`, and the
audit-only note. The 200-row/50-positive rule is driven by `small_group_min_rows` and
`small_group_min_positives`; the 500-row state rule is now the `state_min_test_rows` config field
instead of a second hard-coded constant.

The default PR run reports only sex, age band, wealth, residence, education, and state. Religion and
social group are added to the subgroup set only for the `fairness` and `combined` variants, so the
exclusion decision is tested rather than assumed. `pr_15_plus_config(feature_variant=...)` and the
15-49 config now record the variant, suffix non-default run names, and use the existing
`feature_policy.pr_feature_set` variants. The results report renders mandatory sex and age-band
tables, a richer subgroup table, the household-head note for audit subgroups, and the explicit
statement that excluding a protected attribute does not by itself make a model fair. The model card
carries the same note.

Variant comparison is a pure helper, `src/screener/pr_variants.py`:

- `compare_pr_variants(default, variants)` requires mandatory sex and age-band subgroup rows, reports
  each variant's added/removed features, overall metric deltas versus the default, and the sex,
  age-band, and audit-only subgroup rows. It does not treat a higher variant ROC-AUC as proof that a
  variant is better.
- `render_pr_variant_comparison(comparison)` writes the readable Markdown, including the
  household-head note, the small-group rule, and the no-fairness-from-exclusion statement.
- `scripts/run_pr_variants.py` runs the default plus the requested variants on the same saved PR
  split assignment and writes `variant_comparison.json` and `variant_comparison.md` under
  `artifacts/pr/variants/`. `--smoke` keeps the suite runnable on a small sample; every variant run
  requires the saved split assignment, so the suite cannot silently use a fresh partition.

Tests: `tests/test_pr_model.py` covers the default and variant subgroup contracts.
`tests/test_pipeline.py` covers the full subgroup columns and gaps, the 200/50 flagging with a
reliable sex group and an unreliable oldest age band, the 500-row state omission, the fairness
variant's household-head audit rows, and a real-pipeline variant comparison. `tests/test_pr_variants.py`
covers deltas, feature differences, mandatory-subgroup enforcement, and the report statements.
`tests/test_run_pr_variants_cli.py` covers the parent-partition guard and JSON/Markdown artifact
writing. `tests/test_feature_policy.py` asserts each variant's exact feature set. Full suite:
110 passing. Ruff is clean on all new and changed files.

Real-data variant runs are left to ticket 07; the smoke suite is exercised in tests and by
`python scripts/run_pr_variants.py --sample 15plus --smoke`.

