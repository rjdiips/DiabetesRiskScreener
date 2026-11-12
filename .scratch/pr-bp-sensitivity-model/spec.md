# PR 15+ BP-Based Sensitivity Model

Status: ready-for-agent
Depends on: primary-model-pipeline

## Problem Statement

The primary model covers adults aged 15–49 only, because the IR/MR recodes are restricted to women 15–49 and men 15–54. That leaves out the adults with the highest diabetes prevalence: in the PR household-member file, adults aged 50+ have 27.20% prevalence, compared with 8.75% among adults 15–49. The primary model therefore cannot say whether its risk pattern holds for older adults, and it is approximately 88% female.

The PR file covers 1,808,013 adults aged 15+ with valid glucose and a known target, and it has blood pressure for 99.87% of them. It does not have the rich IR/MR questionnaire, diet, or anthropometry for all ages. The researcher needs a large-sample, BP-based sensitivity model that uses the PR file, reuses the primary training pipeline, and reports honestly what changes when the sample is extended to older adults and the feature set is reduced.

## Solution

Build a PR feature table in the same harmonised schema as the IR/MR table, then reuse the primary training pipeline with a PR BP-based config. The default PR model uses age, sex, urban/rural residence, wealth, education, tobacco, alcohol, and blood pressure. It excludes anthropometry, diet, state, religion, and social group from deployed features so that missingness is not age- or sex-dependent and the model follows the same portability and fairness policy as the primary model.

Run three evaluations:

1. **PR 15+ BP model** — the large-sample sensitivity model, covering 1,805,697 BP-valid adults.
2. **PR 15–49 BP model** — an age-matched comparison that isolates the effect of the smaller feature set.
3. **Primary IR/MR 15–49 model** — the reference model from the primary pipeline.

Write a side-by-side comparison report, a PR feature audit, and a PR sensitivity model-card addendum. The PR model is a sensitivity analysis, not the deployed model.

## User Stories

1. As a researcher, I want a PR feature table covering all adults aged 15+, so that I can model the population the IR/MR files leave out.
2. As a researcher, I want the PR table to use the same harmonised column names as the IR/MR table, so that the same training and reporting code works.
3. As a researcher, I want the PR sample to reconcile with the existing audit at 1,808,013 rows and 253,421 positives, so that counts are traceable.
4. As a researcher, I want adults aged 50+ retained, so that the sensitivity model covers the age group with the highest prevalence.
5. As a researcher, I want the PR model to use blood pressure and basic demographics, so that it works for every adult regardless of anthropometry coverage.
6. As a researcher, I want BP readings cleaned and averaged with the existing rules, so that the BP features are consistent with the primary model.
7. As a researcher, I want the PR model to exclude anthropometry from its default feature set, so that missingness is not determined by age or sex.
8. As a researcher, I want a PR 15–49 age-matched run, so that I can separate the effect of the smaller feature set from the effect of adding older adults.
9. As a researcher, I want a 50+ subgroup report, so that I can see how the BP-based model performs where prevalence is highest.
10. As a researcher, I want sex-specific metrics, so that the more balanced PR sample is used properly.
11. As a researcher, I want unweighted and survey-weighted metrics, so that sample performance and population performance are distinguished.
12. As a researcher, I want the same cluster-grouped split discipline as the primary model, so that the PR evaluation is not optimistically inflated.
13. As a researcher, I want the PR run to reuse the primary cluster split where clusters overlap, so that comparisons are not confounded by different partitions.
14. As a researcher, I want a state-free default PR model, so that it matches the primary model's portability policy.
15. As a researcher, I want state, religion, and social-group variants, so that I can measure what those features would change.
16. As a researcher, I want the target and fact-sheet target reported separately, so that the PR results align with NFHS reporting.
17. As a reviewer, I want a side-by-side comparison of the primary, PR 15–49, and PR 15+ models, so that I can see the trade-off between feature richness and population coverage.
18. As a reviewer, I want the comparison to state clearly that the samples and feature sets differ, so that I do not overinterpret the overall metrics.
19. As a reviewer, I want calibration and threshold metrics for every model, so that the operational comparison is fair.
20. As a reviewer, I want the age gradient and 50+ prevalence shown, so that the higher PR prevalence is explained rather than treated as a surprise.
21. As a reviewer, I want a PR sensitivity report and model-card addendum, so that the analysis is documented and cannot be mistaken for the deployed model.
22. As a fairness auditor, I want PR subgroup metrics by sex, age band, wealth, residence, education, state, religion, and social group, so that gaps are visible.
23. As a fairness auditor, I want small-group flags, so that unreliable subgroup estimates are not used to claim fairness.
24. As a fairness auditor, I want a PR fairness variant that includes religion and social group, so that the primary model's exclusion decision is tested rather than assumed.
25. As a fairness auditor, I want a PR state variant, so that local calibration is compared with the portable state-free model.
26. As a fairness auditor, I want the report to note that PR religion and social group describe the household head, not necessarily the respondent, so that the subgroup interpretation is correct.
27. As a maintainer, I want the PR feature builder to fail loudly on missing columns or invalid codes, so that silent data errors are caught.
28. As a maintainer, I want the PR feature audit to record BP coverage, target prevalence, and missingness, so that data quality is visible.
29. As a maintainer, I want the PR model to reuse the primary pipeline, so that there is one modeling implementation rather than two.
30. As a maintainer, I want the PR config saved with the run, so that the sample filter and feature set are reproducible.
31. As a maintainer, I want synthetic-data tests for the PR feature builder and comparison helper, so that CI does not need the real microdata.
32. As a portfolio author, I want a publishable aggregate PR sensitivity report, so that the larger-sample analysis is shareable without redistributing microdata.
33. As a portfolio author, I want the PR model's limitations stated, so that it is not mistaken for a clinically validated or deployed tool.
34. As a health worker, I want the PR model's sensitivity, specificity, PPV, and NPV at its threshold, so that I can judge whether a simpler BP-based screener is viable.
35. As a health worker, I want the PR model's performance in older adults, so that I know whether it is useful where prevalence is highest.
36. As a data scientist, I want the PR model's calibration compared with the primary model, so that I can see whether the simpler feature set degrades probability quality.
37. As a data scientist, I want the PR model's feature importance and SHAP summary, so that I can explain what drives the BP-based model.
38. As a data scientist, I want the PR model to use the same calibration and threshold-selection logic as the primary model, so that the comparison is methodologically consistent.
39. As a future maintainer, I want the PR table to retain anthropometry where the PR file measures it, so that a future PR anthropometry variant does not require rebuilding the table.
40. As a future maintainer, I want the PR-specific optional cooking-fuel variable retained, so that a household-SES sensitivity variant can be added later.

## Implementation Decisions

### Scope and sample

- The PR feature table covers adults aged 15–98 (98 means 98+), with valid random glucose, a known diagnosis/medication status, and sex recorded as male or female.
- The expected table size is 1,808,013 rows, 253,421 positives, and 14.02% prevalence. It contains 962,219 women and 845,794 men.
- The table includes 1,291,831 adults aged 15–49 and 516,182 adults aged 50+. Prevalence is 8.75% and 27.20% respectively.
- The default PR 15+ model uses the BP-valid subset: 1,805,697 rows, 253,091 positives, and 14.02% prevalence. The 2,316 rows without a valid BP pair are excluded from the BP model but retained in the feature table and described in the audit.
- The PR 15–49 comparison uses the BP-valid 15–49 subset: 1,290,219 rows, 112,898 positives, and 8.75% prevalence.
- The target is the same as the primary model: glucose > 140 mg/dL OR diagnosed diabetes OR glucose-lowering medication. The fact-sheet target (glucose > 140 OR medication) is reported separately.
- The 183 respondents coded as transgender in the PR file are excluded because the audit sample and the primary model use sex codes 1 and 2. The model card states this exclusion.
- The PR model is a sensitivity analysis. The primary IR/MR model remains the deployed model.

### PR feature table

- The new seam is `build_pr_features(raw_pr)`. It accepts a raw PR DataFrame and returns a harmonised feature table. A thin command-line wrapper writes the table and the PR feature audit.
- The table mirrors the IR/MR harmonised schema where a concept exists. Concepts that do not exist in the PR file are explicit missing values, not silently omitted, so that the same training and reporting code can consume the table.
- The PR table adds two PR-specific optional columns, `cooking_fuel` and the derived `cooking_fuel_clean`. They are not in the default BP model.
- The builder reuses the existing cleaning and target helpers: scaled measurement cleaning, BMI from height and weight, waist-to-hip ratio, BP summarisation, binary coding, target construction, and fact-sheet target construction.
- The builder validates required columns, value codes, and the expected sample filters. It raises a clear error rather than silently producing an incomplete table.

### PR source mapping

| Harmonised column | PR source | Decision |
|---|---|---|
| `case_id` | `hhid` + `hvidx` | Stable person identifier |
| `recode` | constant | `"PR"` |
| `sex` | `hv104` | 1 male, 2 female; exclude 3 |
| `female` | `hv104 == 2` | Binary |
| `cluster`, `household`, `line` | `hv001`, `hv002`, `hvidx` | Grouping and merge keys |
| `psu`, `strata` | `hv021`, `hv022` | Survey design |
| `weight`, `weight_scaled` | `hv005`, `hv005 / 1,000,000` | Survey weight |
| `age` | `hv105` | 15–98; 98 means 98+ |
| `age_15_49` | `hv105` between 15 and 49 | Sample flag |
| `state` | `hv024` | Raw state code |
| `residence`, `urban` | `hv025`, `hv025 == 1` | 1 urban, 2 rural |
| `wealth` | `hv270` | 1–5 |
| `education` | `hv106` | 0–3; code 8 becomes missing |
| `religion` | `sh47` | Household-head religion; raw codes retained |
| `social_group` | `sh49` | 1–4; code 8 becomes missing; household-head social group |
| `marital_status`, `children_ever_born` | not available | Missing |
| `height_cm`, `weight_kg`, `bmi` | `ha3`/`ha2`/`ha40` for women; `hb3`/`hb2`/`hb40` for men | Valid ranges; not used by default PR model |
| `waist_cm`, `hip_cm`, `waist_hip_ratio` | `sh305`, `sh306` | Valid ranges; not used by default PR model |
| `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high` | `shb18s/d`, `shb25s/d`, `shb29s/d` | Existing BP summarisation |
| `bp_medication` | `shb21` | 0/1 |
| `ever_checked`, `diagnosed`, `medication` | `shb55`, `shb56`, `shb57` | 0/1 |
| `glucose_raw`, `glucose` | `shb74` | Valid range 20–600 mg/dL |
| `target`, `target_fact_sheet` | constructed | Same definitions as primary |
| `tobacco_any` | `sh25` | 0/1; code 8 becomes missing |
| `smoking_any`, `smokeless_any` | not available | Missing; PR has combined tobacco only |
| `alcohol_any` | `sh26` | 0/1; code 8 becomes missing |
| `alcohol_frequency` | not available | Missing; PR has any use only |
| `diet_*` | not available | Missing |
| `cooking_fuel` | `hv226` | PR-specific raw code, optional |
| `cooking_fuel_clean` | `hv226` in 1, 2, 3 | Electricity, LPG, or biogas; codes 11–12 become missing |

### Default PR feature set

The default PR BP model uses 12 features:

- `female`
- `age`
- `urban`
- `wealth`
- `education`
- `tobacco_any`
- `alcohol_any`
- `systolic_mean`
- `diastolic_mean`
- `bp_readings`
- `bp_high`
- `bp_medication`

The default excludes:

- `state`, `religion`, and `social_group`, to match the primary model's portability and fairness policy;
- `height_cm`, `weight_kg`, `bmi`, `waist_cm`, `hip_cm`, and `waist_hip_ratio`, because they are measured only for younger women and men and would make missingness age- and sex-dependent;
- diet, marital status, children ever born, smoking/smokeless split, and alcohol frequency, because the PR file does not contain them;
- `cooking_fuel`, which is reserved for an optional household-SES variant.

Feature-set variants:

- **State variant:** default plus `state`.
- **Fairness variant:** default plus `religion` and `social_group`.
- **Combined variant:** default plus `state`, `religion`, and `social_group`.
- **Cooking-fuel variant:** default plus `cooking_fuel_clean`.

### Split strategy

- The PR model reuses the primary pipeline's cluster-grouped split logic. The 20% holdout and 5-fold development CV remain unchanged.
- Where a PR cluster appears in the primary split assignment, the PR run reuses that cluster's split and fold. Clusters that do not appear in the primary assignment are assigned deterministically with the same seed and stratification rules.
- The split source is recorded in the run metadata: `reused_primary`, `pr_only`, or `mixed`.
- The pipeline asserts that no cluster and no household appears in more than one split or development fold.
- The PR 15–49 comparison uses the same split assignment as the PR 15+ model, filtered to the 15–49 rows. This keeps the age-matched comparison within the same cluster partition.
- The test set is evaluated once per run, after model selection, calibration, and threshold selection.

### Training and evaluation

- The PR model reuses `run_training_pipeline` with a PR config. There is no second modeling implementation.
- The PR config sets the PR feature table, the `age >= 15` sample, the 12-feature default set, the `target` column, cluster grouping, the survey weight column, and the PR subgroup columns.
- The PR 15–49 comparison sets the sample to `age_15_49 == True` and otherwise uses the same config.
- The pipeline uses the same baselines, model selection, class-imbalance handling, calibration, threshold selection, bootstrap, and artifact conventions as the primary model.
- The PR model's threshold is selected independently on out-of-fold PR predictions. It is not inherited from the primary model.
- The PR model's calibration is fitted independently on out-of-fold PR predictions.
- The PR model's probability is not directly comparable to the primary model's probability because the samples and feature sets differ. The report compares discrimination, calibration, and threshold behaviour, not raw probability levels.

### Subgroups and fairness

- Required PR subgroups: sex, age band (15–24, 25–34, 35–44, 45–49, 50–59, 60–69, 70+), wealth quintile, residence, education, and state where the state has at least 500 test respondents.
- Audit subgroups: religion and social group, using the fairness variant.
- Each subgroup row reports n, positives, prevalence, ROC-AUC, PR-AUC, sensitivity at the chosen threshold, calibration error, and the gap versus the overall test set.
- Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged as unreliable.
- The report notes that PR religion and social group describe the household head, not necessarily the respondent.
- The report does not claim fairness from excluding a protected attribute. It shows what the state and fairness variants change.
- Sex-specific and age-band-specific metrics are mandatory because the PR sample is more balanced than the primary sample and includes a large older population.

### Comparison report

- A pure comparison helper accepts the primary result and the PR 15+ and PR 15–49 results and produces a structured comparison.
- The comparison report contains:
  - a sample table with row counts, sex composition, age composition, prevalence, and weighted prevalence;
  - a feature-set table showing the 32-feature primary model and the 12-feature PR model;
  - overall discrimination, calibration, and threshold metrics for all three runs;
  - reliability curves;
  - sex-specific and age-band-specific metrics;
  - subgroup gaps for the PR model;
  - a narrative explaining that the PR 15–49 run isolates the feature-set trade-off while the PR 15+ run adds older adults;
  - a limitations section.
- The comparison report is written as readable Markdown and as structured JSON.
- The report must not claim that one model is universally better. It must distinguish feature richness, population coverage, and calibration.

### Artifacts and model card

- The PR feature builder writes the PR feature table and a PR feature audit. The table is local-only; the audit is aggregate and publishable.
- Each PR run writes the same artifact set as the primary pipeline: split assignment, resolved config, run metadata, metrics, subgroup metrics, calibration curve, threshold analysis, feature importance, SHAP summary, results report, and model bundle.
- The comparison helper writes a primary-versus-PR report and JSON.
- A PR sensitivity model-card addendum covers intended use, sample and feature differences, age gradient, calibration, threshold, subgroup gaps, label bias, limitations, and the fact that the PR model is not the deployed model.
- The model card states that PR religion and social group describe the household head, that 183 transgender respondents were excluded, and that the PR model does not use anthropometry or diet.
- Model binaries and the PR feature table remain local and gitignored. Aggregate metrics, plots, and the model-card addendum are publishable.

### Reproducibility and data hygiene

- The PR feature audit records the exact row counts, positive counts, prevalence, BP coverage, missingness, cluster count, household count, and weighted prevalence.
- The PR run records the split source, config, seeds, package versions, dataset fingerprint, and row counts.
- The PR feature builder and the PR model run support a small-sample smoke mode.
- Modeling uses Python 3.11 or 3.12 because LightGBM and SHAP wheels may lag on Python 3.14.
- Raw and processed microdata are never committed. Model binaries are never committed.
- The report states that the PR model is a sensitivity analysis, not a clinical diagnostic tool and not the deployed screener.

## Testing Decisions

A good test checks externally observable behaviour: the PR feature table has the expected sample and target counts, the BP features are correct, the harmonised schema is present, the default feature set excludes anthropometry and sensitive attributes, the PR run reuses the training pipeline, and the comparison helper produces a complete table. Tests should not assert private helper structure or exact tree splits.

The primary seam is the PR feature builder, `build_pr_features(raw_pr)`. The PR model reuses the existing `run_training_pipeline` seam, so there is no second modeling seam. The comparison helper is a pure function tested directly.

Prior art is the synthetic-DataFrame style in the existing feature and target tests. Tests use small hand-built PR-shaped DataFrames and known arrays; they do not require the real NFHS-5 microdata.

### PR feature-builder integration test

- Build a synthetic PR-shaped DataFrame with the required columns: household and cluster identifiers, survey design, sex, age, education, wealth, residence, state, religion, social group, cooking fuel, tobacco, alcohol, BP readings, BP medication, glucose, diagnosis, medication, and anthropometry for both sexes.
- Include rows that should be excluded: age below 15, sex code 3, invalid glucose, missing diagnosis and medication status, and invalid BP values.
- Include rows with valid BP pairs, invalid BP pairs, and missing BP.
- Include both women and men so the sex-specific anthropometry path is exercised.
- Assert the sample filter retains only adults 15–98 with sex 1/2, valid glucose, and a known target.
- Assert the target and fact-sheet target values.
- Assert BP means, reading count, high-BP flag, and BP medication.
- Assert height, weight, and BMI are taken from the women's variables for women and the men's variables for men, and are missing outside the measured age ranges.
- Assert education code 8 and social-group code 8 become missing.
- Assert tobacco and alcohol code 8 become missing.
- Assert `recode` is `"PR"`, `age_15_49` is correct, and `case_id` is unique.
- Assert the harmonised schema is present and unavailable concepts are explicit missing values.
- Assert `cooking_fuel` and `cooking_fuel_clean` are correct.
- Assert the builder is deterministic.

### Pure-helper tests

- BP summarisation: valid pairs averaged, invalid pairs ignored, reading count correct, high-BP flag correct, all-invalid rows missing.
- Anthropometry by sex: women use the women's variables, men use the men's variables, missing codes and out-of-range values become missing.
- Education and social-group mapping: code 8 becomes missing; codes 1–4 remain.
- Tobacco and alcohol mapping: 0/1 retained; 8 becomes missing.
- Cooking-fuel mapping: electricity, LPG, and biogas are clean; kerosene through animal dung are not; no-food and other become missing.
- Target construction: high glucose, diagnosis, and medication union; fact-sheet target uses high glucose or medication only.
- Comparison helper: complete table for all three runs, correct differences, handles a missing metric without crashing, and includes the sample/feature-difference notes.

### Pipeline-reuse test

- Extend the existing pipeline integration test with a PR config using a synthetic PR-shaped feature table.
- Assert the PR config uses the 12-feature default set.
- Assert the default PR feature set excludes `state`, `religion`, `social_group`, anthropometry, diet, and cooking fuel.
- Assert the state, fairness, combined, and cooking-fuel variants add the expected columns.
- Assert the run produces the required artifacts and that the model bundle round-trips.
- Assert the split source is recorded and no cluster or household crosses splits or folds.

### Test constraints

- The PR feature-builder integration test must run in under about 30 seconds.
- The pipeline-reuse test must stay within the existing pipeline test budget.
- Tests must not download data or require a GPU.
- Real-data training is verified by a manual smoke run, not by CI.

## Out of Scope

- The primary model pipeline implementation. This spec depends on it.
- A PR anthropometry model, a PR diet model, or a PR questionnaire-only model. The table retains the columns for future variants, but the default PR model does not use them.
- A deployed state-specific PR model, a deployed PR model, or a PR model in the API/demo.
- A 5-fold × 10-seed ensemble. The primary pipeline's single-seed 5-fold convention applies.
- Large-scale hyperparameter optimisation.
- External validation, NFHS-6 replication, clinical validation, and causal inference.
- Fairness mitigation such as reweighting or adversarial debiasing.
- District-level or small-area estimates.
- Model compression, ONNX export, and mobile deployment.
- Redistributing raw or processed NFHS-5 microdata.

## Further Notes

- The PR sample includes 516,182 adults aged 50+ with 27.20% prevalence. The higher PR prevalence is expected because of the age extension, not a data error.
- The PR sample is 53.2% women and 46.8% men, much more balanced than the IR/MR sample. Sex-specific metrics are still mandatory.
- BP is valid for 99.87% of the target-valid PR sample. The 2,316 rows without a valid BP pair are excluded from the BP model but retained in the table and described in the audit.
- PR tobacco and alcohol are combined 0/1 variables. The PR file does not support the smoking/smokeless split or alcohol frequency used in the IR/MR table.
- PR anthropometry is measured only for women 15–49 and men 15–54. The default PR model excludes it so that missingness is not age- and sex-dependent.
- PR religion and social group describe the household head. The fairness report must say this rather than implying they are respondent-level attributes.
- The PR model's probabilities are not directly comparable to the primary model's probabilities. The comparison is about discrimination, calibration, and threshold behaviour on different samples and feature sets.
- The PR 15–49 run is the cleanest feature-set comparison. The PR 15+ run answers the separate question of whether the model extends to older adults.
- The PR model is a sensitivity analysis. The primary IR/MR model remains the model intended for the app.
- The original project specification's multi-seed ensemble and four-model-variant plan remains larger than this spec. This spec delivers the large-sample BP-based sensitivity analysis and the age-matched comparison first.
- Definition of done:
  - the PR feature table reconciles to 1,808,013 rows and 253,421 positives;
  - the BP-valid 15+ sample reconciles to 1,805,697 rows and 253,091 positives;
  - the BP-valid 15–49 sample reconciles to 1,290,219 rows and 112,898 positives;
  - the BP-valid 50+ sample reconciles to 515,478 rows and 140,193 positives;
  - the PR 15+ and PR 15–49 runs complete with the required artifacts;
  - the comparison report contains the sample, feature, overall, calibration, threshold, sex, age-band, and subgroup tables;
  - the PR sensitivity model-card addendum exists;
  - all tests pass;
  - no microdata or model binaries are committed.
