# Primary Model Training and Evaluation

Status: ready-for-agent

## Problem Statement

The project now has a clean, harmonised IR/MR feature table, but no way to establish whether a model built on it is trustworthy. A naive random split would put members of the same household or cluster on both sides of the split, inflating performance. A single accuracy number would hide the class imbalance and the calibration problems that matter for a screening tool. And without subgroup reporting, the model could look good overall while performing poorly for men, older adults, or poorer households.

The researcher needs a reproducible pipeline that turns the feature table into a calibrated, evaluated, and documented primary model. The pipeline must prevent household and cluster leakage, compare a complex model against simple baselines, select a sensitivity-first referral threshold, and report discrimination, calibration, and subgroup performance honestly. The output must be aggregate results and a model bundle that can later be served by the API and demo without redistributing microdata.

## Solution

Build one reproducible primary-model pipeline that:

- takes the existing IR/MR feature table and restricts it to the primary 15–49 sample;
- creates a cluster-grouped holdout test set and cluster-grouped cross-validation folds, with no cluster or household overlap;
- trains a prevalence-only baseline, a logistic-regression baseline, and a LightGBM model;
- selects the model on validation performance, not on the test set;
- calibrates the selected model on out-of-fold predictions and chooses a screening threshold that targets at least 80% sensitivity;
- evaluates discrimination, calibration, clinical utility, and subgroup/fairness performance, with bootstrap confidence intervals and weighted sensitivity analyses;
- writes a model bundle, split assignment, metrics, calibration and threshold tables, subgroup report, SHAP summary, results report, and model card;
- exposes a stable load-and-predict interface for the future API.

The primary model is state-free and excludes caste/tribe and religion from deployed features. State, caste/tribe, and religion remain available for state and fairness sensitivity variants so their impact can be measured rather than assumed.

## User Stories

1. As a researcher, I want a reproducible split assignment saved to disk, so that every reported metric can be traced back to the same respondents.
2. As a researcher, I want clusters (PSUs) kept intact across train, validation, and test, so that community-level leakage does not inflate performance.
3. As a researcher, I want a household-overlap check even when splitting by cluster, so that I can prove no family members cross folds.
4. As a researcher, I want the test set held out until final evaluation, so that model selection and threshold tuning cannot peek at it.
5. As a researcher, I want a prevalence-only baseline, so that I know the floor that any feature-based model must beat.
6. As a researcher, I want a logistic-regression baseline with proper imputation, encoding, and scaling, so that I can judge whether LightGBM earns its complexity.
7. As a researcher, I want LightGBM with native categorical handling and missing-value support, so that I do not discard information through unnecessary imputation.
8. As a researcher, I want class imbalance handled explicitly, so that the model is not rewarded for predicting the majority class.
9. As a researcher, I want model selection based on validation PR-AUC and calibration, so that the test set remains an honest final check.
10. As a researcher, I want hyperparameters tuned only inside the training folds, so that tuning does not leak into validation or test performance.
11. As a researcher, I want calibrated probabilities, so that a predicted 20% risk means roughly 20% risk rather than an arbitrary score.
12. As a researcher, I want isotonic and Platt calibration compared on out-of-fold predictions, so that the calibrator is chosen by evidence rather than convention.
13. As a researcher, I want reliability curves and calibration slope and intercept, so that I can see where the probabilities are miscalibrated.
14. As a researcher, I want bootstrap confidence intervals clustered by PSU, so that uncertainty reflects the survey design rather than pretending every respondent is independent.
15. As a researcher, I want unweighted and survey-weighted metrics reported side by side, so that I can distinguish sample performance from population performance.
16. As a researcher, I want ROC-AUC, PR-AUC, Brier score, and log loss, so that ranking and probability quality are both visible.
17. As a researcher, I want sensitivity at 80% specificity, so that I can compare models at a clinically meaningful operating point.
18. As a researcher, I want the chosen threshold to target at least 80% sensitivity, so that the screener is useful for ruling people in for further testing.
19. As a researcher, I want the threshold selected on out-of-fold predictions, so that the operating point is not tuned on the test set.
20. As a health worker, I want sensitivity, specificity, PPV, NPV, and the confusion matrix at the chosen threshold, so that I understand the practical consequences of referral.
21. As a health worker, I want a configurable target sensitivity, so that a clinic with different referral capacity can choose a stricter or looser threshold.
22. As a reviewer, I want a model card covering intended use, target definition, sample, features, performance, calibration, threshold, and limitations, so that I can judge whether the model is being overclaimed.
23. As a reviewer, I want a readable results report with tables and plots, so that I do not have to parse raw JSON to understand the model.
24. As a reviewer, I want false-negative and false-positive error analysis, so that I can see who the model misses and why.
25. As a reviewer, I want label-bias analysis by wealth, education, and access-related variables, so that I can see whether the model is learning who gets diagnosed rather than who has elevated glucose.
26. As a fairness auditor, I want subgroup metrics by sex, age band, wealth quintile, residence, education, and state, so that I can identify material gaps.
27. As a fairness auditor, I want subgroup sample sizes and positive counts shown alongside every metric, so that small-group estimates are not overinterpreted.
28. As a fairness auditor, I want a fairness sensitivity model that includes caste/tribe and religion, so that I can measure the impact of excluding them from the primary model.
29. As a fairness auditor, I want a state sensitivity model, so that I can compare portable state-free performance with locally adapted performance.
30. As a fairness auditor, I want sex-specific metrics reported prominently, so that the roughly 88% female pooled sample does not hide poor performance for men.
31. As a data scientist, I want global SHAP importance on a fixed sample, so that I can explain what drives the model without running SHAP on every row.
32. As a data scientist, I want local top-factor explanations for a single respondent, so that the future API can show plain-language reasons.
33. As a maintainer, I want the model bundle to include the feature schema, categorical columns, calibrator, threshold, and metadata, so that inference cannot silently diverge from training.
34. As a maintainer, I want the pipeline to fail loudly when target-derived columns, identifiers, or survey-design fields are passed as features, so that leakage is caught before results are published.
35. As a maintainer, I want the split assignment, config, package versions, and dataset fingerprint saved with every run, so that results can be reproduced.
36. As a maintainer, I want an end-to-end test on a small synthetic feature table, so that the pipeline can be verified without the real NFHS-5 microdata.
37. As a future API developer, I want a load-and-predict interface that accepts the same feature columns used in training, so that serving the model does not require reimplementing preprocessing.
38. As a future API developer, I want calibrated probabilities and a threshold flag returned together, so that the API and UI agree on the referral decision.
39. As a portfolio author, I want aggregate metrics, plots, and a model card that can be published without redistributing microdata, so that the work is shareable and auditable.
40. As a portfolio author, I want the model's limitations stated explicitly, so that the project is not mistaken for a clinical diagnostic tool.

## Implementation Decisions

### Scope and sample

- The primary model uses the existing IR/MR feature table, filtered to `age_15_49 == True`.
- The primary sample has 774,861 respondents: 688,308 women and 86,553 men. Prevalence is 8.33%.
- Men aged 50–54 are excluded from the primary model. Their count and prevalence are reported as a sensitivity note; they are not silently dropped.
- The target is the existing `target` column: glucose > 140 mg/dL OR diagnosed diabetes OR glucose-lowering medication. The fact-sheet target (glucose > 140 OR medication) is reported as a secondary comparison.
- Rows with missing target are not allowed. The input table is already filtered, and the pipeline validates this.
- The PR 15+ BP-based sensitivity model is a separate specification because it requires a PR feature builder. It is not part of this pipeline.

### Public interface

- The single public seam is `run_training_pipeline(features, config, output_dir)`. It accepts a feature DataFrame, a `TrainingConfig`, and an output directory, and returns a `TrainingResult`.
- `TrainingConfig` records the seed, grouping column, test fraction, fold count, feature set, threshold target sensitivity, calibration options, bootstrap count, SHAP sample size, and model parameters.
- `TrainingResult` exposes the split assignment, selected model, metrics, subgroup metrics, calibration data, threshold analysis, feature importance, SHAP summary, artifact locations, and the `ModelBundle`.
- A companion `load_model_bundle(path)` loads a saved `ModelBundle` and exposes calibrated prediction. This is the only additional public interface, and it exists so the future API does not reach into serialized internals.
- `ModelBundle.predict_proba(features)` returns calibrated probabilities and `ModelBundle.predict(features)` returns the threshold flag. Both validate the feature schema.
- The command-line entry point is a thin wrapper around `run_training_pipeline`. It is not a second implementation.

### Split strategy

- The primary grouping unit is the cluster (PSU). Keeping clusters intact prevents both household and community leakage.
- A 20% cluster-grouped holdout test set is created first. The remaining 80% is the development set.
- Development uses 5-fold cluster-grouped cross-validation. The test set is untouched until model selection, calibration, and threshold selection are complete.
- Splitting is stratified on target and sex where feasible. Fold prevalence and sex composition are reported so that any imbalance is visible.
- The split assignment records one row per respondent with the split name and fold number. It is saved as an artifact.
- The pipeline asserts that no cluster appears in more than one split or development fold, and that no household appears in more than one split or development fold.
- Household-grouped splitting is supported as a configurable sensitivity check. If the user prefers household grouping as the primary unit, the config can switch the grouping column without changing the rest of the pipeline.
- Cluster overlap with the test set is reported for the household-grouped sensitivity run.

### Feature policy

- The feature table's model-input list is the superset. The primary deployed model uses 32 features: all model inputs except `state`, `religion`, and `social_group`.
- `state` is excluded from the primary model to reduce local access-pattern and diagnosis-pattern leakage. A state sensitivity variant adds it back.
- `religion` and `social_group` are excluded from the primary model because they are sensitive attributes. A fairness sensitivity variant adds them back for audit only.
- The primary model therefore uses 32 features; the state variant uses 33; the fairness variant uses 34; the combined variant uses all 35.
- The following are never model inputs: case identifiers, cluster/household/line identifiers, PSU, strata, sample weights, target components, `age_15_49`, and the raw duplicate encodings `sex` and `residence`.
- `female` and `urban` are the binary encodings used by the model. `sex` and `residence` remain in the table for reporting.
- Nominal categoricals use LightGBM native categorical handling. Ordinal variables such as education, wealth, and diet frequency are treated as ordered integers.
- Missing values are passed through to LightGBM. The logistic baseline uses median imputation for numeric columns and a separate missing category for categorical columns.
- The pipeline validates the feature schema before training and raises a clear error if a forbidden column is present or a required column is missing.

### Baselines and model selection

- Three models are required:
  1. Prevalence-only baseline, which predicts the training prevalence for every respondent.
  2. Logistic regression with imputation, one-hot encoding, scaling, and balanced class weights.
  3. LightGBM with native categorical handling, missing-value support, class weighting, and early stopping.
- A random forest or XGBoost comparison is optional and is not required for acceptance.
- Model selection uses development-set cross-validation. PR-AUC is the primary ranking metric; Brier score and log loss break ties and guard against poor calibration. The test set is not used for selection.
- Hyperparameter tuning uses a small manual grid or a bounded search inside the training folds. The search space, best parameters, and trial count are logged. Tuning on the test set is forbidden.
- A single-seed 5-fold LightGBM run is the required implementation. Multi-seed ensembling is a future extension, not an acceptance requirement.
- The selected model is refit on the full development set after model selection. The test set is evaluated once.

### Class imbalance

- Accuracy is not a headline metric. PR-AUC, sensitivity at fixed specificity, and calibration are the primary measures.
- Logistic regression uses balanced class weights.
- LightGBM uses either `scale_pos_weight` or `is_unbalance`, chosen on validation performance. The chosen setting is recorded.
- Weighted training is a sensitivity analysis, not the primary training mode. The primary model is trained unweighted so that it produces individual-level probabilities.

### Calibration

- Out-of-fold predictions on the development set are used to fit calibrators.
- Isotonic regression and Platt scaling are both fitted. The calibrator is selected by Brier score and log loss on the out-of-fold predictions.
- The selected calibrator is applied to the test set and stored in the model bundle.
- The report includes reliability curves before and after calibration, calibration slope, calibration intercept, Brier score, log loss, and expected calibration error.
- Calibration is reported unweighted and survey-weighted.
- Calibration by wealth quintile, sex, and state is reported where subgroup sizes permit.

### Threshold selection

- The default operating point targets at least 80% sensitivity while maximising specificity. Ties are broken by PPV.
- The threshold is selected on out-of-fold development predictions, never on the test set.
- The target sensitivity is configurable so that a clinic with different referral capacity can change the operating point.
- The report includes the threshold, confusion matrix, sensitivity, specificity, PPV, NPV, and F1 at the chosen threshold.
- A 0.5 threshold may be reported for reference but is not the default.
- Sensitivity at 80% specificity is reported as a threshold-independent comparison across models.

### Metrics and uncertainty

- Overall metrics: n, positives, prevalence, ROC-AUC, PR-AUC, Brier score, log loss, sensitivity at 80% specificity, sensitivity/specificity/PPV/NPV/F1 at the chosen threshold, calibration slope, calibration intercept, and expected calibration error.
- Confidence intervals use a cluster bootstrap that resamples PSUs. The default is 1,000 replicates; subgroup intervals may use fewer replicates when the subgroup is small.
- The report shows the point estimate and 95% interval. It does not treat the sample weight as a sample-size multiplier.
- Unweighted metrics are primary. Survey-weighted metrics are reported as a sensitivity analysis using the harmonised weight column.
- All metrics are computed from calibrated probabilities.

### Subgroup and fairness analysis

- Required subgroups: sex, age band (15–24, 25–34, 35–44, 45–49), wealth quintile, residence, education, and state where the state has at least 500 test respondents.
- Audit subgroups: religion and caste/tribe/social group. These are reported for fairness analysis but are not primary deployed features.
- Each subgroup row reports n, positives, prevalence, ROC-AUC, PR-AUC, sensitivity at the chosen threshold, and calibration error, with a gap versus the overall test set.
- Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged as unreliable. Metrics are still shown but are not used to claim fairness.
- The fairness sensitivity model includes religion and social group. The state sensitivity model includes state. Their metrics are compared with the primary model.
- The report explicitly states that removing a protected attribute does not make a model fair, and that subgroup gaps are descriptive, not causal.
- Sex-specific metrics are mandatory because the pooled sample is approximately 88% female.

### Explainability

- Global SHAP importance uses a fixed sample of the test set, defaulting to 5,000 respondents, so the run remains tractable.
- The report includes mean absolute SHAP values and a summary plot. A CSV of the underlying values is saved.
- A local explanation function returns the top contributing features for one respondent, with direction and impact. It is designed for the future API.
- Explanations are described as associations, not causes. The model card states this explicitly.
- If SHAP is unavailable in the runtime, the pipeline falls back to permutation importance or native LightGBM gain importance and records the fallback.

### Artifacts and model bundle

- Each run writes to a unique run directory named from the timestamp, seed, and a short hash.
- Required artifacts:
  - split assignment;
  - resolved config;
  - run metadata, including dataset fingerprint, row counts, package versions, and git commit when available;
  - overall metrics;
  - subgroup metrics;
  - calibration curve;
  - threshold analysis;
  - feature importance;
  - SHAP summary data and plot;
  - results report;
  - model card;
  - serialized model bundle.
- The model bundle contains the fitted model, calibrator, feature columns, categorical columns, threshold, target definition, training sample metadata, package versions, and run identifier.
- The model bundle is local-only and gitignored. Aggregate metrics, tables, plots, and the model card are publishable.
- The loader validates the feature schema and returns calibrated probabilities plus the threshold flag. It does not reimplement preprocessing.
- The model card covers intended use, target definition, sample, feature policy, performance, calibration, threshold, subgroup gaps, limitations, ethical considerations, non-goals, citation, and version date.
- The results report is readable Markdown with tables and embedded plots. Raw JSON remains available for downstream tooling.

### Reproducibility and data hygiene

- All random seeds are fixed and recorded. The split assignment is saved and reused.
- The resolved config is saved with every run.
- The pipeline records the dataset fingerprint: row count, column count, target prevalence, and a hash of the feature table when feasible.
- The pipeline records Python, pandas, scikit-learn, LightGBM, and SHAP versions.
- Modeling must run on Python 3.11 or 3.12 because LightGBM and SHAP wheels may lag on Python 3.14.
- The command-line entry point supports a small-sample smoke mode so the full pipeline can be exercised quickly without training on all 774,861 rows.
- Raw and processed microdata are never committed. Model binaries are never committed. Only aggregate results, plots, and the model card are publishable.
- The model card and results report state that random blood glucose is a screening indicator, not a diagnosis, and that the model is not for insurance, employment, or care-allocation decisions.

## Testing Decisions

A good test checks externally observable behaviour rather than internal implementation details. It should assert that a split is leakage-free, that a metric is numerically correct on a known input, that a calibrated probability is in range, that a threshold meets its sensitivity target on a separable synthetic dataset, and that the saved model reproduces its predictions after loading. Tests should not assert exact tree structures, exact LightGBM splits, or implementation-specific column order beyond the documented schema.

The primary test seam is the pipeline function that accepts a feature DataFrame, a config, and an output directory. One end-to-end integration test drives that seam with a small synthetic feature table. Pure helpers for split assignment, metrics, calibration, threshold selection, subgroup summaries, feature-policy validation, and artifact round-tripping are tested directly because they are deterministic and cheap to exercise.

Prior art is the existing synthetic-DataFrame style in the feature and target tests. Tests use small hand-built DataFrames and known arrays; they do not require the real NFHS-5 microdata.

### Integration test at the pipeline seam

- Build a synthetic feature table with 300–500 respondents, 30–50 clusters, one to three households per cluster, and one to two members per household.
- Generate the target from age and BMI with noise so both classes and both sexes are present.
- Include the real feature columns needed by the pipeline, including `cluster`, `household`, `female`, `age`, `bmi`, survey weight, subgroup columns, and a target.
- Use a small config: three folds, a 20% test set, a shallow LightGBM model, 20 bootstrap replicates, and a 50-row SHAP sample.
- Assert that every respondent is assigned to exactly one split and one fold.
- Assert that no cluster and no household appears in more than one split or development fold.
- Assert that the test set is approximately 20% of respondents and contains both classes.
- Assert that metrics are finite and in valid ranges, and that all required metric keys are present.
- Assert that calibrated probabilities are in [0, 1].
- Assert that the threshold achieves the requested sensitivity on the synthetic validation data when the data are separable enough.
- Assert that subgroup metrics sum to the test size and that small-group flags are applied.
- Assert that all required artifacts exist and that JSON and CSV files parse.
- Assert that the model bundle loads and reproduces the same calibrated probabilities.
- Assert that two runs with the same seed produce the same split assignment and the same metrics within tolerance.
- Assert that passing a target-derived column, an identifier, or a survey-design column as a feature raises a clear error.
- Assert that the primary feature set excludes `state`, `religion`, and `social_group`, and that the sensitivity variants include them.

### Pure-helper tests

- Split assignment: no group overlap, deterministic output, all rows assigned, fold sizes balanced, and the test set close to the requested fraction.
- Metrics: perfect separation, all-negative edge case, known small arrays with hand-computed ROC-AUC, PR-AUC, Brier score, log loss, and sensitivity at a fixed specificity.
- Calibration: isotonic output is monotonic, Platt output is in [0, 1], and calibration improves Brier score on a deliberately miscalibrated synthetic score.
- Threshold selection: returns the smallest threshold that meets the sensitivity target on a synthetic score, and returns a valid confusion matrix.
- Subgroup summaries: group counts sum to the input size, gaps are computed against the overall value, and small groups are flagged.
- Feature policy: forbidden columns are rejected; the primary, state, fairness, and combined feature sets have the expected sizes.
- Artifact round-trip: the model bundle saves, loads, validates its schema, and predicts the same probabilities.

### Test constraints

- The integration test must run in under about 60 seconds on a normal laptop.
- Tests must not download data or require DVC, cloud storage, or a GPU.
- Real-data training is verified by a manual smoke run, not by the CI test suite.
- The smoke mode uses a small random sample and a shallow model so it can be run before the full training run.

## Out of Scope

- The PR 15+ BP-based sensitivity model and the PR feature builder. See `.scratch/pr-bp-sensitivity-model/spec.md`.
- The questionnaire-only model, the deployed state-specific model, and the combined all-feature model beyond their role as sensitivity comparisons.
- A 5-fold × 10-seed 50-model ensemble. A single-seed 5-fold run is the required implementation; multi-seed ensembling is a future extension.
- Large-scale hyperparameter optimisation. A small grid or bounded search is sufficient for this spec.
- FastAPI service, web demo, bilingual UI, Docker, deployment, monitoring, and rollback.
- External validation, NFHS-6 replication, clinical validation, and causal inference.
- Fairness mitigation such as reweighting, adversarial debiasing, or group-specific thresholds. This spec audits fairness; it does not mitigate it.
- District-level or small-area estimates.
- Model compression, ONNX export, and mobile deployment.
- Redistributing raw or processed NFHS-5 microdata.

## Further Notes

- The primary model is deliberately state-free and excludes caste/tribe and religion. This is a fairness and leakage decision, not an oversight. The state and fairness variants exist to measure what those features would change.
- If the user later decides that the 35-feature superset should be the primary model, the feature-set config can switch to the combined variant. The pipeline should not need structural changes.
- The primary sample is approximately 88% female. Sex-specific metrics are not optional.
- The target includes self-reported diagnosis and medication, so the model can learn access-to-care patterns. Label-bias analysis is a required part of the report, not an appendix.
- The 15–49 restriction excludes men aged 50–54. The model card must state this and the report must show their count.
- Survey weights are used for prevalence and weighted metrics. They are never model features.
- The test set is cluster-grouped. If the user prefers household grouping, the config can switch the grouping column; the pipeline still checks for leakage.
- The chosen threshold is a policy choice. The model card and UI must show it, and the API must return it alongside the probability.
- Calibration is essential because the product will display probabilities. A well-ranked but poorly calibrated model is not acceptable for a screening tool.
- The model card should be honest about the difference between elevated random glucose, diagnosed diabetes, and future diabetes risk. The model does not predict future diabetes.
- The original project specification's multi-seed ensemble and four-model-variant plan is larger than this spec. This spec delivers the primary model and the two most important sensitivity comparisons first; the remaining variants can follow once the pipeline is proven.
