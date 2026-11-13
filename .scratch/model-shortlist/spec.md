# Model shortlist and leakage-safe selection

**Status:** ready-for-agent

## Product scope amendment (13 November 2026)

The product is a self-service web app for individuals, not a health-worker tool. There is no ASHA/ANM/clinic workflow, no referral queue, and no assumed access to a blood glucose test. The measurement model remains the deployed default: height, weight, and waist are self-reported from scales and tape measures that are commonly available at home, and blood pressure is optional where the user has a reading. The questionnaire-only model is the fallback for users who cannot provide measurements. The operating point is a user-facing risk band (lower / moderate / higher), not a binary referral. The shortlist, comparison, ranking, feature-bundle, promotion, and leakage rules below are unchanged.

## Problem Statement

The published primary model was selected from three families (prevalence-only, logistic regression, LightGBM) on a single seed, with class-weighted training and a PR-AUC ranking rule. The main spec asks for an accuracy-first choice among pre-registered candidates, four model variants, calibration and subgroup gates, and a multi-seed ensemble. The shortlist research (`.scratch/model-shortlist/research.md`) shows the credible performance ceiling for this task is ROC-AUC 0.68–0.71, so the work is not a hunt for a step change; it is a controlled comparison that can detect small real gains and protect calibration and fairness.

The comparison needs to score several candidates on the same cluster-grouped split and then deploy one winner. The shared training pipeline currently selects a family on the development folds and immediately evaluates the held-out test set in the same call. If the comparison runner calls that pipeline once per candidate, the test set is read once per candidate. The winner would then be chosen partly on data that is supposed to be untouched, and the reported winner score would be optimistic. The same risk applies to the feature-bundle comparison and to any future multi-candidate run.

The project also needs one frozen deployable artifact before API and UI work begins. Today there is no fixed shortlist, no random-forest, XGBoost, CatBoost, spline, or monotone candidate, no comparison artifact, and the PR sensitivity runs use the old recipe. The pipeline needs a seam that makes the test split unreachable during selection without creating a second modeling implementation.

## Solution

Run one fixed comparison over seven scored candidates plus a prevalence reference, on two primary feature sets (measurement and questionnaire-only) and a pre-registered block-ablation grid inside the measurement pool. Train unweighted, calibrate post hoc, rank by a gate-based rule, select one deployed model from the eligible measurement bundles, freeze its artifact, then rebuild the PR split and rerun the sensitivity models with the same recipe.

Split training into a preparation phase, a selection phase, and a final evaluation phase. The selection phase runs only on the development rows and returns out-of-fold evidence for every candidate, plus a fit handle that can refit any candidate. The final phase fits the chosen candidate and reads the test split exactly once. The comparison runner uses only the selection phase; the freeze path uses the fit handle and final evaluation. The existing single-model entry point keeps its signature and behaviour by composing the phases internally.

A neural net and one algorithmic feature-selection experiment are explored only after the scored comparison and can be promoted only under a pre-registered rule. The shortlist, ranking rule, promotion rule, feature sets, feature bundles, and pipeline phases are fixed before any scores are examined. The held-out test set is used once for final reporting.

## User Stories

1. As a researcher, I want a shortlist fixed before scoring, so that the winner is not the result of trying models until one looks good.
2. As a researcher, I want every candidate trained with the same preprocessing, folds, and tuning budget, so that family and feature-bundle differences are not confounded.
3. As a researcher, I want unweighted training with post-hoc calibration, so that predicted probabilities mean what they say.
4. As a researcher, I want a spline logistic regression and a monotone LightGBM, so that nonlinearity and clinical direction are tested without assuming a tree ensemble is necessary.
5. As a reviewer, I want a comparison table with discrimination, calibration, subgroup gaps, and net benefit for every candidate, so that the choice can be audited.
6. As a reviewer, I want calibration and subgroup gaps to veto a candidate that ranks higher, so that accuracy cannot buy a misleading probability.
7. As a maintainer, I want the questionnaire-only models reported separately, so that the accessibility trade-off is visible without weakening the deployed model.
8. As a maintainer, I want seed averaging only for the winner and only when stochastic, so that compute is not spent averaging deterministic models.
9. As a maintainer, I want the winning configuration and seed recorded in the artifact, so that serving cannot silently diverge from selection.
10. As a portfolio author, I want the PR sensitivity runs rebuilt with the same recipe, so that the published comparison does not mix methods.
11. As a maintainer, I want pre-registered feature bundles tested before the winner is frozen, so a smaller deployment set can be chosen without post-hoc feature selection.
12. As a researcher, I want candidate selection to use out-of-fold predictions only, so that the winner is not chosen on the test set.
13. As a researcher, I want calibration and the risk-band cut-offs fitted per candidate on that candidate's own out-of-fold predictions, so that the reported operating point is honest.
14. As a researcher, I want net benefit at plausible thresholds available from out-of-fold evidence, so that clinical-utility context is not derived from the test set.
15. As a reviewer, I want to see that the test split was untouched during selection, so that I can trust the winner and its score.
16. As a reviewer, I want per-candidate out-of-fold metrics, subgroup gaps, and threshold metrics, so that the ranking and every veto can be audited.
17. As a reviewer, I want test metrics only for the chosen winner, so that non-winners cannot be compared on data they were not selected on.
18. As a reviewer, I want the comparison artifact to state that selection used out-of-fold evidence and that the test set was evaluated once, so that the method is visible without reading code.
19. As a maintainer, I want one selection seam used by both the comparison runner and the existing single-model pipeline, so that there is no second modeling implementation.
20. As a maintainer, I want to refit any candidate from the selection evidence, so that the comparison layer can choose a winner without re-running cross-validation.
21. As a maintainer, I want to pin a chosen family and its tuned parameters for the final run, so that the frozen artifact is exactly the candidate the comparison selected.
22. As a maintainer, I want the existing primary and PR entry points to keep their signatures and artifact sets, so that this work does not become a second migration.
23. As a maintainer, I want the selection phase to reject a frame that contains test rows, so that a caller cannot leak the test split by mistake.
24. As a maintainer, I want a regression test that fails if test-derived metrics appear in comparison artifacts, so that the separation is protected by behaviour rather than convention.
25. As a maintainer, I want the selection result to record the development row count, fold count, and split summary, so that run metadata can distinguish selection rows from test rows.
26. As a future contributor, I want the fit handle to be the single place where seed averaging can be added, so that a multi-seed ensemble does not require another selection path.
27. As a portfolio author, I want the published comparison to say that the test set was used once, so that the project's evaluation discipline is part of the result.
28. As a future contributor, I want the candidate list and pinned winner to be configuration, not hard-coded family names, so that adding a candidate does not change the pipeline's control flow.
29. As a maintainer, I want the feature-bundle rule to prefer a smaller set only when its loss is inside the pre-registered margin, so that parsimony is explicit and auditable.

## Implementation Decisions

### Shortlist

- B0 prevalence-only is the unscored reference floor and cannot win.
- Scored candidates: C1 logistic regression; C2 logistic regression with restricted cubic splines on age, BMI, waist, systolic BP, diastolic BP; C3 random forest; C4 LightGBM; C5 XGBoost; C6 CatBoost; C7 LightGBM with monotone constraints.
- C7 constrains risk non-decreasing in age, BMI, waist, systolic BP, and diastolic BP only. No constraints on wealth, education, diet, or behaviour.
- No stacking, no second neural architecture, no TabNet or transformer. Exploration is limited to one MLP and one feature-selection experiment.

### Feature sets and bundles

- Measurement feature set: the existing 32 primary features.
- Questionnaire-only feature set: the same features without waist, hip, waist-to-hip ratio, systolic and diastolic BP, BP reading count, measured high BP, and BP medication. It keeps age, sex, height, weight, BMI, education, wealth, residence, marital status, children ever born, tobacco, alcohol, and diet.
- Pre-registered measurement bundles: `measurement_full` (32), `measurement_no_anthropometry` (26; drops height, weight, BMI, waist, hip, and waist-to-hip ratio), `measurement_no_bp` (27; drops the five BP columns), and `measurement_no_diet` (23; drops the nine diet columns). The substance block is not ablated because it is cheap to collect and has a strong clinical prior.
- All seven families run on each competing measurement bundle and on the questionnaire-only set. The questionnaire-only results are the fallback for users who cannot provide measurements and do not compete for the default deployment.
- The measurement pool selects the deployed model. Its eligible bundles are compared by the feature-bundle rule below.

### Pipeline phases

- **Preparation seam.** A public preparation step applies the configured sample filters, resolves or loads the cluster-grouped split assignment, runs the existing cluster and household leakage checks, and returns the development frame, test frame, assignment, row counts, and split summary. The comparison runner receives the development frame; the final run receives both frames.
- **Candidate configuration.** A candidate records a model family, a parameter mapping, and a stable label. Configuration can supply a candidate list and a pinned candidate. The trial loop is driven by the configured candidates, not a hard-coded family tuple. With no candidate list, the documented family list and the existing winner order are preserved. With a pinned candidate, the final run scores and fits exactly that family and those parameters.
- **Selection evidence.** `evaluate_candidates(development, config)` scores every configured candidate on the existing development folds and returns out-of-fold evidence. It accepts only development rows, validates that every row has the development split and a valid fold, and rejects any frame containing a test row before fitting anything. It returns evidence, not a winner.
- **Selection evidence contents.** Each candidate record contains the candidate label, family, and parameters; raw out-of-fold scores, calibrated out-of-fold probabilities, target, covered mask, and fold identifier; the selected calibration method and report; the reference 80%-sensitivity operating point with confusion counts and rates; the age-relative score bands; discrimination, calibration, Brier, log loss, and ECE; development subgroup metrics with the existing reliability and audit rules; the reliability curve; net benefit at 0.05–0.20; fold-level metrics; and the tuning record.
- **Metrics prefactor.** Add a decision-curve net-benefit helper at 0.05–0.20. Make the subgroup-metrics helper work for development out-of-fold rows or test rows, carrying an explicit split label so the two reports cannot be confused.
- **Fit handle.** The selection result exposes a fit handle that refits any named candidate on all development rows and returns the model bundle used at inference, with the raw model, category levels, calibrator, score reference, feature columns, and candidate metadata. The fit handle is the only bridge from selection to final evaluation and the extension point for seed averaging.
- **Final evaluation.** The existing single-model pipeline composes the phases: prepare the data, run selection or accept the pinned candidate, fit that candidate, read the test split exactly once, and write the existing artifact set. No code path inside selection computes test metrics, test subgroup rows, test calibration curves, test feature importance, or test SHAP values.
- **Backward compatibility.** The single-model pipeline signature and artifact keys stay backward compatible. The pinned candidate is an optional configuration field, not a required new argument. Existing primary and PR entry points keep their behaviour.
- **Guard rails.** The selection step rejects any frame with a test row, any development row without a valid fold, and any missing or forbidden feature column. A comparison artifact may not contain a test metric block for a non-winner. The winner's test metrics come only from the final run, which records that the test set was evaluated once.

### Training

- Train unweighted. Remove `class_weight="balanced"` from logistic regression and `is_unbalance` from LightGBM.
- No SMOTE, no resampling, no synthetic rows.
- One weighted LightGBM run is kept as a labelled sensitivity check, not a candidate.
- All candidates share the same cluster-grouped split assignment, 20% test holdout, and five development folds.
- Each family receives the same bounded tuning budget: three settings, evaluated on the same folds. CatBoost iterations are capped. Random forest runs full-fat first and is tuned only if it dominates wall-clock time. The search space, trial count, and selected settings are recorded.

### Calibration and operating points

- Fit isotonic and Platt on out-of-fold predictions for every candidate; choose by Brier and log loss.
- Apply the chosen calibrator to test predictions exactly once.
- The deployed score is the user's percentile within their age band, from 0 to 100. Bands are 0–49 lower, 50–79 moderate, and 80–100 higher. The score is a comparison, not a probability.
- Show the absolute probability as a natural frequency alongside the score, so a high percentile in a young age group is not mistaken for a high absolute risk.
- Store the score reference (the age-band distribution) with the frozen artifact; report the band distribution, calibration by band, and subgroup band rates.
- The 80%-sensitivity threshold remains a reference operating point for model selection and reporting, not the app's primary output.
- Report the reference 80%-sensitivity operating point, confusion matrix, sensitivity, specificity, PPV, NPV, F1, calibration slope and intercept, Brier, log loss, and reliability curves for every candidate.

### Ranking rule

- Rank by validation PR-AUC; ROC-AUC is a secondary check.
- A candidate is vetoed if any of the following holds:
  - calibrated Brier is more than 0.005 worse than the best candidate's;
  - calibration slope is outside 0.8–1.2 or intercept outside −0.1–0.1;
  - a subgroup flagged reliable has sensitivity at the reference operating point more than 0.10 away from the overall sensitivity.
- Decision-curve net benefit is reported for plausible thresholds (0.05–0.20) and used as context and as a tie-breaker, not as a veto.
- The test set is not used for ranking, early stopping, calibration, or threshold selection.
- The winner is the highest-ranked non-vetoed candidate in the measurement pool.

### Feature-bundle rule

- Choose the winner within each competing measurement bundle using the ranking rule above.
- A bundle is eligible only when its winner passes every ranking veto.
- Let the reference be the eligible bundle winner with the highest PR-AUC.
- Among eligible bundles whose PR-AUC is within 0.005 of the reference and whose Brier is no more than 0.005 worse than the reference, choose the one with the fewest features; break ties by higher PR-AUC, then lower Brier.
- If no bundle passes, the comparison has no winner and the run fails rather than deploying an ungated model.
- The rule is applied to out-of-fold predictions only. The test set is evaluated once for the chosen bundle's winner.

### Seed averaging

- Apply ten seeds only to the winner.
- Skip seed averaging when the winner is deterministic (logistic regression).
- Average predicted probabilities across the ten refits; fit the calibrator on pooled out-of-fold predictions; report mean and standard deviation across seeds.
- Record the seed list and ensemble size in the artifact.

### Exploration

- Train one MLP on the same folds and the frozen winner's feature set after the seven candidates are scored.
- Promote it only if all three hold: ΔROC-AUC versus the winner is at least 0.005 with a cluster-bootstrap 95% CI excluding zero; calibrated Brier is within 0.005 of the winner; no reliable subgroup's sensitivity gap exceeds 0.10.
- If not promoted, report it as an exploration result and do not deploy it.
- After the winner is frozen, run one labelled algorithmic feature-selection experiment (permutation importance or recursive feature elimination) with the selector fitted inside the development folds only. Promote it only under the same three conditions as the MLP; otherwise report it as exploration and do not deploy it. Promotion requires a new freeze and PR rerun.

### Artifacts and reruns

- Write one comparison artifact per feature set and per competing measurement bundle: `comparison.json` and `comparison.md`, containing per-candidate out-of-fold metrics, calibration, subgroup summary, net benefit, tuning record, veto status, bundle name, and the winner.
- Non-winner comparison records contain no test metric block. The winner record points at the final run, and its test metrics come only from that run.
- The comparison artifact records the selection split, candidate labels, ranking and veto outcomes, and a test-evaluated-once flag.
- Freeze the winner as a run directory with the model bundle, feature schema, calibrator, score reference, bands, metrics, model card, and run metadata.
- After the winner is frozen, rebuild the PR split against the primary split assignment, rerun PR 15+ and PR 15–49 with the same recipe, regenerate the comparison report, and mark the previous PR artifacts as the previous recipe.
- Record the interpreter actually used (3.14.6 today). Do not claim an interpreter the runs did not use, and do not install a different one just to match prose.

### Documentation and spec

- Amend the main spec to state that this shortlist, ranking rule, feature-bundle rule, and leakage-safe selection phases govern launch selection, that the ten-seed ensemble applies only to the stochastic winner, and that the HbA1c sensitivity target is withdrawn because the public NFHS-5 release exposes DBS eligibility but no HbA1c values.
- Replace the README results table from the new run records; update the model card and glossary.
- Keep aggregate results and code public; keep microdata, split assignments, per-respondent SHAP values, and model binaries local.

## Testing Decisions

Good tests assert externally visible behaviour at the highest seam: the preparation and selection contracts, the comparison artifact, and the existing single-model pipeline. They do not assert private helper call order or internal attribute names.

- Unit tests for every family builder on a small synthetic matrix: random forest, XGBoost, CatBoost, spline logistic, and monotone LightGBM produce probabilities in [0, 1].
- Unit tests for the two primary feature sets and all four measurement bundles: no target-derived, identifier, or survey-design columns; the questionnaire-only set excludes every BP and waist/hip column; each ablation bundle is exactly the full measurement set minus one block; the documented counts and block memberships hold.
- Test the preparation seam: development and test rows are disjoint, the assignment matches both frames, and the split summary reports the expected counts.
- Test the selection seam on a synthetic cluster-grouped table: one evidence record per configured candidate and setting, finite calibrated probabilities, a selected calibration method, a selected threshold, out-of-fold metrics, development subgroup rows, and a net-benefit table.
- Test that selection covers exactly the development rows and no test rows.
- Test the guard: a frame containing a test row is rejected with a clear error before any model is fitted.
- Test the sentinel path: a test row whose values would break training does not affect selection.
- Test the fit handle: it refits the named candidate and returns a bundle whose threshold, calibrator, feature columns, and predictions are consistent with the evidence.
- Test the pinned-candidate path: the final run uses the pinned family and parameters and does not score the other candidates.
- Unit tests for the ranking rule: a higher PR-AUC candidate is vetoed by worse Brier, bad slope, or a large subgroup sensitivity gap; a lower PR-AUC candidate wins when the higher one is vetoed.
- Unit tests for the feature-bundle rule: a smaller bundle within the PR-AUC and Brier margins wins; a smaller bundle outside either margin loses; the full bundle wins when no reduced bundle qualifies.
- Unit test for the post-freeze feature-selection experiment: selection runs inside development folds only, and a non-promoted result changes no artifact.
- Unit test for the promotion rule: all three conditions must hold.
- Unit test for seed averaging: stochastic winners get ten refits; logistic winners get one.
- Integration test at the pipeline seam on a synthetic table: all seven families train and produce a comparison artifact with the required keys; no training on real data.
- Test the comparison-artifact contract: non-winner records contain out-of-fold metrics only, the winner record points at the final run, and a test-metric block cannot appear before final evaluation.
- Test backward compatibility: the existing primary and PR pipeline tests still pass, and the final run still writes the same artifact keys and test metrics.
- CLI guard tests: a missing feature table fails with a clear message; the comparison entry point and the primary run entry point reject missing inputs without a traceback.
- Existing tests continue to pass; no model is trained in CI on real data.

## Out of Scope

- FastAPI, web app, deployment, CI, and observability.
- A 50-model ensemble over every candidate.
- Stacking, tabular foundation models, TabNet, transformers, or a second neural architecture.
- SMOTE or any resampling method.
- HbA1c as a feature or target; the public release has no HbA1c values.
- External validation, NFHS-6, district estimates, and fairness mitigation.
- Changing the primary target definition, the exclusion rules, the split assignment, or the operating-point target.
- Open-ended feature search; only the pre-registered measurement bundles and the single post-freeze experiment are in scope.
- Changing the existing default winner order for primary and PR runs.

## Further Notes

- The credible ceiling from the research is ROC-AUC 0.68–0.71. A candidate that wins by 0.002 is not a step change; the comparison exists to justify the deployed choice, not to manufacture a headline.
- NFHS-5 lacks family history and physical activity, so IDRS cannot be replicated. It is a benchmark for ambition, not a validation target.
- The questionnaire-only model is expected to be weaker. Its purpose is to show what a measurement-free deployment would cost, not to replace the measurement model.
- The bundle rule prefers a smaller set only when its loss is inside the pre-registered margin. It is a parsimony rule, not a claim that fewer features are always better.
- The PR 15+ model cannot include full anthropometry for adults 50+, because those measurements are not collected for that age group in the PR file.
- The current pipeline evaluates the test split unconditionally. That is safe for a single model but unsafe for a comparison. The preparation and selection seams exist to make test leakage impossible to reuse by accident.
- The current select-and-refit helper hard-codes three families. Candidate configuration replaces that tuple so the selection seam and the new model families compose.
- The comparison runner must not call the full single-model pipeline once per candidate. It consumes selection evidence and only the winner goes through final evaluation.
- The issue tracker is local markdown under `.scratch/`; this spec is not version-controlled with the source.
