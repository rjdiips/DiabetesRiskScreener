# India Diabetes Risk Screener

A research screener that estimates the probability of elevated random blood glucose or diagnosed diabetes among Indian adults, built on NFHS-5. This glossary fixes the project's own vocabulary; it does not describe implementation.

## Language

### Evaluation

**Reference baseline**:
The model that ignores every input and predicts the overall positive rate. It is the do-nothing floor that any candidate must beat.
_Avoid_: Prevalence model, dummy model, null model

**Cluster-grouped split**:
An evaluation split that keeps every survey cluster (PSU) entirely on one side, so respondents from the same community cannot appear in both training and test data.
_Avoid_: Random split, cluster split

**Household overlap check**:
A verification that no household has members on both sides of a split. It is reported alongside cluster grouping, which normally makes it redundant, to prove that redundancy.
_Avoid_: Leakage check

### Model selection

**Model family**:
The learning algorithm under consideration, such as logistic regression, random forest, or LightGBM. Family is independent of which features the model sees.
_Avoid_: Model type, algorithm, classifier

**Model variant**:
The feature set a model is trained on, such as measurement, questionnaire-only, state, or fairness. Variant is independent of family.
_Avoid_: Model version, feature set

**Candidate**:
One shortlisted combination of a model family, a model variant, and fixed settings, evaluated on the shared folds.
_Avoid_: Experiment, run, trial

**Shortlist**:
The fixed set of candidates named before any scores are examined. Only a shortlisted candidate can become the deployed model; later ideas belong in exploration.
_Avoid_: Candidate pool, long list, model list

**Exploration bucket**:
Models that are trained and reported but cannot become the deployed model unless they pass the promotion rule. It exists so curiosity does not silently widen the shortlist.
_Avoid_: Extra candidates, bonus models

**Promotion rule**:
The pre-agreed evidence a model from the exploration bucket must produce to enter the shortlist, covering effect size, uncertainty, calibration, and subgroup gaps.
_Avoid_: Promotion criteria, upgrade test

**Ranking rule**:
The pre-agreed order and vetoes applied to shortlisted candidates: a headline metric ranks them, while calibration and subgroup gaps can remove a higher-ranked candidate.
_Avoid_: Scoring, selection criteria

**Winner**:
The shortlisted candidate that the ranking rule selects. The winner is not necessarily the candidate with the single highest headline number.
_Avoid_: Best model, top model

**Deployed model**:
The artifact the API serves. It is derived from the winner, possibly as a seed-averaged ensemble, and is the only model the app ever calls.
_Avoid_: Production model, final model

**Seed-averaged ensemble**:
The winner refit with several random seeds, with predicted probabilities averaged across the refits. It reduces seed-to-seed luck; it is not a mixture of different model families.
_Avoid_: Ensemble, bagging, stacking

**Screening threshold**:
The probability cutoff that turns a predicted risk into a referral decision, chosen to meet a target sensitivity and stored with the deployed model. The deployed model stores one cutoff per age band so the same sensitivity target is met across the age gradient.
_Avoid_: Cutoff, decision boundary, operating point

**Age-band threshold**:
The referral cutoff used inside one age band. It is chosen on out-of-fold predictions to meet the sensitivity target for that band, and it replaces a single global cutoff that would miss most young cases and refer almost every older adult.
_Avoid_: Subgroup threshold, stratified cutoff

**Feature bundle**:
A pre-registered subset of the measurement features, created by removing one measurement block (anthropometry, blood pressure, or diet). A bundle competes for deployment only through the feature-bundle rule, which prefers fewer features only when performance is effectively unchanged.
_Avoid_: Feature subset, reduced model

### Product

**Screener app**:
The web surface a health worker uses: the questionnaire, result, model transparency, and about pages, plus the API behind them. The app consumes the deployed model; it does not train or choose one.
_Avoid_: App, UI, frontend, demo

**Referral decision**:
The app's output of low, moderate, or high risk derived from the calibrated probability and the screening threshold. It is guidance to seek a blood test, never a diagnosis.
_Avoid_: Diagnosis, prediction, classification
