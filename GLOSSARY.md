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
The artifact the API serves to the self-service app. It is derived from the winner, possibly as a seed-averaged ensemble, and is the only model the app calls by default.
_Avoid_: Production model, final model

**Seed-averaged ensemble**:
The winner refit with several random seeds, with predicted probabilities averaged across the refits. It reduces seed-to-seed luck; it is not a mixture of different model families.
_Avoid_: Ensemble, bagging, stacking

**Risk score (age-relative)**:
The user's percentile within their age band, from 0 to 100. A score of 85 means the predicted risk is higher than 85% of people of the same age. It is a comparison, not a probability; the absolute probability is shown separately.
_Avoid_: Risk percentage, points score, probability score

**Percentile curve**:
A small distribution of age-relative scores for the user's age band, with the user's position marked. It shows how unusual the score is for that age, not the chance of disease.
_Avoid_: Bell curve, risk curve

**Feature bundle**:
A pre-registered subset of the measurement features, created by removing one measurement block (anthropometry, blood pressure, or diet). A bundle competes for deployment only through the feature-bundle rule, which prefers fewer features only when performance is effectively unchanged.
_Avoid_: Feature subset, reduced model

### Product

**Screener app**:
The web surface an individual uses: the questionnaire, result, model transparency, and about pages, plus the API behind them. The app consumes the deployed model; it does not train or choose one.
_Avoid_: App, UI, frontend, demo

**Risk band**:
The app's coarse label (lower / moderate / higher) derived from the age-relative score: 0–49 lower, 50–79 moderate, 80–100 higher. It means “compared with people your age”, not a probability, a diagnosis, or a guarantee that a blood test is available.
_Avoid_: Referral, diagnosis, prediction, classification
