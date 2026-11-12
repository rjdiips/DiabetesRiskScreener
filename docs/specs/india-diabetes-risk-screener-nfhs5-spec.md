# India Diabetes Risk Screener — NFHS-5 Specification

**Status:** Draft v1.2  
**Owner:** Rajdeep Sen  
**Date:** 26 October 2026  
**Repository:** TBD  
**Live demo:** TBD

---

## 0. Scope amendment (1 November 2026)

Model selection and launch scope follow `.scratch/model-shortlist/spec.md`. This section supersedes any conflicting text below.

- **Fixed shortlist.** The launch comparison is a prevalence-only reference plus seven scored candidates: logistic regression, spline logistic regression, random forest, LightGBM, XGBoost, CatBoost, and monotone-constrained LightGBM. The list is fixed before scoring; no candidate is added after results are seen.
- **Ensemble scope.** The 5-fold × 10-seed ensemble applies only to the selected winner, and only when the winner is stochastic. It is not run for every candidate. A deterministic winner (logistic regression) is deployed as a single fitted model.
- **Feature sets.** All seven families run on both the measurement feature set and the questionnaire-only feature set. The measurement pool selects the deployed model; questionnaire-only results are reported as the accessibility comparison and do not compete for deployment.
- **Training and calibration.** All candidates train unweighted. Class weighting, SMOTE, and resampling are not used for selection. Calibration is fitted post hoc on out-of-fold predictions, and the screening threshold is chosen on out-of-fold predictions at ≥80% sensitivity.
- **Ranking.** PR-AUC ranks candidates, with calibration and subgroup gaps as vetoes. The held-out test set is evaluated once and is not used for ranking, calibration, or threshold selection.
- **HbA1c sensitivity withdrawn.** The public NFHS-5 release exposes dried-blood-spot *eligibility* (`shdbs`, 6.3%) but no HbA1c values, and publishes no biomarker recode. The HbA1c sensitivity target in §13 is therefore not implementable from this data. The fact-sheet target remains the secondary outcome.
- **PR sensitivity runs.** PR 15+ and PR 15–49 are rebuilt with the same recipe after the primary winner is frozen. Until then, the existing PR artifacts are the previous recipe and must not be quoted beside new primary numbers.
- **Neural nets.** A neural net is an exploration result, not a scored candidate. It can be promoted only under the pre-registered rule in the model-shortlist spec.

---

## 1. Summary

**India Diabetes Risk Screener** is an India-specific, questionnaire-and-measurement app that estimates the probability of elevated blood glucose or diagnosed diabetes using data from the National Family Health Survey 2019–21 (NFHS-5).

It is designed for low-resource primary care in India:

- A short form that an ASHA/ANM worker can complete
- Simple measurements: height, weight, waist circumference, blood pressure
- A calibrated risk score with plain-language explanations
- Referral guidance to the nearest Health & Wellness Centre
- The full launch surface: screener, result, model transparency, about, and API
- Fairness analysis across wealth, caste/tribe, religion, gender, and rural/urban groups

The launch app is an **online, responsive web app** with server-side inference. It is **English-only at launch**. Hindi/English bilingual support and an offline-first PWA are explicitly deferred to post-launch and are not launch acceptance criteria. At launch, model selection is **accuracy-first**: choose the best validated discrimination and calibration, and do not trade measured accuracy for browser model size or offline latency.

The app is a **research and portfolio demo**. It is not a diagnostic test, not a validated screening programme, and not medical advice.

---

## 2. Why NFHS-5

NFHS-5 is the strongest accessible India-specific dataset for this project.

| Property | NFHS-5 |
|---|---|
| Survey period | 2019–21 |
| Households | 636,699 |
| Women interviewed | 724,115 |
| Men interviewed | 101,839 |
| Individual records (women 15–49, men 15–54) | ~1.44 million |
| Adults with valid random glucose | ~1.25 million |
| Districts | 707 |
| States/UTs | 28 states and 8 UTs |
| Blood glucose | Random blood glucose for women 15–49 and men 15–54 |
| HbA1c | Dried-blood-spot subsample |
| Blood pressure | Measured |
| Anthropometry | Height, weight, waist, hip |
| Socioeconomic data | Wealth index, education, caste/tribe, religion, rural/urban |
| Access | DHS Program registration and data-use agreement |
| Cost | Free |

### Why it is stronger than BRFSS for India

- It is **India-specific**, not a U.S. survey.
- It is **district-representative**, which supports a district care-cascade story.
- It includes **random blood glucose**, not only self-reported diagnosis.
- It includes **HbA1c on a subsample**, enabling a stronger sensitivity analysis.
- It includes **caste/tribe, religion, wealth, and rural/urban** variables for fairness analysis.
- It supports a **low-resource primary care** product story that is relevant to Indian health-tech employers.

### Why it is not automatically “better” than every alternative

- It is **cross-sectional**: it measures glucose and diagnosis at one point in time, not future diabetes incidence.
- Random blood glucose is **not fasting glucose** and is a **screening indicator**, not a diagnosis.
- A single glucose reading can misclassify.
- The HbA1c subsample is smaller.
- It has **no follow-up**, so it cannot model incident diabetes.
- It has **no readmission outcome**, so it is not a replacement for a clinical readmission project.
- It is from **2019–21**, not 2026.
- NFHS-6 (2023–24) is newer, but public microdata availability is the constraint; use NFHS-5 now and validate on NFHS-6 when microdata is released.

### Comparison with other options

| Dataset | India-specific | Target | Strength | Main limitation |
|---|---|---|---|---|
| **NFHS-5** | Yes | Random glucose / diagnosed diabetes | Best accessible India screener | Cross-sectional, random glucose |
| **NFHS-6** | Yes | Same | Newest India data | Microdata may not be available yet |
| **LASI** | Yes | Diabetes and aging, longitudinal | Strong longitudinal story | Access process; older adults |
| **ICMR-INDIAB** | Yes | Diabetes prevalence | Strong clinical/epidemiology | Not public |
| **Hospital EHR partnership** | Yes | Readmission/complications | Strongest clinical ML | Requires IRB/DUA and a partner |
| **BRFSS** | No | Self-reported diabetes | Large and simple | U.S., self-report |
| **NHANES** | No | Lab-defined diabetes | Biomarkers | U.S., cross-sectional |
| **MIMIC-IV** | No | Readmission/mortality | Modern EHR | U.S., credentialed |
| **UK Biobank** | No | Incident diabetes | Longitudinal, biomarkers | U.K., application and fee |
| **Pima / Kaggle “Indian diabetes”** | No | Not India | Easy | Pima is Native American; many Kaggle sets are synthetic or renamed |

---

## 3. Goals

### Product goals

- Let an ASHA/ANM worker complete a short risk assessment in under three minutes.
- Use only information available before a blood glucose test.
- Return a calibrated probability of elevated blood glucose or diagnosed diabetes.
- Show the top factors behind the estimate.
- Give clear referral guidance, not a diagnosis.
- Ship the full app experience as a responsive web app: screener, result, model transparency, and about.
- Prioritise model accuracy over model size and offline constraints at launch.
- Keep the form usable on low-end phones over a normal mobile connection.

### Engineering goals

- Reproducible pipeline from NFHS-5 recode files to model artifact.
- Household-grouped train/test splits to prevent leakage across respondents in the same household.
- Multi-seed LightGBM ensemble with probability calibration.
- Accuracy-first selection among pre-registered model variants, with calibration and subgroup reporting as gates.
- Survey-weighted evaluation and subgroup reporting.
- Typed FastAPI service and a tested web frontend.
- CI and a public deployment.
- Committed evaluation report and model card.

### Launch scope

| Included at launch | Deferred to post-launch |
|---|---|
| Full-featured English web app: Screener, Result, Model, About | Hindi/English bilingual UI |
| FastAPI service with predict, model-info, health, and docs | Additional languages |
| Server-side inference and SHAP explanations | Offline-first PWA and on-device inference |
| Accuracy-first model selection | Model distillation/ONNX export |
| CI and public deployment | Docker packaging and containerized deployment |

### Portfolio goals

- A distinctly India-specific project, not another U.S. dataset.
- A strong public-health and low-resource engineering story.
- Three resume bullets with real metrics.
- A live demo and GitHub repository.

---

## 4. Non-goals

- **Not** a diagnostic tool.
- **Not** a validated screening programme.
- **Not** a replacement for fasting glucose, HbA1c, or oral glucose tolerance testing.
- **Not** a predictor of future diabetes incidence.
- **Not** a tool for insurance, employment, marriage, or care-allocation decisions.
- **Not** a causal model. SHAP values are associations, not causes.
- **Not** a tool that stores identifiable patient data.
- **Not** an LLM-first product. Any LLM feature must be optional and grounded.

---

## 5. Users and use cases

### Primary user

**ASHA / ANM / community health worker**  
Works at a Health & Wellness Centre or in a community setting. Needs a fast, low-literacy tool that helps decide who should be referred for a blood glucose test.

### Secondary users

- **Self-service user** — wants a private, questionnaire-based estimate.
- **PHC doctor or nurse** — wants a second signal and a clear referral note.
- **District health officer** — wants aggregate risk and care-cascade views.
- **Recruiter or reviewer** — wants to see model quality, fairness, and deployment quality.
- **Data scientist** — wants to inspect target definition, calibration, and subgroup metrics.

### User stories

- As an ASHA worker, I want to ask a short set of questions and get a risk score so I can prioritise referrals.
- As an ASHA worker, I want the app to load quickly on a typical mobile connection so I can use it in the field.
- As a user, I want to know that this is not a diagnosis and that I should get a blood test.
- As a reviewer, I want to see calibration and subgroup performance, not just accuracy.
- As a developer, I want a documented API and a stable feature schema.

---

## 6. Success metrics

### Model quality

| Metric | Target |
|---|---|
| ROC-AUC | Beat logistic-regression baseline |
| PR-AUC | Beat baseline; report 95% CI |
| Model selection | Highest validated ROC-AUC and PR-AUC among pre-registered candidates, with calibration and subgroup reporting as gates; no model-size or offline constraint |
| Brier score | Lower than uncalibrated baseline |
| Calibration | Reliability curve close to diagonal; report slope and intercept |
| Sensitivity at 80% specificity | Report and compare to baseline |
| Sensitivity at chosen threshold | ≥ 80% for the primary screening threshold |
| Subgroup performance | Report by wealth, caste/tribe, religion, gender, rural/urban, state |
| Household leakage | Zero household overlap between train and test |

Targets will be finalised after the baseline run. Do not invent results.

### Engineering quality

| Metric | Target |
|---|---|
| API tests | All routes covered, including validation and error cases |
| Test coverage | ≥ 85% for `src/screener` |
| Online predict latency | p95 < 300 ms |
| Frontend flows | Screener submission, result rendering, model page, and API-error state covered |
| Deployment | Public URL with `/docs` and `/health` |
| Documentation | README, model card, evaluation report, and this spec |

### Product quality

| Metric | Target |
|---|---|
| Time to complete | < 3 minutes |
| Form completion rate | > 80% in usability testing |
| Referral comprehension | Users can explain what “high risk” means |
| Responsive UI | Usable on a low-end phone over a normal mobile connection |
| Accessibility | Low-literacy labels, icons, keyboard/screen-reader support |

Bilingual UI and offline-first PWA targets are tracked in the post-launch backlog, not in the launch metrics.

---

## 7. Dataset

### Primary dataset

**National Family Health Survey 2019–21 (NFHS-5), India**

| Property | Value |
|---|---|
| Survey | NFHS-5 |
| Years | 2019–21 |
| Households | 636,699 |
| Women 15–49 | 724,115 |
| Men 15–54 | 101,839 |
| Individual records (women 15–49, men 15–54) | ~1.44 million |
| Adults with valid random glucose | ~1.25 million |
| Districts | 707 |
| States/UTs | 28 states and 8 UTs |
| Biomarkers | Random blood glucose, HbA1c subsample, blood pressure, anthropometry |
| Data source | DHS Program / IIPS / MoHFW |
| Access | Registration and data-use agreement |
| Cost | Free |

### Data access

1. Register with the DHS Program.
2. Request the NFHS-5 dataset.
3. Download the relevant recode files:
   - Women’s recode (IR)
   - Men’s recode (MR)
   - Household recode (HR)
   - Biomarker recode (BR)
   - Codebook and questionnaire
4. Read the files with `pandas.read_stata` or `pyreadstat`.
5. Keep raw data out of Git.
6. Record file names, release dates, and checksums.
7. Follow the DHS data-use terms; share code and aggregate results, not raw data.

### Data links

- DHS Program: https://dhsprogram.com/
- NFHS-5: https://www.nfhsiiips.in/nfhsuser/nfhs5.php
- NFHS-5 final report: https://dhsprogram.com/pubs/pdf/FR375/FR375.pdf
- NFHS-6: https://www.nfhsiiips.in/nfhsuser/nfhs6.php

### NFHS-6 note

NFHS-6 (2023–24) fact sheets were released in 2026. Use NFHS-5 for the primary build because it has public microdata and a documented codebook. Add NFHS-6 as a temporal validation set when microdata becomes available.

### Citation

> International Institute for Population Sciences (IIPS) and ICF. 2021. National Family Health Survey (NFHS-5), 2019–21: India. Mumbai: IIPS.

> Ministry of Health and Family Welfare (MoHFW), Government of India. National Family Health Survey 2019–21.

---

## 8. Target definition

### Biomarker variables

Use the NFHS-5 codebook to map the exact variable names. The key measures are:

- Random blood glucose (mg/dL)
- HbA1c (%) on the dried-blood-spot subsample
- Self-reported diabetes diagnosis
- Diabetes medication use
- Systolic and diastolic blood pressure
- Height, weight, waist, and hip circumference

Exact variable names can differ between recode files; verify them in the distributed NFHS-5 codebook before writing the pipeline.

### Primary target

**Elevated random blood glucose or diagnosed/treated diabetes:**

```text
positive = random_glucose > 140 mg/dL
           OR diagnosed_diabetes == yes
           OR taking_diabetes_medication == yes
```

This matches the NFHS-5 final-report definition of high or very high blood sugar or taking medicine to control blood sugar.

### Secondary target: undiagnosed diabetes

For a screening-focused analysis:

```text
eligible = no prior diabetes diagnosis
           AND no diabetes medication
positive = random_glucose > 140 mg/dL
```

This is the stronger “screening gap” story, but the positive class is smaller.

### Sensitivity target: HbA1c

In the HbA1c subsample:

```text
positive = HbA1c >= 6.5%
```

Use this as a sensitivity analysis, not the primary target, because the subsample is smaller and not available for everyone.

### Exclusions

- Age under 15.
- Missing or invalid random blood glucose.
- Missing target diagnosis or medication status.
- Pregnant women for the primary diabetes target, because pregnancy affects glucose. Flag and exclude; report the excluded count.
- Respondents with inconsistent values, following the NFHS-5 analytic guidance.

### Age range

Women are sampled at 15–49; men at 15–54. For the primary pooled model:

- Restrict to **15–49 for both sexes** to avoid age-range confounding.
- Run a secondary model including men 50–54.
- Report sex-specific metrics in both cases.

### Important wording

The app estimates:

> The probability of elevated random blood glucose or a reported diabetes diagnosis among Indian adults with similar characteristics.

It does **not** estimate:

> The probability that this person will develop diabetes in the future.

It also does **not** diagnose diabetes. Random glucose is a screening indicator.

---

## 9. Data cleaning and leakage rules

### Cleaning

- Recode `don't know`, `refused`, and missing values using the NFHS-5 codebook.
- Convert heights and weights to metric units if needed.
- Compute BMI from height and weight; use the NFHS-5 BMI variable as a cross-check.
- Use measured waist and hip circumference; compute waist-to-hip ratio.
- Use the average of valid BP readings, following NFHS-5 guidance.
- Recode wealth index into quintiles.
- Recode education into the standard categories.
- Recode caste/tribe and religion into the published categories; group small categories only for stable estimates.
- Keep `"Unknown"` as an explicit category where appropriate.
- Do not impute the target.

### Household grouping

NFHS-5 interviews multiple respondents per household. Treat the household as a group:

- Use `HHID` as the group identifier.
- Use `StratifiedGroupKFold` or `GroupShuffleSplit` by `HHID`.
- Never put two respondents from the same household in different folds.
- Report household overlap checks.

This is one of the most important methodological details in the project.

### Leakage rules

Exclude all target-derived and diabetes-management variables:

- Random blood glucose
- HbA1c
- Self-reported diabetes diagnosis
- Diabetes medication use
- Insulin use
- Any diabetes complication or management variable
- Any variable that is only asked of people with diabetes

Exclude design and identifier variables from model features:

- `HHID`
- `CASEID`
- Cluster/PSU identifiers
- Strata
- Sample weights
- District/state identifiers as model features in the primary model

Use design variables for weighted analysis, not as features.

### Access-to-care bias

Healthcare access can influence whether a person is diagnosed. Do not use access variables in the primary model. Report performance by insurance/access indicators as a bias audit.

### Comorbidity and complication leakage

Do not include diabetes complications or diabetes-management variables. If comorbidities such as hypertension are used, treat them as risk factors available at screening time, not as consequences of diabetes. Run an ablation with and without hypertension.

### Caste, religion, and fairness

Caste/tribe and religion are sensitive. Use them for **fairness auditing**, not necessarily as deployed model features.

- Primary deployed model: exclude caste/tribe and religion.
- Fairness sensitivity model: include them and compare performance and gaps.
- Never present caste or religion as biological causes.
- Do not require users to disclose caste or religion in the app.

---

## 10. Feature set

### Core features: measurement model

This is the primary model for ASHA/ANM use, because height, weight, waist, and BP can be measured at a Health & Wellness Centre.

| Feature | Source | Type | Notes |
|---|---|---|---|
| Age | `V012` / `MV012` | Continuous | 15–49 primary |
| Sex | Constructed sex indicator | Binary | Harmonise women’s and men’s recode files |
| Height | Anthropometry | Continuous | cm |
| Weight | Anthropometry | Continuous | kg |
| BMI | Derived | Continuous | kg/m² |
| Waist circumference | Anthropometry | Continuous | cm |
| Waist-to-hip ratio | Derived | Continuous | Waist / hip |
| Systolic BP | BP measurement | Continuous | Average of valid readings |
| Diastolic BP | BP measurement | Continuous | Average of valid readings |
| Hypertension | BP or self-report | Binary | Measured high BP or diagnosis |
| Education | `V106` / `MV106` | Ordered categorical | None, primary, secondary, higher |
| Wealth quintile | `V190` / `MV190` | Ordered categorical | Poorest to richest |
| Rural/urban | `V025` / `MV025` | Binary | Rural vs urban |
| Tobacco use | `V463A`-style variables | Categorical | Never, former, current |
| Alcohol use | `V463B`-style variables | Categorical | Never, occasional, regular |
| Physical activity | `V`/`MV` activity variables | Categorical | Low, moderate, high |
| Diet | NFHS-5 diet variables | Categorical | Frequency of fruits/vegetables, fried food |
| Household size | `HV009` | Count | Optional |
| Cooking fuel | `HV226` | Categorical | Clean vs biomass; optional social determinant |

### Questionnaire-only model

For self-service use, train a second model without measured BP, waist, or hip:

- Age, sex, height, weight, BMI
- Education, wealth, rural/urban
- Tobacco, alcohol, physical activity, diet
- Self-reported hypertension (if available)

This model will be weaker but more accessible. Report both.

### Full model with state

Add state or region as a categorical feature for local deployment. Compare with the state-free model:

- State-free model: more portable, less likely to encode local access patterns.
- State model: better local calibration, but may reflect state-level diagnosis and access differences.

Report both and choose based on calibration by state.

### Do not use

- Random blood glucose
- HbA1c
- Diabetes diagnosis
- Diabetes medication
- Insulin
- Diabetes complications
- Any post-diagnosis management variable
- Sample weights, PSU, strata as features
- Household ID or case ID as features

---

## 11. Survey weights and evaluation

NFHS-5 uses a complex survey design:

- Women’s weight: `V005` (divide by 1,000,000 for analysis)
- Men’s weight: `MV005`
- PSU: `V021` / `MV021`
- Strata: `V022` / `MV022`
- State: `V024` / `MV024`

### Primary modeling choice

Train the primary model **unweighted** so the output is an individual-level probability. Use survey weights for:

- Weighted prevalence
- Weighted evaluation metrics as a sensitivity analysis
- Population-representative descriptive tables

### Sensitivity analysis

- Train with sample weights.
- Compare weighted and unweighted calibration.
- Report whether weighting changes subgroup conclusions.

### Reporting

Report both:

- Unweighted metrics: performance on the sample.
- Weighted metrics: performance for the survey population.

Use design-based methods or bootstrap for confidence intervals. Do not treat the weight as a sample-size multiplier.

### Pooling women and men

Women and men have separate recode files and weights. To pool:

- Harmonise variable names and categories.
- Use sex-specific weights and design variables.
- Include sex as a feature.
- Report sex-specific metrics.
- Check that the pooled prevalence matches the NFHS-5 report.

---

## 12. Modelling

### Baselines

Train and evaluate:

1. Logistic regression with one-hot encoding, imputation, and scaling
2. Random forest
3. LightGBM
4. XGBoost (optional)

The baseline comparison is required. A complex model is only justified if it beats a simple one on the agreed metrics.

### Primary model

**LightGBM** with:

- Native categorical feature handling
- Class weighting or `scale_pos_weight`
- Early stopping on the validation fold
- Fixed random seeds
- Calibrated probabilities

### Accuracy-first selection

The launch model is the highest-accuracy validated configuration among the pre-registered candidates, judged primarily by validation ROC-AUC and PR-AUC and gated by calibration and subgroup reporting. The held-out test set is used once for final reporting, not for selection. Model size, browser inference, and offline latency are not selection criteria. Accuracy-first does not override leakage rules, the sensitive-attribute exclusion, or the test-set discipline.

### Split strategy

- Hold out **20% of households** as a final test set.
- Use **5-fold stratified group cross-validation** on the remaining 80%, grouped by `HHID`.
- Keep the test set untouched until final evaluation.
- Save the split assignment.
- Report household overlap checks.

### Hyperparameter tuning

- Use Optuna or a small manual grid.
- Tune inside the training folds only.
- Do not tune on the test set.
- Log the search space, best parameters, and trial count.

### Multi-seed ensemble

Use **5 folds × 10 seeds = 50 models**.

For each seed:

1. Create a household-grouped 5-fold split with that seed.
2. Train one model per fold.
3. Collect out-of-fold predictions.
4. Train final models on the full training set.
5. Predict the test set with each final model.

Aggregate by **averaging predicted probabilities**, not hard labels.

Report:

- Mean and standard deviation across seeds
- Single-seed baseline vs ensemble
- PR-AUC, ROC-AUC, Brier, and log loss
- Whether the ensemble improves calibration as well as ranking

### Model variants

1. **Measurement model** — age, sex, BMI, waist, BP, socioeconomic, behaviors. Primary.
2. **Questionnaire-only model** — no BP/waist; self-reported height/weight. Secondary.
3. **State model** — adds state/region. Local deployment comparison.
4. **Fairness sensitivity model** — includes caste/tribe and religion for audit only.

Report all four. The measurement model is the default launch model. The state model is a launch candidate only if it materially improves validated discrimination or calibration; any switch must be recorded with its subgroup and portability trade-offs. The questionnaire-only model remains a secondary accessibility option, and the fairness model remains audit-only.

### Calibration

- Calibrate on out-of-fold predictions using isotonic regression or Platt scaling.
- Choose the calibrator by Brier score and log loss on validation data.
- Plot the reliability curve before and after calibration.
- Report calibration slope and intercept.
- Store the calibrator with the model artifact.
- Show weighted and unweighted calibration curves.
- Report calibration by state and wealth quintile.

### Threshold selection

- Choose a threshold based on referral capacity and a target sensitivity.
- For screening, aim for **≥ 80% sensitivity** while maximising specificity.
- Report the threshold, confusion matrix, sensitivity, specificity, PPV, and NPV.
- Provide a configurable threshold for PHC referral capacity.
- Do not use 0.5 by default.
- Show the threshold in the UI and model card.

---

## 13. Evaluation

### Metrics

Report at minimum:

- ROC-AUC
- PR-AUC
- Brier score
- Log loss
- Sensitivity at 80% specificity
- Sensitivity, specificity, PPV, and NPV at the chosen threshold
- F1 at the chosen threshold
- Calibration curve
- Calibration slope and intercept
- Confusion matrix

### Uncertainty

- Use bootstrap confidence intervals for the main metrics.
- Report mean ± standard deviation across seeds.
- Include subgroup sample sizes.
- Note that ordinary bootstrap intervals are approximate for complex survey data.

### Subgroup analysis

Report by:

- Age group
- Sex
- Wealth quintile
- Caste/tribe
- Religion
- Rural/urban
- State/region
- Education

For each subgroup:

- n
- Unweighted and weighted prevalence
- ROC-AUC
- PR-AUC
- Sensitivity at the chosen threshold
- Calibration error

Do not overclaim fairness from small subgroups. Document gaps and limitations.

### Care cascade

A strong secondary analysis:

- Among adults with diabetes, what proportion are aware?
- Among aware, what proportion are treated?
- Among treated, what proportion are controlled?
- How do these differ by state, wealth, caste/tribe, religion, gender, and rural/urban?

This can become a district care-cascade dashboard. It is a strong public-health story and does not require a new model.

### Error analysis

- Inspect false negatives: people with elevated glucose predicted as low risk.
- Inspect false positives: people with normal glucose predicted as high risk.
- Compare feature distributions for errors and non-errors.
- Write a short “what the model misses” section in the evaluation report.

### Label-bias analysis

Because the target includes diagnosed diabetes, the model may learn who gets diagnosed rather than who has elevated glucose. Report:

- Performance by insurance/access status
- Performance by wealth and education
- Whether access variables would change results
- A clear limitation statement

### Evaluation harness

Create an `evals/` directory with:

- A fixed household-grouped test set
- A runner that loads the model artifact and computes metrics
- `results.md` — readable tables and plots
- `results.json` — raw metrics and metadata
- `subgroup_report.md` — fairness tables
- Model version, dataset version, seed list, and date

Commit the results. They are a snapshot, not a permanent claim.

---

## 14. Explainability and fairness

### Explainability

- Use SHAP `TreeExplainer` for the primary tree model.
- Global: summary plot and mean absolute SHAP values.
- Local: top 5 factors for each prediction, with direction and impact.
- In the UI, use plain-language labels, not raw variable names.
- State clearly that explanations are associations, not causes.

Examples:

- “Higher waist circumference is associated with a higher estimated risk in this model.”
- “Age is associated with a higher estimated risk.”
- “The model uses wealth quintile; this reflects structural patterns, not a biological cause.”

### Fairness

- Compare subgroup metrics.
- Check whether the chosen threshold produces materially different sensitivity across groups.
- Document subgroups with too few samples for reliable conclusions.
- Run a fairness sensitivity model with caste/tribe and religion and compare.
- Do not claim fairness because a protected attribute was removed.
- Do not present caste, tribe, or religion as biological causes.
- Consider whether the app should ask for caste/religion at all. Default: no.

### Model card

Write a model card covering:

- Intended use
- Out-of-scope use
- Training data and population
- Target definition
- Features
- Metrics and calibration
- Subgroup performance
- Limitations, including random glucose, self-report, and diagnosis bias
- Ethical considerations
- Citation and data-use terms

---

## 15. Product experience

### Name

**India Diabetes Risk Screener**

### Tagline

> A short, India-specific risk assessment for elevated blood glucose, with referral guidance.

### Pages

#### 1. Screener

- Short introduction and disclaimer.
- Form grouped into sections:
  - About you
  - Body measurements
  - Blood pressure
  - Health behaviors
  - Household and social context
- Progress indicator.
- Low-literacy labels and icons.
- “Use an example profile” button for demos.
- Sensitive fields marked optional.
- Submit button with loading state.
- Server-backed prediction; show a clear retry state if the API is unreachable. Do not show a cached or fabricated risk score.

#### 2. Result

- Risk percentage.
- Risk band: low / moderate / high.
- Threshold used.
- Top 5 contributing factors.
- “What this means” panel.
- “What this does not mean” panel.
- Referral guidance:
  - High risk: “Please get a blood glucose test at the nearest Health & Wellness Centre.”
  - Moderate risk: “Consider a blood glucose test, especially if you have symptoms or family history.”
  - Low risk: “No action needed today. Maintain healthy habits and get tested if you develop symptoms.”
- Model version.
- Disclaimer.

#### 3. Model

- Metrics table.
- Calibration plot.
- Subgroup table.
- Dataset citation.
- Training date and model version.
- Link to evaluation report and model card.

#### 4. About

- What the project is.
- What it is not.
- Privacy statement.
- Data source and limitations.
- NPCDCS and Health & Wellness Centre alignment.
- Source code and dataset links.

### UX rules

- Do not say “you have diabetes.”
- Use “estimated probability of elevated blood glucose or diagnosed diabetes.”
- Make the disclaimer visible, not buried.
- Keep the form usable on a low-end Android phone over a normal mobile connection.
- Use accessible labels, large tap targets, and sufficient contrast.
- Do not require caste or religion.
- Do not ask for information the model does not use.
- Explain why sensitive fields are asked and allow “prefer not to answer.”
- Provide a print/share referral note without storing PII.

### Example result copy

> **Estimated probability: 22%**  
> This means that, in the NFHS-5 survey data, about 22 in 100 adults with a similar profile had elevated random blood glucose or a reported diabetes diagnosis.  
> This is not a diagnosis. Please get a blood glucose test at your nearest Health & Wellness Centre.  
> The model does not predict whether you will develop diabetes in the future.

---

## 16. API specification

### Base URL

- Local: `http://localhost:8000`
- Production: TBD

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and model-loaded status |
| `GET` | `/api/model-info` | Model version, metrics, threshold, dataset citation |
| `POST` | `/api/predict` | Single prediction with explanations |
| `POST` | `/api/batch` | Optional batch prediction for research |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### `POST /api/predict`

Request:

```json
{
  "age": 45,
  "sex": "female",
  "height_cm": 158,
  "weight_kg": 68,
  "waist_cm": 92,
  "hip_cm": 100,
  "systolic_bp": 130,
  "diastolic_bp": 85,
  "education": "secondary",
  "wealth_quintile": "middle",
  "rural": true,
  "state": "Bihar",
  "tobacco": "never",
  "alcohol": "never",
  "physical_activity": "low",
  "diet": "mixed"
}
```

Response:

```json
{
  "risk": 0.22,
  "risk_band": "high",
  "target": "elevated_random_blood_glucose_or_diagnosed_diabetes",
  "threshold": 0.18,
  "top_factors": [
    {
      "feature": "Waist circumference",
      "value": 92,
      "impact": 0.061,
      "direction": "increases"
    },
    {
      "feature": "Age",
      "value": 45,
      "impact": 0.034,
      "direction": "increases"
    }
  ],
  "referral": "Please get a blood glucose test at the nearest Health & Wellness Centre.",
  "model_version": "nfhs5-india-screener-v1",
  "disclaimer": "Research demo only. Not a diagnosis and not medical advice."
}
```

### `GET /api/model-info`

Return:

- Model version
- Target definition
- Dataset year and citation
- Number of records after exclusions
- Metrics with confidence intervals
- Calibration summary
- Threshold and risk bands
- Subgroup table
- Limitations

### Validation

- Use Pydantic enums and constrained numeric types.
- Reject unknown fields.
- Enforce request body size limits.
- Return `422` for validation errors and `413` for oversized payloads.
- Return a stable error shape:

```json
{
  "error": "validation_error",
  "detail": "..."
}
```

### Performance

- Online single prediction p95 < 300 ms.
- Batch of 1,000 rows < 5 s.
- Model loaded once at startup.
- Health check reports `model_loaded: true`.

### Security

- No PII storage by default.
- Log request metadata only, not feature values.
- Restrict CORS to the deployed frontend origin.
- Rate limit the public demo.
- Do not expose model internals or training data.
- Do not log free-text fields.

---

## 17. Architecture

```text
Web app (React/Vite, English)
  - responsive UI
  - server API for inference and SHAP
        |
        v
FastAPI service
  - request validation (Pydantic)
  - preprocessing pipeline
  - LightGBM ensemble
  - calibration
  - SHAP explainer
        |
        v
Model artifacts
  - model.joblib
  - preprocessor.joblib
  - calibrator.joblib
  - feature_schema.json
  - metrics.json
  - model_card.md
```

### Deployment choice

**Responsive web app + FastAPI API** is preferred:

- The web app is hosted on a static host or CDN.
- FastAPI serves the online model and model-info endpoints.
- For a single-service demo, FastAPI can serve the built frontend as static files.
- Inference is server-side for the launch app; no browser model is required.

### Runtime

- Python 3.14 (the only interpreter the modeling stack is tested on; `requires-python` matches)
- FastAPI
- Uvicorn
- LightGBM
- scikit-learn
- pandas / numpy
- SHAP
- joblib
- pyreadstat
- Pydantic
- React or vanilla JS

### Model loading

- Load the model, calibrator, feature schema, and SHAP explainer during FastAPI startup.
- Fail fast if the feature schema and model do not match.
- Initialise the SHAP explainer with a small background sample, not the full training set.
- Return the model version with every prediction.

### Inference path

- The browser sends the validated feature payload to `POST /api/predict`.
- FastAPI returns the calibrated probability, threshold flag, risk band, top factors, referral copy, and model version.
- The frontend does not run a second model implementation; UI decisions and API decisions come from the same response.
- If the API is unavailable, show a retry state. Do not invent or cache a risk score.

### Post-launch features

These are explicitly out of launch scope and do not block acceptance:

- **Bilingual UI** — Hindi first, then additional languages, with professional translation and usability testing.
- **Offline-first PWA** — cached form, optional on-device model, and online/offline parity tests.
- **Model distillation/ONNX export** — only if offline inference is later prioritised.
- **Docker packaging** — Dockerfile, docker-compose, containerized deployment, and image-tag rollback.

---

## 18. Model artifacts and repository layout

```text
india-diabetes-risk-screener/
  README.md
  SPEC.md
  pyproject.toml
  requirements.txt
  data/
    README.md
    download.md
    raw/                  # gitignored
    processed/            # gitignored
  src/
    screener/
      data.py
      features.py
      train.py
      evaluate.py
      calibrate.py
      explain.py
      cascade.py
      api/
        main.py
        schemas.py
        service.py
        settings.py
  frontend/
    src/
    public/
    package.json
  tests/
    test_features.py
    test_target.py
    test_splits.py
    test_api.py
    test_model_smoke.py
  evals/
    run_eval.py
    results.md
    results.json
    subgroup_report.md
  artifacts/
    model.joblib
    preprocessor.joblib
    calibrator.joblib
    feature_schema.json
    metrics.json
    calibration.json
    model_card.md
  .github/
    workflows/
      ci.yml
```

### Data handling

- Do not commit raw NFHS-5 data.
- Provide `data/download.md` with registration and download steps.
- Record file names, release dates, and checksums.
- Keep processed data out of Git.
- Share code and aggregate results only.

### Artifact handling

- Commit metrics, model card, and evaluation results.
- Commit the model only if the DHS terms permit it.
- Otherwise publish it as a release asset or use Git LFS.

---

## 19. Testing

### Unit tests

- Target recoding
- Missing-code recoding
- BMI and waist-to-hip calculations
- BP averaging
- Age range filtering
- Risk band calculation
- Threshold logic

### Data tests

- Target distribution is in the expected range
- No leakage columns in the feature set
- No design variables in the feature set
- Household-grouped splits have zero overlap
- Expected number of columns after selection
- Weighted prevalence matches the NFHS-5 report within tolerance

### API tests

- `/health`
- `/api/model-info`
- `/api/predict` with valid input
- `/api/predict` with invalid input → 422
- Oversized body → 413
- Unknown fields → 422
- Error shape is stable

### Model smoke tests

- Fixed input produces a probability in `[0, 1]`
- Golden predictions stay within a small tolerance
- Model version is present in the response
- Top factors are returned for every prediction

### Fairness snapshot

- Subgroup metrics are computed without errors
- Small subgroups are flagged
- Fairness report is committed

### Web app tests

- Screener validates required fields and submits a prediction.
- Result page renders risk, band, threshold, top factors, referral, and disclaimer.
- Model page renders metrics, calibration, subgroup table, and model version.
- API-unavailable state shows a retry, not a fabricated score.

### CI

GitHub Actions should run:

- Lint (ruff)
- Tests (pytest)
- Frontend build and frontend tests
- Optional: training smoke test on a 5,000-row sample

---

## 20. Deployment and operations

### Deployment

- Build the web frontend and the FastAPI service.
- Deploy the frontend to a static host or CDN, or serve it from FastAPI.
- Deploy the API to Render, Fly.io, Railway, or a low-cost cloud region.
- Expose `/health` and `/docs`.
- Set environment variables:
  - `MODEL_PATH`
  - `LOG_LEVEL`
  - `CORS_ORIGINS`
  - `RATE_LIMIT_PER_MINUTE`
  - `DISCLAIMER_VERSION`

### Observability

- Structured JSON logs with request ID, latency, status, and model version.
- Do not log feature values or free-text input.
- Track prediction distribution and request volume.
- Optional: monitor feature drift against the training reference.
- Health check should fail if the model is not loaded.
- The launch app requires connectivity for inference; surface API failures clearly.

### Rollback

- Keep the previous model artifact and deployment release.
- Document how to roll back.
- Record the model version in every response.

---

## 21. Security, privacy, and ethics

- NFHS-5 is public-use, de-identified data, but it still contains sensitive attributes.
- Do not attempt re-identification.
- Do not store user-submitted questionnaire answers.
- Do not log PII.
- Do not require caste or religion in the app.
- Do not use the model for insurance, employment, marriage, or care-allocation decisions.
- Do not claim clinical validity.
- Display a clear research-demo disclaimer.
- Document dataset limitations and subgroup gaps.
- Avoid causal language in explanations.
- Rate-limit the public demo.
- Keep dependencies patched.
- Follow the DHS data-use agreement.

### Disclaimer

> This is a research and portfolio demo. It estimates the probability of elevated random blood glucose or a reported diabetes diagnosis based on NFHS-5 survey data. It is not a diagnostic test, does not predict future diabetes, and must not be used for medical decisions. Please consult a qualified health professional and get a blood glucose test if you have concerns.

---

## 22. Documentation

Required documents:

- `README.md` — setup, data access, training, evaluation, API, deployment
- `SPEC.md` — this document
- `model_card.md` — intended use, limitations, metrics, subgroups
- `evals/results.md` — committed evaluation report
- `evals/subgroup_report.md` — fairness tables
- `data/README.md` — dataset citation, terms, download instructions
- `docs/api.md` — optional API examples

The README should include:

- One-paragraph project summary
- Screenshot or GIF
- Dataset citation
- Model metrics table
- API example
- Local setup commands
- Deployment link
- Limitations and disclaimer

---

## 23. Milestones

| Phase | Deliverable | Estimate |
|---|---|---|
| M0 | Spec and repo scaffold | 0.5 day |
| M1 | DHS access, data download, recoding, target | 1–2 days |
| M2 | Baselines and evaluation harness | 1 day |
| M3 | Feature engineering and model variants | 1–2 days |
| M4 | Multi-seed ensemble and calibration | 1–2 days |
| M5 | SHAP, fairness, care cascade | 1–2 days |
| M6 | FastAPI service and tests | 1–2 days |
| M7 | Responsive web app, English UI, full pages | 2–3 days |
| M8 | Deploy, model card | 1–2 days |
| M9 | Resume bullets and README polish | 0.5 day |
| Post-launch | Hindi/English bilingual UI, offline-first PWA, and Docker packaging | 3–5 days, deferred |

Total launch scope: roughly **2–3 weeks part-time**.

---

## 24. Acceptance criteria

### Data

- [ ] NFHS-5 downloaded and checksum recorded
- [ ] Target definition documented
- [ ] Pregnancy and invalid-glucose exclusions documented
- [ ] Leakage columns excluded
- [ ] Household-grouped split saved
- [ ] Weighted prevalence matches the report within tolerance

### Model

- [ ] Logistic-regression baseline reported
- [ ] LightGBM ensemble trained with 5 folds × 10 seeds
- [ ] PR-AUC, ROC-AUC, Brier, and log loss reported
- [ ] Calibration curve and Brier score reported
- [ ] Threshold selected and justified
- [ ] Subgroup metrics reported
- [ ] Measurement, questionnaire-only, and state models compared
- [ ] Highest-accuracy validated configuration selected and documented
- [ ] SHAP global and local explanations working
- [ ] Evaluation report committed

### API

- [ ] `/health`, `/api/model-info`, `/api/predict` implemented
- [ ] Pydantic schemas and validation
- [ ] Error shape tested
- [ ] p95 latency < 300 ms
- [ ] Test suite passes in CI

### App

- [ ] English UI
- [ ] Screener, Result, Model, and About pages implemented
- [ ] Low-literacy labels and icons
- [ ] Risk score, risk band, and top factors displayed
- [ ] Referral guidance displayed
- [ ] “What this does not mean” panel
- [ ] Mobile responsive
- [ ] API-error and retry state
- [ ] Model page with metrics and limitations
- [ ] About page with privacy, limitations, and source links

### Deployment

- [ ] Web frontend builds
- [ ] Public URL live
- [ ] `/docs` and `/health` accessible
- [ ] Model version visible in responses

### Portfolio

- [ ] README with screenshot and metrics
- [ ] Model card
- [ ] Care-cascade report
- [ ] Three resume bullets with real numbers

---

## 25. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Random glucose is not fasting glucose | Misclassification | Call it a screening indicator; report HbA1c sensitivity; do not diagnose |
| Cross-sectional data | Cannot predict future diabetes | State the limitation; do not use “will develop” language |
| Diagnosis bias | Model may learn access to care | Exclude access variables; audit by insurance/wealth; document |
| Household clustering | Leakage and inflated metrics | Group splits by household ID |
| Survey design complexity | Incorrect population estimates | Use weights, strata, and PSUs; report weighted and unweighted metrics |
| Caste/religion sensitivity | Stigma and discrimination | Exclude from deployed model; use only for fairness audit |
| Small subgroups | Unreliable fairness claims | Report n; suppress small cells; avoid overclaiming |
| Missing data | Bias and instability | Document missingness; use native tree handling; add missing indicators |
| Network dependency | App unavailable without connectivity | Clear retry/error state; defer offline-first PWA to post-launch |
| Accuracy-first trade-off | Chasing accuracy can hurt calibration, fairness, or delay launch | Pre-register candidates; select on validation; gate on calibration and subgroup reporting; keep the test set untouched |
| Scope creep | Project never ships | Ship the full measurement model + web app first; defer bilingual UI and offline PWA to post-launch |
| Clinical misinterpretation | Harm | Explicit disclaimer; referral guidance; no diagnostic language |
| Data-use terms | Legal risk | Follow DHS agreement; no raw data sharing |
| NFHS-6 not available | Outdated data | Use NFHS-5; add NFHS-6 validation when released |

---

## 26. Open questions

- Primary target: all diabetes vs undiagnosed diabetes only?
- Use random glucose for the full sample or HbA1c for the subsample?
- Restrict to ages 15–49 for both sexes, or include men 50–54?
- Should the state model be promoted to launch if it materially improves validated accuracy, or kept as an audit variant?
- Exclude caste/religion from the deployed model? (Recommended: yes.)
- Should the app ask for wealth/education, or infer a proxy from a few asset questions?
- Which deployment platform: Render, Fly.io, Railway, or a static host + API?
- Is an LLM-generated plain-language explanation in scope?
- Which deferred feature comes first after launch: Hindi/English UI or offline-first PWA?
- Is this project replacing the BRFSS/UCI projects or sitting alongside them?

---

## 27. Resume bullet templates

Use real numbers after the evaluation is complete.

> **India Diabetes Risk Screener** — Python, LightGBM, FastAPI, SHAP, React
> - Built an India-specific diabetes risk screener on **1.25 million NFHS-5 adults with valid random glucose**, using household-grouped cross-validation and a **10-seed LightGBM ensemble**; improved PR-AUC from **0.XX to 0.YY** over logistic regression.
> - Calibrated risk scores for a **≥80% sensitivity** screening threshold (Brier **0.XX**) and analysed fairness across wealth, caste/tribe, religion, gender, and rural/urban groups.
> - Deployed a full-featured responsive web app and FastAPI service for ASHA/ANM workflows with **40+ tests**, CI, and a model card documenting random-glucose and cross-sectional limitations.

Alternative, shorter:

> - Built and deployed an India-specific diabetes risk screener on **1.25M NFHS-5 adults with valid random glucose**, using a **10-seed LightGBM ensemble**, probability calibration, and SHAP explanations.
> - Designed a responsive, accuracy-first web app for low-resource primary care; reported subgroup fairness and a district care-cascade analysis.

---

## 28. Appendix — variable reference

Exact NFHS-5 variable names vary by recode file. Verify them in the distributed codebook before implementation.

### Identifiers and design

| Concept | Typical NFHS/DHS variable | Use |
|---|---|---|
| Household ID | `HHID` | Grouping |
| Case ID | `CASEID` | Identifier; do not use as feature |
| Cluster/PSU | `V021` / `MV021` | Survey design |
| Strata | `V022` / `MV022` | Survey design |
| Women’s weight | `V005` | Weighted analysis |
| Men’s weight | `MV005` | Weighted analysis |
| State | `V024` / `MV024` | Subgroup/calibration |
| District | `SDIST` (if available) | Subgroup/calibration |

### Demographics and socioeconomic status

| Concept | Typical variable | Use |
|---|---|---|
| Age | `V012` / `MV012` | Feature |
| Sex | Sex variable | Feature |
| Education | `V106` / `MV106` | Feature |
| Wealth quintile | `V190` / `MV190` | Feature/fairness |
| Rural/urban | `V025` / `MV025` | Feature/fairness |
| Religion | `V130` / `MV130` | Fairness audit |
| Caste/tribe | `V131` / `MV131` | Fairness audit |
| Household size | `HV009` | Optional feature |
| Cooking fuel | `HV226` | Optional social determinant |

### Anthropometry and biomarkers

| Concept | Typical variable | Use |
|---|---|---|
| Height | Anthropometry variable | Feature |
| Weight | Anthropometry variable | Feature |
| BMI | `HA40` (women), `HB40` (men) | Feature/cross-check |
| Waist | Anthropometry variable | Feature |
| Hip | Anthropometry variable | Feature |
| Systolic BP | BP variable | Feature |
| Diastolic BP | BP variable | Feature |
| Random blood glucose | Glucose module variable | Target |
| HbA1c | DBS subsample variable | Sensitivity target |
| Diabetes diagnosis | Diabetes module variable | Target/exclusion |
| Diabetes medication | Diabetes module variable | Target/exclusion |

### Behaviors

| Concept | Typical variable | Use |
|---|---|---|
| Tobacco | Tobacco module variable | Feature |
| Alcohol | Alcohol module variable | Feature |
| Physical activity | Activity module variable | Feature |
| Diet | Diet module variables | Feature |

### Data links

- DHS Program: https://dhsprogram.com/
- NFHS-5: https://www.nfhsiiips.in/nfhsuser/nfhs5.php
- NFHS-5 final report: https://dhsprogram.com/pubs/pdf/FR375/FR375.pdf
- NFHS-6: https://www.nfhsiiips.in/nfhsuser/nfhs6.php

---

## 29. Definition of done

The project is done when:

1. The model is trained, calibrated, and evaluated on a household-grouped held-out test set.
2. The target definition and exclusions are documented.
3. Measurement, questionnaire-only, and state models are compared, and the highest-accuracy validated configuration is selected.
4. The evaluation report, subgroup report, and model card are committed.
5. The FastAPI service and responsive web app are deployed.
6. `/docs`, `/health`, and `/api/predict` work.
7. The test suite passes in CI.
8. The README contains a screenshot, metrics, setup instructions, and limitations.
9. The resume has three bullets with real metrics and a link to the live demo.
10. Bilingual UI, offline-first PWA, and Docker packaging are recorded as post-launch features, not launch blockers.
