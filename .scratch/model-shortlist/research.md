# Model shortlist research

Focused literature review for the NFHS-5 diabetes risk screener. Scope: model families that have been evaluated for diabetes risk prediction in India/South Asia, international diabetes risk models, tabular ML benchmarks, and calibration/fairness requirements for a clinical screening tool. Citations are numbered and grouped at the end. Where a claim could not be verified from a primary source, it is marked **unverified** or omitted.

## Summary

- The credible India/South Asia performance ceiling for questionnaire-and-measurement diabetes risk models is roughly **AUC 0.68–0.71**, not 0.95+. IDRS reaches 0.698 in its derivation cohort [1] and 0.693–0.707 in a 113,043-person national validation [2]; an NFHS-5 logistic risk score reports a pseudo-R² of 0.077 [3]; an NFHS-5 tree-model study reports RF accuracy of 64.4% and an LR AUC of 0.678 [4]; a 2022 Bangladesh DHS ML study reports AUC 0.699 [7].
- The project's published primary model (ROC-AUC 0.710, PR-AUC 0.204, Brier 0.072 at 8.33% prevalence) is therefore **already in the credible range**. The shortlist should be designed to detect small, real gains and to protect calibration, not to chase a step change.
- India-specific NFHS/IDHS machine-learning papers that report 0.95–0.99 accuracy or AUC are not credible for this task: they typically use random splits, SMOTE, and/or include glucose or other target-defining variables as predictors [5,6]. They should not set expectations.
- International diabetes risk models are overwhelmingly **regression models** (logistic or Cox), not tree ensembles. FINDRISC, ARIC, Framingham, QDScore, and the NHANES calculator all use age, adiposity, blood pressure/hypertension, family history, activity, and diet [11–15]. No international guideline recommends a particular ML family [18,19].
- **Measurement-based models outperform questionnaire-only models**: external validation of 25 models found C-statistics of 0.74–0.84 for basic (non-laboratory) models and 0.81–0.93 for models adding biomarkers [16]. Adding fasting glucose moved ARIC from 0.71 to 0.78 [12] and Framingham from 0.72 to 0.85 [13].
- NFHS-5 **lacks two core IDRS items** — family history of diabetes and physical activity — so this project cannot replicate IDRS, and an IDRS comparison is not like-for-like. The deployed feature set is a different, broader questionnaire-plus-anthropometry model.
- On tabular data, gradient-boosted trees remain the strongest default, especially at medium sample sizes [20,21,24]. Neural nets can match them at large n but rarely dominate, and differences are often negligible after tuning [22,23]. Clinical-prediction reviews find no consistent ML advantage over logistic regression once bias is accounted for [25].
- **Recommendation: six scored candidates plus a separate prevalence-only reference baseline** — logistic regression, logistic regression with splines, random forest, LightGBM, XGBoost, and CatBoost. A neural net belongs in an **exploration bucket**, not the scored shortlist, with a pre-specified promotion rule.
- Calibration should be judged by **slope, intercept, calibration curve, and Brier score**, not by AUC or ECE alone [32,33]. Avoid SMOTE; resampling and imbalance corrections worsen calibration without improving AUC [40]. Post-hoc isotonic/Platt calibration on out-of-fold predictions is the right correction [39].
- Fairness reporting needs **subgroup calibration and equalized-odds metrics**, not just subgroup AUC; dropping protected attributes does not remove proxy bias [41–44]. Use TRIPOD+AI, PROBAST, WHO ethics guidance, and the DHS data-use terms [36,38,45,46].

## 1. India / South Asia evidence

### What has actually been evaluated

The Indian literature has two strands. The first is the **Indian Diabetes Risk Score (IDRS)**, a deliberately simple questionnaire score derived in Chennai and validated nationally. The second is a small, recent set of **NFHS/IDHS machine-learning papers**, most of which are methodologically weaker than their reported metrics suggest. South Asian DHS-based ML studies are a useful external comparison because they use similar survey data and a similar target definition.

| Study | Population / data | n | Outcome | Models | Performance | Main limitation |
|---|---|---:|---|---|---|---|
| Mohan et al. 2005 [1] | CURES, Chennai; adults | 2,350 | Undiagnosed diabetes by 2-h OGTT | IDRS (age, waist, family history, physical activity), logistic-derived score | AUC 0.698 (95% CI 0.663–0.733); at score ≥60 sensitivity 72.5%, specificity 60.1%, PPV 17.0%, NPV 95.1% | Single city; 2005; internal validation only; no external cohort |
| Deepa et al. 2023 (INDIAB-15) [2] | ICMR-INDIAB, 30 states/UTs; urban and rural | 113,043 | Undiagnosed T2D by OGTT | MDRF-IDRS applied, not re-derived | AUC 0.697 urban, 0.694 rural, 0.693 male, 0.707 female | Evaluates IDRS only; no new model; IDRS items limit feature set |
| Karri et al. 2024 [3] | NFHS-5; women 15–49, men 15–54 | 1,254,547 | Random BG ≥140 mg/dL, or diagnosed diabetes, or glucose-lowering medication | Logistic regression risk score with 6 predictors (age, sex, urban, alcohol, hypertension, waist) | Prevalence 9.3%; pseudo-R² 0.077; no AUC or calibration reported | No discrimination/calibration; single random glucose; no cluster-grouped validation |
| Barman et al. 2025 [4] | NFHS-5; adults ≥30 | 1,087,006 | Diabetes by medication or elevated glucose | Decision tree, random forest, XGBoost; logistic regression comparator | RF best; best accuracy 64.43%; LR AUC 0.678, described as similar to RF | 20.3% prevalence; SMOTE; no household/cluster grouping; target definition differs from ours; calls NFHS glucose “fasting” although NFHS-5 measures random capillary glucose |
| Athwani et al. 2026 [5] | NFHS-5; adults | 825,954 | Diabetes (2% prevalence, apparently self-report) | Random forest with SMOTE and random-search tuning | ROC-AUC 0.811; accuracy 73.48%; recall 74.40%; log loss 0.542 | 70/30 random split; SMOTE; glucose listed as a top feature although glucose defines the target — likely leakage; no calibration |
| Thotad et al. 2023 [6] | NFHS-4 / IDHS 2016 | Not reported in abstract | Diabetes | Logistic regression, naive Bayes, LDA, SVM, kNN, decision tree, XGBoost, RF, KECA variants | RF 99.84% accuracy; AUC 99% | Implausible for a noisy survey target; almost certainly leakage/overlap; no external validation |
| Kibria et al. 2026 [7] | Bangladesh DHS 2022; adults | 13,847 | FPG ≥7.0 mmol/L or glucose-lowering drugs | ANN, RF, AdaBoost, gradient boosting, XGBoost, SVM | Diabetes prevalence 16.3%; best AUC 0.699 (AdaBoost); best accuracy 0.836 (SVM); low specificity | Single survey; no external validation; no calibration; South Asia but not India |
| Islam et al. 2020 [8] | Bangladesh DHS 2011 | 1,569 (127 events) | Diabetes | SVM, RF, LDA, logistic regression, kNN, bagged CART | Bagged CART accuracy 94.3% but AUC 0.600 | Tiny event count; accuracy is misleading under imbalance; no calibration |

### What this means

- **The realistic range is 0.68–0.71 AUC.** The two largest, most credible India-specific evaluations — IDRS in INDIAB-15 [2] and the NFHS-5 logistic score [3] — are consistent with the project's 0.710. The NFHS-5 RF study [4] lands in the same region once its 20.3% prevalence and SMOTE are accounted for.
- **The high-AUC papers are not a reason to expect more.** Athwani's 0.811 [5] and Thotad's 0.99 [6] are best read as warnings about leakage, random splitting, and SMOTE, not as achievable performance. This is consistent with the broader finding that ML papers at high risk of bias report much larger effects [25].
- **The India-specific ML literature does not isolate model-family effects.** Almost every study compares families on a single split with different preprocessing, so a higher RF or XGBoost number is not evidence that the family is better. The project's shared folds and fixed preprocessing are a stronger design than anything in this table.
- **Undiagnosed burden is large and socially patterned.** In NFHS-5, 24.8% of diabetes among 15–49-year-olds was undiagnosed, and undiagnosed prevalence was higher in men and in lower-wealth groups [9]. That makes the target a mix of true glucose elevation and access-dependent diagnosis, which matters for fairness interpretation.
- **IDRS is not directly comparable.** NFHS-5 does not contain family history of diabetes or a physical-activity measure in the deployed feature set, two of the four IDRS components [1,10]. The project's model is therefore a different instrument, and the IDRS AUC is a benchmark for ambition, not a validation target.

## 2. International diabetes risk models

### Guideline anchors

- **ADA Standards of Care 2025** recommends screening all adults from age 35, and earlier for people with overweight/obesity, family history, hypertension, dyslipidaemia, physical inactivity, or other risk conditions [18]. It does not endorse a specific risk score or model family.
- **ICMR 2018** recommends screening all adults over 30, and earlier in the presence of family history, BMI ≥23 kg/m², increased waist circumference (>90 cm men, >80 cm women), hypertension, or dyslipidaemia [19]. This is directly relevant to the Indian feature set: age, BMI, waist, and blood pressure are the intended triage variables.

### Models

| Model | Setting / data | n | Predictors | Outcome | Performance | Questionnaire vs measurement | Main limitation |
|---|---|---:|---|---|---|---|---|
| FINDRISC [11] | Finland; 35–64; 10-year follow-up | 4,435; 182 incident cases | Age, BMI, waist, antihypertensive treatment, history of high glucose, physical activity, diet | Drug-treated T2D | At score ≥9: sensitivity 0.78/0.81 and specificity 0.77/0.76 in two cohorts; PPV 0.13/0.05 | Questionnaire plus simple measures | Finnish; drug-treated outcome only; no OGTT |
| ARIC [12] | US; 45–64; 9-year follow-up | 7,915; 1,292 incident cases | Clinical model: waist, height, hypertension, BP, family history, ethnicity, age; plus FPG and lipids | Incident T2D | AUC 0.71 clinical; 0.78 with FPG; 0.80 with FPG + triglycerides/HDL | Measurement-based | US middle-aged cohort; fasting glucose adds most of the gain |
| Framingham Offspring [13] | US; middle-aged; 7-year follow-up | 160 incident cases | Personal: age, sex, parental history, BMI; simple clinical adds hypertension, low HDL, high triglycerides, impaired fasting glucose | Incident T2D | AROC 0.72 personal; 0.85 simple clinical; complex models 0.850–0.854 | Measurement-based | White cohort; OGTT at baseline; small case count |
| QDScore [14] | England/Wales; 25–79; 10-year follow-up | 2,540,753 derivation; 1,232,832 validation; 78,081 and 37,535 incident cases | Age, sex, ethnicity, BMI, smoking, family history, deprivation, treated hypertension, CVD, corticosteroids | Incident diagnosed T2D | Large prospective derivation and validation; ethnicity strongly predictive (e.g., Bangladeshi women HR 4.07) | Questionnaire / routine EHR | UK primary-care records; diagnosis-based, not glucose-defined; no AUC in abstract |
| NHANES Diabetes Risk Calculator [15] | US NHANES III | Not stated in abstract | Age, waist, gestational diabetes, height, race/ethnicity, hypertension, family history, exercise | Undiagnosed diabetes or prediabetes | AUC 0.85 for undiagnosed diabetes; 0.75 for prediabetes or undiagnosed diabetes; sensitivity 88%, specificity 75% | Questionnaire plus simple measures | US; cross-sectional; self-report/OGTT mixture; older data |
| Abbasi et al. 2012 [16] | EPIC-NL external validation; 20–70 | 38,379; 924 cases | 25 published models, basic and extended | Incident T2D | Basic models C 0.74–0.84; extended models C 0.81–0.93; many overestimated risk | Both | Dutch cohort; incidence, not prevalence; calibration improved after recalibration |
| Noble et al. 2011 [17] | Systematic review | 43 papers; 145 models; 6.88 million participants | Many | Incident T2D | Heterogeneity precluded meta-analysis; some models robust and externally validated, many not; genetic markers added nothing over clinical/sociodemographic factors | Both | Narrative synthesis; heterogeneous definitions and validation quality |

### Questionnaire-only versus measurement-based

The international evidence gives a clear hierarchy:

1. **Questionnaire-only / basic models** (FINDRISC, QDScore, the personal Framingham model) reach roughly **C 0.72–0.84** [11,13,14,16].
2. **Basic models plus simple measurements** (waist, height, BP) reach the upper part of that range; the NHANES calculator reached 0.85 for undiagnosed diabetes [15].
3. **Models adding fasting glucose or other biomarkers** reach **C 0.81–0.93** [12,13,16]. The gain from glucose is large and consistent: ARIC 0.71 → 0.78 [12]; Framingham 0.72 → 0.85 [13].

This project is a **pre-test case-finding model**: it predicts a glucose-defined/diagnosed outcome from non-glucose features. That makes it closest to IDRS and the NHANES calculator, not to ARIC/Framingham. The international literature therefore predicts a ceiling near 0.70–0.80, with the lower end more likely in a heterogeneous national survey with a single random glucose.

Two further implications:

- **Feature choice matters more than family.** Age, adiposity (BMI/waist), blood pressure/hypertension, family history, activity, and diet recur across all models [11–15]. Where family history and activity are missing, the model loses two of the strongest questionnaire items.
- **Regression is the international default.** FINDRISC, ARIC, Framingham, QDScore, and the NHANES calculator are all regression-based [11–15]. There is no international precedent for choosing a tree ensemble over a well-specified regression model for this task; the burden of proof is on the more complex family.

## 3. Tabular ML evidence: trees vs neural nets

### Synthesis

- **Gradient-boosted trees are the strongest default on medium-sized tabular data.** Grinsztajn et al. benchmarked 45 datasets at roughly 10,000 samples and found tree-based models remained state of the art, even before accounting for their speed; they attribute this to trees' robustness to uninformative features, invariance to monotone transformations, and ability to learn irregular functions [20]. Shwartz-Ziv and Armon found XGBoost outperformed the deep models they tested across datasets, including the datasets used to propose those deep models, and required less tuning; a deep-model + XGBoost ensemble did better than XGBoost alone [21]. Borisov et al. compared 11 deep approaches with traditional methods on five real tabular datasets and concluded that gradient-boosted tree ensembles still mostly outperform deep learning [24].
- **Neural nets can match, but not reliably beat, GBDT.** Gorishniy et al. found that a simple ResNet-like architecture and a tabular Transformer were competitive with specialized deep architectures, but no architecture dominated across tasks [22]. McElfresh et al. compared 19 algorithms across 176 datasets and concluded that the “NN vs GBDT” debate is overemphasized: for many datasets the difference is negligible, or light GBDT tuning matters more than the family choice [23]. They also note that TabPFN, a tabular foundation model, was limited to training sets of about 3,000 rows — not applicable to 775k rows [23].
- **In clinical prediction, ML does not reliably beat logistic regression.** Christodoulou et al. reviewed 71 studies and 282 comparisons; among comparisons at low risk of bias, the difference in logit(AUC) between LR and ML was 0.00 (95% CI −0.18 to 0.18). The apparent ML advantage was concentrated in studies at high risk of bias (+0.34) [25]. This is the single most important caution for a shortlist that might otherwise assume “more complex is better.”
- **Diabetes-specific ML reviews report good pooled performance but weak validation and calibration.** De Silva et al. pooled 23 studies and 40 models and found a c-index of 0.812, but no study externally validated its model and only 7 of 23 assessed calibration [26]. Li et al. reviewed 65 studies and 97 models; logistic regression accounted for 97.9% of models, ML showed only marginal AUC gains (with implausible maxima up to 0.998), and more than 80% of models were at high risk of bias [27].
- **The practical conclusion is not “trees always win.”** It is that family differences are usually small, tuning and feature representation matter, and a well-calibrated regression model is a serious contender. A shortlist should include strong regression baselines and at most a few tree variants, not a large horse race.

### Benchmark table

| Evidence | Design | Key result | Implication for this project |
|---|---|---|---|
| Grinsztajn et al. 2022 [20] | 45 tabular datasets, ~10k samples; extensive tuning | Tree-based models remain SOTA on medium-sized data | GBDT is the right default; our 775k rows are larger than their benchmark, so leave room for an NN to close the gap |
| Shwartz-Ziv & Armon 2022 [21] | Multiple public datasets; XGBoost vs deep models | XGBoost outperformed deep models and needed less tuning; ensemble improved on XGBoost alone | Include XGBoost; consider a tree + linear ensemble only as exploration |
| Gorishniy et al. 2021 [22] | Diverse tabular tasks; ResNet-like and Transformer vs specialized DL | Simple deep baselines are competitive; no universal winner | A neural net is a legitimate exploration, not a presumed upgrade |
| McElfresh et al. 2023 [23] | 176 datasets, 19 algorithms | Differences often negligible; tuning can matter more than family; TabPFN limited to ~3k rows | Do not let family choice dominate the shortlist; fix tuning budgets |
| Borisov et al. 2022 [24] | 5 real datasets, 11 DL approaches | GBDT still mostly outperforms DL | GBDT prior is stronger than NN prior |
| Christodoulou et al. 2019 [25] | 71 studies, 282 LR-vs-ML comparisons | No ML advantage at low risk of bias (logit AUC difference 0.00, 95% CI −0.18 to 0.18) | Logistic regression must be shortlisted, not treated as a token baseline |
| De Silva et al. 2020 [26] | 23 studies, 40 diabetes models | Pooled c-index 0.812; no external validation; only 7 assessed calibration | Expect modest discrimination; make calibration a gate |
| Li et al. 2025 [27] | 65 studies, 97 diabetes models | LR dominates; ML gains marginal; >80% high risk of bias | Discount published high-AUC claims; favour clean validation |

### Why the India ML literature overstates performance

Three recurring choices explain most of the gap between the 0.99 papers and the 0.70 credible range:

1. **Random train/test splits** put members of the same household or community on both sides. The project's cluster-grouped split is the correct design for DHS data; random splits leak cluster-level information.
2. **SMOTE and other resampling** are used before evaluation. Resampling does not improve AUC and produces badly calibrated probabilities [40]; it also makes accuracy look better at the expense of PPV.
3. **Target-defining variables are included as predictors.** If glucose, diagnosis, or medication is in the feature matrix, the model is predicting the label from itself. Athwani's glucose feature [5] and Thotad's near-perfect results [6] are the clearest examples.

## 4. Calibration and fairness requirements

### 4.1 Calibration

For a screening tool, a probability is only useful if it means what it says. The minimum reporting set should be:

- **Calibration slope and intercept** (ideal 1.0 and 0.0), estimated on out-of-fold or external predictions, with confidence intervals. Van Calster et al. argue that calibration is the “Achilles heel” of predictive analytics and recommend assessing it explicitly at development and validation, with flexible calibration curves and a sample large enough to estimate them [32].
- **Calibration curve** (loess or isotonic) with confidence bands, not just a single summary. A curve shows whether miscalibration is in the clinically relevant risk range.
- **Brier score** as an overall measure of probabilistic accuracy, and log loss as a secondary measure. Steyerberg et al. recommend reporting discrimination and calibration together, and decision-analytic measures when the model informs decisions [33].
- **Recalibration** on out-of-fold predictions using Platt scaling or isotonic regression. Niculescu-Mizil and Caruana showed that boosted trees, random forests, and SVMs produce biased probabilities and that Platt/isotonic correction improves them; isotonic needs a sufficiently large calibration sample [39]. At 775k rows this is feasible, but subgroup-specific isotonic fits are not.
- **Avoid SMOTE and resampling.** van den Goorbergh et al. showed that random undersampling, random oversampling, and SMOTE produce poorly calibrated models that overestimate the minority-class probability, with no AUC gain; shifting the threshold achieves the same sensitivity/specificity balance without destroying calibration [40]. Class weighting has a similar probability-distorting effect, so it must be paired with post-hoc calibration or dropped in favour of an unweighted model plus a threshold.
- **ECE is a secondary diagnostic.** It is useful for a quick summary and is already in the pipeline, but it depends on binning and should not replace slope/intercept and curves.

### 4.2 Threshold selection and clinical utility

- **Use decision curve analysis / net benefit** to compare candidates across plausible threshold probabilities, rather than selecting on Youden's J or accuracy [34]. Net benefit incorporates the relative harm of false positives and false negatives.
- **The project's sensitivity-first rule is defensible for screening**, but the operating point must be reported with sensitivity, specificity, PPV, NPV, and the number needed to screen. At 8.33% prevalence, the current primary model's 0.81 sensitivity and 0.46 specificity yield PPV ≈ 0.12 and NPV ≈ 0.96 — roughly 8 referrals per detected case. That is a product decision, not a model-family decision.
- **Report the threshold's uncertainty.** A cluster-bootstrap CI for sensitivity, specificity, and PPV at the chosen threshold is more useful than a point estimate.

### 4.3 Fairness

- **Report subgroup calibration, not only subgroup AUC.** A model can have equal AUC across groups but systematically over- or under-predict in one group. Report calibration slope/intercept and observed/expected ratios by sex, age band, wealth quintile, residence, education, and state.
- **Use equalized-odds-type metrics at the screening threshold**: TPR (sensitivity) and FPR (1 − specificity) gaps across groups, with CIs [44]. Demographic parity is usually inappropriate when true prevalence differs by group; equalized odds asks whether the model misses cases or over-refers at different rates.
- **Do not claim fairness from dropping protected attributes.** Obermeyer et al. showed that a widely used risk algorithm was biased because it used health-care cost as a proxy for need; at the same risk score, Black patients were sicker [41]. Vyas et al. show how race-based “corrections” in clinical algorithms can embed inequity, and why removing race without examining proxies is insufficient [42]. The project's state/religion/caste audit-only variants are the right response: exclude them from deployment, but measure what including them would change.
- **Report n and event counts for every subgroup**, and flag small subgroups. The project's existing rule (n ≥ 200 and positives ≥ 50, state rows only at n ≥ 500) is sensible and should be stated in the model card.
- **Account for label bias.** Diagnosed diabetes is access-dependent; undiagnosed diabetes is concentrated in men and lower-wealth groups in NFHS-5 [9]. A subgroup with lower measured prevalence may have more undiagnosed disease, so equal AUC does not prove equal clinical utility.
- **Use cluster bootstrap** for subgroup CIs, because respondents are clustered within PSUs and households.

### 4.4 Reporting, validation, and governance

- **TRIPOD+AI** is the current reporting standard for prediction models using regression or ML; it supersedes TRIPOD 2015 and requires calibration, performance, and fairness-related reporting [35,36].
- **PROBAST** is the appropriate risk-of-bias tool for the model-development and validation literature, and a useful self-check for this project [38].
- **Sample size**: Riley et al. provide the minimum sample-size framework for binary outcomes; at 64,526 positives and 32 predictors, this project is far above the minimum for development, but subgroup analyses and spline terms still need event-count checks [37].
- **External validation** is the main missing piece. NFHS-6 microdata or an ICMR-INDIAB-linked cohort would be the natural next test; until then, state-level and age-band transportability should be reported as internal checks, not as external validation.
- **Governance**: the WHO guidance on ethics and governance of AI for health sets out autonomy, safety, transparency, accountability, inclusiveness/equity, and sustainability principles [45]. The DHS data-use terms prohibit attempts to identify respondents and limit redistribution; only aggregate results and model artifacts should be published [46].

## 5. Candidate assessment for THIS project

### Context that constrains the choice

- **n ≈ 775k development rows, 32 features, 8.33% positive.** This is a large, low-dimensional, mixed-type tabular problem. It is the regime where GBDT is strong and where a well-tuned NN can be competitive, but also where logistic regression with splines is very hard to beat by much.
- **Almost no high-cardinality categoricals in the deployed set.** State, religion, and caste are audit-only. This specifically weakens CatBoost's main structural advantage (ordered target statistics for high-cardinality categorical features) [31].
- **Calibration is a hard gate.** Any family that improves AUC but worsens slope/intercept or subgroup calibration should lose.
- **SHAP is a product requirement.** TreeSHAP is exact and fast for tree ensembles [48,49]; linear SHAP is exact for logistic regression; KernelSHAP for a neural net is slower and approximate, which matters at p95 < 300 ms.
- **The incumbent is already good.** The published primary LightGBM has ROC-AUC 0.710, PR-AUC 0.204, Brier 0.072, and ECE 0.0014. The shortlist should be judged against that, not against zero.

### Assessment

| Family | Evidence strength | Fit to our data and constraints | SHAP / latency feasibility | Recommendation |
|---|---|---|---|---|
| Prevalence-only | Reference baseline; no evidence of benefit | Perfect; establishes the do-nothing floor | Not applicable | Keep as **unscored reference baseline** |
| Logistic regression (main effects) | **Strong**: international risk models are regression; no ML advantage at low risk of bias [16,25] | **Excellent**: handles mixed features with median imputation, scaling, one-hot; stable at 775k; may underfit nonlinearity | Exact coefficients and linear SHAP; <1 ms | **Shortlist C1** |
| Logistic regression + restricted cubic splines | **Moderate**: continuous risk factors are nonlinear; splines are standard [47] | **Good**: lets age, BMI, waist, SBP, DBP bend without hand-cut categories; more parameters but n is large | Exact for spline basis; explanations map back to the original feature; <1 ms | **Shortlist C2** |
| Random forest | **Moderate**: strong tabular baseline [28]; India NFHS studies use RF but are methodologically weak [4,5] | **Good**: captures interactions, robust to outliers; needs probability calibration; slower than GBDT | TreeSHAP exact [49]; roughly 10–50 ms for a few hundred trees; feasible | **Shortlist C3** |
| LightGBM | **Strong**: GBDT is SOTA on medium tabular data [20,21,24,30]; incumbent validated in this repo | **Excellent**: native categoricals, handles missingness, fast; already the benchmark | TreeSHAP exact; ~1–5 ms | **Shortlist C4 (incumbent)** |
| XGBoost | **Strong** benchmark evidence [21,29]; no India-specific NFHS evidence | **Excellent**: same features, histogram method, strong regularization; slightly less native categorical support | TreeSHAP exact; ~1–5 ms | **Shortlist C5** |
| CatBoost | **Moderate** algorithm evidence [31]; no India-specific evidence; most published comparisons use small clinical or Kaggle-style data | **Weaker fit**: few high-cardinality categoricals, so ordered target statistics add little; slower training | TreeSHAP possible but slower; ~5–20 ms | **Shortlist C6, low prior** — include as a GBDT implementation check, not an expected winner |
| MLP / feed-forward neural net | **Mixed**: can match GBDT at large n but rarely dominates [22,23]; India evidence is thin and small | **Moderate**: needs scaling/embeddings, careful tuning, missing-data handling; 775k is the favourable regime | Inference <5 ms, but KernelSHAP is slow/approximate; exact Shapley is expensive | **Exploration bucket**, not scored |
| EBM / GA2M | **Moderate**: interpretable additive model with pairwise interactions | **Good**: fast, inherently interpretable; less familiar to reviewers | Native explanations; <1 ms | **Optional exploration** |
| TabNet / FT-Transformer | **Weak** for this size and feature set | **Poor**: data-hungry, tuning-heavy, no clear advantage over GBDT | Expensive explanations | **Exclude** |

### Notes on specific choices

- **Why include both LR and LR + splines?** They separate two questions: “is a nonlinear model better?” and “is a tree better?” If spline LR closes most of the gap to LightGBM, the simpler, more transportable model wins on governance grounds.
- **Why include both LightGBM and XGBoost?** They are different implementations of the same broad family. Including both tests whether the incumbent's edge is real or an implementation artifact. It is a robustness check, not a search for a fifth family.
- **Why is CatBoost the lowest-prior tree?** Its distinctive advantage is categorical handling [31], and the deployed feature set has almost none. It may still win on regularization or default tuning, but the prior is weak. Cap its tuning budget so it does not consume disproportionate compute.
- **Why not SMOTE?** The India NFHS papers use it heavily [4,5], but the evidence says it harms calibration without improving discrimination [40]. Use class weights or no weighting, then calibrate post hoc and choose the threshold on OOF predictions.
- **Why not a neural net on the shortlist?** The large n is the strongest argument for one, but the benchmark evidence says parity at best [22,23], and the India-specific evidence is thin. A neural net also makes exact local explanations and latency harder. It is worth exploring, but it has not earned a scored slot before seeing results.

## 6. Recommended shortlist

### Reference baseline (not scored)

- **B0. Prevalence-only.** Predicts the overall positive rate for every respondent. It is the floor that every candidate must beat on discrimination, calibration, and net benefit. It is not a candidate and cannot win.

### Scored candidates (6)

| ID | Candidate | One-line rationale |
|---|---|---|
| C1 | **Logistic regression, main effects**, median imputation + scaling + one-hot, balanced weights, post-hoc calibration | The international default and the strongest low-complexity baseline; tests whether complexity is needed at all [16,25]. |
| C2 | **Logistic regression with restricted cubic splines** on age, BMI, waist, SBP, DBP; same preprocessing and calibration | Separates nonlinearity from model family; if it matches GBDT, prefer it for transportability and exact explanations [47]. |
| C3 | **Random forest**, class-balanced, tuned `min_samples_leaf`/`max_features`, post-hoc calibration | A bagged-tree alternative that tests variance reduction and gives a different bias profile from boosting [28]; also the family used in the India NFHS literature [4,5]. |
| C4 | **LightGBM**, existing configuration (native categoricals, `is_unbalance`), post-hoc calibration | The incumbent and the strongest prior on medium tabular data; fast, SHAP-exact, already validated in this repo [20,21,24]. |
| C5 | **XGBoost**, histogram tree method, `scale_pos_weight`, same features and calibration | An independent GBDT implementation; tests whether the incumbent's edge is real or LightGBM-specific [21,29]. |
| C6 | **CatBoost**, same features, modest tuning budget, post-hoc calibration | A third GBDT implementation with different regularization; low prior because the feature set has few high-cardinality categoricals [31]. |

### Explicit decision on neural nets

**A neural net does not belong on the scored shortlist. It belongs in an exploration bucket.** The reasons are:

1. The strongest tabular benchmarks find GBDT at least as good on medium data, and neural nets rarely dominate after tuning [20–24].
2. The clinical-prediction literature finds no consistent ML advantage over regression at low risk of bias [25].
3. India-specific neural-net evidence is thin and mostly small clinical datasets; it does not support a strong prior.
4. Exact local explanations for a neural net are slower and less mature than TreeSHAP, which conflicts with the p95 < 300 ms requirement and the SHAP-based product surface [48,49].

**Exploration protocol (pre-specified):** train a small MLP with embeddings or one-hot inputs on the same folds, calibrate it identically, and promote it only if it beats C4 by a pre-agreed margin — for example, ΔROC-AUC ≥ 0.005 with a cluster-bootstrap 95% CI excluding zero — **and** its calibration slope/intercept and subgroup gaps are no worse. Otherwise report it as an exploration result, not a candidate.

### If a seventh scored slot is wanted

Spend it on a **monotone-constrained LightGBM** (age, BMI, waist, SBP non-decreasing in risk) rather than on a second neural architecture. It tests whether domain constraints cost discrimination and whether they improve transportability/trust. Do not add a second NN, TabNet, or a stacking ensemble before the six candidates have been scored; stacking multiplies the calibration and explanation surface without addressing the core question.

## 7. What the evidence does not settle

1. **Whether any family will beat LightGBM by a clinically meaningful margin on this exact target.** The India literature does not isolate family effects, and the international literature is mostly regression. The shortlist is designed to detect a small gain, not to assume one.
2. **Whether the target should be random glucose >140, HbA1c-defined diabetes, or diagnosed diabetes.** These labels have different measurement error, prevalence, and fairness properties. The HbA1c subsample could be used as a sensitivity analysis, but the literature does not say which target is best for an ASHA-referral tool.
3. **How much cluster-grouped validation lowers performance relative to the random splits used in the India ML papers.** The project's 0.710 is already cluster-grouped; a direct random-split comparison would quantify the optimism in the published literature.
4. **Whether survey weights should be used for training, calibration, and threshold selection.** The project reports weighted metrics but trains unweighted. The literature is inconsistent, and weighting interacts with calibration and subgroup fairness.
5. **Whether the model transports across states and NFHS rounds.** State-level and age-band internal checks are not external validation. NFHS-6 or an ICMR-INDIAB-linked cohort is the natural next test.
6. **Whether a neural net's large-n advantage materialises here.** The benchmarks are mixed and mostly not at 775k rows with 32 low-cardinality features. The exploration protocol is the honest way to find out.
7. **Whether CatBoost's categorical handling matters at all** when the deployed features are continuous, ordinal, or binary and high-cardinality identifiers are audit-only. The prior says no; the experiment can confirm.
8. **What sensitivity/specificity trade-off is acceptable for ASHA referral.** The current 0.81/0.46 operating point refers about 8 people per detected case. The cost of false positives, the availability of confirmatory testing, and the harm of false negatives are programme decisions, not model-selection decisions.
9. **How to handle label bias from diagnosed diabetes.** Diagnosed cases depend on access; undiagnosed cases are concentrated in men and lower-wealth groups [9]. Excluding state/religion/caste from the model does not remove this bias, and subgroup AUC gaps may reflect labels rather than model unfairness.
10. **Whether isotonic calibration on pooled OOF predictions is stable across states and age bands.** Group-specific recalibration may be needed, but it may also overfit small subgroups. This is an open methodological choice for the model card.

## References

### India / South Asia

1. Mohan V, Deepa R, Deepa M, Somannavar S, Datta M. A simplified Indian Diabetes Risk Score for screening for undiagnosed diabetic subjects. *J Assoc Physicians India.* 2005;53:759–763. PMID 16334618. https://pubmed.ncbi.nlm.nih.gov/16334618/
2. Deepa M, Elangovan N, Venkatesan U, et al. Evaluation of Madras Diabetes Research Foundation-Indian Diabetes Risk Score in detecting undiagnosed diabetes in the Indian population: Results from the ICMR-INDIAB population-based study (INDIAB-15). *Indian J Med Res.* 2023. doi:10.4103/ijmr.ijmr_2615_21
3. Karri AK, Guthi VR, Githa PSS. Risk Prediction of high blood glucose among women (15–49 years) and men (15–54 years) in India: An analysis from National Family Health Survey-5 (2019–21). *J Family Med Prim Care.* 2024;13(11):5312–5319. doi:10.4103/jfmpc.jfmpc_929_24
4. Barman B, Choudhury HK, Jajodia B. Interpretable machine learning for diabetes risk prediction: a large-scale analysis of Indian national survey data. *Discover Public Health.* 2025;22. doi:10.1186/s12982-025-01236-8
5. Athwani H, Awasthi K, Athwani M, Pandey R. Machine learning driven insights into socio-economic, demographic, and health determinants of Type-II diabetes: a tuned hyperparameter random forest model using NFHS-5 data. *J Pharm Bioallied Sci.* 2026;18(Suppl 1):S182–S184. doi:10.4103/jpbs.jpbs_1612_25
6. Thotad PN, Bharamagoudar GR, Anami BS. Diabetes disease detection and classification on Indian demographic and health survey data using machine learning methods. *Diabetes Metab Syndr.* 2023;17(1):102690. doi:10.1016/j.dsx.2022.102690
7. Kibria GMA, O'Hagan JR, Shariar G, Khan T, Elfaramawi M. Application of machine learning approaches to develop predictive models for diabetes and hypertension among Bangladeshi adults. *PLOS Glob Public Health.* 2026. doi:10.1371/journal.pgph.0004797
8. Islam MM, Rahman MJ, Chandra Roy D, Maniruzzaman M. Automated detection and classification of diabetes disease based on Bangladesh demographic and health survey data, 2011 using machine learning approach. *Diabetes Metab Syndr.* 2020;14(3):217–219. doi:10.1016/j.dsx.2020.03.004
9. Sahadevan P, Kamal VK, Sasidharan A, Bagepally BS, Kumari D, Pal A. Prevalence and risk factors associated with undiagnosed diabetes in India: Insights from NFHS-5 national survey. *J Glob Health.* 2023;13:04135. doi:10.7189/jogh.13.04135
10. International Institute for Population Sciences (IIPS) and ICF. *National Family Health Survey (NFHS-5), 2019–21: India.* Mumbai: IIPS; 2021. https://dhsprogram.com/pubs/pdf/FR375/FR375.pdf

### International diabetes risk models and guidelines

11. Lindström J, Tuomilehto J. The diabetes risk score: a practical tool to predict type 2 diabetes risk. *Diabetes Care.* 2003;26(3):725–731. doi:10.2337/diacare.26.3.725
12. Schmidt MI, Duncan BB, Bang H, et al. Identifying individuals at high risk for diabetes: the Atherosclerosis Risk in Communities study. *Diabetes Care.* 2005;28(8):2013–2018. doi:10.2337/diacare.28.8.2013
13. Wilson PWF, Meigs JB, Sullivan L, Fox CS, Nathan DM, D'Agostino RB Sr. Prediction of incident diabetes mellitus in middle-aged adults: the Framingham Offspring Study. *Arch Intern Med.* 2007;167(10):1068–1074. doi:10.1001/archinte.167.10.1068
14. Hippisley-Cox J, Coupland C, Robson J, Sheikh A, Brindle P. Predicting risk of type 2 diabetes in England and Wales: prospective derivation and validation of QDScore. *BMJ.* 2009;338:b880. doi:10.1136/bmj.b880
15. Heikes KE, Eddy DM, Arondekar B, Schlessinger L. Diabetes Risk Calculator: a simple tool for detecting undiagnosed diabetes and pre-diabetes. *Diabetes Care.* 2008;31(5):1040–1045. doi:10.2337/dc07-1150
16. Abbasi A, Peelen LM, Corpeleijn E, et al. Prediction models for risk of developing type 2 diabetes: systematic literature search and independent external validation study. *BMJ.* 2012;345:e5900. doi:10.1136/bmj.e5900
17. Noble D, Mathur R, Dent T, Meads C, Greenhalgh T. Risk models and scores for type 2 diabetes: systematic review. *BMJ.* 2011;343:d7163. doi:10.1136/bmj.d7163
18. American Diabetes Association Professional Practice Committee. 2. Diagnosis and classification of diabetes: Standards of Care in Diabetes—2025. *Diabetes Care.* 2025;48(Suppl 1):S27–S49. doi:10.2337/dc25-S002
19. Indian Council of Medical Research. *ICMR Guidelines for Management of Type 2 Diabetes.* New Delhi: ICMR; 2018. https://www.icmr.gov.in/icmrobject/custom_data/pdf/resource-guidelines/ICMR_GuidelinesType2diabetes2018_0.pdf

### Tabular machine learning

20. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? In: *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track.* 2022. arXiv:2207.08815. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html
21. Shwartz-Ziv R, Armon A. Tabular data: deep learning is not all you need. *Inf Fusion.* 2022;81:84–90. doi:10.1016/j.inffus.2021.11.011
22. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting deep learning models for tabular data. In: *Advances in Neural Information Processing Systems 34 (NeurIPS 2021).* 2021. arXiv:2106.11959. https://arxiv.org/abs/2106.11959
23. McElfresh D, Nicholson C, Valverde JM, et al. When do neural nets outperform boosted trees on tabular data? In: *Advances in Neural Information Processing Systems 36 (NeurIPS 2023).* 2023. arXiv:2305.02997. https://arxiv.org/abs/2305.02997
24. Borisov V, Leemann T, Seßler K, Haug J, Pawelczyk M, Kasneci G. Deep neural networks and tabular data: a survey. *IEEE Trans Neural Netw Learn Syst.* 2022;33(12):7686–7701. doi:10.1109/TNNLS.2022.3229161
25. Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *J Clin Epidemiol.* 2019;110:12–22. doi:10.1016/j.jclinepi.2019.02.004
26. De Silva K, Lee WK, Forbes A, Demmer RT, Barton C, Enticott J. Use and performance of machine learning models for type 2 diabetes prediction in community settings: a systematic review and meta-analysis. *Int J Med Inform.* 2020;143:104268. doi:10.1016/j.ijmedinf.2020.104268
27. Li M, Tan A, Song L, Qiu X, Zhou L, Wen Z. Performance of diabetes risk prediction models: a systematic review and meta-analysis. *Endocr Connect.* 2025;14(11):e250353. doi:10.1530/EC-25-0353
28. Breiman L. Random forests. *Mach Learn.* 2001;45(1):5–32. doi:10.1023/A:1010933404324
29. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* 2016:785–794. doi:10.1145/2939672.2939785
30. Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. In: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017).* 2017. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
31. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. In: *Advances in Neural Information Processing Systems 31 (NeurIPS 2018).* 2018. arXiv:1706.09516. https://arxiv.org/abs/1706.09516

### Calibration, fairness, reporting, and governance

32. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW; Topic Group 'Evaluating diagnostic tests and prediction models' of the STRATOS initiative. Calibration: the Achilles heel of predictive analytics. *BMC Med.* 2019;17:230. doi:10.1186/s12916-019-1466-7
33. Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology.* 2010;21(1):128–138. doi:10.1097/EDE.0b013e3181c30fb2
34. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making.* 2006;26(6):565–574. doi:10.1177/0272989X06295361
35. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): the TRIPOD statement. *Ann Intern Med.* 2015;162(1):55–63. doi:10.7326/M14-0697
36. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378. doi:10.1136/bmj-2023-078378
37. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ.* 2020;368:m441. doi:10.1136/bmj.m441
38. Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. *Ann Intern Med.* 2019;170(1):51–58. doi:10.7326/M18-1376
39. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. In: *Proceedings of the 22nd International Conference on Machine Learning (ICML).* 2005:625–632. doi:10.1145/1102351.1102430
40. van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *J Am Med Inform Assoc.* 2022;29(9):1525–1534. doi:10.1093/jamia/ocac093
41. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. *Science.* 2019;366(6464):447–453. doi:10.1126/science.aax2342
42. Vyas DA, Eisenstein LG, Jones DS. Hidden in plain sight — reconsidering the use of race correction in clinical algorithms. *N Engl J Med.* 2020;383(9):874–882. doi:10.1056/NEJMms2004740
43. Chen RJ, Wang JJ, Williamson DFK, et al. Algorithmic fairness in artificial intelligence for medicine and healthcare. *Nat Biomed Eng.* 2023;7:719–742. doi:10.1038/s41551-023-01056-8
44. Hardt M, Price E, Srebro N. Equality of opportunity in supervised learning. In: *Advances in Neural Information Processing Systems 29 (NeurIPS 2016).* 2016. arXiv:1610.02413. https://arxiv.org/abs/1610.02413
45. World Health Organization. *Ethics and Governance of Artificial Intelligence for Health: WHO Guidance.* Geneva: WHO; 2021. https://www.who.int/publications/i/item/9789240029200
46. The DHS Program. *Data Use Terms.* https://dhsprogram.com/data/terms-of-use.cfm
47. Sauerbrei W, Royston P, Binder H. Selection of important variables and determination of functional form for continuous predictors in multivariable model building. *Stat Med.* 2007;26(30):5512–5528. doi:10.1002/sim.3148
48. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017).* 2017. arXiv:1705.07874. https://arxiv.org/abs/1705.07874
49. Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell.* 2020;2:56–67. doi:10.1038/s42256-019-0138-9
