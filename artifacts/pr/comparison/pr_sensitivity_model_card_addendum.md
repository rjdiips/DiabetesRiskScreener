# Model-card addendum: PR BP-based sensitivity model

This addendum documents the PR 15+ and PR 15–49 BP-based sensitivity models. It supplements the per-run model cards and the [primary-versus-PR comparison report](primary_vs_pr_comparison.md). It is written from published aggregate results and can be shared without distributing NFHS-5 microdata.

- **PR 15+ run:** `pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6` (sample `age_15_plus`).
- **PR 15–49 run:** `pr_15to49_bp_20261026T084051779902Z_42_96adb799` (sample `age_15_49`).
- **Primary reference run:** `primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e` (the deployed IR/MR 15–49 model).

**Status.** The PR model is a sensitivity analysis: it is not the deployed model and not the deployed screener. The primary IR/MR 15–49 model remains the model intended for the app. The PR model is not a clinical diagnostic tool and must not be used to diagnose, treat, or allocate care.

## Intended use

The model estimates the probability that an adult has elevated random blood glucose or diagnosed or treated diabetes at the time of the NFHS-5 survey. The target is glucose > 140 mg/dL OR a self-reported high-glucose diagnosis OR current glucose-lowering medication. Random blood glucose is a screening indicator, not a diagnosis.

Its purpose is research and portfolio analysis: to test whether a 12-feature blood-pressure and demographic model can rank diabetes risk across the larger PR sample, and to show what changes when the sample extends to older adults and the primary model's richer feature set is reduced.

It is not intended for individual clinical decisions, clinical triage, insurance, employment, care allocation, or population prevalence estimation. It does not predict future diabetes.

## Sample and feature differences

| Run | Sample | Rows used | Test rows | Test positives | Prevalence | Weighted prevalence |
|---|---|---:|---:|---:|---:|---:|
| Primary IR/MR 15–49 | age_15_49 | 774861 | 154946 | 12943 | 0.0835 | 0.0935 |
| PR 15–49 | age_15_49 | 1290219 | 257801 | 22582 | 0.0876 | 0.0965 |
| PR 15+ | age_15_plus | 1805697 | 360633 | 50596 | 0.1403 | 0.1543 |

- The PR sample extends to adults 50+. In the PR 15+ test set, ages 50+ contribute 102,832 of 360,633 rows (28.51%), with prevalence between 24.77% and 30.14%. That extension explains most of the difference between the PR 15+ and PR 15–49 prevalences.
- 183 respondents coded as transgender (sex code 3) were excluded, because the NFHS-5 audit sample and the primary model use sex codes 1 and 2. The PR model is therefore defined only for respondents coded male or female.
- The PR model uses only rows with a valid systolic/diastolic pair. Rows without a valid pair remain in the local PR feature table but cannot be scored by the BP model.
- The primary model uses 32 features; the PR model uses 12: `female`, `age`, `urban`, `wealth`, `education`, `tobacco_any`, `alcohol_any`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication`. The PR model does not use anthropometry or diet.
- The PR model also excludes state, religion, social group, marital status, children ever born, the smoking/smokeless split, alcohol frequency, and cooking fuel. Anthropometry is measured only for younger adults in the PR file, so using it would make missingness age- and sex-dependent.
- On the age-matched comparison, the primary 32-feature model has ROC-AUC 0.7099 and PR-AUC 0.2035, versus ROC-AUC 0.6979 and PR-AUC 0.1900 for the 12-feature PR 15–49 model. This is descriptive, not a pure ablation: the PR and IR/MR instruments measure different concepts.
- PR religion and social group describe the household head, not necessarily the respondent. They are excluded from the default PR model; the spec's fairness variant is the mechanism for auditing them, and no fairness-variant result is claimed here. A subgroup gap in those variables cannot be read as a respondent-level disparity.
- The PR 15–49 and PR 15+ runs use the same 12 features, the same cluster-grouped partition, and the same modelling logic. The 15–49 run filters the parent partition rather than re-splitting, so the difference between the two PR runs isolates the sample extension. The primary comparison is not a pure feature ablation, because the IR/MR and PR instruments measure different concepts.

## Age gradient

| Age band | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity at threshold | Δ ROC-AUC vs overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15-24 | 85824 | 3096 | 3.61% | 0.5784 | 0.0487 | 0.0048 | -0.1676 |
| 25-34 | 77588 | 5613 | 7.23% | 0.5906 | 0.1023 | 0.1484 | -0.1553 |
| 35-44 | 64319 | 8523 | 13.25% | 0.6117 | 0.2008 | 0.7029 | -0.1342 |
| 45-49 | 30070 | 5350 | 17.79% | 0.6152 | 0.2714 | 0.9806 | -0.1307 |
| 50-59 | 46826 | 11598 | 24.77% | 0.6322 | 0.3699 | 0.9999 | -0.1137 |
| 60-69 | 35898 | 10356 | 28.85% | 0.6466 | 0.4367 | 1.0000 | -0.0993 |
| 70+ | 20108 | 6060 | 30.14% | 0.6314 | 0.4246 | 1.0000 | -0.1145 |

Prevalence rises from 3.61% in ages 15-24 to 30.14% in ages 70+. Adults aged 50+ contribute 102,832 of 360,633 PR 15+ test rows (28.51%). Age-band ROC-AUC ranges from 0.5784 to 0.6466.

At the selected threshold, sensitivity is 0.0048 in ages 15–24, 1.0000 in ages 70+, and higher across the older bands. Because the operating point is tuned to overall sensitivity, it mostly flags older adults and leaves the youngest band almost unflagged. Discrimination within age bands is modest, which is why a younger screening population would need its own calibration and threshold rather than reuse of the PR 15+ threshold.

## Calibration

| Metric | PR 15–49 | PR 15+ |
|---|---:|---:|
| ROC-AUC | 0.6979 | 0.7459 |
| PR-AUC | 0.1900 | 0.3320 |
| Brier | 0.0761 | 0.1077 |
| Log loss | 0.2761 | 0.3568 |
| ECE | 0.0006 | 0.0011 |
| Calibration slope | 0.9946 | 0.9965 |
| Calibration intercept | -0.0114 | -0.0043 |
| Calibrator | isotonic | isotonic |

Both runs selected the calibrator shown in the table on out-of-fold predictions; in the published runs this is isotonic. Test ECE is close to zero in both runs, and the calibration slopes are close to one. Brier and log loss are lower in the PR 15–49 run largely because prevalence is lower there; they are not direct evidence that one model has better probability quality.

| Survey-weighted metric | PR 15–49 | PR 15+ |
|---|---:|---:|
| Prevalence | 0.0965 | 0.1543 |
| ROC-AUC | 0.6955 | 0.7471 |
| PR-AUC | 0.2064 | 0.3595 |
| Brier | 0.0828 | 0.1155 |
| ECE | 0.0085 | 0.0105 |
| Sensitivity | 0.7886 | 0.7904 |
| Specificity | 0.4768 | 0.5634 |
| PPV | 0.1387 | 0.2483 |
| NPV | 0.9548 | 0.9364 |

Survey-weighted metrics use the NFHS-5 survey weight. Training is unweighted; the weighted block is an evaluation view, not a different model.

Reliability above 0.70 predicted risk is sparsely supported: 31 PR 15+ test rows and 0 PR 15–49 test rows fall in those bins. The high-risk tail therefore cannot support a precise calibration claim even though overall ECE is small.

## Threshold

| Run | Threshold | Sensitivity | Specificity | PPV | NPV | F1 | Sensitivity @ 80% specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| PR 15–49 | 0.0599 | 0.7958 | 0.4708 | 0.1262 | 0.9600 | 0.2178 | 0.4425 |
| PR 15+ | 0.1040 | 0.7925 | 0.5614 | 0.2277 | 0.9431 | 0.3538 | 0.5223 |

Each run selects its own referral threshold on its own out-of-fold predictions, targeting at least 80% sensitivity. The PR 15+ threshold is 0.1040; the PR 15–49 threshold is 0.0599. These are policy choices, not intrinsic model properties.

Threshold performance depends on prevalence. A threshold and its PPV or NPV cannot be transported to another age structure, state, or clinical setting without recalibration and a new threshold decision. At 80% specificity, sensitivity is 0.4425 for PR 15–49 and 0.5223 for PR 15+; this higher-specificity operating point is more demanding than the 80%-sensitivity threshold.

## Subgroup gaps

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | Sensitivity | Δ ROC-AUC | Δ Sensitivity | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| sex | female | 192409 | 25204 | 13.10% | 0.7463 | 0.7594 | +0.0003 | -0.0331 | yes |
| sex | male | 168224 | 25392 | 15.09% | 0.7433 | 0.8254 | -0.0026 | +0.0329 | yes |
| wealth | 1 | 74363 | 8284 | 11.14% | 0.7080 | 0.7017 | -0.0379 | -0.0908 | yes |
| wealth | 5 | 62477 | 11271 | 18.04% | 0.7760 | 0.8737 | +0.0301 | +0.0811 | yes |
| residence | rural | 272967 | 35627 | 13.05% | 0.7350 | 0.7708 | -0.0110 | -0.0217 | yes |
| residence | urban | 87666 | 14969 | 17.08% | 0.7659 | 0.8443 | +0.0200 | +0.0518 | yes |
| education | 0 | 94263 | 16418 | 17.42% | 0.6803 | 0.8834 | -0.0656 | +0.0909 | yes |
| education | 3 | 46432 | 5271 | 11.35% | 0.7619 | 0.6750 | +0.0160 | -0.1175 | yes |
| state | 37 | 1215 | 198 | 16.30% | 0.6568 | 0.6212 | -0.0891 | -0.1713 | yes |
| state | 7 | 4918 | 620 | 12.61% | 0.8162 | 0.8919 | +0.0703 | +0.0994 | yes |

Sex: female prevalence is 13.10% versus male 15.09%; sensitivity at the chosen threshold is 0.7594 versus 0.8254. The model under-flags women relative to men in the PR 15+ run even though sex is a model feature.

Wealth: prevalence rises from 11.14% in quintile 1 to 18.04% in quintile 5, and the sensitivity gap moves from -0.0908 to +0.0811. Education shows a similar pattern: prevalence is 17.42% in the lowest education group and 11.35% in the highest, while the sensitivity gap moves from +0.0909 to -0.1175.

State differences in discrimination are wide: the ROC-AUC gap ranges from -0.0891 in one reliable state (prevalence 16.30%) to +0.0703 in another (prevalence 12.61%). State rows appear only where the test set has at least 500 respondents; the codes are raw NFHS state codes.

Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged unreliable. 1 PR 15+ subgroup row is flagged that way and must not be used to claim fairness.

Gaps are descriptive, not causal. Excluding a protected attribute does not by itself make a model fair; the fairness variant exists to measure what including religion and social group changes, and the household-head wording limits that audit.

## Label bias

- The target includes self-reported diagnosis and medication. Both depend on prior testing and access to care, so the model can learn access-to-care patterns as well as disease risk.
- Random glucose is a single survey measurement, not a fasting glucose, oral glucose tolerance test, or HbA1c. A value at or below 140 mg/dL does not rule out diabetes, and a transient high value can be labelled positive.
- Undiagnosed diabetes with a non-elevated random glucose is counted as negative. Prevalence and subgroup gaps therefore reflect both disease and diagnosis opportunity.
- Wealth, residence, education, and state gaps are especially vulnerable to this label bias: richer or urban respondents may be more likely to have been tested and diagnosed. The model's use of wealth and education can reproduce that pathway.
- The NFHS-5 fact-sheet target (glucose > 140 OR medication, without the diagnosis item) is reported separately in each run. The primary target here is slightly broader, so its prevalence is not identical to the published NFHS-5 fact-sheet prevalence.
- The survey is from 2019–21. There is no external validation, no NFHS-6 replication, and no clinical validation.

## Limitations

- Raw probabilities are not directly comparable across the three runs because the samples and feature sets differ. Only discrimination, calibration, threshold behaviour, and subgroup gaps are compared.
- The samples are not nested or exchangeable. The primary model uses the IR/MR recodes, while the PR models use the household-member recode; the primary sample is mostly female and the PR sample is far more balanced.
- The PR 15–49 run is the cleanest feature-set comparison, but it is not a pure ablation of the primary model: the PR and IR/MR instruments measure different concepts and the PR sample covers a different population.
- Older adults are present only in the PR 15+ run. Any difference between the PR 15+ and PR 15–49 runs mixes the age extension with the different age-specific case mix.
- Survey-weighted metrics are reported where the run produced them. The test evaluation is still a single cluster-grouped holdout, not external validation.
- Threshold metrics depend on prevalence. PPV and NPV cannot be transported to another setting without accounting for that setting's prevalence.
- The PR model is a sensitivity analysis, not the deployed screener. It is not a diagnostic test and must not be used for clinical, insurance, employment, or care-allocation decisions.
- The PR model omits anthropometry, diet, marital status, children ever born, the smoking/smokeless split, alcohol frequency, and cooking fuel. Those omissions are deliberate, but they mean the model cannot show how much those concepts would add.
- PR adults without a valid blood-pressure pair are excluded from the BP model. They remain in the feature table, but the model cannot score them without a BP reading.
- High-risk calibration is sparse, and the evaluation is one cluster-grouped holdout with one seed and five development folds, not external validation.
- 183 transgender respondents are not represented, and sex is encoded only as male or female. The model must not be presented as covering all adults.
- Religion and social group describe the household head. Any audit using them is a household-level proxy, not a respondent-level attribute.
- Subgroup gaps are descriptive and not causal. No fairness mitigation, reweighting, or adversarial debiasing was applied.

## Non-goals

- A deployed PR model, a PR model in the API or demo, or a replacement for the primary screener.
- A clinical diagnostic tool, a diagnostic test, or a triage tool.
- Prediction of future diabetes, incident diabetes, or complications.
- Insurance, employment, care-allocation, or other high-stakes decisions.
- Population prevalence estimation, district or small-area estimates, or policy targeting.
- External validation, NFHS-6 replication, clinical validation, or causal inference.
- A PR anthropometry model, a PR diet model, or a questionnaire-only model.
- Fairness mitigation, model compression, ONNX export, or mobile deployment.

## Publication

This addendum contains only aggregate counts, metrics, and run identifiers from the published runs. It contains no respondent records, no split assignments, no per-respondent SHAP values, and no model binaries. Raw and processed NFHS-5 microdata are not redistributed.

## Citation

International Institute for Population Sciences (IIPS) and ICF. 2021. *National Family Health Survey (NFHS-5), 2019-21: India*. Mumbai: IIPS.
