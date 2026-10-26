# pr_15plus_bp results

Run: `pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6`

This is a research and portfolio model. It estimates the probability of elevated random blood glucose or diagnosed diabetes; it is not a diagnosis.

## Run role

PR 15+ BP sensitivity model covering all adults aged 15+ with a valid blood-pressure pair.

- Sample: `age_15_plus`
- Feature variant: `default`

## Sample

| Split | Rows |
|---|---:|
| Development | 1445064 |
| Test | 360633 |
| Total used | 1805697 |

## Model selection

| Family | Mean PR-AUC | Mean Brier | Mean log loss | Selected |
|---|---:|---:|---:|:---:|
| prevalence | 0.1401 | 0.1205 | 0.4052 | no |
| logistic | 0.3102 | 0.2062 | 0.6056 | no |
| lightgbm | 0.3356 | 0.2051 | 0.5950 | yes |

## Test performance at the chosen threshold

Threshold: **0.1040** (target sensitivity 0.80)

| Metric | Value |
|---|---:|
| n | 360633 |
| positives | 50596 |
| prevalence | 0.1403 |
| ROC-AUC | 0.7459 |
| PR-AUC | 0.3320 |
| Brier | 0.1077 |
| Log loss | 0.3568 |
| Sensitivity @ 80% specificity | 0.5223 |
| Sensitivity | 0.7925 |
| Specificity | 0.5614 |
| PPV | 0.2277 |
| NPV | 0.9431 |
| F1 | 0.3538 |
| Calibration slope | 0.9965 |
| Calibration intercept | -0.0043 |
| ECE | 0.0011 |

### 95% cluster-bootstrap intervals

| Metric | Point | Low | High |
|---|---:|---:|---:|
| roc_auc | 0.7459 | 0.7431 | 0.7489 |
| pr_auc | 0.3320 | 0.3262 | 0.3377 |
| brier | 0.1077 | 0.1064 | 0.1090 |

### Survey-weighted metrics

| Metric | Value |
|---|---:|
| n | 360633 |
| positives | 50596 |
| prevalence | 0.1543 |
| ROC-AUC | 0.7471 |
| PR-AUC | 0.3595 |
| Brier | 0.1155 |
| Log loss | 0.3774 |
| Sensitivity @ 80% specificity | 0.5305 |
| Sensitivity | 0.7904 |
| Specificity | 0.5634 |
| PPV | 0.2483 |
| NPV | 0.9364 |
| F1 | 0.3779 |
| Calibration slope | 0.9764 |
| Calibration intercept | 0.0563 |
| ECE | 0.0105 |

### Fact-sheet target (reported separately)

Column: `target_fact_sheet`

| Metric | Value |
|---|---:|
| n | 360578 |
| positives | 47338 |
| prevalence | 0.1313 |
| ROC-AUC | 0.7512 |
| PR-AUC | 0.3211 |
| Brier | 0.1020 |
| Log loss | 0.3411 |
| Sensitivity @ 80% specificity | 0.5334 |
| Sensitivity | 0.8025 |
| Specificity | 0.5592 |
| PPV | 0.2158 |
| NPV | 0.9493 |
| F1 | 0.3401 |
| Calibration slope | 1.0204 |
| Calibration intercept | -0.0537 |
| ECE | 0.0089 |

## Calibration

Calibrator selected on out-of-fold development predictions: **isotonic**.

| Method | Brier | Log loss |
|---|---:|---:|
| isotonic | 0.1076 | 0.3562 |
| platt | 0.1076 | 0.3563 |

## Subgroups

Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged `Reliable = no` and must not be used to claim fairness.

State rows are reported only where the test set has at least 500 respondents.

### Sex-specific metrics (mandatory)

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sex | female | 192409 | 25204 | 0.1310 | 0.7463 | 0.3195 | 0.7594 | 0.0021 | -0.0093 | 0.0003 | -0.0125 | -0.0331 | 0.0010 | yes |
| sex | male | 168224 | 25392 | 0.1509 | 0.7433 | 0.3439 | 0.8254 | 0.0014 | 0.0106 | -0.0026 | 0.0119 | 0.0329 | 0.0003 | yes |

### Age-band-specific metrics (mandatory)

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| age_band | 15-24 | 85824 | 3096 | 0.0361 | 0.5784 | 0.0487 | 0.0048 | 0.0001 | -0.1042 | -0.1676 | -0.2832 | -0.7877 | -0.0010 | yes |
| age_band | 25-34 | 77588 | 5613 | 0.0723 | 0.5906 | 0.1023 | 0.1484 | 0.0017 | -0.0680 | -0.1553 | -0.2297 | -0.6441 | 0.0006 | yes |
| age_band | 35-44 | 64319 | 8523 | 0.1325 | 0.6117 | 0.2008 | 0.7029 | 0.0032 | -0.0078 | -0.1342 | -0.1312 | -0.0896 | 0.0021 | yes |
| age_band | 45-49 | 30070 | 5350 | 0.1779 | 0.6152 | 0.2714 | 0.9806 | 0.0041 | 0.0376 | -0.1307 | -0.0606 | 0.1880 | 0.0030 | yes |
| age_band | 50-59 | 46826 | 11598 | 0.2477 | 0.6322 | 0.3699 | 0.9999 | 0.0044 | 0.1074 | -0.1137 | 0.0380 | 0.2074 | 0.0033 | yes |
| age_band | 60-69 | 35898 | 10356 | 0.2885 | 0.6466 | 0.4367 | 1.0000 | 0.0049 | 0.1482 | -0.0993 | 0.1047 | 0.2075 | 0.0038 | yes |
| age_band | 70+ | 20108 | 6060 | 0.3014 | 0.6314 | 0.4246 | 1.0000 | 0.0079 | 0.1611 | -0.1145 | 0.0926 | 0.2075 | 0.0068 | yes |

### Other subgroups

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wealth | 1 | 74363 | 8284 | 0.1114 | 0.7080 | 0.2252 | 0.7017 | 0.0020 | -0.0289 | -0.0379 | -0.1068 | -0.0908 | 0.0009 | yes |
| wealth | 2 | 79170 | 9569 | 0.1209 | 0.7218 | 0.2585 | 0.7437 | 0.0023 | -0.0194 | -0.0241 | -0.0735 | -0.0489 | 0.0012 | yes |
| wealth | 3 | 75645 | 10529 | 0.1392 | 0.7360 | 0.3031 | 0.7857 | 0.0046 | -0.0011 | -0.0099 | -0.0289 | -0.0068 | 0.0035 | yes |
| wealth | 4 | 68978 | 10943 | 0.1586 | 0.7558 | 0.3697 | 0.8269 | 0.0022 | 0.0183 | 0.0099 | 0.0377 | 0.0344 | 0.0011 | yes |
| wealth | 5 | 62477 | 11271 | 0.1804 | 0.7760 | 0.4257 | 0.8737 | 0.0023 | 0.0401 | 0.0301 | 0.0937 | 0.0811 | 0.0012 | yes |
| residence | rural | 272967 | 35627 | 0.1305 | 0.7350 | 0.2974 | 0.7708 | 0.0014 | -0.0098 | -0.0110 | -0.0346 | -0.0217 | 0.0003 | yes |
| residence | urban | 87666 | 14969 | 0.1708 | 0.7659 | 0.4024 | 0.8443 | 0.0037 | 0.0305 | 0.0200 | 0.0704 | 0.0518 | 0.0026 | yes |
| education | 0 | 94263 | 16418 | 0.1742 | 0.6803 | 0.3031 | 0.8834 | 0.0028 | 0.0339 | -0.0656 | -0.0288 | 0.0909 | 0.0017 | yes |
| education | 1 | 48777 | 8376 | 0.1717 | 0.7284 | 0.3573 | 0.8487 | 0.0012 | 0.0314 | -0.0175 | 0.0253 | 0.0562 | 0.0001 | yes |
| education | 2 | 170972 | 20494 | 0.1199 | 0.7661 | 0.3343 | 0.7267 | 0.0016 | -0.0204 | 0.0202 | 0.0023 | -0.0658 | 0.0005 | yes |
| education | 3 | 46432 | 5271 | 0.1135 | 0.7619 | 0.3417 | 0.6750 | 0.0023 | -0.0268 | 0.0160 | 0.0097 | -0.1175 | 0.0012 | yes |
| education | missing | 189 | 37 | 0.1958 | 0.7729 | 0.4725 | 0.8919 | 0.0307 | 0.0555 | 0.0270 | 0.1405 | 0.0994 | 0.0296 | no |
| state | 1 | 11459 | 1042 | 0.0909 | 0.7657 | 0.2546 | 0.8052 | 0.0434 | -0.0494 | 0.0198 | -0.0774 | 0.0127 | 0.0423 | yes |
| state | 10 | 21633 | 3198 | 0.1478 | 0.7290 | 0.3188 | 0.7192 | 0.0262 | 0.0075 | -0.0169 | -0.0131 | -0.0733 | 0.0251 | yes |
| state | 11 | 1696 | 257 | 0.1515 | 0.7271 | 0.2896 | 0.8482 | 0.0211 | 0.0112 | -0.0188 | -0.0423 | 0.0557 | 0.0200 | yes |
| state | 12 | 8424 | 1021 | 0.1212 | 0.6800 | 0.2204 | 0.7297 | 0.0155 | -0.0191 | -0.0659 | -0.1116 | -0.0628 | 0.0144 | yes |
| state | 13 | 3982 | 522 | 0.1311 | 0.7826 | 0.3436 | 0.8506 | 0.0203 | -0.0092 | 0.0367 | 0.0116 | 0.0581 | 0.0192 | yes |
| state | 14 | 4780 | 687 | 0.1437 | 0.7439 | 0.3181 | 0.8311 | 0.0168 | 0.0034 | -0.0020 | -0.0139 | 0.0386 | 0.0157 | yes |
| state | 15 | 4208 | 632 | 0.1502 | 0.7394 | 0.3322 | 0.8054 | 0.0123 | 0.0099 | -0.0065 | 0.0002 | 0.0129 | 0.0112 | yes |
| state | 16 | 3057 | 527 | 0.1724 | 0.6910 | 0.3084 | 0.6983 | 0.0421 | 0.0321 | -0.0549 | -0.0236 | -0.0942 | 0.0410 | yes |
| state | 17 | 6087 | 723 | 0.1188 | 0.7267 | 0.2562 | 0.6694 | 0.0124 | -0.0215 | -0.0192 | -0.0758 | -0.1231 | 0.0113 | yes |
| state | 18 | 16461 | 2357 | 0.1432 | 0.7292 | 0.3183 | 0.7374 | 0.0153 | 0.0029 | -0.0167 | -0.0137 | -0.0551 | 0.0142 | yes |
| state | 19 | 10828 | 2172 | 0.2006 | 0.7072 | 0.3629 | 0.7145 | 0.0637 | 0.0603 | -0.0387 | 0.0309 | -0.0780 | 0.0626 | yes |
| state | 2 | 4477 | 615 | 0.1374 | 0.7452 | 0.3400 | 0.8488 | 0.0292 | -0.0029 | -0.0007 | 0.0080 | 0.0563 | 0.0281 | yes |
| state | 20 | 12519 | 1599 | 0.1277 | 0.7403 | 0.3057 | 0.7530 | 0.0036 | -0.0126 | -0.0056 | -0.0262 | -0.0395 | 0.0025 | yes |
| state | 21 | 13960 | 2146 | 0.1537 | 0.7290 | 0.3292 | 0.7754 | 0.0161 | 0.0134 | -0.0169 | -0.0028 | -0.0171 | 0.0150 | yes |
| state | 22 | 12640 | 1357 | 0.1074 | 0.7321 | 0.2673 | 0.7605 | 0.0210 | -0.0329 | -0.0138 | -0.0647 | -0.0320 | 0.0199 | yes |
| state | 23 | 24531 | 2809 | 0.1145 | 0.7306 | 0.2602 | 0.7682 | 0.0181 | -0.0258 | -0.0153 | -0.0718 | -0.0243 | 0.0170 | yes |
| state | 24 | 17444 | 2678 | 0.1535 | 0.7309 | 0.3297 | 0.7909 | 0.0106 | 0.0132 | -0.0150 | -0.0023 | -0.0016 | 0.0095 | yes |
| state | 25 | 1790 | 299 | 0.1670 | 0.7436 | 0.3712 | 0.7826 | 0.0204 | 0.0267 | -0.0023 | 0.0392 | -0.0099 | 0.0193 | yes |
| state | 27 | 19325 | 2410 | 0.1247 | 0.7531 | 0.3211 | 0.8361 | 0.0279 | -0.0156 | 0.0072 | -0.0108 | 0.0436 | 0.0268 | yes |
| state | 28 | 5580 | 1190 | 0.2133 | 0.7569 | 0.4622 | 0.8176 | 0.0576 | 0.0730 | 0.0110 | 0.1302 | 0.0251 | 0.0565 | yes |
| state | 29 | 16797 | 2516 | 0.1498 | 0.7547 | 0.3603 | 0.8378 | 0.0081 | 0.0095 | 0.0087 | 0.0284 | 0.0453 | 0.0070 | yes |
| state | 3 | 10695 | 1566 | 0.1464 | 0.7709 | 0.3556 | 0.8710 | 0.0235 | 0.0061 | 0.0250 | 0.0236 | 0.0785 | 0.0224 | yes |
| state | 30 | 957 | 207 | 0.2163 | 0.8149 | 0.5128 | 0.8841 | 0.0464 | 0.0760 | 0.0690 | 0.1808 | 0.0915 | 0.0453 | yes |
| state | 32 | 6066 | 1580 | 0.2605 | 0.7906 | 0.5435 | 0.9165 | 0.0480 | 0.1202 | 0.0447 | 0.2116 | 0.1239 | 0.0469 | yes |
| state | 33 | 13732 | 3243 | 0.2362 | 0.7513 | 0.4729 | 0.8560 | 0.0642 | 0.0959 | 0.0054 | 0.1410 | 0.0635 | 0.0631 | yes |
| state | 34 | 2061 | 515 | 0.2499 | 0.8069 | 0.5588 | 0.8893 | 0.0563 | 0.1096 | 0.0610 | 0.2268 | 0.0968 | 0.0552 | yes |
| state | 35 | 1612 | 299 | 0.1855 | 0.7501 | 0.4353 | 0.8294 | 0.0223 | 0.0452 | 0.0041 | 0.1033 | 0.0369 | 0.0212 | yes |
| state | 36 | 13726 | 2145 | 0.1563 | 0.7569 | 0.3691 | 0.8224 | 0.0091 | 0.0160 | 0.0110 | 0.0371 | 0.0299 | 0.0080 | yes |
| state | 37 | 1215 | 198 | 0.1630 | 0.6568 | 0.2423 | 0.6212 | 0.0641 | 0.0227 | -0.0891 | -0.0897 | -0.1713 | 0.0630 | yes |
| state | 5 | 6770 | 804 | 0.1188 | 0.7493 | 0.2951 | 0.8234 | 0.0269 | -0.0215 | 0.0034 | -0.0369 | 0.0309 | 0.0258 | yes |
| state | 6 | 11866 | 1699 | 0.1432 | 0.7561 | 0.3309 | 0.8240 | 0.0093 | 0.0029 | 0.0102 | -0.0011 | 0.0315 | 0.0082 | yes |
| state | 7 | 4918 | 620 | 0.1261 | 0.8162 | 0.4060 | 0.8919 | 0.0365 | -0.0142 | 0.0703 | 0.0740 | 0.0994 | 0.0354 | yes |
| state | 8 | 20150 | 1729 | 0.0858 | 0.7516 | 0.2224 | 0.7941 | 0.0386 | -0.0545 | 0.0057 | -0.1096 | 0.0016 | 0.0375 | yes |
| state | 9 | 44473 | 5113 | 0.1150 | 0.7371 | 0.2754 | 0.7512 | 0.0097 | -0.0253 | -0.0088 | -0.0566 | -0.0413 | 0.0086 | yes |

### Fairness audit subgroups

This run does not report audit-only subgroup rows; the fairness variant reports religion and social group.

- Describes the household head, not necessarily the respondent.

Subgroup gaps are descriptive, not causal. Excluding a protected attribute does not by itself make a model fair; the fairness variant tests what including religion and social group changes.

## Feature importance

| Feature | Importance |
|---|---:|
| age | 3268935.8818 |
| bp_medication | 348505.9968 |
| systolic_mean | 328991.4992 |
| wealth | 81468.6612 |
| diastolic_mean | 70252.6925 |
| education | 61875.4175 |
| urban | 25280.0751 |
| female | 16165.0260 |
| bp_readings | 9694.3380 |
| bp_high | 9023.9965 |
| tobacco_any | 4303.5194 |
| alcohol_any | 3512.6053 |

## Explainability

Method: **shap** (fallback: no).


## Limitations

- Random blood glucose is a screening indicator, not a diagnosis.
- The target includes self-reported diagnosis and medication, so the model can learn access-to-care patterns.
- Subgroup gaps are descriptive, not causal.
- This model is not for insurance, employment, or care-allocation decisions.
