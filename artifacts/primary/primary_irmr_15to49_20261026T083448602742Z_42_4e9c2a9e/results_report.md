# primary_irmr_15to49 results

Run: `primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e`

This is a research and portfolio model. It estimates the probability of elevated random blood glucose or diagnosed diabetes; it is not a diagnosis.

## Run role

Primary IR/MR 15-49 reference model with the rich anthropometry, diet, and BP feature set. This is the deployed screener; the PR runs are sensitivity analyses.

- Sample: `age_15_49`
- Feature variant: `default`

## Sample

| Split | Rows |
|---|---:|
| Development | 619915 |
| Test | 154946 |
| Total used | 774861 |

## Model selection

| Family | Mean PR-AUC | Mean Brier | Mean log loss | Selected |
|---|---:|---:|---:|:---:|
| prevalence | 0.0832 | 0.0763 | 0.2865 | no |
| logistic | 0.1998 | 0.2163 | 0.6231 | no |
| lightgbm | 0.2074 | 0.2109 | 0.6128 | yes |

## Test performance at the chosen threshold

Threshold: **0.0549** (target sensitivity 0.80)

| Metric | Value |
|---|---:|
| n | 154946 |
| positives | 12943 |
| prevalence | 0.0835 |
| ROC-AUC | 0.7099 |
| PR-AUC | 0.2035 |
| Brier | 0.0721 |
| Log loss | 0.2638 |
| Sensitivity @ 80% specificity | 0.4831 |
| Sensitivity | 0.8096 |
| Specificity | 0.4558 |
| PPV | 0.1194 |
| NPV | 0.9633 |
| F1 | 0.2081 |
| Calibration slope | 0.9842 |
| Calibration intercept | -0.0331 |
| ECE | 0.0014 |

### 95% cluster-bootstrap intervals

| Metric | Point | Low | High |
|---|---:|---:|---:|
| roc_auc | 0.7099 | 0.7046 | 0.7149 |
| pr_auc | 0.2035 | 0.1970 | 0.2105 |
| brier | 0.0721 | 0.0706 | 0.0736 |

### Survey-weighted metrics

| Metric | Value |
|---|---:|
| n | 154946 |
| positives | 12943 |
| prevalence | 0.0935 |
| ROC-AUC | 0.7129 |
| PR-AUC | 0.2265 |
| Brier | 0.0793 |
| Log loss | 0.2843 |
| Sensitivity @ 80% specificity | 0.4962 |
| Sensitivity | 0.8141 |
| Specificity | 0.4517 |
| PPV | 0.1328 |
| NPV | 0.9593 |
| F1 | 0.2284 |
| Calibration slope | 0.9831 |
| Calibration intercept | 0.0633 |
| ECE | 0.0078 |

### Fact-sheet target (reported separately)

Column: `target_fact_sheet`

| Metric | Value |
|---|---:|
| n | 154927 |
| positives | 11658 |
| prevalence | 0.0752 |
| ROC-AUC | 0.7168 |
| PR-AUC | 0.1946 |
| Brier | 0.0656 |
| Log loss | 0.2445 |
| Sensitivity @ 80% specificity | 0.4981 |
| Sensitivity | 0.8183 |
| Specificity | 0.4542 |
| PPV | 0.1087 |
| NPV | 0.9685 |
| F1 | 0.1920 |
| Calibration slope | 1.0171 |
| Calibration intercept | -0.0826 |
| ECE | 0.0083 |

## Calibration

Calibrator selected on out-of-fold development predictions: **isotonic**.

| Method | Brier | Log loss |
|---|---:|---:|
| isotonic | 0.0717 | 0.2624 |
| platt | 0.0717 | 0.2626 |

## Subgroups

Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged `Reliable = no` and must not be used to claim fairness.

State rows are reported only where the test set has at least 500 respondents.

### Sex-specific metrics (mandatory)

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sex | female | 137821 | 11131 | 0.0808 | 0.7074 | 0.1978 | 0.7973 | 0.0020 | -0.0028 | -0.0025 | -0.0056 | -0.0123 | 0.0006 | yes |
| sex | male | 17125 | 1812 | 0.1058 | 0.7152 | 0.2364 | 0.8852 | 0.0065 | 0.0223 | 0.0053 | 0.0329 | 0.0756 | 0.0050 | yes |

### Age-band-specific metrics (mandatory)

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| age_band | 15-24 | 51484 | 1853 | 0.0360 | 0.5893 | 0.0542 | 0.1592 | 0.0005 | -0.0475 | -0.1205 | -0.1493 | -0.6504 | -0.0009 | yes |
| age_band | 25-34 | 47072 | 3260 | 0.0693 | 0.6291 | 0.1167 | 0.7316 | 0.0012 | -0.0143 | -0.0808 | -0.0868 | -0.0780 | -0.0002 | yes |
| age_band | 35-44 | 38422 | 4801 | 0.1250 | 0.6495 | 0.2251 | 0.9935 | 0.0029 | 0.0414 | -0.0604 | 0.0216 | 0.1839 | 0.0014 | yes |
| age_band | 45-49 | 17968 | 3029 | 0.1686 | 0.6634 | 0.2949 | 1.0000 | 0.0076 | 0.0850 | -0.0465 | 0.0914 | 0.1904 | 0.0062 | yes |

### Other subgroups

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wealth | 1 | 31957 | 2172 | 0.0680 | 0.6802 | 0.1418 | 0.7523 | 0.0017 | -0.0156 | -0.0296 | -0.0617 | -0.0573 | 0.0002 | yes |
| wealth | 2 | 34693 | 2653 | 0.0765 | 0.6863 | 0.1682 | 0.7655 | 0.0023 | -0.0071 | -0.0236 | -0.0353 | -0.0441 | 0.0009 | yes |
| wealth | 3 | 32890 | 2799 | 0.0851 | 0.7071 | 0.1960 | 0.8174 | 0.0025 | 0.0016 | -0.0027 | -0.0075 | 0.0078 | 0.0010 | yes |
| wealth | 4 | 29609 | 2838 | 0.0958 | 0.7256 | 0.2372 | 0.8439 | 0.0036 | 0.0123 | 0.0157 | 0.0338 | 0.0343 | 0.0021 | yes |
| wealth | 5 | 25797 | 2481 | 0.0962 | 0.7349 | 0.2550 | 0.8589 | 0.0049 | 0.0126 | 0.0250 | 0.0516 | 0.0493 | 0.0034 | yes |
| residence | rural | 117394 | 9213 | 0.0785 | 0.6984 | 0.1829 | 0.7908 | 0.0022 | -0.0051 | -0.0115 | -0.0206 | -0.0188 | 0.0008 | yes |
| residence | urban | 37552 | 3730 | 0.0993 | 0.7332 | 0.2505 | 0.8560 | 0.0036 | 0.0158 | 0.0233 | 0.0470 | 0.0464 | 0.0021 | yes |
| education | 0 | 33535 | 3282 | 0.0979 | 0.6757 | 0.1958 | 0.9119 | 0.0026 | 0.0143 | -0.0342 | -0.0076 | 0.1023 | 0.0012 | yes |
| education | 1 | 17792 | 1773 | 0.0997 | 0.6983 | 0.2266 | 0.8782 | 0.0021 | 0.0161 | -0.0115 | 0.0231 | 0.0685 | 0.0007 | yes |
| education | 2 | 81103 | 6249 | 0.0771 | 0.7207 | 0.2048 | 0.7587 | 0.0021 | -0.0065 | 0.0108 | 0.0013 | -0.0509 | 0.0006 | yes |
| education | 3 | 22516 | 1639 | 0.0728 | 0.7033 | 0.1859 | 0.7248 | 0.0014 | -0.0107 | -0.0065 | -0.0175 | -0.0848 | -0.0001 | yes |
| state | 1 | 4837 | 238 | 0.0492 | 0.7048 | 0.1435 | 0.8025 | 0.0315 | -0.0343 | -0.0051 | -0.0599 | -0.0071 | 0.0300 | yes |
| state | 10 | 9848 | 857 | 0.0870 | 0.6841 | 0.2065 | 0.7281 | 0.0153 | 0.0035 | -0.0258 | 0.0030 | -0.0815 | 0.0138 | yes |
| state | 11 | 664 | 60 | 0.0904 | 0.7457 | 0.2267 | 0.8833 | 0.0180 | 0.0068 | 0.0358 | 0.0232 | 0.0737 | 0.0166 | yes |
| state | 12 | 3699 | 318 | 0.0860 | 0.6804 | 0.1669 | 0.8333 | 0.0077 | 0.0024 | -0.0294 | -0.0366 | 0.0237 | 0.0063 | yes |
| state | 13 | 1683 | 110 | 0.0654 | 0.7744 | 0.2534 | 0.8727 | 0.0118 | -0.0182 | 0.0645 | 0.0499 | 0.0631 | 0.0104 | yes |
| state | 14 | 1951 | 163 | 0.0835 | 0.7301 | 0.2100 | 0.8650 | 0.0139 | 0.0000 | 0.0202 | 0.0065 | 0.0554 | 0.0124 | yes |
| state | 15 | 1708 | 159 | 0.0931 | 0.7080 | 0.2438 | 0.8742 | 0.0125 | 0.0096 | -0.0019 | 0.0403 | 0.0646 | 0.0111 | yes |
| state | 16 | 1381 | 193 | 0.1398 | 0.6957 | 0.2745 | 0.8497 | 0.0491 | 0.0562 | -0.0142 | 0.0710 | 0.0401 | 0.0476 | yes |
| state | 17 | 3016 | 252 | 0.0836 | 0.6818 | 0.1808 | 0.7659 | 0.0111 | 0.0000 | -0.0281 | -0.0226 | -0.0438 | 0.0097 | yes |
| state | 18 | 7493 | 719 | 0.0960 | 0.7013 | 0.2292 | 0.8387 | 0.0084 | 0.0124 | -0.0086 | 0.0258 | 0.0290 | 0.0069 | yes |
| state | 19 | 4813 | 682 | 0.1417 | 0.6908 | 0.2741 | 0.8387 | 0.0439 | 0.0582 | -0.0191 | 0.0706 | 0.0291 | 0.0425 | yes |
| state | 2 | 1758 | 127 | 0.0722 | 0.7047 | 0.1750 | 0.8110 | 0.0176 | -0.0113 | -0.0052 | -0.0284 | 0.0014 | 0.0162 | yes |
| state | 20 | 5593 | 409 | 0.0731 | 0.6744 | 0.1369 | 0.7702 | 0.0050 | -0.0104 | -0.0355 | -0.0666 | -0.0395 | 0.0036 | yes |
| state | 21 | 5794 | 551 | 0.0951 | 0.6977 | 0.2103 | 0.8512 | 0.0039 | 0.0116 | -0.0122 | 0.0068 | 0.0416 | 0.0024 | yes |
| state | 22 | 5509 | 392 | 0.0712 | 0.6971 | 0.1826 | 0.7959 | 0.0069 | -0.0124 | -0.0128 | -0.0209 | -0.0137 | 0.0054 | yes |
| state | 23 | 10179 | 663 | 0.0651 | 0.6770 | 0.1425 | 0.7179 | 0.0053 | -0.0184 | -0.0329 | -0.0610 | -0.0917 | 0.0038 | yes |
| state | 24 | 7135 | 678 | 0.0950 | 0.6803 | 0.1940 | 0.7596 | 0.0205 | 0.0115 | -0.0296 | -0.0095 | -0.0500 | 0.0191 | yes |
| state | 25 | 747 | 75 | 0.1004 | 0.6825 | 0.1899 | 0.7867 | 0.0168 | 0.0169 | -0.0274 | -0.0135 | -0.0230 | 0.0153 | yes |
| state | 27 | 7409 | 502 | 0.0678 | 0.7120 | 0.1670 | 0.8108 | 0.0119 | -0.0158 | 0.0022 | -0.0365 | 0.0011 | 0.0105 | yes |
| state | 28 | 2367 | 282 | 0.1191 | 0.7310 | 0.2876 | 0.8936 | 0.0152 | 0.0356 | 0.0211 | 0.0841 | 0.0840 | 0.0137 | yes |
| state | 29 | 6911 | 600 | 0.0868 | 0.7058 | 0.2110 | 0.8350 | 0.0043 | 0.0033 | -0.0041 | 0.0075 | 0.0254 | 0.0029 | yes |
| state | 3 | 4653 | 362 | 0.0778 | 0.7394 | 0.2400 | 0.8646 | 0.0261 | -0.0057 | 0.0296 | 0.0365 | 0.0550 | 0.0247 | yes |
| state | 32 | 2172 | 277 | 0.1275 | 0.7380 | 0.2868 | 0.9458 | 0.0114 | 0.0440 | 0.0281 | 0.0833 | 0.1362 | 0.0100 | yes |
| state | 33 | 5429 | 739 | 0.1361 | 0.7057 | 0.2856 | 0.8904 | 0.0270 | 0.0526 | -0.0042 | 0.0821 | 0.0808 | 0.0255 | yes |
| state | 34 | 832 | 103 | 0.1238 | 0.7500 | 0.3011 | 0.9417 | 0.0123 | 0.0403 | 0.0401 | 0.0977 | 0.1321 | 0.0108 | yes |
| state | 35 | 656 | 74 | 0.1128 | 0.6874 | 0.2661 | 0.9054 | 0.0198 | 0.0293 | -0.0225 | 0.0626 | 0.0958 | 0.0184 | yes |
| state | 36 | 5835 | 546 | 0.0936 | 0.7258 | 0.2454 | 0.8553 | 0.0075 | 0.0100 | 0.0159 | 0.0419 | 0.0457 | 0.0061 | yes |
| state | 37 | 530 | 78 | 0.1472 | 0.7050 | 0.2882 | 0.7821 | 0.0707 | 0.0636 | -0.0049 | 0.0847 | -0.0276 | 0.0693 | yes |
| state | 5 | 3049 | 205 | 0.0672 | 0.7346 | 0.1782 | 0.8683 | 0.0170 | -0.0163 | 0.0247 | -0.0253 | 0.0587 | 0.0156 | yes |
| state | 6 | 5183 | 433 | 0.0835 | 0.7243 | 0.2109 | 0.8314 | 0.0066 | 0.0000 | 0.0144 | 0.0075 | 0.0218 | 0.0052 | yes |
| state | 7 | 2275 | 152 | 0.0668 | 0.7883 | 0.2593 | 0.9079 | 0.0381 | -0.0167 | 0.0784 | 0.0558 | 0.0983 | 0.0366 | yes |
| state | 8 | 9066 | 409 | 0.0451 | 0.6990 | 0.1221 | 0.7164 | 0.0192 | -0.0384 | -0.0108 | -0.0813 | -0.0932 | 0.0178 | yes |
| state | 9 | 19986 | 1452 | 0.0727 | 0.6947 | 0.1701 | 0.7280 | 0.0020 | -0.0109 | -0.0151 | -0.0334 | -0.0817 | 0.0005 | yes |

### Fairness audit subgroups

Religion and social group are audit-only here: they are reported to test the feature-inclusion decision, not deployed as default features.

- Reported for the fairness audit only; not a deployed model feature.

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| religion | 1 | 117973 | 9819 | 0.0832 | 0.7080 | 0.2008 | 0.8037 | 0.0014 | -0.0003 | -0.0018 | -0.0027 | -0.0059 | -0.0001 | yes |
| religion | 2 | 18582 | 1599 | 0.0861 | 0.7204 | 0.2223 | 0.8155 | 0.0027 | 0.0025 | 0.0105 | 0.0188 | 0.0059 | 0.0012 | yes |
| religion | 3 | 11156 | 952 | 0.0853 | 0.7083 | 0.2025 | 0.8403 | 0.0046 | 0.0018 | -0.0016 | -0.0009 | 0.0307 | 0.0031 | yes |
| religion | 4 | 3373 | 273 | 0.0809 | 0.7394 | 0.2539 | 0.8571 | 0.0191 | -0.0026 | 0.0295 | 0.0505 | 0.0475 | 0.0177 | yes |
| religion | 5 | 1925 | 119 | 0.0618 | 0.7257 | 0.1462 | 0.8655 | 0.0260 | -0.0217 | 0.0158 | -0.0573 | 0.0559 | 0.0246 | yes |
| religion | 6 | 206 | 27 | 0.1311 | 0.6425 | 0.1815 | 0.8148 | 0.0664 | 0.0475 | -0.0674 | -0.0219 | 0.0052 | 0.0649 | no |
| religion | 7 | 5 | 0 | 0.0000 | nan | nan | nan | 0.0448 | -0.0835 | nan | nan | nan | 0.0433 | no |
| religion | 8 | 18 | 3 | 0.1667 | 0.6333 | 0.4881 | 0.6667 | 0.1234 | 0.0831 | -0.0765 | 0.2846 | -0.1430 | 0.1220 | no |
| religion | 9 | 69 | 6 | 0.0870 | 0.8241 | 0.3621 | 1.0000 | 0.0662 | 0.0034 | 0.1142 | 0.1586 | 0.1904 | 0.0647 | no |
| religion | 96 | 1639 | 145 | 0.0885 | 0.6818 | 0.2063 | 0.8000 | 0.0058 | 0.0049 | -0.0281 | 0.0028 | -0.0096 | 0.0044 | yes |
| social_group | 1 | 30208 | 2401 | 0.0795 | 0.7215 | 0.2102 | 0.8030 | 0.0036 | -0.0040 | 0.0116 | 0.0067 | -0.0066 | 0.0021 | yes |
| social_group | 2 | 28548 | 2124 | 0.0744 | 0.6863 | 0.1666 | 0.7877 | 0.0026 | -0.0091 | -0.0235 | -0.0369 | -0.0220 | 0.0011 | yes |
| social_group | 3 | 59872 | 5088 | 0.0850 | 0.7091 | 0.2040 | 0.8070 | 0.0024 | 0.0014 | -0.0008 | 0.0006 | -0.0026 | 0.0009 | yes |
| social_group | 4 | 28461 | 2537 | 0.0891 | 0.7132 | 0.2170 | 0.8305 | 0.0044 | 0.0056 | 0.0033 | 0.0135 | 0.0209 | 0.0029 | yes |
| social_group | missing | 7857 | 793 | 0.1009 | 0.7195 | 0.2372 | 0.8386 | 0.0118 | 0.0174 | 0.0097 | 0.0337 | 0.0290 | 0.0103 | yes |

Subgroup gaps are descriptive, not causal. Excluding a protected attribute does not by itself make a model fair; the fairness variant tests what including religion and social group changes.

## Feature importance

| Feature | Importance |
|---|---:|
| age | 861309.7369 |
| waist_cm | 183183.2051 |
| bmi | 157881.4857 |
| systolic_mean | 87636.6588 |
| bp_medication | 48909.6484 |
| waist_hip_ratio | 33831.0063 |
| diet_fish | 33090.6392 |
| diastolic_mean | 27511.8595 |
| height_cm | 17384.7635 |
| marital_status | 16371.3920 |
| hip_cm | 12727.1352 |
| weight_kg | 11733.8864 |
| female | 9138.4876 |
| wealth | 5674.9841 |
| bp_readings | 5301.4394 |

## Explainability

Method: **shap** (fallback: no).


## Limitations

- Random blood glucose is a screening indicator, not a diagnosis.
- The target includes self-reported diagnosis and medication, so the model can learn access-to-care patterns.
- Subgroup gaps are descriptive, not causal.
- This model is not for insurance, employment, or care-allocation decisions.
