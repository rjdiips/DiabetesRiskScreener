# pr_15to49_bp results

Run: `pr_15to49_bp_20261026T084051779902Z_42_96adb799`

This is a research and portfolio model. It estimates the probability of elevated random blood glucose or diagnosed diabetes; it is not a diagnosis.

## Run role

Age-matched comparison to pr_15plus_bp: the same PR cluster partition and 12-feature BP set, filtered to adults aged 15-49. The sample filter and the age-band reporting differ; the feature set, partition, and modelling logic are identical, so the result isolates the effect of the smaller feature set.

- Sample: `age_15_49`
- Compared with: `pr_15plus_bp`
- Feature variant: `default`

## Sample

| Split | Rows |
|---|---:|
| Development | 1032418 |
| Test | 257801 |
| Total used | 1290219 |

## Model selection

| Family | Mean PR-AUC | Mean Brier | Mean log loss | Selected |
|---|---:|---:|---:|:---:|
| prevalence | 0.0875 | 0.0798 | 0.2967 | no |
| logistic | 0.1869 | 0.2205 | 0.6326 | no |
| lightgbm | 0.1919 | 0.2189 | 0.6280 | yes |

## Test performance at the chosen threshold

Threshold: **0.0599** (target sensitivity 0.80)

| Metric | Value |
|---|---:|
| n | 257801 |
| positives | 22582 |
| prevalence | 0.0876 |
| ROC-AUC | 0.6979 |
| PR-AUC | 0.1900 |
| Brier | 0.0761 |
| Log loss | 0.2761 |
| Sensitivity @ 80% specificity | 0.4425 |
| Sensitivity | 0.7958 |
| Specificity | 0.4708 |
| PPV | 0.1262 |
| NPV | 0.9600 |
| F1 | 0.2178 |
| Calibration slope | 0.9946 |
| Calibration intercept | -0.0114 |
| ECE | 0.0006 |

### 95% cluster-bootstrap intervals

| Metric | Point | Low | High |
|---|---:|---:|---:|
| roc_auc | 0.6979 | 0.6939 | 0.7020 |
| pr_auc | 0.1900 | 0.1849 | 0.1951 |
| brier | 0.0761 | 0.0748 | 0.0773 |

### Survey-weighted metrics

| Metric | Value |
|---|---:|
| n | 257801 |
| positives | 22582 |
| prevalence | 0.0965 |
| ROC-AUC | 0.6955 |
| PR-AUC | 0.2064 |
| Brier | 0.0828 |
| Log loss | 0.2955 |
| Sensitivity @ 80% specificity | 0.4432 |
| Sensitivity | 0.7886 |
| Specificity | 0.4768 |
| PPV | 0.1387 |
| NPV | 0.9548 |
| F1 | 0.2359 |
| Calibration slope | 0.9765 |
| Calibration intercept | 0.0566 |
| ECE | 0.0085 |

### Fact-sheet target (reported separately)

Column: `target_fact_sheet`

| Metric | Value |
|---|---:|
| n | 257765 |
| positives | 20669 |
| prevalence | 0.0802 |
| ROC-AUC | 0.7042 |
| PR-AUC | 0.1811 |
| Brier | 0.0703 |
| Log loss | 0.2591 |
| Sensitivity @ 80% specificity | 0.4555 |
| Sensitivity | 0.8048 |
| Specificity | 0.4695 |
| PPV | 0.1168 |
| NPV | 0.9650 |
| F1 | 0.2040 |
| Calibration slope | 1.0275 |
| Calibration intercept | -0.0430 |
| ECE | 0.0074 |

## Calibration

Calibrator selected on out-of-fold development predictions: **isotonic**.

| Method | Brier | Log loss |
|---|---:|---:|
| isotonic | 0.0760 | 0.2756 |
| platt | 0.0760 | 0.2757 |

## Subgroups

Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged `Reliable = no` and must not be used to claim fairness.

State rows are reported only where the test set has at least 500 respondents.

### Sex-specific metrics (mandatory)

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sex | female | 139586 | 11247 | 0.0806 | 0.6883 | 0.1738 | 0.7591 | 0.0016 | -0.0070 | -0.0096 | -0.0161 | -0.0366 | 0.0009 | yes |
| sex | male | 118215 | 11335 | 0.0959 | 0.7043 | 0.2049 | 0.8321 | 0.0008 | 0.0083 | 0.0063 | 0.0149 | 0.0363 | 0.0002 | yes |

### Age-band-specific metrics (mandatory)

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| age_band | 15-24 | 85824 | 3096 | 0.0361 | 0.5793 | 0.0490 | 0.0707 | 0.0001 | -0.0515 | -0.1186 | -0.1410 | -0.7250 | -0.0005 | yes |
| age_band | 25-34 | 77588 | 5613 | 0.0723 | 0.5928 | 0.1036 | 0.6921 | 0.0021 | -0.0153 | -0.1051 | -0.0864 | -0.1036 | 0.0014 | yes |
| age_band | 35-44 | 64319 | 8523 | 0.1325 | 0.6127 | 0.2016 | 0.9993 | 0.0030 | 0.0449 | -0.0852 | 0.0116 | 0.2035 | 0.0024 | yes |
| age_band | 45-49 | 30070 | 5350 | 0.1779 | 0.6156 | 0.2701 | 0.9998 | 0.0042 | 0.0903 | -0.0823 | 0.0801 | 0.2040 | 0.0035 | yes |

### Other subgroups

| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wealth | 1 | 52840 | 3884 | 0.0735 | 0.6659 | 0.1330 | 0.7276 | 0.0009 | -0.0141 | -0.0321 | -0.0570 | -0.0682 | 0.0002 | yes |
| wealth | 2 | 57469 | 4589 | 0.0799 | 0.6787 | 0.1582 | 0.7692 | 0.0009 | -0.0077 | -0.0193 | -0.0318 | -0.0265 | 0.0003 | yes |
| wealth | 3 | 55078 | 4931 | 0.0895 | 0.6935 | 0.1844 | 0.7966 | 0.0028 | 0.0019 | -0.0044 | -0.0056 | 0.0008 | 0.0021 | yes |
| wealth | 4 | 49585 | 4875 | 0.0983 | 0.7102 | 0.2171 | 0.8279 | 0.0018 | 0.0107 | 0.0123 | 0.0271 | 0.0321 | 0.0012 | yes |
| wealth | 5 | 42829 | 4303 | 0.1005 | 0.7257 | 0.2370 | 0.8482 | 0.0026 | 0.0129 | 0.0278 | 0.0471 | 0.0525 | 0.0019 | yes |
| residence | rural | 194656 | 16071 | 0.0826 | 0.6883 | 0.1732 | 0.7775 | 0.0012 | -0.0050 | -0.0096 | -0.0168 | -0.0183 | 0.0006 | yes |
| residence | urban | 63145 | 6511 | 0.1031 | 0.7152 | 0.2271 | 0.8409 | 0.0033 | 0.0155 | 0.0173 | 0.0371 | 0.0451 | 0.0027 | yes |
| education | 0 | 42295 | 4278 | 0.1011 | 0.6451 | 0.1716 | 0.8974 | 0.0021 | 0.0136 | -0.0528 | -0.0184 | 0.1016 | 0.0015 | yes |
| education | 1 | 30324 | 3062 | 0.1010 | 0.6744 | 0.1953 | 0.8739 | 0.0010 | 0.0134 | -0.0236 | 0.0053 | 0.0782 | 0.0004 | yes |
| education | 2 | 144106 | 11928 | 0.0828 | 0.7132 | 0.1955 | 0.7612 | 0.0012 | -0.0048 | 0.0153 | 0.0055 | -0.0346 | 0.0006 | yes |
| education | 3 | 40975 | 3303 | 0.0806 | 0.6958 | 0.1857 | 0.7169 | 0.0012 | -0.0070 | -0.0021 | -0.0043 | -0.0788 | 0.0006 | yes |
| education | missing | 101 | 11 | 0.1089 | 0.8692 | 0.5681 | 0.7273 | 0.0607 | 0.0213 | 0.1713 | 0.3781 | -0.0685 | 0.0601 | no |
| state | 1 | 8432 | 430 | 0.0510 | 0.7033 | 0.1318 | 0.7860 | 0.0322 | -0.0366 | 0.0053 | -0.0582 | -0.0097 | 0.0316 | yes |
| state | 10 | 15373 | 1460 | 0.0950 | 0.6779 | 0.1932 | 0.6897 | 0.0221 | 0.0074 | -0.0200 | 0.0033 | -0.1060 | 0.0214 | yes |
| state | 11 | 1209 | 119 | 0.0984 | 0.6983 | 0.1862 | 0.8824 | 0.0161 | 0.0108 | 0.0004 | -0.0038 | 0.0866 | 0.0155 | yes |
| state | 12 | 6318 | 605 | 0.0958 | 0.6660 | 0.1667 | 0.8017 | 0.0110 | 0.0082 | -0.0319 | -0.0233 | 0.0059 | 0.0104 | yes |
| state | 13 | 2828 | 199 | 0.0704 | 0.7413 | 0.1803 | 0.8492 | 0.0184 | -0.0172 | 0.0434 | -0.0097 | 0.0535 | 0.0178 | yes |
| state | 14 | 3313 | 296 | 0.0893 | 0.7237 | 0.2039 | 0.8615 | 0.0123 | 0.0018 | 0.0258 | 0.0139 | 0.0657 | 0.0116 | yes |
| state | 15 | 2942 | 280 | 0.0952 | 0.7074 | 0.2146 | 0.8714 | 0.0135 | 0.0076 | 0.0095 | 0.0246 | 0.0757 | 0.0129 | yes |
| state | 16 | 2289 | 310 | 0.1354 | 0.6744 | 0.2430 | 0.8032 | 0.0463 | 0.0478 | -0.0235 | 0.0530 | 0.0075 | 0.0457 | yes |
| state | 17 | 5045 | 455 | 0.0902 | 0.6932 | 0.1919 | 0.7736 | 0.0089 | 0.0026 | -0.0047 | 0.0019 | -0.0221 | 0.0083 | yes |
| state | 18 | 12338 | 1246 | 0.1010 | 0.6905 | 0.2160 | 0.7897 | 0.0149 | 0.0134 | -0.0074 | 0.0261 | -0.0060 | 0.0143 | yes |
| state | 19 | 7946 | 1188 | 0.1495 | 0.6780 | 0.2708 | 0.7694 | 0.0607 | 0.0619 | -0.0199 | 0.0808 | -0.0264 | 0.0601 | yes |
| state | 2 | 2867 | 229 | 0.0799 | 0.6919 | 0.1985 | 0.8384 | 0.0209 | -0.0077 | -0.0060 | 0.0085 | 0.0427 | 0.0203 | yes |
| state | 20 | 9095 | 731 | 0.0804 | 0.6834 | 0.1554 | 0.7579 | 0.0020 | -0.0072 | -0.0145 | -0.0346 | -0.0379 | 0.0014 | yes |
| state | 21 | 9570 | 955 | 0.0998 | 0.6871 | 0.2097 | 0.7979 | 0.0129 | 0.0122 | -0.0108 | 0.0197 | 0.0021 | 0.0122 | yes |
| state | 22 | 9324 | 672 | 0.0721 | 0.6954 | 0.1661 | 0.7902 | 0.0137 | -0.0155 | -0.0026 | -0.0239 | -0.0056 | 0.0131 | yes |
| state | 23 | 17634 | 1259 | 0.0714 | 0.6773 | 0.1383 | 0.7768 | 0.0132 | -0.0162 | -0.0206 | -0.0517 | -0.0190 | 0.0126 | yes |
| state | 24 | 12256 | 1161 | 0.0947 | 0.6679 | 0.1738 | 0.7778 | 0.0087 | 0.0071 | -0.0301 | -0.0162 | -0.0180 | 0.0081 | yes |
| state | 25 | 1324 | 141 | 0.1065 | 0.6811 | 0.1950 | 0.8014 | 0.0207 | 0.0189 | -0.0168 | 0.0050 | 0.0057 | 0.0201 | yes |
| state | 27 | 13099 | 934 | 0.0713 | 0.6999 | 0.1639 | 0.8148 | 0.0198 | -0.0163 | 0.0020 | -0.0260 | 0.0190 | 0.0192 | yes |
| state | 28 | 3852 | 511 | 0.1327 | 0.7017 | 0.2657 | 0.8611 | 0.0354 | 0.0451 | 0.0038 | 0.0757 | 0.0653 | 0.0348 | yes |
| state | 29 | 11572 | 1020 | 0.0881 | 0.7033 | 0.2012 | 0.8206 | 0.0069 | 0.0005 | 0.0053 | 0.0112 | 0.0248 | 0.0063 | yes |
| state | 3 | 7537 | 619 | 0.0821 | 0.7248 | 0.2051 | 0.8530 | 0.0209 | -0.0055 | 0.0269 | 0.0151 | 0.0572 | 0.0202 | yes |
| state | 30 | 664 | 75 | 0.1130 | 0.7930 | 0.2983 | 0.9200 | 0.0374 | 0.0254 | 0.0951 | 0.1083 | 0.1242 | 0.0368 | yes |
| state | 32 | 3580 | 456 | 0.1274 | 0.7433 | 0.2960 | 0.9101 | 0.0204 | 0.0398 | 0.0454 | 0.1060 | 0.1143 | 0.0198 | yes |
| state | 33 | 8852 | 1281 | 0.1447 | 0.7100 | 0.2915 | 0.8517 | 0.0445 | 0.0571 | 0.0121 | 0.1015 | 0.0559 | 0.0439 | yes |
| state | 34 | 1381 | 184 | 0.1332 | 0.7642 | 0.3405 | 0.8967 | 0.0303 | 0.0456 | 0.0663 | 0.1505 | 0.1010 | 0.0297 | yes |
| state | 35 | 1113 | 131 | 0.1177 | 0.6817 | 0.2597 | 0.8550 | 0.0127 | 0.0301 | -0.0162 | 0.0697 | 0.0592 | 0.0120 | yes |
| state | 36 | 9660 | 955 | 0.0989 | 0.7209 | 0.2424 | 0.8366 | 0.0058 | 0.0113 | 0.0230 | 0.0524 | 0.0409 | 0.0051 | yes |
| state | 37 | 945 | 118 | 0.1249 | 0.6176 | 0.1773 | 0.7119 | 0.0478 | 0.0373 | -0.0803 | -0.0127 | -0.0839 | 0.0472 | yes |
| state | 5 | 4688 | 343 | 0.0732 | 0.7206 | 0.1898 | 0.8222 | 0.0186 | -0.0144 | 0.0227 | -0.0002 | 0.0264 | 0.0180 | yes |
| state | 6 | 8620 | 757 | 0.0878 | 0.7236 | 0.1945 | 0.8296 | 0.0050 | 0.0002 | 0.0256 | 0.0045 | 0.0338 | 0.0044 | yes |
| state | 7 | 3793 | 253 | 0.0667 | 0.7704 | 0.2205 | 0.8972 | 0.0327 | -0.0209 | 0.0725 | 0.0305 | 0.1015 | 0.0321 | yes |
| state | 8 | 14906 | 754 | 0.0506 | 0.7077 | 0.1210 | 0.7918 | 0.0297 | -0.0370 | 0.0097 | -0.0690 | -0.0040 | 0.0291 | yes |
| state | 9 | 32915 | 2401 | 0.0729 | 0.6847 | 0.1531 | 0.7293 | 0.0060 | -0.0146 | -0.0132 | -0.0369 | -0.0665 | 0.0053 | yes |

### Fairness audit subgroups

This run does not report audit-only subgroup rows; the fairness variant reports religion and social group.

- Describes the household head, not necessarily the respondent.

Subgroup gaps are descriptive, not causal. Excluding a protected attribute does not by itself make a model fair; the fairness variant tests what including religion and social group changes.

## Feature importance

| Feature | Importance |
|---|---:|
| age | 1557311.2003 |
| systolic_mean | 204574.5572 |
| bp_medication | 97026.6150 |
| diastolic_mean | 83237.2470 |
| wealth | 39161.1704 |
| education | 21795.3529 |
| bp_high | 19136.0060 |
| female | 18205.6240 |
| bp_readings | 11050.8235 |
| urban | 9975.9092 |
| tobacco_any | 5180.4767 |
| alcohol_any | 3061.9517 |

## Explainability

Method: **shap** (fallback: no).


## Limitations

- Random blood glucose is a screening indicator, not a diagnosis.
- The target includes self-reported diagnosis and medication, so the model can learn access-to-care patterns.
- Subgroup gaps are descriptive, not causal.
- This model is not for insurance, employment, or care-allocation decisions.
