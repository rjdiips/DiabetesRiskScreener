# Primary vs PR sensitivity model comparison

This report compares three runs on the same reporting axes. It does not rank them.

- **Primary IR/MR 15–49** (sample `age_15_49`, run `primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e`): 32 features.
- **PR 15–49** (sample `age_15_49`, run `pr_15to49_bp_20261026T084051779902Z_42_96adb799`): 12 features.
- **PR 15+** (sample `age_15_plus`, run `pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6`): 12 features.

> **Do not compare raw probabilities.** The samples and feature sets differ. The comparison is about discrimination, calibration, threshold behaviour, and subgroup gaps.

See the [PR sensitivity model-card addendum](pr_sensitivity_model_card_addendum.md) for intended use, sample and feature differences, the age gradient, label bias, and non-goals.

## What the three runs are

- The primary IR/MR 15–49 run is the reference model. It uses the rich questionnaire, anthropometry, diet, and BP feature set.
- The PR 15–49 run isolates the feature-set trade-off. It keeps the same adults aged 15–49 and otherwise uses the same PR cluster partition as the PR 15+ run, but replaces the primary feature set with the 12 BP-based features. The comparison with the primary model therefore shows what the smaller feature set changes, although the underlying recodes and population coverage still differ.
- The PR 15+ run adds older adults. It uses the same 12 features and the same cluster partition as the PR 15–49 run but retains adults aged 50+, where prevalence is higher. The difference between the two PR runs mostly reflects that population extension rather than a feature-set change.
- No run is universally better. Feature richness, population coverage, calibration, and threshold behaviour are separate trade-offs, and a higher ROC-AUC or PR-AUC does not by itself make a model preferable.
- Raw probabilities are not directly comparable across the three models. The comparison is about discrimination, calibration, and threshold behaviour on different samples and feature sets.

## Sample differences

| Run | Sample | Rows used | Rows after filters | Test rows | Test positives | Prevalence | Weighted prevalence | Female n | Male n | Female share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary IR/MR 15–49 | age_15_49 | 774861 | 774861 | 154946 | 12943 | 0.0835 | 0.0935 | 137821 | 17125 | 0.8895 |
| PR 15–49 | age_15_49 | 1290219 | 1290219 | 257801 | 22582 | 0.0876 | 0.0965 | 139586 | 118215 | 0.5414 |
| PR 15+ | age_15_plus | 1805697 | 1805697 | 360633 | 50596 | 0.1403 | 0.1543 | 192409 | 168224 | 0.5335 |

Rows used are the filtered run rows; test rows are the rows that produced the metrics. Composition and prevalence are reported on the test set, because that is the set every metric is computed on.

### Age composition

| Run | Age band | n | Test share | Prevalence |
|---|---|---:|---:|---:|
| Primary IR/MR 15–49 | 15-24 | 51484 | 0.3323 | 0.0360 |
| Primary IR/MR 15–49 | 25-34 | 47072 | 0.3038 | 0.0693 |
| Primary IR/MR 15–49 | 35-44 | 38422 | 0.2480 | 0.1250 |
| Primary IR/MR 15–49 | 45-49 | 17968 | 0.1160 | 0.1686 |
| PR 15–49 | 15-24 | 85824 | 0.3329 | 0.0361 |
| PR 15–49 | 25-34 | 77588 | 0.3010 | 0.0723 |
| PR 15–49 | 35-44 | 64319 | 0.2495 | 0.1325 |
| PR 15–49 | 45-49 | 30070 | 0.1166 | 0.1779 |
| PR 15+ | 15-24 | 85824 | 0.2380 | 0.0361 |
| PR 15+ | 25-34 | 77588 | 0.2151 | 0.0723 |
| PR 15+ | 35-44 | 64319 | 0.1784 | 0.1325 |
| PR 15+ | 45-49 | 30070 | 0.0834 | 0.1779 |
| PR 15+ | 50-59 | 46826 | 0.1298 | 0.2477 |
| PR 15+ | 60-69 | 35898 | 0.0995 | 0.2885 |
| PR 15+ | 70+ | 20108 | 0.0558 | 0.3014 |

- Rows used: primary 774861; PR 15–49 1290219; PR 15+ 1805697.
- Test prevalence: primary 0.0835; PR 15–49 0.0876; PR 15+ 0.1403. Higher PR prevalence is expected because the PR sample extends to older adults.
- Female share of the test set: primary 0.8895; PR 15–49 0.5414; PR 15+ 0.5335.
- Age bands reported: primary 15-24, 25-34, 35-44, 45-49; PR 15–49 15-24, 25-34, 35-44, 45-49; PR 15+ 15-24, 25-34, 35-44, 45-49, 50-59, 60-69, 70+.
- The primary run uses the IR/MR recodes; the PR runs use the PR household-member recode. The samples are therefore not nested. Their sex composition is shown in the sample table above.

## Feature-set differences

| Run | Feature count | Features |
|---|---:|---|
| Primary IR/MR 15–49 | 32 | `female`, `age`, `urban`, `wealth`, `education`, `marital_status`, `children_ever_born`, `height_cm`, `weight_kg`, `bmi`, `waist_cm`, `hip_cm`, `waist_hip_ratio`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication`, `tobacco_any`, `smoking_any`, `smokeless_any`, `alcohol_any`, `alcohol_frequency`, `diet_milk`, `diet_pulses`, `diet_green_leafy`, `diet_fruit`, `diet_eggs`, `diet_fish`, `diet_meat`, `diet_fried`, `diet_aerated` |
| PR 15–49 | 12 | `female`, `age`, `urban`, `wealth`, `education`, `tobacco_any`, `alcohol_any`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication` |
| PR 15+ | 12 | `female`, `age`, `urban`, `wealth`, `education`, `tobacco_any`, `alcohol_any`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication` |

- The primary model uses 32 features; the PR 15+ model uses 12. The PR 15–49 model uses 12.
- The PR feature set is built to be usable for every adult with a valid BP pair: it excludes anthropometry, diet, marital status, children ever born, the smoking/smokeless split, alcohol frequency, state, religion, and social group.
- The PR 15+ and PR 15–49 runs use exactly the same feature list; only the sample filter and age-band reporting differ.

### Features present only in the primary model

`marital_status`, `children_ever_born`, `height_cm`, `weight_kg`, `bmi`, `waist_cm`, `hip_cm`, `waist_hip_ratio`, `smoking_any`, `smokeless_any`, `alcohol_frequency`, `diet_milk`, `diet_pulses`, `diet_green_leafy`, `diet_fruit`, `diet_eggs`, `diet_fish`, `diet_meat`, `diet_fried`, `diet_aerated`

### Features present only in the PR models

—

## Overall discrimination, calibration, and threshold metrics

Each run selects its own calibration and screening threshold on its own out-of-fold predictions. The threshold columns therefore describe each model's operating point, not a shared cut-off applied to all three probabilities.

| Metric | Primary IR/MR 15–49 | PR 15–49 | PR 15+ |
|---|---:|---:|---:|
| Test rows | 154946 | 257801 | 360633 |
| Test positives | 12943 | 22582 | 50596 |
| Prevalence | 0.0835 | 0.0876 | 0.1403 |
| ROC-AUC | 0.7099 | 0.6979 | 0.7459 |
| PR-AUC | 0.2035 | 0.1900 | 0.3320 |
| Brier | 0.0721 | 0.0761 | 0.1077 |
| Log loss | 0.2638 | 0.2761 | 0.3568 |
| ECE | 0.0014 | 0.0006 | 0.0011 |
| Calibration slope | 0.9842 | 0.9946 | 0.9965 |
| Calibration intercept | -0.0331 | -0.0114 | -0.0043 |
| Sensitivity @ 80% specificity | 0.4831 | 0.4425 | 0.5223 |
| Threshold | 0.0549 | 0.0599 | 0.1040 |
| Sensitivity | 0.8096 | 0.7958 | 0.7925 |
| Specificity | 0.4558 | 0.4708 | 0.5614 |
| PPV | 0.1194 | 0.1262 | 0.2277 |
| NPV | 0.9633 | 0.9600 | 0.9431 |
| F1 | 0.2081 | 0.2178 | 0.3538 |
| Calibrator | isotonic | isotonic | isotonic |
| Selected model | lightgbm | lightgbm | lightgbm |

## Survey-weighted overall metrics

Survey-weighted rows use each run's survey weight and the threshold selected on that run's unweighted out-of-fold predictions.

| Metric | Primary IR/MR 15–49 | PR 15–49 | PR 15+ |
|---|---:|---:|---:|
| Test rows | 154946 | 257801 | 360633 |
| Test positives | 12943 | 22582 | 50596 |
| Prevalence | 0.0935 | 0.0965 | 0.1543 |
| ROC-AUC | 0.7129 | 0.6955 | 0.7471 |
| PR-AUC | 0.2265 | 0.2064 | 0.3595 |
| Brier | 0.0793 | 0.0828 | 0.1155 |
| Log loss | 0.2843 | 0.2955 | 0.3774 |
| ECE | 0.0078 | 0.0085 | 0.0105 |
| Calibration slope | 0.9831 | 0.9765 | 0.9764 |
| Calibration intercept | 0.0633 | 0.0566 | 0.0563 |
| Sensitivity @ 80% specificity | 0.4962 | 0.4432 | 0.5305 |
| Threshold | 0.0549 | 0.0599 | 0.1040 |
| Sensitivity | 0.8141 | 0.7886 | 0.7904 |
| Specificity | 0.4517 | 0.4768 | 0.5634 |
| PPV | 0.1328 | 0.1387 | 0.2483 |
| NPV | 0.9593 | 0.9548 | 0.9364 |
| F1 | 0.2284 | 0.2359 | 0.3779 |
| Calibrator | isotonic | isotonic | isotonic |
| Selected model | lightgbm | lightgbm | lightgbm |

## Differences between runs

Deltas are descriptive and are shown as left run minus right run. They are not evidence that one model is universally better; the samples and feature sets differ.

| Metric | PR 15–49 minus primary | PR 15+ minus primary | PR 15+ minus PR 15–49 |
|---|---:|---:|---:|
| Prevalence | +0.0041 | +0.0568 | +0.0527 |
| ROC-AUC | -0.0120 | +0.0360 | +0.0480 |
| PR-AUC | -0.0135 | +0.1285 | +0.1420 |
| Brier | +0.0040 | +0.0357 | +0.0317 |
| Log loss | +0.0123 | +0.0930 | +0.0807 |
| ECE | -0.0008 | -0.0003 | +0.0005 |
| Calibration slope | +0.0104 | +0.0123 | +0.0019 |
| Calibration intercept | +0.0217 | +0.0288 | +0.0071 |
| Sensitivity @ 80% specificity | -0.0406 | +0.0392 | +0.0798 |
| Threshold | +0.0051 | +0.0491 | +0.0441 |
| Sensitivity | -0.0139 | -0.0171 | -0.0033 |
| Specificity | +0.0150 | +0.1055 | +0.0905 |
| PPV | +0.0067 | +0.1083 | +0.1016 |
| NPV | -0.0033 | -0.0202 | -0.0169 |
| F1 | +0.0097 | +0.1456 | +0.1360 |

## Survey-weighted differences between runs

| Metric | PR 15–49 minus primary | PR 15+ minus primary | PR 15+ minus PR 15–49 |
|---|---:|---:|---:|
| Prevalence | +0.0030 | +0.0608 | +0.0578 |
| ROC-AUC | -0.0173 | +0.0342 | +0.0516 |
| PR-AUC | -0.0200 | +0.1330 | +0.1531 |
| Brier | +0.0035 | +0.0362 | +0.0327 |
| Log loss | +0.0112 | +0.0931 | +0.0819 |
| ECE | +0.0007 | +0.0027 | +0.0020 |
| Calibration slope | -0.0066 | -0.0067 | -0.0001 |
| Calibration intercept | -0.0067 | -0.0071 | -0.0003 |
| Sensitivity @ 80% specificity | -0.0531 | +0.0343 | +0.0874 |
| Threshold | +0.0051 | +0.0491 | +0.0441 |
| Sensitivity | -0.0256 | -0.0238 | +0.0018 |
| Specificity | +0.0251 | +0.1117 | +0.0866 |
| PPV | +0.0058 | +0.1155 | +0.1096 |
| NPV | -0.0045 | -0.0229 | -0.0184 |
| F1 | +0.0075 | +0.1495 | +0.1420 |

## Reliability curves

| Run | Bin lower | Bin upper | n | Mean predicted | Observed | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Primary IR/MR 15–49 | 0.0000 | 0.1000 | 112513 | 0.0510 | 0.0519 | 0.0009 |
| Primary IR/MR 15–49 | 0.1000 | 0.2000 | 33529 | 0.1389 | 0.1368 | 0.0022 |
| Primary IR/MR 15–49 | 0.2000 | 0.3000 | 6678 | 0.2490 | 0.2487 | 0.0003 |
| Primary IR/MR 15–49 | 0.3000 | 0.4000 | 1667 | 0.3576 | 0.3479 | 0.0097 |
| Primary IR/MR 15–49 | 0.4000 | 0.5000 | 273 | 0.4653 | 0.4103 | 0.0550 |
| Primary IR/MR 15–49 | 0.5000 | 0.6000 | 286 | 0.5326 | 0.5804 | 0.0478 |
| Primary IR/MR 15–49 | 0.6000 | 0.7000 | 0 | n/a | n/a | n/a |
| Primary IR/MR 15–49 | 0.7000 | 0.8000 | 0 | n/a | n/a | n/a |
| Primary IR/MR 15–49 | 0.8000 | 0.9000 | 0 | n/a | n/a | n/a |
| Primary IR/MR 15–49 | 0.9000 | 1.0000 | 0 | n/a | n/a | n/a |
| PR 15–49 | 0.0000 | 0.1000 | 169853 | 0.0523 | 0.0527 | 0.0004 |
| PR 15–49 | 0.1000 | 0.2000 | 74498 | 0.1361 | 0.1350 | 0.0010 |
| PR 15–49 | 0.2000 | 0.3000 | 10612 | 0.2345 | 0.2355 | 0.0010 |
| PR 15–49 | 0.3000 | 0.4000 | 1757 | 0.3398 | 0.3426 | 0.0028 |
| PR 15–49 | 0.4000 | 0.5000 | 1038 | 0.4305 | 0.4277 | 0.0027 |
| PR 15–49 | 0.5000 | 0.6000 | 43 | 0.5267 | 0.5349 | 0.0082 |
| PR 15–49 | 0.6000 | 0.7000 | 0 | n/a | n/a | n/a |
| PR 15–49 | 0.7000 | 0.8000 | 0 | n/a | n/a | n/a |
| PR 15–49 | 0.8000 | 0.9000 | 0 | n/a | n/a | n/a |
| PR 15–49 | 0.9000 | 1.0000 | 0 | n/a | n/a | n/a |
| PR 15+ | 0.0000 | 0.1000 | 173192 | 0.0533 | 0.0537 | 0.0005 |
| PR 15+ | 0.1000 | 0.2000 | 102138 | 0.1493 | 0.1485 | 0.0008 |
| PR 15+ | 0.2000 | 0.3000 | 53944 | 0.2450 | 0.2474 | 0.0023 |
| PR 15+ | 0.3000 | 0.4000 | 16678 | 0.3435 | 0.3399 | 0.0036 |
| PR 15+ | 0.4000 | 0.5000 | 9163 | 0.4376 | 0.4383 | 0.0007 |
| PR 15+ | 0.5000 | 0.6000 | 3775 | 0.5320 | 0.5330 | 0.0010 |
| PR 15+ | 0.6000 | 0.7000 | 1712 | 0.6365 | 0.6192 | 0.0173 |
| PR 15+ | 0.7000 | 0.8000 | 27 | 0.7804 | 0.7037 | 0.0767 |
| PR 15+ | 0.8000 | 0.9000 | 1 | 0.8125 | 1.0000 | 0.1875 |
| PR 15+ | 0.9000 | 1.0000 | 3 | 0.9091 | 1.0000 | 0.0909 |

## Sex-specific metrics

| Run | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Primary IR/MR 15–49 | female | 137821 | 11131 | 0.0808 | 0.7074 | 0.1978 | 0.7973 | 0.0020 | yes |
| Primary IR/MR 15–49 | male | 17125 | 1812 | 0.1058 | 0.7152 | 0.2364 | 0.8852 | 0.0065 | yes |
| PR 15–49 | female | 139586 | 11247 | 0.0806 | 0.6883 | 0.1738 | 0.7591 | 0.0016 | yes |
| PR 15–49 | male | 118215 | 11335 | 0.0959 | 0.7043 | 0.2049 | 0.8321 | 0.0008 | yes |
| PR 15+ | female | 192409 | 25204 | 0.1310 | 0.7463 | 0.3195 | 0.7594 | 0.0021 | yes |
| PR 15+ | male | 168224 | 25392 | 0.1509 | 0.7433 | 0.3439 | 0.8254 | 0.0014 | yes |

## Age-band-specific metrics

| Run | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | Sensitivity | ECE | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Primary IR/MR 15–49 | 15-24 | 51484 | 1853 | 0.0360 | 0.5893 | 0.0542 | 0.1592 | 0.0005 | yes |
| Primary IR/MR 15–49 | 25-34 | 47072 | 3260 | 0.0693 | 0.6291 | 0.1167 | 0.7316 | 0.0012 | yes |
| Primary IR/MR 15–49 | 35-44 | 38422 | 4801 | 0.1250 | 0.6495 | 0.2251 | 0.9935 | 0.0029 | yes |
| Primary IR/MR 15–49 | 45-49 | 17968 | 3029 | 0.1686 | 0.6634 | 0.2949 | 1.0000 | 0.0076 | yes |
| PR 15–49 | 15-24 | 85824 | 3096 | 0.0361 | 0.5793 | 0.0490 | 0.0707 | 0.0001 | yes |
| PR 15–49 | 25-34 | 77588 | 5613 | 0.0723 | 0.5928 | 0.1036 | 0.6921 | 0.0021 | yes |
| PR 15–49 | 35-44 | 64319 | 8523 | 0.1325 | 0.6127 | 0.2016 | 0.9993 | 0.0030 | yes |
| PR 15–49 | 45-49 | 30070 | 5350 | 0.1779 | 0.6156 | 0.2701 | 0.9998 | 0.0042 | yes |
| PR 15+ | 15-24 | 85824 | 3096 | 0.0361 | 0.5784 | 0.0487 | 0.0048 | 0.0001 | yes |
| PR 15+ | 25-34 | 77588 | 5613 | 0.0723 | 0.5906 | 0.1023 | 0.1484 | 0.0017 | yes |
| PR 15+ | 35-44 | 64319 | 8523 | 0.1325 | 0.6117 | 0.2008 | 0.7029 | 0.0032 | yes |
| PR 15+ | 45-49 | 30070 | 5350 | 0.1779 | 0.6152 | 0.2714 | 0.9806 | 0.0041 | yes |
| PR 15+ | 50-59 | 46826 | 11598 | 0.2477 | 0.6322 | 0.3699 | 0.9999 | 0.0044 | yes |
| PR 15+ | 60-69 | 35898 | 10356 | 0.2885 | 0.6466 | 0.4367 | 1.0000 | 0.0049 | yes |
| PR 15+ | 70+ | 20108 | 6060 | 0.3014 | 0.6314 | 0.4246 | 1.0000 | 0.0079 | yes |

## Subgroup gaps for the PR models

Each row reports the gap versus that run's own overall test set. The primary model's comparable subgroup table is in its own results report.

| Run | Subgroup | Group | n | Positives | Prevalence | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| PR 15–49 | sex | female | 139586 | 11247 | 0.0806 | -0.0070 | -0.0096 | -0.0161 | -0.0366 | 0.0009 | yes |
| PR 15–49 | sex | male | 118215 | 11335 | 0.0959 | 0.0083 | 0.0063 | 0.0149 | 0.0363 | 0.0002 | yes |
| PR 15–49 | age_band | 15-24 | 85824 | 3096 | 0.0361 | -0.0515 | -0.1186 | -0.1410 | -0.7250 | -0.0005 | yes |
| PR 15–49 | age_band | 25-34 | 77588 | 5613 | 0.0723 | -0.0153 | -0.1051 | -0.0864 | -0.1036 | 0.0014 | yes |
| PR 15–49 | age_band | 35-44 | 64319 | 8523 | 0.1325 | 0.0449 | -0.0852 | 0.0116 | 0.2035 | 0.0024 | yes |
| PR 15–49 | age_band | 45-49 | 30070 | 5350 | 0.1779 | 0.0903 | -0.0823 | 0.0801 | 0.2040 | 0.0035 | yes |
| PR 15–49 | wealth | 1 | 52840 | 3884 | 0.0735 | -0.0141 | -0.0321 | -0.0570 | -0.0682 | 0.0002 | yes |
| PR 15–49 | wealth | 2 | 57469 | 4589 | 0.0799 | -0.0077 | -0.0193 | -0.0318 | -0.0265 | 0.0003 | yes |
| PR 15–49 | wealth | 3 | 55078 | 4931 | 0.0895 | 0.0019 | -0.0044 | -0.0056 | 0.0008 | 0.0021 | yes |
| PR 15–49 | wealth | 4 | 49585 | 4875 | 0.0983 | 0.0107 | 0.0123 | 0.0271 | 0.0321 | 0.0012 | yes |
| PR 15–49 | wealth | 5 | 42829 | 4303 | 0.1005 | 0.0129 | 0.0278 | 0.0471 | 0.0525 | 0.0019 | yes |
| PR 15–49 | residence | rural | 194656 | 16071 | 0.0826 | -0.0050 | -0.0096 | -0.0168 | -0.0183 | 0.0006 | yes |
| PR 15–49 | residence | urban | 63145 | 6511 | 0.1031 | 0.0155 | 0.0173 | 0.0371 | 0.0451 | 0.0027 | yes |
| PR 15–49 | education | 0 | 42295 | 4278 | 0.1011 | 0.0136 | -0.0528 | -0.0184 | 0.1016 | 0.0015 | yes |
| PR 15–49 | education | 1 | 30324 | 3062 | 0.1010 | 0.0134 | -0.0236 | 0.0053 | 0.0782 | 0.0004 | yes |
| PR 15–49 | education | 2 | 144106 | 11928 | 0.0828 | -0.0048 | 0.0153 | 0.0055 | -0.0346 | 0.0006 | yes |
| PR 15–49 | education | 3 | 40975 | 3303 | 0.0806 | -0.0070 | -0.0021 | -0.0043 | -0.0788 | 0.0006 | yes |
| PR 15–49 | education | missing | 101 | 11 | 0.1089 | 0.0213 | 0.1713 | 0.3781 | -0.0685 | 0.0601 | no |
| PR 15–49 | state | 1 | 8432 | 430 | 0.0510 | -0.0366 | 0.0053 | -0.0582 | -0.0097 | 0.0316 | yes |
| PR 15–49 | state | 10 | 15373 | 1460 | 0.0950 | 0.0074 | -0.0200 | 0.0033 | -0.1060 | 0.0214 | yes |
| PR 15–49 | state | 11 | 1209 | 119 | 0.0984 | 0.0108 | 0.0004 | -0.0038 | 0.0866 | 0.0155 | yes |
| PR 15–49 | state | 12 | 6318 | 605 | 0.0958 | 0.0082 | -0.0319 | -0.0233 | 0.0059 | 0.0104 | yes |
| PR 15–49 | state | 13 | 2828 | 199 | 0.0704 | -0.0172 | 0.0434 | -0.0097 | 0.0535 | 0.0178 | yes |
| PR 15–49 | state | 14 | 3313 | 296 | 0.0893 | 0.0018 | 0.0258 | 0.0139 | 0.0657 | 0.0116 | yes |
| PR 15–49 | state | 15 | 2942 | 280 | 0.0952 | 0.0076 | 0.0095 | 0.0246 | 0.0757 | 0.0129 | yes |
| PR 15–49 | state | 16 | 2289 | 310 | 0.1354 | 0.0478 | -0.0235 | 0.0530 | 0.0075 | 0.0457 | yes |
| PR 15–49 | state | 17 | 5045 | 455 | 0.0902 | 0.0026 | -0.0047 | 0.0019 | -0.0221 | 0.0083 | yes |
| PR 15–49 | state | 18 | 12338 | 1246 | 0.1010 | 0.0134 | -0.0074 | 0.0261 | -0.0060 | 0.0143 | yes |
| PR 15–49 | state | 19 | 7946 | 1188 | 0.1495 | 0.0619 | -0.0199 | 0.0808 | -0.0264 | 0.0601 | yes |
| PR 15–49 | state | 2 | 2867 | 229 | 0.0799 | -0.0077 | -0.0060 | 0.0085 | 0.0427 | 0.0203 | yes |
| PR 15–49 | state | 20 | 9095 | 731 | 0.0804 | -0.0072 | -0.0145 | -0.0346 | -0.0379 | 0.0014 | yes |
| PR 15–49 | state | 21 | 9570 | 955 | 0.0998 | 0.0122 | -0.0108 | 0.0197 | 0.0021 | 0.0122 | yes |
| PR 15–49 | state | 22 | 9324 | 672 | 0.0721 | -0.0155 | -0.0026 | -0.0239 | -0.0056 | 0.0131 | yes |
| PR 15–49 | state | 23 | 17634 | 1259 | 0.0714 | -0.0162 | -0.0206 | -0.0517 | -0.0190 | 0.0126 | yes |
| PR 15–49 | state | 24 | 12256 | 1161 | 0.0947 | 0.0071 | -0.0301 | -0.0162 | -0.0180 | 0.0081 | yes |
| PR 15–49 | state | 25 | 1324 | 141 | 0.1065 | 0.0189 | -0.0168 | 0.0050 | 0.0057 | 0.0201 | yes |
| PR 15–49 | state | 27 | 13099 | 934 | 0.0713 | -0.0163 | 0.0020 | -0.0260 | 0.0190 | 0.0192 | yes |
| PR 15–49 | state | 28 | 3852 | 511 | 0.1327 | 0.0451 | 0.0038 | 0.0757 | 0.0653 | 0.0348 | yes |
| PR 15–49 | state | 29 | 11572 | 1020 | 0.0881 | 0.0005 | 0.0053 | 0.0112 | 0.0248 | 0.0063 | yes |
| PR 15–49 | state | 3 | 7537 | 619 | 0.0821 | -0.0055 | 0.0269 | 0.0151 | 0.0572 | 0.0202 | yes |
| PR 15–49 | state | 30 | 664 | 75 | 0.1130 | 0.0254 | 0.0951 | 0.1083 | 0.1242 | 0.0368 | yes |
| PR 15–49 | state | 32 | 3580 | 456 | 0.1274 | 0.0398 | 0.0454 | 0.1060 | 0.1143 | 0.0198 | yes |
| PR 15–49 | state | 33 | 8852 | 1281 | 0.1447 | 0.0571 | 0.0121 | 0.1015 | 0.0559 | 0.0439 | yes |
| PR 15–49 | state | 34 | 1381 | 184 | 0.1332 | 0.0456 | 0.0663 | 0.1505 | 0.1010 | 0.0297 | yes |
| PR 15–49 | state | 35 | 1113 | 131 | 0.1177 | 0.0301 | -0.0162 | 0.0697 | 0.0592 | 0.0120 | yes |
| PR 15–49 | state | 36 | 9660 | 955 | 0.0989 | 0.0113 | 0.0230 | 0.0524 | 0.0409 | 0.0051 | yes |
| PR 15–49 | state | 37 | 945 | 118 | 0.1249 | 0.0373 | -0.0803 | -0.0127 | -0.0839 | 0.0472 | yes |
| PR 15–49 | state | 5 | 4688 | 343 | 0.0732 | -0.0144 | 0.0227 | -0.0002 | 0.0264 | 0.0180 | yes |
| PR 15–49 | state | 6 | 8620 | 757 | 0.0878 | 0.0002 | 0.0256 | 0.0045 | 0.0338 | 0.0044 | yes |
| PR 15–49 | state | 7 | 3793 | 253 | 0.0667 | -0.0209 | 0.0725 | 0.0305 | 0.1015 | 0.0321 | yes |
| PR 15–49 | state | 8 | 14906 | 754 | 0.0506 | -0.0370 | 0.0097 | -0.0690 | -0.0040 | 0.0291 | yes |
| PR 15–49 | state | 9 | 32915 | 2401 | 0.0729 | -0.0146 | -0.0132 | -0.0369 | -0.0665 | 0.0053 | yes |
| PR 15+ | sex | female | 192409 | 25204 | 0.1310 | -0.0093 | 0.0003 | -0.0125 | -0.0331 | 0.0010 | yes |
| PR 15+ | sex | male | 168224 | 25392 | 0.1509 | 0.0106 | -0.0026 | 0.0119 | 0.0329 | 0.0003 | yes |
| PR 15+ | age_band | 15-24 | 85824 | 3096 | 0.0361 | -0.1042 | -0.1676 | -0.2832 | -0.7877 | -0.0010 | yes |
| PR 15+ | age_band | 25-34 | 77588 | 5613 | 0.0723 | -0.0680 | -0.1553 | -0.2297 | -0.6441 | 0.0006 | yes |
| PR 15+ | age_band | 35-44 | 64319 | 8523 | 0.1325 | -0.0078 | -0.1342 | -0.1312 | -0.0896 | 0.0021 | yes |
| PR 15+ | age_band | 45-49 | 30070 | 5350 | 0.1779 | 0.0376 | -0.1307 | -0.0606 | 0.1880 | 0.0030 | yes |
| PR 15+ | age_band | 50-59 | 46826 | 11598 | 0.2477 | 0.1074 | -0.1137 | 0.0380 | 0.2074 | 0.0033 | yes |
| PR 15+ | age_band | 60-69 | 35898 | 10356 | 0.2885 | 0.1482 | -0.0993 | 0.1047 | 0.2075 | 0.0038 | yes |
| PR 15+ | age_band | 70+ | 20108 | 6060 | 0.3014 | 0.1611 | -0.1145 | 0.0926 | 0.2075 | 0.0068 | yes |
| PR 15+ | wealth | 1 | 74363 | 8284 | 0.1114 | -0.0289 | -0.0379 | -0.1068 | -0.0908 | 0.0009 | yes |
| PR 15+ | wealth | 2 | 79170 | 9569 | 0.1209 | -0.0194 | -0.0241 | -0.0735 | -0.0489 | 0.0012 | yes |
| PR 15+ | wealth | 3 | 75645 | 10529 | 0.1392 | -0.0011 | -0.0099 | -0.0289 | -0.0068 | 0.0035 | yes |
| PR 15+ | wealth | 4 | 68978 | 10943 | 0.1586 | 0.0183 | 0.0099 | 0.0377 | 0.0344 | 0.0011 | yes |
| PR 15+ | wealth | 5 | 62477 | 11271 | 0.1804 | 0.0401 | 0.0301 | 0.0937 | 0.0811 | 0.0012 | yes |
| PR 15+ | residence | rural | 272967 | 35627 | 0.1305 | -0.0098 | -0.0110 | -0.0346 | -0.0217 | 0.0003 | yes |
| PR 15+ | residence | urban | 87666 | 14969 | 0.1708 | 0.0305 | 0.0200 | 0.0704 | 0.0518 | 0.0026 | yes |
| PR 15+ | education | 0 | 94263 | 16418 | 0.1742 | 0.0339 | -0.0656 | -0.0288 | 0.0909 | 0.0017 | yes |
| PR 15+ | education | 1 | 48777 | 8376 | 0.1717 | 0.0314 | -0.0175 | 0.0253 | 0.0562 | 0.0001 | yes |
| PR 15+ | education | 2 | 170972 | 20494 | 0.1199 | -0.0204 | 0.0202 | 0.0023 | -0.0658 | 0.0005 | yes |
| PR 15+ | education | 3 | 46432 | 5271 | 0.1135 | -0.0268 | 0.0160 | 0.0097 | -0.1175 | 0.0012 | yes |
| PR 15+ | education | missing | 189 | 37 | 0.1958 | 0.0555 | 0.0270 | 0.1405 | 0.0994 | 0.0296 | no |
| PR 15+ | state | 1 | 11459 | 1042 | 0.0909 | -0.0494 | 0.0198 | -0.0774 | 0.0127 | 0.0423 | yes |
| PR 15+ | state | 10 | 21633 | 3198 | 0.1478 | 0.0075 | -0.0169 | -0.0131 | -0.0733 | 0.0251 | yes |
| PR 15+ | state | 11 | 1696 | 257 | 0.1515 | 0.0112 | -0.0188 | -0.0423 | 0.0557 | 0.0200 | yes |
| PR 15+ | state | 12 | 8424 | 1021 | 0.1212 | -0.0191 | -0.0659 | -0.1116 | -0.0628 | 0.0144 | yes |
| PR 15+ | state | 13 | 3982 | 522 | 0.1311 | -0.0092 | 0.0367 | 0.0116 | 0.0581 | 0.0192 | yes |
| PR 15+ | state | 14 | 4780 | 687 | 0.1437 | 0.0034 | -0.0020 | -0.0139 | 0.0386 | 0.0157 | yes |
| PR 15+ | state | 15 | 4208 | 632 | 0.1502 | 0.0099 | -0.0065 | 0.0002 | 0.0129 | 0.0112 | yes |
| PR 15+ | state | 16 | 3057 | 527 | 0.1724 | 0.0321 | -0.0549 | -0.0236 | -0.0942 | 0.0410 | yes |
| PR 15+ | state | 17 | 6087 | 723 | 0.1188 | -0.0215 | -0.0192 | -0.0758 | -0.1231 | 0.0113 | yes |
| PR 15+ | state | 18 | 16461 | 2357 | 0.1432 | 0.0029 | -0.0167 | -0.0137 | -0.0551 | 0.0142 | yes |
| PR 15+ | state | 19 | 10828 | 2172 | 0.2006 | 0.0603 | -0.0387 | 0.0309 | -0.0780 | 0.0626 | yes |
| PR 15+ | state | 2 | 4477 | 615 | 0.1374 | -0.0029 | -0.0007 | 0.0080 | 0.0563 | 0.0281 | yes |
| PR 15+ | state | 20 | 12519 | 1599 | 0.1277 | -0.0126 | -0.0056 | -0.0262 | -0.0395 | 0.0025 | yes |
| PR 15+ | state | 21 | 13960 | 2146 | 0.1537 | 0.0134 | -0.0169 | -0.0028 | -0.0171 | 0.0150 | yes |
| PR 15+ | state | 22 | 12640 | 1357 | 0.1074 | -0.0329 | -0.0138 | -0.0647 | -0.0320 | 0.0199 | yes |
| PR 15+ | state | 23 | 24531 | 2809 | 0.1145 | -0.0258 | -0.0153 | -0.0718 | -0.0243 | 0.0170 | yes |
| PR 15+ | state | 24 | 17444 | 2678 | 0.1535 | 0.0132 | -0.0150 | -0.0023 | -0.0016 | 0.0095 | yes |
| PR 15+ | state | 25 | 1790 | 299 | 0.1670 | 0.0267 | -0.0023 | 0.0392 | -0.0099 | 0.0193 | yes |
| PR 15+ | state | 27 | 19325 | 2410 | 0.1247 | -0.0156 | 0.0072 | -0.0108 | 0.0436 | 0.0268 | yes |
| PR 15+ | state | 28 | 5580 | 1190 | 0.2133 | 0.0730 | 0.0110 | 0.1302 | 0.0251 | 0.0565 | yes |
| PR 15+ | state | 29 | 16797 | 2516 | 0.1498 | 0.0095 | 0.0087 | 0.0284 | 0.0453 | 0.0070 | yes |
| PR 15+ | state | 3 | 10695 | 1566 | 0.1464 | 0.0061 | 0.0250 | 0.0236 | 0.0785 | 0.0224 | yes |
| PR 15+ | state | 30 | 957 | 207 | 0.2163 | 0.0760 | 0.0690 | 0.1808 | 0.0915 | 0.0453 | yes |
| PR 15+ | state | 32 | 6066 | 1580 | 0.2605 | 0.1202 | 0.0447 | 0.2116 | 0.1239 | 0.0469 | yes |
| PR 15+ | state | 33 | 13732 | 3243 | 0.2362 | 0.0959 | 0.0054 | 0.1410 | 0.0635 | 0.0631 | yes |
| PR 15+ | state | 34 | 2061 | 515 | 0.2499 | 0.1096 | 0.0610 | 0.2268 | 0.0968 | 0.0552 | yes |
| PR 15+ | state | 35 | 1612 | 299 | 0.1855 | 0.0452 | 0.0041 | 0.1033 | 0.0369 | 0.0212 | yes |
| PR 15+ | state | 36 | 13726 | 2145 | 0.1563 | 0.0160 | 0.0110 | 0.0371 | 0.0299 | 0.0080 | yes |
| PR 15+ | state | 37 | 1215 | 198 | 0.1630 | 0.0227 | -0.0891 | -0.0897 | -0.1713 | 0.0630 | yes |
| PR 15+ | state | 5 | 6770 | 804 | 0.1188 | -0.0215 | 0.0034 | -0.0369 | 0.0309 | 0.0258 | yes |
| PR 15+ | state | 6 | 11866 | 1699 | 0.1432 | 0.0029 | 0.0102 | -0.0011 | 0.0315 | 0.0082 | yes |
| PR 15+ | state | 7 | 4918 | 620 | 0.1261 | -0.0142 | 0.0703 | 0.0740 | 0.0994 | 0.0354 | yes |
| PR 15+ | state | 8 | 20150 | 1729 | 0.0858 | -0.0545 | 0.0057 | -0.1096 | 0.0016 | 0.0375 | yes |
| PR 15+ | state | 9 | 44473 | 5113 | 0.1150 | -0.0253 | -0.0088 | -0.0566 | -0.0413 | 0.0086 | yes |

Subgroups flagged Reliable = no have too few respondents or positives to support a fairness or performance claim.

## Limitations

- Raw probabilities are not directly comparable across the three runs because the samples and feature sets differ. Only discrimination, calibration, threshold behaviour, and subgroup gaps are compared.
- The samples are not nested or exchangeable. The primary model uses the IR/MR recodes, while the PR models use the household-member recode; the primary sample is mostly female and the PR sample is far more balanced.
- The PR 15–49 run is the cleanest feature-set comparison, but it is not a pure ablation of the primary model: the PR and IR/MR instruments measure different concepts and the PR sample covers a different population.
- Older adults are present only in the PR 15+ run. Any difference between the PR 15+ and PR 15–49 runs mixes the age extension with the different age-specific case mix.
- Survey-weighted metrics are reported where the run produced them. The test evaluation is still a single cluster-grouped holdout, not external validation.
- Threshold metrics depend on prevalence. PPV and NPV cannot be transported to another setting without accounting for that setting's prevalence.
- The PR model is a sensitivity analysis, not the deployed screener. It is not a diagnostic test and must not be used for clinical, insurance, employment, or care-allocation decisions.

The PR model is a sensitivity analysis. The primary IR/MR model remains the model intended for the app.
