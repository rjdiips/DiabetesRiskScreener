# Model card: pr_15plus_bp

Version date: 2026-10-26

## Intended use

Research and portfolio screening support for elevated random blood glucose or diagnosed diabetes among Indian adults. Not a diagnosis.

This is a **sensitivity analysis**, not the deployed screener. The primary IR/MR model remains the model intended for the app.

## Target definition

glucose > 140 OR diagnosed OR medication

## Sample

1805697 respondents were used, with 50596 positives in the test set.

## Features

`female`, `age`, `urban`, `wealth`, `education`, `tobacco_any`, `alcohol_any`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication`

The PR default excludes state, religion, social group, anthropometry, diet, cooking fuel, marital status, children ever born, the smoking/smokeless split, and alcohol frequency.

## Performance

- ROC-AUC: 0.7459
- PR-AUC: 0.3320
- Brier: 0.1077
- Sensitivity at the chosen threshold: 0.7925
- Specificity at the chosen threshold: 0.5614

## Calibration and threshold

The calibrator was selected on out-of-fold predictions (`isotonic`). The referral threshold is 0.1040 and targets at least 0.80 sensitivity.

## Limitations and non-goals

- The model does not predict future diabetes.
- The model is not clinically validated.
- It must not be used for insurance, employment, or care-allocation decisions.
- Subgroup gaps are descriptive, not causal.

## Ethical considerations

- The target includes self-reported diagnosis and medication, so the model can learn access-to-care patterns.
- Removing a protected attribute does not by itself make a model fair.
- Describes the household head, not necessarily the respondent.

## Citation

International Institute for Population Sciences (IIPS) and ICF (2021). National Family Health Survey (NFHS-5), 2019-21: India.
