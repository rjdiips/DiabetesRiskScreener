# Model card: primary_irmr_15to49

Version date: 2026-10-26

## Intended use

Research and portfolio screening support for elevated random blood glucose or diagnosed diabetes among Indian adults. Not a diagnosis.

## Target definition

glucose > 140 OR diagnosed OR medication

## Sample

774861 respondents were used, with 12943 positives in the test set.

## Features

`female`, `age`, `urban`, `wealth`, `education`, `marital_status`, `children_ever_born`, `height_cm`, `weight_kg`, `bmi`, `waist_cm`, `hip_cm`, `waist_hip_ratio`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication`, `tobacco_any`, `smoking_any`, `smokeless_any`, `alcohol_any`, `alcohol_frequency`, `diet_milk`, `diet_pulses`, `diet_green_leafy`, `diet_fruit`, `diet_eggs`, `diet_fish`, `diet_meat`, `diet_fried`, `diet_aerated`

## Performance

- ROC-AUC: 0.7099
- PR-AUC: 0.2035
- Brier: 0.0721
- Sensitivity at the chosen threshold: 0.8096
- Specificity at the chosen threshold: 0.4558

## Calibration and threshold

The calibrator was selected on out-of-fold predictions (`isotonic`). The referral threshold is 0.0549 and targets at least 0.80 sensitivity.

## Limitations and non-goals

- The model does not predict future diabetes.
- The model is not clinically validated.
- It must not be used for insurance, employment, or care-allocation decisions.
- Subgroup gaps are descriptive, not causal.

## Ethical considerations

- The target includes self-reported diagnosis and medication, so the model can learn access-to-care patterns.
- Removing a protected attribute does not by itself make a model fair.
- Reported for the fairness audit only; not a deployed model feature.

## Citation

International Institute for Population Sciences (IIPS) and ICF (2021). National Family Health Survey (NFHS-5), 2019-21: India.
