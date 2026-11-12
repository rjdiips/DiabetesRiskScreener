# 07: Measurement, questionnaire-only, and ablation feature bundles

**What to build:** Define the two primary named feature sets and the pre-registered measurement block-ablation bundles. The measurement set is the existing 32 primary features. The questionnaire-only set removes waist, hip, waist-to-hip ratio, systolic and diastolic BP, BP reading count, measured high BP, and BP medication, and keeps age, sex, height, weight, BMI, education, wealth, residence, marital status, children ever born, tobacco, alcohol, and diet. The measurement bundles are `measurement_full` (the 32), `measurement_no_anthropometry` (26; drops `height_cm`, `weight_kg`, `bmi`, `waist_cm`, `hip_cm`, `waist_hip_ratio`), `measurement_no_bp` (27; drops `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication`), and `measurement_no_diet` (23; drops the nine `diet_*` columns). The substance block is not ablated because it is cheap to collect and has a strong clinical prior. All sets must pass the existing feature-policy validation and exclude target-derived, identifier, and survey-design columns.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Two primary feature-set constructors (measurement and questionnaire-only) and three reduced measurement-bundle constructors (`measurement_no_anthropometry`, `measurement_no_bp`, `measurement_no_diet`) exist and are used by the model configs; `measurement_full` is the measurement set.
- [x] The questionnaire-only set contains no BP or waist/hip columns and keeps height, weight, and BMI.
- [x] Each ablation bundle is exactly `measurement_full` minus one block and has the documented column count.
- [x] Both primary sets and every bundle exclude glucose, diagnosis, medication, identifiers, cluster, strata, weights, and the sample flag.
- [x] A test asserts the exact column counts, the block memberships, and the excluded columns for every set.
- [x] Model configs accept any named set without changing the pipeline.
- [x] The questionnaire-only set is marked non-competing for deployment.

## Comments

Implemented on `feature/model-shortlist` (commit `2137072`, merge `b41ad25`; review fixes `790bab3`, merge `533dadb`).

`src/screener/feature_policy.py` defines `MEASUREMENT_BLOCKS`, `QUESTIONNAIRE_EXCLUDED_FEATURES`, `PRIMARY_FEATURE_SETS` (`measurement_full` 32, `questionnaire_only` 24, `measurement_no_anthropometry` 26, `measurement_no_bp` 27, `measurement_no_diet` 23), `NON_COMPETING_FEATURE_SETS`, and `primary_feature_set`. `primary_15_49_config(feature_set=...)` resolves any named set without touching the pipeline and records non-default sets in `feature_variant`. Tests assert exact counts, block membership, exclusions, forbidden-column disjointness, non-competing status, and `validate_feature_frame` for every set. Full suite: 173 passed.

Note for ticket 04: `Candidate` has no variant field yet; the feature-set name reaches candidates through the run configuration, so carry it into selection evidence there.
