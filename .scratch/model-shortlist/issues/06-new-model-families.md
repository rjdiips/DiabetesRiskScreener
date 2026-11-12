# 06: Add scored candidate families and train unweighted

**What to build:** Extend `screener.modeling` beyond prevalence/logistic/lightgbm with random forest, XGBoost, CatBoost, a spline logistic regression (restricted cubic spline basis on age, BMI, waist, systolic BP, and diastolic BP), and a monotone-constrained LightGBM (risk non-decreasing in age, BMI, waist, systolic BP, and diastolic BP only). All families use the existing feature-matrix and preprocessing contract and return positive-class probabilities in [0, 1]. Train every family unweighted: remove `class_weight="balanced"` from logistic regression and `is_unbalance` from LightGBM. Keep one weighted LightGBM configuration as a labelled sensitivity check, not a candidate. Add xgboost and catboost to the project dependencies. The seven scored candidates are C1 logistic, C2 spline logistic, C3 random forest, C4 LightGBM, C5 XGBoost, C6 CatBoost, and C7 monotone LightGBM.

**Blocked by:** 02 (candidate configuration)

**Status:** resolved

- [x] `MODEL_FAMILIES` includes random forest, xgboost, catboost, spline logistic, and monotone LightGBM; each builds, fits, and returns positive-class probabilities in [0, 1].
- [x] Logistic regression and LightGBM train unweighted by default.
- [x] A weighted LightGBM sensitivity configuration exists and is labelled as such.
- [x] XGBoost and CatBoost run on CPU; the code does not require a GPU.
- [x] `pyproject.toml` declares xgboost and catboost.
- [x] A restricted cubic spline basis helper transforms the five continuous features and is unit-tested (shape, monotone basis ordering, no NaNs).
- [x] The spline logistic candidate fits and predicts on a synthetic matrix.
- [x] The monotone LightGBM candidate maps its constraints to the correct feature positions regardless of column order.
- [x] No constraint is applied to wealth, education, residence, diet, or behaviour.
- [x] Tests assert that increasing age/BMI/waist/BP cannot lower the predicted risk in the constrained model on a controlled synthetic example.
- [x] Unit tests cover every family builder on a small synthetic matrix.
- [x] Existing tests pass with the unweighted defaults.

## Comments

Implemented on `main` (commit `1e01ad3`, merge `6673fb1`; spline corrected in review fixes `0b328af`, merge `7b66cae`).

`src/screener/modeling.py` adds `random_forest`, `xgboost`, `catboost`, `logistic_spline`, and `lightgbm_monotone` to `MODEL_FAMILIES`. All build through `build_model`, fit through `fit_model`, and return positive-class probabilities in [0, 1]; random forest, XGBoost, and CatBoost reuse the shared one-hot preprocessing and run CPU-only, with CatBoost iterations capped at 300. Logistic regression no longer sets `class_weight="balanced"` and `DEFAULT_LIGHTGBM_PARAMS` no longer sets `is_unbalance`. The weighted LightGBM run is `SENSITIVITY_CONFIGS["lightgbm_weighted_sensitivity"]` (family `lightgbm`, `is_unbalance: True`), deliberately outside `MODEL_FAMILIES` and every default candidate list. `logistic_spline` uses `RestrictedCubicSplineTransformer`, a genuine Harrell restricted cubic spline basis on `age`, `bmi`, `waist_cm`, `systolic_mean`, and `diastolic_mean` with quantile knots and medians learned inside the fitted pipeline (so folds cannot leak), linear tails, and missing columns ignored for the PR feature set; the public `restricted_cubic_spline_basis` helper is the same transformer. `monotone_constraints(feature_columns)` maps +1 to those five features by name in matrix order and 0 to everything else, so wealth, education, residence, diet, and behaviour stay unconstrained. `pyproject.toml` declares `xgboost>=2.0` and `catboost>=1.2`. `tests/test_modeling.py` covers every family on synthetic matrices, unweighted defaults, the sensitivity label, the spline basis (shape, ordering, NaNs, linear tails, determinism), shuffled constraint positions, and monotone non-decrease under rising risk features. Full suite: 203 passed.
