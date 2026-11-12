# 04: Selection evidence seam

**What to build:** A public selection step, `evaluate_candidates(development, config)`, that scores every configured candidate on the existing development folds and returns out-of-fold evidence without touching the test split. Each candidate's evidence carries calibrated probabilities, the selected calibration method, the screening threshold at the configured sensitivity target, discrimination and calibration metrics, development subgroup gaps, the reliability curve, net benefit, fold-level metrics, and the tuning record. The selection result also records the named feature set or measurement bundle it was produced on, so evidence from two bundles with the same candidate labels cannot be confused. The step rejects any frame containing a test row before fitting anything.

**Blocked by:** 01, 02, 03

**Status:** resolved

- [x] The selection step returns one evidence record per configured candidate and tuning setting.
- [x] Each record names the candidate label, family, and parameters, and records the selected tuning setting.
- [x] The selection result records the feature-set or bundle name from `config.metadata["feature_set"]`, falling back to `config.metadata["feature_variant"]`, then `"default"`, plus the feature count, so bundle runs are self-describing.
- [x] A test shows two configs that differ only in `feature_columns` produce distinctly labelled selection results.
- [x] Calibration is fitted per candidate on that candidate's out-of-fold predictions, and the screening threshold is chosen on those same predictions at the configured sensitivity target.
- [x] Discrimination, calibration, Brier, log loss, ECE, and threshold metrics are computed from out-of-fold predictions.
- [x] Development subgroup metrics use the existing reliability and audit rules and are labelled as development out-of-fold.
- [x] The reliability curve and the net-benefit table at 0.05–0.20 are present in the evidence.
- [x] A frame containing a test row is rejected before any model is fitted.
- [x] A test-set sentinel row with values that would break training does not affect selection.
- [x] The selection step does not compute test metrics, test subgroup rows, test calibration curves, test feature importance, or test SHAP values.
- [x] An integration test on a synthetic cluster-grouped table produces valid evidence for every candidate; no real data is used in CI.

## Comments

Implemented on `main` (commit `c69571a`, merge `d68e503`; review fixes `0b328af`, merge `7b66cae`).

`src/screener/pipeline.py` now exposes `evaluate_candidates(development, config)` plus `SelectionEvidence`/`SelectionRecord` and `SELECTION_SPLIT_LABEL = "development_oof"`. The guard runs before any model is built and rejects any non-development row (naming the test-row count), a missing split/fold column, or a fold outside `1..n_folds`. One record is produced per candidate from the existing `_resolve_candidates`, so configured lists, pinned candidates, the default winner order, and per-family tuning settings all compose; CV reuses `_cross_validate_candidate`. Calibration and the screening threshold are fitted per candidate on that candidate's own covered OOF predictions, and discrimination, calibration, Brier, log loss, ECE, threshold/confusion metrics, development-OOF subgroup rows, the reliability curve, the 0.05–0.20 net-benefit table, and fold metrics all come from OOF evidence. The result and every record carry `feature_set` (`metadata["feature_set"]` → `metadata["feature_variant"]` → `"default"`) and `feature_count`, plus `development_rows`/`fold_count`. No test metrics, subgroups, curves, importance, or SHAP are computed here. `tests/test_selection.py` covers the evidence contents, feature labelling (including two configs differing only in `feature_columns`), test-row and invalid-fold rejection before fitting, the broken-test-row sentinel, and the default candidate path. Full suite: 203 passed.
