# 05: Fit handle and final evaluation

**What to build:** The selection result can refit any named candidate on all development rows through a fit handle. The fit returns the model bundle that inference already uses, with the raw model, category levels, calibrator, threshold, feature columns, and candidate metadata. The existing single-model pipeline is refactored to prepare the data, run selection or accept a pinned candidate, fit that candidate, read the test split exactly once, and write the existing artifact set. With no pinned candidate, the current winner order and primary/PR behaviour are preserved.

**Blocked by:** 04

**Status:** resolved

- [x] The fit handle refits the named candidate on all development rows.
- [x] The fitted bundle's threshold, calibrator, feature columns, and predictions are consistent with that candidate's evidence.
- [x] The single-model pipeline uses the preparation and selection seams, then fits and evaluates the test split once.
- [x] The pipeline signature and existing artifact keys remain backward compatible.
- [x] A pinned candidate uses exactly that family and those parameters, and no other candidate is scored or fitted.
- [x] With no pinned candidate, the default winner order matches the current behaviour.
- [x] Test metrics, test subgroup rows, test calibration curves, test feature importance, and test SHAP values are produced only after the winner is fixed.
- [x] The final run records the pinned candidate and the selection row count separately from the test row count.
- [x] Existing primary and PR pipeline tests continue to pass.
- [x] Tests cover the fit handle, the pinned path, the test-evaluated-once path, and backward compatibility.
