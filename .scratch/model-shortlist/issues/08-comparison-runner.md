# 08: Comparison runner and ranking rule

**What to build:** A runner that evaluates the seven scored candidates plus the prevalence reference on one named feature set or measurement bundle, with a bounded equal tuning budget (three settings per family), unweighted training, isotonic/Platt calibration chosen on out-of-fold predictions, and a threshold chosen at ≥80% sensitivity. Apply the pre-registered ranking rule: PR-AUC ranks, with vetoes for Brier more than 0.005 worse than the best candidate, calibration slope outside 0.8–1.2 or intercept outside −0.1–0.1, or a reliable subgroup sensitivity gap above 0.10. Implement the feature-bundle rule as a pure selection function: choose each bundle's winner with the ranking rule, then choose the bundle with the fewest features whose winner is within 0.005 PR-AUC of, and no more than 0.005 Brier worse than, the best eligible bundle; fail rather than deploy when no bundle passes. Write `comparison.json` and `comparison.md` with per-candidate out-of-fold discrimination, calibration, threshold metrics, subgroup gaps, net benefit at 0.05–0.20, tuning record, veto status, bundle name, and the winner. The runner uses only out-of-fold evidence; the test split is not read until the winner is refit. It must not call the full single-model pipeline once per candidate.

**Blocked by:** 04, 05, 06, 07

**Status:** resolved

- [x] One command runs the comparison for a named feature set or measurement bundle and writes both comparison artifacts.
- [x] All seven candidates use the same splits, folds, preprocessing contract, three-setting tuning budget, and one split assignment.
- [x] Candidate ranking, calibration, thresholds, and vetoes use out-of-fold predictions only; the test split is not read until the winner is refit.
- [x] The ranking rule and each veto are unit-tested on hand-built metric records, including a higher-PR-AUC candidate being vetoed.
- [x] The feature-bundle rule is a pure function and is unit-tested: a smaller bundle within the PR-AUC and Brier margins wins; a smaller bundle outside either margin loses; the full bundle wins when no reduced bundle qualifies.
- [x] Net benefit is computed and reported for 0.05–0.20.
- [x] The comparison artifact records per-candidate out-of-fold metrics, tuning record, veto status, bundle name, and the winner.
- [x] The bundle name is an explicit runner input used to resolve `primary_feature_set` and written verbatim into the artifact; it is never inferred from `feature_variant`, so `measurement_full` stays `measurement_full` and is not written as `default`.
- [x] A test asserts the artifact's bundle name equals the requested name (including `measurement_full`) and matches every evidence record's `feature_set`.
- [x] A synthetic-table integration test produces a valid comparison artifact; no real data in CI.
- [x] The runner does not call the full single-model pipeline once per candidate.
- [x] A CLI guard reports a missing feature table without a traceback.
