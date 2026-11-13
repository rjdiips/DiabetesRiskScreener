# 11: Run the primary comparison, select the bundle, and freeze the winner

**What to build:** Run the comparison on the real measurement bundles from ticket 07, then on the questionnaire-only set. Select the deployed bundle with the pre-registered feature-bundle rule: choose each bundle's winner with the ranking rule, then choose the bundle with the fewest features whose winner is within 0.005 PR-AUC of, and no more than 0.005 Brier worse than, the best eligible bundle; break ties by higher PR-AUC, then lower Brier. Apply the seed-averaging capability from ticket 10 to the selected bundle's winner and freeze it as the deployable artifact. Replace the previous primary run with the new run so one answer set remains. Record the interpreter, seeds, tuning record, comparison artifacts, bundle name, updated model card, and the age-relative score reference and bands. Report the questionnaire-only results as the accessibility comparison; they do not compete for deployment.

**Blocked by:** 08, 09, 10

**Status:** ready-for-agent

- [ ] Every competing measurement bundle completes the full seven-candidate comparison and writes both comparison artifacts.
- [ ] Every competing measurement bundle uses the same primary split assignment.
- [ ] The primary split assignment is either reused from the previous run or regenerated and recorded; ticket 12 rebuilds the PR split against it.
- [ ] The feature-bundle margins (0.005 PR-AUC, 0.005 Brier) and tie-breaks are recorded before any comparison scores are examined.
- [ ] Each bundle's winner is recorded by the ranking rule.
- [ ] The feature-bundle rule selects the deployed bundle and records the PR-AUC and Brier margins.
- [ ] The selected bundle's winner is seed-averaged (or kept single when deterministic) and frozen with model bundle, calibrator, feature schema, score reference, bands, metrics, model card, and run metadata.
- [ ] The final run pins the selected family and tuned parameters; it does not re-run family selection.
- [ ] The primary run entry point accepts the selected bundle name and records it in the run metadata.
- [ ] The frozen artifact records the selected bundle name and feature schema.
- [ ] The questionnaire-only comparison completes and is reported as non-competing.
- [ ] The previous primary run directory is removed so one published answer exists.
- [ ] The run records the interpreter actually used (currently 3.14.6) and does not claim 3.11.
- [ ] The test set is evaluated once for the selected bundle's winner.
- [ ] Sample reconciliation passes for the documented rows and positives.
- [ ] No microdata, split assignment, per-respondent SHAP values, or model binary is committed.

## Comments

- 2026-11-14: The product output changed to an age-relative 0-100 score, three bands (0-49 lower, 50-79 moderate, 80-100 higher), a speedometer gauge, and a percentile curve, with the absolute probability shown as a natural frequency. The 80%-per-age-band-threshold freeze is superseded; the frozen artifact must store the age-band score reference and bands.
