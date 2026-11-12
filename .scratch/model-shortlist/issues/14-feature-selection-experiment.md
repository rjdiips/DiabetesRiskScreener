# 14: Post-freeze feature-selection experiment

**What to build:** After the primary winner is frozen, run one labelled algorithmic feature-selection experiment: permutation importance or recursive feature elimination, with the selector fitted inside the development folds only and nested within the same cluster-grouped CV. Evaluate the selected subset on the same out-of-fold predictions and compare it with the frozen winner. Reuse the existing promotion rule: the experiment is promoted only if ΔROC-AUC versus the frozen winner is at least 0.005 with a cluster-bootstrap 95% CI excluding zero, calibrated Brier is within 0.005, and no reliable subgroup sensitivity gap exceeds 0.10. Otherwise publish it as an exploration result and do not change the deployed artifact. Promotion requires a new freeze and PR rerun; this ticket must not modify the frozen winner.

**Blocked by:** 11

**Status:** ready-for-agent

- [ ] One pre-registered selector is run; no open-ended feature search.
- [ ] Selection happens inside the development folds only; the test set is untouched.
- [ ] The selected subset is reported with its stability across folds and seeds.
- [ ] The promotion rule is reused or unit-tested for pass, fail-on-margin, fail-on-calibration, and fail-on-subgroup.
- [ ] The decision and evidence are written to the comparison artifact.
- [ ] A non-promoted experiment changes no deployed artifact.
- [ ] A promoted experiment opens a new freeze and PR-rerun ticket instead of editing the frozen run.
- [ ] No second selection method or neural architecture is added.
