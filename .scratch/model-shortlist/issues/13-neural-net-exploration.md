# 13: Neural-net exploration and promotion rule

**What to build:** After the scored comparison and the frozen winner exist, train one MLP on the same folds and the frozen winner's feature set. Calibrate it identically. Promote it to the deployed model only if all three conditions hold: ΔROC-AUC versus the winner is at least 0.005 with a cluster-bootstrap 95% CI excluding zero; calibrated Brier is within 0.005 of the winner; no reliable subgroup sensitivity gap exceeds 0.10. Otherwise publish it as an exploration result and do not deploy it.

**Blocked by:** 11

**Status:** ready-for-agent

- [ ] One MLP trains on the same splits, features, and preprocessing as the scored comparison.
- [ ] The promotion rule is a pure function with unit tests for pass, fail-on-margin, fail-on-calibration, and fail-on-subgroup.
- [ ] The decision and its evidence are written to the comparison artifact.
- [ ] A non-promoted MLP changes no deployed artifact.
- [ ] No second neural architecture is added.
