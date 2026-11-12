# 03: Metrics prefactor

**What to build:** Two reusable metric capabilities that the selection evidence needs. First, a decision-curve net-benefit table at threshold probabilities 0.05–0.20. Second, a subgroup-metrics helper that works on either development out-of-fold rows or test rows, with the split it was computed on carried in the output so the two reports cannot be confused. The existing reliability and audit rules for subgroups stay unchanged.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] A net-benefit helper returns net benefit for every threshold in 0.05–0.20 and handles a sample with no positives or no negatives without dividing by zero.
- [x] The net-benefit output has a stable table shape and is unit-tested against a hand-calculated example.
- [x] The subgroup-metrics helper accepts a frame, target, probabilities, configuration, overall metrics, and an explicit split label.
- [x] Every subgroup row carries the split label so development out-of-fold and test rows are distinguishable.
- [x] Existing subgroup reliability, audit-only, and small-group rules behave exactly as before.
- [x] Tests cover the helper on both a development-labelled frame and a test-labelled frame.

## Comments

Implemented on `feature/model-shortlist` (commit `69e1af2`, merge `ac803da`; review fixes `790bab3`, merge `533dadb`).

`src/screener/metrics.py` gains `net_benefit_table` over thresholds 0.05-0.20 (16 rows, optional `sample_weight`) with hand-calculated and empty-class tests. `src/screener/pipeline.py` exposes `subgroup_metrics(frame, target_column, probabilities, config, overall, split_label)`; every row carries `split`, non-finite OOF probabilities are excluded from `n` and the metric call, and the existing reliability/audit-only/small-group rules are unchanged. The pipeline call site passes `TEST_SPLIT`. Tests cover development- and test-labelled frames. Full suite: 173 passed.
