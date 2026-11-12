# 09: Leakage-safe comparison artifacts

**What to build:** Harden the comparison artifact so it cannot carry test-derived evidence for non-winners. Non-winner records contain only out-of-fold metrics. The winner record points at the final run, and its test metrics come only from that run. The artifact records the selection split, candidate labels, ranking and veto outcomes, and a test-evaluated-once flag. A sentinel test set makes a non-winner test metric appear if the contract is broken, so the regression test fails rather than relying on convention.

**Blocked by:** 08

**Status:** resolved

- [x] Non-winner records contain no test metric block.
- [x] The winner record points at the final run and its test metrics come only from that run.
- [x] The artifact records the selection split, candidate labels, ranking and veto outcomes, and a test-evaluated-once flag.
- [x] A regression test with a sentinel test set fails if a non-winner test metric appears.
- [x] The comparison artifact states that the test set was evaluated once.
- [x] No real data is used in CI.
