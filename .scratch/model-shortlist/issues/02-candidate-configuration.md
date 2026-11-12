# 02: Candidate configuration

**What to build:** The pipeline can score a configured list of candidates and can accept a pinned candidate for the final run. A candidate carries a model family, a parameter mapping, and a stable label. The hard-coded three-family trial loop is replaced by the configured candidate list. With no candidate list supplied, the existing three families and the existing winner order are preserved, so primary and PR runs do not change. With a pinned candidate supplied, the pipeline scores and fits exactly that family and those parameters and does not re-run family selection.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] A candidate value object records family, parameters, and a stable label.
- [x] Configuration can supply a candidate list and a pinned candidate.
- [x] The trial loop is driven by the configured candidates, not a hard-coded family tuple.
- [x] Each family's configured tuning settings expand into scored candidates, and the selected setting is recorded in the tuning record.
- [x] A pinned candidate produces exactly one scored candidate and no family selection.
- [x] With no candidate list and no pinned candidate, the existing default winner order still applies.
- [x] Tests cover the configured list, the pinned path, and the unchanged default path.
- [x] Existing primary and PR configuration tests continue to pass.

## Comments

Implemented on `feature/model-shortlist` (commit `0d43983`, merge `d28f7d7`; review fixes `790bab3`, merge `533dadb`).

`Candidate` is a frozen value object in `src/screener/pipeline.py`; `TrainingConfig` carries `candidates` and `pinned_candidate` and serialises both. `_resolve_candidates` drives the trial loop: pinned -> one trial, configured list -> exactly those, neither -> the existing `prevalence`, `logistic`, `lightgbm` order with `model_params` list expansion. Each trial records its label/family/params and a `tuning_record` records the trial count, selected setting, and search space. Duplicate or empty candidate labels are rejected. Tests cover the configured, pinned, default, and tuning-expansion paths; primary/PR config tests pass unchanged. Full suite: 173 passed.
