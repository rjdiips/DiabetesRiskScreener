# 03: One config assembler for the primary and PR models

**What to build:** The primary IR/MR 15–49 model and the two PR sensitivity models are assembled by the same code path, so a pipeline setting added to one family cannot silently miss the other. The subgroup set they share (sex, age band, wealth, residence, education, state) is defined once and parameterised by the age bands, labels, and state minimum-test-rows threshold. Audit-only religion and social group stay a PR-variant concern with their household-head note, and the primary model keeps a note saying those subgroups are reported for the audit and are not deployed features. The 15–49 age bands and labels exist once and are used by both the primary model and the PR 15–49 comparison. The primary config no longer computes a categorical-feature tuple that is always empty, and the constant behind it is gone, so nothing in the code suggests the deployed model can use state, religion, or social group as features.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] The public primary and PR config builders keep their current names and arguments.
- [x] Both families are built through one shared assembly path; only their differences are supplied per family.
- [x] The shared subgroup set is declared once.
- [x] The 15–49 age bands and labels are declared once.
- [x] The primary config passes no categorical columns, and a test pins that.
- [x] The dead categorical-feature constant is deleted.
- [x] Existing config assertions still pass, and every field they cover keeps its current value.
- [x] No model is trained in these tests.

## Comments

`src/screener/_config_assembly.py` now holds `assemble_config`,
`shared_subgroups`, the single `AGE_15_49_BANDS`/`AGE_15_49_LABELS`, and the
metadata helper; `primary_15_49_config` and `_pr_config` supply only what
differs. `PRIMARY_CATEGORICAL_FEATURES` and the always-empty categorical
computation are gone, and a test pins `categorical_columns == ()`. The
duplicated `PRIMARY_AGE_*` and `PR_15_49_AGE_*` constants were deleted rather
than aliased, since nothing in the repo referenced them. `TrainingConfig.to_dict()`
is byte-identical for nine config variants. Landed as `e30fa36`; the review
fix `7787952` made the assembler's shared pipeline settings required (they had
re-declared `TrainingConfig` defaults) and renamed the helper to
`config_metadata`, because the repo already calls the `run_metadata.json`
artifact "run metadata".
