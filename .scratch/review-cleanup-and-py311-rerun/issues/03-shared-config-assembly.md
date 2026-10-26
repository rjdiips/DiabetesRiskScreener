# 03: One config assembler for the primary and PR models

**What to build:** The primary IR/MR 15–49 model and the two PR sensitivity models are assembled by the same code path, so a pipeline setting added to one family cannot silently miss the other. The subgroup set they share (sex, age band, wealth, residence, education, state) is defined once and parameterised by the age bands, labels, and state minimum-test-rows threshold. Audit-only religion and social group stay a PR-variant concern with their household-head note, and the primary model keeps a note saying those subgroups are reported for the audit and are not deployed features. The 15–49 age bands and labels exist once and are used by both the primary model and the PR 15–49 comparison. The primary config no longer computes a categorical-feature tuple that is always empty, and the constant behind it is gone, so nothing in the code suggests the deployed model can use state, religion, or social group as features.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The public primary and PR config builders keep their current names and arguments.
- [ ] Both families are built through one shared assembly path; only their differences are supplied per family.
- [ ] The shared subgroup set is declared once.
- [ ] The 15–49 age bands and labels are declared once.
- [ ] The primary config passes no categorical columns, and a test pins that.
- [ ] The dead categorical-feature constant is deleted.
- [ ] Existing config assertions still pass, and every field they cover keeps its current value.
- [ ] No model is trained in these tests.
