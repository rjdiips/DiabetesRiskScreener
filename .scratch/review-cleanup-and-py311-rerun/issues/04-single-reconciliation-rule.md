# 04: One reconciliation rule, one declaration per documented count

**What to build:** Every sample reconciliation — the PR feature-table audit, the PR BP-valid 15+ sample, the PR BP-valid 15–49 sample, and the primary IR/MR 15–49 sample — applies the same comparison rule from one place, and each documented count is declared once. The reported result keeps its current shape (status, observed summary, per-count checks with expected, observed, and pass), so run metadata and anything that reads it stay unchanged. Changing how a count is compared is now a one-place edit, and a documented recount no longer requires hunting for copies.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] All four reconcilers delegate to a single comparison helper.
- [ ] The returned structure and its serialized form are unchanged.
- [ ] Each documented count is declared once per sample, with no duplicated literals in tests.
- [ ] Tests assert through the public reconcilers, not the helper directly.
- [ ] Existing reconciliation tests keep passing with no assertion changes.
