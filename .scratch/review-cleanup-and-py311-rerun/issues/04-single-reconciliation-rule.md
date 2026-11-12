# 04: One reconciliation rule, one declaration per documented count

**What to build:** Every sample reconciliation — the PR feature-table audit, the PR BP-valid 15+ sample, the PR BP-valid 15–49 sample, and the primary IR/MR 15–49 sample — applies the same comparison rule from one place, and each documented count is declared once. The reported result keeps its current shape (status, observed summary, per-count checks with expected, observed, and pass), so run metadata and anything that reads it stay unchanged. Changing how a count is compared is now a one-place edit, and a documented recount no longer requires hunting for copies.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] All four reconcilers delegate to a single comparison helper.
- [x] The returned structure and its serialized form are unchanged.
- [x] Each documented count is declared once per sample, with no duplicated literals in tests.
- [x] Tests assert through the public reconcilers, not the helper directly.
- [x] Existing reconciliation tests keep passing with no assertion changes.

## Comments

One private `_compare_counts` in `features.py` now backs the PR feature-table
audit, the PR BP-valid 15+ and 15–49 samples, and the primary 15–49 sample; a
second private helper, `_sample_reference`, names the narrowing the three
sample reconcilers share. Returned shapes and key order are unchanged
(`reconcile_pr_audit` stays `{status, checks}` with no `summary`), and each
documented count is declared once in `PRIMARY_REFERENCE`,
`PR_REFERENCE_COUNTS`, and `PR_15_49_REFERENCE`. The two reference-record
assertions in `test_primary_model.py` and `test_pr_model.py` keep their
literals deliberately: the ticket also requires existing assertions to stay,
and those tests are the pin on the documented values. Landed as `3ac6500`; the
review fix `7787952` added `_sample_reference` and corrected the helper
docstring.
