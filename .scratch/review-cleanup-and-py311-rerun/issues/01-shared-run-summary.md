# 01: Shared run summary and row-count accessor

**What to build:** Both model-run entry points (the primary IR/MR 15–49 run and the PR sensitivity runs) print their end-of-run summary from one shared library function instead of two hand-written copies. The summary keeps every field it has today — run directory, run name, sample and sample label, comparison run, smoke flag, row counts, split source, selected model family, reconciliation status, documented rows and positives, documented-match, overall ROC-AUC, PR-AUC, threshold, sensitivity, specificity, and the fact-sheet PR-AUC — and both entry points produce the same shape for the same inputs. Reading the pipeline run result's row counts goes through one accessor rather than repeated nested lookups. A smoke run prints the same summary as before this change.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] A single library function builds the run summary from a pipeline run result plus the smoke flag, reconciliation status, and documented counts.
- [x] Both run entry points call it; neither builds a summary mapping inline.
- [x] The printed keys and values are unchanged for an existing run, verified by a smoke run.
- [x] Row counts are read through one accessor; no entry point indexes nested run metadata for them.
- [x] The summary function is unit-tested against a hand-built fake run result, with no training.
- [x] The function performs no filesystem access.

## Comments

`screener.run_summary.build_run_summary` now builds the printed mapping, and
`TrainingResult.row_counts` is the single accessor the summary reads. The
primary and PR entry points both call the shared function and no longer build
the mapping inline. `sample` defaults to the pipeline's recorded sample label,
so the primary run does not duplicate the constant; the PR runs pass the
CLI sample name ("15plus"/"15-49") and print the metadata label separately.

Verified with a 1,000-row smoke run of each entry point before and after the
change: the PR 15+ JSON is identical except `run_dir` (which carries a new
timestamp), and the primary JSON keeps every existing value and gains only
`sample_label` and `comparison_run`, giving both entry points the same shape
and key order. `tests/test_run_summary.py` covers the full mapping, the
comparison-run and missing fact-sheet cases, the metadata-derived sample, the
row-count accessor, and a run directory that does not exist. Full suite: 135
passing; ruff clean on the changed files.
