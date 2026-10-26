# 01: Shared run summary and row-count accessor

**What to build:** Both model-run entry points (the primary IR/MR 15–49 run and the PR sensitivity runs) print their end-of-run summary from one shared library function instead of two hand-written copies. The summary keeps every field it has today — run directory, run name, sample and sample label, comparison run, smoke flag, row counts, split source, selected model family, reconciliation status, documented rows and positives, documented-match, overall ROC-AUC, PR-AUC, threshold, sensitivity, specificity, and the fact-sheet PR-AUC — and both entry points produce the same shape for the same inputs. Reading the pipeline run result's row counts goes through one accessor rather than repeated nested lookups. A smoke run prints the same summary as before this change.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] A single library function builds the run summary from a pipeline run result plus the smoke flag, reconciliation status, and documented counts.
- [ ] Both run entry points call it; neither builds a summary mapping inline.
- [ ] The printed keys and values are unchanged for an existing run, verified by a smoke run.
- [ ] Row counts are read through one accessor; no entry point indexes nested run metadata for them.
- [ ] The summary function is unit-tested against a hand-built fake run result, with no training.
- [ ] The function performs no filesystem access.
