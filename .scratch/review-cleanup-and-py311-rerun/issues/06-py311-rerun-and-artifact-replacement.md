# 06: Python 3.11 rerun and artifact replacement

**What to build:** The three published models are rerun on Python 3.11, the interpreter the project documents, and the new runs replace the old ones so exactly one set of published results exists. The run order stays as documented: primary model, split rebuild reusing the primary assignment, PR 15+, PR 15–49, comparison, sample verification. The old run directories and the old comparison outputs are removed in the same change, and the README results table, run list, and data documentation quote the new records. Afterwards the sample-verification command passes, every run record names 3.11, and the README's environment statement and the artifacts finally agree.

**Blocked by:** 01, 03, 04, 05

**Status:** ready-for-agent

- [ ] Python 3.11 runs the full stack: a fresh environment with the project and its dev extras installs and passes the suite.
- [ ] A smoke run on 3.11 succeeds before any full run.
- [ ] Primary IR/MR 15–49, PR 15+, and PR 15–49 full runs complete on 3.11.
- [ ] Sample verification passes for every documented count, matching ticket 07's reconciliation.
- [ ] All three runs record `split_source: mixed` and the 3.11 interpreter in their run records.
- [ ] The comparison report is regenerated from the new runs and still contains the sample, feature, overall, weighted, calibration, threshold, sex, age-band, and subgroup tables with computed deltas.
- [ ] The three previously published run directories and the previous comparison outputs are removed; one published answer set remains.
- [ ] The README results table and run paths, and the data documentation, quote the new records.
- [ ] Ticket 07's implementation record notes the interpreter the published runs now use.
- [ ] No microdata, split assignment, per-respondent SHAP values, or model binary is committed.
- [ ] The full suite and ruff pass on 3.11.
