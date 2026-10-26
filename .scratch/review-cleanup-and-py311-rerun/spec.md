# Review cleanup and Python 3.11 rerun

**Status:** ready-for-agent

## Problem Statement

Two problems remain after ticket 07 published the real-data models.

First, the published results were produced on Python 3.14.6, but the project documents 3.11 or 3.12 as the supported modeling stack. Anyone reproducing the runs follows Setup, gets a different interpreter than the one that produced the committed numbers, and cannot tell from the README whether that matters. The run records themselves say 3.14.6, so the documentation and the artifacts disagree about the environment that produced the results.

Second, a two-axis code review of the ticket 07 work found that the primary and PR paths share almost no code even where they are meant to behave identically, and that the two new command-line entry points have no tests. A contributor changing a subgroup definition, an age band, a reconciliation rule, or the summary a run prints must now find and edit several copies, and nothing fails if one copy is missed. The primary config also carries a categorical-feature computation that provably always yields nothing, which suggests the primary model can treat state, religion, or social group as features when it cannot.

## Solution

Rerun the primary IR/MR 15–49 model, the PR 15+ sensitivity model, and the PR 15–49 comparison on Python 3.11, replacing the existing published runs so exactly one set of results exists, and let the run records prove the interpreter. Then collapse the duplicated configuration, subgroup, reconciliation, and summary logic behind the seams the two model families already expose, merge the two run accessors in the comparison module into one, and add CLI guard tests for the untested entry points. Behaviour and published numbers must not change except where the rerun itself changes them.

## User Stories

1. As a maintainer, I want the published models rerun on Python 3.11, so that the committed results were produced by an interpreter the project actually supports.
2. As a reader of the README, I want the environment statement and the run records to agree, so that I can trust the reproducibility claim without opening three JSON files.
3. As a reader of the results table, I want it to quote the newly published runs exactly, so that I can compare the three models without wondering which run each row came from.
4. As a maintainer, I want the rerun to replace the previous runs rather than accumulate beside them, so that there is one canonical set of published metrics.
5. As a maintainer, I want the rerun to follow the documented order (primary, then the shared partition, then both PR samples, then the comparison), so that the cluster partition stays shared and `split_source` stays `mixed`.
6. As a maintainer, I want a single command that re-checks every documented sample count after the rerun, so that I do not re-derive the reconciliation by hand.
7. As a contributor, I want the summary that each run prints to be produced by one shared function, so that the primary and PR runs cannot drift in the fields they report.
8. As a contributor, I want that summary function testable without training a model, so that a field I add is covered in seconds rather than minutes.
9. As a contributor, I want the repeated row-count navigation on the pipeline result hidden behind one accessor, so that a change to the metadata shape breaks one line rather than several.
10. As a contributor, I want the primary and PR configs assembled by the same code path, so that a pipeline setting added to one family does not silently miss the other.
11. As a contributor, I want the shared subgroup definitions (sex, wealth, residence, education, state) expressed once, so that editing a subgroup means editing one place.
12. As a contributor, I want the 15–49 age bands and labels expressed once, so that the primary and PR 15–49 models cannot disagree about what "45–49" means.
13. As a contributor, I want the reconciliation rule expressed once, so that a change in how counts are compared applies to every sample.
14. As a maintainer, I want each documented sample count declared once per sample, so that a recount does not require hunting through constants, tests, and prose for copies.
15. As an auditor, I want the primary model's audit subgroup note to keep describing what those subgroups are, and not to imply they are model features.
16. As a contributor, I want the dead categorical-feature computation removed from the primary config, so that no reader concludes the deployed model uses state, religion, or social group as inputs.
17. As a contributor, I want one accessor in the comparison module for both metric values and run-level fields, so that I do not have to remember which of two near-identical helpers to call.
18. As a reviewer, I want the comparison report to render identically after that merge, so that a refactor does not quietly change the published report.
19. As a reviewer, I want the run entry points to fail fast and clearly when their input tables are missing, so that a mistyped path reports the missing file and the fix rather than a traceback.
20. As a reviewer, I want those CLI guards covered by tests that follow the existing CLI-test pattern, so that the new scripts are not the only untested entry points in the project.
21. As a CI consumer, I want the new tests to run without training a model, so that the suite stays fast.
22. As a reviewer, I want the refactor to be behaviour-preserving, so that the only changed numbers in the diff come from the rerun.
23. As a maintainer, I want the comparison report regenerated from the new runs, so that every section still reflects published artifacts.
24. As a maintainer, I want the data documentation to keep describing the run order and reconciliation accurately after the rerun.
25. As a maintainer, I want ticket 07's record to note the interpreter the published runs now use, so that the historical note does not contradict the artifacts.
26. As a maintainer, I want the publication policy unchanged by this work, so that no microdata, respondent-level split assignment, per-respondent SHAP values, or model binary enters the repository.
27. As a future contributor, I want no remaining claim that 3.14 is the stack in use, so that the environment guidance has one answer.

## Implementation Decisions

**Configuration assembly.** `screener.pr_model` and `screener.primary_model` keep their public config builders as the interface; no new public config API is introduced. Behind them, one internal assembler builds the pipeline config shared shape (target columns, target definition, cluster/household/weight columns, folds, test fraction, sensitivity, bootstrap, SHAP sample size, split assignment, smoke rows, metadata). The primary and PR builders supply only what differs: run name, feature columns, sample filters, age bands and labels, subgroup extras, model role, and sample label.

**Subgroup definitions.** The subgroup set shared by both families (sex, age band, wealth, residence, education, state) is defined once and parameterised by age bands, labels, and the state minimum-test-rows threshold. Audit-only religion and social group remain a PR-variant concern and keep their household-head note. The primary model keeps an audit note that states these are reported for the audit and are not deployed features.

**Age bands.** The 15–49 bands and labels are declared once and referenced by both the primary model and the PR 15–49 comparison.

**Dead categorical computation.** The primary config's categorical-feature tuple and its backing constant are deleted. The primary feature set contains none of those columns, and the config now passes no categorical columns. A test asserts the primary config's categorical columns are empty.

**Sample reconciliation.** One internal helper compares an observed count summary against a reference record and returns the existing `{status, summary, checks}` shape, so serialized run metadata does not change. Every reconciler (PR feature-table audit, PR BP-valid 15+, PR BP-valid 15–49, primary 15–49) delegates to it. Each documented count keeps exactly one declaration per sample: the primary reference record, the PR reference counts, and the 15–49 subset thereof.

**Run summary.** A new library function in `screener` builds the summary mapping that the run entry points print, from a pipeline run result plus the smoke flag, the reconciliation status, and the documented counts. Both the primary and PR run entry points call it. The printed keys and values stay exactly as they are today, including the documented-match boolean and the fact-sheet PR-AUC. The function does not touch the filesystem and does not train.

**Row counts.** The pipeline run result gains or exposes one accessor for the row counts that the summary and the entry points read, so no caller indexes the nested metadata directly.

**Comparison accessor.** The comparison module keeps one accessor that returns either a metric from a named block or a run-level field. The metric-value helper and the run-field helper merge into it; call sites and the report renderer are updated, and the rendered report must be byte-identical for unchanged inputs.

**Rerun and artifact replacement.** The rerun targets Python 3.11 (the floor of `requires-python`). It runs in the order already documented: primary model, split rebuild reusing the primary assignment, PR 15+, PR 15–49, comparison, sample verification. The new run directories replace the three previously published ones, and the comparison artifacts are regenerated from the new runs; the old run directories and old comparison outputs are removed in the same change. Feature tables are not rebuilt unless the sample verification disagrees with the documented counts, in which case the work stops and reports rather than publishing mismatched numbers.

**Documentation.** The README results table, the run list, and the data documentation are updated from the new run records. The environment statement continues to name 3.11 or 3.12 as the supported stack, and the run records now agree with it. Ticket 07's implementation record gains a note naming the interpreter the published runs now use.

**Publication policy.** Unchanged: aggregate metrics, plots, reports, and model cards are committed; microdata, split assignments, per-respondent SHAP values, and model binaries stay local and gitignored.

## Testing Decisions

A good test here asserts externally visible behaviour: the config a public builder returns, the summary a run prints, the report text a comparison produces, and the exit code and message a CLI gives. It must not assert on private helper names, call order, or intermediate data structures, because those are exactly what this work rearranges. Tests must not train a model.

**Seams**

- **Run summary (new):** a library function taking a run result plus the smoke flag, reconciliation status, and documented counts, returning the mapping the entry points print. Unit-tested against a hand-built fake result. This is the only new seam.
- **Config builders (existing):** the public primary and PR config builders, as already exercised by the primary and PR model tests. Assertions cover feature columns, sample filters, subgroups, age bands, categorical columns, metadata role and sample label, and shared pipeline settings.
- **Sample reconcilers (existing):** the public reconcilers, asserted through their documented `{status, summary, checks}` output. Existing assertions on summary values stay valid.
- **Comparison accessor (existing):** asserted directly for metric values, weighted values, run-level fields, and missing names, plus the existing report-rendering tests that pin the generated tables.
- **CLI guards (existing pattern):** load each entry point as a module, call its main with arguments, assert exit code and stderr. Used for a missing feature table on both previously untested entry points.
- **Rerun (operational):** no unit seam. Acceptance is the sample-verification command exiting successfully, every new run record naming the target interpreter, the comparison regenerating with all required sections, and the tests passing.

**Modules tested:** the primary model config, the PR model configs, the sample reconcilers for both families, the run summary function, the comparison accessor and renderer, and the two command-line entry points.

**Prior art:** the existing primary and PR model tests for config assertions, the comparison tests for accessor and renderer behaviour, and the three existing CLI test modules for the importlib-plus-main pattern.

## Out of Scope

- The PR subgroup audit note's "household head, not necessarily the respondent" caveat and the PR sensitivity model-card addendum. Both belong to ticket 08.
- Any change to model families, hyperparameter candidates, calibration options, feature sets, thresholds, or the sensitivity target.
- Retuning or redefining the primary or PR samples; the rerun must reproduce the documented counts.
- Rebuilding the feature tables unless reconciliation fails.
- Publishing anything that is currently local-only, or relaxing the gitignore rules.
- Supporting Python 3.14 as a documented target.
- Writing a glossary or new architecture decision records.
- Republishing the variant runs under the PR variants output.

## Further Notes

**Prerequisite risk.** This machine has only Python 3.14.6; no 3.11 runtime is installed. The rerun needs a 3.11 interpreter installed through the Python launcher, a fresh virtual environment, and a reinstall of the project with its dev extras. If 3.11 cannot be installed or the dependency set does not build on it, the work stops and reports that; it does not publish a second interpreter's results alongside the first.

**Expected drift.** The refactor must not move any number. The rerun may move metrics slightly through library-version differences alone; the README table takes its values from the new run records rather than from the previous table.

**Artifact replacement.** Deleting the old run directories is part of the change, not a follow-up, so that the repository never carries two published answer sets for the same three models.

**Timing.** Three full runs plus the comparison are long-running; the smoke path stays available and should be exercised once on the new interpreter before the full runs, as the ticket 07 workflow requires.
