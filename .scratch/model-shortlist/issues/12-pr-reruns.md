# 12: Rerun PR sensitivity models and update the published docs

**What to build:** After the primary winner is frozen, rebuild the PR split against the new primary split assignment, then rerun PR 15+ and PR 15–49 with the same unweighted recipe and candidate set. Regenerate the primary-versus-PR comparison report and the PR sensitivity model-card addendum. Replace the README results table, run paths, and model card with the new records, and check that no environment claim contradicts the interpreter the runs recorded. Mark the previous PR artifacts as the previous recipe and remove them once replaced so one answer set remains. Verify the main spec's selection, feature-bundle, and leakage-safe pipeline language against the frozen run and amend only if a new conflict appears.

**Blocked by:** 11

**Status:** ready-for-agent

- [ ] The PR split is rebuilt with `--primary-assignment` and records the shared partition.
- [ ] PR 15+ and PR 15–49 rerun with the same candidates and unweighted recipe.
- [ ] The primary-versus-PR comparison is regenerated from the new runs and records the selected primary feature bundle.
- [ ] The comparison states that the PR runs keep the PR 12-feature set and do not inherit the primary bundle choice.
- [ ] The PR sensitivity model-card addendum is regenerated.
- [ ] Old PR runs and the old comparison outputs are removed; one answer set remains.
- [ ] Sample verification passes for every documented count.
- [ ] The README results table and run paths quote the new run records exactly, including the selected feature bundle and count.
- [ ] The model card states the deployed family, selected bundle and feature count, threshold, calibration, subgroup gaps, and limitations, and quotes the new run records.
- [ ] The main spec's model-selection section names the pre-registered feature bundles, the feature-bundle rule, and the leakage-safe selection phases, or records why they are delegated to the shortlist spec.
- [ ] No environment claim contradicts the interpreter recorded in the run.
- [ ] The glossary's exploration-bucket, promotion-rule, and ranking-rule entries are checked against the frozen run and still consistent.
- [ ] No microdata, split assignment, per-respondent SHAP values, or model binary is committed.
