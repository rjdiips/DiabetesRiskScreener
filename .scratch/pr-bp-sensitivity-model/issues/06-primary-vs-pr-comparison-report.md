# 06: Primary vs PR comparison report

**What to build:** Produce a side-by-side comparison of the primary IR/MR 15–49 model, the PR 15–49 age-matched model, and the PR 15+ model. The report makes the sample and feature-set differences explicit and compares discrimination, calibration, and threshold behaviour rather than raw probability levels. It is written as readable Markdown and as structured JSON.

**Blocked by:** 03, 04, 05; primary-model-pipeline (external spec, not yet implemented).

**Status:** ready-for-agent

- [ ] A pure comparison helper accepts the primary result and the PR 15+ and PR 15–49 results and returns a structured comparison.
- [ ] The report includes a sample table with row counts, sex composition, age composition, prevalence, and weighted prevalence.
- [ ] The report includes a feature-set table showing the 32-feature primary model and the 12-feature PR model.
- [ ] The report includes overall discrimination, calibration, and threshold metrics for all three runs.
- [ ] The report includes reliability curves.
- [ ] The report includes sex-specific and age-band-specific metrics.
- [ ] The report includes subgroup gaps for the PR model.
- [ ] The report includes a narrative explaining that the PR 15–49 run isolates the feature-set trade-off while the PR 15+ run adds older adults.
- [ ] The report includes a limitations section and states that raw probabilities are not directly comparable across the three models.
- [ ] The report is written as Markdown and as structured JSON.
- [ ] The report does not claim that one model is universally better.
- [ ] Tests cover the complete table for all three runs, correct differences, graceful handling of a missing metric, and the sample/feature-difference notes.
