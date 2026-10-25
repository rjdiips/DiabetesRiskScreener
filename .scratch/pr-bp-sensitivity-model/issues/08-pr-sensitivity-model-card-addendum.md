# 08: PR sensitivity model-card addendum

**What to build:** Write a model-card addendum for the PR BP sensitivity analysis that can be published without microdata. It makes clear that the PR model is a sensitivity analysis, not the deployed screener, and explains what changes when the sample extends to older adults and the feature set is reduced.

**Blocked by:** 07.

**Status:** ready-for-agent

- [ ] The addendum covers intended use, sample and feature differences, the age gradient, calibration, threshold, subgroup gaps, label bias, limitations, and non-goals.
- [ ] It states that the PR model is not the deployed model and not a clinical diagnostic tool.
- [ ] It states that PR religion and social group describe the household head.
- [ ] It states that 183 transgender respondents were excluded.
- [ ] It states that the PR model does not use anthropometry or diet.
- [ ] It quotes real metrics from the PR runs, not smoke-run placeholders.
- [ ] It is publishable without redistributing microdata.
- [ ] It is linked from the comparison report and the project README.
