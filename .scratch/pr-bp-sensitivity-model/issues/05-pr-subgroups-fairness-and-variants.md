# 05: PR subgroup, fairness, and feature-set variants

**What to build:** Report PR subgroup performance and run the feature-set variants that test the primary model's exclusion decisions. The default model is audited by sex, age band including 50+, wealth, residence, education, and state; religion and social group are audited through the fairness variant. State, fairness, combined, and cooking-fuel variants are compared with the default so the report shows what those features change rather than assuming they are irrelevant.

**Blocked by:** 03.

**Status:** ready-for-agent

- [ ] Subgroup rows exist for sex, age band (15–24, 25–34, 35–44, 45–49, 50–59, 60–69, 70+), wealth quintile, residence, education, and every state with at least 500 test respondents.
- [ ] Each subgroup row reports n, positives, prevalence, ROC-AUC, PR-AUC, sensitivity at the chosen threshold, calibration error, and the gap versus the overall test set.
- [ ] Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged as unreliable.
- [ ] Religion and social group are audited through the fairness variant.
- [ ] The report states that PR religion and social group describe the household head, not necessarily the respondent.
- [ ] The state variant adds `state`; the fairness variant adds `religion` and `social_group`; the combined variant adds all three; the cooking-fuel variant adds `cooking_fuel_clean`.
- [ ] Variant metrics are compared with the default model.
- [ ] The report does not claim fairness from excluding a protected attribute.
- [ ] Sex-specific and age-band-specific metrics are mandatory.
- [ ] Tests assert each variant's feature set and the small-group flagging behaviour.
