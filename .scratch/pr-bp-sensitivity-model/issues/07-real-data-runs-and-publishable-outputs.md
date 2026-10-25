# 07: Real-data PR runs and publishable aggregate outputs

**What to build:** Execute the PR 15+ and PR 15–49 runs on the real NFHS-5 PR table, verify the definition-of-done counts, and publish the aggregate results and comparison report. Microdata and model binaries stay local and gitignored; only aggregate metrics, plots, and reports are publishable.

**Blocked by:** 04, 05, 06.

**Status:** ready-for-agent

- [ ] The PR feature table reconciles to 1,808,013 rows and 253,421 positives.
- [ ] The BP-valid 15+ sample reconciles to 1,805,697 rows and 253,091 positives.
- [ ] The BP-valid 15–49 sample reconciles to 1,290,219 rows and 112,898 positives.
- [ ] The BP-valid 50+ sample reconciles to 515,478 rows and 140,193 positives.
- [ ] The PR 15+ and PR 15–49 runs complete with the required artifacts.
- [ ] The comparison report contains the sample, feature, overall, calibration, threshold, sex, age-band, and subgroup tables.
- [ ] Aggregate metrics, plots, and reports are published; microdata and model binaries are not committed.
- [ ] The project and data documentation record the PR build and run commands and the sample reconciliation.
- [ ] A real-data smoke run completes before the full run.
