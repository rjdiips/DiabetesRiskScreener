# 01: PR feature table and audit

**What to build:** Turn the raw PR household-member file into a harmonised feature table that the primary training pipeline can consume, and publish an aggregate PR feature audit. The table covers adults aged 15+ with valid glucose and a known target, mirrors the IR/MR schema where a concept exists, retains anthropometry where the PR file measures it, and adds the PR-specific cooking-fuel variable. Concepts the PR file does not contain are explicit missing values. The builder fails loudly on missing required columns or invalid required codes.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] `build_pr_features(raw_pr)` returns a table with the harmonised IR/MR columns plus `cooking_fuel` and `cooking_fuel_clean`.
- [x] The sample filter retains adults aged 15–98 with sex codes 1 or 2, valid random glucose (20–600 mg/dL), and a known diagnosis/medication status; it excludes sex code 3, invalid glucose, and rows with no known target.
- [x] `target` is glucose > 140 mg/dL OR diagnosed OR medication; `target_fact_sheet` is glucose > 140 mg/dL OR medication.
- [x] BP summarisation averages valid systolic/diastolic pairs, counts readings, flags high BP, and leaves rows with no valid pair missing.
- [x] Anthropometry uses the women's variables for women and the men's variables for men; missing codes and out-of-range values become missing.
- [x] Education code 8, social-group code 8, tobacco code 8, and alcohol code 8 become missing; social-group codes 1–4 are retained.
- [x] `cooking_fuel_clean` is clean for electricity, LPG, and biogas; not clean for kerosene through animal dung; missing for no-food and other.
- [x] `recode` is `"PR"`, `case_id` is unique, and `age_15_49` is correct.
- [x] Unavailable concepts — marital status, children ever born, smoking/smokeless split, alcohol frequency, and diet — are present as explicit missing columns.
- [x] The builder raises a clear error on missing required columns or invalid required codes.
- [x] The builder is deterministic for the same input.
- [x] A thin command-line wrapper writes the local PR feature table and the aggregate PR feature audit.
- [x] The real-data audit reconciles to 1,808,013 rows, 253,421 positives, 14.02% prevalence, 962,219 women, 845,794 men, 1,291,831 aged 15–49, and 516,182 aged 50+.
- [x] The real-data audit reconciles BP validity to 1,805,697 overall, 1,290,219 aged 15–49, and 515,478 aged 50+.
- [x] The audit records missingness, BP coverage, cluster and household counts, and weighted prevalence.
- [x] Synthetic PR-shaped tests cover the sample filter, target and fact-sheet target, BP summarisation, anthropometry by sex, code handling, harmonised schema, optional columns, and determinism.
- [x] The tests pass without the real NFHS-5 microdata.

## Comments

Implemented uild_pr_features(raw_pr) in src/screener/features.py, with pure helpers for
code cleaning, sex-specific anthropometry, and cooking-fuel classification. Added
uild_pr_feature_audit and 
econcile_pr_audit, plus scripts/build_pr_features.py.

Real-data run reconciles exactly: 1,808,013 rows, 253,421 positives, 14.02%, 962,219 women,
845,794 men, 1,291,831 aged 15-49, 516,182 aged 50+, BP-valid 1,805,697 / 1,290,219 /
515,478, and BP-valid positives 253,091 / 112,898 / 140,193. Synthetic tests: 17 in
	ests/test_pr_features.py; full suite 31 passing. Outputs are local-only under
data/processed/ plus the aggregate data/pr_feature_audit.json.
