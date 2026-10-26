# India Diabetes Risk Screener

A research and portfolio project that estimates the probability of elevated random blood glucose or diagnosed diabetes among Indian adults using NFHS-5 (2019-21).

**Status:** data downloaded, audited, and harmonised IR/MR and PR feature tables built; shared model pipeline, primary IR/MR reference run, PR 15+ and PR 15–49 sensitivity runs, PR subgroup/fairness variants, and the primary-versus-PR comparison report implemented; full real-data runs and publishable aggregate results in `artifacts/`.

## What this is

- An India-specific risk screener built on NFHS-5.
- A calibrated model with household-grouped validation.
- A FastAPI service and a bilingual (English/Hindi) demo.
- A calibration, subgroup, and fairness report.

## What this is not

- Not a diagnostic test.
- Not a predictor of future diabetes.
- Not for insurance, employment, or care-allocation decisions.

## Data

NFHS-5 microdata from the DHS Program. Raw data is not committed. See [`data/README.md`](data/README.md) for file names, checksums, variable names, and the audit.

Key audit numbers:

| Sample | Analysis rows | Target positive | Prevalence |
|---|---:|---:|---:|
| PR, adults 15+ | 1,808,013 | 253,421 | 14.02% |
| PR, adults 15-49 | 1,291,831 | 113,041 | 8.75% |
| IR + MR, women 15-49 and men 15-54 | 782,789 | 66,450 | 8.49% |
| IR + MR, adults 15-49 | 774,861 | 64,526 | 8.33% |

These counts exclude rows with missing diagnosis/medication status and missing sex. Before that exclusion, the valid-glucose counts are 1,812,179 (PR 15+), 1,295,578 (PR 15-49), and 785,388 (IR + MR 15-54).

The IR/MR pooled sample has the full questionnaire, anthropometry, and BP feature set. The PR sample supports a larger BP-based model.

## Repository layout

```text
data/               # local only; raw/ and processed/ are gitignored
scripts/            # audit and pipeline entry points
src/screener/       # data, features, model, API
tests/              # unit and data tests
```

## Setup

Use Python 3.11 or 3.12 for the modeling stack (LightGBM/SHAP wheels may lag on 3.14).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Data audit

```powershell
python scripts/audit_data.py
```

This reads the local NFHS-5 files, prints sample counts, and writes `data/audit.json`.

## Feature table

```powershell
python scripts/build_features.py
```

This harmonises the women's (IR) and men's (MR) recodes, merges men's height, weight, and BMI from the PR household-member file, and writes:

- `data/processed/irmr_features.parquet` — 782,789 rows at ages 15-54, with an `age_15_49` flag for the primary 774,861-row sample
- `data/feature_audit.json` — prevalence, missingness, model-input list, and merge validation
- `data/merge_validation.json` — standalone PR-to-MR merge report

The PR-to-MR merge matches all 101,839 MR rows on `mv001/mv002/mv003` to `hv001/hv002/hvidx`, with zero duplicate keys, zero null keys, and zero sex mismatches. Among matched men, 95,997 have valid height, 96,037 have valid weight, and 95,912 have a valid reported BMI. Computed and reported BMI agree closely (n = 95,904, r = 0.999997, mean difference 0.009).

See [`data/README.md`](data/README.md) for the full schema and cleaning rules.

## PR feature table

```powershell
python scripts/build_pr_features.py
```

This harmonises the PR household-member file into the same column schema as the IR/MR table (plus `cooking_fuel` and `cooking_fuel_clean`) and writes:

- `data/processed/pr_features.parquet` — 1,808,013 adults aged 15+ with valid glucose and a known target
- `data/pr_feature_audit.json` — sample counts, prevalence, BP coverage, missingness, survey-design counts, and a reconciliation against the documented reference counts

The PR table retains adults aged 50+ (516,182 rows, 27.20% prevalence) that the IR/MR files cannot cover. BP is valid for 1,805,697 rows (99.87%). Concepts the PR file does not measure — marital status, children ever born, the smoking/smokeless split, alcohol frequency, and diet — are explicit missing columns so the same training code can consume the table. Add `--limit 100000` for a smoke run.

The PR model built on this table is a sensitivity analysis; the primary IR/MR model remains the deployed model.

## PR split assignment

```powershell
python scripts/build_pr_split.py
python scripts/build_pr_split.py --primary-assignment artifacts/primary/<run>/split_assignment.parquet
```

This assigns a leakage-free cluster-grouped 20% test holdout and five development folds. Where a PR cluster appears in a primary split assignment, the PR rows reuse that cluster's split and fold; the remaining PR-only clusters are assigned deterministically with the same seed and stratification rule. The split source is recorded as `reused_primary`, `pr_only`, or `mixed`.

For the published real-data comparison, the primary split was built first, then the PR partition was rebuilt with `--primary-assignment` so the three runs share one cluster partition where the recodes overlap. The full PR partition reports `split_source: mixed`.

It writes:

- `data/processed/pr_split_assignment.parquet` — one row per respondent; local only, gitignored
- `data/pr_split_summary.json` — aggregate split sizes, fold prevalence, sex composition, 15–49 slice, leakage checks, and the split source

The 15–49 comparison is the same partition filtered to younger respondents, not a re-split. The assignment is deterministic for a fixed seed.

## Primary model run

```powershell
python scripts/run_primary_model.py
python scripts/run_primary_model.py --smoke
python scripts/run_primary_model.py --split-assignment artifacts/primary/<run>/split_assignment.parquet
```

This is the deployed IR/MR 15–49 reference model, run through the same shared `run_training_pipeline`. It uses the 32-feature default set (`PRIMARY_DEFAULT_FEATURES`), the `age_15_49` sample flag (774,861 rows, 64,526 positives), and cluster-grouped development/test splits. The default run generates and saves a split assignment; pass `--split-assignment` to reuse an existing partition. It writes the same artifact set as the PR runs under `artifacts/primary/`.

The ICF-reported trade-off is explicit: the primary model has the richer feature set (anthropometry and diet) but covers adults 15–49 and is about 89% female; the PR models trade that feature richness for a much larger, more balanced sample that includes adults 50+.

## PR sensitivity model run

```powershell
python scripts/run_pr_model.py --sample 15plus
python scripts/run_pr_model.py --sample 15plus --smoke
python scripts/run_pr_model.py --sample 15-49 --smoke --smoke-rows 2000
python scripts/run_pr_model.py --sample 15plus --output artifacts/pr/runs
python scripts/run_pr_model.py --sample 15-49 --output artifacts/pr/runs
```

This is a thin wrapper around the shared `run_training_pipeline`; there is no second modeling implementation. The 15+ run uses the BP-valid sample of 1,805,697 adults (253,091 positives), the 12-feature default set (`female`, `age`, `urban`, `wealth`, `education`, `tobacco_any`, `alcohol_any`, `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication`), and the saved PR split assignment. Calibration and the screening threshold are fitted independently on PR out-of-fold predictions.

The `15-49` run is the age-matched comparison. It uses the feature builder's `age_15_49` flag as its sample filter, narrows the reported age bands to 15–24 … 45–49, and otherwise reuses the 15+ config: same feature set, same saved cluster partition filtered to 15–49 rows (no re-splitting), and its own calibration and threshold fitted on 15–49 out-of-fold predictions. The BP-valid 15–49 sample reconciles to 1,290,219 rows and 112,898 positives, and its run metadata and results report record `sample: age_15_49` and `comparison_run: pr_15plus_bp` so the two runs cannot be confused. `--no-split-assignment` is rejected for this sample, because a fresh partition would confound the comparison.

Each run writes a timestamped directory under `artifacts/pr/runs/` with:

- the split assignment, resolved config, and run metadata (split source, seeds, package versions, dataset fingerprint, row counts);
- overall metrics, subgroup metrics, the calibration curve, threshold analysis, feature importance, and the SHAP summary;
- a results report and a model card;
- a serialized model bundle (local only and gitignored).

The primary target and the NFHS fact-sheet target are reported separately. `--smoke` trains on a stratified sample instead of the full table. The PR model is a sensitivity analysis; the primary IR/MR model remains the deployed model.

## PR subgroups, fairness, and feature-set variants

Every PR run reports sex, age band (15–24, 25–34, 35–44, 45–49, 50–59, 60–69, 70+), wealth quintile, residence, education, and state. State rows are reported only where the test set has at least 500 respondents. Each row carries n, positives, prevalence, ROC-AUC, PR-AUC, sensitivity at the chosen threshold, calibration error (ECE), and the gaps versus the overall test set. Subgroups with fewer than 200 respondents or fewer than 50 positives are flagged `reliable = no` and must not be used to claim fairness.

Religion and social group are audited through the fairness variant rather than the default run. They describe the household head, not necessarily the respondent. Excluding a protected attribute does not by itself make a model fair, so the report shows what including those features changes.

Run the default model and the feature-set variants on the same PR sample and split assignment:

```powershell
python scripts/run_pr_variants.py --sample 15plus --smoke
python scripts/run_pr_variants.py --sample 15plus --variants state fairness
```

The state variant adds `state`; the fairness variant adds `religion` and `social_group`; the combined variant adds all three; the cooking-fuel variant adds `cooking_fuel_clean`. The script writes each run's normal artifact set plus a `variant_comparison.json` and `variant_comparison.md` under `artifacts/pr/variants/`. The comparison reports each variant's feature delta and overall metric delta versus the default, with mandatory sex and age-band tables and any fairness-audit subgroups. A higher variant ROC-AUC or PR-AUC is not treated as automatic evidence that the variant is better.

## Primary vs PR comparison report

```powershell
python scripts/run_pr_comparison.py `
  --primary artifacts/primary/<run> `
  --pr-15-49 artifacts/pr/runs/<run> `
  --pr-15-plus artifacts/pr/runs/<run>
```

This reads the three finished run directories and writes `primary_vs_pr_comparison.json` and `primary_vs_pr_comparison.md` under `artifacts/pr/comparison/`. The report compares sample composition, feature sets, discrimination, calibration, operating thresholds, reliability curves, sex- and age-band-specific metrics, and subgroup gaps. It states that the PR 15–49 run isolates the feature-set trade-off while the PR 15+ run adds older adults, and it does not claim that one model is universally better. Raw probabilities are not compared across models because the samples and feature sets differ.

## Verifying the sample counts

```powershell
python scripts/verify_pr_samples.py
```

This reads the local PR and IR/MR feature tables and checks the definition-of-done counts: the PR feature table (1,808,013 rows / 253,421 positives), the BP-valid 15+ sample (1,805,697 / 253,091), the BP-valid 15–49 sample (1,290,219 / 112,898), the BP-valid 50+ sample (515,478 / 140,193), and the primary 15–49 sample (774,861 / 64,526). It prints `pass` or `review` per check and exits non-zero on a mismatch.

## Published real-data results

The full real-data runs were executed on the local NFHS-5 tables with the pinned modeling stack. The stack targets Python 3.11 or 3.12 (see Setup); each run records the interpreter and package versions it actually used in its `run_metadata.json`. The primary run generated the cluster partition; the PR partition was rebuilt with `--primary-assignment` and records `split_source: mixed`. All three runs selected LightGBM and isotonic calibration.

| Run | Rows | Test rows | Features | ROC-AUC | PR-AUC | Brier | ECE | Threshold | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary IR/MR 15–49 | 774,861 | 154,946 | 32 | 0.7099 | 0.2035 | 0.0721 | 0.0014 | 0.0549 | 0.8096 | 0.4558 | 0.1194 | 0.9633 |
| PR 15–49 | 1,290,219 | 257,801 | 12 | 0.6979 | 0.1900 | 0.0761 | 0.0006 | 0.0599 | 0.7958 | 0.4708 | 0.1262 | 0.9600 |
| PR 15+ | 1,805,697 | 360,633 | 12 | 0.7459 | 0.3320 | 0.1077 | 0.0011 | 0.1040 | 0.7925 | 0.5614 | 0.2277 | 0.9431 |

Survey-weighted prevalence is 9.35% for the primary 15–49 run, 9.65% for PR 15–49, and 15.43% for PR 15+. The PR 15+ PR-AUC is higher partly because prevalence is higher in the extended sample; that is not evidence that the PR model is universally better.

Published aggregates:

- `artifacts/primary/primary_irmr_15to49_20261026T083448602742Z_42_4e9c2a9e/` — results report, model card, metrics, calibration/threshold tables, subgroup metrics, feature importance, SHAP summary and plot.
- `artifacts/pr/runs/pr_15to49_bp_20261026T084051779902Z_42_96adb799/` — PR 15–49 age-matched run.
- `artifacts/pr/runs/pr_15plus_bp_20261026T084255023166Z_42_2a9e78b6/` — PR 15+ run.
- `artifacts/pr/comparison/primary_vs_pr_comparison.md` and `.json` — side-by-side comparison.

The split assignments, per-respondent SHAP values, model bundles, and all raw/processed microdata remain local and gitignored.

## Disclaimer

This is a research and portfolio demo. It is not medical advice and must not be used for clinical decisions. Random blood glucose is a screening indicator, not a diagnosis.

## Citation

International Institute for Population Sciences (IIPS) and ICF. 2021. *National Family Health Survey (NFHS-5), 2019-21: India*. Mumbai: IIPS.
