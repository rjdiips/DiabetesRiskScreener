# NFHS-5 India 2019-21 — Data Notes

**Source:** The DHS Program, India Standard DHS 2019-21 (NFHS-5)  
**Downloaded:** 22-23 October 2026  
**Access:** Approved DHS project `India Diabetes Risk Screener` (India, Survey)  
**Raw data:** local only; never commit to Git

---

## 1. Files

| File | Contents | ZIP bytes | SHA-256 |
|---|---|---:|---|
| `IAPR7EDT.ZIP` | Household Member Recode | 244,403,841 | `320B7B406D7F826DE46AF45AEF5488C1621A0AB269FFD7AB500AAC0B9D0A9B55` |
| `IAHR7EDT.ZIP` | Household Recode | 402,484,212 | `8B45978D438F7C0A0E313ACBD9E14093B854CE3521A32A2544BE3414943BF712` |
| `IAIR7EDT.ZIP` | Individual Recode (women 15-49) | 384,497,648 | `A67F097E54D2A398EC862E36053AA7E5FA6690821CF21B321995288547471CD6` |
| `IAMR7EDT.ZIP` | Men's Recode (men 15-54) | 21,149,371 | `4CE9D854EC12D35C5DE4A719F3F0B5AE2974797A21D8A779E3D03A2378C941BB` |

Extracted Stata files:

| Recode | Folder | Extracted file | Bytes | Rows | Columns |
|---|---|---|---:|---:|---:|
| PR | `pr_household_member/` | `IAPR7EFL.DTA` | 2,061,992,078 | 2,843,917 | 530 |
| HR | `hr_household/` | `IAHR7EFL.DTA` | 5,213,125,543 | 636,699 | 6,482 |
| IR | `ir_women_15_49/` | `IAIR7EFL.DTA` | 5,196,403,097 | 724,115 | 5,972 |
| MR | `mr_men_15_54/` | `IAMR7EFL.DTA` | 100,358,153 | 101,839 | 861 |

The folder names describe the recode: PR is the household-member file, HR is the household file, IR is the women 15-49 file, and MR is the men 15-54 file.

The PR file is the main household-member file. It has one row per household member and contains the glucose, blood pressure, and anthropometry variables. The HR file is wide because it includes member-level variables for up to 35 members; it is not needed for the primary model because the PR file already carries the relevant `HV` household variables.

---

## 2. Key variables

### PR (household member)

| Concept | Variable | Notes |
|---|---|---|
| Household ID | `hhid` | Grouping key; 636,699 unique households |
| Member line | `hvidx` | Combine with `hhid` for a person key |
| Cluster / household | `hv001` / `hv002` | Survey design |
| Household weight | `hv005` | Divide by 1,000,000 for analysis |
| Household size | `hv009` | |
| State | `hv024` | |
| Urban/rural | `hv025` | 1 = urban, 2 = rural |
| Sex | `hv104` | 1 = male, 2 = female, 3 = missing |
| Age | `hv105` | 15-97 in the analysis; 98 = 98+ |
| Education | `hv106` | |
| Wealth quintile | `hv270` | 1 = poorest ... 5 = richest |
| Cooking fuel | `hv226` | |
| Tobacco | `sh25` | 0/1/8 |
| Alcohol | `sh26` | 0/1/8 |
| BP readings | `shb18s/d`, `shb25s/d`, `shb29s/d` | First, second, third systolic/diastolic |
| Waist / hip | `sh305` / `sh306` | cm |
| Women's height/weight/BMI | `ha3` / `ha2` / `ha40` | Women 15-49 |
| Men's height/weight/BMI | `hb3` / `hb2` / `hb40` | Interviewed men 15-54 |
| Glucose ever checked | `shb55` | |
| Told high glucose | `shb56` | 1 = yes |
| Glucose medication | `shb57` | 1 = yes |
| Glucose reading | `shb74` | mg/dL |

### IR (women) and MR (men)

| Concept | IR variable | MR variable |
|---|---|---|
| Case ID | `caseid` | `mcaseid` |
| Cluster / household / line | `v001` / `v002` / `v003` | `mv001` / `mv002` / `mv003` |
| Weight | `v005` | `mv005` |
| Age | `v012` | `mv012` |
| State | `v024` | `mv024` |
| Urban/rural | `v025` | `mv025` |
| Wealth quintile | `v190` | `mv190` |
| Currently has diabetes | `s728a` | `sm627a` |
| Sought treatment | `s728ab` | `sm627ab` |
| Glucose ever checked | `sb55` | `smb55` |
| Told high glucose | `sb56` | `smb56` |
| Glucose medication | `sb57` | `smb57` |
| Glucose reading | `sb74` | `smb74` |

The IR/MR files also carry the full women's and men's questionnaire, which is where the behavioural and diet variables live.

---

## 3. Glucose cleaning

Valid random blood glucose is **20-600 mg/dL**. The following values in `shb74`, `sb74`, and `smb74` are missing/invalid codes and must be dropped:

- `995` — not measured (35,397 rows in PR)
- `996` — refused/other (1,175 rows in PR)
- `998` — don't know (4,662 rows in PR)

`997` and `999` are not present in this file but should be treated as missing if they appear. Do not recode these as high glucose.

---

## 4. Target definition

Primary target:

```text
positive = glucose > 140 mg/dL
           OR told high blood glucose == yes (`shb56` / `sb56` / `smb56`)
           OR currently taking glucose-lowering medicine == yes (`shb57` / `sb57` / `smb57`)
```

`shb56` means "told high blood glucose on two or more occasions by a doctor or other health professional"; it is a self-reported diagnosis proxy, not a clinical diagnosis.

The NFHS-5 fact-sheet indicator uses **high glucose or taking medicine** and does not include the diagnosis item. The target above is slightly broader and matches the app's wording ("elevated random blood glucose or diagnosed diabetes"). Report both definitions:

- **Primary:** glucose > 140 OR diagnosed OR medication
- **Fact-sheet comparison:** glucose > 140 OR medication

Rows with missing diagnosis and medication status are excluded when `require_known_status=True`; the count is about 4,200 among adults 15+.

---

## 5. Sample counts (valid glucose, age 15+)

| Sample | Valid glucose | Target positive | Prevalence |
|---|---:|---:|---:|
| PR, adults 15+ | 1,812,179 | 253,849 | 14.01% |
| PR, adults 15-49 | 1,295,578 | 113,383 | 8.75% |
| IR, women 15-49 | 690,748 | 55,858 | 8.09% |
| MR, men 15-54 | 94,640 | 10,790 | 11.40% |
| IR + MR pooled | 785,388 | 66,648 | 8.49% |

After excluding rows with missing diagnosis/medication status and missing sex, the stricter analysis samples are:

| Sample | Analysis rows | Target positive | Prevalence |
|---|---:|---:|---:|
| PR, adults 15+ | 1,808,013 | 253,421 | 14.02% |
| PR, adults 15-49 | 1,291,831 | 113,041 | 8.75% |
| IR + MR, 15-54 | 782,789 | 66,450 | 8.49% |
| IR + MR, 15-49 | 774,861 | 64,526 | 8.33% |

Target components in PR adults 15+:

| Component | Prevalence |
|---|---:|
| Glucose > 140 | 12.10% |
| Told high glucose (`shb56`) | 3.79% |
| Glucose medication (`shb57`) | 2.86% |
| Union (primary target) | 14.02% |
| Glucose > 140 OR medication (fact-sheet definition) | 13.12% |

Weighted with `hv005`, the fact-sheet definition is 13.53% for women and 15.58% for men, close to the published NFHS-5 values of 13.9% and 14.7%. This is a useful sanity check.

---

## 6. Harmonised feature table (IR + MR)

Build it with:

```powershell
python scripts/build_features.py
```

This reads the local Stata files, harmonises the women's (IR) and men's (MR) recodes, merges men's height/weight/BMI from the PR household-member file, and writes:

| Output | Contents |
|---|---|
| `data/processed/irmr_features.parquet` | One row per analysis respondent; 54 columns |
| `data/feature_audit.json` | Sample counts, prevalence, missingness, model-input list, merge report |
| `data/merge_validation.json` | Standalone PR-to-MR merge validation |

The table covers ages 15-54 and carries an `age_15_49` flag. The primary model filters to `age_15_49 == True` (774,861 rows); the 15-54 table has 782,789 rows. The final table contains 688,308 IR (women) rows with 8.09% prevalence and 94,481 MR (men) rows with 11.40% prevalence.

### Column groups

| Group | Columns |
|---|---|
| Identifiers and survey design | `case_id`, `recode`, `cluster`, `household`, `line`, `psu`, `strata`, `weight`, `weight_scaled` |
| Demographics | `sex`, `female`, `age`, `age_15_49`, `state`, `residence`, `urban`, `wealth`, `education`, `religion`, `social_group`, `marital_status`, `children_ever_born` |
| Anthropometry | `height_cm`, `weight_kg`, `bmi`, `waist_cm`, `hip_cm`, `waist_hip_ratio` |
| Blood pressure | `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high`, `bp_medication` |
| Target components | `ever_checked`, `diagnosed`, `medication`, `glucose_raw`, `glucose`, `target`, `target_fact_sheet` |
| Tobacco and alcohol | `tobacco_any`, `smoking_any`, `smokeless_any`, `alcohol_any`, `alcohol_frequency` |
| Diet frequency | `diet_milk`, `diet_pulses`, `diet_green_leafy`, `diet_fruit`, `diet_eggs`, `diet_fish`, `diet_meat`, `diet_fried`, `diet_aerated` |

The 35 model inputs are listed in `data/feature_audit.json` under `model_feature_columns`. Identifiers, survey-design fields, target components, redundant raw encodings (`sex`, `residence`), and the `age_15_49` sample flag are excluded from `MODEL_EXCLUDED_COLUMNS`. The table keeps both encodings so reports can use the raw survey codes while the model uses `female` and `urban`.

### Cleaning rules

| Measurement | Source | Scaling | Valid range |
|---|---|---|---|
| Women's height / weight / BMI | IR `v438` / `v437` / `v445` | /10, /10, /100 | 100-250 cm, 20-300 kg, 12-60 |
| Men's height / weight / BMI | PR `hb3` / `hb2` / `hb40` | /10, /10, /100 | 100-250 cm, 20-300 kg, 12-60 |
| Waist / hip | IR `s305`/`s306`, MR `smb305`/`smb306` | already cm | 30-250 cm |
| Blood pressure | `*b18s/d`, `*b25s/d`, `*b29s/d` | already mmHg | systolic 50-300, diastolic 30-200 |

Only valid systolic/diastolic **pairs** are averaged. `bp_high` is 1 when mean systolic >= 140 or mean diastolic >= 90. Reported BMI is used when valid; otherwise BMI is computed from cleaned height and weight.

Diet variables use the NFHS frequency codes 0-3 (never / daily / weekly / occasionally) and are kept as ordinal inputs.

### PR-to-MR merge validation

Men's height, weight, and BMI exist in PR (`hb2`, `hb3`, `hb40`) but not in MR. The merge is:

```text
MR mv001 / mv002 / mv003  ->  PR hv001 / hv002 / hvidx
```

Validation results (`data/merge_validation.json`):

| Check | Result |
|---|---:|
| MR rows matched to PR | 101,839 / 101,839 (100%) |
| Duplicate or null keys | 0 |
| Sex mismatch | 0 |
| Age exact match | 88,618 |
| Age within 1 year | 97,118 (95.4%) |
| Age within 2 years | 99,203 (97.4%) |
| Maximum absolute age difference | 34 years |
| Valid height among matched MR rows | 95,997 (94.3%) |
| Valid weight among matched MR rows | 96,037 (94.3%) |
| Valid reported BMI among matched MR rows | 95,912 |
| Computed vs reported BMI (both valid) | 95,904 |
| BMI correlation | 0.999997 |
| Mean BMI difference (computed - reported) | 0.008974 |
| Median absolute BMI difference | 0.004838 |

Status: **pass**. Age differences are expected because the household roster (`hv105`) and the men's questionnaire (`mv012`) can be completed at different times; the merge does not require exact age equality.

In the final analysis table, 94,221 of 94,481 MR rows (99.7%) have merged height, 94,258 have weight, and 94,140 have BMI.

---

## 7. PR feature table

Build it with:

```powershell
python scripts/build_pr_features.py
```

This harmonises the PR household-member file into the same column schema as the IR/MR table and writes:

| Output | Contents |
|---|---|
| `data/processed/pr_features.parquet` | One row per analysis respondent; 56 columns (the 54 harmonised columns plus `cooking_fuel` and `cooking_fuel_clean`) |
| `data/pr_feature_audit.json` | Sample counts, prevalence, BP coverage, missingness, survey-design counts, and reconciliation against the reference counts |

The sample is adults aged 15-98 with sex codes 1 or 2, random glucose in 20-600 mg/dL, and a known diagnosis/medication status. The 183 respondents coded as transgender (sex code 3) are excluded. The table has 1,808,013 rows and 253,421 positives (14.02%): 1,291,831 adults aged 15-49 (8.75%) and 516,182 aged 50+ (27.20%). It contains 962,219 women and 845,794 men.

### PR source mapping

| Harmonised column | PR source | Decision |
|---|---|---|
| `case_id` | `hhid` + `hvidx` | Stable person identifier |
| `recode` | constant | `"PR"` |
| `sex`, `female` | `hv104` | 1 male, 2 female; code 3 excluded |
| `cluster`, `household`, `line` | `hv001`, `hv002`, `hvidx` | Grouping keys |
| `psu`, `strata` | `hv021`, `hv022` | Survey design |
| `weight`, `weight_scaled` | `hv005`, `hv005 / 1,000,000` | Survey weight |
| `age`, `age_15_49` | `hv105` | 15-98; 98 means 98+; 15-49 sample flag |
| `state`, `residence`, `urban` | `hv024`, `hv025` | Raw codes plus binary urban |
| `wealth`, `education` | `hv270`, `hv106` | Education 0-3; code 8 becomes missing |
| `religion`, `social_group` | `sh47`, `sh49` | Household-head attributes; social group 1-4, code 8 becomes missing |
| `height_cm`, `weight_kg`, `bmi` | `ha3`/`ha2`/`ha40` (women), `hb3`/`hb2`/`hb40` (men) | Valid ranges; not used by the default PR model |
| `waist_cm`, `hip_cm`, `waist_hip_ratio` | `sh305`, `sh306` | Valid ranges; not used by the default PR model |
| `systolic_mean`, `diastolic_mean`, `bp_readings`, `bp_high` | `shb18s/d`, `shb25s/d`, `shb29s/d` | Existing BP summarisation |
| `bp_medication` | `shb21` | 0/1 |
| `ever_checked`, `diagnosed`, `medication` | `shb55`, `shb56`, `shb57` | 0/1 |
| `glucose_raw`, `glucose` | `shb74` | Valid range 20-600 mg/dL |
| `target`, `target_fact_sheet` | constructed | Same definitions as IR/MR |
| `tobacco_any`, `alcohol_any` | `sh25`, `sh26` | 0/1; code 8 becomes missing |
| `cooking_fuel`, `cooking_fuel_clean` | `hv226` | PR-specific; clean = electricity (1), LPG (2), or biogas (4); no food (95) and other (96) missing |
| `marital_status`, `children_ever_born`, `smoking_any`, `smokeless_any`, `alcohol_frequency`, `diet_*` | not available | Explicit missing columns |

### PR audit

`data/pr_feature_audit.json` records:

| Check | Result |
|---|---|
| Rows / positives / prevalence | 1,808,013 / 253,421 / 14.02% |
| Women / men | 962,219 / 845,794 |
| Adults 15-49 / 50+ | 1,291,831 / 516,182 |
| BP-valid rows | 1,805,697 (99.87%) |
| BP-valid 15-49 / 50+ | 1,290,219 / 515,478 |
| BP-valid positives 15+ / 15-49 / 50+ | 253,091 / 112,898 / 140,193 |
| Clusters / households | 30,125 / 614,792 |
| Weighted target prevalence | 15.39% (women 14.50%, men 16.42%) |
| Weighted fact-sheet prevalence | 14.48% |
| Clean cooking fuel | 943,827 (52.29% of non-missing) |

The builder validates required columns and value codes and raises a `ValueError` on a malformed input. It is deterministic for the same input. The audit reconciles against these reference counts and reports `pass` or `review`.

### PR split assignment

Build it after the PR feature table:

```powershell
python scripts/build_pr_split.py
python scripts/build_pr_split.py --primary-assignment artifacts/primary/split_assignment.parquet
```

| Output | Contents |
|---|---|
| `data/processed/pr_split_assignment.parquet` | One row per respondent with `case_id`, `cluster`, `household`, `split`, `fold`, and `assignment_source`; local only |
| `data/pr_split_summary.json` | Aggregate split sizes, fold prevalence, sex composition, the 15-49 slice, leakage checks, and the split source |

The partition is cluster-grouped: a 20% test holdout and five development folds. Clusters that overlap the primary split assignment keep the primary split and fold. PR-only clusters are assigned deterministically from the same seed, stratified by cluster-level target prevalence and female share. The summary records `split_source` as `reused_primary`, `pr_only`, or `mixed`. The 15-49 comparison is the same partition filtered to younger rows, with no re-splitting.

Household identity is the `(cluster, household)` pair because DHS `hv002` household numbers repeat across clusters.

---

## 8. Which sample to model?

There are two useful samples:

1. **IR + MR pooled (785,388 valid glucose; 782,789 after target-status exclusion)**  
   Full women's/men's questionnaire, anthropometry, BP, and survey design variables. This is the best primary sample for the measurement model with BMI, waist, and behaviours. Women are 15-49 and men 15-54; for a common age range, restrict to 15-49 (774,861 rows) and report the men 50-54 sensitivity.

2. **PR adults 15+ (1,812,179 valid glucose; 1,808,013 after target-status exclusion)**  
   Larger and includes older adults, with glucose, BP, tobacco, alcohol, education, wealth, and rural/urban. BMI/waist/hip are available for women 15-49 and interviewed men 15-54, but not for adults 50+. This supports a large-sample BP-based model, not the full anthropometry model.

The earlier project spec's "~1.25M adults with valid glucose" refers to the **PR age 15-49** subset, not the full-feature IR/MR sample. Use the actual sample size of the model that is built:

- 782,789 for the full-feature IR/MR model (774,861 if restricted to 15-49)
- 1,808,013 for the PR 15+ BP-based model

Do not claim 1.25M unless the model is specifically the PR 15-49 subset.

---

## 9. Terms and hygiene

- Do not redistribute the raw or processed microdata.
- Do not commit `data/raw/` or `data/processed/` to Git.
- Publish only code, aggregate metrics, plots, and a model card.
- Cite NFHS-5/IIPS and the DHS Program in public outputs.
- Do not attempt re-identification.
- Random glucose is a screening indicator, not a diagnosis.

**Citation:**

> International Institute for Population Sciences (IIPS) and ICF. 2021. *National Family Health Survey (NFHS-5), 2019-21: India*. Mumbai: IIPS.
