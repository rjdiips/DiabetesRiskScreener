"""Build the harmonised IR/MR and PR feature tables and validate the PR-to-MR merge."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import AGE_MAX, AGE_MIN, GLUCOSE_THRESHOLD, IR_PATH, MR_PATH, PR_PATH
from .data import build_target, clean_glucose

IR_FEATURE_COLUMNS = [
    "caseid",
    "v001",
    "v002",
    "v003",
    "v005",
    "v012",
    "v021",
    "v022",
    "v024",
    "v025",
    "v106",
    "v130",
    "s116",
    "v190",
    "v501",
    "v201",
    "v437",
    "v438",
    "v445",
    "s305",
    "s306",
    "sb18s",
    "sb18d",
    "sb25s",
    "sb25d",
    "sb29s",
    "sb29d",
    "sb21",
    "sb55",
    "sb56",
    "sb57",
    "sb73",
    "sb74",
    "v463a",
    "v463b",
    "v463c",
    "v463d",
    "v463f",
    "v463g",
    "v463j",
    "v463k",
    "v463l",
    "v463x",
    "v463z",
    "s720",
    "s721",
    "s731a",
    "s731b",
    "s731c",
    "s731d",
    "s731e",
    "s731f",
    "s731g",
    "s731h",
    "s731i",
]

MR_FEATURE_COLUMNS = [
    "mcaseid",
    "mv001",
    "mv002",
    "mv003",
    "mv005",
    "mv012",
    "mv021",
    "mv022",
    "mv024",
    "mv025",
    "mv106",
    "mv130",
    "sm118",
    "mv190",
    "mv501",
    "mv201",
    "smb305",
    "smb306",
    "smb18s",
    "smb18d",
    "smb25s",
    "smb25d",
    "smb29s",
    "smb29d",
    "smb21",
    "smb55",
    "smb56",
    "smb57",
    "smb73",
    "smb74",
    "mv463a",
    "mv463b",
    "mv463c",
    "mv463d",
    "mv463e",
    "mv463f",
    "mv463g",
    "mv463x",
    "mv463z",
    "sm619",
    "sm620",
    "sm630a",
    "sm630b",
    "sm630c",
    "sm630d",
    "sm630e",
    "sm630f",
    "sm630g",
    "sm630h",
    "sm630i",
]

PR_MEN_ANTHRO_COLUMNS = [
    "hhid",
    "hvidx",
    "hv001",
    "hv002",
    "hv104",
    "hv105",
    "hb2",
    "hb3",
    "hb40",
]

PR_SOURCE_COLUMNS = [
    "hhid",
    "hvidx",
    "hv001",
    "hv002",
    "hv005",
    "hv021",
    "hv022",
    "hv024",
    "hv025",
    "hv104",
    "hv105",
    "hv106",
    "hv270",
    "hv226",
    "sh47",
    "sh49",
    "sh25",
    "sh26",
    "shb18s",
    "shb18d",
    "shb25s",
    "shb25d",
    "shb29s",
    "shb29d",
    "shb21",
    "sh305",
    "sh306",
    "ha2",
    "ha3",
    "ha40",
    "hb2",
    "hb3",
    "hb40",
    "shb55",
    "shb56",
    "shb57",
    "shb74",
]

# DHS value codes accepted for required PR source columns. Values outside these
# sets indicate a mis-specified file rather than a row-level missing code.
PR_REQUIRED_CODES = {
    "hv025": {1, 2},
    "hv104": {1, 2, 3},
    "hv106": {0, 1, 2, 3, 8},
    "hv270": {1, 2, 3, 4, 5},
    "hv226": {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 95, 96},
    "sh49": {1, 2, 3, 4, 8},
    "sh25": {0, 1, 8},
    "sh26": {0, 1, 8},
    "shb21": {0, 1},
    "shb55": {0, 1},
    "shb56": {0, 1},
    "shb57": {0, 1},
}

# NFHS-5 India's hv226 labels: 1 electricity, 2 LPG, 4 biogas, 5 kerosene,
# 6 coal/lignite, 7 charcoal, 8 wood, 9 straw/shrubs/grass, 10 agricultural
# crop, 11 animal dung, 95 no food cooked in house, 96 other.
CLEAN_COOKING_FUEL_CODES = {1, 2, 4}
DIRTY_COOKING_FUEL_CODES = {5, 6, 7, 8, 9, 10, 11}
MISSING_COOKING_FUEL_CODES = {95, 96}

PR_WOMEN_ANTHRO_AGES = (15, 49)
PR_MEN_ANTHRO_AGES = (15, 54)

PR_AGE_MIN = 0
PR_AGE_MAX = 98

# Documented NFHS-5 PR reference counts used to reconcile the real-data audit.
PR_REFERENCE_COUNTS = {
    "rows": 1_808_013,
    "positive": 253_421,
    "female": 962_219,
    "male": 845_794,
    "age_15_49_rows": 1_291_831,
    "age_50_plus_rows": 516_182,
    "bp_valid_rows": 1_805_697,
    "bp_valid_age_15_49": 1_290_219,
    "bp_valid_age_50_plus": 515_478,
    "bp_positive_rows": 253_091,
    "bp_positive_age_15_49": 112_898,
    "bp_positive_age_50_plus": 140_193,
}

IR_DIET_COLUMNS = {
    "diet_milk": "s731a",
    "diet_pulses": "s731b",
    "diet_green_leafy": "s731c",
    "diet_fruit": "s731d",
    "diet_eggs": "s731e",
    "diet_fish": "s731f",
    "diet_meat": "s731g",
    "diet_fried": "s731h",
    "diet_aerated": "s731i",
}

MR_DIET_COLUMNS = {
    "diet_milk": "sm630a",
    "diet_pulses": "sm630b",
    "diet_green_leafy": "sm630c",
    "diet_fruit": "sm630d",
    "diet_eggs": "sm630e",
    "diet_fish": "sm630f",
    "diet_meat": "sm630g",
    "diet_fried": "sm630h",
    "diet_aerated": "sm630i",
}

IR_TOBACCO_COLUMNS = [
    "v463a",
    "v463b",
    "v463c",
    "v463d",
    "v463f",
    "v463g",
    "v463j",
    "v463k",
    "v463l",
    "v463x",
]
IR_SMOKING_COLUMNS = ["v463a", "v463b", "v463f", "v463g"]
IR_SMOKELESS_COLUMNS = ["v463c", "v463d", "v463j", "v463k", "v463l"]

MR_TOBACCO_COLUMNS = [
    "mv463a",
    "mv463b",
    "mv463c",
    "mv463d",
    "mv463e",
    "mv463f",
    "mv463g",
    "mv463x",
]
MR_SMOKING_COLUMNS = ["mv463a", "mv463b", "mv463e", "mv463f"]
MR_SMOKELESS_COLUMNS = ["mv463c", "mv463d", "mv463g"]

FEATURE_COLUMNS = [
    "case_id",
    "recode",
    "sex",
    "female",
    "cluster",
    "household",
    "line",
    "psu",
    "strata",
    "weight",
    "weight_scaled",
    "age",
    "age_15_49",
    "state",
    "residence",
    "urban",
    "wealth",
    "education",
    "religion",
    "social_group",
    "marital_status",
    "children_ever_born",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "hip_cm",
    "waist_hip_ratio",
    "systolic_mean",
    "diastolic_mean",
    "bp_readings",
    "bp_high",
    "bp_medication",
    "ever_checked",
    "diagnosed",
    "medication",
    "glucose_raw",
    "glucose",
    "target",
    "target_fact_sheet",
    "tobacco_any",
    "smoking_any",
    "smokeless_any",
    "alcohol_any",
    "alcohol_frequency",
    "diet_milk",
    "diet_pulses",
    "diet_green_leafy",
    "diet_fruit",
    "diet_eggs",
    "diet_fish",
    "diet_meat",
    "diet_fried",
    "diet_aerated",
]

PR_EXTRA_COLUMNS = ["cooking_fuel", "cooking_fuel_clean"]
PR_HARMONISED_COLUMNS = FEATURE_COLUMNS + PR_EXTRA_COLUMNS

# Concepts the PR household-member file does not measure. They stay explicit
# missing columns so the PR table keeps the IR/MR schema.
PR_UNAVAILABLE_COLUMNS = [
    "marital_status",
    "children_ever_born",
    "smoking_any",
    "smokeless_any",
    "alcohol_frequency",
    *IR_DIET_COLUMNS,
]

ID_COLUMNS = [
    "case_id",
    "recode",
    "cluster",
    "household",
    "line",
    "psu",
    "strata",
    "weight",
    "weight_scaled",
]

TARGET_COLUMNS = [
    "glucose_raw",
    "glucose",
    "ever_checked",
    "diagnosed",
    "medication",
    "target",
    "target_fact_sheet",
]

SAMPLE_COLUMNS = ["age_15_49"]

# Alternative encodings kept in the table for reporting but not fed to the model.
REDUNDANT_ENCODINGS = ["sex", "residence"]

MODEL_EXCLUDED_COLUMNS = ID_COLUMNS + TARGET_COLUMNS + SAMPLE_COLUMNS + REDUNDANT_ENCODINGS
MODEL_FEATURE_COLUMNS = [
    col for col in FEATURE_COLUMNS if col not in MODEL_EXCLUDED_COLUMNS
]


def _read_stata(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read selected DHS columns without converting value labels."""
    return pd.read_stata(path, columns=list(columns), convert_categoricals=False)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _binary(series: pd.Series) -> pd.Series:
    """Return 0/1 as nullable integers, treating other codes as missing."""
    values = _numeric(series)
    return values.where(values.isin([0, 1])).astype("Int64")


def clean_scaled(series: pd.Series, scale: float, minimum: float, maximum: float) -> pd.Series:
    """Divide a DHS integer-scaled measurement by ``scale`` and apply a valid range."""
    values = _numeric(series) / scale
    return values.where(values.between(minimum, maximum))


def clean_cm(series: pd.Series) -> pd.Series:
    """Clean a waist/hip circumference in centimetres."""
    values = _numeric(series)
    return values.where(values.between(30.0, 250.0))


def clean_systolic(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    return values.where(values.between(50.0, 300.0))


def clean_diastolic(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    return values.where(values.between(30.0, 200.0))


def bmi_from_height_weight(height_cm: pd.Series, weight_kg: pd.Series) -> pd.Series:
    """Compute BMI and keep the plausible adult range."""
    height_m = height_cm / 100.0
    bmi = weight_kg / height_m.pow(2)
    return bmi.where(bmi.between(12.0, 60.0))


def waist_hip_ratio(waist_cm: pd.Series, hip_cm: pd.Series) -> pd.Series:
    """Compute waist-to-hip ratio and drop implausible values."""
    ratio = waist_cm / hip_cm
    return ratio.where(ratio.between(0.5, 1.5))


def summarize_bp(
    df: pd.DataFrame,
    systolic_columns: Sequence[str],
    diastolic_columns: Sequence[str],
) -> pd.DataFrame:
    """Average valid BP pairs, count readings, and flag elevated BP."""
    systolic_frames = []
    diastolic_frames = []
    for systolic_col, diastolic_col in zip(systolic_columns, diastolic_columns, strict=True):
        systolic = clean_systolic(df[systolic_col])
        diastolic = clean_diastolic(df[diastolic_col])
        valid_pair = systolic.notna() & diastolic.notna()
        systolic_frames.append(systolic.where(valid_pair))
        diastolic_frames.append(diastolic.where(valid_pair))

    systolic_values = pd.concat(systolic_frames, axis=1)
    diastolic_values = pd.concat(diastolic_frames, axis=1)

    readings = systolic_values.notna().sum(axis=1).astype("Int64")
    systolic_mean = systolic_values.mean(axis=1)
    diastolic_mean = diastolic_values.mean(axis=1)
    elevated = ((systolic_mean >= 140.0) | (diastolic_mean >= 90.0)).astype("Int64")
    elevated = elevated.where(systolic_mean.notna() & diastolic_mean.notna())

    return pd.DataFrame(
        {
            "systolic_mean": systolic_mean,
            "diastolic_mean": diastolic_mean,
            "bp_readings": readings,
            "bp_high": elevated,
        },
        index=df.index,
    )


def any_yes(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return 1 if any listed 0/1 variable is yes, 0 if all are no, NA if all missing."""
    values = df[list(columns)].apply(pd.to_numeric, errors="coerce")
    result = values.eq(1).any(axis=1).astype("Int64")
    return result.where(~values.isna().all(axis=1))


def harmonize_alcohol(
    df: pd.DataFrame,
    any_column: str,
    frequency_column: str,
) -> pd.Series:
    """Harmonise alcohol use: 0 none, 1 daily, 2 weekly, 3 less than weekly."""
    any_use = _binary(df[any_column])
    frequency = _numeric(df[frequency_column])
    result = pd.Series(pd.NA, index=df.index, dtype="Int64")
    result = result.mask(any_use.eq(0), 0)
    result = result.mask(any_use.eq(1), frequency)
    return result.astype("Int64")


def fact_sheet_target(
    df: pd.DataFrame,
    glucose_column: str,
    medication_column: str,
) -> pd.Series:
    """Build the NFHS fact-sheet comparison target: high glucose or medication."""
    glucose = clean_glucose(df[glucose_column])
    medication = _binary(df[medication_column])
    target = ((glucose > GLUCOSE_THRESHOLD) | medication.eq(1)).astype("Int64")
    return target.where(glucose.notna() & medication.notna())


def clean_education(series: pd.Series) -> pd.Series:
    """Return the 0-3 education codes, with code 8 (missing) set to NA."""
    values = _numeric(series)
    return values.where(values.isin([0, 1, 2, 3])).astype("Int64")


def clean_social_group(series: pd.Series) -> pd.Series:
    """Return social-group codes 1-4, with code 8 (missing) set to NA."""
    values = _numeric(series)
    return values.where(values.isin([1, 2, 3, 4])).astype("Int64")


def clean_cooking_fuel_codes(series: pd.Series) -> pd.Series:
    """Retain known NFHS-5 cooking-fuel codes and set unknown codes to NA."""
    values = _numeric(series)
    known_codes = sorted(PR_REQUIRED_CODES["hv226"])
    return values.where(values.isin(known_codes)).astype("Int64")


def cooking_fuel_clean(series: pd.Series) -> pd.Series:
    """Classify clean cooking fuel: electricity, LPG, or biogas.

    Kerosene through animal dung are not clean. No food cooked in the house
    and other fuel are missing.
    """
    values = clean_cooking_fuel_codes(series)
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result = result.mask(values.isin(CLEAN_COOKING_FUEL_CODES), 1)
    result = result.mask(values.isin(DIRTY_COOKING_FUEL_CODES), 0)
    return result


def build_pr_anthropometry(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean PR anthropometry, using the women's or men's variables by sex.

    Women's anthropometry is measured at ages 15-49 and men's at ages 15-54;
    values outside those ranges are treated as missing.
    """
    sex = _numeric(raw["hv104"])
    age = _numeric(raw["hv105"])
    female = sex.eq(2)
    female_measured = female & age.between(*PR_WOMEN_ANTHRO_AGES)
    male_measured = sex.eq(1) & age.between(*PR_MEN_ANTHRO_AGES)

    height_cm = clean_scaled(raw["ha3"], 10.0, 100.0, 250.0).where(female_measured)
    height_cm = height_cm.where(
        female_measured, clean_scaled(raw["hb3"], 10.0, 100.0, 250.0).where(male_measured)
    )
    weight_kg = clean_scaled(raw["ha2"], 10.0, 20.0, 300.0).where(female_measured)
    weight_kg = weight_kg.where(
        female_measured, clean_scaled(raw["hb2"], 10.0, 20.0, 300.0).where(male_measured)
    )
    reported_bmi = clean_scaled(raw["ha40"], 100.0, 12.0, 60.0).where(female_measured)
    reported_bmi = reported_bmi.where(
        female_measured, clean_scaled(raw["hb40"], 100.0, 12.0, 60.0).where(male_measured)
    )
    calculated_bmi = bmi_from_height_weight(height_cm, weight_kg)

    out = pd.DataFrame(index=raw.index)
    out["height_cm"] = height_cm
    out["weight_kg"] = weight_kg
    out["bmi"] = reported_bmi.where(reported_bmi.notna(), calculated_bmi)
    return out


def _validate_pr_input(raw: pd.DataFrame) -> None:
    """Fail loudly on missing required columns or unexpected value codes."""
    missing = [column for column in PR_SOURCE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            "PR input is missing required columns: " + ", ".join(missing)
        )

    for column, allowed in PR_REQUIRED_CODES.items():
        values = _numeric(raw[column])
        unexpected = sorted(set(values.dropna().unique()) - allowed)
        if unexpected:
            raise ValueError(
                f"PR column '{column}' contains unexpected codes: {unexpected}"
            )

    ages = _numeric(raw["hv105"])
    invalid_ages = ages.notna() & ~ages.between(PR_AGE_MIN, PR_AGE_MAX)
    if invalid_ages.any():
        raise ValueError(
            f"PR column 'hv105' contains {int(invalid_ages.sum())} ages outside "
            f"{PR_AGE_MIN}-{PR_AGE_MAX}"
        )

    hhid = raw["hhid"].astype("string").str.strip()
    line = _numeric(raw["hvidx"])
    if hhid.isna().any() or hhid.eq("").any():
        raise ValueError("PR input has missing household identifiers in 'hhid'")
    if line.isna().any():
        raise ValueError("PR input has missing member lines in 'hvidx'")

    duplicates = pd.DataFrame({"hhid": hhid, "hvidx": line}).duplicated().sum()
    if duplicates:
        raise ValueError(
            f"PR input has {int(duplicates)} duplicate hhid/hvidx person keys"
        )


def prepare_pr_men_anthropometry(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean PR height, weight, and BMI for merging into the men's recode."""
    out = pd.DataFrame(index=raw.index)
    out["cluster"] = _numeric(raw["hv001"]).astype("Int64")
    out["household"] = _numeric(raw["hv002"]).astype("Int64")
    out["line"] = _numeric(raw["hvidx"]).astype("Int64")
    out["pr_age"] = _numeric(raw["hv105"]).astype("Int64")
    out["pr_sex"] = _numeric(raw["hv104"]).astype("Int64")
    out["height_cm"] = clean_scaled(raw["hb3"], 10.0, 100.0, 250.0)
    out["weight_kg"] = clean_scaled(raw["hb2"], 10.0, 20.0, 300.0)

    reported_bmi = clean_scaled(raw["hb40"], 100.0, 12.0, 60.0)
    calculated_bmi = bmi_from_height_weight(out["height_cm"], out["weight_kg"])
    out["bmi"] = reported_bmi.where(reported_bmi.notna(), calculated_bmi)
    return out


def validate_pr_mr_merge(mr_raw: pd.DataFrame, pr_raw: pd.DataFrame) -> dict:
    """Validate the one-to-one PR-to-MR merge and anthropometry consistency."""
    mr = mr_raw[["mcaseid", "mv001", "mv002", "mv003", "mv012"]].copy()
    pr = pr_raw[
        ["hhid", "hvidx", "hv001", "hv002", "hv104", "hv105", "hb2", "hb3", "hb40"]
    ].copy()

    mr_keys = ["mv001", "mv002", "mv003"]
    pr_keys = ["hv001", "hv002", "hvidx"]

    merged = mr.merge(
        pr,
        left_on=mr_keys,
        right_on=pr_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    both = merged.loc[merged["_merge"] == "both"].copy()

    age_diff = both["mv012"] - both["hv105"]
    height_cm = clean_scaled(both["hb3"], 10.0, 100.0, 250.0)
    weight_kg = clean_scaled(both["hb2"], 10.0, 20.0, 300.0)
    bmi_reported = clean_scaled(both["hb40"], 100.0, 12.0, 60.0)
    bmi_calculated = bmi_from_height_weight(height_cm, weight_kg)
    both_bmi = bmi_reported.notna() & bmi_calculated.notna()

    matched = int(len(both))
    mr_rows = int(len(mr))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merge_keys": {
            "mr": ["mv001", "mv002", "mv003"],
            "pr": ["hv001", "hv002", "hvidx"],
        },
        "mr_rows": mr_rows,
        "pr_rows": int(len(pr)),
        "mr_key_duplicates": int(mr.duplicated(mr_keys).sum()),
        "pr_key_duplicates": int(pr.duplicated(pr_keys).sum()),
        "mr_key_nulls": int(mr[mr_keys].isna().any(axis=1).sum()),
        "pr_key_nulls": int(pr[pr_keys].isna().any(axis=1).sum()),
        "matched_rows": matched,
        "unmatched_mr_rows": int((merged["_merge"] == "left_only").sum()),
        "match_rate": round(matched / mr_rows, 6) if mr_rows else None,
        "sex_mismatch": int((both["hv104"] != 1).sum()),
        "age_exact": int((age_diff == 0).sum()),
        "age_within_1": int((age_diff.abs() <= 1).sum()),
        "age_within_2": int((age_diff.abs() <= 2).sum()),
        "age_max_abs_diff": int(age_diff.abs().max()) if len(age_diff) else None,
        "age_diff_counts": {
            str(int(value)): int(count)
            for value, count in age_diff.value_counts().sort_index().items()
        },
        "height_valid": int(height_cm.notna().sum()),
        "height_valid_rate": round(float(height_cm.notna().mean()), 6) if matched else None,
        "weight_valid": int(weight_kg.notna().sum()),
        "weight_valid_rate": round(float(weight_kg.notna().mean()), 6) if matched else None,
        "bmi_reported_valid": int(bmi_reported.notna().sum()),
        "bmi_computed_valid": int(bmi_calculated.notna().sum()),
        "bmi_both_valid": int(both_bmi.sum()),
        "bmi_correlation": round(float(bmi_calculated[both_bmi].corr(bmi_reported[both_bmi])), 6)
        if both_bmi.any()
        else None,
        "bmi_mean_diff_calc_minus_reported": round(
            float((bmi_calculated[both_bmi] - bmi_reported[both_bmi]).mean()), 6
        )
        if both_bmi.any()
        else None,
        "bmi_median_abs_diff": round(
            float((bmi_calculated[both_bmi] - bmi_reported[both_bmi]).abs().median()), 6
        )
        if both_bmi.any()
        else None,
        "status": "pass"
        if matched == mr_rows
        and int((merged["_merge"] == "left_only").sum()) == 0
        and int((both["hv104"] != 1).sum()) == 0
        and int(mr.duplicated(mr_keys).sum()) == 0
        and int(pr.duplicated(pr_keys).sum()) == 0
        else "review",
        "notes": [
            "All MR rows matched PR on cluster, household, and line.",
            (
                "Age differences between the household roster and the men's questionnaire "
                "are expected; compare age_within_1 and age_within_2 rather than requiring "
                "exact equality."
            ),
            "BMI computed from PR height and weight is compared with the reported hb40 BMI.",
        ],
    }
    return report


def _base_ir(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["case_id"] = raw["caseid"].astype("string").str.strip()
    out["recode"] = "IR"
    out["sex"] = 2
    out["female"] = 1
    out["cluster"] = _numeric(raw["v001"]).astype("Int64")
    out["household"] = _numeric(raw["v002"]).astype("Int64")
    out["line"] = _numeric(raw["v003"]).astype("Int64")
    out["psu"] = _numeric(raw["v021"]).astype("Int64")
    out["strata"] = _numeric(raw["v022"]).astype("Int64")
    out["weight"] = _numeric(raw["v005"])
    out["weight_scaled"] = _numeric(raw["v005"]) / 1_000_000.0
    out["age"] = _numeric(raw["v012"]).astype("Int64")
    out["state"] = _numeric(raw["v024"]).astype("Int64")
    out["residence"] = _numeric(raw["v025"]).astype("Int64")
    out["urban"] = (raw["v025"] == 1).astype("Int64")
    out["wealth"] = _numeric(raw["v190"]).astype("Int64")
    out["education"] = _numeric(raw["v106"]).astype("Int64")
    out["religion"] = _numeric(raw["v130"]).astype("Int64")
    out["social_group"] = raw["s116"].where(raw["s116"].ne(8)).astype("Int64")
    out["marital_status"] = _numeric(raw["v501"]).astype("Int64")
    out["children_ever_born"] = _numeric(raw["v201"]).astype("Int64")
    return out


def _base_mr(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["case_id"] = raw["mcaseid"].astype("string").str.strip()
    out["recode"] = "MR"
    out["sex"] = 1
    out["female"] = 0
    out["cluster"] = _numeric(raw["mv001"]).astype("Int64")
    out["household"] = _numeric(raw["mv002"]).astype("Int64")
    out["line"] = _numeric(raw["mv003"]).astype("Int64")
    out["psu"] = _numeric(raw["mv021"]).astype("Int64")
    out["strata"] = _numeric(raw["mv022"]).astype("Int64")
    out["weight"] = _numeric(raw["mv005"])
    out["weight_scaled"] = _numeric(raw["mv005"]) / 1_000_000.0
    out["age"] = _numeric(raw["mv012"]).astype("Int64")
    out["state"] = _numeric(raw["mv024"]).astype("Int64")
    out["residence"] = _numeric(raw["mv025"]).astype("Int64")
    out["urban"] = (raw["mv025"] == 1).astype("Int64")
    out["wealth"] = _numeric(raw["mv190"]).astype("Int64")
    out["education"] = _numeric(raw["mv106"]).astype("Int64")
    out["religion"] = _numeric(raw["mv130"]).astype("Int64")
    out["social_group"] = raw["sm118"].where(raw["sm118"].ne(8)).astype("Int64")
    out["marital_status"] = _numeric(raw["mv501"]).astype("Int64")
    out["children_ever_born"] = _numeric(raw["mv201"]).astype("Int64")
    return out


def build_ir_features(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the women's feature block."""
    if raw is None:
        raw = _read_stata(IR_PATH, IR_FEATURE_COLUMNS)

    out = _base_ir(raw)
    out["height_cm"] = clean_scaled(raw["v438"], 10.0, 100.0, 250.0)
    out["weight_kg"] = clean_scaled(raw["v437"], 10.0, 20.0, 300.0)
    reported_bmi = clean_scaled(raw["v445"], 100.0, 12.0, 60.0)
    calculated_bmi = bmi_from_height_weight(out["height_cm"], out["weight_kg"])
    out["bmi"] = reported_bmi.where(reported_bmi.notna(), calculated_bmi)
    out["waist_cm"] = clean_cm(raw["s305"])
    out["hip_cm"] = clean_cm(raw["s306"])
    out["waist_hip_ratio"] = waist_hip_ratio(out["waist_cm"], out["hip_cm"])

    out = out.join(
        summarize_bp(
            raw,
            ["sb18s", "sb25s", "sb29s"],
            ["sb18d", "sb25d", "sb29d"],
        )
    )
    out["bp_medication"] = _binary(raw["sb21"])
    out["ever_checked"] = _binary(raw["sb55"])
    out["diagnosed"] = _binary(raw["sb56"])
    out["medication"] = _binary(raw["sb57"])
    out["glucose_raw"] = _numeric(raw["sb74"])
    out["glucose"] = clean_glucose(raw["sb74"])
    out["target"] = build_target(raw, "sb74", "sb56", "sb57")
    out["target_fact_sheet"] = fact_sheet_target(raw, "sb74", "sb57")

    out["tobacco_any"] = any_yes(raw, IR_TOBACCO_COLUMNS)
    out["smoking_any"] = any_yes(raw, IR_SMOKING_COLUMNS)
    out["smokeless_any"] = any_yes(raw, IR_SMOKELESS_COLUMNS)
    out["alcohol_any"] = _binary(raw["s720"])
    out["alcohol_frequency"] = harmonize_alcohol(raw, "s720", "s721")
    for name, column in IR_DIET_COLUMNS.items():
        out[name] = _numeric(raw[column]).astype("Int64")
    return out


def build_mr_features(
    raw: pd.DataFrame | None = None,
    pr_anthropometry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the men's feature block, merging PR height, weight, and BMI."""
    if raw is None:
        raw = _read_stata(MR_PATH, MR_FEATURE_COLUMNS)
    if pr_anthropometry is None:
        pr_raw = _read_stata(PR_PATH, PR_MEN_ANTHRO_COLUMNS)
        pr_anthropometry = prepare_pr_men_anthropometry(pr_raw)

    out = _base_mr(raw)
    out["waist_cm"] = clean_cm(raw["smb305"])
    out["hip_cm"] = clean_cm(raw["smb306"])
    out["waist_hip_ratio"] = waist_hip_ratio(out["waist_cm"], out["hip_cm"])

    out = out.join(
        summarize_bp(
            raw,
            ["smb18s", "smb25s", "smb29s"],
            ["smb18d", "smb25d", "smb29d"],
        )
    )
    out["bp_medication"] = _binary(raw["smb21"])
    out["ever_checked"] = _binary(raw["smb55"])
    out["diagnosed"] = _binary(raw["smb56"])
    out["medication"] = _binary(raw["smb57"])
    out["glucose_raw"] = _numeric(raw["smb74"])
    out["glucose"] = clean_glucose(raw["smb74"])
    out["target"] = build_target(raw, "smb74", "smb56", "smb57")
    out["target_fact_sheet"] = fact_sheet_target(raw, "smb74", "smb57")

    out["tobacco_any"] = any_yes(raw, MR_TOBACCO_COLUMNS)
    out["smoking_any"] = any_yes(raw, MR_SMOKING_COLUMNS)
    out["smokeless_any"] = any_yes(raw, MR_SMOKELESS_COLUMNS)
    out["alcohol_any"] = _binary(raw["sm619"])
    out["alcohol_frequency"] = harmonize_alcohol(raw, "sm619", "sm620")
    for name, column in MR_DIET_COLUMNS.items():
        out[name] = _numeric(raw[column]).astype("Int64")

    anthro = pr_anthropometry[
        ["cluster", "household", "line", "height_cm", "weight_kg", "bmi"]
    ]
    out = out.merge(
        anthro,
        on=["cluster", "household", "line"],
        how="left",
        validate="one_to_one",
    )
    return out


def _summary(df: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    subset = df if mask is None else df.loc[mask]
    rows = int(len(subset))
    positive = int(subset["target"].sum())
    return {
        "rows": rows,
        "positive": positive,
        "prevalence": round(positive / rows, 6) if rows else None,
    }


def _missingness(df: pd.DataFrame, columns: Sequence[str]) -> dict:
    out = {}
    for column in columns:
        missing = int(df[column].isna().sum())
        out[column] = {
            "missing": missing,
            "missing_rate": round(missing / len(df), 6) if len(df) else None,
        }
    return out


def build_feature_audit(df: pd.DataFrame, merge_report: dict) -> dict:
    """Summarise the feature table and merge validation."""
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_definition": "glucose > 140 OR diagnosed == 1 OR medication == 1",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "overall": _summary(df),
        "age_15_49": _summary(df, df["age_15_49"]),
        "age_15_54": _summary(df),
        "by_recode": {
            recode: _summary(df, df["recode"].eq(recode))
            for recode in sorted(df["recode"].unique())
        },
        "by_sex": {
            "female": _summary(df, df["female"].eq(1)),
            "male": _summary(df, df["female"].eq(0)),
        },
        "missingness": _missingness(
            df,
            [
                "height_cm",
                "weight_kg",
                "bmi",
                "waist_cm",
                "hip_cm",
                "waist_hip_ratio",
                "systolic_mean",
                "diastolic_mean",
                "bp_medication",
                "tobacco_any",
                "alcohol_any",
                "social_group",
                "diet_fruit",
            ],
        ),
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "excluded_columns": MODEL_EXCLUDED_COLUMNS,
        "merge_validation": merge_report,
    }
    return audit


def build_irmr_features() -> tuple[pd.DataFrame, dict, dict]:
    """Load, harmonise, merge, and filter the IR/MR analysis table."""
    ir_raw = _read_stata(IR_PATH, IR_FEATURE_COLUMNS)
    ir = build_ir_features(ir_raw)
    del ir_raw

    mr_raw = _read_stata(MR_PATH, MR_FEATURE_COLUMNS)
    pr_raw = _read_stata(PR_PATH, PR_MEN_ANTHRO_COLUMNS)
    merge_report = validate_pr_mr_merge(mr_raw, pr_raw)
    pr_anthropometry = prepare_pr_men_anthropometry(pr_raw)
    mr = build_mr_features(mr_raw, pr_anthropometry)
    del mr_raw, pr_raw, pr_anthropometry

    df = pd.concat([ir, mr], ignore_index=True, sort=False)
    del ir, mr
    df["age_15_49"] = df["age"].between(15, 49)
    df = df.loc[df["glucose"].notna() & df["target"].notna()].copy()
    df = df[FEATURE_COLUMNS].copy()
    audit = build_feature_audit(df, merge_report)
    return df, audit, merge_report


def build_pr_features(raw_pr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the harmonised PR household-member feature table.

    The table mirrors the IR/MR schema, adds the PR cooking-fuel variables, and
    keeps adults aged 15-98 with sex codes 1 or 2, valid random glucose, and a
    known diagnosis/medication status.
    """
    if raw_pr is None:
        raw_pr = _read_stata(PR_PATH, PR_SOURCE_COLUMNS)
    _validate_pr_input(raw_pr)

    out = pd.DataFrame(index=raw_pr.index)
    hhid = raw_pr["hhid"].astype("string").str.strip()
    line = _numeric(raw_pr["hvidx"]).astype("Int64")
    out["case_id"] = hhid + "-" + line.astype("string")
    out["recode"] = "PR"
    out["sex"] = _numeric(raw_pr["hv104"]).astype("Int64")
    out["female"] = (raw_pr["hv104"] == 2).astype("Int64")
    out["cluster"] = _numeric(raw_pr["hv001"]).astype("Int64")
    out["household"] = _numeric(raw_pr["hv002"]).astype("Int64")
    out["line"] = line
    out["psu"] = _numeric(raw_pr["hv021"]).astype("Int64")
    out["strata"] = _numeric(raw_pr["hv022"]).astype("Int64")
    out["weight"] = _numeric(raw_pr["hv005"])
    out["weight_scaled"] = _numeric(raw_pr["hv005"]) / 1_000_000.0
    out["age"] = _numeric(raw_pr["hv105"]).astype("Int64")
    out["age_15_49"] = out["age"].between(15, 49)
    out["state"] = _numeric(raw_pr["hv024"]).astype("Int64")
    out["residence"] = _numeric(raw_pr["hv025"]).astype("Int64")
    out["urban"] = (raw_pr["hv025"] == 1).astype("Int64")
    out["wealth"] = _numeric(raw_pr["hv270"]).astype("Int64")
    out["education"] = clean_education(raw_pr["hv106"])
    out["religion"] = _numeric(raw_pr["sh47"]).astype("Int64")
    out["social_group"] = clean_social_group(raw_pr["sh49"])
    for column in PR_UNAVAILABLE_COLUMNS:
        out[column] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    out = out.join(build_pr_anthropometry(raw_pr))
    out["waist_cm"] = clean_cm(raw_pr["sh305"])
    out["hip_cm"] = clean_cm(raw_pr["sh306"])
    out["waist_hip_ratio"] = waist_hip_ratio(out["waist_cm"], out["hip_cm"])

    out = out.join(
        summarize_bp(
            raw_pr,
            ["shb18s", "shb25s", "shb29s"],
            ["shb18d", "shb25d", "shb29d"],
        )
    )
    out["bp_medication"] = _binary(raw_pr["shb21"])
    out["ever_checked"] = _binary(raw_pr["shb55"])
    out["diagnosed"] = _binary(raw_pr["shb56"])
    out["medication"] = _binary(raw_pr["shb57"])
    out["glucose_raw"] = _numeric(raw_pr["shb74"])
    out["glucose"] = clean_glucose(raw_pr["shb74"])
    out["target"] = build_target(raw_pr, "shb74", "shb56", "shb57")
    out["target_fact_sheet"] = fact_sheet_target(raw_pr, "shb74", "shb57")

    out["tobacco_any"] = _binary(raw_pr["sh25"])
    out["alcohol_any"] = _binary(raw_pr["sh26"])
    out["cooking_fuel"] = clean_cooking_fuel_codes(raw_pr["hv226"])
    out["cooking_fuel_clean"] = cooking_fuel_clean(raw_pr["hv226"])

    keep = (
        out["age"].between(AGE_MIN, AGE_MAX)
        & out["sex"].isin([1, 2])
        & out["glucose"].notna()
        & out["target"].notna()
    )
    return out.loc[keep, PR_HARMONISED_COLUMNS].reset_index(drop=True)


def _weighted_mean(df: pd.DataFrame, column: str) -> float | None:
    """Return the survey-weighted mean of a 0/1 column, or None if unavailable."""
    weights = _numeric(df["weight_scaled"])
    values = _numeric(df[column])
    valid = weights.notna() & (weights > 0) & values.notna()
    if not valid.any():
        return None
    return round(float((values[valid] * weights[valid]).sum() / weights[valid].sum()), 6)


def _observed_pr_counts(audit: dict) -> dict:
    """Flatten the audit counts that reconcile against the reference counts."""
    return {
        "rows": audit["rows"],
        "positive": audit["overall"]["positive"],
        "female": audit["by_sex"]["female"]["rows"],
        "male": audit["by_sex"]["male"]["rows"],
        "age_15_49_rows": audit["age_15_49"]["rows"],
        "age_50_plus_rows": audit["age_50_plus"]["rows"],
        "bp_valid_rows": audit["blood_pressure"]["valid_rows"],
        "bp_valid_age_15_49": audit["blood_pressure"]["age_15_49"],
        "bp_valid_age_50_plus": audit["blood_pressure"]["age_50_plus"],
        "bp_positive_rows": audit["blood_pressure"]["positive_rows"],
        "bp_positive_age_15_49": audit["blood_pressure"]["positive_age_15_49"],
        "bp_positive_age_50_plus": audit["blood_pressure"]["positive_age_50_plus"],
    }


def reconcile_pr_audit(
    audit: dict,
    reference: dict | None = None,
) -> dict:
    """Compare observed PR audit counts with the documented NFHS-5 reference."""
    reference = PR_REFERENCE_COUNTS if reference is None else reference
    observed = _observed_pr_counts(audit)
    checks = {
        name: {
            "expected": expected,
            "observed": observed.get(name),
            "pass": observed.get(name) == expected,
        }
        for name, expected in reference.items()
    }
    return {
        "status": "pass" if all(check["pass"] for check in checks.values()) else "review",
        "checks": checks,
    }


def build_pr_feature_audit(df: pd.DataFrame) -> dict:
    """Summarise the PR feature table for the aggregate audit artifact."""
    rows = int(len(df))
    bp_valid = df["systolic_mean"].notna() & df["diastolic_mean"].notna()
    age_50_plus = df["age"].ge(50)
    high_glucose = df["glucose"] > GLUCOSE_THRESHOLD
    fact_sheet = df["target_fact_sheet"].eq(1)
    fact_sheet_known = int(df["target_fact_sheet"].notna().sum())

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recode": "PR",
        "target_definition": "glucose > 140 OR diagnosed == 1 OR medication == 1",
        "fact_sheet_target_definition": "glucose > 140 OR medication == 1",
        "rows": rows,
        "columns": int(len(df.columns)),
        "overall": _summary(df),
        "age_15_49": _summary(df, df["age_15_49"]),
        "age_50_plus": _summary(df, age_50_plus),
        "by_sex": {
            "female": _summary(df, df["female"].eq(1)),
            "male": _summary(df, df["female"].eq(0)),
        },
        "target_components": {
            "glucose_gt_140": int(high_glucose.sum()),
            "diagnosed": int(df["diagnosed"].eq(1).sum()),
            "medication": int(df["medication"].eq(1).sum()),
            "target_union": int(df["target"].sum()),
            "target_fact_sheet": int(fact_sheet.sum()),
            "target_fact_sheet_known": fact_sheet_known,
            "prevalence_union": round(float(df["target"].mean()), 6) if rows else None,
            "prevalence_fact_sheet": round(int(fact_sheet.sum()) / fact_sheet_known, 6)
            if fact_sheet_known
            else None,
        },
        "weighted_prevalence": {
            "target": _weighted_mean(df, "target"),
            "target_fact_sheet": _weighted_mean(df, "target_fact_sheet"),
            "by_sex": {
                "female": _weighted_mean(df.loc[df["female"].eq(1)], "target"),
                "male": _weighted_mean(df.loc[df["female"].eq(0)], "target"),
            },
        },
        "blood_pressure": {
            "valid_rows": int(bp_valid.sum()),
            "valid_rate": round(float(bp_valid.mean()), 6) if rows else None,
            "missing_rows": int((~bp_valid).sum()),
            "age_15_49": int((bp_valid & df["age_15_49"]).sum()),
            "age_50_plus": int((bp_valid & age_50_plus).sum()),
            "positive_rows": int((bp_valid & df["target"].eq(1)).sum()),
            "positive_age_15_49": int(
                (bp_valid & df["age_15_49"] & df["target"].eq(1)).sum()
            ),
            "positive_age_50_plus": int(
                (bp_valid & age_50_plus & df["target"].eq(1)).sum()
            ),
            "readings": {
                str(int(readings)): int(count)
                for readings, count in df["bp_readings"].value_counts().sort_index().items()
            },
            "high_rate": round(float(df["bp_high"].mean()), 6)
            if df["bp_high"].notna().any()
            else None,
            "medication_known": int(df["bp_medication"].notna().sum()),
            "medication_positive": int(df["bp_medication"].eq(1).sum()),
        },
        "survey_design": {
            "clusters": int(df["cluster"].nunique()),
            "households": int(df[["cluster", "household"]].drop_duplicates().shape[0]),
            "psus": int(df["psu"].nunique()),
            "strata": int(df["strata"].nunique()),
            "states": int(df["state"].nunique()),
        },
        "cooking_fuel": {
            "clean": int(df["cooking_fuel_clean"].eq(1).sum()),
            "not_clean": int(df["cooking_fuel_clean"].eq(0).sum()),
            "missing": int(df["cooking_fuel_clean"].isna().sum()),
            "clean_rate": round(float(df["cooking_fuel_clean"].mean()), 6)
            if df["cooking_fuel_clean"].notna().any()
            else None,
        },
        "missingness": _missingness(
            df,
            [
                "state",
                "education",
                "religion",
                "social_group",
                "height_cm",
                "weight_kg",
                "bmi",
                "waist_cm",
                "hip_cm",
                "waist_hip_ratio",
                "systolic_mean",
                "diastolic_mean",
                "bp_medication",
                "tobacco_any",
                "alcohol_any",
                "cooking_fuel_clean",
                "target_fact_sheet",
            ],
        ),
        "harmonised_columns": PR_HARMONISED_COLUMNS,
        "unavailable_columns": PR_UNAVAILABLE_COLUMNS,
    }
    audit["reconciliation"] = reconcile_pr_audit(audit)
    return audit

