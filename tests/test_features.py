import pandas as pd

from screener.features import (
    MODEL_FEATURE_COLUMNS,
    any_yes,
    bmi_from_height_weight,
    clean_cm,
    clean_scaled,
    fact_sheet_target,
    harmonize_alcohol,
    prepare_pr_men_anthropometry,
    summarize_bp,
    validate_pr_mr_merge,
    waist_hip_ratio,
)


def test_clean_scaled_drops_missing_codes_and_out_of_range():
    values = pd.Series([9994, 9995, 500, 3000, 199])
    cleaned = clean_scaled(values, 10.0, 20.0, 300.0)

    assert pd.isna(cleaned.iloc[0])
    assert pd.isna(cleaned.iloc[1])
    assert cleaned.iloc[2] == 50
    assert cleaned.iloc[3] == 300
    assert pd.isna(cleaned.iloc[4])


def test_clean_cm_and_waist_hip_ratio():
    waist = clean_cm(pd.Series([80, 30, 250, 251, 999.5]))
    hip = clean_cm(pd.Series([100, 100, 100, 100, 100]))
    ratio = waist_hip_ratio(waist, hip)

    assert waist.iloc[0] == 80
    assert waist.iloc[1] == 30
    assert waist.iloc[2] == 250
    assert pd.isna(waist.iloc[3])
    assert pd.isna(waist.iloc[4])
    assert ratio.iloc[0] == 0.8
    assert pd.isna(ratio.iloc[2])  # 2.5 is outside the plausible range


def test_bmi_from_height_weight_uses_plausible_range():
    height = pd.Series([170.0, 100.0, 170.0])
    weight = pd.Series([65.0, 300.0, 10.0])
    bmi = bmi_from_height_weight(height, weight)

    assert round(float(bmi.iloc[0]), 2) == 22.49
    assert pd.isna(bmi.iloc[1])
    assert pd.isna(bmi.iloc[2])


def test_summarize_bp_averages_valid_pairs_and_flags_high():
    df = pd.DataFrame(
        {
            "sb18s": [120, 0, 996],
            "sb18d": [80, 0, 996],
            "sb25s": [130, 150, 996],
            "sb25d": [85, 95, 996],
            "sb29s": [140, 996, 996],
            "sb29d": [90, 996, 996],
        }
    )
    bp = summarize_bp(df, ["sb18s", "sb25s", "sb29s"], ["sb18d", "sb25d", "sb29d"])

    assert bp["bp_readings"].tolist() == [3, 1, 0]
    assert round(float(bp["systolic_mean"].iloc[0]), 1) == 130.0
    assert round(float(bp["diastolic_mean"].iloc[0]), 1) == 85.0
    assert bp["bp_high"].iloc[0] == 0
    assert bp["bp_high"].iloc[1] == 1
    assert pd.isna(bp["systolic_mean"].iloc[2])
    assert pd.isna(bp["bp_high"].iloc[2])


def test_any_yes_returns_na_when_all_inputs_missing():
    df = pd.DataFrame(
        {
            "a": [0, 1, 0, None],
            "b": [0, 0, 1, None],
        }
    )
    result = any_yes(df, ["a", "b"])

    assert result.tolist()[:3] == [0, 1, 1]
    assert pd.isna(result.iloc[3])


def test_harmonize_alcohol_maps_non_drinkers_to_zero():
    df = pd.DataFrame(
        {
            "any": [0, 0, 1, 1, 1],
            "frequency": [0, None, 1, 2, None],
        }
    )
    result = harmonize_alcohol(df, "any", "frequency")

    assert result.iloc[0] == 0
    assert result.iloc[1] == 0
    assert result.iloc[2] == 1
    assert result.iloc[3] == 2
    assert pd.isna(result.iloc[4])


def test_fact_sheet_target_uses_high_glucose_or_medication():
    df = pd.DataFrame(
        {
            "glucose": [100, 141, 100, 995],
            "medication": [0, 0, 1, 0],
        }
    )
    target = fact_sheet_target(df, "glucose", "medication")

    assert target.tolist()[:3] == [0, 1, 1]
    assert pd.isna(target.iloc[3])


def test_prepare_pr_men_anthropometry_cleans_and_falls_back_to_computed_bmi():
    raw = pd.DataFrame(
        {
            "hv001": [1, 1, 1],
            "hv002": [1, 1, 1],
            "hvidx": [1, 2, 3],
            "hv105": [30, 40, 50],
            "hv104": [1, 1, 1],
            "hb3": [1700, 1750, 9994],
            "hb2": [650, 700, 9994],
            "hb40": [2249, 9998, 9998],
        }
    )
    prepared = prepare_pr_men_anthropometry(raw)

    assert prepared.loc[0, "height_cm"] == 170.0
    assert prepared.loc[0, "weight_kg"] == 65.0
    assert prepared.loc[0, "bmi"] == 22.49
    assert round(float(prepared.loc[1, "bmi"]), 2) == 22.86
    assert pd.isna(prepared.loc[2, "height_cm"])
    assert pd.isna(prepared.loc[2, "weight_kg"])
    assert pd.isna(prepared.loc[2, "bmi"])


def test_validate_pr_mr_merge_reports_full_match():
    mr = pd.DataFrame(
        {
            "mcaseid": ["a", "b"],
            "mv001": [1, 1],
            "mv002": [1, 1],
            "mv003": [1, 2],
            "mv012": [30, 40],
        }
    )
    pr = pd.DataFrame(
        {
            "hhid": ["h1", "h1"],
            "hvidx": [1, 2],
            "hv001": [1, 1],
            "hv002": [1, 1],
            "hv104": [1, 1],
            "hv105": [30, 40],
            "hb2": [650, 700],
            "hb3": [1700, 1750],
            "hb40": [2249, 2286],
        }
    )
    report = validate_pr_mr_merge(mr, pr)

    assert report["matched_rows"] == 2
    assert report["unmatched_mr_rows"] == 0
    assert report["match_rate"] == 1.0
    assert report["sex_mismatch"] == 0
    assert report["age_exact"] == 2
    assert report["height_valid"] == 2
    assert report["weight_valid"] == 2
    assert report["status"] == "pass"


def test_model_feature_columns_exclude_target_and_ids():
    excluded = {
        "case_id",
        "cluster",
        "household",
        "line",
        "weight",
        "glucose",
        "glucose_raw",
        "ever_checked",
        "diagnosed",
        "medication",
        "target",
        "target_fact_sheet",
        "age_15_49",
        "sex",
        "residence",
    }

    assert excluded.isdisjoint(MODEL_FEATURE_COLUMNS)
    assert "age" in MODEL_FEATURE_COLUMNS
    assert "female" in MODEL_FEATURE_COLUMNS
    assert "urban" in MODEL_FEATURE_COLUMNS
    assert "bmi" in MODEL_FEATURE_COLUMNS
    assert "systolic_mean" in MODEL_FEATURE_COLUMNS
