"""Synthetic-data tests for the PR household-member feature builder."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from screener.features import (
    FEATURE_COLUMNS,
    PR_HARMONISED_COLUMNS,
    build_pr_anthropometry,
    build_pr_feature_audit,
    build_pr_features,
    clean_cooking_fuel_codes,
    clean_education,
    clean_social_group,
    cooking_fuel_clean,
    reconcile_pr_audit,
)

BASE_ROW = {
    "hhid": "h1",
    "hvidx": 1,
    "hv001": 1,
    "hv002": 1,
    "hv005": 1_000_000,
    "hv021": 1,
    "hv022": 1,
    "hv024": 1,
    "hv025": 1,
    "hv104": 1,
    "hv105": 30,
    "hv106": 2,
    "hv270": 3,
    "hv226": 1,
    "sh47": 1,
    "sh49": 3,
    "sh25": 0,
    "sh26": 0,
    "shb18s": 120,
    "shb18d": 80,
    "shb25s": 996,
    "shb25d": 996,
    "shb29s": 996,
    "shb29d": 996,
    "shb21": 0,
    "sh305": 80,
    "sh306": 100,
    "ha2": 550,
    "ha3": 1600,
    "ha40": 2150,
    "hb2": 650,
    "hb3": 1700,
    "hb40": 2249,
    "shb55": 1,
    "shb56": 0,
    "shb57": 0,
    "shb74": 100,
}


def _row(case: str, **overrides) -> dict:
    hhid, line = case.split("-")
    row = {
        **BASE_ROW,
        "hhid": hhid,
        "hvidx": int(line),
        "hv001": int(hhid.removeprefix("h")),
        "hv002": 1,
        "hv021": int(hhid.removeprefix("h")),
        "hv022": 1,
    }
    row.update(overrides)
    return row


def pr_frame() -> pd.DataFrame:
    """A PR-shaped frame with retained adults and every exclusion path."""
    rows = [
        # Retained: woman with three valid BP readings and women's anthropometry.
        _row(
            "h1-1",
            hv104=2,
            hv105=30,
            shb25s=130,
            shb25d=85,
            shb29s=140,
            shb29d=90,
            hv226=2,
        ),
        # Retained: man with one valid BP pair and men's anthropometry.
        _row(
            "h1-2",
            hv104=1,
            hv105=40,
            shb18s=130,
            shb18d=85,
            shb21=1,
            ha2=9994,
            ha3=9994,
            ha40=9998,
            hb2=700,
            hb3=1750,
            hb40=9998,
            hv226=5,
            sh49=1,
            sh25=1,
            shb74=150,
        ),
        # Retained: older woman, no valid BP pair, medication-only target.
        _row(
            "h2-1",
            hv104=2,
            hv105=60,
            shb18s=996,
            shb18d=996,
            ha2=9994,
            ha3=9994,
            ha40=9998,
            shb57=1,
            hv226=95,
        ),
        # Excluded: sex code 3.
        _row("h2-2", hv104=3),
        # Excluded: age below 15.
        _row("h3-1", hv105=10),
        # Excluded: invalid glucose missing code.
        _row("h3-2", shb74=995),
        # Excluded: diagnosis and medication status both unknown.
        _row("h4-1", hv104=2, hv105=45, shb56=None, shb57=None),
        # Retained: man with missing codes for education, social group/tobacco/alcohol.
        _row(
            "h4-2",
            hv104=1,
            hv105=25,
            hv106=8,
            sh49=8,
            sh25=8,
            sh26=8,
            shb21=None,
            shb18s=130,
            shb18d=80,
            shb25s=150,
            shb25d=95,
            ha2=9994,
            ha3=9994,
            ha40=9998,
            hb2=9994,
            hb3=9994,
            hb40=9998,
            hv226=4,
        ),
        # Retained: high-glucose woman with no cooking-fuel classification.
        _row(
            "h5-1",
            hv104=2,
            hv105=20,
            shb74=141,
            shb18s=110,
            shb18d=70,
            ha2=6000,
            ha3=9994,
            ha40=9998,
            hv226=96,
            sh49=4,
        ),
        # Retained: diagnosed man whose fact-sheet target is negative.
        _row(
            "h5-2",
            hv104=1,
            hv105=55,
            shb56=1,
            shb21=1,
            shb25s=0,
            shb25d=0,
            shb29s=0,
            shb29d=0,
            hv226=1,
        ),
    ]
    return pd.DataFrame(rows)


RETAINED = ["h1-1", "h1-2", "h2-1", "h4-2", "h5-1", "h5-2"]


@pytest.fixture
def features() -> pd.DataFrame:
    return build_pr_features(pr_frame()).set_index("case_id")


def test_pr_builder_filters_to_adults_with_valid_glucose_and_known_target(features):
    assert features.index.tolist() == RETAINED
    assert features["age"].between(15, 98).all()
    assert features["sex"].isin([1, 2]).all()
    assert features["glucose"].notna().all()
    assert features["target"].notna().all()


def test_pr_builder_constructs_targets(features):
    assert features.loc["h1-1", "target"] == 0
    assert features.loc["h1-2", "target"] == 1
    assert features.loc["h2-1", "target"] == 1
    assert features.loc["h5-1", "target"] == 1
    assert features.loc["h5-2", "target"] == 1

    assert features.loc["h1-2", "target_fact_sheet"] == 1
    assert features.loc["h5-1", "target_fact_sheet"] == 1
    assert features.loc["h5-2", "target_fact_sheet"] == 0


def test_pr_builder_summarises_blood_pressure(features):
    assert features.loc["h1-1", "bp_readings"] == 3
    assert features.loc["h1-1", "systolic_mean"] == 130
    assert features.loc["h1-1", "diastolic_mean"] == 85
    assert features.loc["h1-1", "bp_high"] == 0

    assert features.loc["h1-2", "bp_readings"] == 1
    assert features.loc["h1-2", "systolic_mean"] == 130
    assert features.loc["h1-2", "diastolic_mean"] == 85
    assert features.loc["h1-2", "bp_medication"] == 1

    assert features.loc["h4-2", "bp_readings"] == 2
    assert features.loc["h4-2", "systolic_mean"] == 140
    assert features.loc["h4-2", "diastolic_mean"] == 87.5
    assert features.loc["h4-2", "bp_high"] == 1
    assert pd.isna(features.loc["h4-2", "bp_medication"])

    assert features.loc["h2-1", "bp_readings"] == 0
    assert pd.isna(features.loc["h2-1", "systolic_mean"])
    assert pd.isna(features.loc["h2-1", "diastolic_mean"])
    assert pd.isna(features.loc["h2-1", "bp_high"])


def test_pr_builder_uses_sex_specific_anthropometry(features):
    assert features.loc["h1-1", "height_cm"] == 160
    assert features.loc["h1-1", "weight_kg"] == 55
    assert features.loc["h1-1", "bmi"] == 21.5

    assert features.loc["h1-2", "height_cm"] == 175
    assert features.loc["h1-2", "weight_kg"] == 70
    assert round(float(features.loc["h1-2", "bmi"]), 2) == 22.86

    for case in ["h2-1", "h4-2", "h5-1", "h5-2"]:
        assert pd.isna(features.loc[case, "height_cm"])
        assert pd.isna(features.loc[case, "weight_kg"])
        assert pd.isna(features.loc[case, "bmi"])


def test_pr_builder_handles_missing_and_retained_codes(features):
    assert pd.isna(features.loc["h4-2", "education"])
    assert pd.isna(features.loc["h4-2", "social_group"])
    assert pd.isna(features.loc["h4-2", "tobacco_any"])
    assert pd.isna(features.loc["h4-2", "alcohol_any"])

    assert features.loc["h1-1", "education"] == 2
    assert features.loc["h1-1", "social_group"] == 3
    assert features.loc["h1-2", "social_group"] == 1
    assert features.loc["h5-1", "social_group"] == 4


def test_pr_builder_classifies_cooking_fuel(features):
    expected = {
        "h1-1": (2, 1),
        "h1-2": (5, 0),
        "h2-1": (95, pd.NA),
        "h4-2": (4, 1),
        "h5-1": (96, pd.NA),
        "h5-2": (1, 1),
    }
    for case, (fuel, clean) in expected.items():
        assert features.loc[case, "cooking_fuel"] == fuel
        if pd.isna(clean):
            assert pd.isna(features.loc[case, "cooking_fuel_clean"])
        else:
            assert features.loc[case, "cooking_fuel_clean"] == clean


def test_pr_builder_emits_the_harmonised_schema(features):
    assert list(features.reset_index().columns) == PR_HARMONISED_COLUMNS
    assert set(FEATURE_COLUMNS).issubset(features.reset_index().columns)
    assert (features["recode"] == "PR").all()
    assert features.index.is_unique


def test_pr_builder_tracks_the_15_49_sample_flag(features):
    assert features.loc["h1-1", "age_15_49"]
    assert features.loc["h1-2", "age_15_49"]
    assert not features.loc["h2-1", "age_15_49"]
    assert features.loc["h5-1", "age_15_49"]
    assert not features.loc["h5-2", "age_15_49"]


def test_pr_builder_leaves_unavailable_concepts_missing(features):
    unavailable = [
        "marital_status",
        "children_ever_born",
        "smoking_any",
        "smokeless_any",
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
    for column in unavailable:
        assert column in features.columns
        assert features[column].isna().all()


def test_pr_builder_is_deterministic():
    first = build_pr_features(pr_frame())
    second = build_pr_features(pr_frame())

    assert_frame_equal(first, second)


def test_pr_builder_raises_on_missing_required_columns():
    with pytest.raises(ValueError, match="shb74"):
        build_pr_features(pr_frame().drop(columns=["shb74"]))


def test_pr_builder_raises_on_unexpected_required_codes():
    bad = pr_frame()
    bad.loc[0, "hv104"] = 9

    with pytest.raises(ValueError, match="hv104"):
        build_pr_features(bad)


def test_education_and_social_group_missing_code_becomes_na():
    education = clean_education(pd.Series([0, 1, 2, 3, 8]))
    social_group = clean_social_group(pd.Series([1, 2, 3, 4, 8]))

    assert education.tolist()[:4] == [0, 1, 2, 3]
    assert pd.isna(education.iloc[4])
    assert social_group.tolist()[:4] == [1, 2, 3, 4]
    assert pd.isna(social_group.iloc[4])


def test_cooking_fuel_mapping_covers_clean_dirty_and_missing_codes():
    codes = pd.Series([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 95, 96])
    assert clean_cooking_fuel_codes(codes).tolist() == codes.tolist()

    clean = cooking_fuel_clean(codes)
    assert clean.tolist()[:3] == [1, 1, 1]
    assert clean.tolist()[3:10] == [0, 0, 0, 0, 0, 0, 0]
    assert clean.tolist()[10:].count(pd.NA) == 2


def test_build_pr_anthropometry_selects_the_sex_specific_variables():
    raw = pd.DataFrame(
        {
            "hv104": [2, 1, 2, 1],
            "hv105": [30, 40, 50, 55],
            "ha3": [1600, 9994, 1650, 9994],
            "ha2": [550, 9994, 600, 9994],
            "ha40": [2150, 9998, 2300, 9998],
            "hb3": [1700, 1750, 9994, 1800],
            "hb2": [650, 700, 9994, 750],
            "hb40": [2249, 2286, 9998, 2400],
        }
    )
    anthropometry = build_pr_anthropometry(raw)

    assert anthropometry.loc[0, "height_cm"] == 160
    assert anthropometry.loc[0, "weight_kg"] == 55
    assert anthropometry.loc[0, "bmi"] == 21.5
    assert anthropometry.loc[1, "height_cm"] == 175
    assert anthropometry.loc[1, "weight_kg"] == 70
    assert anthropometry.loc[1, "bmi"] == 22.86
    # Women are measured at 15-49, so a 50-year-old woman has no anthropometry.
    assert pd.isna(anthropometry.loc[2, "height_cm"])
    # Men are measured at 15-54, so a 55-year-old man has no anthropometry.
    assert pd.isna(anthropometry.loc[3, "height_cm"])
    assert pd.isna(anthropometry.loc[3, "bmi"])


def test_pr_feature_audit_records_sample_bp_and_missingness():
    audit = build_pr_feature_audit(build_pr_features(pr_frame()))

    assert audit["recode"] == "PR"
    assert audit["rows"] == 6
    assert audit["columns"] == len(PR_HARMONISED_COLUMNS)
    assert audit["overall"]["positive"] == 4
    assert audit["by_sex"]["female"]["rows"] == 3
    assert audit["by_sex"]["male"]["rows"] == 3
    assert audit["age_15_49"]["rows"] == 4
    assert audit["age_50_plus"]["rows"] == 2

    assert audit["blood_pressure"]["valid_rows"] == 5
    assert audit["blood_pressure"]["missing_rows"] == 1
    assert audit["blood_pressure"]["age_15_49"] == 4
    assert audit["blood_pressure"]["age_50_plus"] == 1
    assert audit["blood_pressure"]["positive_rows"] == 3
    assert audit["blood_pressure"]["positive_age_15_49"] == 2
    assert audit["blood_pressure"]["positive_age_50_plus"] == 1

    assert audit["survey_design"]["clusters"] == 4
    assert audit["survey_design"]["households"] == 4
    assert audit["weighted_prevalence"]["target"] == 0.666667

    assert audit["missingness"]["education"]["missing"] == 1
    assert audit["missingness"]["cooking_fuel_clean"]["missing"] == 2
    assert audit["cooking_fuel"]["clean"] == 3
    assert audit["cooking_fuel"]["not_clean"] == 1

    assert audit["reconciliation"]["status"] == "review"
    assert audit["reconciliation"]["checks"]["rows"]["observed"] == 6


def test_pr_audit_reconciles_against_matching_reference_counts():
    audit = build_pr_feature_audit(build_pr_features(pr_frame()))
    observed = {
        "rows": 6,
        "positive": 4,
        "female": 3,
        "male": 3,
        "age_15_49_rows": 4,
        "age_50_plus_rows": 2,
        "bp_valid_rows": 5,
        "bp_valid_age_15_49": 4,
        "bp_valid_age_50_plus": 1,
        "bp_positive_rows": 3,
        "bp_positive_age_15_49": 2,
        "bp_positive_age_50_plus": 1,
    }

    reconciliation = reconcile_pr_audit(audit, observed)

    assert reconciliation["status"] == "pass"
    assert all(check["pass"] for check in reconciliation["checks"].values())
