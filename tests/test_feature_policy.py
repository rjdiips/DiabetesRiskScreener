"""Feature-policy tests: the default feature sets are a fixed contract."""

import pandas as pd
import pytest

from screener.feature_policy import (
    FORBIDDEN_FEATURE_COLUMNS,
    NON_COMPETING_FEATURE_SETS,
    PR_15_PLUS_FEATURES,
    PR_EXCLUDED_FEATURES,
    PR_VARIANT_NAMES,
    PRIMARY_DEFAULT_FEATURES,
    PRIMARY_FAIRNESS_FEATURES,
    PRIMARY_FEATURE_SETS,
    PRIMARY_STATE_FEATURES,
    pr_feature_set,
    primary_feature_set,
    validate_feature_frame,
)

# The pre-registered blocks and counts from the model-shortlist spec.
ANTHROPOMETRY_BLOCK = (
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "hip_cm",
    "waist_hip_ratio",
)
BP_BLOCK = (
    "systolic_mean",
    "diastolic_mean",
    "bp_readings",
    "bp_high",
    "bp_medication",
)
DIET_BLOCK = (
    "diet_milk",
    "diet_pulses",
    "diet_green_leafy",
    "diet_fruit",
    "diet_eggs",
    "diet_fish",
    "diet_meat",
    "diet_fried",
    "diet_aerated",
)
BUNDLES = (
    ("measurement_no_anthropometry", ANTHROPOMETRY_BLOCK, 26),
    ("measurement_no_bp", BP_BLOCK, 27),
    ("measurement_no_diet", DIET_BLOCK, 23),
)


def test_pr_default_feature_set_is_exactly_the_twelve_documented_features():
    assert PR_15_PLUS_FEATURES == (
        "female",
        "age",
        "urban",
        "wealth",
        "education",
        "tobacco_any",
        "alcohol_any",
        "systolic_mean",
        "diastolic_mean",
        "bp_readings",
        "bp_high",
        "bp_medication",
    )


def test_pr_default_excludes_sensitive_anthropometry_and_unavailable_concepts():
    assert set(PR_15_PLUS_FEATURES).isdisjoint(PR_EXCLUDED_FEATURES)
    for column in (
        "state",
        "religion",
        "social_group",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_cm",
        "hip_cm",
        "waist_hip_ratio",
        "diet_fruit",
        "cooking_fuel",
        "cooking_fuel_clean",
        "marital_status",
        "children_ever_born",
        "smoking_any",
        "smokeless_any",
        "alcohol_frequency",
    ):
        assert column in PR_EXCLUDED_FEATURES


def test_pr_feature_set_variants_add_only_the_expected_columns():
    assert PR_VARIANT_NAMES == ("state", "fairness", "combined", "cooking_fuel")
    assert pr_feature_set("default") == PR_15_PLUS_FEATURES
    assert pr_feature_set("state") == PR_15_PLUS_FEATURES + ("state",)
    assert pr_feature_set("fairness") == PR_15_PLUS_FEATURES + (
        "religion",
        "social_group",
    )
    assert pr_feature_set("combined") == PR_15_PLUS_FEATURES + (
        "state",
        "religion",
        "social_group",
    )
    assert pr_feature_set("cooking_fuel") == PR_15_PLUS_FEATURES + (
        "cooking_fuel_clean",
    )
    with pytest.raises(ValueError, match="unknown PR feature-set variant"):
        pr_feature_set("everything")


def test_primary_default_has_32_features_and_variants_add_sensitive_columns():
    assert len(PRIMARY_DEFAULT_FEATURES) == 32
    assert "state" not in PRIMARY_DEFAULT_FEATURES
    assert "religion" not in PRIMARY_DEFAULT_FEATURES
    assert "social_group" not in PRIMARY_DEFAULT_FEATURES
    assert PRIMARY_STATE_FEATURES == PRIMARY_DEFAULT_FEATURES + ("state",)
    assert PRIMARY_FAIRNESS_FEATURES == PRIMARY_DEFAULT_FEATURES + (
        "religion",
        "social_group",
    )


def test_forbidden_columns_are_rejected_before_training():
    frame = pd.DataFrame({"female": [0, 1], "target": [0, 1]})

    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_frame(frame, ["female", "target"])
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_frame(frame, ["female", "case_id"])
    assert "target" in FORBIDDEN_FEATURE_COLUMNS
    assert "case_id" in FORBIDDEN_FEATURE_COLUMNS
    assert "weight_scaled" in FORBIDDEN_FEATURE_COLUMNS


def test_missing_feature_columns_are_rejected():
    frame = pd.DataFrame({"female": [0, 1]})

    with pytest.raises(ValueError, match="missing feature columns"):
        validate_feature_frame(frame, ["female", "age"])


def test_a_clean_feature_frame_passes():
    frame = pd.DataFrame({"female": [0, 1], "age": [25, 40]})

    validate_feature_frame(frame, ["female", "age"])


def test_measurement_full_is_the_32_feature_primary_default():
    assert primary_feature_set() == PRIMARY_DEFAULT_FEATURES
    assert primary_feature_set("measurement_full") == PRIMARY_DEFAULT_FEATURES
    assert len(primary_feature_set("measurement_full")) == 32


def test_questionnaire_only_drops_measurement_columns_and_keeps_the_rest():
    columns = primary_feature_set("questionnaire_only")

    assert len(columns) == 24
    for dropped in ("waist_cm", "hip_cm", "waist_hip_ratio", *BP_BLOCK):
        assert dropped not in columns
    for kept in (
        "age",
        "female",
        "height_cm",
        "weight_kg",
        "bmi",
        "education",
        "wealth",
        "urban",
        "marital_status",
        "children_ever_born",
        "tobacco_any",
        "alcohol_any",
        *DIET_BLOCK,
    ):
        assert kept in columns


def test_each_bundle_is_the_full_set_minus_exactly_one_block():
    full = set(primary_feature_set("measurement_full"))

    for name, block, expected_size in BUNDLES:
        columns = primary_feature_set(name)
        assert len(columns) == expected_size
        assert set(columns) == full - set(block)


def test_unknown_primary_feature_set_is_rejected():
    with pytest.raises(ValueError, match="unknown primary feature-set"):
        primary_feature_set("everything")


def test_every_primary_set_excludes_forbidden_and_target_derived_columns():
    prohibited = {
        "glucose_raw",
        "glucose",
        "ever_checked",
        "diagnosed",
        "medication",
        "target",
        "target_fact_sheet",
        "case_id",
        "recode",
        "cluster",
        "household",
        "line",
        "psu",
        "strata",
        "weight",
        "weight_scaled",
        "age_15_49",
        "sex",
        "residence",
    }

    for name in PRIMARY_FEATURE_SETS:
        columns = set(primary_feature_set(name))
        assert columns.isdisjoint(prohibited), name
        assert columns.isdisjoint(FORBIDDEN_FEATURE_COLUMNS), name


def test_every_primary_set_passes_feature_frame_validation():
    columns = sorted(
        {
            column
            for name in PRIMARY_FEATURE_SETS
            for column in primary_feature_set(name)
        }
    )
    frame = pd.DataFrame({column: [0.0, 1.0] for column in columns})

    for name in PRIMARY_FEATURE_SETS:
        validate_feature_frame(frame, primary_feature_set(name))


def test_questionnaire_only_is_the_only_non_competing_set():
    assert set(PRIMARY_FEATURE_SETS) == {
        "measurement_full",
        "measurement_no_anthropometry",
        "measurement_no_bp",
        "measurement_no_diet",
        "questionnaire_only",
    }
    assert set(PRIMARY_FEATURE_SETS) - NON_COMPETING_FEATURE_SETS == {
        "measurement_full",
        "measurement_no_anthropometry",
        "measurement_no_bp",
        "measurement_no_diet",
    }
    assert NON_COMPETING_FEATURE_SETS == frozenset({"questionnaire_only"})
