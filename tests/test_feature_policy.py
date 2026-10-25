"""Feature-policy tests: the default feature sets are a fixed contract."""

import pandas as pd
import pytest

from screener.feature_policy import (
    FORBIDDEN_FEATURE_COLUMNS,
    PR_15_PLUS_FEATURES,
    PR_EXCLUDED_FEATURES,
    PR_VARIANT_NAMES,
    PRIMARY_DEFAULT_FEATURES,
    PRIMARY_FAIRNESS_FEATURES,
    PRIMARY_STATE_FEATURES,
    pr_feature_set,
    validate_feature_frame,
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
