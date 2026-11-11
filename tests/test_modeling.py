"""Unit tests for the model-family factories on small synthetic matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.modeling import (
    DEFAULT_CATBOOST_PARAMS,
    DEFAULT_LIGHTGBM_PARAMS,
    DEFAULT_LOGISTIC_PARAMS,
    MODEL_FAMILIES,
    SENSITIVITY_CONFIGS,
    SPLINE_FEATURES,
    build_model,
    fit_model,
    model_matrix,
    monotone_constraints,
    predict_positive_probability,
    restricted_cubic_spline_basis,
)

FEATURES = (
    "age",
    "bmi",
    "waist_cm",
    "systolic_mean",
    "diastolic_mean",
    "wealth",
    "education",
    "residence",
    "diet",
)
CATEGORICALS = ("education", "residence", "diet")

# Small settings keep the suite fast; the defaults are exercised separately.
FAST_PARAMS = {
    "random_forest": {"n_estimators": 30},
    "xgboost": {"n_estimators": 30},
    "catboost": {"iterations": 30},
    "lightgbm": {"n_estimators": 40},
    "lightgbm_monotone": {"n_estimators": 40},
}


def synthetic_frame(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    """A small frame with the continuous, numeric, and categorical features."""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 75, rows)
    bmi = rng.normal(25.0, 4.0, rows)
    waist = 60.0 + 0.45 * age + rng.normal(0.0, 5.0, rows)
    systolic = 90.0 + 0.6 * age + rng.normal(0.0, 8.0, rows)
    diastolic = 60.0 + 0.25 * age + rng.normal(0.0, 5.0, rows)
    risk = (
        0.04 * (age - 20)
        + 0.08 * (bmi - 20)
        + 0.03 * (waist - 60)
        + 0.02 * (systolic - 90)
        + 0.02 * (diastolic - 60)
    )
    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "waist_cm": waist,
            "systolic_mean": systolic,
            "diastolic_mean": diastolic,
            "wealth": rng.integers(1, 6, rows),
            "education": rng.choice(["none", "primary", "secondary"], rows),
            "residence": rng.choice(["urban", "rural"], rows),
            "diet": rng.choice(["daily", "weekly", "rarely"], rows),
            "target": (risk > np.median(risk)).astype(int),
        }
    )


def matrix_and_target(
    frame: pd.DataFrame | None = None,
    feature_columns: tuple[str, ...] = FEATURES,
    categorical_columns: tuple[str, ...] = CATEGORICALS,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = synthetic_frame() if frame is None else frame
    matrix, _ = model_matrix(frame, feature_columns, categorical_columns)
    return matrix, frame["target"].to_numpy()


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_every_family_builds_fits_and_predicts_probabilities(family):
    matrix, target = matrix_and_target()

    model = build_model(
        family,
        FAST_PARAMS.get(family, {}),
        feature_columns=FEATURES,
        categorical_columns=CATEGORICALS,
        seed=42,
    )
    fit_model(
        model,
        matrix,
        target,
        family=family,
        categorical_columns=CATEGORICALS,
    )
    probabilities = predict_positive_probability(model, matrix)

    assert probabilities.shape == (len(matrix),)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()


def test_logistic_and_lightgbm_defaults_are_unweighted():
    logistic = build_model(
        "logistic",
        feature_columns=FEATURES,
        categorical_columns=CATEGORICALS,
    )
    assert logistic.named_steps["classifier"].class_weight is None
    assert "class_weight" not in DEFAULT_LOGISTIC_PARAMS

    lightgbm = build_model(
        "lightgbm",
        feature_columns=FEATURES,
        categorical_columns=CATEGORICALS,
    )
    assert lightgbm.get_params().get("is_unbalance", False) is False
    assert "is_unbalance" not in DEFAULT_LIGHTGBM_PARAMS


def test_weighted_lightgbm_is_a_labelled_sensitivity_check_only():
    weight = SENSITIVITY_CONFIGS["lightgbm_weighted_sensitivity"]

    assert weight["family"] == "lightgbm"
    assert weight["params"]["is_unbalance"] is True
    assert "lightgbm_weighted_sensitivity" not in MODEL_FAMILIES
    assert not any("sensitivity" in family for family in MODEL_FAMILIES)


def test_xgboost_defaults_are_cpu_only_and_categorical_aware():
    model = build_model(
        "xgboost",
        feature_columns=FEATURES,
        categorical_columns=CATEGORICALS,
        seed=42,
    )
    params = model.get_params()

    assert params["device"] == "cpu"
    assert params["enable_categorical"] is True


def test_catboost_defaults_are_cpu_only_and_cap_iterations():
    model = build_model(
        "catboost",
        feature_columns=FEATURES,
        categorical_columns=CATEGORICALS,
        seed=42,
    )
    params = model.get_params()

    assert params["task_type"] == "CPU"
    assert params["iterations"] == DEFAULT_CATBOOST_PARAMS["iterations"]
    assert params["iterations"] <= 500


def test_model_families_include_every_scored_candidate():
    assert {
        "prevalence",
        "logistic",
        "lightgbm",
        "random_forest",
        "xgboost",
        "catboost",
        "logistic_spline",
        "lightgbm_monotone",
    } <= set(MODEL_FAMILIES)


def test_unknown_family_raises_with_the_family_list():
    with pytest.raises(ValueError) as error:
        build_model("tabnet", feature_columns=FEATURES)

    message = str(error.value)
    assert "unknown model family" in message
    assert all(family in message for family in MODEL_FAMILIES)


def test_restricted_cubic_spline_basis_shape_order_and_determinism():
    frame = synthetic_frame()
    matrix, _ = matrix_and_target(frame)

    basis = restricted_cubic_spline_basis(matrix)
    repeated = restricted_cubic_spline_basis(matrix)

    # Six basis functions per feature (five quantile knots, cubic, no bias),
    # grouped in the documented feature order.
    assert basis.shape == (len(matrix), 6 * len(SPLINE_FEATURES))
    for position, feature in enumerate(SPLINE_FEATURES):
        block = basis.columns[position * 6 : (position + 1) * 6]
        assert all(name.startswith(f"{feature}_") for name in block)
    assert not basis.isna().any().any()
    pd.testing.assert_frame_equal(basis, repeated)


def test_restricted_cubic_spline_basis_ignores_columns_not_present():
    frame = synthetic_frame()
    columns = ("age", "wealth", "education", "systolic_mean", "diastolic_mean")
    matrix, _ = model_matrix(frame, columns, ("education",))

    basis = restricted_cubic_spline_basis(matrix)

    assert basis.shape == (len(matrix), 3 * 6)
    assert all(
        not any(name.startswith(f"{feature}_") for name in basis.columns)
        for feature in ("bmi", "waist_cm")
    )


def test_spline_logistic_fits_and_predicts_without_the_anthropometry_columns():
    frame = synthetic_frame()
    columns = ("age", "wealth", "education", "systolic_mean", "diastolic_mean")
    categorical = ("education",)
    matrix, _ = model_matrix(frame, columns, categorical)

    model = build_model(
        "logistic_spline",
        {},
        feature_columns=columns,
        categorical_columns=categorical,
        seed=42,
    )
    fit_model(
        model,
        matrix,
        frame["target"].to_numpy(),
        family="logistic_spline",
        categorical_columns=categorical,
    )
    probabilities = predict_positive_probability(model, matrix)

    assert probabilities.shape == (len(matrix),)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()


def test_monotone_constraints_map_to_shuffled_feature_positions():
    columns = (
        "diet",
        "systolic_mean",
        "wealth",
        "age",
        "education",
        "bmi",
        "residence",
        "waist_cm",
        "diastolic_mean",
    )
    expected = [1 if column in SPLINE_FEATURES else 0 for column in columns]

    constraints = monotone_constraints(columns)
    assert constraints == expected

    # Wealth, education, residence, and diet are never constrained.
    assert [value for column, value in zip(columns, constraints) if column in {
        "wealth",
        "education",
        "residence",
        "diet",
    }] == [0, 0, 0, 0]

    model = build_model(
        "lightgbm_monotone",
        {"n_estimators": 20},
        feature_columns=columns,
        categorical_columns=("diet", "education", "residence"),
        seed=42,
    )
    assert list(model.get_params()["monotone_constraints"]) == expected


def monotone_frame(rows: int = 600, seed: int = 3) -> pd.DataFrame:
    """Risk rises with every constrained feature and carries mild noise."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20.0, 80.0, rows)
    bmi = rng.uniform(17.0, 40.0, rows)
    waist = rng.uniform(60.0, 120.0, rows)
    systolic = rng.uniform(90.0, 180.0, rows)
    diastolic = rng.uniform(55.0, 110.0, rows)
    risk = (
        0.04 * (age - 20.0)
        + 0.12 * (bmi - 17.0)
        + 0.05 * (waist - 60.0)
        + 0.03 * (systolic - 90.0)
        + 0.03 * (diastolic - 55.0)
    )
    target = (risk + rng.normal(0.0, 1.0, rows) > np.median(risk)).astype(int)
    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "waist_cm": waist,
            "systolic_mean": systolic,
            "diastolic_mean": diastolic,
            "target": target,
        }
    )


def test_monotone_lightgbm_never_lowers_risk_when_a_constrained_feature_rises():
    frame = monotone_frame()
    matrix, _ = model_matrix(frame, SPLINE_FEATURES, ())
    model = build_model(
        "lightgbm_monotone",
        {
            "n_estimators": 80,
            "learning_rate": 0.15,
            "num_leaves": 7,
            "min_child_samples": 10,
            "n_jobs": 1,
            "verbosity": -1,
        },
        feature_columns=SPLINE_FEATURES,
        categorical_columns=(),
        seed=42,
    )
    fit_model(
        model,
        matrix,
        frame["target"].to_numpy(),
        family="lightgbm_monotone",
    )

    base = matrix.iloc[:40].copy()
    baseline = predict_positive_probability(model, base)
    for column, delta in (
        ("age", 15.0),
        ("bmi", 6.0),
        ("waist_cm", 15.0),
        ("systolic_mean", 20.0),
        ("diastolic_mean", 15.0),
    ):
        raised = base.copy()
        raised[column] = raised[column] + delta
        probabilities = predict_positive_probability(model, raised)
        assert np.all(probabilities >= baseline - 1e-9), column
