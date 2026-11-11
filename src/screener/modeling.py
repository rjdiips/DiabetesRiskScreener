"""Model factories and the shared feature-matrix preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .model_bundle import ConstantModel

MODEL_FAMILIES = (
    "prevalence",
    "logistic",
    "lightgbm",
    "random_forest",
    "xgboost",
    "catboost",
    "logistic_spline",
    "lightgbm_monotone",
)

# The five continuous features that get spline terms and monotone constraints.
SPLINE_FEATURES = ("age", "bmi", "waist_cm", "systolic_mean", "diastolic_mean")
MONOTONE_FEATURES = SPLINE_FEATURES
# Restricted cubic spline basis: quantile knots fitted on the training matrix,
# so the knot placement cannot leak from held-out rows.
SPLINE_KNOTS = 5
SPLINE_DEGREE = 3

DEFAULT_LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "importance_type": "gain",
    "n_jobs": 1,
    "verbosity": -1,
    "force_col_wise": True,
}

DEFAULT_LOGISTIC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1_000,
}

DEFAULT_RANDOM_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "n_jobs": 1,
}

DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "device": "cpu",
    "eval_metric": "logloss",
    "n_jobs": 1,
}

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "loss_function": "Logloss",
    "task_type": "CPU",
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}

# One weighted LightGBM run is kept as a labelled sensitivity check only: it is
# not a candidate and must never enter MODEL_FAMILIES or a default candidate list.
LIGHTGBM_WEIGHTED_SENSITIVITY = "lightgbm_weighted_sensitivity"
SENSITIVITY_CONFIGS: dict[str, dict[str, Any]] = {
    LIGHTGBM_WEIGHTED_SENSITIVITY: {
        "family": "lightgbm",
        "params": {"is_unbalance": True},
    },
}


def model_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    categorical_columns: tuple[str, ...] | list[str] = (),
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return a numeric/categorical matrix plus the fitted category levels."""
    categorical = set(categorical_columns)
    matrix = pd.DataFrame(index=frame.index)
    levels: dict[str, list[str]] = {}

    for column in feature_columns:
        if column in categorical:
            values = frame[column].astype("string").fillna("missing")
            categories = sorted(values.unique().tolist())
            matrix[column] = pd.Categorical(values, categories=categories)
            levels[column] = categories
        else:
            matrix[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype("float64")
    return matrix, levels


def inference_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    categorical_columns: tuple[str, ...] | list[str],
    category_levels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build a matrix for inference using the training category levels."""
    categorical = set(categorical_columns)
    matrix = pd.DataFrame(index=frame.index)
    levels = category_levels or {}

    for column in feature_columns:
        if column in categorical:
            values = frame[column].astype("string").fillna("missing")
            matrix[column] = pd.Categorical(
                values, categories=levels.get(column)
            )
        else:
            matrix[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype("float64")
    return matrix


def _spline_transformer() -> SplineTransformer:
    """The pre-registered restricted cubic spline basis configuration."""
    return SplineTransformer(
        n_knots=SPLINE_KNOTS,
        degree=SPLINE_DEGREE,
        knots="quantile",
        include_bias=False,
        extrapolation="linear",
    )


def restricted_cubic_spline_basis(
    matrix: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] = SPLINE_FEATURES,
) -> pd.DataFrame:
    """Return the restricted cubic spline basis for the columns present.

    The knots are quantiles of the supplied matrix, so callers pass training
    rows only. Columns missing from the matrix (for example BMI and waist in
    the PR feature set) are ignored.
    """
    present = [column for column in feature_columns if column in matrix.columns]
    if not present:
        return pd.DataFrame(index=matrix.index)
    transformer = _spline_transformer()
    values = transformer.fit_transform(matrix[present].to_numpy(dtype="float64"))
    names = transformer.get_feature_names_out(present)
    return pd.DataFrame(values, index=matrix.index, columns=list(names))


def monotone_constraints(
    feature_columns: tuple[str, ...] | list[str],
) -> list[int]:
    """Map +1 monotone constraints to the risk features in matrix column order."""
    return [1 if column in MONOTONE_FEATURES else 0 for column in feature_columns]


def _categorical_encoder() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def _one_hot_preprocess(
    numeric: list[str],
    categorical: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", _categorical_encoder(), categorical),
        ],
        remainder="drop",
    )


def _build_logistic_spline(
    params: dict[str, Any],
    feature_columns: tuple[str, ...] | list[str],
    categorical_columns: tuple[str, ...] | list[str],
):
    settings = {**DEFAULT_LOGISTIC_PARAMS, **params}
    categorical = list(categorical_columns)
    numeric = [column for column in feature_columns if column not in categorical]
    spline = [column for column in numeric if column in SPLINE_FEATURES]
    linear = [column for column in numeric if column not in spline]

    transformers = []
    if spline:
        transformers.append(
            (
                "spline",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("spline", _spline_transformer()),
                        ("scale", StandardScaler()),
                    ]
                ),
                spline,
            )
        )
    if linear:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear,
            )
        )
    if categorical:
        transformers.append(("categorical", _categorical_encoder(), categorical))
    classifier = LogisticRegression(
        C=settings["C"],
        max_iter=settings["max_iter"],
    )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("classifier", classifier),
        ]
    )


def build_model(
    family: str,
    params: dict[str, Any] | None = None,
    *,
    feature_columns: tuple[str, ...] | list[str] = (),
    categorical_columns: tuple[str, ...] | list[str] = (),
    seed: int = 42,
):
    """Construct one model family with the shared preprocessing contract."""
    params = dict(params or {})
    if family == "prevalence":
        return ConstantModel()
    if family == "logistic":
        settings = {**DEFAULT_LOGISTIC_PARAMS, **params}
        categorical = list(categorical_columns)
        numeric = [
            column for column in feature_columns if column not in categorical
        ]
        classifier = LogisticRegression(
            C=settings["C"],
            max_iter=settings["max_iter"],
        )
        return Pipeline(
            [
                (
                    "preprocess",
                    _one_hot_preprocess(
                        numeric,
                        categorical,
                        scale_numeric=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )
    if family == "logistic_spline":
        return _build_logistic_spline(
            params,
            feature_columns,
            categorical_columns,
        )
    if family == "random_forest":
        settings = {**DEFAULT_RANDOM_FOREST_PARAMS, **params}
        settings.setdefault("random_state", seed)
        categorical = list(categorical_columns)
        numeric = [
            column for column in feature_columns if column not in categorical
        ]
        return Pipeline(
            [
                (
                    "preprocess",
                    _one_hot_preprocess(
                        numeric,
                        categorical,
                        scale_numeric=False,
                    ),
                ),
                ("classifier", RandomForestClassifier(**settings)),
            ]
        )
    if family in ("lightgbm", "lightgbm_monotone"):
        settings = {**DEFAULT_LIGHTGBM_PARAMS, **params}
        settings.setdefault("random_state", seed)
        if family == "lightgbm_monotone":
            settings["monotone_constraints"] = monotone_constraints(
                feature_columns
            )
        return LGBMClassifier(**settings)
    if family == "xgboost":
        from xgboost import XGBClassifier

        settings = {**DEFAULT_XGBOOST_PARAMS, **params}
        settings.setdefault("random_state", seed)
        return XGBClassifier(**settings)
    if family == "catboost":
        from catboost import CatBoostClassifier

        settings = {**DEFAULT_CATBOOST_PARAMS, **params}
        settings.setdefault("random_seed", seed)
        return CatBoostClassifier(**settings)
    raise ValueError(
        f"unknown model family '{family}'; expected one of "
        + ", ".join(MODEL_FAMILIES)
    )


def fit_model(
    model,
    matrix: pd.DataFrame,
    target,
    *,
    family: str,
    categorical_columns: tuple[str, ...] | list[str] = (),
):
    """Fit one model, letting the boosting families use native categories."""
    categorical = [
        column for column in categorical_columns if column in matrix.columns
    ]
    if family in ("lightgbm", "lightgbm_monotone"):
        model.fit(
            matrix,
            np.asarray(target),
            categorical_feature=categorical or "auto",
        )
    elif family == "catboost":
        model.fit(
            matrix,
            np.asarray(target),
            cat_features=categorical or None,
        )
    else:
        model.fit(matrix, np.asarray(target))
    return model


def predict_positive_probability(model, matrix: pd.DataFrame) -> np.ndarray:
    """Return the positive-class probability from any supported model."""
    probabilities = model.predict_proba(matrix)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        return probabilities[:, 1]
    return probabilities
