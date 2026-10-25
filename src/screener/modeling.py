"""Model factories and the shared feature-matrix preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .model_bundle import ConstantModel

MODEL_FAMILIES = ("prevalence", "logistic", "lightgbm")

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
    "is_unbalance": True,
    "importance_type": "gain",
    "n_jobs": 1,
    "verbosity": -1,
    "force_col_wise": True,
}

DEFAULT_LOGISTIC_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1_000,
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
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="missing"
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        classifier = LogisticRegression(
            C=settings["C"],
            max_iter=settings["max_iter"],
            class_weight="balanced",
        )
        return Pipeline(
            [
                ("preprocess", preprocess),
                ("classifier", classifier),
            ]
        )
    if family == "lightgbm":
        settings = {**DEFAULT_LIGHTGBM_PARAMS, **params}
        settings.setdefault("random_state", seed)
        return LGBMClassifier(**settings)
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
    """Fit one model, letting LightGBM use native categorical handling."""
    if family == "lightgbm":
        categorical = [column for column in categorical_columns if column in matrix.columns]
        model.fit(
            matrix,
            np.asarray(target),
            categorical_feature=categorical or "auto",
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
