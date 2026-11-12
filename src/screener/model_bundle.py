"""The serialisable model bundle and its schema-validating prediction surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ConstantModel:
    """The prevalence-only baseline."""

    def __init__(self, prevalence: float = 0.5) -> None:
        self.prevalence = float(prevalence)

    def fit(self, features, target, **_: Any) -> ConstantModel:
        self.prevalence = float(np.mean(target))
        return self

    def predict_proba(self, features) -> np.ndarray:
        rows = len(features)
        positive = np.full(rows, self.prevalence, dtype=float)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class ModelBundle:
    """Everything inference needs, with no reach into serialized internals."""

    model: Any
    calibrator: Any
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ()
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    threshold: float = 0.5
    threshold_rule: dict = field(default_factory=dict)
    target_definition: str = ""
    metadata: dict = field(default_factory=dict)
    model_family: str = "lightgbm"

    def _matrix(self, features) -> pd.DataFrame:
        missing = [
            column for column in self.feature_columns if column not in features.columns
        ]
        if missing:
            raise ValueError(
                "inference frame is missing training features: " + ", ".join(missing)
            )

        matrix = pd.DataFrame(index=features.index)
        for column in self.feature_columns:
            if column in self.categorical_columns:
                levels = self.category_levels.get(column)
                values = features[column].astype("string").fillna("missing")
                matrix[column] = pd.Categorical(values, categories=levels)
            else:
                matrix[column] = pd.to_numeric(
                    features[column], errors="coerce"
                ).astype("float64")
        return matrix

    def predict_proba(self, features) -> np.ndarray:
        """Return calibrated probabilities after validating the schema."""
        raw = self.model.predict_proba(self._matrix(features))
        scores = np.asarray(raw)[:, 1] if np.ndim(raw) == 2 else np.asarray(raw)
        calibrated = np.asarray(self.calibrator.predict(scores), dtype=float)
        return np.clip(calibrated, 0.0, 1.0)

    def thresholds(self, features) -> np.ndarray:
        """Return the referral threshold for every row.

        A bundle without a rule uses its single threshold. A rule with age bins
        returns the band's threshold, and rows outside every band fall back to
        the rule's default.
        """
        rows = len(features)
        rule = self.threshold_rule or {}
        if not rule:
            return np.full(rows, float(self.threshold), dtype=float)
        fallback = float(rule.get("default_threshold", self.threshold))
        values = np.full(rows, fallback, dtype=float)
        column = rule.get("column")
        bins = rule.get("bins") or []
        labels = rule.get("labels") or []
        thresholds = rule.get("thresholds") or {}
        if column in features.columns and bins and labels:
            bands = pd.cut(
                pd.to_numeric(features[column], errors="coerce"),
                bins=list(bins),
                labels=list(labels),
                right=False,
            )
            bands = bands.astype("string").fillna("missing").astype(str).to_numpy()
            values = np.asarray(
                [float(thresholds.get(band, fallback)) for band in bands],
                dtype=float,
            )
        return values

    def predict(self, features) -> np.ndarray:
        """Return the referral flag at the bundle's chosen threshold(s)."""
        return (self.predict_proba(features) >= self.thresholds(features)).astype(int)


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Persist a bundle. The binary stays local and is gitignored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a bundle and refuse anything that is not a ``ModelBundle``."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle")
    return bundle
