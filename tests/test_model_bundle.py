"""The model bundle must round-trip and validate the training schema."""

import numpy as np
import pandas as pd
import pytest

from screener.calibration import IdentityCalibrator
from screener.model_bundle import (
    ConstantModel,
    ModelBundle,
    load_model_bundle,
    save_model_bundle,
)


def _bundle(threshold: float = 0.4) -> ModelBundle:
    model = ConstantModel(prevalence=0.35)
    return ModelBundle(
        model=model,
        calibrator=IdentityCalibrator(),
        feature_columns=("age", "female"),
        categorical_columns=(),
        category_levels={},
        threshold=threshold,
        target_definition="glucose > 140 OR diagnosed OR medication",
        metadata={"run_id": "test-run"},
        model_family="prevalence",
    )


def test_bundle_predicts_calibrated_probabilities_and_the_threshold_flag():
    bundle = _bundle(threshold=0.4)
    frame = pd.DataFrame({"age": [20, 60], "female": [0, 1]})

    probabilities = bundle.predict_proba(frame)

    assert np.allclose(probabilities, 0.35)
    assert list(bundle.predict(frame)) == [0, 0]

    bundle.threshold = 0.3
    assert list(bundle.predict(frame)) == [1, 1]


def test_bundle_applies_one_threshold_per_age_band():
    bundle = _bundle(threshold=0.9)
    bundle.threshold_rule = {
        "column": "age",
        "bins": [15.0, 25.0, 999.0],
        "labels": ["15-24", "25+"],
        "thresholds": {"15-24": 0.5, "25+": 0.3},
        "default_threshold": 0.9,
    }
    frame = pd.DataFrame({"age": [20, 30, None], "female": [0, 0, 0]})

    assert list(bundle.thresholds(frame)) == [0.5, 0.3, 0.9]
    assert list(bundle.predict(frame)) == [0, 1, 0]


def test_bundle_rejects_a_frame_missing_a_training_feature():
    bundle = _bundle()

    with pytest.raises(ValueError, match="missing"):
        bundle.predict_proba(pd.DataFrame({"age": [20]}))


def test_bundle_round_trips_through_disk(tmp_path):
    bundle = _bundle()
    path = save_model_bundle(bundle, tmp_path / "model_bundle.joblib")

    loaded = load_model_bundle(path)
    frame = pd.DataFrame({"age": [20, 60], "female": [0, 1]})

    assert loaded.feature_columns == bundle.feature_columns
    assert loaded.threshold == bundle.threshold
    assert loaded.target_definition == bundle.target_definition
    assert np.allclose(loaded.predict_proba(frame), bundle.predict_proba(frame))
