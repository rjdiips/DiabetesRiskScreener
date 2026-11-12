"""Seed-averaging capability tests on a synthetic stochastic winner."""

from dataclasses import replace

import numpy as np
import pytest
from test_pipeline import LIGHTGBM_TEST_PARAMS, pr_shaped_features

from screener.pipeline import Candidate, evaluate_candidates, prepare_training_data
from screener.pr_model import pr_15_plus_config
from screener.seed_averaging import AveragedModel, fit_seed_averaged
from screener.splits import assign_pr_splits

N_FOLDS = 3


def evidence_for(features, candidates, *, feature_set="default"):
    config = replace(
        pr_15_plus_config(
            split_assignment=assign_pr_splits(features, seed=42, n_folds=N_FOLDS),
            seed=42,
            model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
            metadata={"feature_set": feature_set},
        ),
        n_folds=N_FOLDS,
        candidates=tuple(candidates),
    )
    prepared = prepare_training_data(features, config)
    return prepared, evaluate_candidates(prepared.development, config)


def test_deterministic_winner_is_fitted_once_without_averaging():
    features = pr_shaped_features()
    prepared, evidence = evidence_for(
        features,
        [Candidate(label="logistic-winner", family="logistic", params={"C": 0.5})],
    )

    fit = fit_seed_averaged(evidence, "logistic-winner")

    assert fit.report["deterministic"] is True
    assert fit.report["averaged"] is False
    assert fit.report["ensemble_size"] == 1
    assert fit.report["seeds"] == [42]
    assert not isinstance(fit.bundle.model, AveragedModel)
    assert fit.bundle.metadata["ensemble_size"] == 1
    probabilities = fit.bundle.predict_proba(prepared.development)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_stochastic_winner_is_refit_on_every_seed_and_calibrated_pooled():
    features = pr_shaped_features()
    prepared, evidence = evidence_for(
        features,
        [Candidate(label="lightgbm-winner", family="lightgbm", params=LIGHTGBM_TEST_PARAMS)],
    )

    fit = fit_seed_averaged(
        evidence,
        "lightgbm-winner",
        seeds=(42, 43, 44),
    )

    assert isinstance(fit.bundle.model, AveragedModel)
    assert len(fit.bundle.model.models) == 3
    assert fit.report["deterministic"] is False
    assert fit.report["averaged"] is True
    assert fit.report["ensemble_size"] == 3
    assert fit.report["seeds"] == [42, 43, 44]
    assert [row["seed"] for row in fit.report["per_seed"]] == [42, 43, 44]
    assert fit.report["mean"]["brier"] is not None
    assert fit.report["std"]["brier"] is not None
    assert fit.report["calibration_method"] in {"isotonic", "platt", "identity"}
    assert fit.report["threshold"] == fit.bundle.threshold
    assert fit.bundle.metadata["seeds"] == [42, 43, 44]
    assert fit.bundle.metadata["ensemble_size"] == 3
    assert fit.report["feature_count"] == len(fit.report["feature_columns"]) > 0

    probabilities = fit.bundle.predict_proba(prepared.development)
    assert probabilities.shape == (len(prepared.development),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    # Pooled calibration makes the ensemble no worse than its best single seed.
    assert (
        fit.report["averaged_validation"]["brier"]
        <= fit.report["best_single_seed_brier"] + 1e-12
    )


def test_seed_averaging_is_feature_set_agnostic():
    features = pr_shaped_features()
    prepared, evidence = evidence_for(
        features,
        [Candidate(label="lightgbm-winner", family="lightgbm", params=LIGHTGBM_TEST_PARAMS)],
        feature_set="measurement_no_bp",
    )

    fit = fit_seed_averaged(evidence, "lightgbm-winner", seeds=(42, 43))

    assert fit.report["seeds"] == [42, 43]
    assert fit.bundle.metadata["feature_set"] == "measurement_no_bp"
    assert fit.bundle.feature_columns == evidence.fit_handle.config.feature_columns
    assert fit.bundle.predict_proba(prepared.development).shape == (
        len(prepared.development),
    )


def test_seed_averaging_rejects_duplicate_seeds_and_unknown_candidates():
    features = pr_shaped_features()
    _, evidence = evidence_for(
        features,
        [Candidate(label="lightgbm-winner", family="lightgbm", params=LIGHTGBM_TEST_PARAMS)],
    )

    with pytest.raises(ValueError, match="distinct seeds"):
        fit_seed_averaged(evidence, "lightgbm-winner", seeds=(42, 42))
    with pytest.raises(ValueError, match="unknown candidate"):
        fit_seed_averaged(evidence, "missing")
