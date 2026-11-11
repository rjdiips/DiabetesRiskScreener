"""Selection-evidence seam tests on a synthetic cluster-grouped table.

``evaluate_candidates`` is the only entry point under test: it must score the
configured candidates on the development folds and never reach the test split.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_pipeline import LIGHTGBM_TEST_PARAMS, pr_shaped_features

from screener.feature_policy import PR_15_PLUS_FEATURES
from screener.metrics import NET_BENEFIT_THRESHOLDS
from screener.pipeline import (
    SELECTION_SPLIT_LABEL,
    Candidate,
    evaluate_candidates,
    prepare_training_data,
)
from screener.pr_model import pr_15_plus_config
from screener.splits import assign_pr_splits

N_FOLDS = 3


def selection_config(features, *, metadata=None, feature_columns=None):
    """A fast PR config on the synthetic table, with three development folds."""
    assignment = assign_pr_splits(features, seed=42, n_folds=N_FOLDS)
    config = pr_15_plus_config(
        split_assignment=assignment,
        seed=42,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
        metadata=metadata,
    )
    config = replace(config, n_folds=N_FOLDS)
    if feature_columns is not None:
        config = replace(config, feature_columns=tuple(feature_columns))
    return config


def test_selection_returns_one_record_per_configured_candidate():
    features = pr_shaped_features()
    configured = (
        Candidate(label="prevalence-reference", family="prevalence"),
        Candidate(label="logistic-restricted", family="logistic", params={"C": 0.5}),
        Candidate(
            label="lightgbm-tuned",
            family="lightgbm",
            params=LIGHTGBM_TEST_PARAMS,
        ),
    )
    config = replace(selection_config(features), candidates=configured)
    prepared = prepare_training_data(features, config)

    evidence = evaluate_candidates(prepared.development, config)

    assert [record.label for record in evidence.records] == [
        "prevalence-reference",
        "logistic-restricted",
        "lightgbm-tuned",
    ]
    assert [record.family for record in evidence.records] == [
        "prevalence",
        "logistic",
        "lightgbm",
    ]
    assert [record.params for record in evidence.records] == [
        {},
        {"C": 0.5},
        LIGHTGBM_TEST_PARAMS,
    ]
    search_space = [candidate.to_dict() for candidate in configured]
    for record in evidence.records:
        assert record.tuning_record["trial_count"] == len(configured)
        assert record.tuning_record["selected"] == {
            "label": record.label,
            "family": record.family,
            "params": record.params,
        }
        assert record.tuning_record["search_space"] == search_space


def test_selection_record_carries_oof_evidence_for_exactly_the_development_rows():
    features = pr_shaped_features()
    config = selection_config(features)
    prepared = prepare_training_data(features, config)
    development = prepared.development

    evidence = evaluate_candidates(development, config)

    assert evidence.development_rows == len(development)
    assert evidence.fold_count == N_FOLDS
    assert len(evidence.records) == 3

    for record in evidence.records:
        n = len(development)
        assert record.oof_scores.shape == (n,)
        assert record.oof_probabilities.shape == (n,)
        assert record.target.shape == (n,)
        assert record.covered.shape == (n,)
        assert record.fold.shape == (n,)

        # The evidence is aligned with the development frame, row for row.
        assert np.array_equal(
            record.target, development[config.target_column].to_numpy()
        )
        assert np.array_equal(
            record.fold,
            pd.to_numeric(development["fold"]).astype(int).to_numpy(),
        )
        assert set(record.fold) == set(range(1, N_FOLDS + 1))
        assert record.covered.all()

        covered = record.covered
        assert np.isfinite(record.oof_scores[covered]).all()
        probabilities = record.oof_probabilities[covered]
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.isnan(record.oof_probabilities[~covered]).all()

        assert record.calibration_method in {
            *config.calibration_methods,
            "identity",
        }
        assert record.calibration_report["selected"] == record.calibration_method

        threshold_report = record.threshold_report
        assert threshold_report["target_sensitivity"] == config.target_sensitivity
        assert 0 <= threshold_report["threshold"] <= 1
        assert threshold_report["threshold"] == record.metrics["threshold"]
        assert threshold_report["confusion"]

        metrics = record.metrics
        assert metrics["n"] == int(covered.sum())
        for key in (
            "roc_auc",
            "pr_auc",
            "brier",
            "log_loss",
            "ece",
            "sensitivity",
            "specificity",
            "calibration_slope",
            "calibration_intercept",
        ):
            assert metrics[key] is not None, key

        # Subgroups are development out-of-fold and cover the scored rows only.
        assert {row["split"] for row in record.subgroups} == {
            SELECTION_SPLIT_LABEL
        }
        sex = [row for row in record.subgroups if row["subgroup"] == "sex"]
        assert {row["group"] for row in sex} == {"male", "female"}
        assert sum(row["n"] for row in sex) == int(covered.sum())

        assert len(record.reliability_curve) == config.reliability_bins
        assert len(record.net_benefit) == len(NET_BENEFIT_THRESHOLDS)
        assert [row["threshold"] for row in record.net_benefit] == list(
            NET_BENEFIT_THRESHOLDS
        )
        assert record.fold_metrics
        assert {row["fold"] for row in record.fold_metrics} == set(
            range(1, N_FOLDS + 1)
        )

        # Tables are JSON-safe and can be written straight into a comparison artifact.
        for table in (
            record.tuning_record,
            record.calibration_report,
            record.threshold_report,
            record.metrics,
            record.subgroups,
            record.reliability_curve,
            record.net_benefit,
            record.fold_metrics,
        ):
            json.dumps(table)


def test_selection_results_are_labelled_by_feature_set_and_count():
    features = pr_shaped_features()
    full = selection_config(features, metadata={"feature_set": "measurement_full"})
    questionnaire = selection_config(
        features,
        metadata={"feature_variant": "questionnaire_only"},
        feature_columns=PR_15_PLUS_FEATURES[:6],
    )
    assert full.feature_columns != questionnaire.feature_columns

    full_evidence = evaluate_candidates(
        prepare_training_data(features, full).development, full
    )
    questionnaire_evidence = evaluate_candidates(
        prepare_training_data(features, questionnaire).development,
        questionnaire,
    )

    assert full_evidence.feature_set == "measurement_full"
    assert full_evidence.feature_count == len(full.feature_columns) == 12
    assert questionnaire_evidence.feature_set == "questionnaire_only"
    assert questionnaire_evidence.feature_count == len(questionnaire.feature_columns)
    assert full_evidence.feature_set != questionnaire_evidence.feature_set

    for evidence in (full_evidence, questionnaire_evidence):
        for record in evidence.records:
            assert record.feature_set == evidence.feature_set
            assert record.feature_count == evidence.feature_count


def test_configs_differing_only_in_feature_columns_are_distinctly_labelled():
    features = pr_shaped_features()
    wider = selection_config(features)
    narrower = selection_config(features, feature_columns=PR_15_PLUS_FEATURES[:6])
    assert wider.metadata == narrower.metadata
    assert wider.feature_columns != narrower.feature_columns

    wider_evidence = evaluate_candidates(
        prepare_training_data(features, wider).development, wider
    )
    narrower_evidence = evaluate_candidates(
        prepare_training_data(features, narrower).development, narrower
    )

    assert wider_evidence.feature_set == narrower_evidence.feature_set == "default"
    assert wider_evidence.feature_count != narrower_evidence.feature_count
    assert (wider_evidence.feature_set, wider_evidence.feature_count) != (
        narrower_evidence.feature_set,
        narrower_evidence.feature_count,
    )


def test_default_candidate_path_still_scores_the_default_order():
    features = pr_shaped_features()
    deeper = {**LIGHTGBM_TEST_PARAMS, "num_leaves": 15}
    config = replace(
        selection_config(features),
        model_params={"lightgbm": [LIGHTGBM_TEST_PARAMS, deeper]},
    )
    prepared = prepare_training_data(features, config)

    evidence = evaluate_candidates(prepared.development, config)

    assert [record.family for record in evidence.records] == [
        "prevalence",
        "logistic",
        "lightgbm",
        "lightgbm",
    ]
    assert [record.label for record in evidence.records] == [
        "prevalence",
        "logistic",
        "lightgbm[0]",
        "lightgbm[1]",
    ]
    assert [record.params for record in evidence.records] == [
        {},
        {},
        LIGHTGBM_TEST_PARAMS,
        deeper,
    ]
    assert evidence.feature_set == "default"
    assert evidence.feature_count == len(config.feature_columns)
    for record in evidence.records:
        assert record.tuning_record["trial_count"] == 4
        assert record.tuning_record["selected"]["label"] == record.label
        assert record.tuning_record["selected"]["params"] == record.params


def test_a_broken_test_row_in_the_full_table_does_not_affect_selection():
    features = pr_shaped_features()
    config = selection_config(features)
    prepared = prepare_training_data(features, config)
    baseline = evaluate_candidates(prepared.development, config)

    sentinel = prepared.test["case_id"].iloc[0]
    corrupted = features.copy()
    is_sentinel = corrupted["case_id"].eq(sentinel)
    corrupted.loc[is_sentinel, "wealth"] = np.nan
    corrupted.loc[is_sentinel, "education"] = 1e12
    corrupted["tobacco_any"] = corrupted["tobacco_any"].astype(object)
    corrupted.loc[is_sentinel, "tobacco_any"] = "not-a-number"

    with_sentinel = prepare_training_data(corrupted, config)
    assert with_sentinel.test["case_id"].eq(sentinel).any()
    assert not with_sentinel.development["case_id"].eq(sentinel).any()

    evidence = evaluate_candidates(with_sentinel.development, config)

    assert [record.label for record in evidence.records] == [
        record.label for record in baseline.records
    ]
    for record, reference in zip(evidence.records, baseline.records):
        assert np.allclose(
            record.oof_probabilities,
            reference.oof_probabilities,
            equal_nan=True,
        )
        assert record.metrics == reference.metrics
        assert record.threshold_report == reference.threshold_report


def test_selection_rejects_a_test_row_before_any_model_is_fitted(monkeypatch):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared = prepare_training_data(features, config)
    leaked = pd.concat(
        [prepared.development, prepared.test.head(2)],
        ignore_index=True,
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("build_model must not run for a leaked frame")

    monkeypatch.setattr("screener.pipeline.build_model", fail_build)

    with pytest.raises(ValueError, match="test rows"):
        evaluate_candidates(leaked, config)

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_candidates(prepared.development.drop(columns="split"), config)


def test_selection_rejects_a_development_fold_outside_the_folds(monkeypatch):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared = prepare_training_data(features, config)
    bad = prepared.development.copy()
    bad.loc[bad.index[0], "fold"] = 99

    def fail_build(*args, **kwargs):
        raise AssertionError("build_model must not run for an invalid frame")

    monkeypatch.setattr("screener.pipeline.build_model", fail_build)

    with pytest.raises(ValueError, match="valid fold"):
        evaluate_candidates(bad, config)
