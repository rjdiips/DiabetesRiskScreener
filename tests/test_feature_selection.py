"""Tests for the post-freeze feature-selection experiment."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_pipeline import pr_shaped_features

from screener import feature_selection
from screener.comparison import run_comparison
from screener.exploration import attach_exploration, write_exploration
from screener.feature_selection import (
    EXPERIMENT_NAME,
    SELECTOR_METHOD,
    run_feature_selection_experiment,
)
from screener.pipeline import Candidate, prepare_training_data
from screener.pr_model import pr_15_plus_config
from screener.splits import assign_pr_splits


def selection_config(features):
    assignment = assign_pr_splits(features, seed=42, n_folds=3)
    return replace(
        pr_15_plus_config(
            split_assignment=assignment,
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
            metadata={"feature_set": "questionnaire_only"},
        ),
        n_folds=3,
        candidates=(
            Candidate(label="logistic", family="logistic", params={"C": 1.0}),
            Candidate(label="B0-prevalence", family="prevalence"),
        ),
    )


def comparison_artifact(features, config, output_dir):
    prepared = prepare_training_data(features, config)
    artifact = run_comparison(
        prepared.development,
        config,
        output_dir=output_dir,
        bundle="questionnaire_only",
    )
    return prepared, artifact


def winner_candidate(artifact):
    winner = artifact["winner"]
    return Candidate(
        label=winner["label"],
        family=winner["family"],
        params=winner["params"],
    )


def run_experiment(prepared, config, artifact):
    return run_feature_selection_experiment(
        prepared.development,
        config,
        winner_candidate=winner_candidate(artifact),
        n_repeats=2,
        replicates=20,
    )


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def test_feature_selection_reports_one_development_only_result(tmp_path):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)

    result = run_experiment(prepared, config, artifact)

    assert result["experiment"] == EXPERIMENT_NAME
    assert result["split"] == "development_oof"
    assert result["development_rows"] == len(prepared.development)
    assert result["requires_new_freeze"] == result["decision"]["promoted"]
    assert len(result["per_fold"]) == 3

    configured = set(config.feature_columns)
    assert {row["feature"] for row in result["stability"]} == configured
    assert sum(row["selected_folds"] for row in result["stability"]) == sum(
        len(row["selected_features"]) for row in result["per_fold"]
    )
    assert set(result["selected_features"]) | set(result["dropped_features"]) == configured
    assert not set(result["selected_features"]) & set(result["dropped_features"])
    assert result["feature_count_before"] == len(config.feature_columns)
    assert result["feature_count_after"] == len(result["selected_features"])

    assert set(result["decision"]["conditions"]) == {
        "roc_auc_margin",
        "ci_excludes_zero",
        "brier_within_margin",
        "subgroup_gap_ok",
    }
    assert {"roc_auc", "brier"} <= set(result["oof_metrics"]["winner"])
    assert {"roc_auc", "brier"} <= set(result["oof_metrics"]["experiment"])
    assert "delta_roc_auc_ci" in result
    assert "max_reliable_sensitivity_gap" in result
    assert {key for key in walk_keys(result) if "test" in key.lower()} == set()


def test_a_development_frame_containing_a_test_row_is_rejected_before_fitting(
    tmp_path,
):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    poisoned = pd.concat(
        [prepared.development, prepared.test.head(1)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="test rows"):
        run_feature_selection_experiment(
            poisoned,
            config,
            winner_candidate=winner_candidate(artifact),
            n_repeats=2,
            replicates=20,
        )


def test_a_non_promoted_result_leaves_the_winner_unchanged(tmp_path):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    winner_before = copy.deepcopy(artifact["winner"])
    result = copy.deepcopy(run_experiment(prepared, config, artifact))
    result["decision"]["promoted"] = False
    result["requires_new_freeze"] = False

    paths = write_exploration(
        artifact,
        "feature_selection",
        result,
        output_dir=tmp_path,
    )

    assert artifact["winner"] == winner_before
    assert artifact["test_evaluated_once"] is False
    written = json.loads(paths["comparison_json"].read_text(encoding="utf-8"))
    assert written["winner"] == winner_before
    assert written["test_evaluated_once"] is False


def test_a_promoted_decision_requires_a_new_freeze_and_keeps_the_winner(tmp_path):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    winner_before = copy.deepcopy(artifact["winner"])
    promoted = copy.deepcopy(run_experiment(prepared, config, artifact))
    promoted["decision"]["promoted"] = True
    promoted["requires_new_freeze"] = True

    attach_exploration(artifact, "feature_selection", promoted)

    assert promoted["requires_new_freeze"] is True
    assert artifact["winner"] == winner_before
    assert artifact["test_evaluated_once"] is False


def test_positive_importance_everywhere_keeps_the_configured_set(
    tmp_path, monkeypatch
):
    features = pr_shaped_features()
    config = selection_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)

    def fake_importance(model, matrix, target, *, n_repeats, random_state, scoring):
        assert scoring == "roc_auc"
        return SimpleNamespace(importances_mean=[1.0] * len(matrix.columns))

    monkeypatch.setattr(
        feature_selection, "permutation_importance", fake_importance
    )
    result = run_experiment(prepared, config, artifact)

    assert result["selected_features"] == list(config.feature_columns)
    assert result["dropped_features"] == []
    assert result["feature_count_after"] == result["feature_count_before"]
    assert result["selector"]["method"] == SELECTOR_METHOD
    assert SELECTOR_METHOD == "permutation_importance"
    assert all(row["stability"] == 1.0 for row in result["stability"])
