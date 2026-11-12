"""Tests for the post-freeze MLP exploration and its artifact attachment."""

import copy
from dataclasses import replace

import pandas as pd
import pytest
from test_pipeline import pr_shaped_features

from screener.comparison import run_comparison
from screener.exploration import (
    attach_exploration,
    run_mlp_exploration,
    write_exploration,
)
from screener.pipeline import Candidate, prepare_training_data
from screener.pr_model import pr_15_plus_config
from screener.splits import assign_pr_splits

MLP_TEST_PARAMS = {
    "hidden_layer_sizes": (8,),
    "max_iter": 20,
    "batch_size": 128,
    "n_iter_no_change": 3,
}


def exploration_config(features):
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


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def test_mlp_exploration_reports_one_development_only_result(tmp_path):
    features = pr_shaped_features()
    config = exploration_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    mlp = Candidate(label="X1-mlp", family="mlp", params=MLP_TEST_PARAMS)

    result = run_mlp_exploration(
        prepared.development,
        config,
        winner_candidate=winner_candidate(artifact),
        mlp_candidate=mlp,
        replicates=50,
    )

    assert result["experiment"] == "mlp"
    assert result["candidate"]["label"] == "X1-mlp"
    assert result["candidate"]["family"] == "mlp"
    assert result["split"] == "development_oof"
    assert result["development_rows"] == len(prepared.development)
    assert result["winner"]["label"] == artifact["winner"]["label"]
    assert {"roc_auc", "brier"} <= set(result["winner_metrics"])
    assert {"roc_auc", "brier"} <= set(result["oof_metrics"]["experiment"])
    assert "delta_roc_auc" in result
    assert "delta_brier" in result
    assert "delta_roc_auc_ci" in result
    assert "max_reliable_sensitivity_gap" in result
    assert set(result["decision"]) == {
        "promoted",
        "conditions",
        "delta_roc_auc",
        "delta_brier",
        "margins",
    }
    assert set(result["decision"]["conditions"]) == {
        "roc_auc_margin",
        "ci_excludes_zero",
        "brier_within_margin",
        "subgroup_gap_ok",
    }
    assert {key for key in walk_keys(result) if "test" in key.lower()} == set()


def test_attachment_keeps_the_winner_and_the_test_flag_unchanged(tmp_path):
    features = pr_shaped_features()
    config = exploration_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    winner_before = copy.deepcopy(artifact["winner"])
    mlp = Candidate(label="X1-mlp", family="mlp", params=MLP_TEST_PARAMS)
    result = run_mlp_exploration(
        prepared.development,
        config,
        winner_candidate=winner_candidate(artifact),
        mlp_candidate=mlp,
        replicates=50,
    )

    attached = attach_exploration(artifact, "mlp", result)

    assert attached is artifact
    assert artifact["exploration"]["mlp"] is result
    assert artifact["winner"] == winner_before
    assert artifact["test_evaluated_once"] is False
    assert {key for key in walk_keys(artifact) if "test" in key.lower()} == {
        "test_evaluated_once"
    }

    non_promoted = copy.deepcopy(result)
    non_promoted["decision"]["promoted"] = False
    attach_exploration(artifact, "mlp_not_promoted", non_promoted)
    assert artifact["winner"] == winner_before
    assert artifact["test_evaluated_once"] is False


def test_writing_exploration_rewrites_the_comparison_report(tmp_path):
    features = pr_shaped_features()
    config = exploration_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    mlp = Candidate(label="X1-mlp", family="mlp", params=MLP_TEST_PARAMS)
    result = run_mlp_exploration(
        prepared.development,
        config,
        winner_candidate=winner_candidate(artifact),
        mlp_candidate=mlp,
        replicates=50,
    )

    paths = write_exploration(artifact, "mlp", result, output_dir=tmp_path)

    assert paths["exploration_json"].name == "exploration_mlp.json"
    assert paths["exploration_json"].exists()
    assert paths["comparison_json"].exists()
    report = paths["comparison_markdown"].read_text(encoding="utf-8")
    assert "## Exploration" in report
    assert "does not change the deployed artifact unless promoted" in report
    assert "Promoted:" in report


def test_a_development_frame_containing_a_test_row_is_rejected_before_fitting(
    tmp_path,
):
    features = pr_shaped_features()
    config = exploration_config(features)
    prepared, artifact = comparison_artifact(features, config, tmp_path)
    poisoned = pd.concat(
        [prepared.development, prepared.test.head(1)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="test rows"):
        run_mlp_exploration(
            poisoned,
            config,
            winner_candidate=winner_candidate(artifact),
            replicates=50,
        )
