"""Unit tests for the pre-registered ranking and feature-bundle rules."""

import json
from dataclasses import replace

import pytest
from test_pipeline import LIGHTGBM_TEST_PARAMS, pr_shaped_features

from screener.comparison import (
    BUNDLE_BRIER_MARGIN,
    BUNDLE_PR_AUC_MARGIN,
    SIMPLICITY_PR_AUC_MARGIN,
    attach_winner_test_metrics,
    choose_bundle,
    comparison_candidates,
    comparison_config,
    rank_candidates,
    run_comparison,
    select_winner,
    write_comparison_artifacts,
)
from screener.pipeline import Candidate, prepare_training_data
from screener.pr_model import pr_15_plus_config
from screener.splits import assign_pr_splits

FAST_PARAMS = {
    "logistic": {"C": 1.0},
    "logistic_spline": {"C": 1.0},
    "random_forest": {"n_estimators": 20, "min_samples_leaf": 5, "n_jobs": 1},
    "lightgbm": LIGHTGBM_TEST_PARAMS,
    "xgboost": {"n_estimators": 20, "max_depth": 3, "n_jobs": 1},
    "catboost": {
        "iterations": 20,
        "depth": 3,
        "thread_count": 1,
        "verbose": False,
    },
    "lightgbm_monotone": LIGHTGBM_TEST_PARAMS,
}


def fast_config(features, *, feature_set="questionnaire_only"):
    """A three-fold comparison config with one fast setting per family."""
    assignment = assign_pr_splits(features, seed=42, n_folds=3)
    candidates = tuple(
        Candidate(label=family, family=family, params=params)
        for family, params in FAST_PARAMS.items()
    ) + (Candidate(label="B0-prevalence", family="prevalence"),)
    return replace(
        pr_15_plus_config(
            split_assignment=assignment,
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
            metadata={"feature_set": feature_set},
        ),
        n_folds=3,
        candidates=candidates,
    )


def candidate(
    label,
    *,
    family="lightgbm",
    pr_auc=0.30,
    roc_auc=0.70,
    brier=0.060,
    slope=1.0,
    intercept=0.0,
    gap=None,
):
    return {
        "label": label,
        "family": family,
        "params": {},
        "rank": None,
        "vetoes": [],
        "oof_metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier": brier,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        },
        "max_reliable_sensitivity_gap": gap,
    }


def test_higher_pr_auc_candidate_is_vetoed_by_worse_brier():
    ranked = rank_candidates(
        [
            candidate("higher", pr_auc=0.31, brier=0.070),
            candidate("lower", pr_auc=0.30, brier=0.060),
        ]
    )

    assert [item["label"] for item in ranked] == ["higher", "lower"]
    assert ranked[0]["rank"] == 1
    assert any("brier" in reason for reason in ranked[0]["vetoes"])
    assert ranked[1]["vetoes"] == []
    assert next(item for item in ranked if not item["vetoes"])["label"] == "lower"


def test_higher_pr_auc_candidate_is_vetoed_by_bad_calibration_slope():
    ranked = rank_candidates(
        [
            candidate("higher", pr_auc=0.31, slope=0.5),
            candidate("lower", pr_auc=0.30, slope=1.0),
        ]
    )

    assert any("slope" in reason for reason in ranked[0]["vetoes"])
    assert ranked[1]["vetoes"] == []


def test_higher_pr_auc_candidate_is_vetoed_by_a_large_subgroup_gap():
    ranked = rank_candidates(
        [
            candidate("higher", pr_auc=0.31, gap=0.25),
            candidate("lower", pr_auc=0.30, gap=0.05),
        ]
    )

    assert any("subgroup" in reason for reason in ranked[0]["vetoes"])
    assert ranked[1]["vetoes"] == []


def test_subgroup_gap_margin_is_fifteen_points():
    ranked = rank_candidates(
        [
            candidate("slightly_unequal", pr_auc=0.31, gap=0.14),
            candidate("unequal", pr_auc=0.30, gap=0.16),
        ]
    )

    assert ranked[0]["vetoes"] == []
    assert any("subgroup" in reason for reason in ranked[1]["vetoes"])


def test_age_band_gap_is_excluded_from_the_veto():
    item = candidate("age_gradient", pr_auc=0.31, gap=0.70)
    item["max_reliable_veto_sensitivity_gap"] = None

    ranked = rank_candidates([item])

    assert ranked[0]["vetoes"] == []


def test_near_tied_regression_model_is_preferred():
    ranked = rank_candidates(
        [
            candidate("tree", family="lightgbm", pr_auc=0.310),
            candidate(
                "spline",
                family="logistic_spline",
                pr_auc=0.310 - SIMPLICITY_PR_AUC_MARGIN + 0.001,
            ),
        ]
    )

    winner, reason = select_winner([item for item in ranked if not item["vetoes"]])

    assert winner["label"] == "spline"
    assert reason == "simplicity_preference"


def test_tree_wins_when_the_regression_model_is_outside_the_simplicity_margin():
    ranked = rank_candidates(
        [
            candidate("tree", family="lightgbm", pr_auc=0.320),
            candidate(
                "spline",
                family="logistic_spline",
                pr_auc=0.320 - SIMPLICITY_PR_AUC_MARGIN - 0.001,
            ),
        ]
    )

    winner, reason = select_winner([item for item in ranked if not item["vetoes"]])

    assert winner["label"] == "tree"
    assert reason == "ranking"


def test_lower_pr_auc_candidate_wins_when_the_higher_one_is_vetoed():
    ranked = rank_candidates(
        [
            candidate("higher", pr_auc=0.31, intercept=0.5),
            candidate("lower", pr_auc=0.29, intercept=0.0),
        ]
    )

    winner = next(item for item in ranked if not item["vetoes"])
    assert winner["label"] == "lower"


def test_prevalence_reference_is_never_ranked_or_can_win():
    ranked = rank_candidates(
        [
            candidate("B0-prevalence", family="prevalence", pr_auc=0.99),
            candidate("c1", pr_auc=0.30),
        ]
    )

    assert [item["label"] for item in ranked] == ["c1"]


def bundle(name, features, pr_auc, brier, *, eligible=True):
    return {
        "bundle": name,
        "feature_count": features,
        "eligible": eligible,
        "winner": {"pr_auc": pr_auc, "brier": brier},
    }


def test_smaller_bundle_within_both_margins_wins():
    chosen = choose_bundle(
        [
            bundle("measurement_full", 32, 0.300, 0.060),
            bundle("measurement_no_bp", 27, 0.299, 0.061),
        ]
    )

    assert chosen["bundle"] == "measurement_no_bp"
    assert chosen["feature_count"] == 27


def test_smaller_bundle_outside_the_pr_auc_margin_loses():
    chosen = choose_bundle(
        [
            bundle("measurement_full", 32, 0.300, 0.060),
            bundle(
                "measurement_no_bp",
                27,
                0.300 - BUNDLE_PR_AUC_MARGIN - 0.001,
                0.060,
            ),
        ]
    )

    assert chosen["bundle"] == "measurement_full"


def test_smaller_bundle_outside_the_brier_margin_loses():
    chosen = choose_bundle(
        [
            bundle("measurement_full", 32, 0.300, 0.060),
            bundle(
                "measurement_no_bp",
                27,
                0.300,
                0.060 + BUNDLE_BRIER_MARGIN + 0.001,
            ),
        ]
    )

    assert chosen["bundle"] == "measurement_full"


def test_no_bundle_passes_the_veto_fails_rather_than_deploying():
    with pytest.raises(ValueError, match="no feature bundle"):
        choose_bundle(
            [
                bundle("measurement_full", 32, 0.300, 0.060, eligible=False),
                bundle("measurement_no_bp", 27, 0.299, 0.061, eligible=False),
            ]
        )


def test_comparison_candidates_are_seven_families_three_settings_plus_reference():
    candidates = comparison_candidates()

    assert len(candidates) == 1 + 7 * 3
    assert candidates[0].family == "prevalence"
    assert candidates[0].label == "B0-prevalence"
    families = [item.family for item in candidates[1:]]
    for family in FAST_PARAMS:
        assert families.count(family) == 3


def test_comparison_config_records_the_feature_set():
    config = comparison_config(feature_set="measurement_no_bp")

    assert config.metadata["feature_set"] == "measurement_no_bp"
    assert config.metadata["run_role"] == "comparison"
    assert len(config.candidates) == 22


def test_comparison_runner_writes_out_of_fold_only_artifacts(tmp_path):
    features = pr_shaped_features()
    config = fast_config(features)
    prepared = prepare_training_data(features, config)

    artifact = run_comparison(
        prepared.development,
        config,
        output_dir=tmp_path,
        bundle="questionnaire_only",
    )

    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.md").exists()
    on_disk = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert artifact["bundle"] == "questionnaire_only"
    assert artifact["feature_set"] == "questionnaire_only"
    assert artifact["selection_split"] == "development_oof"
    assert artifact["test_evaluated_once"] is False
    assert len(artifact["candidates"]) == len(FAST_PARAMS) + 1
    assert len(artifact["ranking"]) == len(FAST_PARAMS)
    assert artifact["winner"] is not None
    assert artifact["winner"]["family"] != "prevalence"
    assert artifact["winner"]["final_run"] is None

    # No test-derived block can appear anywhere in a comparison artifact.
    keys = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                keys.append(key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(artifact)
    assert {key for key in keys if "test" in key.lower()} == {"test_evaluated_once"}


def test_winner_test_metrics_attach_only_to_the_winner(tmp_path):
    features = pr_shaped_features()
    config = fast_config(features)
    prepared = prepare_training_data(features, config)
    artifact = run_comparison(
        prepared.development,
        config,
        output_dir=tmp_path,
        bundle="measurement_full",
    )
    winner_label = artifact["winner"]["label"]

    attach_winner_test_metrics(
        artifact,
        final_run="artifacts/primary/run_x",
        test_metrics={"roc_auc": 0.71, "pr_auc": 0.20},
    )

    assert artifact["test_evaluated_once"] is True
    assert artifact["winner"]["final_run"] == "artifacts/primary/run_x"
    assert artifact["winner"]["test_metrics"] == {"roc_auc": 0.71, "pr_auc": 0.20}
    assert all("test_metrics" not in item for item in artifact["candidates"])
    assert all(
        item["label"] == winner_label or "final_run" not in item
        for item in artifact["candidates"]
    )
    write_comparison_artifacts(artifact, tmp_path / "attached")
    assert "artifacts/primary/run_x" in (
        tmp_path / "attached" / "comparison.md"
    ).read_text(encoding="utf-8")


def test_writer_refuses_a_non_winner_test_block(tmp_path):
    features = pr_shaped_features()
    config = fast_config(features)
    prepared = prepare_training_data(features, config)
    artifact = run_comparison(
        prepared.development,
        config,
        output_dir=tmp_path,
        bundle="measurement_full",
    )
    non_winner = next(
        item
        for item in artifact["candidates"]
        if item["label"] != artifact["winner"]["label"]
        and item["family"] != "prevalence"
    )
    non_winner["test_metrics"] = {"roc_auc": 0.99}

    with pytest.raises(ValueError, match="non-winner"):
        write_comparison_artifacts(artifact, tmp_path / "bad")


def test_sentinel_test_rows_cannot_reach_the_comparison_artifact(tmp_path):
    features = pr_shaped_features()
    config = fast_config(features)
    prepared = prepare_training_data(features, config)
    baseline = run_comparison(
        prepared.development,
        config,
        output_dir=tmp_path / "baseline",
        bundle="measurement_full",
    )

    sentinel = prepared.test["case_id"].iloc[0]
    corrupted = features.copy()
    mask = corrupted["case_id"].eq(sentinel)
    corrupted.loc[mask, "target"] = 1
    corrupted.loc[mask, "wealth"] = 987_654_321
    corrupted.loc[mask, "systolic_mean"] = 987_654_321.0
    poisoned = prepare_training_data(corrupted, config)
    assert sentinel in set(poisoned.test["case_id"])
    assert sentinel not in set(poisoned.development["case_id"])

    artifact = run_comparison(
        poisoned.development,
        config,
        output_dir=tmp_path / "poisoned",
        bundle="measurement_full",
    )

    assert artifact == baseline
    text = (tmp_path / "poisoned" / "comparison.json").read_text(encoding="utf-8")
    assert sentinel not in text
    assert "987654321" not in text
