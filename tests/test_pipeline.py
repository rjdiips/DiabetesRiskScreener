"""Pipeline-reuse test on a synthetic PR-shaped feature table.

The test drives the public ``run_training_pipeline`` seam with the PR config,
so it proves there is no second modeling implementation and that the PR run
writes the same artifact set as the primary run.
"""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.feature_policy import PR_15_PLUS_FEATURES, PR_EXCLUDED_FEATURES
from screener.metrics import binary_metrics
from screener.model_bundle import load_model_bundle
from screener.pipeline import (
    Candidate,
    SampleFilter,
    TrainingConfig,
    prepare_training_data,
    run_training_pipeline,
    subgroup_metrics,
)
from screener.pr_model import pr_15_49_config, pr_15_plus_config
from screener.pr_variants import compare_pr_variants
from screener.splits import assign_pr_splits

LIGHTGBM_TEST_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.1,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 10,
    "n_jobs": 1,
    "verbosity": -1,
}


def pr_shaped_features(
    n_clusters: int = 60,
    households_per_cluster: int = 2,
    members_per_household: int = 4,
) -> pd.DataFrame:
    """A synthetic PR table with every feature, subgroup, and target column."""
    rng = np.random.default_rng(11)
    rows = []
    for cluster in range(1, n_clusters + 1):
        for household_offset in range(households_per_cluster):
            household = cluster * 100 + household_offset
            for member in range(members_per_household):
                age = 15 + (cluster * 7 + household_offset * 11 + member * 13) % 66
                female = int((cluster + household_offset + member) % 2 == 0)
                urban = int(cluster % 2 == 0)
                wealth = 1 + (cluster + member) % 5
                education = (cluster + household_offset + member) % 4
                tobacco_any = int((cluster + member) % 3 == 0)
                alcohol_any = int((cluster + household_offset) % 4 == 0)
                systolic_mean = 100.0 + age * 0.55 + rng.normal(0, 8)
                diastolic_mean = 65.0 + age * 0.25 + rng.normal(0, 5)
                bp_high = int(systolic_mean >= 140 or diastolic_mean >= 90)
                bp_medication = int((cluster + member) % 7 == 0)
                target = int(
                    age >= 55
                    or (bp_high and age >= 30)
                    or (cluster + member) % 7 == 0
                )
                target_fact_sheet = int(systolic_mean > 140 or bp_medication == 1)
                rows.append(
                    {
                        "case_id": f"c{cluster}-h{household_offset}-m{member}",
                        "cluster": cluster,
                        "household": household,
                        "line": member + 1,
                        "female": female,
                        "age": age,
                        "age_15_49": age <= 49,
                        "urban": urban,
                        "residence": 1 if urban else 2,
                        "wealth": wealth,
                        "education": education,
                        "state": 1 + cluster % 10,
                        "religion": 1 + cluster % 4,
                        "social_group": 1 + cluster % 3,
                        "tobacco_any": tobacco_any,
                        "alcohol_any": alcohol_any,
                        "systolic_mean": systolic_mean,
                        "diastolic_mean": diastolic_mean,
                        "bp_readings": 3,
                        "bp_high": bp_high,
                        "bp_medication": bp_medication,
                        "weight_scaled": 0.5 + rng.random(),
                        "target": target,
                        "target_fact_sheet": target_fact_sheet,
                        "cooking_fuel_clean": int(cluster % 2 == 0),
                    }
                )
    frame = pd.DataFrame(rows)
    # One respondent without a valid BP pair must be excluded by the PR config.
    frame.loc[frame.index[-1], ["systolic_mean", "diastolic_mean"]] = np.nan
    return frame


def test_pr_config_sets_the_twelve_feature_default_and_the_pr_sample():
    config = pr_15_plus_config()
    resolved = config.to_dict()

    assert config.feature_columns == PR_15_PLUS_FEATURES
    assert len(config.feature_columns) == 12
    assert resolved["feature_count"] == 12
    assert resolved["feature_table"].endswith("pr_features.parquet")
    assert resolved["target_column"] == "target"
    assert resolved["group_column"] == "cluster"
    assert resolved["weight_column"] == "weight_scaled"
    assert any(
        item["column"] == "age" and item["minimum"] == 15
        for item in resolved["sample_filters"]
    )
    assert any(
        item["column"] == "systolic_mean" and item["require_notna"]
        for item in resolved["sample_filters"]
    )
    subgroup_names = {item["name"] for item in resolved["subgroups"]}
    assert subgroup_names == {
        "sex",
        "age_band",
        "wealth",
        "residence",
        "education",
        "state",
    }
    assert set(PR_EXCLUDED_FEATURES).isdisjoint(config.feature_columns)


def test_pr_15_49_config_filters_the_same_partition_to_younger_adults():
    config = pr_15_49_config()
    resolved = config.to_dict()

    # The sample is the builder's `age_15_49` flag, not an independent age range.
    assert any(
        item["column"] == "age_15_49" and item["values"] == [True]
        for item in resolved["sample_filters"]
    )
    assert not any(
        item["column"] == "age" and item["maximum"] is not None
        for item in resolved["sample_filters"]
    )
    assert [item["name"] for item in resolved["subgroups"]][1] == "age_band"
    age_band = next(item for item in resolved["subgroups"] if item["name"] == "age_band")
    assert "50-59" not in age_band["labels"]
    assert resolved["metadata"]["sample"] == "age_15_49"
    assert resolved["metadata"]["comparison_run"] == "pr_15plus_bp"

    plus = pr_15_plus_config()
    assert plus.metadata["sample"] == "age_15_plus"
    assert "comparison_run" not in plus.metadata
    assert plus.feature_columns == config.feature_columns


def test_pr_run_reuses_the_pipeline_and_writes_every_artifact(tmp_path):
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    config = pr_15_plus_config(
        split_assignment=assignment,
        seed=42,
        bootstrap_replicates=20,
        shap_sample_size=50,
        state_min_test_rows=5,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path)

    required = {
        "split_assignment",
        "resolved_config",
        "run_metadata",
        "metrics",
        "subgroup_metrics",
        "calibration_curve",
        "threshold_analysis",
        "feature_importance",
        "shap_summary",
        "results_report",
        "model_bundle",
    }
    assert required <= set(result.artifacts)
    for path in result.artifacts.values():
        assert path.exists(), path

    metrics = json.loads(result.artifacts["metrics"].read_text(encoding="utf-8"))
    assert metrics["overall"]["n"] == result.metrics["split"]["test"]["rows"]
    assert metrics["overall"]["roc_auc"] is not None
    assert metrics["overall"]["threshold"] == result.metrics["overall"]["threshold"]
    assert metrics["calibration"]["selected"] in {"isotonic", "platt"}
    assert metrics["threshold_selection"]["target_sensitivity"] == 0.8

    # The fact-sheet target is reported separately from the primary target.
    assert metrics["secondary_target"]["column"] == "target_fact_sheet"
    assert metrics["secondary_target"]["n"] <= metrics["overall"]["n"]

    assert result.metrics["split"]["leakage"] == {
        "cluster_single_slot": True,
        "household_single_slot": True,
    }

    subgroup_names = set(result.subgroup_metrics["subgroup"])
    assert subgroup_names <= {
        "sex",
        "age_band",
        "wealth",
        "residence",
        "education",
        "state",
    }
    assert not {"religion", "social_group"} & subgroup_names
    state_rows = result.subgroup_metrics[
        result.subgroup_metrics["subgroup"] == "state"
    ]
    assert not state_rows.empty
    assert state_rows["n"].min() >= 5
    sex_rows = result.subgroup_metrics[result.subgroup_metrics["subgroup"] == "sex"]

    assert sex_rows["n"].sum() == metrics["overall"]["n"]
    assert set(sex_rows["group"]) == {"male", "female"}

    assert not result.shap_summary.empty
    assert {"feature", "mean_abs_shap"} <= set(result.shap_summary.columns)

    metadata = json.loads(
        result.artifacts["run_metadata"].read_text(encoding="utf-8")
    )
    assert metadata["split_source"] in {"reused_primary", "pr_only", "mixed"}
    assert metadata["seeds"]["split"] == 42
    assert metadata["package_versions"]["pandas"]
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    assert metadata["dependency_lock"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert len(metadata["dataset_fingerprint"]) == 64
    assert metadata["row_counts"]["rows_used"] == (
        metrics["overall"]["n"] + metrics["split"]["development"]["rows"]
    )
    assert metadata["row_counts"]["rows_after_filters"] == len(features) - 1

    resolved = json.loads(
        result.artifacts["resolved_config"].read_text(encoding="utf-8")
    )
    assert resolved["feature_columns"] == list(PR_15_PLUS_FEATURES)
    assert resolved["split_assignment_supplied"] is True

    report = result.artifacts["results_report"].read_text(encoding="utf-8")
    model_card = result.artifacts["model_card"].read_text(encoding="utf-8")
    assert "Δ Prevalence" in report
    assert "household head" in report
    assert "household head" in model_card

    bundle = load_model_bundle(result.artifacts["model_bundle"])
    frame = features.dropna(subset=["systolic_mean", "diastolic_mean"]).head(10)
    assert np.allclose(
        bundle.predict_proba(frame),
        result.model.predict_proba(frame),
    )
    assert bundle.threshold == metrics["overall"]["threshold"]


def test_pr_15_49_run_filters_the_flag_and_reuses_the_15_plus_partition(tmp_path):
    features = pr_shaped_features()
    parent_assignment = assign_pr_splits(features, seed=42)
    config = pr_15_49_config(
        split_assignment=parent_assignment,
        seed=42,
        bootstrap_replicates=20,
        shap_sample_size=50,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path)

    # The sample is exactly the BP-valid age_15_49 flag, not a second age range.
    expected = features[
        features["age_15_49"]
        & features["systolic_mean"].notna()
        & features["diastolic_mean"].notna()
    ]
    assert set(result.split_assignment["case_id"]) == set(expected["case_id"])
    assert result.run_metadata["row_counts"]["rows_after_filters"] == len(expected)
    assert result.metrics["split"]["rows"] == len(expected)

    # The partition is the PR 15+ assignment filtered to those rows: no re-split.
    merged = result.split_assignment.merge(
        parent_assignment,
        on="case_id",
        suffixes=("_run", "_parent"),
        validate="one_to_one",
    )
    assert len(merged) == len(result.split_assignment)
    assert merged["split_run"].eq(merged["split_parent"]).all()
    assert merged["cluster_run"].eq(merged["cluster_parent"]).all()
    assert (
        merged["fold_run"]
        .astype("Int64")
        .fillna(-1)
        .eq(merged["fold_parent"].astype("Int64").fillna(-1))
        .all()
    )

    # Calibration and the threshold are fitted on this run's own OOF predictions:
    # the threshold confusion counts cover the 15-49 development rows, not the
    # larger 15+ development set, and the calibrator is refitted for this run.
    metrics = json.loads(result.artifacts["metrics"].read_text(encoding="utf-8"))
    assert metrics["calibration"]["selected"] in {"isotonic", "platt", "identity"}
    confusion = metrics["threshold_selection"]["confusion"]
    assert sum(confusion.values()) == metrics["split"]["development"]["rows"]
    assert metrics["threshold_selection"]["target_sensitivity"] == 0.8

    # Every artifact of the PR 15+ run is written for the age-matched run too.
    required = {
        "split_assignment",
        "resolved_config",
        "run_metadata",
        "metrics",
        "subgroup_metrics",
        "calibration_curve",
        "threshold_analysis",
        "feature_importance",
        "shap_summary",
        "results_report",
        "model_bundle",
    }
    assert required <= set(result.artifacts)
    for path in result.artifacts.values():
        assert path.exists(), path

    age_groups = set(
        result.subgroup_metrics.loc[
            result.subgroup_metrics["subgroup"].eq("age_band"), "group"
        ]
    )
    assert age_groups == {"15-24", "25-34", "35-44", "45-49"}

    # Run metadata and the report state which run this is and what it compares to.
    metadata = json.loads(
        result.artifacts["run_metadata"].read_text(encoding="utf-8")
    )
    assert metadata["name"] == "pr_15to49_bp"
    assert metadata["sample"] == "age_15_49"
    assert metadata["comparison_run"] == "pr_15plus_bp"
    assert metadata["split_source"] in {"reused_primary", "pr_only", "mixed"}

    report = result.artifacts["results_report"].read_text(encoding="utf-8")
    assert "# pr_15to49_bp results" in report
    assert "## Run role" in report
    assert "pr_15plus_bp" in report
    assert "age_15_49" in report


def test_smoke_mode_trains_on_a_small_sample_not_the_full_table(tmp_path):
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    config = pr_15_plus_config(
        split_assignment=assignment,
        seed=42,
        bootstrap_replicates=10,
        shap_sample_size=25,
        smoke_rows=200,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path)

    assert result.run_metadata["row_counts"]["rows_after_filters"] == len(features) - 1
    assert result.run_metadata["row_counts"]["rows_used"] == 200
    assert result.run_metadata["row_counts"]["smoke"] is True
    assert result.artifacts["model_bundle"].exists()


def test_pipeline_loads_a_split_assignment_from_the_configured_path(tmp_path):
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    assignment_path = tmp_path / "pr_split_assignment.parquet"
    assignment.to_parquet(assignment_path, index=False)

    config = pr_15_plus_config(
        split_assignment=None,
        split_assignment_path=str(assignment_path),
        seed=42,
        bootstrap_replicates=10,
        shap_sample_size=25,
        smoke_rows=200,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path / "runs")

    assert result.run_metadata["split_source"] == "pr_only"
    assert result.run_metadata["row_counts"]["rows_used"] == 200
    assert result.artifacts["model_bundle"].exists()


def test_pipeline_rejects_a_target_column_as_a_feature(tmp_path):
    features = pr_shaped_features()
    config = pr_15_plus_config()
    bad = replace(
        config,
        feature_columns=config.feature_columns + ("target",),
    )

    with pytest.raises(ValueError, match="forbidden"):
        run_training_pipeline(features, bad, tmp_path)


def test_subgroup_metrics_report_gaps_and_flag_small_groups(tmp_path):
    features = pr_shaped_features(n_clusters=400)
    assignment = assign_pr_splits(features, seed=42)
    config = pr_15_plus_config(
        split_assignment=assignment,
        seed=42,
        bootstrap_replicates=5,
        shap_sample_size=10,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path)
    subgroups = result.subgroup_metrics

    required_columns = {
        "subgroup",
        "group",
        "n",
        "positives",
        "prevalence",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "calibration_error",
        "gap_roc_auc",
        "gap_pr_auc",
        "gap_sensitivity",
        "gap_calibration_error",
        "reliable",
        "unreliable_reason",
        "audit_only",
        "note",
    }
    assert required_columns <= set(subgroups.columns)
    assert set(subgroups["split"]) == {"test"}

    # Sex and age band are mandatory and cover the whole test set once each.
    sex = subgroups[subgroups["subgroup"] == "sex"]
    age_band = subgroups[subgroups["subgroup"] == "age_band"]
    assert set(sex["group"]) == {"male", "female"}
    assert set(age_band["group"]) == {
        "15-24",
        "25-34",
        "35-44",
        "45-49",
        "50-59",
        "60-69",
        "70+",
    }
    assert sex["n"].sum() == result.metrics["overall"]["n"]
    assert age_band["n"].sum() == result.metrics["overall"]["n"]

    # The synthetic test set has hundreds of respondents per sex, so those rows
    # clear the 200-row/50-positive rule; the oldest band does not.
    assert sex["reliable"].all()
    oldest = age_band[age_band["group"] == "70+"]
    assert len(oldest) == 1
    assert not oldest.iloc[0]["reliable"]
    assert oldest.iloc[0]["unreliable_reason"]
    assert age_band["n"].min() < config.small_group_min_rows

    # Every state in this small synthetic run has fewer than 500 test rows, so
    # no state row is reported.
    assert subgroups[subgroups["subgroup"] == "state"].empty

    # The gaps are measured against the overall test set.
    for column, overall_key in (
        ("gap_roc_auc", "roc_auc"),
        ("gap_pr_auc", "pr_auc"),
        ("gap_sensitivity", "sensitivity"),
    ):
        row = sex.iloc[0]
        expected = round(row[column.replace("gap_", "")] - result.metrics["overall"][overall_key], 6)
        assert row[column] == expected


def test_subgroup_metrics_label_development_and_test_rows_with_the_same_rules():
    features = pr_shaped_features(n_clusters=100)
    config = pr_15_plus_config()
    probabilities = np.linspace(0.05, 0.95, len(features))
    overall = binary_metrics(
        features[config.target_column].to_numpy(),
        probabilities,
        threshold=0.5,
        n_bins=config.reliability_bins,
    )

    development = subgroup_metrics(
        features,
        config.target_column,
        probabilities,
        config,
        overall,
        "development",
    )
    test = subgroup_metrics(
        features,
        config.target_column,
        probabilities,
        config,
        overall,
        "test",
    )

    assert set(development["split"]) == {"development"}
    assert set(test["split"]) == {"test"}
    pd.testing.assert_frame_equal(
        development.drop(columns="split"), test.drop(columns="split")
    )

    # The existing reliability and small-group rules are unchanged.
    sex = development[development["subgroup"] == "sex"]
    assert set(sex["group"]) == {"male", "female"}
    assert sex["reliable"].all()
    oldest = development[
        development["subgroup"].eq("age_band") & development["group"].eq("70+")
    ]
    assert not oldest.iloc[0]["reliable"]
    assert oldest.iloc[0]["unreliable_reason"]


def test_subgroup_metrics_exclude_unscored_development_rows_from_the_counts():
    features = pr_shaped_features(n_clusters=100)
    config = pr_15_plus_config()
    probabilities = np.linspace(0.05, 0.95, len(features))
    target = features[config.target_column]
    uncovered = np.zeros(len(features), dtype=bool)
    uncovered[features.index[target.eq(1)][:3]] = True
    uncovered[features.index[target.eq(0)][:1]] = True
    probabilities[uncovered] = np.nan
    overall = binary_metrics(
        target.to_numpy()[~uncovered],
        probabilities[~uncovered],
        threshold=0.5,
        n_bins=config.reliability_bins,
    )

    development = subgroup_metrics(
        features,
        config.target_column,
        probabilities,
        config,
        overall,
        "development",
    )

    sex = development[development["subgroup"] == "sex"]
    scored = features[~uncovered]
    assert set(development["split"]) == {"development"}
    assert sex["n"].sum() == len(scored) < len(features)
    assert sex["positives"].sum() == int(scored[config.target_column].sum())
    assert sex["positives"].sum() < int(target.sum())


def test_fairness_variant_reports_the_household_head_audit_subgroups(tmp_path):
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    config = pr_15_plus_config(
        split_assignment=assignment,
        seed=42,
        bootstrap_replicates=5,
        shap_sample_size=10,
        feature_variant="fairness",
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path)

    assert config.feature_columns == PR_15_PLUS_FEATURES + (
        "religion",
        "social_group",
    )
    audit_rows = result.subgroup_metrics[
        result.subgroup_metrics["subgroup"].isin(["religion", "social_group"])
    ]
    assert set(audit_rows["subgroup"]) == {"religion", "social_group"}
    assert audit_rows["audit_only"].all()
    assert audit_rows["note"].str.contains("household head").all()

    report = result.artifacts["results_report"].read_text(encoding="utf-8")
    assert "household head" in report
    assert "does not by itself make a model fair" in report

    model_card = result.artifacts["model_card"].read_text(encoding="utf-8")
    assert "household head" in model_card
    assert "feature-set variant" in model_card
    assert "`fairness`" in model_card
    assert "The PR default excludes" not in model_card


def test_variant_comparison_compares_real_pipeline_runs(tmp_path):
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    shared = {
        "split_assignment": assignment,
        "seed": 42,
        "bootstrap_replicates": 5,
        "shap_sample_size": 10,
        "model_params": {"lightgbm": LIGHTGBM_TEST_PARAMS},
    }

    default = run_training_pipeline(
        features,
        pr_15_plus_config(**shared),
        tmp_path / "default",
    )
    state = run_training_pipeline(
        features,
        pr_15_plus_config(feature_variant="state", **shared),
        tmp_path / "state",
    )

    comparison = compare_pr_variants(default, {"state": state})

    assert comparison["variants"]["state"]["added_features"] == ["state"]
    expected_delta = round(
        state.metrics["overall"]["roc_auc"] - default.metrics["overall"]["roc_auc"],
        6,
    )
    assert comparison["variants"]["state"]["deltas"]["roc_auc"] == expected_delta
    assert comparison["variants"]["state"]["subgroups"]["sex"]
    assert comparison["variants"]["state"]["subgroups"]["age_band"]
    assert comparison["small_group_rules"]["small_group_min_rows"] == 200


def test_preparation_seam_splits_disjoint_frames_that_match_the_assignment():
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    config = pr_15_plus_config(split_assignment=assignment, seed=42)

    prepared = prepare_training_data(features, config)

    development = prepared.development
    test = prepared.test
    assert set(development["case_id"]).isdisjoint(set(test["case_id"]))
    assert len(development) + len(test) == prepared.row_counts["rows_used"]
    assert len(development) + len(test) == len(features) - 1
    assert prepared.split_source == "pr_only"

    # The assignment is one row per used respondent and matches both frames.
    by_split = prepared.assignment.groupby("split")["case_id"].agg(set).to_dict()
    assert by_split["development"] == set(development["case_id"])
    assert by_split["test"] == set(test["case_id"])

    merged = development[["case_id", "split", "fold"]].merge(
        prepared.assignment[["case_id", "split", "fold"]],
        on="case_id",
        suffixes=("_frame", "_assignment"),
        validate="one_to_one",
    )
    assert len(merged) == len(development)
    assert merged["split_frame"].eq("development").all()
    assert merged["split_assignment"].eq("development").all()
    assert merged["fold_frame"].eq(merged["fold_assignment"]).all()

    # Every development respondent has a usable fold for cross-validation.
    folds = development["fold"]
    assert folds.notna().all()
    assert set(folds.astype(int)) == set(range(1, config.n_folds + 1))

    assert prepared.split_summary["rows"] == len(development) + len(test)
    assert prepared.split_summary["development"]["rows"] == len(development)
    assert prepared.split_summary["test"]["rows"] == len(test)


def test_preparation_seam_never_returns_a_test_row_in_development():
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    config = pr_15_plus_config(split_assignment=assignment, seed=42)

    prepared = prepare_training_data(features, config)

    test_only = set(
        assignment.loc[assignment["split"].eq("test"), "case_id"].astype(str)
    )
    assert test_only & set(prepared.development["case_id"]) == set()
    assert test_only <= set(prepared.test["case_id"])
    development_splits = prepared.assignment.set_index("case_id").loc[
        sorted(prepared.development["case_id"]), "split"
    ]
    assert development_splits.eq("development").all()


def test_preparation_seam_rejects_an_assignment_that_misses_a_respondent():
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    partial = assignment[~assignment["case_id"].eq(features.iloc[0]["case_id"])]
    config = pr_15_plus_config(split_assignment=partial, seed=42)

    with pytest.raises(ValueError, match="does not cover"):
        prepare_training_data(features, config)


def test_preparation_seam_rejects_a_development_fold_outside_the_folds():
    features = pr_shaped_features()
    assignment = assign_pr_splits(features, seed=42)
    assignment.loc[assignment["split"].eq("development"), "fold"] = 99
    config = pr_15_plus_config(split_assignment=assignment, seed=42)

    with pytest.raises(ValueError, match="valid fold|between 1 and"):
        prepare_training_data(features, config)


def test_preparation_seam_rejects_a_filter_that_empties_the_sample():
    features = pr_shaped_features()
    config = replace(
        pr_15_plus_config(),
        sample_filters=(SampleFilter(column="age", minimum=200),),
    )

    with pytest.raises(ValueError, match="retain no rows"):
        prepare_training_data(features, config)


def test_preparation_seam_reports_missing_features_and_assignment(tmp_path):
    features = pr_shaped_features()
    missing_column = replace(
        pr_15_plus_config(),
        feature_columns=pr_15_plus_config().feature_columns + ("not_a_feature",),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_training_data(features, missing_column)

    missing_path = pr_15_plus_config(
        split_assignment_path=str(tmp_path / "missing_assignment.parquet")
    )
    with pytest.raises(ValueError, match="split assignment file"):
        prepare_training_data(features, missing_path)


def test_preparation_seam_rejects_a_missing_or_empty_feature_table():
    config = pr_15_plus_config()

    for features in (None, pd.DataFrame()):
        with pytest.raises(ValueError, match="feature table is empty"):
            prepare_training_data(features, config)


def test_training_config_serialises_candidates_and_the_pinned_candidate():
    candidate = Candidate(label="logistic-final", family="logistic", params={"C": 0.5})
    defaults = TrainingConfig().to_dict()

    assert defaults["candidates"] == []
    assert defaults["pinned_candidate"] is None
    assert defaults["seed_averaging"] is False
    assert defaults["seed_averaging_seeds"] == []

    resolved = replace(
        TrainingConfig(),
        candidates=(candidate,),
        pinned_candidate=candidate,
        seed_averaging=True,
        seed_averaging_seeds=(42, 43),
    ).to_dict()
    expected = {"label": "logistic-final", "family": "logistic", "params": {"C": 0.5}}
    assert resolved["candidates"] == [expected]
    assert resolved["pinned_candidate"] == expected
    assert resolved["seed_averaging"] is True
    assert resolved["seed_averaging_seeds"] == [42, 43]
    with pytest.raises(FrozenInstanceError):
        candidate.family = "lightgbm"


def test_configured_candidates_drive_the_trial_loop(tmp_path):
    features = pr_shaped_features()
    config = replace(
        pr_15_plus_config(
            split_assignment=assign_pr_splits(features, seed=42),
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
        ),
        candidates=(
            Candidate(label="prevalence-reference", family="prevalence"),
            Candidate(
                label="logistic-restricted",
                family="logistic",
                params={"C": 0.5},
            ),
        ),
    )

    result = run_training_pipeline(features, config, tmp_path)

    trials = result.metrics["model_selection"]["trials"]
    assert [trial["label"] for trial in trials] == [
        "prevalence-reference",
        "logistic-restricted",
    ]
    assert [trial["family"] for trial in trials] == ["prevalence", "logistic"]
    assert trials[1]["params"] == {"C": 0.5}
    assert sum(trial["selected"] for trial in trials) == 1

    # No family outside the configured list is scored, even with tuning params.
    assert "lightgbm" not in {trial["family"] for trial in trials}

    selected = next(trial for trial in trials if trial["selected"])
    resolved = json.loads(
        result.artifacts["resolved_config"].read_text(encoding="utf-8")
    )
    assert result.metrics["model_selection"]["selected_label"] == selected["label"]
    assert resolved["selected_label"] == selected["label"]
    assert resolved["selected_family"] == selected["family"]
    assert resolved["resolved_model_params"] == selected["params"]


def test_pinned_candidate_scores_and_fits_exactly_one_candidate(tmp_path):
    features = pr_shaped_features()
    pinned = Candidate(label="chosen-candidate", family="logistic", params={"C": 0.5})
    config = replace(
        pr_15_plus_config(
            split_assignment=assign_pr_splits(features, seed=42),
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
        ),
        candidates=(
            Candidate(label="prevalence-reference", family="prevalence"),
            Candidate(label="lightgbm-candidate", family="lightgbm"),
        ),
        pinned_candidate=pinned,
    )

    result = run_training_pipeline(features, config, tmp_path)

    trials = result.metrics["model_selection"]["trials"]
    assert len(trials) == 1
    assert trials[0]["label"] == "chosen-candidate"
    assert trials[0]["family"] == "logistic"
    assert trials[0]["params"] == {"C": 0.5}
    assert trials[0]["selected"] is True

    resolved = json.loads(
        result.artifacts["resolved_config"].read_text(encoding="utf-8")
    )
    assert resolved["selected_label"] == "chosen-candidate"
    assert resolved["selected_family"] == "logistic"
    assert resolved["resolved_model_params"] == {"C": 0.5}
    assert result.model.model_family == "logistic"


def test_default_trial_order_is_unchanged(tmp_path):
    features = pr_shaped_features()
    config = pr_15_plus_config(
        split_assignment=assign_pr_splits(features, seed=42),
        seed=42,
        bootstrap_replicates=5,
        shap_sample_size=10,
        model_params={"lightgbm": LIGHTGBM_TEST_PARAMS},
    )

    result = run_training_pipeline(features, config, tmp_path)

    trials = result.metrics["model_selection"]["trials"]
    assert [trial["family"] for trial in trials] == [
        "prevalence",
        "logistic",
        "lightgbm",
    ]
    assert [trial["label"] for trial in trials] == [
        "prevalence",
        "logistic",
        "lightgbm",
    ]
    assert trials[2]["params"] == LIGHTGBM_TEST_PARAMS
    assert sum(trial["selected"] for trial in trials) == 1

    tuning_record = result.metrics["model_selection"]["tuning_record"]
    selected = next(trial for trial in trials if trial["selected"])
    assert tuning_record["trial_count"] == 3
    assert tuning_record["selected"] == {
        "label": selected["label"],
        "family": selected["family"],
        "params": selected["params"],
    }
    assert tuning_record["search_space"] == [
        {"label": trial["label"], "family": trial["family"], "params": trial["params"]}
        for trial in trials
    ]


def test_tuning_settings_expand_into_scored_candidates(tmp_path):
    features = pr_shaped_features()
    deeper = {**LIGHTGBM_TEST_PARAMS, "num_leaves": 15}
    config = pr_15_plus_config(
        split_assignment=assign_pr_splits(features, seed=42),
        seed=42,
        bootstrap_replicates=5,
        shap_sample_size=10,
        model_params={"lightgbm": [LIGHTGBM_TEST_PARAMS, deeper]},
    )

    result = run_training_pipeline(features, config, tmp_path)

    trials = result.metrics["model_selection"]["trials"]
    assert [trial["family"] for trial in trials] == [
        "prevalence",
        "logistic",
        "lightgbm",
        "lightgbm",
    ]
    assert len({trial["label"] for trial in trials}) == 4
    assert [trial["params"] for trial in trials[2:]] == [LIGHTGBM_TEST_PARAMS, deeper]

    selected = next(trial for trial in trials if trial["selected"])
    resolved = json.loads(
        result.artifacts["resolved_config"].read_text(encoding="utf-8")
    )
    assert resolved["resolved_model_params"] == selected["params"]

    tuning_record = result.metrics["model_selection"]["tuning_record"]
    assert tuning_record["trial_count"] == 4
    assert tuning_record["selected"] == {
        "label": selected["label"],
        "family": selected["family"],
        "params": selected["params"],
    }
    assert [item["label"] for item in tuning_record["search_space"]] == [
        trial["label"] for trial in trials
    ]
    assert [item["params"] for item in tuning_record["search_space"]] == [
        trial["params"] for trial in trials
    ]


def test_seed_averaging_refits_a_stochastic_winner_across_every_seed(tmp_path):
    features = pr_shaped_features()
    params = {**LIGHTGBM_TEST_PARAMS, "n_estimators": 20}
    config = replace(
        pr_15_plus_config(
            split_assignment=assign_pr_splits(features, seed=42, n_folds=3),
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
        ),
        n_folds=3,
        pinned_candidate=Candidate(
            label="lightgbm-winner",
            family="lightgbm",
            params=params,
        ),
        seed_averaging=True,
        seed_averaging_seeds=(42, 43),
    )

    result = run_training_pipeline(features, config, tmp_path)

    report = result.metrics["seed_averaging"]
    assert report["deterministic"] is False
    assert report["ensemble_size"] == 2
    assert report["seeds"] == [42, 43]
    assert result.run_metadata["seed_averaging"]["ensemble_size"] == 2
    assert result.model.metadata["ensemble_size"] == 2
    assert result.model.metadata["seeds"] == [42, 43]
    assert result.metrics["overall"]["n"] == result.metrics["split"]["test"]["rows"]


def test_seed_averaging_keeps_a_deterministic_winner_single(tmp_path):
    features = pr_shaped_features()
    config = replace(
        pr_15_plus_config(
            split_assignment=assign_pr_splits(features, seed=42, n_folds=3),
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
        ),
        n_folds=3,
        pinned_candidate=Candidate(
            label="logistic-winner",
            family="logistic",
            params={"C": 0.5},
        ),
        seed_averaging=True,
        seed_averaging_seeds=(42, 43),
    )

    result = run_training_pipeline(features, config, tmp_path)

    report = result.metrics["seed_averaging"]
    assert report["deterministic"] is True
    assert report["ensemble_size"] == 1
    assert report["seeds"] == [42]
    assert result.run_metadata["seed_averaging"]["ensemble_size"] == 1
    assert result.model.metadata["ensemble_size"] == 1


def test_no_seed_averaging_leaves_the_report_empty(tmp_path):
    features = pr_shaped_features()
    config = pr_15_plus_config(
        split_assignment=assign_pr_splits(features, seed=42, n_folds=3),
        seed=42,
        bootstrap_replicates=5,
        shap_sample_size=10,
        model_params={"lightgbm": {**LIGHTGBM_TEST_PARAMS, "n_estimators": 20}},
    )

    result = run_training_pipeline(features, config, tmp_path)

    assert result.metrics["seed_averaging"] is None
    assert result.run_metadata["seed_averaging"] is None
    assert result.model.metadata["ensemble_size"] == 1


def test_selected_bundle_and_pin_are_recorded_in_metadata(tmp_path):
    features = pr_shaped_features()
    config = replace(
        pr_15_plus_config(
            split_assignment=assign_pr_splits(features, seed=42, n_folds=3),
            seed=42,
            bootstrap_replicates=5,
            shap_sample_size=10,
            metadata={
                "selected_bundle": "measurement_no_bp",
                "comparison_artifact": "artifacts/comparison/measurement_no_bp",
            },
        ),
        n_folds=3,
        pinned_candidate=Candidate(
            label="logistic-winner",
            family="logistic",
            params={"C": 0.5},
        ),
    )

    result = run_training_pipeline(features, config, tmp_path)

    assert result.run_metadata["selected_bundle"] == "measurement_no_bp"
    assert result.model.metadata["selected_bundle"] == "measurement_no_bp"
    assert result.run_metadata["config"]["pinned_candidate"]["label"] == (
        "logistic-winner"
    )


def test_candidate_labels_must_be_unique_and_non_empty(tmp_path):
    features = pr_shaped_features()
    base = pr_15_plus_config(
        split_assignment=assign_pr_splits(features, seed=42),
        seed=42,
        bootstrap_replicates=5,
        shap_sample_size=10,
    )
    duplicate = replace(
        base,
        candidates=(
            Candidate(label="same-label", family="prevalence"),
            Candidate(label="same-label", family="logistic"),
        ),
    )
    with pytest.raises(ValueError, match="duplicate candidate labels: same-label"):
        run_training_pipeline(features, duplicate, tmp_path)

    empty = replace(
        base,
        pinned_candidate=Candidate(label="", family="prevalence"),
    )
    with pytest.raises(ValueError, match="candidate labels must not be empty"):
        run_training_pipeline(features, empty, tmp_path)
