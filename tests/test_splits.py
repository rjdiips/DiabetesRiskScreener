"""Synthetic-data tests for the leakage-free cluster split assignment."""

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from screener.splits import (
    DEVELOPMENT_SPLIT,
    TEST_SPLIT,
    age_15_49_slice,
    assign_pr_splits,
    assign_splits,
    load_split_assignment,
    save_split_assignment,
    summarise_splits,
)


def _features(n_clusters: int = 40, members_per_household: int = 3) -> pd.DataFrame:
    """A PR-shaped frame with two households per cluster and mixed targets/sexes."""
    rows = []
    for cluster in range(1, n_clusters + 1):
        high_prevalence = cluster % 5 == 0
        for household_offset in range(2):
            household = cluster * 100 + household_offset
            for member in range(members_per_household):
                age = 18 + ((cluster * 3 + member * 7 + household_offset) % 60)
                female = int((cluster + household_offset + member) % 2 == 1)
                target = int(
                    (cluster + member) % 5 == 0
                    or (high_prevalence and member % 2 == 0)
                )
                rows.append(
                    {
                        "case_id": f"c{cluster}-h{household_offset}-m{member}",
                        "cluster": cluster,
                        "household": household,
                        "female": female,
                        "target": target,
                        "age": age,
                        "age_15_49": age <= 49,
                    }
                )
    return pd.DataFrame(rows)


def _primary_assignment(
    features: pd.DataFrame, clusters: list[int]
) -> pd.DataFrame:
    """Hand-built primary assignment for the selected clusters."""
    slots = {}
    development_index = 0
    for index, cluster in enumerate(clusters):
        if index % 5 == 0:
            slots[cluster] = (TEST_SPLIT, pd.NA)
        else:
            slots[cluster] = (DEVELOPMENT_SPLIT, (development_index % 5) + 1)
            development_index += 1

    primary = features[features["cluster"].isin(clusters)].copy()
    primary["split"] = primary["cluster"].map(lambda c: slots[c][0])
    primary["fold"] = primary["cluster"].map(lambda c: slots[c][1])
    return primary[["case_id", "cluster", "household", "split", "fold"]]


def _slot(frame: pd.DataFrame) -> pd.Series:
    fold = frame["fold"].astype("Int64").fillna(-1).astype(str)
    return frame["split"].astype(str) + ":" + fold


def test_assign_splits_assigns_every_respondent_to_one_slot():
    features = _features()
    assignment = assign_splits(features, seed=42)

    assert list(assignment.columns) == [
        "case_id",
        "cluster",
        "household",
        "split",
        "fold",
    ]
    assert len(assignment) == len(features)
    assert assignment["case_id"].is_unique
    assert set(assignment["split"]) == {TEST_SPLIT, DEVELOPMENT_SPLIT}

    test = assignment[assignment["split"] == TEST_SPLIT]
    development = assignment[assignment["split"] == DEVELOPMENT_SPLIT]
    assert test["fold"].isna().all()
    assert development["fold"].notna().all()
    assert sorted(development["fold"].astype(int).unique()) == [1, 2, 3, 4, 5]


def test_assign_splits_has_no_cluster_or_household_leakage():
    assignment = assign_splits(_features(), seed=42)
    assignment = assignment.assign(slot=_slot(assignment))

    assert assignment.groupby("cluster")["slot"].nunique().eq(1).all()
    assert assignment.groupby("household")["slot"].nunique().eq(1).all()


def test_assign_splits_is_deterministic_for_a_fixed_seed():
    features = _features()

    first = assign_splits(features, seed=42)
    second = assign_splits(features, seed=42)
    other_seed = assign_splits(features, seed=7)

    assert_frame_equal(first, second)
    assert not first[["cluster", "split", "fold"]].equals(
        other_seed[["cluster", "split", "fold"]]
    )


def test_assign_splits_holds_out_about_twenty_percent_and_balances_folds():
    features = _features(n_clusters=60)
    assignment = assign_splits(features, seed=42)

    test_rows = assignment["split"].eq(TEST_SPLIT).sum()
    assert 0.10 * len(assignment) <= test_rows <= 0.30 * len(assignment)

    folds = assignment[assignment["split"] == DEVELOPMENT_SPLIT].groupby("fold").size()
    assert len(folds) == 5
    assert folds.max() - folds.min() <= 0.2 * folds.mean()


def test_summary_reports_fold_prevalence_and_sex_composition():
    features = _features()
    assignment = assign_splits(features, seed=42)
    summary = summarise_splits(assignment, features)

    assert summary["rows"] == len(features)
    assert summary["clusters"] == features["cluster"].nunique()
    assert summary["households"] == features["household"].nunique()
    assert summary["test"]["rows"] == assignment["split"].eq(TEST_SPLIT).sum()
    assert (
        summary["development"]["rows"]
        == assignment["split"].eq(DEVELOPMENT_SPLIT).sum()
    )
    assert summary["test"]["rows"] + summary["development"]["rows"] == len(features)

    assert len(summary["folds"]) == 5
    assert [fold["fold"] for fold in summary["folds"]] == [1, 2, 3, 4, 5]
    for fold in summary["folds"]:
        assert fold["rows"] > 0
        assert fold["rows"] == fold["female"] + fold["male"]
        assert 0 <= fold["prevalence"] <= 1
        assert 0 <= fold["female_share"] <= 1
        assert fold["positives"] == round(fold["prevalence"] * fold["rows"])

    assert summary["leakage"] == {
        "cluster_single_slot": True,
        "household_single_slot": True,
    }
    assert summary["age_15_49"]["rows"] == int(features["age_15_49"].sum())


def test_pr_only_assignment_uses_the_same_rule_as_the_primary_split():
    features = _features()

    pr_assignment = assign_pr_splits(features, seed=42)
    primary_assignment = assign_splits(features, seed=42)

    assert_frame_equal(
        pr_assignment[["case_id", "cluster", "household", "split", "fold"]],
        primary_assignment,
    )


def test_primary_assignment_may_be_cluster_level_with_a_train_alias():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])
    primary = primary.drop_duplicates("cluster").copy()
    primary["split"] = primary["split"].replace(
        {DEVELOPMENT_SPLIT: "train"}
    )

    assignment = assign_pr_splits(features, primary, seed=42)
    primary_slots = primary.set_index("cluster")

    for cluster, rows in assignment[
        assignment["cluster"].isin(primary_slots.index)
    ].groupby("cluster"):
        assert rows["split"].iloc[0] == primary_slots.loc[cluster, "split"].replace(
            "train", DEVELOPMENT_SPLIT
        )
        if rows["split"].iloc[0] == DEVELOPMENT_SPLIT:
            assert rows["fold"].astype("Int64").eq(
                int(primary_slots.loc[cluster, "fold"])
            ).all()


def test_reused_clusters_keep_their_slot_when_the_seed_changes():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])

    first = assign_pr_splits(features, primary, seed=1)
    second = assign_pr_splits(features, primary, seed=2)

    columns = ["cluster", "split", "fold"]
    reused_first = first[first["assignment_source"].eq("reused_primary")]
    reused_second = second[second["assignment_source"].eq("reused_primary")]
    assert_frame_equal(
        reused_first[columns].reset_index(drop=True),
        reused_second[columns].reset_index(drop=True),
    )


def test_pr_split_reuses_the_primary_split_and_fold_for_overlapping_clusters():
    features = _features()
    overlapping = [1, 2, 3, 4, 5, 6, 7, 8]
    primary = _primary_assignment(features, overlapping)

    assignment = assign_pr_splits(features, primary, seed=42)
    primary_slots = primary.drop_duplicates("cluster").set_index("cluster")

    reused = assignment[assignment["cluster"].isin(overlapping)]
    assert reused["assignment_source"].eq("reused_primary").all()

    for cluster, rows in reused.groupby("cluster"):
        assert rows["split"].nunique() == 1
        assert rows["split"].iloc[0] == primary_slots.loc[cluster, "split"]
        if rows["split"].iloc[0] == DEVELOPMENT_SPLIT:
            assert rows["fold"].astype("Int64").eq(
                int(primary_slots.loc[cluster, "fold"])
            ).all()
        else:
            assert rows["fold"].isna().all()

    pr_only = assignment[~assignment["cluster"].isin(overlapping)]
    assert pr_only["assignment_source"].eq("pr_only").all()


def test_pr_split_source_is_mixed_when_only_some_clusters_overlap():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])

    assignment = assign_pr_splits(features, primary, seed=42)
    summary = summarise_splits(assignment, features)

    assert summary["split_source"] == "mixed"
    assert set(assignment["assignment_source"]) == {"reused_primary", "pr_only"}


def test_pr_split_source_is_reused_primary_when_every_cluster_overlaps():
    features = _features(n_clusters=10)
    primary = _primary_assignment(features, list(range(1, 11)))

    assignment = assign_pr_splits(features, primary, seed=42)
    summary = summarise_splits(assignment, features)

    assert summary["split_source"] == "reused_primary"
    assert assignment["assignment_source"].eq("reused_primary").all()


def test_pr_split_source_is_pr_only_without_a_primary_assignment():
    features = _features()

    assignment = assign_pr_splits(features)
    summary = summarise_splits(assignment, features)

    assert summary["split_source"] == "pr_only"
    assert assignment["assignment_source"].eq("pr_only").all()


def test_pr_only_clusters_are_assigned_deterministically():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])

    first = assign_pr_splits(features, primary, seed=42)
    second = assign_pr_splits(features, primary, seed=42)

    assert_frame_equal(first, second)
    first_slots = first.assign(slot=_slot(first))
    assert first_slots.groupby("cluster")["slot"].nunique().eq(1).all()
    assert first_slots.groupby("household")["slot"].nunique().eq(1).all()


def test_pr_split_accepts_zero_based_primary_folds():
    features = _features(n_clusters=10)
    primary = _primary_assignment(features, list(range(1, 11)))
    development_clusters = sorted(
        primary.loc[primary["split"].eq(DEVELOPMENT_SPLIT), "cluster"].unique()
    )
    for index, cluster in enumerate(development_clusters):
        primary.loc[primary["cluster"].eq(cluster), "fold"] = index % 5

    assignment = assign_pr_splits(features, primary, seed=42)

    for cluster, rows in assignment.groupby("cluster"):
        if rows["split"].iloc[0] == TEST_SPLIT:
            assert rows["fold"].isna().all()
            continue
        expected = int(
            primary.loc[primary["cluster"] == cluster, "fold"].dropna().iloc[0]
        )
        assert rows["fold"].astype("Int64").eq(expected + 1).all()


def test_age_15_49_assignment_is_the_same_partition_filtered():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])
    assignment = assign_pr_splits(features, primary, seed=42)

    subset = age_15_49_slice(assignment, features)
    expected_ids = set(features.loc[features["age_15_49"], "case_id"])

    assert set(subset["case_id"]) == expected_ids
    assert len(subset) == int(features["age_15_49"].sum())
    assert subset["case_id"].is_unique

    merged = subset.merge(assignment, on="case_id", suffixes=("_subset", "_full"))
    assert merged["split_subset"].eq(merged["split_full"]).all()
    assert merged["fold_subset"].astype("Int64").fillna(-1).eq(
        merged["fold_full"].astype("Int64").fillna(-1)
    ).all()


def test_pr_only_clusters_fill_the_remaining_holdout_quota():
    features = _features(n_clusters=60)
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])
    assignment = assign_pr_splits(features, primary, seed=42)

    test_share = assignment["split"].eq(TEST_SPLIT).mean()
    assert 0.15 <= test_share <= 0.25

    primary_test = primary.loc[
        primary["split"].eq(TEST_SPLIT), "cluster"
    ].unique()
    reused_test = assignment[assignment["cluster"].isin(primary_test)]
    assert reused_test["split"].eq(TEST_SPLIT).all()

    pr_only = assignment[assignment["assignment_source"].eq("pr_only")]
    assert pr_only["split"].eq(TEST_SPLIT).any()
    assert pr_only["split"].eq(DEVELOPMENT_SPLIT).any()


def test_degenerate_primary_partition_is_rejected():
    features = _features(n_clusters=10)
    primary = _primary_assignment(features, list(range(1, 11)))
    development = primary["split"].eq(DEVELOPMENT_SPLIT)
    primary.loc[development, "fold"] = 2

    with pytest.raises(ValueError, match="development folds are empty"):
        assign_pr_splits(features, primary)


def test_summary_rejects_a_partial_assignment():
    features = _features()
    assignment = assign_splits(features)

    with pytest.raises(ValueError, match="does not cover every respondent"):
        summarise_splits(assignment.iloc[:-1], features)


def test_summary_rejects_assignment_ids_outside_the_feature_table():
    features = _features()
    assignment = assign_splits(features)
    assignment.loc[assignment.index[0], "case_id"] = "unknown-id"

    with pytest.raises(ValueError, match="not in the feature table"):
        summarise_splits(assignment, features)


def test_summary_reports_15_49_folds_that_are_empty():
    features = _features()
    assignment = assign_splits(features, seed=42)

    fold_by_id = assignment[["case_id", "fold"]]
    fold_one_ids = set(
        fold_by_id.loc[fold_by_id["fold"].fillna(-1).eq(1), "case_id"]
    )
    features.loc[features["case_id"].isin(fold_one_ids), "age_15_49"] = False

    summary = summarise_splits(assignment, features)

    assert 1 in summary["age_15_49"]["missing_folds"]
    assert summary["age_15_49"]["rows"] == int(features["age_15_49"].sum())


def test_split_assignment_round_trips_through_the_artifact(tmp_path):
    features = _features()
    assignment = assign_pr_splits(features, seed=42)
    summary = summarise_splits(assignment, features)

    paths = save_split_assignment(
        assignment,
        tmp_path / "split_assignment.parquet",
        summary=summary,
    )

    loaded = load_split_assignment(paths["assignment"])
    assert_frame_equal(loaded, assignment)
    assert json.loads(paths["summary"].read_text(encoding="utf-8")) == summary


def test_household_identity_is_scoped_to_its_cluster():
    features = _features()
    # DHS household numbers repeat across clusters, so the same number in two
    # clusters is two households, not a leakage error.
    features.loc[features.index[0], "household"] = features.loc[
        features.index[-1], "household"
    ]

    assignment = assign_splits(features)
    summary = summarise_splits(assignment, features)

    assert len(assignment) == len(features)
    assert summary["households"] == features[["cluster", "household"]].drop_duplicates().shape[0]
    assert summary["leakage"]["household_single_slot"]


def test_assignment_rejects_missing_required_columns():
    features = _features()

    with pytest.raises(ValueError, match="target"):
        assign_splits(features.drop(columns=["target"]))


def test_assignment_rejects_a_missing_target():
    features = _features()
    features.loc[features.index[0], "target"] = None

    with pytest.raises(ValueError, match="target"):
        assign_splits(features)


def test_primary_assignment_rejects_conflicting_splits_for_a_cluster():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])
    cluster_two = primary["cluster"].eq(2)
    dev_row = primary[cluster_two & primary["split"].eq(DEVELOPMENT_SPLIT)].index[0]
    primary.loc[dev_row, "split"] = TEST_SPLIT
    primary.loc[dev_row, "fold"] = pd.NA

    with pytest.raises(ValueError, match="cluster"):
        assign_pr_splits(features, primary)


def test_primary_assignment_rejects_a_fold_outside_the_configured_folds():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])
    primary.loc[primary["cluster"].eq(2), "fold"] = 9

    with pytest.raises(ValueError, match="fold"):
        assign_pr_splits(features, primary)


def test_primary_assignment_rejects_missing_columns():
    features = _features()
    primary = _primary_assignment(features, [1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="fold"):
        assign_pr_splits(features, primary.drop(columns=["fold"]))
