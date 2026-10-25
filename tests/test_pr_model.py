"""Pure tests for the PR sample counts and the BP-valid reconciliation."""

import pandas as pd
import pytest

from screener.pr_model import (
    PR_15_49_REFERENCE,
    pr_15_49_config,
    pr_15_plus_config,
    pr_sample_summary,
    reconcile_pr_15_49_sample,
    reconcile_pr_bp_sample,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": 20,
                "age_15_49": True,
                "systolic_mean": 110.0,
                "diastolic_mean": 70.0,
                "target": 0,
            },
            {
                "age": 30,
                "age_15_49": True,
                "systolic_mean": 130.0,
                "diastolic_mean": 85.0,
                "target": 1,
            },
            {
                "age": 50,
                "age_15_49": False,
                "systolic_mean": 150.0,
                "diastolic_mean": 95.0,
                "target": 1,
            },
            {
                "age": 60,
                "age_15_49": False,
                "systolic_mean": None,
                "diastolic_mean": None,
                "target": 1,
            },
            {
                "age": 10,
                "age_15_49": False,
                "systolic_mean": 100.0,
                "diastolic_mean": 65.0,
                "target": 0,
            },
        ]
    )


def test_pr_sample_summary_counts_the_bp_valid_adults():
    summary = pr_sample_summary(_sample())

    assert summary["rows"] == 3
    assert summary["positives"] == 2
    assert summary["prevalence"] == round(2 / 3, 6)
    assert summary["age_15_49_rows"] == 2
    assert summary["age_15_49_positives"] == 1
    assert summary["age_50_plus_rows"] == 1
    assert summary["age_50_plus_positives"] == 1
    assert summary["bp_valid_rate"] == 0.75


def test_pr_sample_summary_honours_the_age_ceiling():
    summary = pr_sample_summary(_sample(), age_max=49)

    assert summary["rows"] == 2
    assert summary["positives"] == 1


def test_reconciliation_passes_when_counts_match_the_reference():
    reference = {
        "bp_valid_rows": 3,
        "bp_positive_rows": 2,
        "bp_valid_age_15_49": 2,
        "bp_positive_age_15_49": 1,
        "bp_valid_age_50_plus": 1,
        "bp_positive_age_50_plus": 1,
    }

    report = reconcile_pr_bp_sample(_sample(), reference)

    assert report["status"] == "pass"
    assert all(check["pass"] for check in report["checks"].values())


def test_reconciliation_flags_a_mismatch_instead_of_silently_passing():
    reference = {
        "bp_valid_rows": 1_805_697,
        "bp_positive_rows": 253_091,
        "bp_valid_age_15_49": 1_290_219,
        "bp_positive_age_15_49": 112_898,
        "bp_valid_age_50_plus": 515_478,
        "bp_positive_age_50_plus": 140_193,
    }

    report = reconcile_pr_bp_sample(_sample(), reference)

    assert report["status"] == "review"
    assert report["checks"]["bp_valid_rows"]["expected"] == 1_805_697
    assert report["checks"]["bp_valid_rows"]["observed"] == 3


def test_pr_configs_share_the_same_partition_and_feature_set():
    plus = pr_15_plus_config()
    younger = pr_15_49_config()

    assert plus.feature_columns == younger.feature_columns
    assert plus.group_column == younger.group_column == "cluster"
    assert plus.household_column == younger.household_column == "household"
    assert plus.weight_column == younger.weight_column == "weight_scaled"
    assert plus.secondary_target_column == younger.secondary_target_column
    assert plus.split_assignment_path == younger.split_assignment_path


def test_pr_15_49_sample_summary_uses_the_flag_not_the_age_range():
    # The flag disagrees with the raw age in both rows, so only a flag-based
    # filter can produce the expected counts.
    frame = pd.DataFrame(
        [
            {
                "age": 30,
                "age_15_49": False,
                "systolic_mean": 120.0,
                "diastolic_mean": 80.0,
                "target": 1,
            },
            {
                "age": 55,
                "age_15_49": True,
                "systolic_mean": 130.0,
                "diastolic_mean": 85.0,
                "target": 0,
            },
        ]
    )

    summary = pr_sample_summary(frame, sample_flag="age_15_49")

    assert summary["rows"] == 1
    assert summary["positives"] == 0


def test_pr_15_49_reconciliation_uses_the_age_15_49_flag():
    reference = {
        "bp_valid_age_15_49": 2,
        "bp_positive_age_15_49": 1,
    }

    report = reconcile_pr_15_49_sample(_sample(), reference)

    assert report["status"] == "pass"
    assert set(report["checks"]) == {"bp_valid_age_15_49", "bp_positive_age_15_49"}
    assert report["summary"]["rows"] == 2
    assert report["summary"]["positives"] == 1


def test_pr_15_49_reconciliation_uses_the_documented_reference_counts():
    assert PR_15_49_REFERENCE == {
        "bp_valid_age_15_49": 1_290_219,
        "bp_positive_age_15_49": 112_898,
    }


def test_pr_15_49_reconciliation_requires_the_flag_column():
    frame = _sample().drop(columns=["age_15_49"])

    with pytest.raises(ValueError, match="age_15_49"):
        reconcile_pr_15_49_sample(frame)


def test_pr_configs_record_distinct_run_roles():
    plus = pr_15_plus_config()
    younger = pr_15_49_config()

    assert plus.metadata["sample"] == "age_15_plus"
    assert "comparison_run" not in plus.metadata
    assert younger.metadata["sample"] == "age_15_49"
    assert younger.metadata["comparison_run"] == "pr_15plus_bp"
    assert plus.metadata["run_role"] != younger.metadata["run_role"]
    assert plus.name != younger.name
