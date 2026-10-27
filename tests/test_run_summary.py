"""Unit tests for the shared end-of-run summary.

The fake result is hand-built: these tests never train a model or touch the
filesystem.
"""

from pathlib import Path

import pandas as pd

from screener.pipeline import TrainingConfig, TrainingResult
from screener.run_summary import build_run_summary


def fake_run_result(
    *,
    row_counts: dict | None = None,
    run_metadata: dict | None = None,
    metrics: dict | None = None,
    run_dir: Path | None = None,
) -> TrainingResult:
    if row_counts is None:
        row_counts = {
            "rows_after_filters": 774_861,
            "rows_used": 5_000,
            "test_rows": 1_000,
        }
    if run_metadata is None:
        run_metadata = {
            "name": "pr_15plus_bp",
            "sample": "age_15_plus",
            "comparison_run": None,
            "split_source": "mixed",
            "selected_model": {"family": "lightgbm", "params": {}},
            "row_counts": row_counts,
        }
    if metrics is None:
        metrics = {
            "overall": {
                "roc_auc": 0.8,
                "pr_auc": 0.42,
                "threshold": 0.31,
                "sensitivity": 0.8,
                "specificity": 0.62,
            },
            "secondary_target": {"pr_auc": 0.55},
        }
    return TrainingResult(
        config=TrainingConfig(),
        split_assignment=pd.DataFrame(),
        model=None,
        metrics=metrics,
        subgroup_metrics=pd.DataFrame(),
        calibration_curve=pd.DataFrame(),
        threshold_analysis=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        shap_summary=pd.DataFrame(),
        run_metadata=run_metadata,
        artifacts={},
        run_dir=run_dir or Path("artifacts/pr/runs/run-1"),
    )


def test_summary_keeps_every_documented_field():
    result = fake_run_result()

    summary = build_run_summary(
        result,
        sample="15plus",
        smoke=True,
        reconciliation_status="pass",
        documented_rows=774_861,
        documented_positives=64_526,
    )

    assert summary == {
        "run_dir": str(Path("artifacts/pr/runs/run-1")),
        "run_name": "pr_15plus_bp",
        "sample": "15plus",
        "sample_label": "age_15_plus",
        "comparison_run": None,
        "smoke": True,
        "rows_after_filters": 774_861,
        "rows_used": 5_000,
        "test_rows": 1_000,
        "split_source": "mixed",
        "selected_model": "lightgbm",
        "reconciliation": "pass",
        "documented_rows": 774_861,
        "documented_positives": 64_526,
        "documented_match": True,
        "roc_auc": 0.8,
        "pr_auc": 0.42,
        "threshold": 0.31,
        "sensitivity": 0.8,
        "specificity": 0.62,
        "fact_sheet_pr_auc": 0.55,
    }


def test_summary_reports_the_comparison_run_and_missing_fact_sheet_target():
    row_counts = {"rows_after_filters": 770_000, "rows_used": 770_000, "test_rows": 900}
    result = fake_run_result(
        row_counts=row_counts,
        run_metadata={
            "name": "pr_15to49_bp",
            "sample": "age_15_49",
            "comparison_run": "pr_15plus_bp",
            "split_source": "reused_primary",
            "selected_model": {"family": "logistic", "params": {}},
            "row_counts": row_counts,
        },
        metrics={
            "overall": {
                "roc_auc": 0.75,
                "pr_auc": 0.38,
                "threshold": 0.29,
                "sensitivity": 0.81,
                "specificity": 0.6,
            }
        },
    )

    summary = build_run_summary(
        result,
        sample="15-49",
        smoke=False,
        reconciliation_status="review",
        documented_rows=774_861,
        documented_positives=64_526,
    )

    assert summary["sample"] == "15-49"
    assert summary["sample_label"] == "age_15_49"
    assert summary["comparison_run"] == "pr_15plus_bp"
    assert summary["selected_model"] == "logistic"
    assert summary["reconciliation"] == "review"
    assert summary["fact_sheet_pr_auc"] is None
    # The documented match compares the filtered row count, not the smoke sample.
    assert summary["documented_match"] is False


def test_summary_defaults_the_sample_to_the_recorded_label():
    result = fake_run_result()

    summary = build_run_summary(
        result,
        smoke=False,
        reconciliation_status="pass",
        documented_rows=774_861,
        documented_positives=64_526,
    )

    assert summary["sample"] == "age_15_plus"
    assert summary["sample_label"] == "age_15_plus"


def test_run_result_exposes_the_recorded_row_counts():
    row_counts = {"rows_after_filters": 10, "rows_used": 8, "test_rows": 2}
    result = fake_run_result(row_counts=row_counts)

    assert result.row_counts == row_counts


def test_summary_does_not_need_the_run_directory_to_exist():
    missing = Path("artifacts/never-written/run-404")
    result = fake_run_result(run_dir=missing)

    summary = build_run_summary(
        result,
        sample="15plus",
        smoke=False,
        reconciliation_status="pass",
        documented_rows=774_861,
        documented_positives=64_526,
    )

    assert summary["run_dir"] == str(missing)
