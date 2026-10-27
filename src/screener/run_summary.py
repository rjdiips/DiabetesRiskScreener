"""The end-of-run summary that every model entry point prints.

This module builds the mapping from a finished :class:`TrainingResult` and the
run-specific context the entry point already has. It does not train, read, or
write anything.
"""

from __future__ import annotations

from typing import Any

from .pipeline import TrainingResult


def build_run_summary(
    result: TrainingResult,
    *,
    smoke: bool,
    reconciliation_status: str,
    documented_rows: int | None,
    documented_positives: int | None,
    sample: str | None = None,
) -> dict[str, Any]:
    """Build the printed summary for one finished run.

    ``sample`` names the run the entry point was asked for; when omitted, the
    pipeline's recorded sample label is used. ``sample_label`` always comes
    from the pipeline metadata, so the two can differ (for example ``15plus``
    and ``age_15_plus``). The documented counts are the reference the run was
    reconciled against.
    """
    row_counts = result.row_counts
    overall = result.metrics["overall"]
    sample_label = result.run_metadata.get("sample")

    return {
        "run_dir": str(result.run_dir),
        "run_name": result.run_metadata["name"],
        "sample": sample_label if sample is None else sample,
        "sample_label": sample_label,
        "comparison_run": result.run_metadata.get("comparison_run"),
        "smoke": smoke,
        "rows_after_filters": row_counts["rows_after_filters"],
        "rows_used": row_counts["rows_used"],
        "test_rows": row_counts["test_rows"],
        "split_source": result.run_metadata["split_source"],
        "selected_model": result.run_metadata["selected_model"]["family"],
        "reconciliation": reconciliation_status,
        "documented_rows": documented_rows,
        "documented_positives": documented_positives,
        "documented_match": row_counts["rows_after_filters"] == documented_rows,
        "roc_auc": overall["roc_auc"],
        "pr_auc": overall["pr_auc"],
        "threshold": overall["threshold"],
        "sensitivity": overall["sensitivity"],
        "specificity": overall["specificity"],
        "fact_sheet_pr_auc": result.metrics.get("secondary_target", {}).get("pr_auc"),
    }


__all__ = ["build_run_summary"]
