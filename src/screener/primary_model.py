"""Primary IR/MR 15-49 model config on the shared training pipeline.

The primary model is the deployed reference. It uses the rich IR/MR feature
set, restricted to adults aged 15-49, and is state-free and free of sensitive
attributes in its deployed features. State, religion, and social group remain
available for reporting and audit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._config_assembly import (
    AGE_15_49_BANDS,
    AGE_15_49_LABELS,
    assemble_config,
    config_metadata,
    shared_subgroups,
)
from .feature_policy import primary_feature_set
from .features import _compare_counts, _sample_reference
from .pipeline import SampleFilter, SubgroupSpec, TrainingConfig

PRIMARY_RUN_NAME = "primary_irmr_15to49"
PRIMARY_SAMPLE_LABEL = "age_15_49"

# The documented primary analysis sample: IR/MR recodes restricted to 15-49.
PRIMARY_REFERENCE = {
    "rows": 774_861,
    "positives": 64_526,
}

_PRIMARY_RUN_ROLE = (
    "Primary IR/MR 15-49 reference model with the rich anthropometry, diet, "
    "and BP feature set. This is the deployed screener; the PR runs are "
    "sensitivity analyses."
)

_AUDIT_SUBGROUP_NOTE = (
    "Reported for the fairness audit only; not a deployed model feature."
)


def primary_15_49_config(
    *,
    feature_table: str | None = "data/processed/irmr_features.parquet",
    feature_set: str = "measurement_full",
    split_assignment: pd.DataFrame | None = None,
    split_assignment_path: str | None = None,
    seed: int = 42,
    target_sensitivity: float = 0.8,
    bootstrap_replicates: int = 1_000,
    shap_sample_size: int = 5_000,
    model_params: dict[str, Any] | None = None,
    smoke_rows: int | None = None,
    state_min_test_rows: int = 500,
    metadata: dict | None = None,
) -> TrainingConfig:
    """The primary IR/MR 15-49 model from the shared training pipeline.

    ``feature_set`` names any set in ``PRIMARY_FEATURE_SETS``; the default is
    the full measurement bundle.
    """
    subgroups = shared_subgroups(
        AGE_15_49_BANDS,
        AGE_15_49_LABELS,
        state_min_test_rows,
    ) + (
        SubgroupSpec(
            name="religion",
            column="religion",
            audit_only=True,
            note=_AUDIT_SUBGROUP_NOTE,
        ),
        SubgroupSpec(
            name="social_group",
            column="social_group",
            audit_only=True,
            note=_AUDIT_SUBGROUP_NOTE,
        ),
    )
    return assemble_config(
        name=PRIMARY_RUN_NAME,
        feature_table=feature_table,
        feature_columns=primary_feature_set(feature_set),
        sample_filters=(SampleFilter(column="age_15_49", values=(True,)),),
        subgroups=subgroups,
        metadata=config_metadata(
            model_role="primary",
            sample=PRIMARY_SAMPLE_LABEL,
            run_role=_PRIMARY_RUN_ROLE,
            audit_subgroup_note=_AUDIT_SUBGROUP_NOTE,
            feature_variant=(
                "default" if feature_set == "measurement_full" else feature_set
            ),
            extra=metadata,
        ),
        seed=seed,
        target_sensitivity=target_sensitivity,
        bootstrap_replicates=bootstrap_replicates,
        shap_sample_size=shap_sample_size,
        model_params=model_params,
        split_assignment=split_assignment,
        split_assignment_path=split_assignment_path,
        smoke_rows=smoke_rows,
        state_min_test_rows=state_min_test_rows,
    )


def primary_sample_summary(features: pd.DataFrame) -> dict:
    """Count the configured primary sample and its positives."""
    required = ["age_15_49", "target"]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(
            "primary feature table is missing required columns: "
            + ", ".join(missing)
        )
    mask = features["age_15_49"].fillna(False).astype(bool)
    subset = features.loc[mask]
    rows = len(subset)
    positives = int(subset["target"].sum())
    return {
        "rows": rows,
        "positives": positives,
        "prevalence": round(positives / rows, 6) if rows else None,
    }


def reconcile_primary_sample(
    features: pd.DataFrame,
    reference: dict | None = None,
) -> dict:
    """Compare the primary sample counts with the documented reference."""
    summary = primary_sample_summary(features)
    expected = PRIMARY_REFERENCE if reference is None else reference
    observed = {"rows": summary["rows"], "positives": summary["positives"]}
    return _compare_counts(
        observed,
        _sample_reference(expected, observed),
        summary,
    )


__all__ = [
    "PRIMARY_REFERENCE",
    "PRIMARY_RUN_NAME",
    "PRIMARY_SAMPLE_LABEL",
    "primary_15_49_config",
    "primary_sample_summary",
    "reconcile_primary_sample",
]
