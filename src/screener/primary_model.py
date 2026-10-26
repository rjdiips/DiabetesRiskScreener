"""Primary IR/MR 15-49 model config on the shared training pipeline.

The primary model is the deployed reference. It uses the rich IR/MR feature
set, restricted to adults aged 15-49, and is state-free and free of sensitive
attributes in its deployed features. State, religion, and social group remain
available for reporting and audit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .feature_policy import PRIMARY_DEFAULT_FEATURES
from .pipeline import SampleFilter, SubgroupSpec, TrainingConfig

PRIMARY_RUN_NAME = "primary_irmr_15to49"
PRIMARY_SAMPLE_LABEL = "age_15_49"

PRIMARY_AGE_BANDS = (15, 25, 35, 45, 50)
PRIMARY_AGE_LABELS = ("15-24", "25-34", "35-44", "45-49")

# The documented primary analysis sample: IR/MR recodes restricted to 15-49.
PRIMARY_REFERENCE = {
    "rows": 774_861,
    "positives": 64_526,
}

PRIMARY_CATEGORICAL_FEATURES = ("state", "religion", "social_group")

_PRIMARY_RUN_ROLE = (
    "Primary IR/MR 15-49 reference model with the rich anthropometry, diet, "
    "and BP feature set. This is the deployed screener; the PR runs are "
    "sensitivity analyses."
)

_AUDIT_SUBGROUP_NOTE = (
    "Reported for the fairness audit only; not a deployed model feature."
)


def _primary_subgroups(
    state_min_test_rows: int,
) -> tuple[SubgroupSpec, ...]:
    return (
        SubgroupSpec(
            name="sex",
            column="female",
            value_labels=((0, "male"), (1, "female")),
        ),
        SubgroupSpec(
            name="age_band",
            column="age",
            bins=PRIMARY_AGE_BANDS,
            labels=PRIMARY_AGE_LABELS,
        ),
        SubgroupSpec(name="wealth", column="wealth"),
        SubgroupSpec(
            name="residence",
            column="residence",
            value_labels=((1, "urban"), (2, "rural")),
        ),
        SubgroupSpec(name="education", column="education"),
        SubgroupSpec(
            name="state",
            column="state",
            minimum_test_rows=state_min_test_rows,
        ),
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


def primary_15_49_config(
    *,
    feature_table: str | None = "data/processed/irmr_features.parquet",
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
    """The primary IR/MR 15-49 model from the shared training pipeline."""
    feature_columns = PRIMARY_DEFAULT_FEATURES
    categorical = tuple(
        column
        for column in feature_columns
        if column in PRIMARY_CATEGORICAL_FEATURES
    )
    return TrainingConfig(
        name=PRIMARY_RUN_NAME,
        feature_table=feature_table,
        feature_columns=feature_columns,
        target_column="target",
        secondary_target_column="target_fact_sheet",
        target_definition="glucose > 140 OR diagnosed OR medication",
        group_column="cluster",
        household_column="household",
        weight_column="weight_scaled",
        sample_filters=(SampleFilter(column="age_15_49", values=(True,)),),
        subgroups=_primary_subgroups(state_min_test_rows),
        categorical_columns=categorical,
        seed=seed,
        n_folds=5,
        test_fraction=0.2,
        target_sensitivity=target_sensitivity,
        bootstrap_replicates=bootstrap_replicates,
        shap_sample_size=shap_sample_size,
        state_min_test_rows=state_min_test_rows,
        model_params=model_params or {},
        split_assignment=split_assignment,
        split_assignment_path=split_assignment_path,
        smoke_rows=smoke_rows,
        metadata={
            "model_role": "primary",
            "sample": PRIMARY_SAMPLE_LABEL,
            "run_role": _PRIMARY_RUN_ROLE,
            "feature_variant": "default",
            "audit_subgroup_note": _AUDIT_SUBGROUP_NOTE,
            **(metadata or {}),
        },
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
    checks = {
        name: {
            "expected": expected[name],
            "observed": summary[name],
            "pass": summary[name] == expected[name],
        }
        for name in ("rows", "positives")
    }
    return {
        "status": "pass" if all(check["pass"] for check in checks.values()) else "review",
        "summary": summary,
        "checks": checks,
    }


__all__ = [
    "PRIMARY_AGE_BANDS",
    "PRIMARY_AGE_LABELS",
    "PRIMARY_REFERENCE",
    "PRIMARY_RUN_NAME",
    "PRIMARY_SAMPLE_LABEL",
    "primary_15_49_config",
    "primary_sample_summary",
    "reconcile_primary_sample",
]
