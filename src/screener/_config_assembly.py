"""One assembly path for the primary and PR model configs.

Both model families share the pipeline shape below. A family supplies only its
run name, feature columns, sample filters, subgroups, and metadata, so a
pipeline setting added here reaches both families.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .pipeline import SampleFilter, SubgroupSpec, TrainingConfig

# The 15-49 bands the primary model and the PR 15-49 comparison both report.
AGE_15_49_BANDS = (15, 25, 35, 45, 50)
AGE_15_49_LABELS = ("15-24", "25-34", "35-44", "45-49")


def shared_subgroups(
    age_bands: tuple[float, ...],
    age_labels: tuple[str, ...],
    state_min_test_rows: int,
) -> tuple[SubgroupSpec, ...]:
    """The subgroup set both model families report."""
    return (
        SubgroupSpec(
            name="sex",
            column="female",
            value_labels=((0, "male"), (1, "female")),
        ),
        SubgroupSpec(
            name="age_band",
            column="age",
            bins=tuple(age_bands),
            labels=tuple(age_labels),
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
    )


def run_metadata(
    *,
    model_role: str,
    sample: str,
    run_role: str,
    audit_subgroup_note: str,
    feature_variant: str = "default",
    extra: dict | None = None,
) -> dict:
    """The run metadata both families record, in a fixed key order."""
    return {
        "model_role": model_role,
        "sample": sample,
        "run_role": run_role,
        "feature_variant": feature_variant,
        "audit_subgroup_note": audit_subgroup_note,
        **(extra or {}),
    }


def assemble_config(
    *,
    name: str,
    feature_table: str | None,
    feature_columns: tuple[str, ...],
    sample_filters: tuple[SampleFilter, ...],
    subgroups: tuple[SubgroupSpec, ...],
    metadata: dict,
    categorical_columns: tuple[str, ...] = (),
    seed: int,
    target_sensitivity: float,
    bootstrap_replicates: int,
    shap_sample_size: int,
    model_params: dict[str, Any] | None,
    split_assignment: pd.DataFrame | None,
    split_assignment_path: str | None,
    smoke_rows: int | None,
    state_min_test_rows: int,
) -> TrainingConfig:
    """Build a TrainingConfig with the pipeline shape both families share.

    Every pipeline setting is required, so the assembler never restates a
    ``TrainingConfig`` default that a family builder could leave behind.
    """
    return TrainingConfig(
        name=name,
        feature_table=feature_table,
        feature_columns=feature_columns,
        target_column="target",
        secondary_target_column="target_fact_sheet",
        target_definition="glucose > 140 OR diagnosed OR medication",
        group_column="cluster",
        household_column="household",
        weight_column="weight_scaled",
        sample_filters=sample_filters,
        subgroups=subgroups,
        categorical_columns=categorical_columns,
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
        metadata=metadata,
    )
