"""PR 15+ and PR 15-49 model configs built on the shared training pipeline.

The PR model is a sensitivity analysis, not the deployed screener. It uses the
blood-pressure-based feature set so that every adult aged 15+ can be scored
without depending on anthropometry that the PR file measures only for younger
adults.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_policy import PR_15_PLUS_FEATURES, pr_feature_set
from .features import PR_REFERENCE_COUNTS
from .pipeline import SampleFilter, SubgroupSpec, TrainingConfig

PR_15_PLUS_AGE_BANDS = (15, 25, 35, 45, 50, 60, 70, np.inf)
PR_15_PLUS_AGE_LABELS = (
    "15-24",
    "25-34",
    "35-44",
    "45-49",
    "50-59",
    "60-69",
    "70+",
)
PR_15_49_AGE_BANDS = (15, 25, 35, 45, 50)
PR_15_49_AGE_LABELS = ("15-24", "25-34", "35-44", "45-49")

PR_CATEGORICAL_FEATURES = ("state", "religion", "social_group")

# Religion and social group are reported only by the variants that include them,
# because the default model's audit deliberately tests the exclusion decision.
PR_AUDIT_FEATURE_VARIANTS = ("fairness", "combined")
PR_AUDIT_SUBGROUP_NAMES = ("religion", "social_group")

PR_15_PLUS_RUN_NAME = "pr_15plus_bp"
PR_15_49_RUN_NAME = "pr_15to49_bp"

# The age-matched comparison filters on the sample flag built by the PR feature
# builder, not on a second age range, so the configured sample is exactly the
# documented one and cannot drift from `build_pr_features`.
PR_15_49_SAMPLE_FLAG = "age_15_49"

PR_15_PLUS_SAMPLE_LABEL = "age_15_plus"
PR_15_49_SAMPLE_LABEL = PR_15_49_SAMPLE_FLAG

PR_15_49_REFERENCE = {
    "bp_valid_age_15_49": PR_REFERENCE_COUNTS["bp_valid_age_15_49"],
    "bp_positive_age_15_49": PR_REFERENCE_COUNTS["bp_positive_age_15_49"],
}

_PR_HOUSEHOLD_HEAD_NOTE = "Describes the household head, not necessarily the respondent."

_PR_15_PLUS_ROLE = (
    "PR 15+ BP sensitivity model covering all adults aged 15+ with a valid "
    "blood-pressure pair."
)
_PR_15_49_ROLE = (
    f"Age-matched comparison to {PR_15_PLUS_RUN_NAME}: the same PR cluster "
    "partition and 12-feature BP set, filtered to adults aged 15-49. The sample "
    "filter and the age-band reporting differ; the feature set, partition, and "
    "modelling logic are identical, so the result isolates the effect of the "
    "smaller feature set."
)


def _pr_subgroups(
    age_bins,
    age_labels,
    feature_variant: str,
    state_min_test_rows: int,
) -> tuple[SubgroupSpec, ...]:
    subgroups = (
        SubgroupSpec(
            name="sex",
            column="female",
            value_labels=((0, "male"), (1, "female")),
        ),
        SubgroupSpec(
            name="age_band",
            column="age",
            bins=tuple(age_bins),
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
    if feature_variant not in PR_AUDIT_FEATURE_VARIANTS:
        return subgroups
    return subgroups + (
        SubgroupSpec(
            name="religion",
            column="religion",
            audit_only=True,
            note=_PR_HOUSEHOLD_HEAD_NOTE,
        ),
        SubgroupSpec(
            name="social_group",
            column="social_group",
            audit_only=True,
            note=_PR_HOUSEHOLD_HEAD_NOTE,
        ),
    )


def _variant_run_name(name: str, feature_variant: str) -> str:
    return name if feature_variant == "default" else f"{name}_{feature_variant}"


def _pr_config(
    *,
    name: str,
    feature_table: str | None,
    sample: str,
    run_role: str,
    comparison_run: str | None,
    sample_filter: SampleFilter | None,
    age_bins,
    age_labels,
    split_assignment: pd.DataFrame | None,
    split_assignment_path: str | None,
    seed: int,
    target_sensitivity: float,
    bootstrap_replicates: int,
    shap_sample_size: int,
    model_params: dict[str, Any] | None,
    smoke_rows: int | None,
    feature_variant: str,
    state_min_test_rows: int,
    metadata: dict | None,
) -> TrainingConfig:
    feature_columns = pr_feature_set(feature_variant)
    categorical = tuple(
        column for column in feature_columns if column in PR_CATEGORICAL_FEATURES
    )
    sample_filters = (
        SampleFilter(column="age", minimum=15),
        *((sample_filter,) if sample_filter is not None else ()),
        SampleFilter(column="systolic_mean", require_notna=True),
        SampleFilter(column="diastolic_mean", require_notna=True),
    )

    run_metadata = {
        "model_role": "sensitivity",
        "sample": sample,
        "run_role": run_role,
        "feature_variant": feature_variant,
        "audit_subgroup_note": _PR_HOUSEHOLD_HEAD_NOTE,
        **(metadata or {}),
    }
    if comparison_run is not None:
        run_metadata["comparison_run"] = _variant_run_name(
            comparison_run, feature_variant
        )

    return TrainingConfig(
        name=_variant_run_name(name, feature_variant),
        feature_table=feature_table,
        feature_columns=feature_columns,
        target_column="target",
        secondary_target_column="target_fact_sheet",
        target_definition="glucose > 140 OR diagnosed OR medication",
        group_column="cluster",
        household_column="household",
        weight_column="weight_scaled",
        sample_filters=sample_filters,
        subgroups=_pr_subgroups(
            age_bins,
            age_labels,
            feature_variant,
            state_min_test_rows,
        ),
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
        metadata=run_metadata,
    )


def pr_15_plus_config(
    *,
    feature_table: str | None = "data/processed/pr_features.parquet",
    split_assignment: pd.DataFrame | None = None,
    split_assignment_path: str | None = None,
    seed: int = 42,
    target_sensitivity: float = 0.8,
    bootstrap_replicates: int = 1_000,
    shap_sample_size: int = 5_000,
    model_params: dict[str, Any] | None = None,
    smoke_rows: int | None = None,
    feature_variant: str = "default",
    state_min_test_rows: int = 500,
    metadata: dict | None = None,
) -> TrainingConfig:
    """The large-sample PR 15+ BP-based sensitivity model."""
    return _pr_config(
        name=PR_15_PLUS_RUN_NAME,
        feature_table=feature_table,
        sample=PR_15_PLUS_SAMPLE_LABEL,
        run_role=_PR_15_PLUS_ROLE,
        comparison_run=None,
        sample_filter=None,
        age_bins=PR_15_PLUS_AGE_BANDS,
        age_labels=PR_15_PLUS_AGE_LABELS,
        split_assignment=split_assignment,
        split_assignment_path=split_assignment_path,
        seed=seed,
        target_sensitivity=target_sensitivity,
        bootstrap_replicates=bootstrap_replicates,
        shap_sample_size=shap_sample_size,
        model_params=model_params,
        smoke_rows=smoke_rows,
        feature_variant=feature_variant,
        state_min_test_rows=state_min_test_rows,
        metadata=metadata,
    )


def pr_15_49_config(
    *,
    feature_table: str | None = "data/processed/pr_features.parquet",
    split_assignment: pd.DataFrame | None = None,
    split_assignment_path: str | None = None,
    seed: int = 42,
    target_sensitivity: float = 0.8,
    bootstrap_replicates: int = 1_000,
    shap_sample_size: int = 5_000,
    model_params: dict[str, Any] | None = None,
    smoke_rows: int | None = None,
    feature_variant: str = "default",
    state_min_test_rows: int = 500,
    metadata: dict | None = None,
) -> TrainingConfig:
    """The age-matched PR 15-49 comparison on the same cluster partition.

    Supply the PR 15+ split assignment (or its path) so the run reuses the
    parent partition; the sample filter alone selects the younger rows.
    """
    return _pr_config(
        name=PR_15_49_RUN_NAME,
        feature_table=feature_table,
        sample=PR_15_49_SAMPLE_LABEL,
        run_role=_PR_15_49_ROLE,
        comparison_run=PR_15_PLUS_RUN_NAME,
        sample_filter=SampleFilter(column=PR_15_49_SAMPLE_FLAG, values=(True,)),
        age_bins=PR_15_49_AGE_BANDS,
        age_labels=PR_15_49_AGE_LABELS,
        split_assignment=split_assignment,
        split_assignment_path=split_assignment_path,
        seed=seed,
        target_sensitivity=target_sensitivity,
        bootstrap_replicates=bootstrap_replicates,
        shap_sample_size=shap_sample_size,
        model_params=model_params,
        smoke_rows=smoke_rows,
        feature_variant=feature_variant,
        state_min_test_rows=state_min_test_rows,
        metadata=metadata,
    )


def pr_sample_summary(
    features: pd.DataFrame,
    *,
    age_min: int = 15,
    age_max: int | None = None,
    sample_flag: str | None = None,
) -> dict:
    """Count the BP-valid PR sample used by the sensitivity model.

    ``sample_flag`` restricts the count with a boolean sample column, so the
    age-matched summary matches the configured ``age_15_49`` run filter.
    """
    required = ["age", "target", "systolic_mean", "diastolic_mean"]
    if sample_flag is not None:
        required.append(sample_flag)
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(
            "PR feature table is missing required columns: " + ", ".join(missing)
        )

    ages = pd.to_numeric(features["age"], errors="coerce")
    in_age = ages.ge(age_min)
    if age_max is not None:
        in_age &= ages.le(age_max)
    if sample_flag is not None:
        in_age &= features[sample_flag].fillna(False).astype(bool)
    bp_valid = features["systolic_mean"].notna() & features["diastolic_mean"].notna()
    subset = features.loc[in_age & bp_valid]

    rows = len(subset)
    positives = int(subset["target"].sum())
    return {
        "age_min": age_min,
        "age_max": age_max,
        "rows": rows,
        "positives": positives,
        "prevalence": round(positives / rows, 6) if rows else None,
        "bp_valid_rate": round(float((in_age & bp_valid).sum() / in_age.sum()), 6)
        if in_age.any()
        else None,
        "age_15_49_rows": len(subset.loc[subset["age"].between(15, 49)]),
        "age_15_49_positives": int(
            subset.loc[subset["age"].between(15, 49), "target"].sum()
        ),
        "age_50_plus_rows": len(subset.loc[subset["age"].ge(50)]),
        "age_50_plus_positives": int(
            subset.loc[subset["age"].ge(50), "target"].sum()
        ),
    }


def _reconciliation_report(summary: dict, observed: dict, reference: dict) -> dict:
    checks = {
        name: {
            "expected": reference[name],
            "observed": value,
            "pass": value == reference[name],
        }
        for name, value in observed.items()
    }
    return {
        "status": "pass" if all(check["pass"] for check in checks.values()) else "review",
        "summary": summary,
        "checks": checks,
    }


def reconcile_pr_bp_sample(
    features: pd.DataFrame,
    reference: dict | None = None,
) -> dict:
    """Compare the BP-valid PR sample with the documented reference counts."""
    summary = pr_sample_summary(features)
    observed = {
        "bp_valid_rows": summary["rows"],
        "bp_positive_rows": summary["positives"],
        "bp_valid_age_15_49": summary["age_15_49_rows"],
        "bp_positive_age_15_49": summary["age_15_49_positives"],
        "bp_valid_age_50_plus": summary["age_50_plus_rows"],
        "bp_positive_age_50_plus": summary["age_50_plus_positives"],
    }
    return _reconciliation_report(
        summary,
        observed,
        PR_REFERENCE_COUNTS if reference is None else reference,
    )


def reconcile_pr_15_49_sample(
    features: pd.DataFrame,
    reference: dict | None = None,
) -> dict:
    """Compare the BP-valid 15-49 sample with the documented reference counts.

    The sample comes from the ``age_15_49`` flag, the same filter the
    age-matched run configures, so the reconciliation cannot drift from the run.
    """
    summary = pr_sample_summary(features, sample_flag=PR_15_49_SAMPLE_FLAG)
    observed = {
        "bp_valid_age_15_49": summary["rows"],
        "bp_positive_age_15_49": summary["positives"],
    }
    return _reconciliation_report(
        summary,
        observed,
        PR_15_49_REFERENCE if reference is None else reference,
    )


__all__ = [
    "PR_15_49_AGE_BANDS",
    "PR_15_49_AGE_LABELS",
    "PR_15_49_REFERENCE",
    "PR_15_49_RUN_NAME",
    "PR_15_49_SAMPLE_FLAG",
    "PR_15_49_SAMPLE_LABEL",
    "PR_15_PLUS_AGE_BANDS",
    "PR_15_PLUS_AGE_LABELS",
    "PR_15_PLUS_FEATURES",
    "PR_15_PLUS_RUN_NAME",
    "PR_15_PLUS_SAMPLE_LABEL",
    "PR_AUDIT_FEATURE_VARIANTS",
    "PR_AUDIT_SUBGROUP_NAMES",
    "pr_15_49_config",
    "pr_15_plus_config",
    "pr_sample_summary",
    "reconcile_pr_15_49_sample",
    "reconcile_pr_bp_sample",
]
