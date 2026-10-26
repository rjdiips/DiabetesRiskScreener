"""Pure tests for the primary IR/MR 15-49 config and sample reconciliation."""

import pandas as pd

from screener.feature_policy import PRIMARY_DEFAULT_FEATURES
from screener.primary_model import (
    PRIMARY_REFERENCE,
    primary_15_49_config,
    primary_sample_summary,
    reconcile_primary_sample,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"age": 20, "age_15_49": True, "target": 0},
            {"age": 30, "age_15_49": True, "target": 1},
            {"age": 50, "age_15_49": False, "target": 1},
        ]
    )


def test_primary_config_uses_the_32_feature_default_and_the_15_49_flag():
    config = primary_15_49_config()
    resolved = config.to_dict()

    assert config.feature_columns == PRIMARY_DEFAULT_FEATURES
    assert len(config.feature_columns) == 32
    assert config.sample_filters[0].column == "age_15_49"
    assert config.sample_filters[0].values == (True,)
    assert resolved["target_column"] == "target"
    assert resolved["secondary_target_column"] == "target_fact_sheet"
    assert resolved["group_column"] == "cluster"
    assert resolved["household_column"] == "household"
    assert resolved["weight_column"] == "weight_scaled"


def test_primary_config_reports_the_mandatory_and_audit_subgroups():
    config = primary_15_49_config()
    names = tuple(spec.name for spec in config.subgroups)

    assert names == (
        "sex",
        "age_band",
        "wealth",
        "residence",
        "education",
        "state",
        "religion",
        "social_group",
    )
    state = next(spec for spec in config.subgroups if spec.name == "state")
    assert state.minimum_test_rows == 500
    audit = [spec.name for spec in config.subgroups if spec.audit_only]
    assert audit == ["religion", "social_group"]


def test_primary_config_records_the_run_role_and_default_feature_variant():
    config = primary_15_49_config()

    assert config.metadata["model_role"] == "primary"
    assert config.metadata["sample"] == "age_15_49"
    assert config.metadata["feature_variant"] == "default"
    assert config.name == "primary_irmr_15to49"


def test_primary_sample_summary_and_reconciliation():
    summary = primary_sample_summary(_sample())
    assert summary["rows"] == 2
    assert summary["positives"] == 1

    report = reconcile_primary_sample(
        _sample(),
        {"rows": 2, "positives": 1},
    )
    assert report["status"] == "pass"
    assert set(report["checks"]) == {"rows", "positives"}


def test_primary_reconciliation_uses_the_documented_reference_counts():
    assert PRIMARY_REFERENCE == {"rows": 774_861, "positives": 64_526}
