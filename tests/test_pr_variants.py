"""Pure tests for the PR feature-set variant comparison."""

from pathlib import Path

import pandas as pd
import pytest

from screener.feature_policy import PR_15_PLUS_FEATURES
from screener.pipeline import TrainingConfig, TrainingResult
from screener.pr_variants import compare_pr_variants, render_pr_variant_comparison


def _overall(**overrides) -> dict:
    values = {
        "n": 1_000,
        "positives": 100,
        "prevalence": 0.1,
        "roc_auc": 0.8,
        "pr_auc": 0.4,
        "brier": 0.1,
        "log_loss": 0.3,
        "sensitivity_at_specificity_0_8": 0.7,
        "sensitivity": 0.8,
        "specificity": 0.6,
        "ppv": 0.2,
        "npv": 0.96,
        "f1": 0.32,
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "ece": 0.02,
        "threshold": 0.12,
    }
    values.update(overrides)
    return values


def _subgroup_row(
    subgroup: str,
    group: str,
    *,
    n: int = 500,
    positives: int = 100,
    reliable: bool = True,
    audit_only: bool = False,
    note: str = "",
) -> dict:
    return {
        "subgroup": subgroup,
        "group": group,
        "n": n,
        "positives": positives,
        "prevalence": positives / n,
        "roc_auc": 0.79,
        "pr_auc": 0.39,
        "sensitivity": 0.78,
        "specificity": 0.6,
        "calibration_error": 0.021,
        "gap_prevalence": 0.0,
        "gap_roc_auc": -0.01,
        "gap_pr_auc": -0.01,
        "gap_sensitivity": -0.02,
        "gap_calibration_error": 0.001,
        "reliable": reliable,
        "unreliable_reason": "" if reliable else "fewer than 200 respondents",
        "audit_only": audit_only,
        "note": note,
    }


def _subgroups(*extra) -> pd.DataFrame:
    rows = [
        _subgroup_row("sex", "male"),
        _subgroup_row("sex", "female"),
        _subgroup_row("age_band", "15-24"),
        _subgroup_row("age_band", "25-34"),
        _subgroup_row("age_band", "35-44"),
        _subgroup_row("age_band", "45-49"),
        _subgroup_row("age_band", "50-59"),
        _subgroup_row("age_band", "60-69"),
        _subgroup_row("age_band", "70+", n=100, positives=20, reliable=False),
    ]
    rows.extend(extra)
    return pd.DataFrame(rows)


def _result(
    name: str,
    feature_columns,
    *,
    overall: dict,
    subgroups: pd.DataFrame | None = None,
    feature_variant: str = "default",
) -> TrainingResult:
    config = TrainingConfig(
        name=name,
        feature_columns=tuple(feature_columns),
        metadata={"feature_variant": feature_variant},
    )
    return TrainingResult(
        config=config,
        split_assignment=pd.DataFrame(),
        model=None,
        metrics={
            "overall": overall,
            "calibration": {"selected": "isotonic"},
        },
        subgroup_metrics=_subgroups() if subgroups is None else subgroups,
        calibration_curve=pd.DataFrame(),
        threshold_analysis=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        shap_summary=pd.DataFrame(),
        run_metadata={
            "run_id": name,
            "feature_variant": feature_variant,
            "selected_model": {"family": "lightgbm"},
            "subgroup_rules": {
                "small_group_min_rows": 200,
                "small_group_min_positives": 50,
                "state_min_test_rows": 500,
            },
        },
        artifacts={},
        run_dir=Path("."),
    )


def test_compare_pr_variants_reports_deltas_features_and_mandatory_subgroups():
    default = _result("default", PR_15_PLUS_FEATURES, overall=_overall())
    state = _result(
        "state",
        PR_15_PLUS_FEATURES + ("state",),
        overall=_overall(roc_auc=0.81, pr_auc=0.41),
        feature_variant="state",
    )
    fairness = _result(
        "fairness",
        PR_15_PLUS_FEATURES + ("religion", "social_group"),
        overall=_overall(roc_auc=0.79),
        subgroups=_subgroups(
            _subgroup_row(
                "religion",
                "1",
                audit_only=True,
                note="Describes the household head, not necessarily the respondent.",
            ),
            _subgroup_row(
                "social_group",
                "1",
                audit_only=True,
                note="Describes the household head, not necessarily the respondent.",
            ),
        ),
        feature_variant="fairness",
    )

    comparison = compare_pr_variants(
        default,
        {"state": state, "fairness": fairness},
    )

    assert comparison["default"]["feature_columns"] == list(PR_15_PLUS_FEATURES)
    assert comparison["variants"]["state"]["added_features"] == ["state"]
    assert comparison["variants"]["fairness"]["added_features"] == [
        "religion",
        "social_group",
    ]
    assert comparison["variants"]["state"]["deltas"]["roc_auc"] == pytest.approx(0.01)
    assert comparison["variants"]["fairness"]["deltas"]["roc_auc"] == pytest.approx(-0.01)
    assert comparison["variants"]["state"]["metrics"]["roc_auc"] == 0.81
    assert comparison["variants"]["state"]["subgroups"]["sex"]
    assert comparison["variants"]["state"]["subgroups"]["age_band"]
    audit = comparison["variants"]["fairness"]["subgroups"]["audit"]
    assert {row["subgroup"] for row in audit} == {"religion", "social_group"}
    assert comparison["small_group_rules"] == {
        "small_group_min_rows": 200,
        "small_group_min_positives": 50,
        "state_min_test_rows": 500,
    }
    assert comparison["mandatory_subgroups"] == ["sex", "age_band"]


def test_compare_pr_variants_rejects_a_run_without_mandatory_subgroups():
    incomplete = _subgroups()
    incomplete = incomplete[incomplete["subgroup"].ne("age_band")]
    default = _result(
        "default",
        PR_15_PLUS_FEATURES,
        overall=_overall(),
        subgroups=incomplete,
    )

    with pytest.raises(ValueError, match="age_band"):
        compare_pr_variants(default, {})


def test_compare_pr_variants_rejects_a_fairness_variant_without_audit_rows():
    default = _result("default", PR_15_PLUS_FEATURES, overall=_overall())
    fairness = _result(
        "pr_15plus_bp_fairness",
        PR_15_PLUS_FEATURES + ("religion", "social_group"),
        overall=_overall(),
        subgroups=_subgroups(),
        feature_variant="fairness",
    )

    with pytest.raises(ValueError, match="religion"):
        compare_pr_variants(default, {"fairness": fairness})


def test_render_pr_variant_comparison_states_the_fairness_and_small_group_limits():
    default = _result("default", PR_15_PLUS_FEATURES, overall=_overall())
    fairness = _result(
        "fairness",
        PR_15_PLUS_FEATURES + ("religion", "social_group"),
        overall=_overall(roc_auc=0.79),
        subgroups=_subgroups(
            _subgroup_row(
                "religion",
                "1",
                audit_only=True,
                note="Describes the household head, not necessarily the respondent.",
            ),
            _subgroup_row(
                "social_group",
                "1",
                audit_only=True,
                note="Describes the household head, not necessarily the respondent.",
            ),
        ),
        feature_variant="fairness",
    )

    report = render_pr_variant_comparison(
        compare_pr_variants(default, {"fairness": fairness})
    )

    assert "# PR feature-set variant comparison" in report
    assert "## Feature sets" in report
    assert "## Overall metrics" in report
    assert "## Sex-specific metrics" in report
    assert "## Age-band-specific metrics" in report
    assert "Δ ROC-AUC" in report
    assert "Δ ECE" in report
    assert "fairness" in report
    assert "household head" in report
    assert "does not by itself make a model fair" in report
    assert "200 respondents" in report
    assert "50 positives" in report
    assert "not automatically preferable" in report
