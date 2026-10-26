"""Pure tests for the primary-versus-PR comparison helper and report."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from screener.feature_policy import PR_15_PLUS_FEATURES, PRIMARY_DEFAULT_FEATURES
from screener.pipeline import TrainingConfig, TrainingResult
from screener.pr_comparison import (
    PR_MODEL_CARD_ADDENDUM_NAME,
    RUN_KEYS,
    compare_primary_vs_pr,
    load_run_result,
    render_pr_model_card_addendum,
    render_primary_vs_pr_report,
    run_metric,
    write_primary_vs_pr_report,
)


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
        "weighted": {
            "n": 1_000,
            "positives": 111,
            "prevalence": 0.111,
            "roc_auc": 0.81,
            "pr_auc": 0.41,
            "brier": 0.09,
            "log_loss": 0.29,
            "sensitivity_at_specificity_0_8": 0.71,
            "sensitivity": 0.81,
            "specificity": 0.61,
            "ppv": 0.21,
            "npv": 0.97,
            "f1": 0.33,
            "calibration_slope": 1.01,
            "calibration_intercept": 0.01,
            "ece": 0.019,
            "threshold": 0.12,
        },
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
        "note": "",
    }


def _subgroups(include_older: bool = True, include_state: bool = True) -> pd.DataFrame:
    rows = [
        _subgroup_row("sex", "male", n=450, positives=90),
        _subgroup_row("sex", "female", n=550, positives=110),
        _subgroup_row("age_band", "15-24", n=250, positives=25),
        _subgroup_row("age_band", "25-34", n=250, positives=25),
        _subgroup_row("age_band", "35-44", n=250, positives=25),
        _subgroup_row("age_band", "45-49", n=250, positives=25),
    ]
    if include_older:
        rows.extend(
            [
                _subgroup_row("age_band", "50-59", n=200, positives=60),
                _subgroup_row("age_band", "60-69", n=100, positives=40),
                _subgroup_row("age_band", "70+", n=100, positives=40),
            ]
        )
    if include_state:
        rows.append(_subgroup_row("state", "1", n=600, positives=80))
    return pd.DataFrame(rows)


def _curve(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bin_lower": 0.0,
                "bin_upper": 0.5,
                "count": 400,
                "mean_predicted": 0.1 + offset,
                "observed": 0.09 + offset,
                "gap": 0.01,
            },
            {
                "bin_lower": 0.5,
                "bin_upper": 1.0,
                "count": 100,
                "mean_predicted": 0.7 + offset,
                "observed": 0.72 + offset,
                "gap": 0.02,
            },
        ]
    )


def _result(
    name: str,
    feature_columns,
    *,
    overall: dict,
    subgroups: pd.DataFrame | None = None,
    curve: pd.DataFrame | None = None,
    sample: str = "age_15_49",
    calibration: str = "isotonic",
) -> TrainingResult:
    config = TrainingConfig(
        name=name,
        feature_columns=tuple(feature_columns),
        metadata={"sample": sample},
    )
    return TrainingResult(
        config=config,
        split_assignment=pd.DataFrame(),
        model=None,
        metrics={
            "overall": overall,
            "calibration": {"selected": calibration},
            "threshold_selection": {"target_sensitivity": 0.8},
            "split": {
                "test": {"rows": 1_000, "female": 550, "male": 450, "female_share": 0.55}
            },
        },
        subgroup_metrics=_subgroups() if subgroups is None else subgroups,
        calibration_curve=_curve() if curve is None else curve,
        threshold_analysis=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        shap_summary=pd.DataFrame(),
        run_metadata={
            "run_id": name,
            "sample": sample,
            "selected_model": {"family": "lightgbm"},
            "row_counts": {
                "rows_used": 5_000,
                "rows_after_filters": 5_000,
                "test_rows": 1_000,
                "test_positives": 100,
            },
        },
        artifacts={},
        run_dir=Path("."),
    )


def _three_runs():
    primary = _result(
        "primary_15_49",
        PRIMARY_DEFAULT_FEATURES,
        overall=_overall(roc_auc=0.80, pr_auc=0.40, threshold=0.10),
        subgroups=_subgroups(include_older=False),
        sample="age_15_49",
    )
    pr_15_49 = _result(
        "pr_15to49_bp",
        PR_15_PLUS_FEATURES,
        overall=_overall(roc_auc=0.78, pr_auc=0.38, threshold=0.12),
        subgroups=_subgroups(include_older=False),
        curve=_curve(offset=0.01),
        sample="age_15_49",
        calibration="platt",
    )
    pr_15_49.metrics["overall"]["weighted"]["roc_auc"] = 0.79
    pr_15_plus = _result(
        "pr_15plus_bp",
        PR_15_PLUS_FEATURES,
        overall=_overall(roc_auc=0.82, pr_auc=0.42, threshold=0.14),
        curve=_curve(offset=0.02),
        sample="age_15_plus",
    )
    return primary, pr_15_49, pr_15_plus


def test_compare_primary_vs_pr_has_complete_tables_for_all_three_runs():
    comparison = compare_primary_vs_pr(*_three_runs())

    assert comparison["comparison_order"] == list(RUN_KEYS)
    assert list(comparison["runs"]) == list(RUN_KEYS)
    assert len(comparison["sample_table"]) == 3
    assert len(comparison["feature_table"]) == 3
    assert [row["feature_count"] for row in comparison["feature_table"]] == [32, 12, 12]
    assert comparison["runs"]["primary"]["feature_columns"] == list(
        PRIMARY_DEFAULT_FEATURES
    )
    assert comparison["runs"]["pr_15_plus"]["feature_columns"] == list(
        PR_15_PLUS_FEATURES
    )

    for key in RUN_KEYS:
        assert comparison["runs"][key]["metrics"]["overall"]["roc_auc"] is not None
        assert comparison["runs"][key]["metrics"]["weighted"]["prevalence"] is not None
        assert comparison["reliability_curves"][key]
        assert comparison["sex_metrics"][key]
        assert comparison["age_band_metrics"][key]

    assert comparison["subgroup_gaps"]["pr_15_49"]
    assert comparison["subgroup_gaps"]["pr_15_plus"]
    assert comparison["feature_differences"]["pr_only_features"] == []
    assert "bmi" in comparison["feature_differences"]["primary_only_features"]

    # Every value is JSON-safe.
    json.dumps(comparison, allow_nan=False)


def test_report_shows_calibrator_and_selected_family():
    comparison = compare_primary_vs_pr(*_three_runs())

    report = render_primary_vs_pr_report(comparison)

    assert "| Calibrator | isotonic | platt | isotonic |" in report
    assert "| Selected model | lightgbm | lightgbm | lightgbm |" in report


def test_compare_primary_vs_pr_computes_correct_differences():
    comparison = compare_primary_vs_pr(*_three_runs())

    pr_49 = comparison["deltas"]["pr_15_49_vs_primary"]
    pr_plus = comparison["deltas"]["pr_15_plus_vs_primary"]
    plus_vs_49 = comparison["deltas"]["pr_15_plus_vs_pr_15_49"]

    assert pr_49["roc_auc"] == pytest.approx(-0.02)
    assert pr_plus["roc_auc"] == pytest.approx(0.02)
    assert plus_vs_49["roc_auc"] == pytest.approx(0.04)
    assert pr_49["pr_auc"] == pytest.approx(-0.02)
    assert pr_plus["threshold"] == pytest.approx(0.04)
    assert comparison["weighted_deltas"]["pr_15_49_vs_primary"][
        "roc_auc"
    ] == pytest.approx(-0.02)
    assert comparison["delta_direction"]["pr_15_49_vs_primary"] == (
        "PR 15–49 minus primary"
    )


def test_run_metric_hides_the_nested_comparison_shape():
    comparison = compare_primary_vs_pr(*_three_runs())

    assert run_metric(comparison, "primary", "roc_auc") == pytest.approx(0.80)
    assert run_metric(
        comparison, "pr_15_49", "roc_auc", weighted=True
    ) == pytest.approx(0.79)
    assert run_metric(comparison, "pr_15_plus", "does_not_exist") is None


def test_missing_metric_and_blocks_are_reported_as_not_available():
    primary, pr_15_49, pr_15_plus = _three_runs()
    primary.metrics["overall"].pop("weighted")
    pr_15_49.metrics["overall"].pop("roc_auc")
    pr_15_49 = replace(
        pr_15_49,
        calibration_curve=pd.DataFrame(),
        subgroup_metrics=pd.DataFrame(),
    )

    comparison = compare_primary_vs_pr(primary, pr_15_49, pr_15_plus)
    report = render_primary_vs_pr_report(comparison)

    assert comparison["runs"]["pr_15_49"]["metrics"]["overall"]["roc_auc"] is None
    assert comparison["deltas"]["pr_15_49_vs_primary"]["roc_auc"] is None
    assert comparison["runs"]["primary"]["metrics"]["weighted"]["prevalence"] is None
    assert comparison["reliability_curves"]["pr_15_49"] == []
    assert comparison["sex_metrics"]["pr_15_49"] == []
    assert comparison["age_band_metrics"]["pr_15_49"] == []
    assert "n/a" in report
    assert "PR 15–49 | n/a" in report


def test_pr_only_features_are_the_union_of_both_pr_runs():
    primary, pr_15_49, pr_15_plus = _three_runs()
    pr_15_49 = _result(
        "pr_15to49_bp",
        PR_15_PLUS_FEATURES + ("state",),
        overall=_overall(),
        subgroups=_subgroups(include_older=False),
        sample="age_15_49",
    )

    comparison = compare_primary_vs_pr(primary, pr_15_49, pr_15_plus)

    assert comparison["feature_differences"]["pr_only_features"] == ["state"]
    assert "do not use the same feature list" in comparison["feature_differences"][
        "notes"
    ][-1]


def test_report_states_sample_and_feature_differences_without_declaring_a_winner():
    comparison = compare_primary_vs_pr(*_three_runs())
    report = render_primary_vs_pr_report(comparison)

    assert "# Primary vs PR sensitivity model comparison" in report
    assert "## Sample differences" in report
    assert "## Feature-set differences" in report
    assert "## Overall discrimination, calibration, and threshold metrics" in report
    assert "## Reliability curves" in report
    assert "## Sex-specific metrics" in report
    assert "## Age-band-specific metrics" in report
    assert "## Subgroup gaps for the PR models" in report
    assert "## Limitations" in report

    assert "32 features" in report
    assert "12 features" in report
    assert "isolates the feature-set trade-off" in report
    assert "adds older adults" in report
    assert "Raw probabilities are not directly comparable" in report
    assert "No run is universally better" in report
    assert "not a pure ablation" in report
    assert "not the deployed screener" in report
    assert "`bmi`" in report
    assert "Female n" in report
    assert "female share" in report.lower()


def test_model_card_addendum_covers_required_sections_and_statements():
    comparison = compare_primary_vs_pr(*_three_runs())

    addendum = render_pr_model_card_addendum(comparison)

    for heading in (
        "## Intended use",
        "## Sample and feature differences",
        "## Age gradient",
        "## Calibration",
        "## Threshold",
        "## Subgroup gaps",
        "## Label bias",
        "## Limitations",
        "## Non-goals",
        "## Publication",
    ):
        assert heading in addendum

    assert "not the deployed screener" in addendum
    assert "not a clinical diagnostic tool" in addendum
    assert "household head" in addendum
    assert "183" in addendum
    assert "transgender" in addendum
    assert "does not use anthropometry or diet" in addendum

    # Metrics and sample rows come from the run results, not placeholders.
    assert "0.8200" in addendum
    assert "0.7800" in addendum
    assert "0.1400" in addendum
    assert "0.1200" in addendum
    assert "5000" in addendum
    assert "1,808,013" not in addendum

    assert "split_assignment.parquet" not in addendum
    assert "shap_values.csv" not in addendum
    assert "model_bundle.joblib" not in addendum


def test_comparison_report_links_the_model_card_addendum():
    comparison = compare_primary_vs_pr(*_three_runs())

    report = render_primary_vs_pr_report(comparison)

    assert (
        f"[PR sensitivity model-card addendum]({PR_MODEL_CARD_ADDENDUM_NAME})"
        in report
    )


def test_write_primary_vs_pr_report_writes_json_markdown_and_addendum(tmp_path):
    comparison = compare_primary_vs_pr(*_three_runs())

    paths = write_primary_vs_pr_report(comparison, tmp_path, stem="comparison")

    assert paths["json"] == tmp_path / "comparison.json"
    assert paths["markdown"] == tmp_path / "comparison.md"
    assert paths["model_card_addendum"] == tmp_path / PR_MODEL_CARD_ADDENDUM_NAME
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["runs"]["primary"]["feature_count"] == 32
    report = paths["markdown"].read_text(encoding="utf-8")
    assert "## Limitations" in report
    assert f"({PR_MODEL_CARD_ADDENDUM_NAME})" in report
    addendum = paths["model_card_addendum"].read_text(encoding="utf-8")
    assert "# Model-card addendum" in addendum
    assert "not the deployed screener" in addendum
    assert "(comparison.md)" in addendum


def test_load_run_result_reads_the_written_artifacts(tmp_path):
    primary, _, _ = _three_runs()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "name": primary.config.name,
                "feature_columns": list(primary.config.feature_columns),
                "metadata": primary.config.metadata,
                "state_min_test_rows": 500,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(primary.run_metadata), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(primary.metrics, allow_nan=False), encoding="utf-8"
    )
    primary.subgroup_metrics.to_csv(run_dir / "subgroup_metrics.csv", index=False)
    primary.calibration_curve.to_csv(run_dir / "calibration_curve.csv", index=False)

    loaded = load_run_result(run_dir)

    assert loaded.config.name == primary.config.name
    assert loaded.config.feature_columns == primary.config.feature_columns
    assert loaded.run_metadata["run_id"] == primary.run_metadata["run_id"]
    assert loaded.metrics["overall"]["roc_auc"] == primary.metrics["overall"]["roc_auc"]
    assert len(loaded.subgroup_metrics) == len(primary.subgroup_metrics)
    assert len(loaded.calibration_curve) == len(primary.calibration_curve)
