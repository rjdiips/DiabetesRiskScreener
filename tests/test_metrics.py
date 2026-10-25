"""Hand-computed tests for the screening metrics and reliability helpers."""

import math

import numpy as np

from screener.metrics import (
    binary_metrics,
    bootstrap_confidence_intervals,
    expected_calibration_error,
    reliability_curve,
    sensitivity_at_specificity,
)


def test_perfect_separation_has_ideal_discrimination_and_known_brier():
    metrics = binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
    )

    assert metrics["n"] == 4
    assert metrics["positives"] == 2
    assert metrics["prevalence"] == 0.5
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert math.isclose(metrics["brier"], 0.025, rel_tol=1e-12)
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["ppv"] == 1.0
    assert metrics["npv"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_single_class_target_reports_undefined_discrimination_not_a_crash():
    metrics = binary_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)

    assert metrics["n"] == 3
    assert metrics["positives"] == 0
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["calibration_slope"] is None
    assert math.isclose(metrics["brier"], (0.01 + 0.04 + 0.09) / 3, rel_tol=1e-12)
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == 1.0
    assert metrics["ppv"] is None
    assert metrics["npv"] == 1.0


def test_confusion_metrics_are_computed_at_the_given_threshold():
    metrics = binary_metrics(
        [0, 1, 1, 0],
        [0.2, 0.7, 0.6, 0.4],
        threshold=0.5,
    )

    assert metrics["confusion"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_sensitivity_at_fixed_specificity_respects_the_constraint():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_score = [0.05, 0.15, 0.25, 0.45, 0.35, 0.55, 0.75, 0.95]

    value = sensitivity_at_specificity(y_true, y_score, 0.75)

    assert 0.0 <= value <= 1.0
    assert value >= 0.75


def test_expected_calibration_error_matches_a_worked_two_bin_example():
    # Two equal-width bins: [0, 0.5) and [0.5, 1.0].
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.3, 0.7, 0.9]

    # Bin 1: mean predicted 0.2, observed 0.0 -> gap 0.2
    # Bin 2: mean predicted 0.8, observed 1.0 -> gap 0.2
    assert math.isclose(
        expected_calibration_error(y_true, y_score, n_bins=2),
        0.2,
        rel_tol=1e-12,
    )


def test_reliability_curve_covers_every_observation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.3, 0.7, 0.9]

    curve = reliability_curve(y_true, y_score, n_bins=2)

    assert [row["bin_lower"] for row in curve] == [0.0, 0.5]
    assert sum(row["count"] for row in curve) == 4
    assert math.isclose(curve[0]["mean_predicted"], 0.2, rel_tol=1e-12)
    assert math.isclose(curve[1]["observed"], 1.0, rel_tol=1e-12)


def test_weighted_metrics_report_weighted_prevalence():
    metrics = binary_metrics(
        [0, 0, 1],
        [0.1, 0.2, 0.9],
        sample_weight=[1, 1, 10],
        threshold=0.5,
    )

    assert metrics["positives"] == 1
    assert math.isclose(metrics["prevalence"], 10 / 12, rel_tol=1e-12)
    assert metrics["sensitivity"] == 1.0


def test_bootstrap_confidence_intervals_are_deterministic_and_bracketed():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=60)
    y_score = np.clip(0.4 * y_true + rng.normal(0.2, 0.15, size=60), 0, 1)
    clusters = np.repeat(np.arange(12), 5)

    first = bootstrap_confidence_intervals(
        y_true, y_score, clusters, replicates=25, seed=42
    )
    second = bootstrap_confidence_intervals(
        y_true, y_score, clusters, replicates=25, seed=42
    )

    assert first == second
    assert first["roc_auc"]["point"] == binary_metrics(y_true, y_score)["roc_auc"]
    assert first["roc_auc"]["low"] <= first["roc_auc"]["point"]
    assert first["roc_auc"]["high"] >= first["roc_auc"]["point"]
    assert 0 <= first["roc_auc"]["low"] <= 1
    assert 0 <= first["roc_auc"]["high"] <= 1
