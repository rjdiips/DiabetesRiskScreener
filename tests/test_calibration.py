"""Tests for probability calibration and sensitivity-first threshold selection."""

import numpy as np

from screener.calibration import (
    fit_calibrator,
    select_calibrator,
    select_threshold,
    threshold_analysis,
)


def test_isotonic_calibrator_is_monotonic_and_bounded():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]

    calibrator = fit_calibrator(y_true, y_score, "isotonic")
    predictions = calibrator.predict(np.array([0.05, 0.2, 0.5, 0.7, 0.95]))

    assert np.all((predictions >= 0) & (predictions <= 1))
    assert np.all(np.diff(predictions) >= -1e-12)


def test_platt_calibrator_returns_probabilities():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]

    calibrator = fit_calibrator(y_true, y_score, "platt")
    predictions = calibrator.predict(np.array([0.05, 0.2, 0.5, 0.7, 0.95]))

    assert np.all((predictions >= 0) & (predictions <= 1))


def test_calibrator_selection_improves_brier_on_an_overconfident_score():
    # Intentionally overconfident: true probability is about the score / 2.
    rng = np.random.default_rng(7)
    y_score = rng.uniform(0, 1, size=500)
    y_true = rng.binomial(1, y_score / 2)

    calibrator, report = select_calibrator(y_true, y_score)

    assert report["selected"] in {"isotonic", "platt"}
    assert report["selected_brier"] <= report["uncalibrated_brier"]
    assert report["method_reports"][report["selected"]]["brier"] == report["selected_brier"]
    assert calibrator is not None


def test_threshold_selection_meets_the_sensitivity_target_and_maximises_specificity():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.05, 0.15, 0.25, 0.45, 0.35, 0.55, 0.75, 0.95])

    selection = select_threshold(y_true, y_score, target_sensitivity=0.75)

    assert selection["sensitivity"] >= 0.75
    # Threshold 0.55 keeps three of four positives and predicts negative for
    # every negative, so it is the max-specificity point meeting the target.
    assert selection["specificity"] == 1.0
    assert selection["threshold"] == 0.55


def test_threshold_analysis_includes_the_chosen_operating_point():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]

    table = threshold_analysis(y_true, y_score, target_sensitivity=0.8, max_points=20)

    assert len(table) >= 1
    assert set(table.columns) >= {
        "threshold",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
    }
    assert table["sensitivity"].max() >= 0.8
