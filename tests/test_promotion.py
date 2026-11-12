"""Unit tests for the pre-registered promotion rule and paired bootstrap."""

import numpy as np
import pytest

from screener.promotion import (
    PROMOTION_BRIER_MARGIN,
    PROMOTION_ROC_AUC_MARGIN,
    PROMOTION_SUBGROUP_MARGIN,
    paired_roc_auc_delta_ci,
    promotion_decision,
)


def metrics(*, roc_auc=0.70, brier=0.060):
    return {"roc_auc": roc_auc, "brier": brier}


def decide(**overrides):
    settings = {
        "winner_metrics": metrics(),
        "experiment_metrics": metrics(roc_auc=0.71, brier=0.061),
        "delta_roc_auc_ci": (0.002, 0.018),
        "max_reliable_sensitivity_gap": 0.05,
    }
    settings.update(overrides)
    return promotion_decision(**settings)


def test_all_four_conditions_promote():
    decision = decide()

    assert decision["promoted"] is True
    assert decision["conditions"] == {
        "roc_auc_margin": True,
        "ci_excludes_zero": True,
        "brier_within_margin": True,
        "subgroup_gap_ok": True,
    }
    assert decision["delta_roc_auc"] == pytest.approx(0.01)
    assert decision["delta_brier"] == pytest.approx(0.001)
    assert decision["margins"]["roc_auc"] == PROMOTION_ROC_AUC_MARGIN


def test_roc_auc_margin_fails_below_the_pre_registered_margin():
    decision = decide(
        experiment_metrics=metrics(
            roc_auc=0.70 + PROMOTION_ROC_AUC_MARGIN - 0.0001,
            brier=0.061,
        )
    )

    assert decision["conditions"]["roc_auc_margin"] is False
    assert decision["promoted"] is False


def test_ci_must_exclude_zero():
    assert decide(delta_roc_auc_ci=(-0.001, 0.02))["promoted"] is False
    assert decide(delta_roc_auc_ci=None)["conditions"]["ci_excludes_zero"] is False


def test_brier_must_stay_within_the_margin():
    decision = decide(
        experiment_metrics=metrics(
            roc_auc=0.71,
            brier=0.060 + PROMOTION_BRIER_MARGIN + 0.0001,
        )
    )

    assert decision["conditions"]["brier_within_margin"] is False
    assert decision["promoted"] is False


def test_a_reliable_subgroup_gap_over_the_margin_fails():
    decision = decide(
        max_reliable_sensitivity_gap=PROMOTION_SUBGROUP_MARGIN + 0.0001
    )

    assert decision["conditions"]["subgroup_gap_ok"] is False
    assert decision["promoted"] is False


def test_no_reliable_subgroup_gap_passes():
    assert decide(max_reliable_sensitivity_gap=None)["promoted"] is True
    assert decide(max_reliable_sensitivity_gap=0.10)["promoted"] is True


def test_missing_metrics_fail_instead_of_raising():
    decision = promotion_decision(
        winner_metrics={},
        experiment_metrics=metrics(roc_auc=0.71),
        delta_roc_auc_ci=(0.002, 0.018),
        max_reliable_sensitivity_gap=None,
    )

    assert decision["promoted"] is False
    assert decision["delta_roc_auc"] is None
    assert decision["delta_brier"] is None
    assert decision["conditions"]["roc_auc_margin"] is False
    assert decision["conditions"]["brier_within_margin"] is False


def test_paired_bootstrap_ci_brackets_zero_for_identical_predictions():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(20), 5)
    target = rng.integers(0, 2, len(clusters))
    probabilities = rng.random(len(clusters))

    low, high = paired_roc_auc_delta_ci(
        target,
        probabilities,
        probabilities.copy(),
        clusters,
        replicates=50,
        seed=42,
    )

    assert low == 0.0
    assert high == 0.0


def test_paired_bootstrap_ci_is_positive_for_a_clearly_better_experiment():
    rng = np.random.default_rng(4)
    clusters = np.repeat(np.arange(25), 4)
    target = np.tile([0, 0, 1, 1], 25)
    winner = rng.random(len(target))
    experiment = target + rng.normal(0.0, 0.01, len(target))

    first = paired_roc_auc_delta_ci(
        target, winner, experiment, clusters, replicates=50, seed=42
    )
    second = paired_roc_auc_delta_ci(
        target, winner, experiment, clusters, replicates=50, seed=42
    )

    assert first == second
    assert first[0] > 0.0
