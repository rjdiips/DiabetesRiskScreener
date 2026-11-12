"""The pre-registered promotion rule for post-freeze exploration.

The rule is pure: it takes already-computed out-of-fold metrics and returns a
decision, so a caller cannot quietly change the thresholds. A missing metric
fails the decision instead of raising, because an incomplete experiment must
never be promoted.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

PROMOTION_ROC_AUC_MARGIN = 0.005
PROMOTION_BRIER_MARGIN = 0.005
PROMOTION_SUBGROUP_MARGIN = 0.10


def _metric(metrics: dict, name: str) -> float | None:
    value = (metrics or {}).get(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _interval(bounds) -> tuple[float, float] | None:
    if bounds is None:
        return None
    try:
        low, high = (float(bounds[0]), float(bounds[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    return low, high


def promotion_decision(
    *,
    winner_metrics: dict,
    experiment_metrics: dict,
    delta_roc_auc_ci: tuple[float, float] | None,
    max_reliable_sensitivity_gap: float | None,
) -> dict:
    """Apply all four pre-registered promotion conditions."""
    winner_roc_auc = _metric(winner_metrics, "roc_auc")
    experiment_roc_auc = _metric(experiment_metrics, "roc_auc")
    winner_brier = _metric(winner_metrics, "brier")
    experiment_brier = _metric(experiment_metrics, "brier")

    delta_roc_auc = (
        None
        if winner_roc_auc is None or experiment_roc_auc is None
        else experiment_roc_auc - winner_roc_auc
    )
    delta_brier = (
        None
        if winner_brier is None or experiment_brier is None
        else experiment_brier - winner_brier
    )
    interval = _interval(delta_roc_auc_ci)
    gap = None
    try:
        gap = None if max_reliable_sensitivity_gap is None else float(
            max_reliable_sensitivity_gap
        )
    except (TypeError, ValueError):
        raise ValueError("max_reliable_sensitivity_gap must be numeric or None")

    conditions = {
        "roc_auc_margin": (
            delta_roc_auc is not None
            and delta_roc_auc >= PROMOTION_ROC_AUC_MARGIN
        ),
        "ci_excludes_zero": interval is not None and interval[0] > 0.0,
        "brier_within_margin": (
            delta_brier is not None and delta_brier <= PROMOTION_BRIER_MARGIN
        ),
        "subgroup_gap_ok": (
            gap is None
            or (np.isfinite(gap) and gap <= PROMOTION_SUBGROUP_MARGIN)
        ),
    }
    return {
        "promoted": all(conditions.values()),
        "conditions": conditions,
        "delta_roc_auc": delta_roc_auc,
        "delta_brier": delta_brier,
        "margins": {
            "roc_auc": PROMOTION_ROC_AUC_MARGIN,
            "brier": PROMOTION_BRIER_MARGIN,
            "subgroup_sensitivity": PROMOTION_SUBGROUP_MARGIN,
        },
    }


def paired_roc_auc_delta_ci(
    target,
    winner_probabilities,
    experiment_probabilities,
    clusters,
    *,
    replicates: int = 1000,
    seed: int = 42,
) -> tuple[float, float] | tuple[None, None]:
    """Cluster-bootstrap 95% CI for the paired ROC-AUC difference.

    Unique clusters are resampled with replacement and every row of a sampled
    cluster is kept, so repeated respondents stay together. Replicates in which
    either sample lacks both classes cannot produce a ROC-AUC and are skipped.
    """
    target = np.asarray(target)
    winner = np.asarray(winner_probabilities, dtype=float)
    experiment = np.asarray(experiment_probabilities, dtype=float)
    clusters = np.asarray(clusters)
    if not (len(target) == len(winner) == len(experiment) == len(clusters)):
        raise ValueError("target, probabilities, and clusters must align")
    if len(target) == 0:
        raise ValueError("paired bootstrap needs at least one row")

    unique = np.unique(clusters)
    positions = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(replicates)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([positions[cluster] for cluster in sampled])
        sample_target = target[index]
        if len(np.unique(sample_target)) < 2:
            continue
        deltas.append(
            roc_auc_score(sample_target, experiment[index])
            - roc_auc_score(sample_target, winner[index])
        )
    if not deltas:
        return None, None
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


__all__ = [
    "PROMOTION_BRIER_MARGIN",
    "PROMOTION_ROC_AUC_MARGIN",
    "PROMOTION_SUBGROUP_MARGIN",
    "paired_roc_auc_delta_ci",
    "promotion_decision",
]
