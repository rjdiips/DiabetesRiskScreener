"""Classification, calibration, and bootstrap metrics for screening models.

Every metric is computed from calibrated probabilities. Undefined values are
reported as ``None`` rather than ``nan`` so that a single-class subgroup does
not poison a JSON artifact.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)

_EPSILON = 1e-6


def _clean(
    y_true,
    y_score,
    sample_weight=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if w is not None:
        mask &= np.isfinite(w) & (w >= 0)
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    if w is not None:
        w = w[mask]
    return y, p, w


def _value(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _rate(numerator: float, denominator: float) -> float | None:
    return _value(numerator / denominator) if denominator else None


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _calibration_slope_intercept(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None

    clipped = np.clip(p, _EPSILON, 1 - _EPSILON)
    logit = np.log(clipped) - np.log1p(-clipped)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)
    try:
        model.fit(logit.reshape(-1, 1), y, sample_weight=weights)
    except ValueError:
        return None, None
    return _value(model.coef_[0, 0]), _value(model.intercept_[0])


def _binned(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None,
) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        mask = index == bin_index
        count = int(mask.sum())
        row = {
            "bin_lower": round(float(edges[bin_index]), 6),
            "bin_upper": round(float(edges[bin_index + 1]), 6),
            "count": count,
        }
        if count:
            row["mean_predicted"] = round(_weighted_mean(p[mask], weights[mask] if weights is not None else None), 6)
            row["observed"] = round(_weighted_mean(y[mask], weights[mask] if weights is not None else None), 6)
            row["gap"] = round(abs(row["observed"] - row["mean_predicted"]), 6)
        else:
            row["mean_predicted"] = None
            row["observed"] = None
            row["gap"] = None
        rows.append(row)
    return rows


def expected_calibration_error(
    y_true,
    y_score,
    n_bins: int = 10,
    sample_weight=None,
) -> float | None:
    """Return the weighted mean absolute gap between predicted and observed risk."""
    y, p, w = _clean(y_true, y_score, sample_weight)
    if not len(y):
        return None

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = float(np.sum(w)) if w is not None else float(len(y))
    if total <= 0:
        return None

    error = 0.0
    for bin_index in range(n_bins):
        mask = index == bin_index
        if not mask.any():
            continue
        bin_weight = (
            float(np.sum(w[mask])) if w is not None else float(mask.sum())
        )
        bin_weights = w[mask] if w is not None else None
        predicted = _weighted_mean(p[mask], bin_weights)
        observed = _weighted_mean(y[mask], bin_weights)
        error += (bin_weight / total) * abs(observed - predicted)
    return round(error, 6)


def reliability_curve(
    y_true,
    y_score,
    n_bins: int = 10,
    sample_weight=None,
) -> list[dict]:
    """Return per-bin predicted versus observed risk."""
    y, p, w = _clean(y_true, y_score, sample_weight)
    return _binned(y, p, n_bins, w)


def sensitivity_at_specificity(
    y_true,
    y_score,
    specificity: float = 0.8,
    sample_weight=None,
) -> float | None:
    """Return the best sensitivity that keeps specificity at or above the floor."""
    y, p, w = _clean(y_true, y_score, sample_weight)
    if not len(y) or len(np.unique(y)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
    acceptable = (1.0 - fpr) >= specificity - 1e-12
    if not acceptable.any():
        return 0.0
    return _value(np.max(tpr[acceptable]))


def binary_metrics(
    y_true,
    y_score,
    *,
    sample_weight=None,
    threshold: float | None = None,
    specificity: float = 0.8,
    n_bins: int = 10,
) -> dict:
    """Compute discrimination, calibration, and operating-point metrics."""
    y, p, w = _clean(y_true, y_score, sample_weight)
    n = len(y)
    positives = int(y.sum())
    prevalence = _rate(positives, n)
    if w is not None:
        total_weight = float(np.sum(w))
        if total_weight > 0:
            prevalence = _rate(float(np.sum(w * y)), total_weight)
    predicted = (p >= threshold).astype(int) if threshold is not None else None

    confusion = None
    sensitivity = specificity_value = ppv = npv = f1 = None
    if predicted is not None:
        tp = int(np.sum((predicted == 1) & (y == 1)))
        fp = int(np.sum((predicted == 1) & (y == 0)))
        tn = int(np.sum((predicted == 0) & (y == 0)))
        fn = int(np.sum((predicted == 0) & (y == 1)))
        confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        sensitivity = _rate(tp, tp + fn)
        specificity_value = _rate(tn, tn + fp)
        ppv = _rate(tp, tp + fp)
        npv = _rate(tn, tn + fn)
        if ppv is not None and sensitivity:
            f1 = _value(2 * ppv * sensitivity / (ppv + sensitivity))
        elif ppv is not None:
            f1 = 0.0

        if w is not None:
            weighted_tp = float(np.sum(w[(predicted == 1) & (y == 1)]))
            weighted_fp = float(np.sum(w[(predicted == 1) & (y == 0)]))
            weighted_tn = float(np.sum(w[(predicted == 0) & (y == 0)]))
            weighted_fn = float(np.sum(w[(predicted == 0) & (y == 1)]))
            sensitivity = _rate(weighted_tp, weighted_tp + weighted_fn)
            specificity_value = _rate(weighted_tn, weighted_tn + weighted_fp)
            ppv = _rate(weighted_tp, weighted_tp + weighted_fp)
            npv = _rate(weighted_tn, weighted_tn + weighted_fn)
            if ppv is not None and sensitivity:
                f1 = _value(2 * ppv * sensitivity / (ppv + sensitivity))
            elif ppv is not None:
                f1 = 0.0

    if len(np.unique(y)) == 2:
        roc_auc = _value(roc_auc_score(y, p, sample_weight=w))
        pr_auc = _value(average_precision_score(y, p, sample_weight=w))
        log_loss = _value(sklearn_log_loss(y, np.clip(p, _EPSILON, 1 - _EPSILON), sample_weight=w, labels=[0, 1]))
        slope, intercept = _calibration_slope_intercept(y, p, w)
    else:
        roc_auc = None
        pr_auc = None
        log_loss = _value(sklearn_log_loss(y, np.clip(p, _EPSILON, 1 - _EPSILON), sample_weight=w, labels=[0, 1]))
        slope = intercept = None

    brier = _value(brier_score_loss(y, p, sample_weight=w))

    return {
        "n": n,
        "positives": positives,
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": log_loss,
        "sensitivity_at_specificity_0_8": sensitivity_at_specificity(y, p, specificity, w),
        "threshold": None if threshold is None else float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity_value,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "confusion": confusion,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece": expected_calibration_error(y, p, n_bins=n_bins, sample_weight=w),
    }


def _metric_function(name: str) -> Callable[[np.ndarray, np.ndarray], float | None]:
    if name == "roc_auc":
        return lambda y, p: _value(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    if name == "pr_auc":
        return lambda y, p: _value(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None
    if name == "brier":
        return lambda y, p: _value(brier_score_loss(y, p))
    raise ValueError(f"unsupported bootstrap metric: {name}")


def bootstrap_confidence_intervals(
    y_true,
    y_score,
    clusters,
    *,
    replicates: int = 1_000,
    seed: int = 42,
    metrics: tuple[str, ...] = ("roc_auc", "pr_auc", "brier"),
    alpha: float = 0.05,
) -> dict:
    """Cluster-resample the test set and report percentile intervals."""
    y, p, _ = _clean(y_true, y_score)
    y_all = np.asarray(y_true, dtype=float)
    p_all = np.asarray(y_score, dtype=float)
    keep = np.isfinite(y_all) & np.isfinite(p_all)
    cluster_values = np.asarray(clusters)[keep]
    unique = np.unique(cluster_values)
    index_by_cluster = {
        cluster: np.flatnonzero(cluster_values == cluster) for cluster in unique
    }
    rng = np.random.default_rng(seed)

    results: dict[str, dict] = {}
    for name in metrics:
        function = _metric_function(name)
        point = function(y, p)
        values = []
        for _ in range(replicates):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            index = np.concatenate([index_by_cluster[cluster] for cluster in sampled])
            value = function(y[index], p[index])
            if value is not None:
                values.append(value)
        if not values:
            low = high = None
        else:
            low = float(np.percentile(values, 100 * alpha / 2))
            high = float(np.percentile(values, 100 * (1 - alpha / 2)))
        results[name] = {"point": point, "low": low, "high": high}
    return results
