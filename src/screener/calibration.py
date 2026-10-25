"""Probability calibration and sensitivity-first threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .metrics import binary_metrics

_EPSILON = 1e-6


def _clean(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], np.clip(p[mask], 0.0, 1.0)


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, _EPSILON, 1 - _EPSILON)
    return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)


class IdentityCalibrator:
    """A transparent fallback when there is not enough signal to fit a calibrator."""

    method = "identity"

    def predict(self, scores) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


class IsotonicCalibrator:
    method = "isotonic"

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        self.model.fit(scores, y_true)
        return self

    def predict(self, scores) -> np.ndarray:
        return np.clip(self.model.predict(np.asarray(scores, dtype=float)), 0.0, 1.0)


class PlattCalibrator:
    method = "platt"

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        self.model.fit(_logit(scores), y_true)
        return self

    def predict(self, scores) -> np.ndarray:
        return np.clip(
            self.model.predict_proba(_logit(np.asarray(scores, dtype=float)))[:, 1],
            0.0,
            1.0,
        )


_CALIBRATORS = {
    "isotonic": IsotonicCalibrator,
    "platt": PlattCalibrator,
}


def fit_calibrator(y_true, y_score, method: str):
    """Fit one named calibrator, falling back to identity for degenerate data."""
    if method not in _CALIBRATORS:
        raise ValueError(
            f"unknown calibration method '{method}'; expected one of "
            + ", ".join(sorted(_CALIBRATORS))
        )

    y, p = _clean(y_true, y_score)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return IdentityCalibrator()
    return _CALIBRATORS[method]().fit(p, y)


def _calibrator_scores(calibrator, y: np.ndarray, p: np.ndarray) -> dict:
    calibrated = np.clip(calibrator.predict(p), 0.0, 1.0)
    return {
        "brier": float(brier_score_loss(y, calibrated)),
        "log_loss": float(
            log_loss(y, np.clip(calibrated, _EPSILON, 1 - _EPSILON), labels=[0, 1])
        ),
    }


def select_calibrator(
    y_true,
    y_score,
    methods: tuple[str, ...] = ("isotonic", "platt"),
) -> tuple[object, dict]:
    """Fit every candidate calibrator and keep the best OOF Brier score."""
    y, p = _clean(y_true, y_score)
    uncalibrated = {
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(
            log_loss(y, np.clip(p, _EPSILON, 1 - _EPSILON), labels=[0, 1])
        )
        if len(y)
        else None,
    }

    reports: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for method in methods:
        calibrator = fit_calibrator(y, p, method)
        reports[method] = _calibrator_scores(calibrator, y, p)
        fitted[method] = calibrator

    if not fitted:
        return IdentityCalibrator(), {
            "selected": "identity",
            "uncalibrated_brier": uncalibrated["brier"],
            "uncalibrated_log_loss": uncalibrated["log_loss"],
            "selected_brier": uncalibrated["brier"],
            "selected_log_loss": uncalibrated["log_loss"],
            "method_reports": {},
        }

    selected = min(
        fitted,
        key=lambda name: (
            reports[name]["brier"],
            reports[name]["log_loss"],
        ),
    )
    return fitted[selected], {
        "selected": selected,
        "uncalibrated_brier": uncalibrated["brier"],
        "uncalibrated_log_loss": uncalibrated["log_loss"],
        "selected_brier": reports[selected]["brier"],
        "selected_log_loss": reports[selected]["log_loss"],
        "method_reports": reports,
    }


def _confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    predicted = p >= threshold
    positive = y == 1
    return {
        "tp": int(np.sum(predicted & positive)),
        "fp": int(np.sum(predicted & ~positive)),
        "tn": int(np.sum(~predicted & ~positive)),
        "fn": int(np.sum(~predicted & positive)),
    }


def _rates(confusion: dict) -> dict:
    tp, fp, tn, fn = (
        confusion["tp"],
        confusion["fp"],
        confusion["tn"],
        confusion["fn"],
    )

    def rate(numerator: float, denominator: float) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    sensitivity = rate(tp, tp + fn)
    specificity = rate(tn, tn + fp)
    ppv = rate(tp, tp + fp)
    npv = rate(tn, tn + fn)
    f1 = (
        round(2 * ppv * sensitivity / (ppv + sensitivity), 6)
        if ppv is not None and sensitivity
        else (0.0 if ppv is not None else None)
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


def _operating_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised sensitivity/specificity at every unique score threshold."""
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    labels = y[order]

    tp_at_least = np.cumsum(labels == 1)
    fp_at_least = np.cumsum(labels == 0)

    negative_unique, first = np.unique(-scores, return_index=True)
    thresholds = -negative_unique
    last = np.append(first[1:] - 1, len(scores) - 1)

    tp = tp_at_least[last].astype(float)
    fp = fp_at_least[last].astype(float)
    positives = float(tp_at_least[-1])
    negatives = float(fp_at_least[-1])

    sensitivity = tp / positives if positives else np.zeros_like(tp)
    specificity = 1.0 - (fp / negatives if negatives else np.zeros_like(fp))
    ppv = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    return thresholds, sensitivity, specificity, ppv, tp


def select_threshold(
    y_true,
    y_score,
    *,
    target_sensitivity: float = 0.8,
) -> dict:
    """Choose the most specific threshold that still meets the sensitivity target."""
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")

    y, p = _clean(y_true, y_score)
    if not len(y):
        return {
            "threshold": 0.5,
            "target_sensitivity": target_sensitivity,
            "sensitivity": None,
            "specificity": None,
            "ppv": None,
            "npv": None,
            "f1": None,
            "confusion": None,
        }

    thresholds, sensitivity, specificity, ppv, _ = _operating_points(y, p)
    valid = np.flatnonzero(sensitivity >= target_sensitivity - 1e-12)

    if len(valid):
        # Thresholds run from high to low, so the first acceptable point has
        # the greatest specificity. Prefer the higher PPV on an exact tie.
        first = int(valid[0])
        tied = valid[np.isclose(specificity[valid], specificity[first], atol=1e-12)]
        if len(tied):
            best_ppv = np.max(ppv[tied])
            tied = tied[np.isclose(ppv[tied], best_ppv, atol=1e-12)]
            first = int(tied[0])
        threshold = float(thresholds[first])
    else:
        threshold = float(np.min(p))

    confusion = _confusion_at(y, p, threshold)
    rates = _rates(confusion)
    return {
        "threshold": round(threshold, 6),
        "target_sensitivity": target_sensitivity,
        **rates,
        "confusion": confusion,
    }


def threshold_analysis(
    y_true,
    y_score,
    *,
    target_sensitivity: float = 0.8,
    max_points: int = 200,
) -> pd.DataFrame:
    """Return a compact table of operating points for the report."""
    y, p = _clean(y_true, y_score)
    if not len(y):
        return pd.DataFrame(
            columns=[
                "threshold",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "f1",
                "tp",
                "fp",
                "tn",
                "fn",
                "selected",
            ]
        )

    thresholds = np.unique(p)
    if len(thresholds) > max_points:
        index = np.linspace(0, len(thresholds) - 1, max_points).round().astype(int)
        thresholds = np.unique(thresholds[index])

    selected = select_threshold(y, p, target_sensitivity=target_sensitivity)["threshold"]
    thresholds = np.unique(np.append(thresholds, selected))

    rows = []
    for threshold in thresholds:
        confusion = _confusion_at(y, p, threshold)
        rates = _rates(confusion)
        rows.append(
            {
                "threshold": round(float(threshold), 6),
                **rates,
                "tp": confusion["tp"],
                "fp": confusion["fp"],
                "tn": confusion["tn"],
                "fn": confusion["fn"],
                "selected": bool(abs(float(threshold) - selected) < 1e-9),
            }
        )
    frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return frame


def metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    """Convenience wrapper so reports share one metric definition."""
    return binary_metrics(y_true, y_score, threshold=threshold)
