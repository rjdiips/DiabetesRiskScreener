"""Ten-seed averaging for the selected stochastic winner.

The capability is family- and feature-set-agnostic: it refits whatever candidate
the selection evidence names, pools that candidate's out-of-fold scores across
seeds to fit one calibrator, and averages the ten refitted models at inference.
Deterministic families skip averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import (
    select_calibrator,
    select_threshold,
    select_thresholds_by_group,
    thresholds_for_groups,
)
from .metrics import binary_metrics
from .model_bundle import ModelBundle
from .modeling import predict_positive_probability
from .pipeline import (
    SelectionEvidence,
    _age_band_spec,
    _bundle_threshold_rule,
    threshold_group_labels,
)

DEFAULT_SEEDS = tuple(range(42, 52))

DETERMINISTIC_FAMILIES = frozenset({"prevalence", "logistic", "logistic_spline"})

_REPORTED_METRICS = ("pr_auc", "roc_auc", "brier", "log_loss")


class AveragedModel:
    """A fitted ensemble whose positive-class probability is the seed mean."""

    def __init__(self, models) -> None:
        self.models = tuple(models)
        if not self.models:
            raise ValueError("a seed-averaged model needs at least one fitted model")

    def predict_proba(self, matrix) -> np.ndarray:
        scores = np.mean(
            [predict_positive_probability(model, matrix) for model in self.models],
            axis=0,
        )
        return np.column_stack([1.0 - scores, scores])


@dataclass(frozen=True)
class SeedAveragedFit:
    """The deployable bundle plus the seed-averaging record."""

    bundle: ModelBundle
    report: dict


def is_deterministic(family: str) -> bool:
    """Whether a family is deterministic and must not be seed-averaged."""
    return family in DETERMINISTIC_FAMILIES


def _mean_std(per_seed: list[dict]) -> tuple[dict, dict]:
    mean = {}
    std = {}
    for name in _REPORTED_METRICS:
        values = [row[name] for row in per_seed if row.get(name) is not None]
        mean[name] = float(np.mean(values)) if values else None
        std[name] = float(np.std(values)) if values else None
    return mean, std


def fit_seed_averaged(
    evidence: SelectionEvidence,
    label: str,
    *,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
) -> SeedAveragedFit:
    """Refit the named candidate across seeds and calibrate the ensemble.

    Deterministic candidates are fitted once and reported without averaging.
    Stochastic candidates are refit on every seed; one calibrator is fitted on
    the pooled out-of-fold scores, and the threshold is chosen on that pooled
    evidence.
    """
    handle = evidence.fit_handle
    record = handle.record(label)
    config = handle.config

    if is_deterministic(record.family):
        bundle = handle.fit(label)
        report = {
            "candidate": label,
            "family": record.family,
            "deterministic": True,
            "averaged": False,
            "seeds": [config.seed],
            "ensemble_size": 1,
            "calibration_method": bundle.calibrator.method,
            "threshold": bundle.threshold,
            "age_band_thresholds": bundle.threshold_rule,
            "feature_columns": list(bundle.feature_columns),
            "feature_count": len(bundle.feature_columns),
            "model_version": f"{record.family}-single-v1",
        }
        bundle.metadata.update(
            {"seeds": list(report["seeds"]), "ensemble_size": 1}
        )
        return SeedAveragedFit(bundle=bundle, report=report)

    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("seed averaging needs at least one seed")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seed averaging needs distinct seeds")

    models = []
    seed_oof = []
    pooled_scores = []
    pooled_target = []
    base_bundle = None
    for seed in seed_list:
        oof, covered = handle.out_of_fold(label, seed=seed)
        if not np.array_equal(covered, record.covered):
            raise ValueError("seed refits must cover the same development rows")
        fitted = handle.fit(label, seed=seed)
        if base_bundle is None:
            base_bundle = fitted
        models.append(fitted.model)
        seed_oof.append((seed, oof, covered))
        pooled_scores.append(oof[covered])
        pooled_target.append(record.target[covered])

    pooled_scores = np.concatenate(pooled_scores)
    pooled_target = np.concatenate(pooled_target)
    calibrator, calibration_report = select_calibrator(
        pooled_target,
        pooled_scores,
        methods=config.calibration_methods,
    )
    calibrated_pooled = np.clip(calibrator.predict(pooled_scores), 0.0, 1.0)
    threshold_report = select_threshold(
        pooled_target,
        calibrated_pooled,
        target_sensitivity=config.target_sensitivity,
    )
    threshold = threshold_report["threshold"]

    group_labels = threshold_group_labels(handle.development, config)
    group_rule = None
    bundle_threshold_rule = {}
    covered_thresholds = None
    if group_labels is not None:
        pooled_groups = np.concatenate(
            [group_labels[covered] for _, _, covered in seed_oof]
        )
        group_rule = select_thresholds_by_group(
            pooled_target,
            calibrated_pooled,
            pooled_groups,
            target_sensitivity=config.target_sensitivity,
        )
        bundle_threshold_rule = _bundle_threshold_rule(
            _age_band_spec(config), group_rule
        )
        covered_thresholds = thresholds_for_groups(
            group_rule, group_labels[record.covered]
        )

    per_seed = []
    for seed, oof, covered in seed_oof:
        probabilities = np.clip(calibrator.predict(oof[covered]), 0.0, 1.0)
        metrics = binary_metrics(
            record.target[covered],
            probabilities,
            threshold=(
                covered_thresholds if covered_thresholds is not None else threshold
            ),
            n_bins=config.reliability_bins,
        )
        per_seed.append(
            {
                "seed": seed,
                **{name: metrics.get(name) for name in _REPORTED_METRICS},
            }
        )

    averaged_scores = np.mean([oof for _, oof, _ in seed_oof], axis=0)
    averaged_probabilities = np.clip(
        calibrator.predict(averaged_scores[record.covered]),
        0.0,
        1.0,
    )
    averaged_metrics = binary_metrics(
        record.target[record.covered],
        averaged_probabilities,
        threshold=(
            covered_thresholds if covered_thresholds is not None else threshold
        ),
        n_bins=config.reliability_bins,
    )
    mean, std = _mean_std(per_seed)
    best_single_brier = min(
        row["brier"] for row in per_seed if row["brier"] is not None
    )

    bundle = ModelBundle(
        model=AveragedModel(models),
        calibrator=calibrator,
        feature_columns=base_bundle.feature_columns,
        categorical_columns=base_bundle.categorical_columns,
        category_levels=base_bundle.category_levels,
        threshold=threshold,
        threshold_rule=bundle_threshold_rule,
        target_definition=config.target_definition,
        metadata={
            "candidate_label": label,
            "model_family": record.family,
            "seeds": list(seed_list),
            "ensemble_size": len(seed_list),
            "calibration_method": calibration_report["selected"],
            "feature_set": record.feature_set,
            "feature_count": record.feature_count,
            "model_version": f"{record.family}-seed-averaged-v1",
        },
        model_family=record.family,
    )
    report = {
        "candidate": label,
        "family": record.family,
        "deterministic": False,
        "averaged": True,
        "seeds": list(seed_list),
        "ensemble_size": len(seed_list),
        "calibration_method": calibration_report["selected"],
        "calibration": calibration_report,
        "threshold": threshold,
        "threshold_report": threshold_report,
        "age_band_thresholds": bundle_threshold_rule,
        "feature_columns": list(bundle.feature_columns),
        "feature_count": len(bundle.feature_columns),
        "model_version": bundle.metadata["model_version"],
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "best_single_seed_brier": best_single_brier,
        "averaged_validation": averaged_metrics,
    }
    return SeedAveragedFit(bundle=bundle, report=report)


__all__ = [
    "DEFAULT_SEEDS",
    "DETERMINISTIC_FAMILIES",
    "AveragedModel",
    "SeedAveragedFit",
    "fit_seed_averaged",
    "is_deterministic",
]
