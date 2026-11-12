"""Post-freeze feature-selection experiment on the frozen winner.

One pre-registered selector runs inside each development fold: fit the winner
family, score permutation importance on the fold's validation rows, keep the
features with positive mean importance, refit on the training rows only, and
predict the validation fold. The selector therefore never sees the validation
row it is scored on, and the test split is never read here. A promoted result
only records that a new freeze is required; it never edits the winner.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .calibration import select_calibrator, select_threshold
from .comparison import reliable_sensitivity_gap
from .metrics import binary_metrics
from .modeling import (
    build_model,
    fit_model,
    model_matrix,
    predict_positive_probability,
)
from .pipeline import (
    SELECTION_SPLIT_LABEL,
    Candidate,
    TrainingConfig,
    _jsonable,
    evaluate_candidates,
    subgroup_metrics,
)
from .promotion import paired_roc_auc_delta_ci, promotion_decision

EXPERIMENT_NAME = "feature_selection"
SELECTOR_METHOD = "permutation_importance"
PERMUTATION_REPEATS = 3
STABILITY_THRESHOLD = 0.8


def _select_fold(
    matrix,
    target,
    folds,
    config: TrainingConfig,
    candidate: Candidate,
    fold: int,
    n_repeats: int,
):
    """Run the selector inside one fold and return its row plus OOF prediction."""
    valid = folds == fold
    train = ~valid
    target_train = target[train]
    if (
        not train.any()
        or not valid.any()
        or np.unique(target_train).size < 2
        or np.unique(target[valid]).size < 2
    ):
        return (
            {"fold": int(fold), "skipped": "single-class fold", "selected_features": []},
            None,
            None,
        )

    model = build_model(
        candidate.family,
        candidate.params,
        feature_columns=config.feature_columns,
        categorical_columns=config.categorical_columns,
        seed=config.seed,
    )
    fit_model(
        model,
        matrix.loc[train],
        target_train,
        family=candidate.family,
        categorical_columns=config.categorical_columns,
    )
    importance = np.asarray(
        permutation_importance(
            model,
            matrix.loc[valid],
            target[valid],
            n_repeats=n_repeats,
            random_state=config.seed,
            scoring="roc_auc",
        ).importances_mean,
        dtype=float,
    )
    selected = [
        feature
        for feature, value in zip(config.feature_columns, importance)
        if np.isfinite(value) and value > 0
    ]
    # An empty fold selection falls back to the configured set, as pre-registered.
    fallback = not selected
    kept = tuple(config.feature_columns) if fallback else tuple(selected)

    kept_matrix = matrix[list(kept)]
    refit = build_model(
        candidate.family,
        candidate.params,
        feature_columns=kept,
        categorical_columns=config.categorical_columns,
        seed=config.seed,
    )
    fit_model(
        refit,
        kept_matrix.loc[train],
        target_train,
        family=candidate.family,
        categorical_columns=config.categorical_columns,
    )
    probabilities = predict_positive_probability(refit, kept_matrix.loc[valid])
    row = {
        "fold": int(fold),
        "selected_features": list(kept),
        "fallback": fallback,
        "metrics": binary_metrics(
            target[valid],
            probabilities,
            threshold=0.5,
            n_bins=config.reliability_bins,
        ),
    }
    return row, importance, (np.flatnonzero(valid), probabilities)


def run_feature_selection_experiment(
    development,
    config: TrainingConfig,
    *,
    winner_candidate: Candidate,
    n_repeats: int = PERMUTATION_REPEATS,
    replicates: int = 1000,
) -> dict:
    """Score one nested feature-selection experiment against the frozen winner."""
    if config.group_column not in development.columns:
        raise ValueError(
            "feature selection needs the development cluster column "
            f"'{config.group_column}'"
        )

    winner = evaluate_candidates(
        development,
        replace(config, pinned_candidate=winner_candidate),
    ).record(winner_candidate.label)

    matrix, _ = model_matrix(
        development,
        config.feature_columns,
        config.categorical_columns,
    )
    target = development[config.target_column].to_numpy()
    folds = pd.to_numeric(development["fold"], errors="coerce").astype(int).to_numpy()

    oof = np.full(len(development), np.nan)
    fold_rows = []
    importances = []
    for fold in sorted(np.unique(folds)):
        row, importance, prediction = _select_fold(
            matrix,
            target,
            folds,
            config,
            winner_candidate,
            int(fold),
            n_repeats,
        )
        fold_rows.append(row)
        if importance is None:
            continue
        importances.append(importance)
        positions, probabilities = prediction
        oof[positions] = probabilities

    scored_folds = len(importances)
    if not scored_folds:
        raise ValueError("no development fold produced out-of-fold predictions")

    counts = {feature: 0 for feature in config.feature_columns}
    for row in fold_rows:
        if "skipped" in row:
            continue
        for feature in row["selected_features"]:
            counts[feature] += 1
    mean_importance = np.mean(np.vstack(importances), axis=0)
    stability = [
        {
            "feature": feature,
            "selected_folds": counts[feature],
            "stability": counts[feature] / scored_folds,
            "mean_importance": mean_importance[index],
        }
        for index, feature in enumerate(config.feature_columns)
    ]

    covered = np.isfinite(oof)
    if not covered.any():
        raise ValueError("the experiment has no out-of-fold predictions")
    covered_target = target[covered]
    calibrator, _ = select_calibrator(
        covered_target,
        oof[covered],
        methods=config.calibration_methods,
    )
    probabilities = np.full(len(development), np.nan)
    probabilities[covered] = np.clip(
        calibrator.predict(oof[covered]), 0.0, 1.0
    )
    calibrated = probabilities[covered]
    threshold_report = select_threshold(
        covered_target,
        calibrated,
        target_sensitivity=config.target_sensitivity,
    )
    overall = binary_metrics(
        covered_target,
        calibrated,
        threshold=threshold_report["threshold"],
        n_bins=config.reliability_bins,
    )
    subgroups = subgroup_metrics(
        development,
        config.target_column,
        probabilities,
        config,
        overall,
        SELECTION_SPLIT_LABEL,
    )

    shared = winner.covered & covered
    if not shared.any():
        raise ValueError("the winner and experiment share no out-of-fold predictions")
    shared_target = target[shared]
    winner_metrics = binary_metrics(
        shared_target,
        winner.oof_probabilities[shared],
        threshold=winner.threshold_report["threshold"],
        n_bins=config.reliability_bins,
    )
    experiment_metrics = binary_metrics(
        shared_target,
        probabilities[shared],
        threshold=threshold_report["threshold"],
        n_bins=config.reliability_bins,
    )
    low, high = paired_roc_auc_delta_ci(
        shared_target,
        winner.oof_probabilities[shared],
        probabilities[shared],
        development[config.group_column].to_numpy()[shared],
        replicates=replicates,
        seed=config.seed,
    )
    interval = None if low is None else (low, high)
    max_gap = reliable_sensitivity_gap(subgroups.to_dict("records"))
    decision = promotion_decision(
        winner_metrics=winner_metrics,
        experiment_metrics=experiment_metrics,
        delta_roc_auc_ci=interval,
        max_reliable_sensitivity_gap=max_gap,
    )

    selected_features = [
        row["feature"]
        for row in stability
        if row["stability"] >= STABILITY_THRESHOLD
    ]
    kept = set(selected_features)
    return _jsonable(
        {
            "experiment": EXPERIMENT_NAME,
            "split": SELECTION_SPLIT_LABEL,
            "development_rows": len(development),
            "candidate": winner_candidate.to_dict(),
            "feature_set": winner.feature_set,
            "feature_count_before": winner.feature_count,
            "feature_count_after": len(selected_features),
            "selector": {
                "method": SELECTOR_METHOD,
                "scoring": "roc_auc",
                "n_repeats": n_repeats,
                "stability_threshold": STABILITY_THRESHOLD,
            },
            "selected_features": selected_features,
            "dropped_features": [
                feature
                for feature in config.feature_columns
                if feature not in kept
            ],
            "per_fold": fold_rows,
            "stability": stability,
            "oof_metrics": {
                "winner": winner.metrics,
                "experiment": overall,
            },
            "winner_metrics": winner_metrics,
            "experiment_metrics": experiment_metrics,
            "delta_roc_auc": decision["delta_roc_auc"],
            "delta_brier": decision["delta_brier"],
            "delta_roc_auc_ci": interval,
            "max_reliable_sensitivity_gap": max_gap,
            "decision": decision,
            "requires_new_freeze": bool(decision["promoted"]),
        }
    )


def experiment_summary(result: dict) -> dict:
    """The compact summary the CLI prints next to the artifact paths."""
    return {
        "experiment": result["experiment"],
        "features_before": result["feature_count_before"],
        "features_after": result["feature_count_after"],
        "selected_features": result["selected_features"],
        "requires_new_freeze": result["requires_new_freeze"],
        "decision": result["decision"],
    }


__all__ = [
    "EXPERIMENT_NAME",
    "PERMUTATION_REPEATS",
    "SELECTOR_METHOD",
    "STABILITY_THRESHOLD",
    "experiment_summary",
    "run_feature_selection_experiment",
]
