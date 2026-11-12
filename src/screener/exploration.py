"""Post-freeze MLP exploration on the frozen winner's feature set.

This module consumes development rows only. It shares the comparison's
candidate scoring and calibration path, then records the result next to the
comparison artifact. Exploration never edits the winner unless the
pre-registered promotion rule says so.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from .comparison import (
    assert_no_test_metrics,
    reliable_sensitivity_gap,
    write_comparison_artifacts,
)
from .pipeline import (
    SELECTION_SPLIT_LABEL,
    Candidate,
    TrainingConfig,
    _jsonable,
    evaluate_candidates,
)
from .promotion import paired_roc_auc_delta_ci, promotion_decision

MLP_EXPLORATION_LABEL = "X1-mlp"


def _score(development, config: TrainingConfig, candidate: Candidate):
    evidence = evaluate_candidates(
        development,
        replace(config, pinned_candidate=candidate),
    )
    return evidence.record(candidate.label)


def _discrimination(target, probabilities) -> dict:
    values = np.unique(target)
    if len(values) < 2:
        return {"roc_auc": None, "brier": float(brier_score_loss(target, probabilities))}
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "brier": float(brier_score_loss(target, probabilities)),
    }


def run_mlp_exploration(
    development,
    config: TrainingConfig,
    *,
    winner_candidate: Candidate,
    mlp_candidate: Candidate | None = None,
    replicates: int = 1000,
) -> dict:
    """Score one MLP against the frozen winner on the same development folds."""
    candidate = mlp_candidate or Candidate(
        label=MLP_EXPLORATION_LABEL,
        family="mlp",
        params={},
    )
    if config.group_column not in development.columns:
        raise ValueError(
            f"exploration needs the development cluster column '{config.group_column}'"
        )

    winner = _score(development, config, winner_candidate)
    experiment = _score(development, config, candidate)
    covered = winner.covered & experiment.covered
    if not covered.any():
        raise ValueError("the winner and MLP share no out-of-fold predictions")

    target = winner.target[covered]
    winner_metrics = _discrimination(target, winner.oof_probabilities[covered])
    experiment_metrics = _discrimination(
        target, experiment.oof_probabilities[covered]
    )
    low, high = paired_roc_auc_delta_ci(
        target,
        winner.oof_probabilities[covered],
        experiment.oof_probabilities[covered],
        development[config.group_column].to_numpy()[covered],
        replicates=replicates,
        seed=config.seed,
    )
    interval = None if low is None else (low, high)
    max_gap = reliable_sensitivity_gap(experiment.subgroups)
    decision = promotion_decision(
        winner_metrics=winner_metrics,
        experiment_metrics=experiment_metrics,
        delta_roc_auc_ci=interval,
        max_reliable_sensitivity_gap=max_gap,
    )

    return _jsonable(
        {
            "experiment": "mlp",
            "split": SELECTION_SPLIT_LABEL,
            "feature_set": experiment.feature_set,
            "feature_count": experiment.feature_count,
            "development_rows": len(development),
            "candidate": candidate.to_dict(),
            "winner": winner_candidate.to_dict(),
            "winner_metrics": winner_metrics,
            "experiment_metrics": experiment_metrics,
            "oof_metrics": {
                "winner": winner.metrics,
                "experiment": experiment.metrics,
            },
            "delta_roc_auc": decision["delta_roc_auc"],
            "delta_brier": decision["delta_brier"],
            "delta_roc_auc_ci": interval,
            "max_reliable_sensitivity_gap": max_gap,
            "decision": decision,
        }
    )


def attach_exploration(
    comparison_artifact: dict,
    name: str,
    result: dict,
) -> dict:
    """Attach one exploration result without touching the winner or test block."""
    comparison_artifact.setdefault("exploration", {})[name] = result
    assert_no_test_metrics(comparison_artifact)
    return comparison_artifact


def write_exploration(
    comparison_artifact: dict,
    name: str,
    result: dict,
    output_dir: str | Path,
) -> dict:
    """Write the exploration JSON and rewrite the comparison artifacts."""
    attach_exploration(comparison_artifact, name, result)
    output = Path(output_dir)
    exploration_path = output / f"exploration_{name}.json"
    exploration_path.parent.mkdir(parents=True, exist_ok=True)
    exploration_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths = write_comparison_artifacts(comparison_artifact, output)
    return {"exploration_json": exploration_path, **paths}


__all__ = [
    "MLP_EXPLORATION_LABEL",
    "attach_exploration",
    "run_mlp_exploration",
    "write_exploration",
]
