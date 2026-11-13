"""Post-hoc GPU-assisted tree search on the chosen feature bundle.

This is an exploration, not the pre-registered comparison. It scores a wider
XGBoost, CatBoost, and LightGBM grid on the existing development folds and
reports whether any candidate would beat the deployed baseline. The test split
is not read.

Run from the project root:

    python scripts/run_gpu_exploration.py
    python scripts/run_gpu_exploration.py --smoke --smoke-rows 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

# The RTX 4060 is CUDA device 0; the Radeon 780M is the integrated display GPU
# and cannot run CUDA. Pin the compute device before importing the libraries.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from screener.comparison import (
    candidate_artifact,
    rank_candidates,
    select_winner,
)
from screener.config import ARTIFACTS_DIR, PROCESSED_DIR, relative_to_project
from screener.pipeline import Candidate, evaluate_candidates, prepare_training_data
from screener.primary_model import primary_15_49_config


def _grid() -> tuple[Candidate, ...]:
    candidates = [
        Candidate(label="baseline-logistic-C10", family="logistic", params={"C": 10.0})
    ]
    for depth in (3, 4, 6):
        for child_weight in (5, 20):
            for learning_rate in (0.03, 0.05):
                candidates.append(
                    Candidate(
                        label=(
                            f"gpu-xgb-d{depth}-mcw{child_weight}"
                            f"-lr{learning_rate}"
                        ),
                        family="xgboost",
                        params={
                            "n_estimators": 600,
                            "learning_rate": learning_rate,
                            "max_depth": depth,
                            "min_child_weight": child_weight,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "reg_lambda": 1.0,
                            "tree_method": "hist",
                            "device": "cuda:0",
                            "n_jobs": 4,
                        },
                    )
                )
    for depth in (3, 4, 6):
        for l2 in (3.0, 10.0):
            for learning_rate in (0.03, 0.05):
                candidates.append(
                    Candidate(
                        label=f"gpu-cat-d{depth}-l2{l2:g}-lr{learning_rate}",
                        family="catboost",
                        params={
                            "iterations": 600,
                            "learning_rate": learning_rate,
                            "depth": depth,
                            "l2_leaf_reg": l2,
                            "task_type": "GPU",
                            "devices": "0",
                            "thread_count": 4,
                        },
                    )
                )
    for leaves in (15, 31, 63):
        for learning_rate in (0.03, 0.05):
            candidates.append(
                Candidate(
                    label=f"cpu-lgbm-leaves{leaves}-lr{learning_rate}",
                    family="lightgbm",
                    params={
                        "n_estimators": 600,
                        "learning_rate": learning_rate,
                        "num_leaves": leaves,
                        "max_depth": -1,
                        "n_jobs": 4,
                    },
                )
            )
    return tuple(candidates)


def _summary(evidence) -> dict:
    records = [candidate_artifact(record) for record in evidence.records]
    ranked = rank_candidates(records)
    eligible = [item for item in ranked if not item["vetoes"]]
    winner, reason = select_winner(eligible) if eligible else (None, "none")
    rows = []
    for item in ranked:
        metrics = item["oof_metrics"]
        rows.append(
            {
                "label": item["label"],
                "family": item["family"],
                "params": item["params"],
                "rank": item["rank"],
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "brier": metrics.get("brier"),
                "calibration_slope": metrics.get("calibration_slope"),
                "calibration_intercept": metrics.get("calibration_intercept"),
                "vetoes": item["vetoes"],
                "max_reliable_veto_sensitivity_gap": item.get(
                    "max_reliable_veto_sensitivity_gap"
                ),
            }
        )
    baseline = next(
        item for item in rows if item["label"] == "baseline-logistic-C10"
    )
    return {
        "exploration": "post_hoc_gpu_tree_search",
        "note": (
            "Exploration only. The deployed selection remains the pre-registered "
            "comparison; the test split was not read."
        ),
        "bundle": evidence.feature_set,
        "development_rows": evidence.development_rows,
        "fold_count": evidence.fold_count,
        "baseline": baseline,
        "winner": None
        if winner is None
        else {
            "label": winner["label"],
            "family": winner["family"],
            "params": winner["params"],
            "pr_auc": winner["oof_metrics"].get("pr_auc"),
            "brier": winner["oof_metrics"].get("brier"),
            "selection_reason": reason,
        },
        "candidates": rows,
    }


def _render(summary: dict) -> str:
    baseline = summary["baseline"]
    lines = [
        "# Post-hoc GPU tree search",
        "",
        summary["note"],
        "",
        f"- Bundle: `{summary['bundle']}`",
        f"- Development rows: {summary['development_rows']}",
        f"- Folds: {summary['fold_count']}",
        (
            f"- Baseline: `{baseline['label']}` "
            f"PR-AUC {baseline['pr_auc']:.4f}, Brier {baseline['brier']:.4f}"
        ),
        "",
        "| Rank | Candidate | Family | PR-AUC | ROC-AUC | Brier | Vetoes |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for item in summary["candidates"]:
        lines.append(
            f"| {item['rank']} | {item['label']} | {item['family']} | "
            f"{item['pr_auc']:.4f} | {item['roc_auc']:.4f} | "
            f"{item['brier']:.4f} | {'; '.join(item['vetoes']) or 'none'} |"
        )
    winner = summary["winner"]
    if winner:
        lines += [
            "",
            (
                f"Best candidate under the amended rule: `{winner['label']}` "
                f"({winner['selection_reason']})."
            ),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "irmr_features.parquet",
    )
    parser.add_argument(
        "--split-assignment",
        type=Path,
        default=ARTIFACTS_DIR / "comparison" / "split_assignment.parquet",
    )
    parser.add_argument("--feature-set", default="measurement_no_diet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "experiments" / "gpu_search",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=5_000)
    args = parser.parse_args(argv)

    if not args.features.exists():
        print(f"Feature table not found: {args.features}", file=sys.stderr)
        return 1
    if not args.split_assignment.exists():
        print(f"Split assignment not found: {args.split_assignment}", file=sys.stderr)
        return 1

    features = pd.read_parquet(args.features)
    config = primary_15_49_config(
        feature_table=relative_to_project(args.features),
        feature_set=args.feature_set,
        split_assignment=pd.read_parquet(args.split_assignment),
        split_assignment_path=relative_to_project(args.split_assignment),
        seed=args.seed,
        smoke_rows=args.smoke_rows if args.smoke else None,
        metadata={"run_role": "gpu_exploration"},
    )
    config = replace(config, candidates=_grid())
    prepared = prepare_training_data(features, config)
    evidence = evaluate_candidates(prepared.development, config)
    summary = _summary(evidence)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gpu_search.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "gpu_search.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps(
        {
            "bundle": summary["bundle"],
            "baseline": summary["baseline"],
            "winner": summary["winner"],
            "candidates": len(summary["candidates"]),
            "output": str(output),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
