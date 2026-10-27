"""Run the primary IR/MR 15-49 model through the shared training pipeline.

This is a thin command-line wrapper around ``run_training_pipeline``. The
primary model is the deployed reference; the PR runs are sensitivity analyses.

Run from the project root:

    python scripts/run_primary_model.py
    python scripts/run_primary_model.py --smoke
    python scripts/run_primary_model.py --split-assignment artifacts/primary/<run>/split_assignment.parquet

Outputs go to ``artifacts/primary/<run_id>/``. The model bundle and split
assignment are local-only; aggregate metrics, plots, and the model card are
publishable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from screener.config import ARTIFACTS_DIR, PROCESSED_DIR, relative_to_project
from screener.pipeline import run_training_pipeline
from screener.primary_model import (
    primary_15_49_config,
    reconcile_primary_sample,
)
from screener.run_summary import build_run_summary


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
        default=None,
        help="Reuse a saved primary split assignment instead of generating one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "primary",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity", type=float, default=0.8)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument("--shap-sample", type=int, default=5_000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train on a small stratified sample instead of the full table.",
    )
    parser.add_argument("--smoke-rows", type=int, default=5_000)
    args = parser.parse_args(argv)

    if not args.features.exists():
        print(
            f"IR/MR feature table not found: {args.features}\n"
            "Run scripts/build_features.py first.",
            file=sys.stderr,
        )
        return 1

    split_assignment = None
    split_path = None
    if args.split_assignment is not None:
        if not args.split_assignment.exists():
            print(
                f"Primary split assignment not found: {args.split_assignment}",
                file=sys.stderr,
            )
            return 1
        split_assignment = pd.read_parquet(args.split_assignment)
        split_path = relative_to_project(args.split_assignment)

    features = pd.read_parquet(args.features)
    reconciliation = reconcile_primary_sample(features)
    summary = reconciliation["summary"]
    config = primary_15_49_config(
        feature_table=relative_to_project(args.features),
        split_assignment=split_assignment,
        split_assignment_path=split_path,
        seed=args.seed,
        target_sensitivity=args.sensitivity,
        bootstrap_replicates=args.bootstrap,
        shap_sample_size=args.shap_sample,
        smoke_rows=args.smoke_rows if args.smoke else None,
        metadata={"sample_reconciliation": reconciliation},
    )

    result = run_training_pipeline(features, config, args.output)

    print(
        json.dumps(
            build_run_summary(
                result,
                smoke=args.smoke,
                reconciliation_status=reconciliation["status"],
                documented_rows=summary.get("rows"),
                documented_positives=summary.get("positives"),
            ),
            indent=2,
        )
    )
    print(f"\nWrote {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
