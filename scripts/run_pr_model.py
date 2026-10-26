"""Run the PR BP-based sensitivity model through the shared training pipeline.

This is a thin command-line wrapper around ``run_training_pipeline``. It does
not implement any modeling logic of its own.

Run from the project root:

    python scripts/run_pr_model.py --sample 15plus
    python scripts/run_pr_model.py --sample 15plus --smoke
    python scripts/run_pr_model.py --sample 15-49 --smoke --smoke-rows 2000

Outputs go to ``artifacts/pr/<run_id>/``. The model bundle is local-only and
gitignored; aggregate metrics, plots, and the model card are publishable.
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
from screener.pr_model import (
    pr_15_49_config,
    pr_15_plus_config,
    reconcile_pr_15_49_sample,
    reconcile_pr_bp_sample,
)

# The age-matched comparison reuses the PR 15+ partition. Running both samples
# from the same saved assignment is what keeps the comparison unconfounded.
SAMPLE_15_PLUS = "15plus"
SAMPLE_15_49 = "15-49"

_CONFIG_BUILDERS = {
    SAMPLE_15_PLUS: pr_15_plus_config,
    SAMPLE_15_49: pr_15_49_config,
}
_RECONCILERS = {
    SAMPLE_15_PLUS: reconcile_pr_bp_sample,
    SAMPLE_15_49: reconcile_pr_15_49_sample,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        choices=(SAMPLE_15_PLUS, SAMPLE_15_49),
        default=SAMPLE_15_PLUS,
        help="PR 15+ sensitivity model or the age-matched 15-49 comparison.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "pr_features.parquet",
    )
    parser.add_argument(
        "--split-assignment",
        type=Path,
        default=PROCESSED_DIR / "pr_split_assignment.parquet",
        help="PR split assignment produced by scripts/build_pr_split.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "pr",
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
    parser.add_argument(
        "--feature-variant",
        default="default",
        help="PR feature-set variant (default, state, fairness, combined, cooking_fuel).",
    )
    parser.add_argument(
        "--no-split-assignment",
        action="store_true",
        help="Ignore the saved PR split and assign a fresh partition.",
    )
    args = parser.parse_args(argv)

    if args.sample == SAMPLE_15_49 and args.no_split_assignment:
        print(
            "The 15-49 comparison must reuse the PR 15+ split assignment; "
            "do not pass --no-split-assignment.",
            file=sys.stderr,
        )
        return 2

    if not args.features.exists():
        print(
            f"PR feature table not found: {args.features}\n"
            "Run scripts/build_pr_features.py first.",
            file=sys.stderr,
        )
        return 1

    features = pd.read_parquet(args.features)

    split_assignment = None
    split_path = None
    if not args.no_split_assignment:
        if not args.split_assignment.exists():
            print(
                f"PR split assignment not found: {args.split_assignment}\n"
                "Run scripts/build_pr_split.py first, or pass --no-split-assignment.",
                file=sys.stderr,
            )
            return 1
        split_assignment = pd.read_parquet(args.split_assignment)
        split_path = relative_to_project(args.split_assignment)

    reconciliation = _RECONCILERS[args.sample](features)
    summary = reconciliation["summary"]
    config = _CONFIG_BUILDERS[args.sample](
        feature_table=relative_to_project(args.features),
        split_assignment=split_assignment,
        split_assignment_path=split_path,
        seed=args.seed,
        target_sensitivity=args.sensitivity,
        bootstrap_replicates=args.bootstrap,
        shap_sample_size=args.shap_sample,
        smoke_rows=args.smoke_rows if args.smoke else None,
        feature_variant=args.feature_variant,
        metadata={"sample_reconciliation": reconciliation},
    )

    result = run_training_pipeline(features, config, args.output)
    overall = result.metrics["overall"]

    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "run_name": result.run_metadata["name"],
                "sample": args.sample,
                "sample_label": result.run_metadata["sample"],
                "comparison_run": result.run_metadata.get("comparison_run"),
                "smoke": args.smoke,
                "rows_after_filters": result.run_metadata["row_counts"][
                    "rows_after_filters"
                ],
                "rows_used": result.run_metadata["row_counts"]["rows_used"],
                "test_rows": result.run_metadata["row_counts"]["test_rows"],
                "split_source": result.run_metadata["split_source"],
                "selected_model": result.run_metadata["selected_model"]["family"],
                "reconciliation": reconciliation["status"],
                "documented_rows": summary.get("rows"),
                "documented_positives": summary.get("positives"),
                "documented_match": result.run_metadata["row_counts"][
                    "rows_after_filters"
                ]
                == summary.get("rows"),
                "roc_auc": overall["roc_auc"],
                "pr_auc": overall["pr_auc"],
                "threshold": overall["threshold"],
                "sensitivity": overall["sensitivity"],
                "specificity": overall["specificity"],
                "fact_sheet_pr_auc": result.metrics.get("secondary_target", {}).get(
                    "pr_auc"
                ),
            },
            indent=2,
        )
    )
    print(f"\nWrote {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
