"""Run PR feature-set variants and compare each with the default PR model.

The variants are configuration on top of the shared training pipeline; this
script adds no modeling logic. Every run uses the same PR sample and saved split
assignment, so the comparison is not confounded by a different partition.

Run from the project root:

    python scripts/run_pr_variants.py --sample 15plus --smoke
    python scripts/run_pr_variants.py --sample 15plus --variants state fairness

Outputs go to ``artifacts/pr/variants/``. Each run keeps its own timestamped
artifact directory, and the comparison is written as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from screener.config import ARTIFACTS_DIR, PROCESSED_DIR, relative_to_project
from screener.feature_policy import PR_VARIANT_NAMES
from screener.pipeline import run_training_pipeline
from screener.pr_model import (
    pr_15_49_config,
    pr_15_plus_config,
    reconcile_pr_15_49_sample,
    reconcile_pr_bp_sample,
)
from screener.pr_variants import compare_pr_variants, render_pr_variant_comparison

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


def _run(
    *,
    features: pd.DataFrame,
    split_assignment: pd.DataFrame,
    split_path: str,
    builder,
    variant: str,
    args,
    reconciliation: dict,
    output: Path,
):
    config = builder(
        feature_table=relative_to_project(args.features),
        split_assignment=split_assignment,
        split_assignment_path=split_path,
        seed=args.seed,
        target_sensitivity=args.sensitivity,
        bootstrap_replicates=args.bootstrap,
        shap_sample_size=args.shap_sample,
        smoke_rows=args.smoke_rows if args.smoke else None,
        feature_variant=variant,
        metadata={"sample_reconciliation": reconciliation},
    )
    return run_training_pipeline(features, config, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        choices=(SAMPLE_15_PLUS, SAMPLE_15_49),
        default=SAMPLE_15_PLUS,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=PR_VARIANT_NAMES,
        default=list(PR_VARIANT_NAMES),
        help="Feature-set variants to run alongside the default model.",
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
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "pr" / "variants",
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
            f"PR feature table not found: {args.features}\n"
            "Run scripts/build_pr_features.py first.",
            file=sys.stderr,
        )
        return 1

    if not args.split_assignment.exists():
        print(
            f"PR split assignment not found: {args.split_assignment}\n"
            "Run scripts/build_pr_split.py first; the variant suite needs one "
            "shared partition for every run.",
            file=sys.stderr,
        )
        return 1

    features = pd.read_parquet(args.features)
    split_assignment = pd.read_parquet(args.split_assignment)
    split_path = relative_to_project(args.split_assignment)

    reconciliation = _RECONCILERS[args.sample](features)
    builder = _CONFIG_BUILDERS[args.sample]

    default_result = _run(
        features=features,
        split_assignment=split_assignment,
        split_path=split_path,
        builder=builder,
        variant="default",
        args=args,
        reconciliation=reconciliation,
        output=args.output / "runs",
    )
    variant_results = {}
    for variant in args.variants:
        variant_results[variant] = _run(
            features=features,
            split_assignment=split_assignment,
            split_path=split_path,
            builder=builder,
            variant=variant,
            args=args,
            reconciliation=reconciliation,
            output=args.output / "runs",
        )

    comparison = compare_pr_variants(default_result, variant_results)
    comparison["runs"] = {
        "default": {
            "run_dir": str(default_result.run_dir),
            "artifacts": {
                name: str(path)
                for name, path in default_result.artifacts.items()
            },
        },
        **{
            variant: {
                "run_dir": str(result.run_dir),
                "artifacts": {
                    name: str(path) for name, path in result.artifacts.items()
                },
            }
            for variant, result in variant_results.items()
        },
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    comparison_dir = args.output / f"comparison_{args.sample}_{stamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    json_path = comparison_dir / "variant_comparison.json"
    markdown_path = comparison_dir / "variant_comparison.md"
    json_path.write_text(
        json.dumps(comparison, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_pr_variant_comparison(comparison),
        encoding="utf-8",
    )

    summary = {
        "comparison_dir": str(comparison_dir),
        "comparison_json": str(json_path),
        "comparison_markdown": str(markdown_path),
        "sample": args.sample,
        "smoke": args.smoke,
        "reconciliation": reconciliation["status"],
        "runs": {
            "default": {
                "run_dir": str(default_result.run_dir),
                "feature_count": len(default_result.config.feature_columns),
                "roc_auc": default_result.metrics["overall"]["roc_auc"],
                "pr_auc": default_result.metrics["overall"]["pr_auc"],
            },
            **{
                variant: {
                    "run_dir": str(result.run_dir),
                    "feature_count": len(result.config.feature_columns),
                    "feature_columns": list(result.config.feature_columns),
                    "roc_auc": result.metrics["overall"]["roc_auc"],
                    "pr_auc": result.metrics["overall"]["pr_auc"],
                    "delta_roc_auc": comparison["variants"][variant]["deltas"][
                        "roc_auc"
                    ],
                    "delta_pr_auc": comparison["variants"][variant]["deltas"][
                        "pr_auc"
                    ],
                }
                for variant, result in variant_results.items()
            },
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
