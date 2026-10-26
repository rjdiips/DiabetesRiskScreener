"""Write the primary-versus-PR comparison report from finished runs.

The three runs are read from their artifact directories. This script adds no
modelling or comparison logic beyond loading the artifacts and writing the
report, so it works for a smoke run and for the full real-data run.

Run from the project root:

    python scripts/run_pr_comparison.py \
        --primary artifacts/primary/<run> \
        --pr-15-49 artifacts/pr/runs/<run> \
        --pr-15-plus artifacts/pr/runs/<run>

Outputs default to ``artifacts/pr/comparison/``. The comparison is written as
``primary_vs_pr_comparison.json`` and ``primary_vs_pr_comparison.md``, and the
PR sensitivity model-card addendum as
``pr_sensitivity_model_card_addendum.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.config import ARTIFACTS_DIR
from screener.pr_comparison import (
    compare_primary_vs_pr,
    load_run_result,
    run_metric,
    write_primary_vs_pr_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="Artifact directory for the primary IR/MR 15-49 run.",
    )
    parser.add_argument(
        "--pr-15-49",
        "--pr-1549",
        dest="pr_15_49",
        type=Path,
        required=True,
        help="Artifact directory for the PR 15-49 age-matched run.",
    )
    parser.add_argument(
        "--pr-15-plus",
        "--pr-15plus",
        dest="pr_15_plus",
        type=Path,
        required=True,
        help="Artifact directory for the PR 15+ run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "pr" / "comparison",
    )
    parser.add_argument(
        "--stem",
        default="primary_vs_pr_comparison",
        help="Base name for the JSON and Markdown outputs.",
    )
    args = parser.parse_args(argv)

    try:
        primary = load_run_result(args.primary)
        pr_15_49 = load_run_result(args.pr_15_49)
        pr_15_plus = load_run_result(args.pr_15_plus)
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Could not load the comparison runs: {error}", file=sys.stderr)
        return 1

    comparison = compare_primary_vs_pr(primary, pr_15_49, pr_15_plus)
    paths = write_primary_vs_pr_report(
        comparison,
        args.output,
        stem=args.stem,
    )

    summary = {
        "comparison_json": str(paths["json"]),
        "comparison_markdown": str(paths["markdown"]),
        "model_card_addendum": str(paths["model_card_addendum"]),
        "runs": {
            key: {
                "run_id": comparison["runs"][key]["run_id"],
                "feature_count": comparison["runs"][key]["feature_count"],
                "roc_auc": run_metric(comparison, key, "roc_auc"),
                "pr_auc": run_metric(comparison, key, "pr_auc"),
            }
            for key in ("primary", "pr_15_49", "pr_15_plus")
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
