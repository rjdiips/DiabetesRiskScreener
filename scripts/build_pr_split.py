"""Assign leakage-free cluster-grouped splits for the PR sensitivity model.

Reuses a primary split assignment where the same clusters appear. Without a
primary assignment, the PR table gets a deterministic PR-only partition.

Run from the project root:

    python scripts/build_pr_split.py
    python scripts/build_pr_split.py --primary-assignment artifacts/primary/split_assignment.parquet

Outputs:
    data/processed/pr_split_assignment.parquet
    data/pr_split_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from screener.config import DATA_DIR, PROCESSED_DIR  # noqa: E402
from screener.splits import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    assign_pr_splits,
    save_split_assignment,
    summarise_splits,
)


def _read_assignment(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "pr_features.parquet",
        help="Path to the PR feature table.",
    )
    parser.add_argument(
        "--primary-assignment",
        type=Path,
        default=None,
        help="Optional primary split assignment (parquet or csv).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "pr_split_assignment.parquet",
        help="Local-only split assignment artifact.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DATA_DIR / "pr_split_summary.json",
        help="Aggregate, publishable split summary.",
    )
    args = parser.parse_args()

    if not args.features.exists():
        print(
            f"PR feature table not found: {args.features}\n"
            "Run scripts/build_pr_features.py first.",
            file=sys.stderr,
        )
        return 1

    features = pd.read_parquet(args.features)
    primary = None
    if args.primary_assignment is not None:
        if not args.primary_assignment.exists():
            print(
                f"Primary split assignment not found: {args.primary_assignment}",
                file=sys.stderr,
            )
            return 1
        primary = _read_assignment(args.primary_assignment)

    assignment = assign_pr_splits(
        features,
        primary,
        seed=args.seed,
        test_fraction=args.test_fraction,
        n_folds=args.folds,
    )
    summary = summarise_splits(assignment, features)
    actual_test_fraction = (
        summary["test"]["rows"] / summary["rows"] if summary["rows"] else None
    )
    summary["config"] = {
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "n_folds": args.folds,
        "grouping": "cluster",
        "primary_assignment_supplied": args.primary_assignment is not None,
        "actual_test_fraction": round(actual_test_fraction, 6)
        if actual_test_fraction is not None
        else None,
    }
    if (
        actual_test_fraction is not None
        and abs(actual_test_fraction - args.test_fraction) > 0.02
    ):
        summary["warnings"] = [
            "actual test fraction "
            f"{actual_test_fraction:.4f} differs from the target "
            f"{args.test_fraction:.4f}; check the primary split assignment"
        ]

    paths = save_split_assignment(
        assignment,
        args.output,
        summary=summary,
        summary_path=args.summary,
    )

    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "clusters": summary["clusters"],
                "households": summary["households"],
                "split_source": summary.get("split_source"),
                "test_rows": summary["test"]["rows"],
                "test_fraction": round(
                    summary["test"]["rows"] / summary["rows"], 6
                )
                if summary["rows"]
                else None,
                "fold_rows": [fold["rows"] for fold in summary["folds"]],
                "age_15_49_rows": summary.get("age_15_49", {}).get("rows"),
                "leakage": summary["leakage"],
            },
            indent=2,
        )
    )
    print(f"\nWrote {paths['assignment']}")
    print(f"Wrote {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
