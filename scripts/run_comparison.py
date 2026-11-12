"""Run the fixed candidate comparison for one named feature set or bundle.

The command scores the pre-registered candidates on the development folds only
and writes ``comparison.json`` and ``comparison.md``. The test split is not
scored here; the winner is frozen through the primary run afterwards.

Run from the project root:

    python scripts/run_comparison.py --feature-set measurement_full
    python scripts/run_comparison.py --feature-set questionnaire_only --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from screener.comparison import comparison_config, run_comparison
from screener.config import ARTIFACTS_DIR, PROCESSED_DIR, relative_to_project
from screener.feature_policy import PRIMARY_FEATURE_SETS
from screener.pipeline import prepare_training_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "irmr_features.parquet",
    )
    parser.add_argument(
        "--feature-set",
        default="measurement_full",
        choices=sorted(PRIMARY_FEATURE_SETS),
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
        default=ARTIFACTS_DIR / "comparison",
    )
    parser.add_argument(
        "--save-split-assignment",
        type=Path,
        default=None,
        help="Write the generated split assignment here so every bundle shares it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity", type=float, default=0.8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Score a small stratified sample instead of the full table.",
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
    config = comparison_config(
        feature_set=args.feature_set,
        feature_table=relative_to_project(args.features),
        split_assignment=split_assignment,
        split_assignment_path=split_path,
        seed=args.seed,
        target_sensitivity=args.sensitivity,
        smoke_rows=args.smoke_rows if args.smoke else None,
    )
    prepared = prepare_training_data(features, config)
    saved_split = None
    if args.save_split_assignment is not None:
        args.save_split_assignment.parent.mkdir(parents=True, exist_ok=True)
        prepared.assignment.to_parquet(args.save_split_assignment, index=False)
        saved_split = relative_to_project(args.save_split_assignment)
    artifact = run_comparison(
        prepared.development,
        config,
        output_dir=args.output / args.feature_set,
        bundle=args.feature_set,
        smoke=args.smoke,
    )

    winner = artifact["winner"]
    print(
        json.dumps(
            {
                "bundle": artifact["bundle"],
                "selection_rows": artifact["selection_rows"],
                "candidates": len(artifact["candidates"]),
                "winner": None if winner is None else winner["label"],
                "vetoed": sorted(artifact["vetoes"]),
                "output": str(args.output / args.feature_set),
                "split_assignment": saved_split,
            },
            indent=2,
        )
    )
    if winner is None:
        print(
            "\nNo candidate passed every ranking veto; no deployable winner.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
