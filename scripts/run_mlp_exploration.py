"""Run the post-freeze MLP exploration against a frozen comparison winner.

The command reads the comparison artifact, rebuilds the winner's config, scores
the winner and one MLP on the development folds only, and writes
``exploration_mlp.json`` next to the comparison artifacts. The test split is
never read or scored here.

Run from the project root:

    python scripts/run_mlp_exploration.py \
        --split-assignment data/processed/primary_split_assignment.parquet \
        --comparison-artifact artifacts/comparison/measurement_full/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from screener.config import PROCESSED_DIR, relative_to_project
from screener.exploration import run_mlp_exploration, write_exploration
from screener.pipeline import Candidate, prepare_training_data
from screener.primary_model import primary_15_49_config


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
        help="The split assignment the comparison was selected on.",
    )
    parser.add_argument(
        "--comparison-artifact",
        type=Path,
        required=True,
        help="Path to the comparison.json holding the frozen winner.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Bundle name; defaults to the artifact's bundle.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=1_000)
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
    if args.split_assignment is None:
        print(
            "A split assignment is required: pass --split-assignment so the "
            "exploration uses the comparison's folds.",
            file=sys.stderr,
        )
        return 1
    if not args.split_assignment.exists():
        print(
            f"Primary split assignment not found: {args.split_assignment}",
            file=sys.stderr,
        )
        return 1
    if not args.comparison_artifact.exists():
        print(
            f"Comparison artifact not found: {args.comparison_artifact}",
            file=sys.stderr,
        )
        return 1

    artifact = json.loads(args.comparison_artifact.read_text(encoding="utf-8"))
    winner = artifact.get("winner")
    if winner is None:
        print(
            "The comparison artifact has no winner to explore.",
            file=sys.stderr,
        )
        return 1
    bundle = args.bundle or artifact.get("bundle")
    feature_set = artifact.get("feature_set") or bundle
    if not feature_set:
        print(
            "The comparison artifact does not name a feature set or bundle.",
            file=sys.stderr,
        )
        return 1

    features = pd.read_parquet(args.features)
    config = primary_15_49_config(
        feature_table=relative_to_project(args.features),
        feature_set=feature_set,
        split_assignment=pd.read_parquet(args.split_assignment),
        split_assignment_path=relative_to_project(args.split_assignment),
        seed=args.seed,
        smoke_rows=args.smoke_rows if args.smoke else None,
        metadata={"feature_set": feature_set},
    )
    prepared = prepare_training_data(features, config)
    result = run_mlp_exploration(
        prepared.development,
        config,
        winner_candidate=Candidate(
            label=winner["label"],
            family=winner["family"],
            params=winner.get("params") or {},
        ),
        replicates=args.replicates,
    )
    paths = write_exploration(
        artifact,
        "mlp",
        result,
        output_dir=args.comparison_artifact.parent,
    )

    print(
        json.dumps(
            {
                "bundle": bundle,
                "feature_set": feature_set,
                "selection_rows": result["development_rows"],
                "promoted": result["decision"]["promoted"],
                "decision": result["decision"],
                "exploration": str(paths["exploration_json"]),
                "comparison_json": str(paths["comparison_json"]),
                "comparison_markdown": str(paths["comparison_markdown"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
