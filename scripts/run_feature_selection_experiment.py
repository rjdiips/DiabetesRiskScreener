"""Run the post-freeze feature-selection experiment against a frozen winner.

The command reads the comparison artifact, rebuilds the winner's config and
folds, runs one pre-registered nested permutation-importance selection on the
development rows only, and writes ``exploration_feature_selection.json`` next
to the comparison artifacts. The test split is never read or scored here.

Run from the project root:

    python scripts/run_feature_selection_experiment.py \
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
from screener.exploration import write_exploration
from screener.feature_selection import (
    EXPERIMENT_NAME,
    PERMUTATION_REPEATS,
    experiment_summary,
    run_feature_selection_experiment,
)
from screener.pipeline import Candidate, prepare_training_data
from screener.primary_model import primary_15_49_config

FREEZE_TICKET = ".scratch/model-shortlist/issues/15-promoted-experiment-freeze.md"


def _write_freeze_ticket(exploration_path: Path) -> Path:
    path = ROOT / FREEZE_TICKET
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# 15: Promoted experiment freeze",
                "",
                "**What to build:** Freeze the promoted feature-selection recipe and",
                "rerun the PR pipeline with the new recipe. The previous frozen run",
                "is unchanged; promotion only opens this freeze.",
                "",
                "**Blocked by:** 14",
                "",
                "**Status:** ready-for-agent",
                "",
                f"Context: `{relative_to_project(exploration_path)}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


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
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to the comparison artifact's parent.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=PERMUTATION_REPEATS)
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
            "experiment uses the comparison's folds.",
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
    bundle = artifact.get("bundle")
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
    result = run_feature_selection_experiment(
        prepared.development,
        config,
        winner_candidate=Candidate(
            label=winner["label"],
            family=winner["family"],
            params=winner.get("params") or {},
        ),
        n_repeats=args.n_repeats,
        replicates=args.replicates,
    )
    output_dir = args.output or args.comparison_artifact.parent
    paths = write_exploration(
        artifact,
        EXPERIMENT_NAME,
        result,
        output_dir=output_dir,
    )

    if result["requires_new_freeze"]:
        ticket = _write_freeze_ticket(paths["exploration_json"])
        print(
            "PROMOTED: the feature-selection experiment passed the promotion "
            "rule. A NEW freeze and PR rerun are required. The frozen run is "
            f"unchanged. Opened {relative_to_project(ticket)}.",
            file=sys.stderr,
        )

    print(
        json.dumps(
            {
                "bundle": bundle,
                "feature_set": feature_set,
                "selection_rows": result["development_rows"],
                **experiment_summary(result),
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
