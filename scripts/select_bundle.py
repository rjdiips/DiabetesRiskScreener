"""Select the deployed feature bundle from the written comparison artifacts.

The command reads one ``comparison.json`` per competing measurement bundle,
applies the pre-registered feature-bundle rule, and writes
``bundle_selection.json`` and ``bundle_selection.md``. The questionnaire-only
set is reported by its own comparison but does not compete here.

Run from the project root:

    python scripts/select_bundle.py
    python scripts/select_bundle.py --comparison-root artifacts/comparison --output artifacts/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.bundle_selection import select_bundle, write_bundle_selection
from screener.config import ARTIFACTS_DIR
from screener.feature_policy import NON_COMPETING_FEATURE_SETS, PRIMARY_FEATURE_SETS

DEFAULT_BUNDLES = tuple(
    sorted(set(PRIMARY_FEATURE_SETS) - NON_COMPETING_FEATURE_SETS)
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=ARTIFACTS_DIR / "comparison",
    )
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=list(DEFAULT_BUNDLES),
        help="Competing measurement bundles to read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for the selection artifacts (default: comparison root).",
    )
    args = parser.parse_args(argv)
    output = args.output or args.comparison_root

    artifacts = []
    for bundle in args.bundles:
        path = args.comparison_root / bundle / "comparison.json"
        if not path.exists():
            print(
                f"Comparison artifact not found for bundle '{bundle}': {path}",
                file=sys.stderr,
            )
            return 1
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("test_evaluated_once") is not False:
            print(
                f"Comparison artifact for bundle '{bundle}' has already evaluated "
                "the test split; bundle selection needs out-of-fold evidence only.",
                file=sys.stderr,
            )
            return 1
        artifacts.append(artifact)

    result = select_bundle(artifacts)
    write_bundle_selection(result, output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
