"""Verify the documented PR sample counts on the local feature tables.

This checks the definition-of-done counts for the PR sensitivity analysis:
the full PR feature table, the BP-valid 15+ sample, the BP-valid 15-49 sample,
the BP-valid 50+ sample, and, when the IR/MR table is present, the primary
15-49 sample. It reads local microdata and prints only aggregate counts.

Run from the project root:

    python scripts/verify_pr_samples.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from screener.config import PROCESSED_DIR
from screener.features import build_pr_feature_audit, reconcile_pr_audit
from screener.pr_model import reconcile_pr_15_49_sample, reconcile_pr_bp_sample
from screener.primary_model import reconcile_primary_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-features",
        type=Path,
        default=PROCESSED_DIR / "pr_features.parquet",
    )
    parser.add_argument(
        "--primary-features",
        type=Path,
        default=PROCESSED_DIR / "irmr_features.parquet",
    )
    args = parser.parse_args(argv)

    if not args.pr_features.exists():
        print(
            f"PR feature table not found: {args.pr_features}\n"
            "Run scripts/build_pr_features.py first.",
            file=sys.stderr,
        )
        return 1

    pr_features = pd.read_parquet(args.pr_features)
    report = {
        "pr_feature_table": reconcile_pr_audit(build_pr_feature_audit(pr_features)),
        "pr_bp_15_plus": reconcile_pr_bp_sample(pr_features),
        "pr_bp_15_49": reconcile_pr_15_49_sample(pr_features),
    }

    if args.primary_features.exists():
        primary_features = pd.read_parquet(args.primary_features)
        report["primary_15_49"] = reconcile_primary_sample(primary_features)

    statuses = {name: block["status"] for name, block in report.items()}
    print(json.dumps({"status": statuses, "reports": report}, indent=2))
    return 0 if all(status == "pass" for status in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
