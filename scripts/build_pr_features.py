"""Build the PR household-member feature table and write the aggregate PR audit.

Run from the project root:

    python scripts/build_pr_features.py
    python scripts/build_pr_features.py --limit 100000   # smoke mode

Outputs:
    data/processed/pr_features.parquet
    data/pr_feature_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from screener.config import PR_PATH, PROCESSED_DIR  # noqa: E402
from screener.features import (  # noqa: E402
    PR_SOURCE_COLUMNS,
    build_pr_feature_audit,
    build_pr_features,
)


def _read_prefix(limit: int) -> pd.DataFrame:
    """Read the first ``limit`` PR rows for a smoke run."""
    reader = pd.read_stata(
        PR_PATH,
        columns=PR_SOURCE_COLUMNS,
        convert_categoricals=False,
        iterator=True,
    )
    return reader.read(nrows=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Read only the first N PR rows (smoke mode).",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_pr = None if args.limit is None else _read_prefix(args.limit)
    print("Building PR feature table ...")
    features = build_pr_features(raw_pr)
    audit = build_pr_feature_audit(features)

    features_path = PROCESSED_DIR / "pr_features.parquet"
    audit_path = ROOT / "data" / "pr_feature_audit.json"

    print(f"Writing {features_path} ...")
    features.to_parquet(features_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    summary = {
        "rows": audit["rows"],
        "positive": audit["overall"]["positive"],
        "prevalence": audit["overall"]["prevalence"],
        "age_15_49": audit["age_15_49"],
        "age_50_plus": audit["age_50_plus"],
        "by_sex": audit["by_sex"],
        "bp_valid_rows": audit["blood_pressure"]["valid_rows"],
        "bp_valid_rate": audit["blood_pressure"]["valid_rate"],
        "bp_positive_rows": audit["blood_pressure"]["positive_rows"],
        "weighted_prevalence": audit["weighted_prevalence"],
        "reconciliation": audit["reconciliation"]["status"],
    }
    if args.limit is not None:
        summary["reconciliation"] = "skipped (smoke mode)"
    print(json.dumps(summary, indent=2))

    failures = [
        name
        for name, check in audit["reconciliation"]["checks"].items()
        if not check["pass"]
    ]
    if args.limit is not None:
        print(
            "Smoke mode: skipped the full-data reference reconciliation.",
        )
    elif failures:
        print(
            "Reconciliation did not match the documented reference counts: "
            + ", ".join(failures),
            file=sys.stderr,
        )
        return 1

    print(f"\nWrote {features_path}")
    print(f"Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
