"""Build the harmonised IR/MR feature table and write aggregate audit outputs.

Run from the project root:

    python scripts/build_features.py

Outputs:
    data/processed/irmr_features.parquet
    data/feature_audit.json
    data/merge_validation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.config import PROCESSED_DIR  # noqa: E402
from screener.features import build_irmr_features  # noqa: E402


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building IR + MR feature table ...")
    features, audit, merge_report = build_irmr_features()

    features_path = PROCESSED_DIR / "irmr_features.parquet"
    audit_path = ROOT / "data" / "feature_audit.json"
    merge_path = ROOT / "data" / "merge_validation.json"

    print(f"Writing {features_path} ...")
    features.to_parquet(features_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    merge_path.write_text(json.dumps(merge_report, indent=2), encoding="utf-8")

    summary = {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "age_15_54": audit["age_15_54"],
        "age_15_49": audit["age_15_49"],
        "by_recode": audit["by_recode"],
        "by_sex": audit["by_sex"],
        "merge_status": merge_report["status"],
        "merge_matched_rows": merge_report["matched_rows"],
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {features_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {merge_path}")


if __name__ == "__main__":
    main()
