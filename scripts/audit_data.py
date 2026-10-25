"""Audit the local NFHS-5 files and write aggregate counts.

Run from the project root:

    python scripts/audit_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from screener.config import AGE_MIN, GLUCOSE_MAX, GLUCOSE_MIN, GLUCOSE_THRESHOLD  # noqa: E402
from screener.data import (  # noqa: E402
    build_target,
    clean_glucose,
    load_irmr_analysis,
    load_pr,
)


def _summary(df: pd.DataFrame, label: str) -> dict:
    n = int(len(df))
    positives = int(df["target"].sum()) if n else 0
    return {
        "label": label,
        "rows": n,
        "positive": positives,
        "prevalence": round(positives / n, 6) if n else None,
    }


def _components(df: pd.DataFrame) -> dict:
    high = df["glucose"] > GLUCOSE_THRESHOLD
    diagnosed = df["diagnosed"].eq(1)
    medication = df["medication"].eq(1)
    union = high | diagnosed | medication
    fact_sheet = high | medication
    return {
        "n": int(len(df)),
        "glucose_gt_140": int(high.sum()),
        "diagnosed": int(diagnosed.sum()),
        "medication": int(medication.sum()),
        "union": int(union.sum()),
        "fact_sheet_high_or_medication": int(fact_sheet.sum()),
        "prevalence_union": round(float(union.mean()), 6) if len(df) else None,
        "prevalence_fact_sheet": round(float(fact_sheet.mean()), 6) if len(df) else None,
    }


def main() -> None:
    print("Loading PR ...")
    pr = load_pr()
    pr_rows_all = len(pr)
    pr_households_all = int(pr["hhid"].nunique())
    pr["glucose"] = clean_glucose(pr["shb74"])
    pr["target"] = build_target(pr, "shb74", "shb56", "shb57")
    pr = pr.loc[pr["glucose"].notna() & pr["target"].notna() & pr["hv104"].isin([1, 2])].copy()

    pr_15 = pr.loc[pr["hv105"].between(15, 98)].copy()
    pr_15_49 = pr.loc[pr["hv105"].between(15, 49)].copy()

    # PR uses shb* names; harmonise for the component helper.
    pr_15_components = pr_15.rename(
        columns={"shb56": "diagnosed", "shb57": "medication"}
    )

    pr_weight = pr_15["hv005"] / 1_000_000
    fact_sheet = (pr_15["glucose"] > GLUCOSE_THRESHOLD) | pr_15["shb57"].eq(1)
    weighted_fact_sheet = float((fact_sheet * pr_weight).sum() / pr_weight.sum())

    sex_weighted = {}
    for sex, name in [(2, "female"), (1, "male")]:
        subset = pr_15.loc[pr_15["hv104"] == sex]
        w = subset["hv005"] / 1_000_000
        fact = (subset["glucose"] > GLUCOSE_THRESHOLD) | subset["shb57"].eq(1)
        sex_weighted[name] = round(float((fact * w).sum() / w.sum()), 6)

    print("Loading IR + MR ...")
    irmr = load_irmr_analysis(age_min=15, age_max=54)
    irmr_15_49 = irmr.loc[irmr["age"].between(15, 49)].copy()

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_definition": "glucose > 140 OR diagnosed == 1 OR medication == 1",
        "glucose_valid_range": [GLUCOSE_MIN, GLUCOSE_MAX],
        "age_min": AGE_MIN,
        "pr": {
            "rows_all_members": pr_rows_all,
            "households_all": pr_households_all,
            "analysis_households": int(pr["hhid"].nunique()),
            "adults_15plus": _summary(pr_15, "PR adults 15+"),
            "adults_15_49": _summary(pr_15_49, "PR adults 15-49"),
            "components_15plus": _components(pr_15_components),
            "weighted_fact_sheet_15plus": round(weighted_fact_sheet, 6),
            "weighted_fact_sheet_by_sex": sex_weighted,
        },
        "irmr": {
            "pooled_15_54": _summary(irmr, "IR + MR, women 15-49 and men 15-54"),
            "pooled_15_49": _summary(irmr_15_49, "IR + MR, adults 15-49"),
            "components_15_54": _components(irmr),
        },
    }

    out_path = ROOT / "data" / "audit.json"
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nAudit")
    print(json.dumps(audit, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
