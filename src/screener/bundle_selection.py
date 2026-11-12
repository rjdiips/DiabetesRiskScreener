"""Select the deployed feature bundle with the pre-registered rule.

Only measurement bundles compete: the questionnaire-only set is the
accessibility comparison. Scores are out-of-fold development evidence; the test
split is not read here.
"""

from __future__ import annotations

import json
from pathlib import Path

from .comparison import PREREGISTERED_RULE, choose_bundle
from .feature_policy import NON_COMPETING_FEATURE_SETS


def bundle_records(comparison_artifacts: list[dict]) -> list[dict]:
    """Map each comparison artifact to its bundle and ranking-rule winner."""
    records = []
    for artifact in comparison_artifacts:
        winner = artifact.get("winner")
        records.append(
            {
                "bundle": artifact.get("bundle") or artifact.get("feature_set"),
                "feature_count": artifact.get("feature_count"),
                "eligible": winner is not None,
                "winner": None
                if winner is None
                else {
                    "label": winner["label"],
                    "family": winner["family"],
                    "params": winner["params"],
                    "pr_auc": winner["pr_auc"],
                    "brier": winner["brier"],
                },
            }
        )
    return records


def _competes(record: dict) -> bool:
    return record["bundle"] not in NON_COMPETING_FEATURE_SETS


def select_bundle(artifacts: list[dict]) -> dict:
    """Apply the feature-bundle rule to the competing measurement bundles."""
    records = [
        record for record in bundle_records(artifacts) if _competes(record)
    ]
    chosen = choose_bundle(records)
    eligible = [record for record in records if record["eligible"]]
    reference = max(eligible, key=lambda item: item["winner"]["pr_auc"])
    return {
        "rule": {
            key: PREREGISTERED_RULE[key]
            for key in (
                "ranking",
                "bundle_pr_auc_margin",
                "bundle_brier_margin",
            )
        },
        "reference": reference,
        "chosen": chosen,
        "pinned_candidate": {
            "label": chosen["winner"]["label"],
            "family": chosen["winner"]["family"],
            "params": chosen["winner"]["params"],
        },
        "bundles": records,
    }


def _number(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_bundle_selection(result: dict) -> str:
    """The human-readable feature-bundle selection report."""
    rule = result["rule"]
    chosen = result["chosen"]
    lines = [
        "# Feature-bundle selection",
        "",
        "Scores are out-of-fold development evidence only; the test split was",
        "not read while selecting the bundle.",
        "",
        "## Pre-registered rule",
        "",
        f"- Rank each bundle's candidates by {rule['ranking']}.",
        (
            f"- Reference: the eligible bundle winner with the highest PR-AUC; "
            f"a bundle qualifies within {rule['bundle_pr_auc_margin']} PR-AUC and "
            f"no more than {rule['bundle_brier_margin']} worse Brier."
        ),
        (
            "- Choose the qualifying bundle with the fewest features; break ties "
            "by higher PR-AUC, then lower Brier."
        ),
        "- The questionnaire-only set does not compete for deployment.",
        "",
        "## Bundles",
        "",
        "| Bundle | Features | Eligible | Winner | PR-AUC | Brier |",
        "|---|---:|:---:|---|---:|---:|",
    ]
    for item in result["bundles"]:
        winner = item["winner"]
        lines.append(
            f"| {item['bundle']} | {item['feature_count']} | "
            f"{'yes' if item['eligible'] else 'no'} | "
            f"{winner['label'] if winner else 'none'} | "
            f"{_number(winner['pr_auc'] if winner else None)} | "
            f"{_number(winner['brier'] if winner else None)} |"
        )
    lines += [
        "",
        (
            f"Chosen bundle: **{chosen['bundle']}** "
            f"({chosen['feature_count']} features), "
            f"winner `{result['pinned_candidate']['label']}`."
        ),
        "",
    ]
    return "\n".join(lines)


def write_bundle_selection(result: dict, output_dir: str | Path) -> dict:
    """Write ``bundle_selection.json`` and ``bundle_selection.md``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "bundle_selection.json"
    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path = output / "bundle_selection.md"
    markdown_path.write_text(
        render_bundle_selection(result),
        encoding="utf-8",
    )
    return {
        "bundle_selection_json": json_path,
        "bundle_selection_markdown": markdown_path,
    }


__all__ = [
    "bundle_records",
    "render_bundle_selection",
    "select_bundle",
    "write_bundle_selection",
]
