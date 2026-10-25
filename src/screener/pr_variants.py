"""Compare PR feature-set variants against the default PR model.

The variants exist to test the default model's exclusion decisions rather than
assume the excluded columns are irrelevant. This module is pure: it accepts
finished pipeline results and returns a structured comparison plus a Markdown
report. Running the variants is the caller's job.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import TrainingResult
from .pr_model import PR_AUDIT_FEATURE_VARIANTS, PR_AUDIT_SUBGROUP_NAMES

MANDATORY_SUBGROUPS = ("sex", "age_band")

COMPARISON_METRICS = (
    "n",
    "positives",
    "prevalence",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "sensitivity_at_specificity_0_8",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "calibration_slope",
    "calibration_intercept",
    "ece",
    "threshold",
)

VARIANT_LABELS = {
    "state": "State",
    "fairness": "Fairness",
    "combined": "Combined",
    "cooking_fuel": "Cooking fuel",
}

VARIANT_MESSAGES = (
    (
        "Variants are compared on the same PR sample and split assignment. Each "
        "variant fits its own calibration and screening threshold on out-of-fold "
        "predictions, so the comparison is about discrimination, calibration, and "
        "threshold behaviour rather than raw probability levels."
    ),
    (
        "A higher ROC-AUC or PR-AUC for a variant is not automatically preferable; "
        "added features can improve discrimination while harming portability or "
        "fairness."
    ),
    (
        "Excluding a protected attribute does not by itself make a model fair. The "
        "fairness variant tests what including religion and social group changes."
    ),
    (
        "Religion and social group describe the household head, not necessarily the "
        "respondent."
    ),
    (
        "Subgroup estimates flagged reliable = no have too few respondents or "
        "positives to support a fairness claim."
    ),
    (
        "The PR model is a sensitivity analysis. The primary IR/MR model remains "
        "the deployed screener."
    ),
)


def _plain(value: Any) -> Any:
    """Convert numpy and pandas scalars into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _require_mandatory_subgroups(result: TrainingResult) -> None:
    frame = result.subgroup_metrics
    present = set() if frame.empty else set(frame["subgroup"].astype(str))
    missing = [name for name in MANDATORY_SUBGROUPS if name not in present]
    if missing:
        raise ValueError(
            f"{result.config.name} is missing mandatory subgroup metrics: "
            + ", ".join(missing)
        )


def _require_audit_subgroups(variant: str, result: TrainingResult) -> None:
    frame = result.subgroup_metrics
    present = set() if frame.empty else set(frame["subgroup"].astype(str))
    missing = [name for name in PR_AUDIT_SUBGROUP_NAMES if name not in present]
    if missing:
        raise ValueError(
            f"the {variant} variant is missing fairness audit subgroup metrics: "
            + ", ".join(missing)
        )


def _run_metrics(result: TrainingResult) -> dict:
    overall = result.metrics.get("overall", {})
    metrics = {name: _plain(overall.get(name)) for name in COMPARISON_METRICS}
    metrics["calibration"] = _plain(
        result.metrics.get("calibration", {}).get("selected")
    )
    metrics["selected_family"] = _plain(
        result.run_metadata.get("selected_model", {}).get("family")
    )
    return metrics


def _run_snapshot(result: TrainingResult) -> dict:
    return {
        "name": result.config.name,
        "run_id": result.run_metadata.get("run_id"),
        "feature_variant": result.config.metadata.get("feature_variant"),
        "feature_columns": list(result.config.feature_columns),
        "feature_count": len(result.config.feature_columns),
        "metrics": _run_metrics(result),
    }


def _subgroup_rows(frame: pd.DataFrame, names) -> list[dict]:
    if frame.empty or "subgroup" not in frame.columns:
        return []
    rows = frame[frame["subgroup"].isin(list(names))]
    return [_plain(row) for row in rows.to_dict("records")]


def _audit_rows(frame: pd.DataFrame) -> list[dict]:
    if frame.empty or "audit_only" not in frame.columns:
        return []
    rows = frame[frame["audit_only"].astype(bool)]
    return [_plain(row) for row in rows.to_dict("records")]


def _subgroup_snapshot(result: TrainingResult) -> dict:
    frame = result.subgroup_metrics
    return {
        "sex": _subgroup_rows(frame, ("sex",)),
        "age_band": _subgroup_rows(frame, ("age_band",)),
        "audit": _audit_rows(frame),
    }


def _deltas(metrics: dict, reference: dict) -> dict:
    deltas = {}
    for name in COMPARISON_METRICS:
        value = metrics.get(name)
        baseline = reference.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(baseline, (int, float))
            and not isinstance(baseline, bool)
        ):
            deltas[name] = round(float(value) - float(baseline), 6)
        else:
            deltas[name] = None
    return deltas


def _small_group_rules(result: TrainingResult) -> dict:
    rules = result.run_metadata.get("subgroup_rules")
    if rules:
        return {
            "small_group_min_rows": _plain(rules.get("small_group_min_rows")),
            "small_group_min_positives": _plain(
                rules.get("small_group_min_positives")
            ),
            "state_min_test_rows": _plain(rules.get("state_min_test_rows")),
        }
    return {
        "small_group_min_rows": result.config.small_group_min_rows,
        "small_group_min_positives": result.config.small_group_min_positives,
        "state_min_test_rows": result.config.state_min_test_rows,
    }


def compare_pr_variants(
    default: TrainingResult,
    variants: Mapping[str, TrainingResult],
) -> dict:
    """Compare each feature-set variant with the default PR run.

    ``default`` and every variant must carry the mandatory sex and age-band
    subgroup rows. The comparison never treats a variant as universally better;
    it reports the feature delta and the metric delta.
    """
    _require_mandatory_subgroups(default)
    default_snapshot = _run_snapshot(default)
    default_snapshot["subgroups"] = _subgroup_snapshot(default)

    comparison = {
        "default": default_snapshot,
        "variants": {},
        "small_group_rules": _small_group_rules(default),
        "mandatory_subgroups": list(MANDATORY_SUBGROUPS),
        "messages": list(VARIANT_MESSAGES),
    }

    for name, result in variants.items():
        _require_mandatory_subgroups(result)
        if name in PR_AUDIT_FEATURE_VARIANTS:
            _require_audit_subgroups(name, result)
        snapshot = _run_snapshot(result)
        snapshot["added_features"] = [
            column
            for column in snapshot["feature_columns"]
            if column not in default_snapshot["feature_columns"]
        ]
        snapshot["removed_features"] = [
            column
            for column in default_snapshot["feature_columns"]
            if column not in snapshot["feature_columns"]
        ]
        snapshot["deltas"] = _deltas(
            snapshot["metrics"], default_snapshot["metrics"]
        )
        snapshot["subgroups"] = _subgroup_snapshot(result)
        comparison["variants"][name] = snapshot

    return comparison


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_delta(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def _variant_label(name: str) -> str:
    return VARIANT_LABELS.get(name, name)


def _overall_table(comparison: dict) -> list[str]:
    variants = comparison["variants"]
    header = ["| Metric | Default |"]
    separator = ["|---|---:|"]
    for name in variants:
        header.append(f" {_variant_label(name)} | Δ {_variant_label(name)} |")
        separator.append("---:|---:|")

    rows = [
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Brier", "brier"),
        ("Log loss", "log_loss"),
        ("Sensitivity @ 80% specificity", "sensitivity_at_specificity_0_8"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("PPV", "ppv"),
        ("NPV", "npv"),
        ("F1", "f1"),
        ("ECE", "ece"),
        ("Calibration slope", "calibration_slope"),
        ("Calibration intercept", "calibration_intercept"),
        ("Threshold", "threshold"),
        ("n", "n"),
        ("Positives", "positives"),
        ("Prevalence", "prevalence"),
    ]
    lines = ["".join(header), "".join(separator)]
    for label, key in rows:
        cells = [
            f"| {label} | {_format(comparison['default']['metrics'].get(key))} |"
        ]
        for variant in variants.values():
            cells.append(f" {_format(variant['metrics'].get(key))} |")
            cells.append(f" {_format_delta(variant['deltas'].get(key))} |")
        lines.append("".join(cells))

    lines.append(
        f"| Calibration | {_format(comparison['default']['metrics'].get('calibration'))} |"
        + "".join(
            f" {_format(variant['metrics'].get('calibration'))} | n/a |"
            for variant in variants.values()
        )
    )
    lines.append(
        f"| Selected family | {_format(comparison['default']['metrics'].get('selected_family'))} |"
        + "".join(
            f" {_format(variant['metrics'].get('selected_family'))} | n/a |"
            for variant in variants.values()
        )
    )
    return lines


def _feature_table(comparison: dict) -> list[str]:
    lines = [
        "| Run | Variant | Features | Added vs default | Removed vs default |",
        "|---|---|---:|---|---|",
        (
            f"| `{comparison['default']['name']}` | default | "
            f"{comparison['default']['feature_count']} | — | — |"
        ),
    ]
    for name, variant in comparison["variants"].items():
        added = ", ".join(f"`{column}`" for column in variant["added_features"]) or "—"
        removed = (
            ", ".join(f"`{column}`" for column in variant["removed_features"]) or "—"
        )
        lines.append(
            f"| `{variant['name']}` | {_variant_label(name)} | "
            f"{variant['feature_count']} | {added} | {removed} |"
        )
    return lines


def _subgroup_table(comparison: dict, name: str) -> list[str]:
    lines = [
        (
            "| Run | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | "
            "Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | "
            "Δ Sensitivity | Δ ECE | Reliable |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    runs = [("default", comparison["default"])]
    runs += list(comparison["variants"].items())
    for label, run in runs:
        for row in run["subgroups"].get(name, []):
            lines.append(
                f"| {_variant_label(label)} | {row['group']} | {row['n']} | "
                f"{row['positives']} | {_format(row.get('prevalence'))} | "
                f"{_format(row.get('roc_auc'))} | {_format(row.get('pr_auc'))} | "
                f"{_format(row.get('sensitivity'))} | "
                f"{_format(row.get('calibration_error'))} | "
                f"{_format(row.get('gap_prevalence'))} | "
                f"{_format(row.get('gap_roc_auc'))} | "
                f"{_format(row.get('gap_pr_auc'))} | "
                f"{_format(row.get('gap_sensitivity'))} | "
                f"{_format(row.get('gap_calibration_error'))} | "
                f"{'yes' if row.get('reliable') else 'no'} |"
            )
    return lines


def _audit_table(comparison: dict) -> list[str]:
    rows = []
    runs = [("default", comparison["default"])]
    runs += list(comparison["variants"].items())
    for label, run in runs:
        for row in run["subgroups"].get("audit", []):
            rows.append((label, row))
    if not rows:
        return ["No audit-only subgroup rows were produced for these runs."]

    lines = [
        (
            "| Run | Subgroup | Group | n | Positives | ROC-AUC | PR-AUC | "
            "Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | "
            "Δ Sensitivity | Δ ECE | Reliable |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for label, row in rows:
        lines.append(
            f"| {_variant_label(label)} | {row['subgroup']} | {row['group']} | "
            f"{row['n']} | {row['positives']} | {_format(row.get('roc_auc'))} | "
            f"{_format(row.get('pr_auc'))} | {_format(row.get('sensitivity'))} | "
            f"{_format(row.get('calibration_error'))} | "
            f"{_format(row.get('gap_prevalence'))} | "
            f"{_format(row.get('gap_roc_auc'))} | "
            f"{_format(row.get('gap_pr_auc'))} | "
            f"{_format(row.get('gap_sensitivity'))} | "
            f"{_format(row.get('gap_calibration_error'))} | "
            f"{'yes' if row.get('reliable') else 'no'} |"
        )
    return lines


def render_pr_variant_comparison(comparison: dict) -> str:
    """Render a structured variant comparison as readable Markdown."""
    rules = comparison["small_group_rules"]
    lines = [
        "# PR feature-set variant comparison",
        "",
        (
            f"Default run: `{comparison['default']['run_id']}` "
            f"(`{comparison['default']['name']}`)."
        ),
        "",
        (
            "Each variant adds a documented feature set to the default PR model and "
            "is evaluated on the same PR sample and split assignment."
        ),
        "",
        "## Feature sets",
        "",
        *_feature_table(comparison),
        "",
        "## Overall metrics",
        "",
        *_overall_table(comparison),
        "",
        "## Sex-specific metrics (mandatory)",
        "",
        *_subgroup_table(comparison, "sex"),
        "",
        "## Age-band-specific metrics (mandatory)",
        "",
        *_subgroup_table(comparison, "age_band"),
        "",
        "## Fairness audit subgroups",
        "",
        (
            "Religion and social group are audit-only. They describe the household "
            "head, not necessarily the respondent."
        ),
        "",
        *_audit_table(comparison),
        "",
        (
            f"Subgroups with fewer than {rules['small_group_min_rows']} respondents "
            f"or fewer than {rules['small_group_min_positives']} positives are "
            "flagged `Reliable = no` and must not be used to claim fairness."
        ),
        "",
        "## What this shows and does not show",
        "",
        *[f"- {message}" for message in comparison["messages"]],
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "COMPARISON_METRICS",
    "MANDATORY_SUBGROUPS",
    "VARIANT_LABELS",
    "VARIANT_MESSAGES",
    "compare_pr_variants",
    "render_pr_variant_comparison",
]
