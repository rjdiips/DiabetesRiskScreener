"""Compare the primary IR/MR model with the PR sensitivity models.

The three runs answer different questions:

* the primary IR/MR 15-49 model is the deployed reference with the rich feature
  set;
* the PR 15-49 model replaces that feature set with the 12 BP/demographic
  features while keeping the age range;
* the PR 15+ model uses the same 12 features but adds adults aged 50+.

This module is pure: it accepts finished pipeline results and returns a
structured comparison plus a Markdown report. Loading runs from artifact
directories and writing the report are separate functions, so the comparison
can be tested without training a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import TrainingConfig, TrainingResult
from .reporting import format_delta, format_value, plain

RUN_KEYS = ("primary", "pr_15_49", "pr_15_plus")

RUN_LABELS = {
    "primary": "Primary IR/MR 15–49",
    "pr_15_49": "PR 15–49",
    "pr_15_plus": "PR 15+",
}

# The order is the side-by-side order used by every table and by the JSON.
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

METRIC_ROWS = (
    ("Test rows", "n"),
    ("Test positives", "positives"),
    ("Prevalence", "prevalence"),
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
    ("Brier", "brier"),
    ("Log loss", "log_loss"),
    ("ECE", "ece"),
    ("Calibration slope", "calibration_slope"),
    ("Calibration intercept", "calibration_intercept"),
    ("Sensitivity @ 80% specificity", "sensitivity_at_specificity_0_8"),
    ("Threshold", "threshold"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("PPV", "ppv"),
    ("NPV", "npv"),
    ("F1", "f1"),
)

DELTA_METRICS = (
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "ece",
    "calibration_slope",
    "calibration_intercept",
    "sensitivity_at_specificity_0_8",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "threshold",
    "prevalence",
)

DELTA_DIRECTIONS = {
    "pr_15_49_vs_primary": "PR 15–49 minus primary",
    "pr_15_plus_vs_primary": "PR 15+ minus primary",
    "pr_15_plus_vs_pr_15_49": "PR 15+ minus PR 15–49",
}

MODEL_NOTES = (
    (
        "The primary IR/MR 15–49 run is the reference model. It uses the rich "
        "questionnaire, anthropometry, diet, and BP feature set."
    ),
    (
        "The PR 15–49 run isolates the feature-set trade-off. It keeps the same "
        "adults aged 15–49 and otherwise uses the same PR cluster partition as the "
        "PR 15+ run, but replaces the primary feature set with the 12 BP-based "
        "features. The comparison with the primary model therefore shows what the "
        "smaller feature set changes, although the underlying recodes and population "
        "coverage still differ."
    ),
    (
        "The PR 15+ run adds older adults. It uses the same 12 features and the same "
        "cluster partition as the PR 15–49 run but retains adults aged 50+, where "
        "prevalence is higher. The difference between the two PR runs mostly reflects "
        "that population extension rather than a feature-set change."
    ),
    (
        "No run is universally better. Feature richness, population coverage, "
        "calibration, and threshold behaviour are separate trade-offs, and a higher "
        "ROC-AUC or PR-AUC does not by itself make a model preferable."
    ),
    (
        "Raw probabilities are not directly comparable across the three models. The "
        "comparison is about discrimination, calibration, and threshold behaviour on "
        "different samples and feature sets."
    ),
)

LIMITATIONS = (
    (
        "Raw probabilities are not directly comparable across the three runs because "
        "the samples and feature sets differ. Only discrimination, calibration, "
        "threshold behaviour, and subgroup gaps are compared."
    ),
    (
        "The samples are not nested or exchangeable. The primary model uses the IR/MR "
        "recodes, while the PR models use the household-member recode; the primary "
        "sample is mostly female and the PR sample is far more balanced."
    ),
    (
        "The PR 15–49 run is the cleanest feature-set comparison, but it is not a pure "
        "ablation of the primary model: the PR and IR/MR instruments measure different "
        "concepts and the PR sample covers a different population."
    ),
    (
        "Older adults are present only in the PR 15+ run. Any difference between the PR "
        "15+ and PR 15–49 runs mixes the age extension with the different age-specific "
        "case mix."
    ),
    (
        "Survey-weighted metrics are reported where the run produced them. The test "
        "evaluation is still a single cluster-grouped holdout, not external validation."
    ),
    (
        "Threshold metrics depend on prevalence. PPV and NPV cannot be transported to "
        "another setting without accounting for that setting's prevalence."
    ),
    (
        "The PR model is a sensitivity analysis, not the deployed screener. It is not a "
        "diagnostic test and must not be used for clinical, insurance, employment, or "
        "care-allocation decisions."
    ),
)

_OVERALL_NOTE = (
    "Each run selects its own calibration and screening threshold on its own "
    "out-of-fold predictions. The threshold columns therefore describe each model's "
    "operating point, not a shared cut-off applied to all three probabilities."
)

_WEIGHTED_NOTE = (
    "Survey-weighted rows use each run's survey weight and the threshold selected on "
    "that run's unweighted out-of-fold predictions."
)

_DELTA_NOTE = (
    "Deltas are descriptive and are shown as left run minus right run. They are not "
    "evidence that one model is universally better; the samples and feature sets "
    "differ."
)

_SMALL_GROUP_NOTE = (
    "Subgroups flagged Reliable = no have too few respondents or positives to support "
    "a fairness or performance claim."
)


def _metric_block(metrics: Mapping[str, Any] | None) -> dict:
    metrics = {} if metrics is None else metrics
    return {name: plain(metrics.get(name)) for name in COMPARISON_METRICS}


def _subgroup_rows(frame: pd.DataFrame, name: str) -> list[dict]:
    if frame.empty or "subgroup" not in frame.columns:
        return []
    rows = frame[frame["subgroup"].astype(str).eq(name)]
    return [plain(row) for row in rows.to_dict("records")]


def _all_subgroup_rows(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    return [plain(row) for row in frame.to_dict("records")]


def _share(part: Any, total: Any) -> float | None:
    if not isinstance(part, (int, float)) or not isinstance(total, (int, float)):
        return None
    if isinstance(part, bool) or isinstance(total, bool) or total <= 0:
        return None
    return round(float(part) / float(total), 6)


def _find_count(rows: list[dict], *labels: str) -> int | None:
    wanted = {label.lower() for label in labels}
    for row in rows:
        group = str(row.get("group", "")).strip().lower()
        n = row.get("n")
        if group in wanted and isinstance(n, (int, float)) and not isinstance(n, bool):
            return int(n)
    return None


def _sex_composition(
    sex_rows: list[dict],
    split_test: Mapping[str, Any] | None,
) -> dict:
    split_test = {} if split_test is None else split_test
    female = _find_count(sex_rows, "female")
    male = _find_count(sex_rows, "male")
    total = (
        sum(
            int(row["n"])
            for row in sex_rows
            if isinstance(row.get("n"), (int, float))
            and not isinstance(row.get("n"), bool)
        )
        if sex_rows
        else None
    )

    if female is None and isinstance(split_test.get("female"), (int, float)):
        female = int(split_test["female"])
    if male is None and isinstance(split_test.get("male"), (int, float)):
        male = int(split_test["male"])
    if total is None and female is not None and male is not None:
        total = female + male

    female_share = _share(female, total)
    if female_share is None:
        raw_share = split_test.get("female_share")
        if isinstance(raw_share, (int, float)) and not isinstance(raw_share, bool):
            female_share = round(float(raw_share), 6)

    return {
        "female": female,
        "male": male,
        "total": total,
        "female_share": female_share,
        "male_share": _share(male, total),
    }


def _age_composition(age_rows: list[dict]) -> list[dict]:
    total = sum(
        int(row["n"])
        for row in age_rows
        if isinstance(row.get("n"), (int, float))
        and not isinstance(row.get("n"), bool)
    )
    composition = []
    for row in age_rows:
        n = row.get("n")
        composition.append(
            {
                "group": plain(row.get("group")),
                "n": plain(n),
                "share": _share(n, total),
                "prevalence": plain(row.get("prevalence")),
                "positives": plain(row.get("positives")),
            }
        )
    return composition


def _sample_summary(result: TrainingResult) -> dict:
    overall = result.metrics.get("overall", {}) or {}
    weighted = overall.get("weighted", {}) if isinstance(overall, Mapping) else {}
    metadata = result.run_metadata or {}
    row_counts = metadata.get("row_counts", {}) or {}
    split = result.metrics.get("split", {}) or {}
    split_test = split.get("test", {}) if isinstance(split, Mapping) else {}

    sex_rows = _subgroup_rows(result.subgroup_metrics, "sex")
    age_rows = _subgroup_rows(result.subgroup_metrics, "age_band")

    test_rows = overall.get("n")
    if test_rows is None:
        test_rows = row_counts.get("test_rows")
    test_positives = overall.get("positives")
    if test_positives is None:
        test_positives = row_counts.get("test_positives")

    rows_used = row_counts.get("rows_used")
    if rows_used is None:
        rows_used = test_rows

    return {
        "sample": plain(
            metadata.get("sample") or result.config.metadata.get("sample")
        ),
        "model_role": plain(
            metadata.get("model_role") or result.config.metadata.get("model_role")
        ),
        "rows_used": plain(rows_used),
        "rows_after_filters": plain(row_counts.get("rows_after_filters")),
        "test_rows": plain(test_rows),
        "test_positives": plain(test_positives),
        "prevalence": plain(overall.get("prevalence")),
        "weighted_prevalence": plain(
            weighted.get("prevalence") if isinstance(weighted, Mapping) else None
        ),
        "sex": _sex_composition(sex_rows, split_test),
        "age": _age_composition(age_rows),
        "age_bands_reported": [
            row.get("group") for row in age_rows if row.get("group") is not None
        ],
        "test_female_share": plain(
            split_test.get("female_share") if isinstance(split_test, Mapping) else None
        ),
    }


def _run_snapshot(key: str, result: TrainingResult) -> dict:
    overall = result.metrics.get("overall", {}) or {}
    weighted = overall.get("weighted", {}) if isinstance(overall, Mapping) else {}
    split = result.metrics.get("split", {}) or {}
    threshold_selection = result.metrics.get("threshold_selection", {}) or {}
    selected_model = result.run_metadata.get("selected_model", {}) or {}

    return {
        "key": key,
        "label": RUN_LABELS[key],
        "name": plain(result.config.name),
        "run_id": plain(result.run_metadata.get("run_id")),
        "run_dir": str(result.run_dir),
        "sample": plain(
            result.run_metadata.get("sample")
            or result.config.metadata.get("sample")
        ),
        "model_role": plain(
            result.run_metadata.get("model_role")
            or result.config.metadata.get("model_role")
        ),
        "feature_variant": plain(
            result.run_metadata.get("feature_variant")
            or result.config.metadata.get("feature_variant")
        ),
        "feature_count": len(result.config.feature_columns),
        "feature_columns": list(result.config.feature_columns),
        "metrics": {
            "overall": _metric_block(overall),
            "weighted": _metric_block(weighted if isinstance(weighted, Mapping) else {}),
            "calibrator": plain(
                result.metrics.get("calibrator")
                or result.metrics.get("calibration", {}).get("selected")
            ),
            "threshold_selection": plain(threshold_selection),
            "selected_family": plain(selected_model.get("family")),
        },
        "sample_summary": _sample_summary(result),
        "subgroups": {
            "sex": _subgroup_rows(result.subgroup_metrics, "sex"),
            "age_band": _subgroup_rows(result.subgroup_metrics, "age_band"),
            "all": _all_subgroup_rows(result.subgroup_metrics),
        },
        "reliability_curve": _all_subgroup_rows(result.calibration_curve),
        "split": plain(split),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
    deltas = {}
    for name in DELTA_METRICS:
        left_value = _numeric(left.get(name))
        right_value = _numeric(right.get(name))
        deltas[name] = (
            round(left_value - right_value, 6)
            if left_value is not None and right_value is not None
            else None
        )
    return deltas


def _sample_differences(runs: Mapping[str, dict]) -> list[str]:
    summaries = {key: runs[key]["sample_summary"] for key in RUN_KEYS}
    prevalence_note = (
        "Test prevalence: primary "
        f"{_format_plain(summaries['primary']['prevalence'])}; PR 15–49 "
        f"{_format_plain(summaries['pr_15_49']['prevalence'])}; PR 15+ "
        f"{_format_plain(summaries['pr_15_plus']['prevalence'])}."
    )
    primary_prevalence = _numeric(summaries["primary"]["prevalence"])
    pr_plus_prevalence = _numeric(summaries["pr_15_plus"]["prevalence"])
    if (
        primary_prevalence is not None
        and pr_plus_prevalence is not None
        and pr_plus_prevalence > primary_prevalence
    ):
        prevalence_note += (
            " Higher PR prevalence is expected because the PR sample extends to "
            "older adults."
        )

    notes = [
        (
            "Rows used: primary "
            f"{_format_plain(summaries['primary']['rows_used'])}; PR 15–49 "
            f"{_format_plain(summaries['pr_15_49']['rows_used'])}; PR 15+ "
            f"{_format_plain(summaries['pr_15_plus']['rows_used'])}."
        ),
        prevalence_note,
        (
            "Female share of the test set: primary "
            f"{_format_plain(summaries['primary']['sex']['female_share'])}; PR 15–49 "
            f"{_format_plain(summaries['pr_15_49']['sex']['female_share'])}; PR 15+ "
            f"{_format_plain(summaries['pr_15_plus']['sex']['female_share'])}."
        ),
        (
            "Age bands reported: primary "
            f"{_format_plain(summaries['primary']['age_bands_reported'])}; PR 15–49 "
            f"{_format_plain(summaries['pr_15_49']['age_bands_reported'])}; PR 15+ "
            f"{_format_plain(summaries['pr_15_plus']['age_bands_reported'])}."
        ),
        (
            "The primary run uses the IR/MR recodes; the PR runs use the PR "
            "household-member recode. The samples are therefore not nested. Their "
            "sex composition is shown in the sample table above."
        ),
    ]
    return notes


def _feature_differences(runs: Mapping[str, dict]) -> dict:
    primary_features = list(runs["primary"]["feature_columns"])
    pr_features = list(runs["pr_15_plus"]["feature_columns"])
    pr_15_49_features = list(runs["pr_15_49"]["feature_columns"])
    primary_set = set(primary_features)
    pr_set = set(pr_features)
    pr_15_49_set = set(pr_15_49_features)
    all_pr_set = pr_set | pr_15_49_set
    notes = [
        (
            "The primary model uses "
            f"{len(primary_features)} features; the PR 15+ model uses "
            f"{len(pr_features)}. The PR 15–49 model uses "
            f"{len(pr_15_49_features)}."
        ),
        (
            "The PR feature set is built to be usable for every adult with a valid "
            "BP pair: it excludes anthropometry, diet, marital status, children ever "
            "born, the smoking/smokeless split, alcohol frequency, state, religion, "
            "and social group."
        ),
    ]
    if pr_features == pr_15_49_features:
        notes.append(
            "The PR 15+ and PR 15–49 runs use exactly the same feature list; only the "
            "sample filter and age-band reporting differ."
        )
    else:
        notes.append(
            "The PR 15+ and PR 15–49 runs do not use the same feature list, so their "
            "difference is not a clean sample-extension comparison."
        )

    return {
        "shared_features": [
            feature
            for feature in primary_features
            if feature in pr_set and feature in pr_15_49_set
        ],
        "primary_only_features": [
            feature for feature in primary_features if feature not in all_pr_set
        ],
        "pr_only_features": list(
            dict.fromkeys(
                feature
                for feature in pr_features + pr_15_49_features
                if feature not in primary_set
            )
        ),
        "notes": notes,
    }


def _format_plain(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "n/a"
    return str(value)


def _delta_block(runs: Mapping[str, dict], block: str) -> dict:
    return {
        "pr_15_49_vs_primary": _metric_deltas(
            runs["pr_15_49"]["metrics"][block],
            runs["primary"]["metrics"][block],
        ),
        "pr_15_plus_vs_primary": _metric_deltas(
            runs["pr_15_plus"]["metrics"][block],
            runs["primary"]["metrics"][block],
        ),
        "pr_15_plus_vs_pr_15_49": _metric_deltas(
            runs["pr_15_plus"]["metrics"][block],
            runs["pr_15_49"]["metrics"][block],
        ),
    }


def compare_primary_vs_pr(
    primary: TrainingResult,
    pr_15_49: TrainingResult,
    pr_15_plus: TrainingResult,
) -> dict:
    """Return a structured comparison of the primary and two PR runs.

    The helper is deliberately tolerant of missing metrics: a run that did not
    produce a weighted metric, a reliability curve, or a subgroup block yields
    ``None`` or an empty list rather than an exception.
    """
    results = {
        "primary": primary,
        "pr_15_49": pr_15_49,
        "pr_15_plus": pr_15_plus,
    }
    runs = {key: _run_snapshot(key, results[key]) for key in RUN_KEYS}

    deltas = _delta_block(runs, "overall")
    weighted_deltas = _delta_block(runs, "weighted")

    comparison = {
        "comparison_order": list(RUN_KEYS),
        "labels": dict(RUN_LABELS),
        "runs": runs,
        "sample_table": [
            {
                "run": key,
                "label": RUN_LABELS[key],
                **runs[key]["sample_summary"],
            }
            for key in RUN_KEYS
        ],
        "feature_table": [
            {
                "run": key,
                "label": RUN_LABELS[key],
                "feature_count": runs[key]["feature_count"],
                "features": runs[key]["feature_columns"],
            }
            for key in RUN_KEYS
        ],
        "deltas": deltas,
        "weighted_deltas": weighted_deltas,
        "delta_direction": dict(DELTA_DIRECTIONS),
        "reliability_curves": {
            key: runs[key]["reliability_curve"] for key in RUN_KEYS
        },
        "sex_metrics": {key: runs[key]["subgroups"]["sex"] for key in RUN_KEYS},
        "age_band_metrics": {
            key: runs[key]["subgroups"]["age_band"] for key in RUN_KEYS
        },
        "subgroup_gaps": {
            "pr_15_49": [
                row
                for row in runs["pr_15_49"]["subgroups"]["all"]
                if not row.get("audit_only")
            ],
            "pr_15_plus": [
                row
                for row in runs["pr_15_plus"]["subgroups"]["all"]
                if not row.get("audit_only")
            ],
        },
        "sample_differences": _sample_differences(runs),
        "feature_differences": _feature_differences(runs),
        "narrative": list(MODEL_NOTES),
        "limitations": list(LIMITATIONS),
        "messages": {
            "overall": _OVERALL_NOTE,
            "weighted": _WEIGHTED_NOTE,
            "deltas": _DELTA_NOTE,
            "small_groups": _SMALL_GROUP_NOTE,
        },
    }
    return plain(comparison)


def _feature_list(features) -> str:
    return ", ".join(f"`{feature}`" for feature in features) or "—"


def _sample_table(comparison: Mapping[str, Any]) -> list[str]:
    lines = [
        (
            "| Run | Sample | Rows used | Rows after filters | Test rows | "
            "Test positives | Prevalence | Weighted prevalence | Female n | "
            "Male n | Female share |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["sample_table"]:
        sex = row["sex"]
        lines.append(
            f"| {row['label']} | {_format_plain(row['sample'])} | "
            f"{_format_plain(row['rows_used'])} | "
            f"{_format_plain(row['rows_after_filters'])} | "
            f"{_format_plain(row['test_rows'])} | "
            f"{_format_plain(row['test_positives'])} | "
            f"{_format_plain(row['prevalence'])} | "
            f"{_format_plain(row['weighted_prevalence'])} | "
            f"{_format_plain(sex['female'])} | "
            f"{_format_plain(sex['male'])} | "
            f"{_format_plain(sex['female_share'])} |"
        )
    return lines


def _age_composition_table(comparison: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Run | Age band | n | Test share | Prevalence |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison["sample_table"]:
        bands = row["age"]
        if not bands:
            lines.append(f"| {row['label']} | n/a | n/a | n/a | n/a |")
            continue
        for band in bands:
            lines.append(
                f"| {row['label']} | {_format_plain(band['group'])} | "
                f"{_format_plain(band['n'])} | {_format_plain(band['share'])} | "
                f"{_format_plain(band['prevalence'])} |"
            )
    return lines


def _feature_table(comparison: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Run | Feature count | Features |",
        "|---|---:|---|",
    ]
    for row in comparison["feature_table"]:
        lines.append(
            f"| {row['label']} | {row['feature_count']} | "
            f"{_feature_list(row['features'])} |"
        )
    return lines


def _overall_table(comparison: Mapping[str, Any], *, weighted: bool = False) -> list[str]:
    block = "weighted" if weighted else "overall"
    lines = [
        "| Metric | Primary IR/MR 15–49 | PR 15–49 | PR 15+ |",
        "|---|---:|---:|---:|",
    ]
    for label, key in METRIC_ROWS:
        cells = [
            (
                f"| {label} | "
                f"{format_value(comparison['runs']['primary']['metrics'][block].get(key))} |"
            )
        ]
        for run_key in ("pr_15_49", "pr_15_plus"):
            cells.append(
                f" {format_value(comparison['runs'][run_key]['metrics'][block].get(key))} |"
            )
        lines.append("".join(cells))
    lines.append(
        "| Calibrator | "
        f"{_format_plain(comparison['runs']['primary']['metrics']['calibrator'])} | "
        f"{_format_plain(comparison['runs']['pr_15_49']['metrics']['calibrator'])} | "
        f"{_format_plain(comparison['runs']['pr_15_plus']['metrics']['calibrator'])} |"
    )
    lines.append(
        "| Selected model | "
        f"{_format_plain(comparison['runs']['primary']['metrics']['selected_family'])} | "
        f"{_format_plain(comparison['runs']['pr_15_49']['metrics']['selected_family'])} | "
        f"{_format_plain(comparison['runs']['pr_15_plus']['metrics']['selected_family'])} |"
    )
    return lines


def _delta_table(comparison: Mapping[str, Any], *, weighted: bool = False) -> list[str]:
    deltas = comparison["weighted_deltas"] if weighted else comparison["deltas"]
    lines = [
        (
            "| Metric | PR 15–49 minus primary | PR 15+ minus primary | "
            "PR 15+ minus PR 15–49 |"
        ),
        "|---|---:|---:|---:|",
    ]
    for label, key in METRIC_ROWS:
        if key not in DELTA_METRICS:
            continue
        lines.append(
            f"| {label} | "
            f"{format_delta(deltas['pr_15_49_vs_primary'].get(key))} | "
            f"{format_delta(deltas['pr_15_plus_vs_primary'].get(key))} | "
            f"{format_delta(deltas['pr_15_plus_vs_pr_15_49'].get(key))} |"
        )
    return lines


def _curve_table(comparison: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Run | Bin lower | Bin upper | n | Mean predicted | Observed | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    wrote_row = False
    for run_key in RUN_KEYS:
        curve = comparison["reliability_curves"].get(run_key) or []
        if not curve:
            lines.append(
                f"| {RUN_LABELS[run_key]} | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        for row in curve:
            wrote_row = True
            lines.append(
                f"| {RUN_LABELS[run_key]} | {_format_plain(row.get('bin_lower'))} | "
                f"{_format_plain(row.get('bin_upper'))} | "
                f"{_format_plain(row.get('count'))} | "
                f"{_format_plain(row.get('mean_predicted'))} | "
                f"{_format_plain(row.get('observed'))} | "
                f"{_format_plain(row.get('gap'))} |"
            )
    if not wrote_row:
        lines.append("| All runs | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _subgroup_table(
    comparison: Mapping[str, Any],
    name: str,
) -> list[str]:
    lines = [
        (
            "| Run | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | "
            "Sensitivity | ECE | Reliable |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    wrote_row = False
    metrics_by_run = comparison[
        "sex_metrics" if name == "sex" else "age_band_metrics"
    ]
    for run_key in RUN_KEYS:
        rows = metrics_by_run[run_key]
        if not rows:
            lines.append(
                f"| {RUN_LABELS[run_key]} | n/a | n/a | n/a | n/a | n/a | n/a | "
                "n/a | n/a | n/a |"
            )
            continue
        for row in rows:
            wrote_row = True
            lines.append(
                f"| {RUN_LABELS[run_key]} | {_format_plain(row.get('group'))} | "
                f"{_format_plain(row.get('n'))} | "
                f"{_format_plain(row.get('positives'))} | "
                f"{_format_plain(row.get('prevalence'))} | "
                f"{_format_plain(row.get('roc_auc'))} | "
                f"{_format_plain(row.get('pr_auc'))} | "
                f"{_format_plain(row.get('sensitivity'))} | "
                f"{_format_plain(row.get('calibration_error'))} | "
                f"{'yes' if row.get('reliable') else 'no'} |"
            )
    if not wrote_row:
        lines.append(
            f"| All runs | {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )
    return lines


def _gap_table(comparison: Mapping[str, Any]) -> list[str]:
    lines = [
        (
            "| Run | Subgroup | Group | n | Positives | Prevalence | "
            "Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | Δ Sensitivity | Δ ECE | Reliable |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    wrote_row = False
    for run_key in ("pr_15_49", "pr_15_plus"):
        rows = comparison["subgroup_gaps"][run_key]
        if not rows:
            lines.append(
                f"| {RUN_LABELS[run_key]} | n/a | n/a | n/a | n/a | n/a | n/a | "
                "n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        for row in rows:
            wrote_row = True
            lines.append(
                f"| {RUN_LABELS[run_key]} | {_format_plain(row.get('subgroup'))} | "
                f"{_format_plain(row.get('group'))} | "
                f"{_format_plain(row.get('n'))} | "
                f"{_format_plain(row.get('positives'))} | "
                f"{_format_plain(row.get('prevalence'))} | "
                f"{_format_plain(row.get('gap_prevalence'))} | "
                f"{_format_plain(row.get('gap_roc_auc'))} | "
                f"{_format_plain(row.get('gap_pr_auc'))} | "
                f"{_format_plain(row.get('gap_sensitivity'))} | "
                f"{_format_plain(row.get('gap_calibration_error'))} | "
                f"{'yes' if row.get('reliable') else 'no'} |"
            )
    if not wrote_row:
        lines.append(
            "| PR models | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )
    return lines


def _run_heading(comparison: Mapping[str, Any], key: str) -> str:
    run = comparison["runs"][key]
    extras = []
    if run.get("sample"):
        extras.append(f"sample `{run['sample']}`")
    if run.get("run_id"):
        extras.append(f"run `{run['run_id']}`")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"- **{run['label']}**{suffix}: {run['feature_count']} features."


def render_primary_vs_pr_report(comparison: Mapping[str, Any]) -> str:
    """Render a structured primary-versus-PR comparison as readable Markdown."""
    feature_notes = comparison["feature_differences"]["notes"]
    lines = [
        "# Primary vs PR sensitivity model comparison",
        "",
        (
            "This report compares three runs on the same reporting axes. It does not "
            "rank them."
        ),
        "",
        *[_run_heading(comparison, key) for key in RUN_KEYS],
        "",
        (
            "> **Do not compare raw probabilities.** The samples and feature sets "
            "differ. The comparison is about discrimination, calibration, threshold "
            "behaviour, and subgroup gaps."
        ),
        "",
        "## What the three runs are",
        "",
        *[f"- {note}" for note in comparison["narrative"]],
        "",
        "## Sample differences",
        "",
        *_sample_table(comparison),
        "",
        (
            "Rows used are the filtered run rows; test rows are the rows that produced "
            "the metrics. Composition and prevalence are reported on the test set, "
            "because that is the set every metric is computed on."
        ),
        "",
        "### Age composition",
        "",
        *_age_composition_table(comparison),
        "",
        *[f"- {note}" for note in comparison["sample_differences"]],
        "",
        "## Feature-set differences",
        "",
        *_feature_table(comparison),
        "",
        *[f"- {note}" for note in feature_notes],
        "",
        "### Features present only in the primary model",
        "",
        (
            _feature_list(comparison["feature_differences"]["primary_only_features"])
            or "—"
        ),
        "",
        "### Features present only in the PR models",
        "",
        _feature_list(comparison["feature_differences"]["pr_only_features"]) or "—",
        "",
        "## Overall discrimination, calibration, and threshold metrics",
        "",
        f"{comparison['messages']['overall']}",
        "",
        *_overall_table(comparison),
        "",
        "## Survey-weighted overall metrics",
        "",
        f"{comparison['messages']['weighted']}",
        "",
        *_overall_table(comparison, weighted=True),
        "",
        "## Differences between runs",
        "",
        f"{comparison['messages']['deltas']}",
        "",
        *_delta_table(comparison),
        "",
        "## Survey-weighted differences between runs",
        "",
        *_delta_table(comparison, weighted=True),
        "",
        "## Reliability curves",
        "",
        *_curve_table(comparison),
        "",
        "## Sex-specific metrics",
        "",
        *_subgroup_table(comparison, "sex"),
        "",
        "## Age-band-specific metrics",
        "",
        *_subgroup_table(comparison, "age_band"),
        "",
        "## Subgroup gaps for the PR models",
        "",
        (
            "Each row reports the gap versus that run's own overall test set. The "
            "primary model's comparable subgroup table is in its own results report."
        ),
        "",
        *_gap_table(comparison),
        "",
        f"{comparison['messages']['small_groups']}",
        "",
        "## Limitations",
        "",
        *[f"- {note}" for note in comparison["limitations"]],
        "",
        (
            "The PR model is a sensitivity analysis. The primary IR/MR model remains "
            "the model intended for the app."
        ),
        "",
    ]
    return "\n".join(lines)


def write_primary_vs_pr_report(
    comparison: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "primary_vs_pr_comparison",
) -> dict[str, Path]:
    """Write the structured JSON and the readable Markdown report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(plain(comparison), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_primary_vs_pr_report(comparison),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"run artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"run artifact is not a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"run artifact is missing: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"run artifact is empty: {path}") from None


def load_run_result(run_dir: str | Path) -> TrainingResult:
    """Load one finished pipeline run from its artifact directory."""
    directory = Path(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {directory}")

    resolved = _read_json(directory / "resolved_config.json")
    metadata = _read_json(directory / "run_metadata.json")
    metrics = _read_json(directory / "metrics.json")
    subgroup_metrics = _read_csv(directory / "subgroup_metrics.csv")
    calibration_curve = _read_csv(directory / "calibration_curve.csv")

    config = TrainingConfig(
        name=resolved.get("name") or metadata.get("name") or directory.name,
        feature_columns=tuple(resolved.get("feature_columns") or ()),
        metadata=resolved.get("metadata") or {},
        state_min_test_rows=resolved.get("state_min_test_rows", 500),
        small_group_min_rows=resolved.get("small_group_min_rows", 200),
        small_group_min_positives=resolved.get("small_group_min_positives", 50),
    )
    artifacts = {
        name: directory / filename
        for name, filename in {
            "resolved_config": "resolved_config.json",
            "run_metadata": "run_metadata.json",
            "metrics": "metrics.json",
            "subgroup_metrics": "subgroup_metrics.csv",
            "calibration_curve": "calibration_curve.csv",
        }.items()
    }
    return TrainingResult(
        config=config,
        split_assignment=pd.DataFrame(),
        model=None,
        metrics=metrics,
        subgroup_metrics=subgroup_metrics,
        calibration_curve=calibration_curve,
        threshold_analysis=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        shap_summary=pd.DataFrame(),
        run_metadata=metadata,
        artifacts={name: path for name, path in artifacts.items() if path.exists()},
        run_dir=directory,
    )


__all__ = [
    "COMPARISON_METRICS",
    "DELTA_METRICS",
    "RUN_KEYS",
    "RUN_LABELS",
    "compare_primary_vs_pr",
    "load_run_result",
    "render_primary_vs_pr_report",
    "write_primary_vs_pr_report",
]
