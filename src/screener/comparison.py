"""The fixed candidate comparison: scoring, ranking, and comparison artifacts.

The comparison consumes development-only selection evidence. It never receives
the test split, so ranking, calibration, and thresholds cannot be influenced by
the rows reserved for the final evaluation.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .pipeline import (
    SELECTION_SPLIT_LABEL,
    Candidate,
    SelectionEvidence,
    SelectionRecord,
    TrainingConfig,
    evaluate_candidates,
)
from .primary_model import primary_15_49_config

# Pre-registered rule, fixed before any comparison score is examined.
BRIER_MARGIN = 0.005
CALIBRATION_SLOPE_RANGE = (0.8, 1.2)
CALIBRATION_INTERCEPT_RANGE = (-0.1, 0.1)
SUBGROUP_SENSITIVITY_MARGIN = 0.15
# Age-band sensitivity is handled by the per-band operating points, not by the
# gap veto; a single cut-off cannot equalise sensitivity across an age gradient.
SUBGROUP_VETO_EXCLUDED = frozenset({"age_band"})
BUNDLE_PR_AUC_MARGIN = 0.005
BUNDLE_BRIER_MARGIN = 0.005

# Prefer a transportable regression model when it is effectively tied with a
# tree ensemble. The margin is part of the amended selection rule.
SIMPLICITY_PR_AUC_MARGIN = 0.01
SIMPLICITY_FAMILIES = ("logistic_spline", "logistic")

PREVALENCE_LABEL = "B0-prevalence"

COMPARISON_FAMILIES = (
    ("C1-logistic", "logistic"),
    ("C2-logistic-spline", "logistic_spline"),
    ("C3-random-forest", "random_forest"),
    ("C4-lightgbm", "lightgbm"),
    ("C5-xgboost", "xgboost"),
    ("C6-catboost", "catboost"),
    ("C7-lightgbm-monotone", "lightgbm_monotone"),
)

# Three settings per family: the same bounded tuning budget for every candidate.
TUNING_SETTINGS: dict[str, tuple[dict, ...]] = {
    "logistic": ({"C": 0.1}, {"C": 1.0}, {"C": 10.0}),
    "logistic_spline": ({"C": 0.1}, {"C": 1.0}, {"C": 10.0}),
    "random_forest": (
        {"n_estimators": 300, "min_samples_leaf": 1},
        {"n_estimators": 300, "min_samples_leaf": 5},
        {"n_estimators": 300, "min_samples_leaf": 20},
    ),
    "lightgbm": (
        {"num_leaves": 15, "max_depth": 4},
        {"num_leaves": 31, "max_depth": 6},
        {"num_leaves": 7, "max_depth": 3},
    ),
    "xgboost": ({"max_depth": 4}, {"max_depth": 6}, {"max_depth": 3}),
    "catboost": ({"depth": 4}, {"depth": 6}, {"depth": 3}),
    "lightgbm_monotone": (
        {"num_leaves": 15, "max_depth": 4},
        {"num_leaves": 31, "max_depth": 6},
        {"num_leaves": 7, "max_depth": 3},
    ),
}

PREREGISTERED_RULE = {
    "ranking": "validation PR-AUC, ROC-AUC as a secondary check",
    "brier_margin": BRIER_MARGIN,
    "calibration_slope_range": list(CALIBRATION_SLOPE_RANGE),
    "calibration_intercept_range": list(CALIBRATION_INTERCEPT_RANGE),
    "subgroup_sensitivity_margin": SUBGROUP_SENSITIVITY_MARGIN,
    "subgroup_sensitivity_excluded": sorted(SUBGROUP_VETO_EXCLUDED),
    "simplicity_pr_auc_margin": SIMPLICITY_PR_AUC_MARGIN,
    "simplicity_families": list(SIMPLICITY_FAMILIES),
    "bundle_pr_auc_margin": BUNDLE_PR_AUC_MARGIN,
    "bundle_brier_margin": BUNDLE_BRIER_MARGIN,
    "tuning_budget": "three settings per family",
    "reference": "B0 prevalence-only is unscored and cannot win",
    "amendment": (
        "Age-band sensitivity is set by age-band thresholds; the subgroup gap "
        "veto applies to sex, wealth, residence, education, and state at 0.15. "
        "A near-tied regression model is preferred over a tree ensemble within "
        "0.01 PR-AUC."
    ),
    "amendment_basis": (
        "The first pass of the 0.10 all-subgroup veto vetoed every candidate "
        "because age-band sensitivity is structurally unequal at one global "
        "cut-off. This amended rule was fixed before the rerun; the first-pass "
        "artifacts are kept locally for audit."
    ),
}


def comparison_candidates() -> tuple[Candidate, ...]:
    """The seven scored families, three settings each, plus the reference."""
    candidates = [Candidate(label=PREVALENCE_LABEL, family="prevalence")]
    for code, family in COMPARISON_FAMILIES:
        for index, params in enumerate(TUNING_SETTINGS[family], start=1):
            candidates.append(
                Candidate(
                    label=f"{code}[{index}]",
                    family=family,
                    params=params,
                )
            )
    return tuple(candidates)


def comparison_config(*, feature_set: str, **primary_kwargs) -> TrainingConfig:
    """The primary config with the pre-registered candidate list."""
    config = primary_15_49_config(feature_set=feature_set, **primary_kwargs)
    return replace(
        config,
        candidates=comparison_candidates(),
        metadata={
            **config.metadata,
            "feature_set": feature_set,
            "run_role": "comparison",
            "tuning_budget": PREREGISTERED_RULE["tuning_budget"],
        },
    )


def reliable_sensitivity_gap(subgroups, *, exclude=()) -> float | None:
    """The largest absolute sensitivity gap among reliable subgroup rows."""
    gaps = [
        abs(row["gap_sensitivity"])
        for row in subgroups
        if row.get("reliable")
        and row.get("gap_sensitivity") is not None
        and row.get("subgroup") not in exclude
    ]
    return max(gaps) if gaps else None


def veto_reasons(
    metrics: dict,
    *,
    best_brier: float | None,
    sensitivity_gap: float | None,
) -> list[str]:
    """The pre-registered vetoes for one candidate's out-of-fold metrics."""
    reasons = []
    brier = metrics.get("brier")
    if (
        brier is not None
        and best_brier is not None
        and brier - best_brier > BRIER_MARGIN
    ):
        reasons.append(
            f"brier {brier:.4f} is more than {BRIER_MARGIN} worse than "
            f"the best {best_brier:.4f}"
        )
    slope = metrics.get("calibration_slope")
    low, high = CALIBRATION_SLOPE_RANGE
    if slope is None or not low <= slope <= high:
        reasons.append(f"calibration slope {slope} is outside {low}-{high}")
    intercept = metrics.get("calibration_intercept")
    low, high = CALIBRATION_INTERCEPT_RANGE
    if intercept is None or not low <= intercept <= high:
        reasons.append(f"calibration intercept {intercept} is outside {low}-{high}")
    if (
        sensitivity_gap is not None
        and sensitivity_gap > SUBGROUP_SENSITIVITY_MARGIN
    ):
        reasons.append(
            f"reliable subgroup sensitivity gap {sensitivity_gap:.4f} exceeds "
            f"{SUBGROUP_SENSITIVITY_MARGIN}"
        )
    return reasons


def _rank_key(record: dict) -> tuple:
    metrics = record["oof_metrics"]
    pr_auc = metrics.get("pr_auc")
    roc_auc = metrics.get("roc_auc")
    return (
        pr_auc is not None,
        pr_auc if pr_auc is not None else -1.0,
        roc_auc if roc_auc is not None else -1.0,
    )


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """Mark vetoes and rank the scored candidates by out-of-fold PR-AUC."""
    scored = [item for item in candidates if item["family"] != "prevalence"]
    briers = [
        item["oof_metrics"].get("brier")
        for item in scored
        if item["oof_metrics"].get("brier") is not None
    ]
    best_brier = min(briers) if briers else None
    for item in scored:
        item["vetoes"] = veto_reasons(
            item["oof_metrics"],
            best_brier=best_brier,
            sensitivity_gap=item.get(
                "max_reliable_veto_sensitivity_gap",
                item["max_reliable_sensitivity_gap"],
            ),
        )
    ranked = sorted(scored, key=_rank_key, reverse=True)
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def select_winner(eligible: list[dict]) -> tuple[dict, str]:
    """Pick the non-vetoed winner, preferring a near-tied regression model.

    A tree ensemble is only preferred when it beats the best eligible
    regression model by more than the pre-registered simplicity margin.
    """
    best = max(eligible, key=_rank_key)
    if best["family"] in SIMPLICITY_FAMILIES:
        return best, "ranking"
    simple = [item for item in eligible if item["family"] in SIMPLICITY_FAMILIES]
    if not simple:
        return best, "ranking"
    candidate = max(simple, key=_rank_key)
    pr_auc = candidate["oof_metrics"].get("pr_auc")
    best_pr_auc = best["oof_metrics"].get("pr_auc")
    brier = candidate["oof_metrics"].get("brier")
    best_brier = best["oof_metrics"].get("brier")
    if (
        pr_auc is not None
        and best_pr_auc is not None
        and brier is not None
        and best_brier is not None
        and pr_auc >= best_pr_auc - SIMPLICITY_PR_AUC_MARGIN
        and brier <= best_brier + BRIER_MARGIN
    ):
        return candidate, "simplicity_preference"
    return best, "ranking"


def candidate_artifact(record: SelectionRecord) -> dict:
    """One comparison record: out-of-fold evidence only, never test metrics."""
    return {
        "label": record.label,
        "family": record.family,
        "params": record.params,
        "rank": None,
        "vetoes": [],
        "oof_metrics": record.metrics,
        "calibration": record.calibration_report,
        "threshold": record.threshold_report,
        "subgroups": record.subgroups,
        "reliability_curve": record.reliability_curve,
        "net_benefit": record.net_benefit,
        "fold_metrics": record.fold_metrics,
        "max_reliable_sensitivity_gap": reliable_sensitivity_gap(record.subgroups),
        "max_reliable_veto_sensitivity_gap": reliable_sensitivity_gap(
            record.subgroups, exclude=SUBGROUP_VETO_EXCLUDED
        ),
    }


def _family_tuning(item: dict) -> dict:
    family = item["family"]
    settings = [
        {
            "label": f"{code}[{index}]",
            "params": params,
        }
        for code, family_name in COMPARISON_FAMILIES
        if family_name == family
        for index, params in enumerate(TUNING_SETTINGS[family], start=1)
    ]
    return {
        "trial_count": len(settings),
        "selected": {"label": item["label"], "params": item["params"]},
        "settings": settings,
    }


def build_comparison_artifact(
    evidence: SelectionEvidence,
    config: TrainingConfig,
    *,
    bundle: str | None = None,
    smoke: bool = False,
) -> dict:
    """Assemble the comparison artifact from development-only evidence."""
    candidates = [candidate_artifact(record) for record in evidence.records]
    ranked = rank_candidates(candidates)
    eligible = [item for item in ranked if not item["vetoes"]]
    if eligible:
        winner, selection_reason = select_winner(eligible)
    else:
        winner, selection_reason = None, "no_eligible_candidate"
    for item in candidates:
        if item["family"] != "prevalence":
            item["tuning"] = _family_tuning(item)

    bundle_name = bundle or evidence.feature_set
    return {
        "schema_version": 1,
        "bundle": bundle_name,
        "feature_set": evidence.feature_set,
        "feature_count": evidence.feature_count,
        "selection_split": SELECTION_SPLIT_LABEL,
        "selection_rows": evidence.development_rows,
        "fold_count": evidence.fold_count,
        "test_evaluated_once": False,
        "smoke": smoke,
        "rule": PREREGISTERED_RULE,
        "candidates": candidates,
        "candidate_labels": [item["label"] for item in candidates],
        "ranking": [item["label"] for item in ranked],
        "vetoes": {
            item["label"]: item["vetoes"]
            for item in candidates
            if item["vetoes"]
        },
        "winner": None
        if winner is None
        else {
            "label": winner["label"],
            "family": winner["family"],
            "params": winner["params"],
            "bundle": bundle_name,
            "pr_auc": winner["oof_metrics"].get("pr_auc"),
            "brier": winner["oof_metrics"].get("brier"),
            "selection_reason": selection_reason,
            "top_ranked_label": ranked[0]["label"] if ranked else None,
            "pr_auc_margin_vs_top": (
                None
                if not ranked
                else (ranked[0]["oof_metrics"].get("pr_auc") or 0.0)
                - (winner["oof_metrics"].get("pr_auc") or 0.0)
            ),
            "final_run": None,
        },
        "config_name": config.name,
    }


def choose_bundle(bundles: list[dict]) -> dict:
    """The pre-registered feature-bundle rule.

    Each entry names a bundle, its feature count, and its ranking-rule winner
    metrics (``pr_auc`` and ``brier``). A bundle is eligible only when its
    winner passes every veto.
    """
    eligible = [item for item in bundles if item.get("eligible") and item["winner"]]
    if not eligible:
        raise ValueError(
            "no feature bundle has a winner that passes every ranking veto; "
            "the comparison has no deployable winner"
        )
    reference = max(eligible, key=lambda item: item["winner"]["pr_auc"])
    reference_metrics = reference["winner"]
    within = [
        item
        for item in eligible
        if item["winner"]["pr_auc"]
        >= reference_metrics["pr_auc"] - BUNDLE_PR_AUC_MARGIN
        and item["winner"]["brier"] <= reference_metrics["brier"] + BUNDLE_BRIER_MARGIN
    ]
    return min(
        within,
        key=lambda item: (
            item["feature_count"],
            -item["winner"]["pr_auc"],
            item["winner"]["brier"],
        ),
    )


ALLOWED_TEST_KEYS = {"test_evaluated_once", "test_metrics"}


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def assert_no_test_metrics(artifact: dict) -> None:
    """Refuse a comparison artifact that carries test evidence for a non-winner."""
    winner = artifact.get("winner") or {}
    winner_label = winner.get("label")
    for candidate in artifact.get("candidates", []):
        test_keys = sorted(
            {key for key in _keys(candidate) if "test" in key.lower()}
        )
        if test_keys and candidate.get("label") != winner_label:
            raise ValueError(
                f"non-winner candidate '{candidate.get('label')}' carries "
                "test-derived keys: " + ", ".join(test_keys)
            )
        if test_keys:
            raise ValueError(
                "test metrics belong in the winner record, not a candidate record"
            )

    unexpected = sorted(
        {key for key in _keys(artifact) if "test" in key.lower()}
        - ALLOWED_TEST_KEYS
    )
    if unexpected:
        raise ValueError(
            "comparison artifact carries test-derived keys: "
            + ", ".join(unexpected)
        )


def attach_winner_test_metrics(
    artifact: dict,
    *,
    final_run: str,
    test_metrics: dict,
) -> dict:
    """Point the winner at its final run and attach that run's test metrics."""
    if artifact.get("winner") is None:
        raise ValueError("the comparison has no winner to attach test metrics to")
    assert_no_test_metrics(artifact)
    artifact["winner"]["final_run"] = final_run
    artifact["winner"]["test_metrics"] = test_metrics
    artifact["test_evaluated_once"] = True
    assert_no_test_metrics(artifact)
    return artifact


def _net_benefit(item: dict, threshold: float = 0.10):
    for row in item.get("net_benefit", []):
        if abs(float(row.get("threshold", -1)) - threshold) < 1e-9:
            return row.get("net_benefit")
    return None


def _metric_row(item: dict) -> str:
    metrics = item["oof_metrics"]
    veto = "; ".join(item["vetoes"]) or "none"
    gap = item.get("max_reliable_veto_sensitivity_gap")
    return (
        f"| {item['rank']} | {item['label']} | {item['family']} | "
        f"{_number(metrics.get('pr_auc'))} | {_number(metrics.get('roc_auc'))} | "
        f"{_number(metrics.get('brier'))} | "
        f"{_number(metrics.get('calibration_slope'))} | "
        f"{_number(metrics.get('calibration_intercept'))} | "
        f"{_number(metrics.get('sensitivity'))} | "
        f"{_number(metrics.get('specificity'))} | "
        f"{_number(gap)} | {_number(_net_benefit(item))} | "
        f"{item.get('calibration', {}).get('selected', 'n/a')} | {veto} |"
    )


def _number(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _exploration_lines(artifact: dict) -> list[str]:
    experiments = artifact.get("exploration") or {}
    if not experiments:
        return []
    lines = [
        "",
        "## Exploration",
        "",
        (
            "Post-freeze experiments are reported for transparency. Exploration "
            "does not change the deployed artifact unless promoted; a "
            "non-promoted result changes no winner or test block."
        ),
        "",
    ]
    for name, result in experiments.items():
        decision = result.get("decision") or {}
        conditions = decision.get("conditions") or {}
        interval = result.get("delta_roc_auc_ci")
        interval_text = (
            " (95% CI n/a)"
            if not interval
            else f" (95% CI {_number(interval[0])} to {_number(interval[1])})"
        )
        failed = [condition for condition, held in conditions.items() if not held]
        lines += [
            f"### {name}",
            "",
            f"- Experiment: `{result.get('experiment', 'unknown')}`",
            f"- Promoted: {'yes' if decision.get('promoted') else 'no'}",
            (
                f"- Δ ROC-AUC: {_number(result.get('delta_roc_auc'))}"
                f"{interval_text}"
            ),
            f"- Δ Brier: {_number(result.get('delta_brier'))}",
            (
                "- Max reliable subgroup sensitivity gap: "
                f"{_number(result.get('max_reliable_sensitivity_gap'))}"
            ),
            (
                "- Failed conditions: "
                + (", ".join(failed) if failed else "none")
            ),
            "",
        ]
    return lines


def render_comparison(artifact: dict) -> str:
    """The human-readable comparison report."""
    winner = artifact["winner"]
    lines = [
        f"# Model comparison: {artifact['bundle']}",
        "",
        "Selection used out-of-fold predictions on the development split only.",
        "The test split was untouched during selection and is evaluated once for",
        "the chosen winner; non-winner records contain no test metrics.",
        "",
        (
            f"- Feature set: `{artifact['feature_set']}` "
            f"({artifact['feature_count']} features)"
        ),
        f"- Selection rows: {artifact['selection_rows']}",
        f"- Folds: {artifact['fold_count']}",
        f"- Test evaluated once: {str(artifact['test_evaluated_once']).lower()}",
        f"- Winner: {winner['label'] if winner else 'none (no eligible candidate)'}"
        + (
            f" (selected by `{winner['selection_reason']}`)"
            if winner and winner.get("selection_reason")
            else ""
        ),
        "",
        "## Ranking rule",
        "",
        f"- Rank by {PREREGISTERED_RULE['ranking']}.",
        f"- Veto when Brier is more than {BRIER_MARGIN} worse than the best.",
        (
            f"- Veto when calibration slope is outside "
            f"{CALIBRATION_SLOPE_RANGE[0]}–{CALIBRATION_SLOPE_RANGE[1]} or "
            f"intercept outside {CALIBRATION_INTERCEPT_RANGE[0]}–"
            f"{CALIBRATION_INTERCEPT_RANGE[1]}."
        ),
        (
            f"- Veto when a reliable subgroup sensitivity gap exceeds "
            f"{SUBGROUP_SENSITIVITY_MARGIN}."
        ),
        (
            "- Age-band sensitivity is set by age-band thresholds and is not part "
            "of the gap veto; the veto applies to sex, wealth, residence, "
            "education, and state."
        ),
        (
            f"- Prefer logistic or spline logistic over a tree ensemble when it is "
            f"within {SIMPLICITY_PR_AUC_MARGIN} PR-AUC and no more than "
            f"{BRIER_MARGIN} worse Brier."
        ),
        f"- Tuning budget: {PREREGISTERED_RULE['tuning_budget']}.",
        "",
        "## Candidates",
        "",
        (
            "| Rank | Candidate | Family | PR-AUC | ROC-AUC | Brier | Slope | "
            "Intercept | Sensitivity | Specificity | Veto subgroup gap | "
            "Net benefit @0.10 | Calibrator | Vetoes |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    ranked = sorted(
        (item for item in artifact["candidates"] if item["rank"] is not None),
        key=lambda item: item["rank"],
    )
    lines.extend(_metric_row(item) for item in ranked)

    lines.extend(
        [
            "",
            "## Tuning record",
            "",
            "| Family | Selected setting | Parameters | Trials |",
            "|---|---|---|---:|",
        ]
    )
    seen = set()
    ranked_by_family = sorted(
        (item for item in artifact["candidates"] if item["rank"] is not None),
        key=lambda item: item["rank"],
    )
    for item in ranked_by_family:
        tuning = item.get("tuning")
        if not tuning or item["family"] in seen:
            continue
        seen.add(item["family"])
        selected = tuning["selected"]
        lines.append(
            f"| {item['family']} | {selected['label']} | "
            f"`{json.dumps(selected['params'], sort_keys=True)}` | "
            f"{tuning['trial_count']} |"
        )
    lines.extend(_exploration_lines(artifact))
    if winner and winner.get("final_run"):
        lines.extend(
            [
                "",
                f"Final run: `{winner['final_run']}`",
                "",
                "The test split was evaluated once for the chosen winner; no",
                "non-winner record carries test metrics.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_comparison_artifacts(artifact: dict, output_dir: str | Path) -> dict:
    """Write ``comparison.json`` and ``comparison.md`` under ``output_dir``."""
    assert_no_test_metrics(artifact)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "comparison.json"
    json_path.write_text(
        json.dumps(artifact, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path = output / "comparison.md"
    markdown_path.write_text(render_comparison(artifact), encoding="utf-8")
    return {"comparison_json": json_path, "comparison_markdown": markdown_path}


def run_comparison(
    development: pd.DataFrame,
    config: TrainingConfig,
    *,
    output_dir: str | Path,
    bundle: str | None = None,
    smoke: bool = False,
) -> dict:
    """Score the configured candidates on development rows and write both artifacts."""
    evidence = evaluate_candidates(development, config)
    artifact = build_comparison_artifact(
        evidence,
        config,
        bundle=bundle,
        smoke=smoke,
    )
    write_comparison_artifacts(artifact, output_dir)
    return artifact


__all__ = [
    "BRIER_MARGIN",
    "BUNDLE_BRIER_MARGIN",
    "BUNDLE_PR_AUC_MARGIN",
    "COMPARISON_FAMILIES",
    "PREREGISTERED_RULE",
    "PREVALENCE_LABEL",
    "SUBGROUP_SENSITIVITY_MARGIN",
    "TUNING_SETTINGS",
    "assert_no_test_metrics",
    "attach_winner_test_metrics",
    "build_comparison_artifact",
    "choose_bundle",
    "comparison_candidates",
    "comparison_config",
    "rank_candidates",
    "render_comparison",
    "run_comparison",
    "veto_reasons",
    "write_comparison_artifacts",
]
