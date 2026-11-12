"""One reproducible training pipeline for the primary and PR models.

``run_training_pipeline`` is the public modeling entry point, and
``prepare_training_data`` exposes its split seam. The primary model and the PR
sensitivity model differ only in their :class:`TrainingConfig`: sample filters,
feature set, subgroup columns, and the split assignment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import (
    IdentityCalibrator,
    fit_calibrator,
    select_calibrator,
    select_threshold,
    threshold_analysis,
)
from .feature_policy import validate_feature_frame
from .metrics import (
    binary_metrics,
    bootstrap_confidence_intervals,
    net_benefit_table,
    reliability_curve,
)
from .model_bundle import ModelBundle, save_model_bundle
from .modeling import (
    build_model,
    fit_model,
    inference_matrix,
    model_matrix,
    predict_positive_probability,
)
from .splits import (
    DEVELOPMENT_SPLIT,
    TEST_SPLIT,
    assert_no_group_leakage,
    assign_splits,
    load_split_assignment,
    split_source,
    summarise_splits,
    validate_split_assignment,
)

_TARGET = "target"
SELECTION_SPLIT_LABEL = "development_oof"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value):
    """Recursively convert numpy and pandas values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _require_columns(frame: pd.DataFrame, columns, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


@dataclass(frozen=True)
class SampleFilter:
    """One declarative row filter recorded in the resolved config."""

    column: str
    minimum: float | None = None
    maximum: float | None = None
    values: tuple = ()
    require_notna: bool = False

    def apply(self, frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        column = frame[self.column]
        if self.require_notna:
            mask &= column.notna()
        if self.values:
            mask &= column.isin(list(self.values))
        if self.minimum is not None or self.maximum is not None:
            numeric = pd.to_numeric(column, errors="coerce")
            mask &= numeric.between(
                -np.inf if self.minimum is None else self.minimum,
                np.inf if self.maximum is None else self.maximum,
            )
        return mask

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "values": list(self.values),
            "require_notna": self.require_notna,
        }


@dataclass(frozen=True)
class SubgroupSpec:
    """A reporting subgroup, optionally derived from a binned column."""

    name: str
    column: str
    bins: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()
    value_labels: tuple[tuple[Any, str], ...] = ()
    minimum_test_rows: int = 0
    audit_only: bool = False
    note: str = ""

    def group_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.bins:
            values = pd.cut(
                pd.to_numeric(frame[self.column], errors="coerce"),
                bins=list(self.bins),
                labels=list(self.labels),
                right=False,
            )
            return values.astype("string").fillna("missing").astype(str)
        series = frame[self.column]
        if self.value_labels:
            mapping = dict(self.value_labels)
            return series.map(mapping).astype("string").fillna("missing").astype(str)
        return series.astype("string").fillna("missing").astype(str)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "column": self.column,
            "bins": list(self.bins),
            "labels": list(self.labels),
            "value_labels": [[value, label] for value, label in self.value_labels],
            "minimum_test_rows": self.minimum_test_rows,
            "audit_only": self.audit_only,
            "note": self.note,
        }


@dataclass(frozen=True)
class Candidate:
    """One configured model candidate: family, parameters, and a stable label."""

    label: str
    family: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "family": self.family,
            "params": self.params,
        }


@dataclass(frozen=True)
class SelectionRecord:
    """Out-of-fold evidence for one scored candidate on the development folds."""

    label: str
    family: str
    params: dict
    tuning_record: dict
    feature_set: str
    feature_count: int
    oof_scores: np.ndarray
    oof_probabilities: np.ndarray
    target: np.ndarray
    covered: np.ndarray
    fold: np.ndarray
    calibration_method: str
    calibration_report: dict
    threshold_report: dict
    metrics: dict
    subgroups: list[dict]
    reliability_curve: list[dict]
    net_benefit: list[dict]
    fold_metrics: list[dict]


@dataclass(frozen=True)
class SelectionEvidence:
    """The development-only selection result: one record per candidate."""

    feature_set: str
    feature_count: int
    development_rows: int
    fold_count: int
    records: list[SelectionRecord]
    fit_handle: FitHandle

    def record(self, label: str) -> SelectionRecord:
        """Return the evidence record for one scored candidate."""
        return self.fit_handle.record(label)

    def fit(self, label: str) -> ModelBundle:
        """Refit one scored candidate on all development rows."""
        return self.fit_handle.fit(label)


@dataclass(frozen=True)
class FitHandle:
    """Refits any scored candidate on all development rows.

    This is the only bridge from selection evidence to final evaluation, so a
    candidate fitted here is exactly the one the evidence describes.
    """

    development: pd.DataFrame = field(repr=False)
    config: TrainingConfig = field(repr=False)
    records: tuple[SelectionRecord, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(record.label for record in self.records)

    def record(self, label: str) -> SelectionRecord:
        for record in self.records:
            if record.label == label:
                return record
        raise ValueError(
            f"unknown candidate '{label}'; scored candidates are: "
            + ", ".join(self.names())
        )

    def out_of_fold(
        self,
        label: str,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return raw out-of-fold scores for one candidate, optionally reseeded."""
        record = self.record(label)
        trial = _cross_validate_candidate(
            self.development,
            self.config,
            Candidate(
                label=record.label,
                family=record.family,
                params=record.params,
            ),
            seed=seed,
        )
        return trial["oof"], trial["covered"]

    def fit(self, label: str, *, seed: int | None = None) -> ModelBundle:
        """Fit the named candidate on every development row and calibrate it."""
        record = self.record(label)
        model_seed = self.config.seed if seed is None else int(seed)
        matrix, levels = model_matrix(
            self.development,
            self.config.feature_columns,
            self.config.categorical_columns,
        )
        model = build_model(
            record.family,
            record.params,
            feature_columns=self.config.feature_columns,
            categorical_columns=self.config.categorical_columns,
            seed=model_seed,
        )
        fit_model(
            model,
            matrix,
            self.development[self.config.target_column].to_numpy(),
            family=record.family,
            categorical_columns=self.config.categorical_columns,
        )

        covered = record.covered
        method = record.calibration_method
        if method == "identity":
            calibrator = IdentityCalibrator()
        else:
            calibrator = fit_calibrator(
                record.target[covered],
                record.oof_scores[covered],
                method,
            )
        return ModelBundle(
            model=model,
            calibrator=calibrator,
            feature_columns=self.config.feature_columns,
            categorical_columns=self.config.categorical_columns,
            category_levels=levels,
            threshold=record.threshold_report["threshold"],
            target_definition=self.config.target_definition,
            metadata={
                "candidate_label": record.label,
                "model_family": record.family,
                "params": record.params,
                "feature_set": record.feature_set,
                "feature_count": record.feature_count,
                "calibration_method": record.calibration_method,
                "seed": model_seed,
                "target_sensitivity": self.config.target_sensitivity,
            },
            model_family=record.family,
        )


@dataclass
class TrainingConfig:
    """Everything that distinguishes one run from another."""

    name: str = "model"
    feature_table: str | None = None
    feature_columns: tuple[str, ...] = ()
    target_column: str = _TARGET
    secondary_target_column: str | None = "target_fact_sheet"
    target_definition: str = "glucose > 140 OR diagnosed OR medication"
    group_column: str = "cluster"
    household_column: str = "household"
    weight_column: str | None = "weight_scaled"
    sample_filters: tuple[SampleFilter, ...] = ()
    subgroups: tuple[SubgroupSpec, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    seed: int = 42
    test_fraction: float = 0.2
    n_folds: int = 5
    target_sensitivity: float = 0.8
    bootstrap_replicates: int = 1_000
    shap_sample_size: int = 5_000
    reliability_bins: int = 10
    small_group_min_rows: int = 200
    small_group_min_positives: int = 50
    state_min_test_rows: int = 500
    model_params: dict[str, Any] = field(default_factory=dict)
    candidates: tuple[Candidate, ...] = ()
    pinned_candidate: Candidate | None = None
    calibration_methods: tuple[str, ...] = ("isotonic", "platt")
    split_assignment: pd.DataFrame | None = field(default=None, repr=False)
    split_assignment_path: str | None = None
    smoke_rows: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "feature_table": self.feature_table,
            "feature_columns": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "target_column": self.target_column,
            "secondary_target_column": self.secondary_target_column,
            "target_definition": self.target_definition,
            "group_column": self.group_column,
            "household_column": self.household_column,
            "weight_column": self.weight_column,
            "sample_filters": [item.to_dict() for item in self.sample_filters],
            "subgroups": [item.to_dict() for item in self.subgroups],
            "categorical_columns": list(self.categorical_columns),
            "seed": self.seed,
            "test_fraction": self.test_fraction,
            "n_folds": self.n_folds,
            "target_sensitivity": self.target_sensitivity,
            "bootstrap_replicates": self.bootstrap_replicates,
            "shap_sample_size": self.shap_sample_size,
            "reliability_bins": self.reliability_bins,
            "small_group_min_rows": self.small_group_min_rows,
            "small_group_min_positives": self.small_group_min_positives,
            "state_min_test_rows": self.state_min_test_rows,
            "model_params": self.model_params,
            "candidates": [item.to_dict() for item in self.candidates],
            "pinned_candidate": (
                self.pinned_candidate.to_dict()
                if self.pinned_candidate is not None
                else None
            ),
            "calibration_methods": list(self.calibration_methods),
            "split_assignment_supplied": self.split_assignment is not None,
            "split_assignment_path": self.split_assignment_path,
            "smoke_rows": self.smoke_rows,
            "metadata": self.metadata,
        }


@dataclass
class TrainingResult:
    """The in-memory result plus the locations of every artifact."""

    config: TrainingConfig
    split_assignment: pd.DataFrame
    model: ModelBundle
    metrics: dict
    subgroup_metrics: pd.DataFrame
    calibration_curve: pd.DataFrame
    threshold_analysis: pd.DataFrame
    feature_importance: pd.DataFrame
    shap_summary: pd.DataFrame
    run_metadata: dict
    artifacts: dict[str, Path]
    run_dir: Path

    @property
    def row_counts(self) -> dict:
        """The row counts recorded for this run."""
        return self.run_metadata["row_counts"]


@dataclass(frozen=True)
class PreparedData:
    """The development and test rows produced by :func:`prepare_training_data`.

    ``development`` and ``test`` are disjoint and together are exactly the used
    sample. ``assignment`` holds one row per used respondent, ``filtered`` is the
    filtered frame before the split columns are merged, and the counts and
    summary describe the same partition.
    """

    filtered: pd.DataFrame
    development: pd.DataFrame
    test: pd.DataFrame
    assignment: pd.DataFrame
    row_counts: dict
    split_summary: dict
    split_source: str


def _stratified_sample(
    frame: pd.DataFrame,
    target_column: str,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    """Take a class-balanced smoke sample without touching the full table again."""
    if rows >= len(frame) or rows <= 0:
        return frame

    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby(target_column, sort=True):
        take = max(1, round(rows * len(group) / len(frame)))
        take = min(take, len(group))
        index = rng.choice(group.index.to_numpy(), size=take, replace=False)
        pieces.append(frame.loc[index])
    sampled = pd.concat(pieces)
    if len(sampled) > rows:
        sampled = sampled.sample(n=rows, random_state=seed)
    return sampled.sort_index()


def _prepare_sample(features: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, dict]:
    if features is None or features.empty:
        raise ValueError("feature table is empty")
    required = [
        "case_id",
        config.group_column,
        config.household_column,
        config.target_column,
        *config.feature_columns,
    ]
    if config.weight_column:
        required.append(config.weight_column)
    if config.secondary_target_column:
        required.append(config.secondary_target_column)
    for item in config.sample_filters:
        required.append(item.column)
    for subgroup in config.subgroups:
        required.append(subgroup.column)
    _require_columns(features, list(dict.fromkeys(required)), "feature table")
    validate_feature_frame(features, config.feature_columns)

    if features["case_id"].isna().any():
        raise ValueError("case_id must not be missing")
    if features["case_id"].duplicated().any():
        raise ValueError("case_id must be unique")

    mask = pd.Series(True, index=features.index)
    for item in config.sample_filters:
        mask &= item.apply(features)
    frame = features.loc[mask].copy()

    if frame.empty:
        raise ValueError("the configured sample filters retain no rows")
    if frame[config.target_column].isna().any():
        raise ValueError("target must not be missing after the sample filter")
    if not frame[config.target_column].isin([0, 1]).all():
        raise ValueError("target must be 0 or 1")

    filtered_rows = len(frame)
    if config.smoke_rows is not None:
        frame = _stratified_sample(
            frame,
            config.target_column,
            int(config.smoke_rows),
            config.seed,
        )

    frame = frame.reset_index(drop=True)
    frame["case_id"] = frame["case_id"].astype(str)
    counts = {
        "rows_after_filters": filtered_rows,
        "rows_used": len(frame),
        "positives_used": int(frame[config.target_column].sum()),
        "smoke": config.smoke_rows is not None,
    }
    return frame, counts


def _resolve_assignment(
    features: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, str]:
    assignment = config.split_assignment
    if assignment is None and config.split_assignment_path:
        path = Path(config.split_assignment_path)
        if not path.exists():
            raise ValueError(f"split assignment file does not exist: {path}")
        assignment = load_split_assignment(path)

    if assignment is not None:
        assignment = validate_split_assignment(assignment)
        assignment["case_id"] = assignment["case_id"].astype(str)
        ids = set(features["case_id"])
        assignment = assignment[assignment["case_id"].isin(ids)].copy()
        if len(assignment) != len(features):
            missing = len(features) - len(assignment)
            raise ValueError(
                f"split assignment does not cover {missing} sampled respondents"
            )
        assert_no_group_leakage(assignment)
        source = split_source(assignment) or "supplied"
        return assignment.reset_index(drop=True), source

    assignment = assign_splits(
        features,
        seed=config.seed,
        test_fraction=config.test_fraction,
        n_folds=config.n_folds,
    )
    return assignment, "generated"


def prepare_training_data(
    features: pd.DataFrame,
    config: TrainingConfig,
) -> PreparedData:
    """Apply the sample filters and split the used rows into development and test.

    This is the point where the two frames become separate objects: callers that
    select a model receive ``prepared.development`` and never the test rows.
    """
    frame, row_counts = _prepare_sample(features, config)
    assignment, source = _resolve_assignment(frame, config)

    data = frame.merge(
        assignment[["case_id", "split", "fold"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(data) != len(frame):
        raise ValueError("split assignment could not be matched to every respondent")
    assert_no_group_leakage(assignment)
    split_summary = summarise_splits(assignment, frame)

    development = data[data["split"].eq(DEVELOPMENT_SPLIT)].reset_index(drop=True)
    test = data[data["split"].eq(TEST_SPLIT)].reset_index(drop=True)
    if development.empty or test.empty:
        raise ValueError("the split assignment must contain development and test rows")

    folds = pd.to_numeric(development["fold"], errors="coerce")
    if folds.isna().any() or not folds.isin(range(1, config.n_folds + 1)).all():
        raise ValueError(
            f"every development respondent needs a valid fold between 1 and {config.n_folds}"
        )

    return PreparedData(
        filtered=frame,
        development=development,
        test=test,
        assignment=assignment,
        row_counts=row_counts,
        split_summary=split_summary,
        split_source=source,
    )


def _validate_development_frame(
    development: pd.DataFrame,
    config: TrainingConfig,
) -> None:
    """Reject anything but development rows with a valid fold, before fitting."""
    _require_columns(development, ["split", "fold"], "selection frame")
    leaked = ~development["split"].eq(DEVELOPMENT_SPLIT)
    if leaked.any():
        raise ValueError(
            "selection accepts development rows only; found "
            f"{int(leaked.sum())} test rows"
        )
    folds = pd.to_numeric(development["fold"], errors="coerce")
    valid = (
        folds.notna()
        & folds.mod(1).eq(0)
        & folds.isin(range(1, config.n_folds + 1))
    )
    if not valid.all():
        raise ValueError(
            "every development respondent needs a valid fold between "
            f"1 and {config.n_folds}; found {int((~valid).sum())} invalid folds"
        )


def evaluate_candidates(
    development: pd.DataFrame,
    config: TrainingConfig,
) -> SelectionEvidence:
    """Score every configured candidate on the development folds.

    The frame must hold development rows only; a test row is rejected before
    any model is built. Calibration and the screening threshold are fitted on
    each candidate's own out-of-fold predictions, and every table is JSON-safe.
    """
    _validate_development_frame(development, config)
    _require_columns(development, [config.target_column], "selection frame")
    validate_feature_frame(development, config.feature_columns)

    candidates = _resolve_candidates(config)
    if not candidates:
        raise ValueError("no model trials were produced")

    target = development[config.target_column].to_numpy()
    fold = pd.to_numeric(development["fold"], errors="coerce").astype(int).to_numpy()
    feature_set = (
        config.metadata.get("feature_set")
        or config.metadata.get("feature_variant")
        or "default"
    )
    feature_count = len(config.feature_columns)
    search_space = [candidate.to_dict() for candidate in candidates]

    records = []
    for candidate in candidates:
        trial = _cross_validate_candidate(development, config, candidate)
        covered = trial["covered"]
        if not covered.any():
            raise ValueError(
                f"candidate '{candidate.label}' has no out-of-fold predictions"
            )
        covered_target = target[covered]
        oof_scores = trial["oof"]
        calibrator, calibration_report = select_calibrator(
            covered_target,
            oof_scores[covered],
            methods=config.calibration_methods,
        )
        probabilities = np.full(len(development), np.nan)
        probabilities[covered] = np.clip(
            calibrator.predict(oof_scores[covered]), 0.0, 1.0
        )
        calibrated = probabilities[covered]
        threshold_report = select_threshold(
            covered_target,
            calibrated,
            target_sensitivity=config.target_sensitivity,
        )
        overall = binary_metrics(
            covered_target,
            calibrated,
            threshold=threshold_report["threshold"],
            n_bins=config.reliability_bins,
        )
        subgroups = subgroup_metrics(
            development,
            config.target_column,
            probabilities,
            config,
            overall,
            SELECTION_SPLIT_LABEL,
        )
        records.append(
            SelectionRecord(
                label=candidate.label,
                family=candidate.family,
                params=trial["params"],
                tuning_record={
                    "trial_count": len(candidates),
                    "selected": {
                        "label": candidate.label,
                        "family": candidate.family,
                        "params": candidate.params,
                    },
                    "search_space": search_space,
                },
                feature_set=feature_set,
                feature_count=feature_count,
                oof_scores=oof_scores,
                oof_probabilities=probabilities,
                target=target,
                covered=covered,
                fold=fold,
                calibration_method=calibration_report["selected"],
                calibration_report=_jsonable(calibration_report),
                threshold_report=_jsonable(threshold_report),
                metrics=_jsonable(overall),
                subgroups=_jsonable(subgroups.to_dict("records")),
                reliability_curve=_jsonable(
                    reliability_curve(
                        covered_target,
                        calibrated,
                        n_bins=config.reliability_bins,
                    )
                ),
                net_benefit=_jsonable(
                    net_benefit_table(covered_target, calibrated)
                ),
                fold_metrics=_jsonable(trial["fold_metrics"]),
            )
        )

    records = tuple(records)
    return SelectionEvidence(
        feature_set=feature_set,
        feature_count=feature_count,
        development_rows=len(development),
        fold_count=int(development["fold"].nunique()),
        records=list(records),
        fit_handle=FitHandle(
            development=development,
            config=config,
            records=records,
        ),
    )


def _mean(values) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _selection_key(result: dict) -> tuple:
    pr_auc = result["mean_pr_auc"]
    return (
        pr_auc is not None,
        pr_auc if pr_auc is not None else -1.0,
        -(result["mean_brier"] if result["mean_brier"] is not None else 1.0),
        -(result["mean_log_loss"] if result["mean_log_loss"] is not None else 1.0),
    )


def _param_candidates(config: TrainingConfig, family: str) -> list[dict]:
    configured = config.model_params.get(family, {})
    if isinstance(configured, list):
        return [dict(item) for item in configured] or [{}]
    return [dict(configured)]


def _resolve_candidates(config: TrainingConfig) -> list[Candidate]:
    """The candidates to score: the pinned one, the configured list, or the defaults."""
    if config.pinned_candidate is not None:
        candidates = [config.pinned_candidate]
    elif config.candidates:
        candidates = list(config.candidates)
    else:
        candidates = []
        for family in ("prevalence", "logistic", "lightgbm"):
            settings = _param_candidates(config, family)
            for index, params in enumerate(settings):
                label = family if len(settings) == 1 else f"{family}[{index}]"
                candidates.append(Candidate(label=label, family=family, params=params))

    labels = [candidate.label for candidate in candidates]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError("duplicate candidate labels: " + ", ".join(duplicates))
    if any(not label for label in labels):
        raise ValueError("candidate labels must not be empty")
    return candidates


def _cross_validate_candidate(
    data: pd.DataFrame,
    config: TrainingConfig,
    candidate: Candidate,
    *,
    seed: int | None = None,
) -> dict:
    family = candidate.family
    params = candidate.params
    model_seed = config.seed if seed is None else int(seed)
    development = data[data["split"].eq(DEVELOPMENT_SPLIT)].reset_index(drop=True)
    matrix, levels = model_matrix(
        development,
        config.feature_columns,
        config.categorical_columns,
    )
    folds = sorted(development["fold"].dropna().astype(int).unique())
    oof = np.full(len(development), np.nan)
    fold_metrics = []

    for fold in folds:
        valid = development["fold"].eq(fold).to_numpy()
        train = ~valid
        target_train = development.loc[train, config.target_column].to_numpy()
        if not train.any() or not valid.any():
            continue
        if len(np.unique(target_train)) < 2:
            fold_metrics.append({"fold": int(fold), "skipped": "single-class training fold"})
            continue

        model = build_model(
            family,
            params,
            feature_columns=config.feature_columns,
            categorical_columns=config.categorical_columns,
            seed=model_seed,
        )
        fit_model(
            model,
            matrix.loc[train],
            target_train,
            family=family,
            categorical_columns=config.categorical_columns,
        )
        probabilities = predict_positive_probability(model, matrix.loc[valid])
        oof[valid] = probabilities
        metrics = binary_metrics(
            development.loc[valid, config.target_column],
            probabilities,
            threshold=0.5,
            n_bins=config.reliability_bins,
        )
        fold_metrics.append({"fold": int(fold), **metrics})

    covered = np.isfinite(oof)
    if not covered.any():
        overall = {}
    else:
        overall = binary_metrics(
            development.loc[covered, config.target_column],
            oof[covered],
            threshold=0.5,
            n_bins=config.reliability_bins,
        )

    return {
        "label": candidate.label,
        "family": family,
        "params": params,
        "oof": oof,
        "covered": covered,
        "fold_metrics": fold_metrics,
        "overall_oof": overall,
        "mean_pr_auc": _mean(
            [item.get("pr_auc") for item in fold_metrics if "skipped" not in item]
        ),
        "mean_brier": _mean(
            [item.get("brier") for item in fold_metrics if "skipped" not in item]
        ),
        "mean_log_loss": _mean(
            [item.get("log_loss") for item in fold_metrics if "skipped" not in item]
        ),
        "category_levels": levels,
    }


def _record_summary(record: SelectionRecord, *, selected: bool = False) -> dict:
    """The per-candidate summary the existing artifact set records."""
    folds = [item for item in record.fold_metrics if "skipped" not in item]
    values = {
        "mean_pr_auc": _mean([item.get("pr_auc") for item in folds]),
        "mean_brier": _mean([item.get("brier") for item in folds]),
        "mean_log_loss": _mean([item.get("log_loss") for item in folds]),
    }
    return {
        "label": record.label,
        "family": record.family,
        "params": record.params,
        **values,
        "selected": selected,
    }


def _select_record(
    evidence: SelectionEvidence,
) -> tuple[SelectionRecord, dict]:
    """Pick the best scoring record and build the recorded selection summary."""
    if not evidence.records:
        raise ValueError("no model trials were produced")
    best = max(evidence.records, key=lambda item: _selection_key(_record_summary(item)))
    summary = {
        "selected_label": best.label,
        "selected_family": best.family,
        "selected_params": best.params,
        "tuning_record": best.tuning_record,
        "trials": [
            _record_summary(record, selected=record is best)
            for record in evidence.records
        ],
    }
    return best, summary


def _gap(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return round(value - reference, 6)


def _unreliable_reason(
    rows_ok: bool,
    positives_ok: bool,
    min_rows: int,
    min_positives: int,
) -> str:
    reasons = []
    if not rows_ok:
        reasons.append(f"fewer than {min_rows} respondents")
    if not positives_ok:
        reasons.append(f"fewer than {min_positives} positives")
    return " and ".join(reasons)


def _audit_notes(config: TrainingConfig) -> list[str]:
    notes = {
        spec.note for spec in config.subgroups if spec.audit_only and spec.note
    }
    metadata_note = config.metadata.get("audit_subgroup_note")
    if metadata_note:
        notes.add(metadata_note)
    return sorted(notes)


def subgroup_metrics(
    frame: pd.DataFrame,
    target_column: str,
    probabilities: np.ndarray,
    config: TrainingConfig,
    overall: dict,
    split_label: str,
) -> pd.DataFrame:
    """Return one audit row per subgroup, labelled with the split it scored."""
    rows = []
    target = frame[target_column].to_numpy()
    scored = np.isfinite(probabilities)
    for spec in config.subgroups:
        labels = spec.group_values(frame)
        for group in sorted(labels.unique()):
            mask = labels.eq(group).to_numpy() & scored
            n = int(mask.sum())
            if not n or (spec.minimum_test_rows and n < spec.minimum_test_rows):
                continue
            metrics = binary_metrics(
                target[mask],
                probabilities[mask],
                threshold=overall["threshold"],
                n_bins=config.reliability_bins,
            )
            rows_ok = n >= config.small_group_min_rows
            positives_ok = (
                metrics["positives"] >= config.small_group_min_positives
            )
            reliable = bool(rows_ok and positives_ok)
            rows.append(
                {
                    "split": split_label,
                    "subgroup": spec.name,
                    "group": group,
                    "n": n,
                    "positives": metrics["positives"],
                    "prevalence": metrics["prevalence"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "calibration_error": metrics["ece"],
                    "gap_prevalence": _gap(
                        metrics["prevalence"], overall["prevalence"]
                    ),
                    "gap_roc_auc": _gap(metrics["roc_auc"], overall["roc_auc"]),
                    "gap_pr_auc": _gap(metrics["pr_auc"], overall["pr_auc"]),
                    "gap_sensitivity": _gap(
                        metrics["sensitivity"], overall["sensitivity"]
                    ),
                    "gap_calibration_error": _gap(
                        metrics["ece"], overall["ece"]
                    ),
                    "reliable": reliable,
                    "unreliable_reason": ""
                    if reliable
                    else _unreliable_reason(
                        rows_ok,
                        positives_ok,
                        config.small_group_min_rows,
                        config.small_group_min_positives,
                    ),
                    "audit_only": spec.audit_only,
                    "note": spec.note,
                }
            )
    return pd.DataFrame(rows)


def _feature_importance(
    model,
    family: str,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, str]:
    if family == "lightgbm":
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "gain"
    elif family == "logistic":
        preprocess = model.named_steps["preprocess"]
        classifier = model.named_steps["classifier"]
        names = list(preprocess.get_feature_names_out())
        values = np.abs(np.asarray(classifier.coef_[0], dtype=float))
        feature_columns = tuple(
            name.split("__", 1)[-1] for name in names
        )
        kind = "abs_coefficient"
    else:
        return (
            pd.DataFrame(columns=["feature", "importance", "importance_type"]),
            "none",
        )

    frame = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance": values,
            "importance_type": kind,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True), kind


def _shap_summary(
    model,
    family: str,
    matrix: pd.DataFrame,
    target: np.ndarray,
    feature_columns: tuple[str, ...],
    *,
    run_dir: Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    empty = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    if matrix.empty:
        return empty, pd.DataFrame(), {"method": "not_applicable", "fallback": True}

    if family != "lightgbm":
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                model,
                matrix,
                target,
                n_repeats=3,
                random_state=seed,
                scoring="roc_auc",
            )
            summary = (
                pd.DataFrame(
                    {
                        "feature": list(feature_columns),
                        "mean_abs_shap": result.importances_mean,
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            return summary, pd.DataFrame(), {
                "method": "permutation_importance_fallback",
                "fallback": True,
            }
        except Exception as error:  # noqa: BLE001 - environment dependent
            return empty, pd.DataFrame(), {
                "method": "unavailable",
                "fallback": True,
                "error": str(error),
            }

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix, check_additivity=False)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        if values.ndim != 2 or values.shape[1] != len(feature_columns):
            raise ValueError("unexpected SHAP value shape")

        summary = (
            pd.DataFrame(
                {
                    "feature": list(feature_columns),
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        values_frame = pd.DataFrame(values, columns=list(feature_columns))

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(values, matrix, show=False, plot_type="bar")
            figure = plt.gcf()
            figure.savefig(run_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close(figure)
        except Exception:  # noqa: BLE001, S110
            pass

        return summary, values_frame, {"method": "shap", "fallback": False}
    except Exception as error:  # noqa: BLE001 - environment dependent
        return empty, pd.DataFrame(), {
            "method": "unavailable",
            "fallback": True,
            "error": str(error),
        }


def _package_versions() -> dict[str, str | None]:
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "shap",
        "joblib",
        "pyarrow",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _fingerprint(frame: pd.DataFrame, columns) -> str:
    hashed = pd.util.hash_pandas_object(frame[list(columns)], index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _run_directory(output_dir: Path, config: TrainingConfig, fingerprint: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_dir / f"{config.name}_{stamp}_{config.seed}_{fingerprint[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _format(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metrics_table(metrics: dict) -> list[str]:
    rows = [
        ("n", metrics.get("n")),
        ("positives", metrics.get("positives")),
        ("prevalence", metrics.get("prevalence")),
        ("ROC-AUC", metrics.get("roc_auc")),
        ("PR-AUC", metrics.get("pr_auc")),
        ("Brier", metrics.get("brier")),
        ("Log loss", metrics.get("log_loss")),
        ("Sensitivity @ 80% specificity", metrics.get("sensitivity_at_specificity_0_8")),
        ("Sensitivity", metrics.get("sensitivity")),
        ("Specificity", metrics.get("specificity")),
        ("PPV", metrics.get("ppv")),
        ("NPV", metrics.get("npv")),
        ("F1", metrics.get("f1")),
        ("Calibration slope", metrics.get("calibration_slope")),
        ("Calibration intercept", metrics.get("calibration_intercept")),
        ("ECE", metrics.get("ece")),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    lines += [f"| {name} | {_format(value)} |" for name, value in rows]
    return lines


def _subgroup_table(subgroups: pd.DataFrame) -> list[str]:
    lines = [
        (
            "| Subgroup | Group | n | Positives | Prevalence | ROC-AUC | PR-AUC | "
            "Sensitivity | ECE | Δ Prevalence | Δ ROC-AUC | Δ PR-AUC | "
            "Δ Sensitivity | Δ ECE | Reliable |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in subgroups.itertuples(index=False):
        lines.append(
            f"| {row.subgroup} | {row.group} | {row.n} | {row.positives} | "
            f"{_format(row.prevalence)} | {_format(row.roc_auc)} | "
            f"{_format(row.pr_auc)} | {_format(row.sensitivity)} | "
            f"{_format(row.calibration_error)} | {_format(row.gap_prevalence)} | "
            f"{_format(row.gap_roc_auc)} | "
            f"{_format(row.gap_pr_auc)} | {_format(row.gap_sensitivity)} | "
            f"{_format(row.gap_calibration_error)} | "
            f"{'yes' if row.reliable else 'no'} |"
        )
    return lines


def _results_report(
    config: TrainingConfig,
    metrics: dict,
    subgroups: pd.DataFrame,
    importance: pd.DataFrame,
    metadata: dict,
) -> str:
    overall = metrics["overall"]
    lines = [
        f"# {config.name} results",
        "",
        f"Run: `{metadata['run_id']}`",
        "",
        (
            "This is a research and portfolio model. It estimates the probability of "
            "elevated random blood glucose or diagnosed diabetes; it is not a diagnosis."
        ),
        "",
    ]

    run_role = metadata.get("run_role")
    if run_role:
        sample = metadata.get("sample")
        comparison = metadata.get("comparison_run")
        feature_variant = metadata.get("feature_variant")
        lines += ["## Run role", "", str(run_role), ""]
        if sample:
            lines.append(f"- Sample: `{sample}`")
        if comparison:
            lines.append(f"- Compared with: `{comparison}`")
        if feature_variant:
            lines.append(f"- Feature variant: `{feature_variant}`")
        lines.append("")

    lines += [
        "## Sample",
        "",
        "| Split | Rows |",
        "|---|---:|",
        f"| Development | {metadata['row_counts']['development_rows']} |",
        f"| Test | {metadata['row_counts']['test_rows']} |",
        f"| Total used | {metadata['row_counts']['rows_used']} |",
        "",
        "## Model selection",
        "",
        "| Family | Mean PR-AUC | Mean Brier | Mean log loss | Selected |",
        "|---|---:|---:|---:|:---:|",
    ]
    for trial in metrics["model_selection"]["trials"]:
        lines.append(
            "| {family} | {pr} | {brier} | {loss} | {selected} |".format(
                family=trial["family"],
                pr=_format(trial["mean_pr_auc"]),
                brier=_format(trial["mean_brier"]),
                loss=_format(trial["mean_log_loss"]),
                selected="yes" if trial["selected"] else "no",
            )
        )

    lines += [
        "",
        "## Test performance at the chosen threshold",
        "",
        (
            f"Threshold: **{_format(overall['threshold'])}** "
            f"(target sensitivity {_format(config.target_sensitivity, 2)})"
        ),
        "",
        *_metrics_table(overall),
        "",
        "### 95% cluster-bootstrap intervals",
        "",
    ]
    intervals = overall.get("confidence_intervals") or {}
    if intervals:
        lines += [
            "| Metric | Point | Low | High |",
            "|---|---:|---:|---:|",
        ]
        for name, interval in intervals.items():
            lines.append(
                f"| {name} | {_format(interval.get('point'))} | "
                f"{_format(interval.get('low'))} | {_format(interval.get('high'))} |"
            )
    else:
        lines.append("Not available for this run.")

    weighted = overall.get("weighted")
    if weighted:
        lines += [
            "",
            "### Survey-weighted metrics",
            "",
            *_metrics_table(weighted),
        ]

    lines += [
        "",
        "### Fact-sheet target (reported separately)",
        "",
    ]
    secondary = metrics.get("secondary_target")
    if secondary:
        lines += [
            f"Column: `{secondary['column']}`",
            "",
            *_metrics_table(secondary),
        ]
    else:
        lines.append("Not available for this run.")

    lines += [
        "",
        "## Calibration",
        "",
        (
            "Calibrator selected on out-of-fold development predictions: "
            f"**{metrics['calibration']['selected']}**."
        ),
        "",
        "| Method | Brier | Log loss |",
        "|---|---:|---:|",
    ]
    for method, report in metrics["calibration"]["method_reports"].items():
        lines.append(
            f"| {method} | {_format(report['brier'])} | {_format(report['log_loss'])} |"
        )

    lines += ["", "## Subgroups", ""]
    if subgroups.empty:
        lines.append("No subgroup rows were produced.")
    else:
        rows = subgroups
        sex_rows = rows[rows["subgroup"].eq("sex")]
        age_rows = rows[rows["subgroup"].eq("age_band")]
        other_rows = rows[
            ~rows["subgroup"].isin(["sex", "age_band"]) & ~rows["audit_only"]
        ]
        audit_rows = rows[rows["audit_only"]]

        lines += [
            (
                f"Subgroups with fewer than {config.small_group_min_rows} respondents "
                f"or fewer than {config.small_group_min_positives} positives are "
                "flagged `Reliable = no` and must not be used to claim fairness."
            ),
            "",
            (
                f"State rows are reported only where the test set has at least "
                f"{config.state_min_test_rows} respondents."
            ),
            "",
            "### Sex-specific metrics (mandatory)",
            "",
            *(
                _subgroup_table(sex_rows)
                if not sex_rows.empty
                else ["_No sex rows were produced._"]
            ),
            "",
            "### Age-band-specific metrics (mandatory)",
            "",
            *(
                _subgroup_table(age_rows)
                if not age_rows.empty
                else ["_No age-band rows were produced._"]
            ),
            "",
            "### Other subgroups",
            "",
            *(
                _subgroup_table(other_rows)
                if not other_rows.empty
                else ["_No other subgroup rows were produced._"]
            ),
        ]

        if not audit_rows.empty:
            notes = _audit_notes(config)
            lines += [
                "",
                "### Fairness audit subgroups",
                "",
                (
                    "Religion and social group are audit-only here: they are "
                    "reported to test the feature-inclusion decision, not deployed "
                    "as default features."
                ),
                "",
            ]
            lines += [f"- {note}" for note in notes]
            lines += ["", *_subgroup_table(audit_rows)]
        else:
            notes = _audit_notes(config)
            if notes:
                lines += [
                    "",
                    "### Fairness audit subgroups",
                    "",
                    (
                        "This run does not report audit-only subgroup rows; the "
                        "fairness variant reports religion and social group."
                    ),
                    "",
                    *[f"- {note}" for note in notes],
                ]

        lines += [
            "",
            (
                "Subgroup gaps are descriptive, not causal. Excluding a protected "
                "attribute does not by itself make a model fair; the fairness "
                "variant tests what including religion and social group changes."
            ),
        ]

    lines += ["", "## Feature importance", ""]
    if importance.empty:
        lines.append("Not available for the selected model.")
    else:
        lines += ["| Feature | Importance |", "|---|---:|"]
        for row in importance.head(15).itertuples(index=False):
            lines.append(f"| {row.feature} | {_format(row.importance)} |")

    shap_report = metadata.get("shap", {})
    lines += [
        "",
        "## Explainability",
        "",
        (
            f"Method: **{shap_report.get('method', 'unknown')}** "
            f"(fallback: {'yes' if shap_report.get('fallback') else 'no'})."
        ),
        "",
    ]

    lines += [
        "",
        "## Limitations",
        "",
        "- Random blood glucose is a screening indicator, not a diagnosis.",
        (
            "- The target includes self-reported diagnosis and medication, so the model "
            "can learn access-to-care patterns."
        ),
        "- Subgroup gaps are descriptive, not causal.",
        "- This model is not for insurance, employment, or care-allocation decisions.",
        "",
    ]
    return "\n".join(lines)


def _model_card(config: TrainingConfig, metrics: dict, metadata: dict) -> str:
    overall = metrics["overall"]
    sensitivity = config.metadata.get("model_role") == "sensitivity"
    lines = [
        f"# Model card: {config.name}",
        "",
        f"Version date: {metadata['finished_at'][:10]}",
        "",
        "## Intended use",
        "",
        (
            "Research and portfolio screening support for elevated random blood "
            "glucose or diagnosed diabetes among Indian adults. Not a diagnosis."
        ),
        "",
    ]
    if sensitivity:
        lines += [
            (
                "This is a **sensitivity analysis**, not the deployed screener. The "
                "primary IR/MR model remains the model intended for the app."
            ),
            "",
        ]
    lines += [
        "## Target definition",
        "",
        config.target_definition,
        "",
        "## Sample",
        "",
        (
            f"{metadata['row_counts']['rows_used']} respondents were used, with "
            f"{overall['positives']} positives in the test set."
        ),
        "",
        "## Features",
        "",
        ", ".join(f"`{column}`" for column in config.feature_columns),
        "",
    ]
    feature_variant = config.metadata.get("feature_variant")
    if sensitivity:
        if feature_variant in (None, "default"):
            lines += [
                (
                    "The PR default excludes state, religion, social group, "
                    "anthropometry, diet, cooking fuel, marital status, children ever "
                    "born, the smoking/smokeless split, and alcohol frequency."
                ),
                "",
            ]
        else:
            lines += [
                (
                    f"This run is the `{feature_variant}` feature-set variant; its "
                    "feature list is above."
                ),
                "",
            ]
    lines += [
        "## Performance",
        "",
        f"- ROC-AUC: {_format(overall['roc_auc'])}",
        f"- PR-AUC: {_format(overall['pr_auc'])}",
        f"- Brier: {_format(overall['brier'])}",
        f"- Sensitivity at the chosen threshold: {_format(overall['sensitivity'])}",
        f"- Specificity at the chosen threshold: {_format(overall['specificity'])}",
        "",
        "## Calibration and threshold",
        "",
        (
            f"The calibrator was selected on out-of-fold predictions "
            f"(`{metrics['calibration']['selected']}`). The referral threshold is "
            f"{_format(overall['threshold'])} and targets at least "
            f"{_format(config.target_sensitivity, 2)} sensitivity."
        ),
        "",
        "## Limitations and non-goals",
        "",
        "- The model does not predict future diabetes.",
        "- The model is not clinically validated.",
        "- It must not be used for insurance, employment, or care-allocation decisions.",
        "- Subgroup gaps are descriptive, not causal.",
        "",
        "## Ethical considerations",
        "",
        (
            "- The target includes self-reported diagnosis and medication, so the model "
            "can learn access-to-care patterns."
        ),
        "- Removing a protected attribute does not by itself make a model fair.",
        *[f"- {note}" for note in _audit_notes(config)],
        "",
        "## Citation",
        "",
        (
            "International Institute for Population Sciences (IIPS) and ICF (2021). "
            "National Family Health Survey (NFHS-5), 2019-21: India."
        ),
        "",
    ]
    return "\n".join(lines)


def run_training_pipeline(
    features: pd.DataFrame,
    config: TrainingConfig,
    output_dir: str | Path,
) -> TrainingResult:
    """Train, calibrate, evaluate, and document one model run."""
    started_at = _utc_now()
    prepared = prepare_training_data(features, config)
    frame = prepared.filtered
    row_counts = prepared.row_counts
    assignment = prepared.assignment
    source = prepared.split_source
    split_summary = prepared.split_summary
    development = prepared.development
    test = prepared.test

    if test[config.target_column].nunique() < 2:
        raise ValueError("the test split must contain both target classes")

    evidence = evaluate_candidates(development, config)
    selected_record, selection = _select_record(evidence)
    bundle = evidence.fit_handle.fit(selected_record.label)
    final_model = bundle.model
    calibrator = bundle.calibrator
    threshold = bundle.threshold
    calibration_report = selected_record.calibration_report
    threshold_report = selected_record.threshold_report
    selection["category_levels"] = bundle.category_levels

    covered = selected_record.covered
    oof_target = selected_record.target[covered]
    calibrated_oof = selected_record.oof_probabilities[covered]

    matrix_test = inference_matrix(
        test,
        config.feature_columns,
        config.categorical_columns,
        bundle.category_levels,
    )
    probabilities = bundle.predict_proba(test)

    target_test = test[config.target_column].to_numpy()
    overall = binary_metrics(
        target_test,
        probabilities,
        threshold=threshold,
        n_bins=config.reliability_bins,
    )
    overall["confidence_intervals"] = bootstrap_confidence_intervals(
        target_test,
        probabilities,
        test[config.group_column].to_numpy(),
        replicates=config.bootstrap_replicates,
        seed=config.seed,
    )
    if config.weight_column:
        weights = pd.to_numeric(test[config.weight_column], errors="coerce").to_numpy()
        overall["weighted"] = binary_metrics(
            target_test,
            probabilities,
            sample_weight=weights,
            threshold=threshold,
            n_bins=config.reliability_bins,
        )

    metrics = {
        "overall": overall,
        "model_selection": selection,
        "calibration": calibration_report,
        "threshold_selection": threshold_report,
        "split": split_summary,
        "calibrator": getattr(calibrator, "method", "unknown"),
    }

    secondary = config.secondary_target_column
    if secondary and secondary in test.columns and test[secondary].notna().any():
        secondary_mask = test[secondary].notna().to_numpy()
        metrics["secondary_target"] = {
            "column": secondary,
            **binary_metrics(
                test.loc[secondary_mask, secondary].to_numpy(),
                probabilities[secondary_mask],
                threshold=threshold,
                n_bins=config.reliability_bins,
            ),
        }

    subgroup_rows = subgroup_metrics(
        test, config.target_column, probabilities, config, overall, TEST_SPLIT
    )
    calibration_curve = pd.DataFrame(
        reliability_curve(
            target_test,
            probabilities,
            n_bins=config.reliability_bins,
        )
    )
    operating_points = threshold_analysis(
        oof_target,
        calibrated_oof,
        target_sensitivity=config.target_sensitivity,
    )
    importance, importance_kind = _feature_importance(
        final_model, selected_record.family, config.feature_columns
    )

    shap_sample = min(config.shap_sample_size, len(test))
    shap_matrix = matrix_test.iloc[:shap_sample] if shap_sample else matrix_test.iloc[:0]
    shap_target = target_test[:shap_sample]

    fingerprint = _fingerprint(
        frame,
        [
            "case_id",
            config.group_column,
            config.household_column,
            config.target_column,
            *config.feature_columns,
        ],
    )
    run_dir = _run_directory(Path(output_dir), config, fingerprint)
    shap_summary, shap_values, shap_report = _shap_summary(
        final_model,
        selected_record.family,
        shap_matrix,
        shap_target,
        config.feature_columns,
        run_dir=run_dir,
        seed=config.seed,
    )

    run_id = run_dir.name
    finished_at = _utc_now()
    metadata = {
        "run_id": run_id,
        "name": config.name,
        "run_role": config.metadata.get("run_role"),
        "sample": config.metadata.get("sample"),
        "comparison_run": config.metadata.get("comparison_run"),
        "started_at": started_at,
        "finished_at": finished_at,
        "split_source": source,
        "split_assignment_source": source,
        "seeds": {"split": config.seed, "model": config.seed, "bootstrap": config.seed},
        "config": config.to_dict(),
        "package_versions": _package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "dataset_fingerprint": fingerprint,
        "row_counts": {
            **row_counts,
            "development_rows": len(development),
            "test_rows": len(test),
            "test_positives": int(target_test.sum()),
            "clusters": int(frame[config.group_column].nunique()),
            "households": int(
                frame[[config.group_column, config.household_column]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "feature_count": len(config.feature_columns),
        "feature_variant": config.metadata.get("feature_variant"),
        "subgroup_rules": {
            "small_group_min_rows": config.small_group_min_rows,
            "small_group_min_positives": config.small_group_min_positives,
            "state_min_test_rows": config.state_min_test_rows,
        },
        "selected_model": {
            "family": selected_record.family,
            "params": selected_record.params,
            "importance_type": importance_kind,
        },
        "shap": shap_report,
        "extra": config.metadata,
    }

    bundle.metadata.update(
        {
            "run_id": run_id,
            "dataset_fingerprint": fingerprint,
        }
    )

    artifacts: dict[str, Path] = {}
    split_frame = assignment[
        [
            "case_id",
            config.group_column,
            config.household_column,
            "split",
            "fold",
            *(
                ["assignment_source"]
                if "assignment_source" in assignment.columns
                else []
            ),
        ]
    ].copy()
    artifacts["split_assignment"] = run_dir / "split_assignment.parquet"
    split_frame.to_parquet(artifacts["split_assignment"], index=False)

    artifacts["resolved_config"] = _write_json(
        run_dir / "resolved_config.json",
        {
            "run_id": run_id,
            **config.to_dict(),
            "resolved_model_params": selection["selected_params"],
            "selected_label": selection["selected_label"],
            "selected_family": selection["selected_family"],
        },
    )
    artifacts["run_metadata"] = _write_json(run_dir / "run_metadata.json", metadata)
    artifacts["metrics"] = _write_json(run_dir / "metrics.json", metrics)
    artifacts["subgroup_metrics"] = _write_frame(
        subgroup_rows, run_dir / "subgroup_metrics.csv"
    )
    artifacts["calibration_curve"] = _write_frame(
        calibration_curve, run_dir / "calibration_curve.csv"
    )
    artifacts["threshold_analysis"] = _write_frame(
        operating_points, run_dir / "threshold_analysis.csv"
    )
    artifacts["feature_importance"] = _write_frame(
        importance, run_dir / "feature_importance.csv"
    )
    artifacts["shap_summary"] = _write_frame(
        shap_summary, run_dir / "shap_summary.csv"
    )
    if not shap_values.empty:
        artifacts["shap_values"] = _write_frame(
            shap_values, run_dir / "shap_values.csv"
        )
    artifacts["results_report"] = run_dir / "results_report.md"
    artifacts["results_report"].write_text(
        _results_report(config, metrics, subgroup_rows, importance, metadata),
        encoding="utf-8",
    )
    artifacts["model_card"] = run_dir / "model_card.md"
    artifacts["model_card"].write_text(
        _model_card(config, metrics, metadata),
        encoding="utf-8",
    )
    artifacts["model_bundle"] = save_model_bundle(
        bundle, run_dir / "model_bundle.joblib"
    )

    return TrainingResult(
        config=config,
        split_assignment=split_frame,
        model=bundle,
        metrics=metrics,
        subgroup_metrics=subgroup_rows,
        calibration_curve=calibration_curve,
        threshold_analysis=operating_points,
        feature_importance=importance,
        shap_summary=shap_summary,
        run_metadata=metadata,
        artifacts=artifacts,
        run_dir=run_dir,
    )
