"""Leakage-free cluster-grouped split assignment.

The split rule lives in one place so the primary IR/MR model and the PR
sensitivity model share the same 20% holdout and 5-fold development partition.
The PR assignment reuses the primary split and fold wherever a cluster appears
in both samples, and assigns the remaining clusters with the same seed and
stratification rule.

The artifact is one row per respondent: identifiers, ``split`` and ``fold``.
Neither ``cluster`` nor ``household`` may cross a split or a development fold.
Clusters are stratified by target prevalence and female share.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

TEST_SPLIT = "test"
DEVELOPMENT_SPLIT = "development"

SPLIT_SOURCE_REUSED = "reused_primary"
SPLIT_SOURCE_PR_ONLY = "pr_only"
SPLIT_SOURCE_MIXED = "mixed"

DEFAULT_SEED = 42
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_FOLDS = 5

_ID = "case_id"
_CLUSTER = "cluster"
_HOUSEHOLD = "household"
_TARGET = "target"
_FEMALE = "female"
_AGE_15_49 = "age_15_49"
_SPLIT = "split"
_FOLD = "fold"
_ASSIGNMENT_SOURCE = "assignment_source"

_REQUIRED_FEATURE_COLUMNS = [_ID, _CLUSTER, _HOUSEHOLD, _TARGET, _FEMALE]
_REQUIRED_ASSIGNMENT_COLUMNS = [_ID, _CLUSTER, _HOUSEHOLD, _SPLIT, _FOLD]

_PRIMARY_SPLIT_ALIASES = {
    "test": TEST_SPLIT,
    "holdout": TEST_SPLIT,
    "test_set": TEST_SPLIT,
    "development": DEVELOPMENT_SPLIT,
    "dev": DEVELOPMENT_SPLIT,
    "train": DEVELOPMENT_SPLIT,
    "training": DEVELOPMENT_SPLIT,
    "valid": DEVELOPMENT_SPLIT,
    "validation": DEVELOPMENT_SPLIT,
}


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _id_key(series: pd.Series) -> pd.Series:
    """Return a stable string key for cluster, household, or case identifiers."""
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64").astype(str)
    if pd.api.types.is_float_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and (numeric % 1 == 0).all():
            return numeric.astype("int64").astype(str)
    return series.astype("string").str.strip().astype(str)


def _group_keys(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the cluster key and the cluster-scoped household key."""
    cluster_key = _id_key(frame[_CLUSTER])
    # DHS household numbers (hv002) are only unique within a cluster, so a
    # household is identified by the (cluster, household) pair.
    household_key = cluster_key + ":" + _id_key(frame[_HOUSEHOLD])
    return cluster_key, household_key


def _validated_features(features: pd.DataFrame) -> pd.DataFrame:
    """Validate the fields needed for grouping and stratified assignment."""
    _require_columns(features, _REQUIRED_FEATURE_COLUMNS, "feature table")

    frame = features.copy()
    if frame[_ID].isna().any():
        raise ValueError("case_id must not be missing")
    if frame[_ID].duplicated().any():
        raise ValueError("case_id must be unique")

    for column in (_CLUSTER, _HOUSEHOLD):
        if frame[column].isna().any():
            raise ValueError(f"{column} must not be missing")

    if frame[_TARGET].isna().any():
        raise ValueError("target must not be missing")
    if not frame[_TARGET].isin([0, 1]).all():
        raise ValueError("target must be 0 or 1")

    if frame[_FEMALE].isna().any():
        raise ValueError("female must not be missing")
    if not frame[_FEMALE].isin([0, 1]).all():
        raise ValueError("female must be 0 or 1")

    frame["_cluster_key"], frame["_household_key"] = _group_keys(frame)

    return frame


def _quantile_band(values: pd.Series, quantiles: int) -> pd.Series:
    """Split cluster-level values into deterministic quantile bands."""
    numeric = values.astype(float)
    if numeric.nunique(dropna=False) <= 1:
        return pd.Series(0, index=numeric.index, dtype="int64")

    bands = pd.qcut(numeric, q=quantiles, labels=False, duplicates="drop")
    return bands.fillna(0).astype("int64")


def _cluster_table(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate respondents to one row per cluster with stratification bands."""
    table = (
        features.groupby("_cluster_key", sort=True)
        .agg(
            rows=(_ID, "size"),
            positives=(_TARGET, "sum"),
            females=(_FEMALE, "sum"),
        )
        .reset_index()
    )
    table["prevalence"] = table["positives"] / table["rows"]
    table["female_share"] = table["females"] / table["rows"]
    table["target_band"] = _quantile_band(table["prevalence"], 3)
    table["sex_band"] = _quantile_band(table["female_share"], 2)
    table["stratum"] = (
        table["target_band"].astype(str) + ":" + table["sex_band"].astype(str)
    )
    return table


def _stable_hash(seed: int, key: str) -> int:
    """A process-independent hash so assignment is reproducible across runs."""
    digest = hashlib.blake2b(
        f"{seed}:{key}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _validate_split_options(test_fraction: float, n_folds: int) -> None:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")


def _allocate_counts(
    desired: dict[str, float],
    capacities: dict[str, int],
    total: int,
    seed: int,
) -> dict[str, int]:
    """Round fractional cluster quotas to integers, respecting capacities."""
    if total <= 0:
        return {key: 0 for key in capacities}

    total = min(total, sum(capacities.values()))
    desired_sum = sum(max(0.0, value) for value in desired.values())
    if desired_sum <= 0:
        capacity_total = sum(capacities.values())
        scaled = {
            key: total * capacities[key] / capacity_total for key in capacities
        }
    else:
        scaled = {
            key: total * max(0.0, desired.get(key, 0.0)) / desired_sum
            for key in capacities
        }

    counts = {
        key: min(capacities[key], int(scaled[key])) for key in capacities
    }
    remainder = total - sum(counts.values())
    if remainder <= 0:
        return counts

    order = sorted(
        capacities,
        key=lambda key: (
            -(scaled[key] - int(scaled[key])),
            _stable_hash(seed, f"test:{key}"),
        ),
    )
    while remainder > 0:
        progressed = False
        for key in order:
            if remainder == 0:
                break
            if counts[key] < capacities[key]:
                counts[key] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return counts


def _select_test_clusters(
    table: pd.DataFrame,
    new_clusters: pd.DataFrame,
    reused_slots: dict[str, tuple[str, int | None]],
    *,
    seed: int,
    test_fraction: float,
) -> set[str]:
    """Choose PR-only clusters so the whole partition keeps its test share.

    The primary test clusters are fixed. PR-only clusters fill the remaining
    holdout quota, preferring strata that the primary assignment left below
    their target, then spreading any shortfall across all new clusters.
    """
    if new_clusters.empty:
        return set()

    rows_by_cluster = dict(zip(table["_cluster_key"], table["rows"]))
    stratum_by_cluster = dict(zip(table["_cluster_key"], table["stratum"]))
    total_rows = float(table["rows"].sum())
    target_rows = test_fraction * total_rows

    reused_test_rows = 0.0
    reused_test_by_stratum: dict[str, float] = {}
    for cluster, (split, _) in reused_slots.items():
        if split != TEST_SPLIT or cluster not in rows_by_cluster:
            continue
        rows = float(rows_by_cluster[cluster])
        reused_test_rows += rows
        stratum = stratum_by_cluster[cluster]
        reused_test_by_stratum[stratum] = (
            reused_test_by_stratum.get(stratum, 0.0) + rows
        )

    remaining_rows = target_rows - reused_test_rows
    if remaining_rows <= 0:
        return set()

    new_rows_total = float(new_clusters["rows"].sum())
    remaining_rows = min(remaining_rows, new_rows_total)
    mean_cluster_rows = new_rows_total / len(new_clusters)
    target_count = round(remaining_rows / mean_cluster_rows)
    target_count = max(0, min(len(new_clusters), target_count))
    if target_count == 0:
        return set()
    if target_count == len(new_clusters):
        return set(new_clusters["_cluster_key"])

    stratum_rows = table.groupby("stratum", sort=True)["rows"].sum().to_dict()
    new_by_stratum = new_clusters.groupby("stratum").agg(
        new_rows=("rows", "sum"),
        mean_rows=("rows", "mean"),
        count=("_cluster_key", "size"),
    )
    strata = list(new_by_stratum.index)

    quotas = {
        stratum: max(
            0.0,
            test_fraction * float(stratum_rows.get(stratum, 0.0))
            - reused_test_by_stratum.get(stratum, 0.0),
        )
        for stratum in strata
    }
    quota_total = sum(quotas.values())
    if quota_total >= remaining_rows:
        allocated_rows = {
            stratum: remaining_rows * quotas[stratum] / quota_total
            for stratum in strata
        }
    else:
        shortfall = remaining_rows - quota_total
        allocated_rows = {
            stratum: quotas[stratum]
            + shortfall
            * float(new_by_stratum.loc[stratum, "new_rows"])
            / new_rows_total
            for stratum in strata
        }

    desired = {
        stratum: allocated_rows[stratum]
        / float(new_by_stratum.loc[stratum, "mean_rows"])
        for stratum in strata
    }
    capacities = {
        stratum: int(new_by_stratum.loc[stratum, "count"])
        for stratum in strata
    }
    counts = _allocate_counts(desired, capacities, target_count, seed)

    selected: set[str] = set()
    for stratum, count in counts.items():
        if count <= 0:
            continue
        stratum_clusters = new_clusters[
            new_clusters["stratum"].eq(stratum)
        ].sort_values(["hash", "_cluster_key"], kind="stable")
        selected.update(stratum_clusters.head(count)["_cluster_key"])

    return selected


def _assign_fold_slots(
    table: pd.DataFrame,
    new_clusters: pd.DataFrame,
    slots: dict[str, tuple[str, int | None]],
    reused_slots: dict[str, tuple[str, int | None]],
    *,
    n_folds: int,
) -> None:
    """Assign PR-only development clusters to the least-filled fold.

    Each cluster goes to the fold that is furthest below its target both within
    the cluster's stratum and across the whole development set. Stratification
    keeps target prevalence and sex composition similar; the global term keeps
    the five fold sizes balanced when the primary folds carry unequal PR rows.
    """
    rows_by_cluster = dict(zip(table["_cluster_key"], table["rows"]))
    stratum_by_cluster = dict(zip(table["_cluster_key"], table["stratum"]))
    stratum_rows = table.groupby("stratum", sort=True)["rows"].sum().to_dict()

    test_rows_by_stratum: dict[str, float] = {}
    reused_fold_rows: dict[str, dict[int, float]] = {}
    global_reused_fold_rows: dict[int, float] = {}
    total_test_rows = 0.0
    for cluster, (split, fold) in slots.items():
        rows = float(rows_by_cluster[cluster])
        if split == TEST_SPLIT:
            total_test_rows += rows
            stratum = stratum_by_cluster[cluster]
            test_rows_by_stratum[stratum] = (
                test_rows_by_stratum.get(stratum, 0.0) + rows
            )
        elif cluster in reused_slots and fold is not None:
            stratum = stratum_by_cluster[cluster]
            by_fold = reused_fold_rows.setdefault(stratum, {})
            by_fold[fold] = by_fold.get(fold, 0.0) + rows
            global_reused_fold_rows[fold] = (
                global_reused_fold_rows.get(fold, 0.0) + rows
            )

    total_rows = float(table["rows"].sum())
    global_target = (total_rows - total_test_rows) / n_folds
    global_deficits = {
        fold: global_target - global_reused_fold_rows.get(fold, 0.0)
        for fold in range(1, n_folds + 1)
    }

    stratum_targets: dict[str, float] = {}
    stratum_deficits: dict[str, dict[int, float]] = {}
    for stratum, rows in stratum_rows.items():
        development_rows = float(rows) - test_rows_by_stratum.get(stratum, 0.0)
        target = development_rows / n_folds
        stratum_targets[stratum] = target
        by_fold = reused_fold_rows.get(stratum, {})
        stratum_deficits[stratum] = {
            fold: target - by_fold.get(fold, 0.0)
            for fold in range(1, n_folds + 1)
        }

    folds = list(range(1, n_folds + 1))
    global_scale = max(global_target, 1.0)
    for cluster, stratum, rows in zip(
        new_clusters["_cluster_key"],
        new_clusters["stratum"],
        new_clusters["rows"],
    ):
        if cluster in slots:
            continue
        deficits = stratum_deficits[stratum]
        stratum_scale = max(stratum_targets[stratum], 1.0)
        fold = max(
            folds,
            key=lambda candidate: (
                deficits[candidate] / stratum_scale
                + global_deficits[candidate] / global_scale,
                -candidate,
            ),
        )
        slots[cluster] = (DEVELOPMENT_SPLIT, fold)
        deficits[fold] -= float(rows)
        global_deficits[fold] -= float(rows)


def _assign_cluster_slots(
    features: pd.DataFrame,
    reused_slots: dict[str, tuple[str, int | None]],
    *,
    seed: int,
    test_fraction: float,
    n_folds: int,
) -> dict[str, tuple[str, int | None]]:
    """Assign every cluster to one split/fold slot.

    Reused primary clusters keep their slot. PR-only clusters are first chosen
    to fill the global test quota within their target-prevalence and
    sex-composition strata, then spread across the development folds.
    """
    _validate_split_options(test_fraction, n_folds)
    table = _cluster_table(features)

    rows_by_cluster = dict(zip(table["_cluster_key"], table["rows"]))
    slots: dict[str, tuple[str, int | None]] = {}
    for cluster, (split, fold) in reused_slots.items():
        if cluster not in rows_by_cluster:
            continue
        if split == DEVELOPMENT_SPLIT and fold not in range(1, n_folds + 1):
            raise ValueError(
                f"primary cluster {cluster} has development fold {fold}, "
                f"but n_folds is {n_folds}"
            )
        slots[cluster] = (split, fold)

    new_clusters = table[~table["_cluster_key"].isin(reused_slots)].copy()
    new_clusters["hash"] = [
        _stable_hash(seed, cluster) for cluster in new_clusters["_cluster_key"]
    ]
    new_clusters = new_clusters.sort_values(
        ["stratum", "hash", "_cluster_key"], kind="stable"
    )

    test_clusters = _select_test_clusters(
        table,
        new_clusters,
        reused_slots,
        seed=seed,
        test_fraction=test_fraction,
    )
    for cluster in test_clusters:
        slots[cluster] = (TEST_SPLIT, None)

    _assign_fold_slots(
        table,
        new_clusters,
        slots,
        reused_slots,
        n_folds=n_folds,
    )

    return slots


def _build_assignment(
    features: pd.DataFrame,
    slots: dict[str, tuple[str, int | None]],
    *,
    include_source: bool,
    reused_clusters: set[str] | None = None,
) -> pd.DataFrame:
    """Build the one-row-per-respondent assignment from the cluster slot map."""
    pairs = [slots[cluster] for cluster in features["_cluster_key"]]
    assignment = features[[_ID, _CLUSTER, _HOUSEHOLD]].copy()
    assignment[_SPLIT] = [split for split, _ in pairs]
    assignment[_FOLD] = pd.array([fold for _, fold in pairs], dtype="Int64")

    if include_source:
        reused_clusters = reused_clusters or set()
        assignment[_ASSIGNMENT_SOURCE] = [
            SPLIT_SOURCE_REUSED
            if cluster in reused_clusters
            else SPLIT_SOURCE_PR_ONLY
            for cluster in features["_cluster_key"]
        ]

    return assignment.reset_index(drop=True)


def _assert_partition_complete(
    assignment: pd.DataFrame,
    *,
    n_folds: int,
) -> None:
    """Reject an assignment that cannot support a test set and every fold."""
    splits = set(assignment[_SPLIT])
    if TEST_SPLIT not in splits:
        raise ValueError("split assignment has no test respondents")
    if DEVELOPMENT_SPLIT not in splits:
        raise ValueError("split assignment has no development respondents")

    development = assignment[assignment[_SPLIT].eq(DEVELOPMENT_SPLIT)]
    present = set(development[_FOLD].dropna().astype(int))
    missing = sorted(set(range(1, n_folds + 1)) - present)
    if missing:
        raise ValueError(
            "development folds are empty: " + ", ".join(str(fold) for fold in missing)
        )


def _normalise_primary_assignment(
    primary_assignment: pd.DataFrame,
    *,
    n_folds: int,
) -> dict[str, tuple[str, int | None]]:
    """Read a primary split assignment into a per-cluster split/fold mapping."""
    _require_columns(
        primary_assignment,
        [_CLUSTER, _SPLIT, _FOLD],
        "primary split assignment",
    )

    frame = primary_assignment[[_CLUSTER, _SPLIT, _FOLD]].copy()
    if frame[_CLUSTER].isna().any():
        raise ValueError("primary split assignment has a missing cluster")
    frame["_cluster_key"] = _id_key(frame[_CLUSTER])

    split = frame[_SPLIT].astype(str).str.strip().str.lower()
    unknown = sorted(set(split) - set(_PRIMARY_SPLIT_ALIASES))
    if unknown:
        raise ValueError(
            "primary split assignment has unknown split names: "
            + ", ".join(unknown)
        )
    frame["_split"] = split.map(_PRIMARY_SPLIT_ALIASES)

    fold = pd.to_numeric(frame[_FOLD], errors="coerce")
    if (fold.notna() & (fold % 1 != 0)).any():
        raise ValueError("primary development fold must be an integer")
    frame["_fold"] = fold
    frame.loc[frame["_split"].eq(TEST_SPLIT), "_fold"] = float("nan")

    development = frame["_split"].eq(DEVELOPMENT_SPLIT)
    development_folds = frame.loc[development, "_fold"].dropna().astype(int)
    if development_folds.empty:
        raise ValueError(
            "primary split assignment has development rows without a fold"
        )

    # Accept both 0-based and 1-based fold numbering.
    if development_folds.min() == 0:
        if development_folds.max() > n_folds - 1:
            raise ValueError(
                f"primary development fold exceeds the configured {n_folds} folds"
            )
        frame.loc[development, "_fold"] = frame.loc[development, "_fold"] + 1

    mapping: dict[str, tuple[str, int | None]] = {}
    for cluster, group in frame.groupby("_cluster_key", sort=False):
        splits = group["_split"].unique()
        if len(splits) != 1:
            raise ValueError(
                f"primary split assignment gives cluster {cluster} conflicting splits"
            )

        split_name = splits[0]
        if split_name == TEST_SPLIT:
            mapping[cluster] = (TEST_SPLIT, None)
            continue

        folds = group["_fold"].dropna().unique()
        if len(folds) != 1:
            raise ValueError(
                f"primary split assignment gives cluster {cluster} "
                "a missing or conflicting development fold"
            )
        fold_number = int(folds[0])
        if fold_number not in range(1, n_folds + 1):
            raise ValueError(
                f"primary development fold {fold_number} is outside the "
                f"configured {n_folds} folds"
            )
        mapping[cluster] = (DEVELOPMENT_SPLIT, fold_number)

    return mapping


def _normalise_assignment(assignment: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise a saved or in-memory split assignment."""
    _require_columns(
        assignment,
        _REQUIRED_ASSIGNMENT_COLUMNS,
        "split assignment",
    )

    frame = assignment.copy()
    split = frame[_SPLIT].astype(str).str.strip().str.lower()
    unknown = sorted(set(split) - {TEST_SPLIT, DEVELOPMENT_SPLIT})
    if unknown:
        raise ValueError(
            "split assignment has unknown split names: " + ", ".join(unknown)
        )
    frame[_SPLIT] = split

    fold = pd.to_numeric(frame[_FOLD], errors="coerce")
    if (fold.notna() & (fold % 1 != 0)).any():
        raise ValueError("fold must be an integer")
    fold = fold.astype("Int64")
    test = split.eq(TEST_SPLIT)
    fold = fold.mask(test, pd.NA)
    if fold[~test].isna().any():
        raise ValueError("development rows must have a fold")
    frame[_FOLD] = fold

    if frame[_ID].isna().any():
        raise ValueError("case_id must not be missing")
    if frame[_ID].duplicated().any():
        raise ValueError("case_id must be unique")
    for column in (_CLUSTER, _HOUSEHOLD):
        if frame[column].isna().any():
            raise ValueError(f"{column} must not be missing")

    return frame


def _slot_key(assignment: pd.DataFrame) -> pd.Series:
    fold = assignment[_FOLD].astype("Int64").fillna(-1).astype(str)
    return assignment[_SPLIT].astype(str) + ":" + fold


def check_split_leakage(assignment: pd.DataFrame) -> dict[str, bool]:
    """Report whether any cluster or household crosses a split or fold."""
    frame = _normalise_assignment(assignment)
    frame["_cluster_key"], frame["_household_key"] = _group_keys(frame)
    frame["_slot"] = _slot_key(frame)

    return {
        "cluster_single_slot": bool(
            frame.groupby("_cluster_key")["_slot"].nunique().eq(1).all()
        ),
        "household_single_slot": bool(
            frame.groupby("_household_key")["_slot"].nunique().eq(1).all()
        ),
    }


def assert_no_group_leakage(assignment: pd.DataFrame) -> None:
    """Raise when a cluster or household appears in more than one slot."""
    checks = check_split_leakage(assignment)
    if not checks["cluster_single_slot"]:
        raise ValueError("a cluster appears in more than one split or development fold")
    if not checks["household_single_slot"]:
        raise ValueError("a household appears in more than one split or development fold")


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _slot_stats(frame: pd.DataFrame) -> dict:
    rows = len(frame)
    positives = int(frame[_TARGET].sum())
    female = int(frame[_FEMALE].sum())
    return {
        "rows": rows,
        "clusters": int(frame["_cluster_key"].nunique()),
        "households": int(frame["_household_key"].nunique()),
        "positives": positives,
        "prevalence": _rate(positives, rows),
        "female": female,
        "male": rows - female,
        "female_share": _rate(female, rows),
    }


def _slice_summary(frame: pd.DataFrame) -> dict:
    test = frame[frame[_SPLIT].eq(TEST_SPLIT)]
    development = frame[frame[_SPLIT].eq(DEVELOPMENT_SPLIT)]

    folds = []
    for fold in sorted(development[_FOLD].dropna().unique()):
        stats = _slot_stats(development[development[_FOLD].eq(fold)])
        stats["fold"] = int(fold)
        folds.append(stats)

    return {
        "rows": len(frame),
        "clusters": int(frame["_cluster_key"].nunique()),
        "households": int(frame["_household_key"].nunique()),
        "test": _slot_stats(test),
        "development": _slot_stats(development),
        "folds": folds,
    }


def _split_source(assignment: pd.DataFrame) -> str | None:
    if _ASSIGNMENT_SOURCE not in assignment.columns:
        return None

    sources = set(assignment[_ASSIGNMENT_SOURCE].dropna().astype(str))
    unknown = sources - {SPLIT_SOURCE_REUSED, SPLIT_SOURCE_PR_ONLY}
    if unknown:
        raise ValueError(
            "split assignment has unknown assignment_source values: "
            + ", ".join(sorted(unknown))
        )
    if sources == {SPLIT_SOURCE_REUSED}:
        return SPLIT_SOURCE_REUSED
    if sources == {SPLIT_SOURCE_PR_ONLY}:
        return SPLIT_SOURCE_PR_ONLY
    return SPLIT_SOURCE_MIXED


def assign_splits(
    features: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    n_folds: int = DEFAULT_FOLDS,
) -> pd.DataFrame:
    """Assign a cluster-grouped 20% test holdout and development folds."""
    prepared = _validated_features(features)
    slots = _assign_cluster_slots(
        prepared,
        {},
        seed=seed,
        test_fraction=test_fraction,
        n_folds=n_folds,
    )

    assignment = _build_assignment(
        prepared,
        slots,
        include_source=False,
    )
    _assert_partition_complete(assignment, n_folds=n_folds)
    assert_no_group_leakage(assignment)
    return assignment


def assign_pr_splits(
    features: pd.DataFrame,
    primary_assignment: pd.DataFrame | None = None,
    *,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    n_folds: int = DEFAULT_FOLDS,
) -> pd.DataFrame:
    """Assign PR splits, reusing the primary slot for overlapping clusters.

    ``assignment_source`` records whether each row inherited its slot from the
    primary assignment or was assigned from the PR-only clusters.
    """
    _validate_split_options(test_fraction, n_folds)
    prepared = _validated_features(features)
    cluster_keys = set(prepared["_cluster_key"])

    primary_mapping: dict[str, tuple[str, int | None]] = {}
    if primary_assignment is not None and len(primary_assignment):
        primary_mapping = _normalise_primary_assignment(
            primary_assignment,
            n_folds=n_folds,
        )

    reused_slots = {
        cluster: slot
        for cluster, slot in primary_mapping.items()
        if cluster in cluster_keys
    }
    slots = _assign_cluster_slots(
        prepared,
        reused_slots,
        seed=seed,
        test_fraction=test_fraction,
        n_folds=n_folds,
    )

    assignment = _build_assignment(
        prepared,
        slots,
        include_source=True,
        reused_clusters=set(reused_slots),
    )
    _assert_partition_complete(assignment, n_folds=n_folds)
    assert_no_group_leakage(assignment)
    return assignment


def age_15_49_slice(assignment: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Filter an assignment to the 15-49 respondents without re-splitting."""
    _require_columns(features, [_ID, _AGE_15_49], "feature table")
    frame = _normalise_assignment(assignment)

    age_15_49_ids = set(
        features.loc[features[_AGE_15_49].fillna(False).astype(bool), _ID]
        .astype(str)
    )
    mask = frame[_ID].astype(str).isin(age_15_49_ids)
    return frame.loc[mask].reset_index(drop=True)


def summarise_splits(assignment: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Summarise split sizes, fold prevalence, and sex composition.

    The 15-49 comparison is the same partition filtered to the younger rows;
    the summary reports it separately and does not re-split.
    """
    normalised = _normalise_assignment(assignment)
    prepared = _validated_features(features)

    feature_ids = set(prepared[_ID].astype(str))
    assigned_ids = set(normalised[_ID].astype(str))
    unknown_ids = sorted(assigned_ids - feature_ids)
    if unknown_ids:
        raise ValueError(
            "split assignment contains case_id values not in the feature table: "
            + ", ".join(unknown_ids[:3])
        )
    missing_ids = sorted(feature_ids - assigned_ids)
    if missing_ids:
        raise ValueError(
            "split assignment does not cover every respondent: "
            + ", ".join(missing_ids[:3])
        )

    columns = [_ID, _TARGET, _FEMALE, "_cluster_key", "_household_key"]
    if _AGE_15_49 in prepared.columns:
        columns.append(_AGE_15_49)

    assignment_ids = normalised[[_ID, _SPLIT, _FOLD]].copy()
    assignment_ids[_ID] = assignment_ids[_ID].astype(str)
    feature_rows = prepared[columns].copy()
    feature_rows[_ID] = feature_rows[_ID].astype(str)

    merged = assignment_ids.merge(
        feature_rows,
        on=_ID,
        how="left",
        validate="one_to_one",
    )
    if merged[_TARGET].isna().any():
        raise ValueError("split assignment could not be matched to every respondent")

    summary = _slice_summary(merged)
    if _AGE_15_49 in merged.columns:
        younger = merged[merged[_AGE_15_49].fillna(False).astype(bool)]
        younger_summary = _slice_summary(younger)
        full_folds = {fold["fold"] for fold in summary["folds"]}
        younger_folds = {fold["fold"] for fold in younger_summary["folds"]}
        younger_summary["missing_folds"] = sorted(full_folds - younger_folds)
        summary[_AGE_15_49] = younger_summary

    summary["leakage"] = check_split_leakage(normalised)
    source = _split_source(normalised)
    if source is not None:
        summary["split_source"] = source

    return summary


def save_split_assignment(
    assignment: pd.DataFrame,
    path: str | Path,
    *,
    summary: dict | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write the split assignment and optional aggregate summary to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalised = _normalise_assignment(assignment)
    normalised.to_parquet(path, index=False)

    paths = {"assignment": path}
    if summary is not None:
        if summary_path is None:
            summary_path = path.with_name(f"{path.stem}_summary.json")
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths["summary"] = summary_path
    return paths


def load_split_assignment(path: str | Path) -> pd.DataFrame:
    """Load a saved split assignment and validate its partition fields."""
    return _normalise_assignment(pd.read_parquet(path))


def validate_split_assignment(assignment: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise an in-memory or loaded split assignment."""
    return _normalise_assignment(assignment)


def split_source(assignment: pd.DataFrame) -> str | None:
    """Return the recorded split provenance, if the assignment carries it."""
    return _split_source(assignment)

