"""Feature sets and the guard rails that keep target leakage out of training.

The primary model deploys a state-free, sensitive-attribute-free feature set.
The PR sensitivity model uses a smaller blood-pressure-based set. Both lists
are fixed here so a run cannot silently drift from the documented policy.
"""

from __future__ import annotations

from .features import (
    ID_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PR_EXTRA_COLUMNS,
    REDUNDANT_ENCODINGS,
    SAMPLE_COLUMNS,
    TARGET_COLUMNS,
)

PRIMARY_EXCLUDED_FEATURES = ("state", "religion", "social_group")

PRIMARY_DEFAULT_FEATURES: tuple[str, ...] = tuple(
    column
    for column in MODEL_FEATURE_COLUMNS
    if column not in PRIMARY_EXCLUDED_FEATURES
)
PRIMARY_STATE_FEATURES = PRIMARY_DEFAULT_FEATURES + ("state",)
PRIMARY_FAIRNESS_FEATURES = PRIMARY_DEFAULT_FEATURES + ("religion", "social_group")
PRIMARY_COMBINED_FEATURES = PRIMARY_DEFAULT_FEATURES + (
    "state",
    "religion",
    "social_group",
)

# The pre-registered measurement blocks the shortlist ablates one at a time.
# Substance is not ablated: it is cheap to collect and has a strong prior.
MEASUREMENT_BLOCKS: dict[str, tuple[str, ...]] = {
    "anthropometry": (
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_cm",
        "hip_cm",
        "waist_hip_ratio",
    ),
    "blood_pressure": (
        "systolic_mean",
        "diastolic_mean",
        "bp_readings",
        "bp_high",
        "bp_medication",
    ),
    "diet": (
        "diet_milk",
        "diet_pulses",
        "diet_green_leafy",
        "diet_fruit",
        "diet_eggs",
        "diet_fish",
        "diet_meat",
        "diet_fried",
        "diet_aerated",
    ),
}

# The questionnaire-only set keeps height, weight, and BMI because a scale is
# far more available than a tape measure or a cuff.
QUESTIONNAIRE_EXCLUDED_FEATURES = (
    "waist_cm",
    "hip_cm",
    "waist_hip_ratio",
    *MEASUREMENT_BLOCKS["blood_pressure"],
)


def _without(
    columns: tuple[str, ...],
    removed: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(column for column in columns if column not in removed)


PRIMARY_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "measurement_full": PRIMARY_DEFAULT_FEATURES,
    "measurement_no_anthropometry": _without(
        PRIMARY_DEFAULT_FEATURES,
        MEASUREMENT_BLOCKS["anthropometry"],
    ),
    "measurement_no_bp": _without(
        PRIMARY_DEFAULT_FEATURES,
        MEASUREMENT_BLOCKS["blood_pressure"],
    ),
    "measurement_no_diet": _without(
        PRIMARY_DEFAULT_FEATURES,
        MEASUREMENT_BLOCKS["diet"],
    ),
    "questionnaire_only": _without(
        PRIMARY_DEFAULT_FEATURES,
        QUESTIONNAIRE_EXCLUDED_FEATURES,
    ),
}

# The questionnaire-only set is the accessibility comparison, not a deployment
# candidate.
NON_COMPETING_FEATURE_SETS = frozenset({"questionnaire_only"})


PR_15_PLUS_FEATURES: tuple[str, ...] = (
    "female",
    "age",
    "urban",
    "wealth",
    "education",
    "tobacco_any",
    "alcohol_any",
    "systolic_mean",
    "diastolic_mean",
    "bp_readings",
    "bp_high",
    "bp_medication",
)

PR_EXCLUDED_FEATURES: tuple[str, ...] = (
    "state",
    "religion",
    "social_group",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "hip_cm",
    "waist_hip_ratio",
    "marital_status",
    "children_ever_born",
    "smoking_any",
    "smokeless_any",
    "alcohol_frequency",
    "diet_milk",
    "diet_pulses",
    "diet_green_leafy",
    "diet_fruit",
    "diet_eggs",
    "diet_fish",
    "diet_meat",
    "diet_fried",
    "diet_aerated",
    *PR_EXTRA_COLUMNS,
)

_PR_VARIANTS = {
    "default": PR_15_PLUS_FEATURES,
    "state": PR_15_PLUS_FEATURES + ("state",),
    "fairness": PR_15_PLUS_FEATURES + ("religion", "social_group"),
    "combined": PR_15_PLUS_FEATURES + ("state", "religion", "social_group"),
    "cooking_fuel": PR_15_PLUS_FEATURES + ("cooking_fuel_clean",),
}

PR_FEATURE_VARIANTS = _PR_VARIANTS
PR_VARIANT_NAMES = tuple(
    name for name in PR_FEATURE_VARIANTS if name != "default"
)

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    [
        *ID_COLUMNS,
        *TARGET_COLUMNS,
        *SAMPLE_COLUMNS,
        *REDUNDANT_ENCODINGS,
        "split",
        "fold",
        "assignment_source",
    ]
)


def primary_feature_set(name: str = "measurement_full") -> tuple[str, ...]:
    """Return a named primary feature set: a measurement bundle or questionnaire-only."""
    try:
        return PRIMARY_FEATURE_SETS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown primary feature-set '{name}'; expected one of "
            + ", ".join(sorted(PRIMARY_FEATURE_SETS))
        ) from error


def pr_feature_set(variant: str = "default") -> tuple[str, ...]:
    """Return a documented PR feature-set variant."""
    try:
        return PR_FEATURE_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown PR feature-set variant '{variant}'; expected one of "
            + ", ".join(sorted(PR_FEATURE_VARIANTS))
        ) from error


def validate_feature_frame(
    features,
    feature_columns: tuple[str, ...] | list[str],
) -> None:
    """Reject identifiers, target-derived columns, and missing inputs."""
    columns = list(feature_columns)
    if len(columns) != len(set(columns)):
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        raise ValueError(
            "duplicate feature columns: " + ", ".join(duplicates)
        )

    missing = [column for column in columns if column not in features.columns]
    forbidden = [column for column in columns if column in FORBIDDEN_FEATURE_COLUMNS]
    if forbidden:
        raise ValueError(
            "forbidden feature columns: " + ", ".join(forbidden)
        )
    if missing:
        raise ValueError(
            "feature frame is missing feature columns: " + ", ".join(missing)
        )
