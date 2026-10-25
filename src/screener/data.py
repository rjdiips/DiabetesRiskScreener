"""Load NFHS-5 recode files and build the diabetes-risk target."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import (
    AGE_MAX,
    AGE_MAX_PRIMARY,
    AGE_MIN,
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    GLUCOSE_THRESHOLD,
    IR_PATH,
    MR_PATH,
    PR_PATH,
)

PR_COLUMNS = [
    "hhid",
    "hvidx",
    "hv001",
    "hv002",
    "hv005",
    "hv009",
    "hv024",
    "hv025",
    "hv104",
    "hv105",
    "hv106",
    "hv270",
    "hv226",
    "sh25",
    "sh26",
    "shb18s",
    "shb18d",
    "shb25s",
    "shb25d",
    "shb29s",
    "shb29d",
    "sh305",
    "sh306",
    "ha2",
    "ha3",
    "ha40",
    "hb2",
    "hb3",
    "hb40",
    "shb55",
    "shb56",
    "shb57",
    "shb73",
    "shb74",
]

IR_COLUMNS = [
    "caseid",
    "v001",
    "v002",
    "v003",
    "v005",
    "v012",
    "v024",
    "v025",
    "v190",
    "s728a",
    "s728ab",
    "sb55",
    "sb56",
    "sb57",
    "sb73",
    "sb74",
]

MR_COLUMNS = [
    "mcaseid",
    "mv001",
    "mv002",
    "mv003",
    "mv005",
    "mv012",
    "mv024",
    "mv025",
    "mv190",
    "sm627a",
    "sm627ab",
    "smb55",
    "smb56",
    "smb57",
    "smb73",
    "smb74",
]

IR_RENAME = {
    "caseid": "case_id",
    "v001": "cluster",
    "v002": "household",
    "v003": "line",
    "v005": "weight",
    "v012": "age",
    "v024": "state",
    "v025": "residence",
    "v190": "wealth",
    "s728a": "currently_diabetic",
    "s728ab": "sought_treatment",
    "sb55": "ever_checked",
    "sb56": "diagnosed",
    "sb57": "medication",
    "sb73": "glucose_time",
    "sb74": "glucose_raw",
}

MR_RENAME = {
    "mcaseid": "case_id",
    "mv001": "cluster",
    "mv002": "household",
    "mv003": "line",
    "mv005": "weight",
    "mv012": "age",
    "mv024": "state",
    "mv025": "residence",
    "mv190": "wealth",
    "sm627a": "currently_diabetic",
    "sm627ab": "sought_treatment",
    "smb55": "ever_checked",
    "smb56": "diagnosed",
    "smb57": "medication",
    "smb73": "glucose_time",
    "smb74": "glucose_raw",
}


def _read_stata(path, columns: Iterable[str]) -> pd.DataFrame:
    """Read selected columns from a DHS Stata file without value-label conversion."""
    return pd.read_stata(path, columns=list(columns), convert_categoricals=False)


def load_pr(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the household-member recode (PR)."""
    return _read_stata(PR_PATH, columns or PR_COLUMNS)


def load_ir(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the women's individual recode (IR)."""
    return _read_stata(IR_PATH, columns or IR_COLUMNS)


def load_mr(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the men's individual recode (MR)."""
    return _read_stata(MR_PATH, columns or MR_COLUMNS)


def clean_glucose(series: pd.Series) -> pd.Series:
    """Return glucose in mg/dL, with missing and out-of-range codes set to NA."""
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(GLUCOSE_MIN, GLUCOSE_MAX))


def build_target(
    df: pd.DataFrame,
    glucose_col: str,
    diagnosed_col: str,
    medication_col: str,
    *,
    require_known_status: bool = True,
) -> pd.Series:
    """Build the primary target: high glucose OR diagnosed OR medication.

    The result is a nullable integer series: 1 positive, 0 negative, NA unknown.
    """
    glucose = clean_glucose(df[glucose_col])
    diagnosed = df[diagnosed_col]
    medication = df[medication_col]

    target = (glucose > GLUCOSE_THRESHOLD) | diagnosed.eq(1) | medication.eq(1)
    target = target.astype("Int64")

    known_status = diagnosed.notna() | medication.notna()
    if require_known_status:
        target = target.where(glucose.notna() & known_status)
    else:
        target = target.where(glucose.notna())
    return target


def load_pr_analysis(
    *,
    age_min: int = AGE_MIN,
    age_max: int = AGE_MAX,
    require_known_status: bool = True,
) -> pd.DataFrame:
    """Load the PR file and return adults with valid glucose and a known target."""
    df = load_pr()
    df = df.loc[df["hv105"].between(age_min, age_max) & df["hv104"].isin([1, 2])].copy()
    df["glucose"] = clean_glucose(df["shb74"])
    df["target"] = build_target(
        df, "shb74", "shb56", "shb57", require_known_status=require_known_status
    )
    return df.loc[df["glucose"].notna() & df["target"].notna()].copy()


def load_irmr_analysis(
    *,
    age_min: int = AGE_MIN,
    age_max: int = AGE_MAX_PRIMARY,
    require_known_status: bool = True,
) -> pd.DataFrame:
    """Load and pool the women's and men's recodes with harmonised columns."""
    ir = load_ir().rename(columns=IR_RENAME)
    ir["sex"] = 2
    ir["recode"] = "IR"

    mr = load_mr().rename(columns=MR_RENAME)
    mr["sex"] = 1
    mr["recode"] = "MR"

    df = pd.concat([ir, mr], ignore_index=True, sort=False)
    df = df.loc[df["age"].between(age_min, age_max)].copy()
    df["glucose"] = clean_glucose(df["glucose_raw"])
    df["target"] = build_target(
        df,
        "glucose_raw",
        "diagnosed",
        "medication",
        require_known_status=require_known_status,
    )
    return df.loc[df["glucose"].notna() & df["target"].notna()].copy()
