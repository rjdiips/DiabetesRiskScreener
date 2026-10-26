"""Project paths and shared constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "nfhs5"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PR_PATH = RAW_DIR / "pr_household_member" / "IAPR7EFL.DTA"
HR_PATH = RAW_DIR / "hr_household" / "IAHR7EFL.DTA"
IR_PATH = RAW_DIR / "ir_women_15_49" / "IAIR7EFL.DTA"
MR_PATH = RAW_DIR / "mr_men_15_54" / "IAMR7EFL.DTA"

# Random blood glucose is reported in mg/dL.
GLUCOSE_MIN = 20.0
GLUCOSE_MAX = 600.0
GLUCOSE_THRESHOLD = 140.0
GLUCOSE_MISSING_CODES = (995.0, 996.0, 997.0, 998.0, 999.0)

AGE_MIN = 15
AGE_MAX = 98
AGE_MAX_PRIMARY = 49


def relative_to_project(path: str | Path) -> str:
    """Render a path for a published record.

    Run metadata and resolved configs are published, so they must not embed a
    contributor's local directory layout. Paths inside the project become
    forward-slash paths relative to the project root; anything else is kept
    as-is with forward slashes.
    """
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()
