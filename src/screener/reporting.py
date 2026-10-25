"""Small JSON and Markdown formatting helpers shared by report modules."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def plain(value: Any) -> Any:
    """Convert numpy and pandas scalars into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


__all__ = ["format_delta", "format_value", "plain"]
