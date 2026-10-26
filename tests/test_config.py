"""Tests for project path helpers."""

from __future__ import annotations

from screener.config import PROCESSED_DIR, PROJECT_ROOT, relative_to_project


def test_relative_to_project_renders_project_paths_relative():
    rendered = relative_to_project(PROCESSED_DIR / "pr_features.parquet")

    assert rendered == "data/processed/pr_features.parquet"
    assert "\\" not in rendered


def test_relative_to_project_leaves_outside_paths_absolute():
    outside = PROJECT_ROOT.parent / "elsewhere" / "features.parquet"

    rendered = relative_to_project(outside)

    assert rendered == outside.as_posix()
    assert "\\" not in rendered
