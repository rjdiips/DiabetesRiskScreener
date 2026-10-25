"""CLI guards and artifact writing for the PR variant suite."""

import importlib.util
import json
from pathlib import Path

from test_pipeline import pr_shaped_features

from screener.splits import assign_pr_splits

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pr_variants.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pr_variants_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variant_suite_requires_the_saved_split_assignment(tmp_path, capsys):
    features = pr_shaped_features()
    features_path = tmp_path / "pr_features.parquet"
    features.to_parquet(features_path, index=False)

    script = _load_script()
    exit_code = script.main(
        [
            "--features",
            str(features_path),
            "--split-assignment",
            str(tmp_path / "missing.parquet"),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    assert "variant suite needs one shared partition" in capsys.readouterr().err


def test_variant_suite_writes_the_comparison_as_json_and_markdown(tmp_path):
    features = pr_shaped_features()
    features_path = tmp_path / "pr_features.parquet"
    features.to_parquet(features_path, index=False)
    assignment = assign_pr_splits(features, seed=42)
    assignment_path = tmp_path / "pr_split_assignment.parquet"
    assignment.to_parquet(assignment_path, index=False)

    script = _load_script()
    exit_code = script.main(
        [
            "--features",
            str(features_path),
            "--split-assignment",
            str(assignment_path),
            "--output",
            str(tmp_path / "out"),
            "--variants",
            "state",
            "fairness",
            "--smoke",
            "--smoke-rows",
            "400",
            "--bootstrap",
            "5",
            "--shap-sample",
            "10",
        ]
    )

    assert exit_code == 0
    comparison_dirs = list((tmp_path / "out").glob("comparison_*"))
    assert len(comparison_dirs) == 1
    json_path = comparison_dirs[0] / "variant_comparison.json"
    markdown_path = comparison_dirs[0] / "variant_comparison.md"
    assert json_path.exists()
    assert markdown_path.exists()

    comparison = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(comparison["variants"]) == {"state", "fairness"}
    assert comparison["variants"]["state"]["added_features"] == ["state"]
    assert comparison["variants"]["fairness"]["added_features"] == [
        "religion",
        "social_group",
    ]
    assert comparison["runs"]["default"]["run_dir"]
    assert comparison["runs"]["state"]["run_dir"]
    assert comparison["small_group_rules"]["small_group_min_rows"] == 200

    report = markdown_path.read_text(encoding="utf-8")
    assert "# PR feature-set variant comparison" in report
    assert "household head" in report
    assert "does not by itself make a model fair" in report
