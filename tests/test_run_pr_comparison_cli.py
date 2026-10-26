"""CLI loading and artifact writing for the primary-versus-PR report."""

import importlib.util
import json
from pathlib import Path

from test_pr_comparison import _three_runs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pr_comparison.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pr_comparison_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(directory: Path, result) -> Path:
    directory.mkdir(parents=True)
    (directory / "resolved_config.json").write_text(
        json.dumps(
            {
                "name": result.config.name,
                "feature_columns": list(result.config.feature_columns),
                "metadata": result.config.metadata,
                "state_min_test_rows": 500,
            }
        ),
        encoding="utf-8",
    )
    (directory / "run_metadata.json").write_text(
        json.dumps(result.run_metadata), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(result.metrics, allow_nan=False), encoding="utf-8"
    )
    result.subgroup_metrics.to_csv(directory / "subgroup_metrics.csv", index=False)
    result.calibration_curve.to_csv(
        directory / "calibration_curve.csv", index=False
    )
    result.threshold_analysis.to_csv(
        directory / "threshold_analysis.csv", index=False
    )
    return directory


def test_cli_writes_the_primary_vs_pr_comparison(tmp_path):
    primary, pr_15_49, pr_15_plus = _three_runs()
    primary_dir = _write_run(tmp_path / "primary", primary)
    pr_15_49_dir = _write_run(tmp_path / "pr_15_49", pr_15_49)
    pr_15_plus_dir = _write_run(tmp_path / "pr_15_plus", pr_15_plus)

    script = _load_script()
    exit_code = script.main(
        [
            "--primary",
            str(primary_dir),
            "--pr-15-49",
            str(pr_15_49_dir),
            "--pr-15-plus",
            str(pr_15_plus_dir),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    json_path = tmp_path / "out" / "primary_vs_pr_comparison.json"
    markdown_path = tmp_path / "out" / "primary_vs_pr_comparison.md"
    addendum_path = tmp_path / "out" / "pr_sensitivity_model_card_addendum.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert addendum_path.exists()

    comparison = json.loads(json_path.read_text(encoding="utf-8"))
    assert list(comparison["runs"]) == ["primary", "pr_15_49", "pr_15_plus"]
    assert comparison["runs"]["primary"]["feature_count"] == 32
    assert comparison["runs"]["pr_15_plus"]["feature_count"] == 12
    assert comparison["deltas"]["pr_15_plus_vs_pr_15_49"]["roc_auc"] == 0.04

    report = markdown_path.read_text(encoding="utf-8")
    assert "## Sample differences" in report
    assert "## Limitations" in report
    assert "(pr_sensitivity_model_card_addendum.md)" in report

    addendum = addendum_path.read_text(encoding="utf-8")
    assert "# Model-card addendum" in addendum
    assert "not the deployed screener" in addendum


def test_cli_reports_a_missing_run_directory(tmp_path, capsys):
    script = _load_script()
    exit_code = script.main(
        [
            "--primary",
            str(tmp_path / "missing"),
            "--pr-15-49",
            str(tmp_path / "missing"),
            "--pr-15-plus",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    assert "Could not load the comparison runs" in capsys.readouterr().err
