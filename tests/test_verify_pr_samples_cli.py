"""CLI guards for the PR sample verification script."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_pr_samples.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_pr_samples_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_reports_a_missing_pr_feature_table(tmp_path, capsys):
    missing = tmp_path / "missing.parquet"

    script = _load_script()
    exit_code = script.main(
        ["--pr-features", str(missing), "--primary-features", str(missing)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert str(missing) in captured.err
    assert "scripts/build_pr_features.py" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
