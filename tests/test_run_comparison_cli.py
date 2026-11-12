"""CLI guards for the candidate comparison run."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_comparison.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_comparison_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_reports_a_missing_feature_table_before_scoring(tmp_path, capsys):
    missing = tmp_path / "missing.parquet"
    output = tmp_path / "out"

    script = _load_script()
    exit_code = script.main(["--features", str(missing), "--output", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert str(missing) in captured.err
    assert "scripts/build_features.py" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_run_reports_a_missing_split_assignment(tmp_path, capsys):
    features = tmp_path / "features.parquet"
    features.write_bytes(b"placeholder")
    missing = tmp_path / "missing_assignment.parquet"

    script = _load_script()
    exit_code = script.main(
        ["--features", str(features), "--split-assignment", str(missing)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err
