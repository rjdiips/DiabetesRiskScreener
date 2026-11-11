"""CLI guards for the primary IR/MR 15-49 run."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_primary_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_primary_model_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_reports_a_missing_feature_table_before_training(tmp_path, capsys):
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
