"""CLI tests for selecting and freezing the deployed feature bundle."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_bundle.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("select_bundle_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison(bundle, features, pr_auc, brier):
    return {
        "bundle": bundle,
        "feature_set": bundle,
        "feature_count": features,
        "test_evaluated_once": False,
        "winner": {
            "label": f"{bundle}-winner",
            "family": "logistic",
            "params": {"C": 0.5},
            "pr_auc": pr_auc,
            "brier": brier,
        },
    }


def test_missing_comparison_artifact_exits_cleanly(tmp_path, capsys):
    script = _load_script()
    exit_code = script.main(
        [
            "--comparison-root",
            str(tmp_path / "comparison"),
            "--bundles",
            "measurement_full",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "measurement_full" in captured.err
    assert "comparison.json" in captured.err
    assert "Traceback" not in captured.err


def test_valid_comparison_pair_writes_the_selection_and_prints_the_pin(
    tmp_path, capsys
):
    root = tmp_path / "comparison"
    for bundle, features, pr_auc, brier in (
        ("measurement_full", 32, 0.300, 0.060),
        ("measurement_no_bp", 27, 0.299, 0.061),
    ):
        directory = root / bundle
        directory.mkdir(parents=True)
        (directory / "comparison.json").write_text(
            json.dumps(comparison(bundle, features, pr_auc, brier)),
            encoding="utf-8",
        )

    script = _load_script()
    exit_code = script.main(
        [
            "--comparison-root",
            str(root),
            "--bundles",
            "measurement_full",
            "measurement_no_bp",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Traceback" not in captured.err
    printed = json.loads(captured.out)
    assert printed["pinned_candidate"]["label"] == "measurement_no_bp-winner"
    assert (tmp_path / "out" / "bundle_selection.json").exists()
    assert (tmp_path / "out" / "bundle_selection.md").exists()
    on_disk = json.loads(
        (tmp_path / "out" / "bundle_selection.json").read_text(encoding="utf-8")
    )
    assert on_disk["chosen"]["bundle"] == "measurement_no_bp"
