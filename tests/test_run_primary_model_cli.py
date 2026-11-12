"""CLI guards and the pinned final run for the primary IR/MR 15-49 model."""

import importlib.util
import json
from pathlib import Path

from test_pipeline import pr_shaped_features

from screener.feature_policy import PRIMARY_FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_primary_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_primary_model_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primary_shaped_features():
    """A synthetic table with every questionnaire-only primary feature."""
    frame = pr_shaped_features(n_clusters=40)
    for offset, column in enumerate(PRIMARY_FEATURE_SETS["questionnaire_only"]):
        if column not in frame:
            frame[column] = (frame["cluster"] + offset + frame["age"]) % 5
    return frame


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


def test_pinned_candidate_with_comparison_artifact_freezes_the_winner(
    tmp_path, capsys
):
    features = primary_shaped_features()
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    params = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 10,
        "n_jobs": 1,
        "verbosity": -1,
    }
    pin = {
        "label": "C4-lightgbm[1]",
        "family": "lightgbm",
        "params": params,
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()
    comparison_path = comparison_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "bundle": "questionnaire_only",
                "feature_set": "questionnaire_only",
                "feature_count": len(PRIMARY_FEATURE_SETS["questionnaire_only"]),
                "selection_rows": 100,
                "fold_count": 5,
                "test_evaluated_once": False,
                "candidates": [
                    {
                        "label": "C1-logistic[1]",
                        "family": "logistic",
                        "params": {"C": 1.0},
                        "rank": 2,
                        "vetoes": [],
                        "oof_metrics": {
                            "pr_auc": 0.2,
                            "roc_auc": 0.6,
                            "brier": 0.1,
                            "calibration_slope": 1.0,
                            "calibration_intercept": 0.0,
                            "sensitivity": 0.8,
                            "specificity": 0.4,
                        },
                        "calibration": {"selected": "platt"},
                        "max_reliable_sensitivity_gap": 0.02,
                    }
                ],
                "winner": {
                    "label": pin["label"],
                    "family": pin["family"],
                    "params": params,
                    "pr_auc": 0.3,
                    "brier": 0.09,
                    "final_run": None,
                },
            }
        ),
        encoding="utf-8",
    )

    script = _load_script()
    exit_code = script.main(
        [
            "--features",
            str(features_path),
            "--feature-set",
            "questionnaire_only",
            "--output",
            str(tmp_path / "out"),
            "--smoke",
            "--smoke-rows",
            "200",
            "--bootstrap",
            "5",
            "--shap-sample",
            "10",
            "--pin-candidate",
            str(pin_path),
            "--comparison-artifact",
            str(comparison_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert "Traceback" not in captured.err

    run_dirs = list((tmp_path / "out").glob("primary_irmr_15to49_*"))
    assert len(run_dirs) == 1
    metrics = json.loads(
        (run_dirs[0] / "metrics.json").read_text(encoding="utf-8")
    )
    assert [trial["label"] for trial in metrics["model_selection"]["trials"]] == [
        pin["label"]
    ]
    assert metrics["model_selection"]["selected_family"] == "lightgbm"
    assert metrics["model_selection"]["selected_params"] == params

    metadata = json.loads(
        (run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["selected_bundle"] == "questionnaire_only"
    assert metadata["config"]["pinned_candidate"] == pin

    artifact = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert artifact["test_evaluated_once"] is True
    assert artifact["winner"]["final_run"].endswith(run_dirs[0].name)
    assert artifact["winner"]["test_metrics"]["roc_auc"] is not None
    assert all(
        "final_run" not in item and "test_metrics" not in item
        for item in artifact["candidates"]
    )
    assert (comparison_dir / "comparison.md").exists()
