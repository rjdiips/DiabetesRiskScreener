"""CLI guards for the PR 15-49 age-matched run."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pr_model.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_pr_model_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_15_49_run_refuses_to_bypass_the_15_plus_split_assignment(capsys):
    script = _load_script()

    exit_code = script.main(["--sample", "15-49", "--no-split-assignment"])

    assert exit_code == 2
    assert "must reuse the PR 15+ split assignment" in capsys.readouterr().err


def test_comparison_candidates_flag_expands_the_run_to_the_shortlist():
    script = _load_script()
    from screener.comparison import comparison_candidates

    default_config = script._CONFIG_BUILDERS[script.SAMPLE_15_PLUS]()

    shortlist = script.configure_candidates(default_config, True)

    assert default_config.candidates == ()
    assert len(shortlist.candidates) == 22
    assert shortlist.candidates == comparison_candidates()
    assert shortlist.metadata["candidate_set"] == "shortlist"
    assert (
        script.configure_candidates(default_config, False).metadata["candidate_set"]
        == "default"
    )
