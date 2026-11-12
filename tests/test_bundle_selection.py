"""The pre-registered feature-bundle selection over comparison artifacts."""

import json

import pytest

from screener.bundle_selection import (
    bundle_records,
    render_bundle_selection,
    select_bundle,
    write_bundle_selection,
)


def artifact(bundle, features, pr_auc, brier, *, winner=True):
    return {
        "bundle": bundle,
        "feature_set": bundle,
        "feature_count": features,
        "winner": (
            {
                "label": f"{bundle}-winner",
                "family": "logistic",
                "params": {"C": 0.5},
                "pr_auc": pr_auc,
                "brier": brier,
            }
            if winner
            else None
        ),
    }


def test_smaller_bundle_within_both_margins_is_selected():
    result = select_bundle(
        [
            artifact("measurement_full", 32, 0.300, 0.060),
            artifact("measurement_no_bp", 27, 0.299, 0.061),
        ]
    )

    assert result["chosen"]["bundle"] == "measurement_no_bp"
    assert result["reference"]["bundle"] == "measurement_full"
    assert result["pinned_candidate"] == {
        "label": "measurement_no_bp-winner",
        "family": "logistic",
        "params": {"C": 0.5},
    }
    assert result["rule"]["bundle_pr_auc_margin"] == 0.005
    assert result["rule"]["bundle_brier_margin"] == 0.005


def test_ineligible_bundle_is_excluded_from_the_reference():
    result = select_bundle(
        [
            artifact("measurement_full", 32, 0.400, 0.050, winner=False),
            artifact("measurement_no_bp", 27, 0.300, 0.060),
            artifact("measurement_no_diet", 23, 0.299, 0.060),
        ]
    )

    assert [item["bundle"] for item in result["bundles"]] == [
        "measurement_full",
        "measurement_no_bp",
        "measurement_no_diet",
    ]
    assert not result["bundles"][0]["eligible"]
    assert result["reference"]["bundle"] == "measurement_no_bp"
    assert result["chosen"]["bundle"] == "measurement_no_diet"


def test_questionnaire_only_does_not_compete():
    result = select_bundle(
        [
            artifact("questionnaire_only", 17, 0.900, 0.010),
            artifact("measurement_full", 32, 0.300, 0.060),
            artifact("measurement_no_bp", 27, 0.299, 0.061),
        ]
    )

    assert [item["bundle"] for item in result["bundles"]] == [
        "measurement_full",
        "measurement_no_bp",
    ]
    assert result["chosen"]["bundle"] == "measurement_no_bp"
    assert result["pinned_candidate"]["label"] == "measurement_no_bp-winner"


def test_no_eligible_bundle_raises_rather_than_deploying():
    with pytest.raises(ValueError, match="no feature bundle"):
        select_bundle(
            [
                artifact("measurement_full", 32, 0.300, 0.060, winner=False),
                artifact("measurement_no_bp", 27, 0.299, 0.061, winner=False),
            ]
        )


def test_bundle_records_keep_only_the_selection_fields():
    records = bundle_records(
        [artifact("measurement_full", 32, 0.300, 0.060)]
    )

    assert records == [
        {
            "bundle": "measurement_full",
            "feature_count": 32,
            "eligible": True,
            "winner": {
                "label": "measurement_full-winner",
                "family": "logistic",
                "params": {"C": 0.5},
                "pr_auc": 0.300,
                "brier": 0.060,
            },
        }
    ]


def test_bundle_selection_writes_both_artifacts(tmp_path):
    result = select_bundle(
        [
            artifact("measurement_full", 32, 0.300, 0.060),
            artifact("measurement_no_bp", 27, 0.299, 0.061),
        ]
    )

    paths = write_bundle_selection(result, tmp_path)

    assert paths["bundle_selection_json"].exists()
    assert paths["bundle_selection_markdown"].exists()
    on_disk = json.loads(
        paths["bundle_selection_json"].read_text(encoding="utf-8")
    )
    assert on_disk == result
    report = render_bundle_selection(result)
    assert "measurement_no_bp" in report
    assert "out-of-fold" in report
    assert "0.005" in report
