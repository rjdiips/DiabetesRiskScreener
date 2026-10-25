import pandas as pd

from screener.data import build_target, clean_glucose


def test_clean_glucose_keeps_valid_range_and_drops_codes():
    values = pd.Series([20, 100, 140, 141, 600, 601, 995, 996, 998, None])
    cleaned = clean_glucose(values)

    assert cleaned.iloc[0] == 20
    assert cleaned.iloc[1] == 100
    assert cleaned.iloc[2] == 140
    assert cleaned.iloc[3] == 141
    assert cleaned.iloc[4] == 600
    assert cleaned.iloc[5:].isna().all()


def test_build_target_union_and_unknown_status():
    df = pd.DataFrame(
        {
            "glucose": [100, 141, 100, 100, 100],
            "diagnosed": [0, 0, 1, 0, None],
            "medication": [0, 0, 0, 1, None],
        }
    )
    target = build_target(df, "glucose", "diagnosed", "medication")

    assert target.iloc[0] == 0
    assert target.iloc[1] == 1
    assert target.iloc[2] == 1
    assert target.iloc[3] == 1
    assert pd.isna(target.iloc[4])


def test_build_target_can_treat_missing_status_as_negative():
    df = pd.DataFrame(
        {
            "glucose": [100],
            "diagnosed": [None],
            "medication": [None],
        }
    )
    target = build_target(
        df, "glucose", "diagnosed", "medication", require_known_status=False
    )

    assert target.iloc[0] == 0


def test_invalid_glucose_is_unknown_even_if_diagnosed():
    df = pd.DataFrame(
        {
            "glucose": [995],
            "diagnosed": [1],
            "medication": [0],
        }
    )
    target = build_target(df, "glucose", "diagnosed", "medication")

    assert pd.isna(target.iloc[0])
