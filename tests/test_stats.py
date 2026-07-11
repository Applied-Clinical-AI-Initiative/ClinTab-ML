import numpy as np
import pandas as pd
import pytest

from clintab import stats


def _sample_df():
    # age needs >10 distinct values for the detector to call it "continuous"
    # rather than "categorical" (its rule for integer-like numeric columns).
    return pd.DataFrame({
        "age": [45, 62, 58, 71, 39, 80, 55, 63, np.nan, 47, 52, 68, 74, 41, 59],
        "sex": ["M", "F", "F", "M", "M", "F", "F", "M", "F", "M", "M", "F", "M", "F", "M"],
        "died": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "admit_date": ["2021-01-05", "2021-02-14", "2021-03-01", "2021-04-22",
                       "2021-05-30", "2021-06-11", "2021-07-19", "2021-08-08",
                       "2021-09-09", "2021-10-10", "2021-11-11", "2021-12-12",
                       "2022-01-01", "2022-02-02", "2022-03-03"],
    })


def test_detect_column_types_classifies_expected_roles():
    df = _sample_df()
    cols = {c["name"]: c for c in stats.detect_column_types(df)}

    assert cols["age"]["type"] == "continuous"
    assert cols["sex"]["type"] == "binary"
    assert cols["died"]["type"] == "binary"
    assert cols["admit_date"]["type"] == "date"


def test_detect_column_types_reports_missingness():
    df = _sample_df()
    cols = {c["name"]: c for c in stats.detect_column_types(df)}
    assert cols["age"]["pct_missing"] == pytest.approx(6.67)
    assert cols["age"]["high_missing"] is False


def test_minority_fraction():
    s = pd.Series(["A", "A", "A", "B", "A", "A", "A", "B"])
    assert stats.minority_fraction(s) == 0.25


def test_minority_fraction_empty_series_returns_none():
    assert stats.minority_fraction(pd.Series([np.nan, np.nan])) is None


def test_summarize_continuous_and_categorical_rows():
    df = _sample_df()
    coltypes = {c["name"]: c["type"] for c in stats.detect_column_types(df)}
    out = stats.summarize(df, coltypes)

    assert out["card"]["total_n"] == 15
    assert out["card"]["date_range"]["column"] == "admit_date"

    age_row = next(r for r in out["continuous"] if r["variable"] == "age")
    assert age_row["n"] == 14

    sex_row = next(r for r in out["categorical"] if r["variable"] == "sex")
    cats = {c["category"]: c["count"] for c in sex_row["categories"]}
    assert cats == {"M": 8, "F": 7}


def test_apply_missing_handling_remove_drops_column():
    df = _sample_df()
    out = stats.apply_missing_handling(df, {"age": "remove"})
    assert "age" not in out.columns


def test_apply_missing_handling_zero_fills_na():
    df = _sample_df()
    out = stats.apply_missing_handling(df, {"age": "zero"})
    assert out["age"].isna().sum() == 0
    assert out.loc[8, "age"] == 0


def test_apply_missing_handling_include_leaves_column_untouched():
    df = _sample_df()
    out = stats.apply_missing_handling(df, {"age": "include"})
    assert out["age"].isna().sum() == 1
