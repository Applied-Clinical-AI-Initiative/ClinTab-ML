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


def test_apply_missing_handling_impute_numeric_uses_median():
    df = pd.DataFrame({"age": [10, 20, 30, np.nan]})
    out = stats.apply_missing_handling(df, {"age": "impute"})
    assert out["age"].isna().sum() == 0
    assert out.loc[3, "age"] == 20.0


def test_apply_missing_handling_impute_categorical_uses_mode():
    df = pd.DataFrame({"sex": ["M", "M", "F", np.nan]})
    out = stats.apply_missing_handling(df, {"sex": "impute"})
    assert out["sex"].isna().sum() == 0
    assert out.loc[3, "sex"] == "M"


def test_apply_missing_handling_impute_all_missing_column_left_as_is():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    out = stats.apply_missing_handling(df, {"x": "impute"})
    assert out["x"].isna().sum() == 3


def test_apply_missing_handling_include_leaves_column_untouched():
    df = _sample_df()
    out = stats.apply_missing_handling(df, {"age": "include"})
    assert out["age"].isna().sum() == 1


# ---------------------------------------------------------------------------
# training-fitted missing handling
# ---------------------------------------------------------------------------
def test_missing_percentages_only_lists_columns_with_missing_values():
    df = pd.DataFrame({"a": [1, np.nan, 3, np.nan], "b": [1, 2, 3, 4]})
    assert stats.missing_percentages(df) == {"a": 50.0}
    assert stats.missing_percentages(df.iloc[:0]) == {}


def test_class_balance_flags_imbalance_without_suggesting_anything():
    out = stats.class_balance(pd.Series([0] * 9 + [1]))
    assert out == {"minority_fraction": 0.1, "imbalanced": True}
    assert stats.class_balance(pd.Series([0, 1, 0, 1]))["imbalanced"] is False
    assert stats.class_balance(pd.Series([np.nan])) is None


def test_fill_values_come_from_the_training_partition_only():
    train = pd.DataFrame({"alb": [3.0, 3.2, 3.4, np.nan]})
    test = pd.DataFrame({"alb": [1.0, 1.0, 1.0, np.nan]})        # would drag a pooled median down
    fitted = stats.fit_missing_handling(train, {"alb": "impute"})
    assert fitted["fills"] == {"alb": 3.2}
    out = stats.apply_fitted_missing_handling(test, {"alb": "impute"}, fitted)
    assert out.loc[3, "alb"] == 3.2
    assert list(out["alb"][:3]) == [1.0, 1.0, 1.0]                # observed values untouched


def test_integer_coded_categorical_gets_its_mode_not_a_median():
    train = pd.DataFrame({"asa": [2, 2, 3, 3, 3, 4, np.nan]})
    fitted = stats.fit_missing_handling(train, {"asa": "impute"}, {"asa": "categorical"})
    assert fitted["fills"]["asa"] == 3
    fitted = stats.fit_missing_handling(train, {"asa": "impute"}, {"asa": "continuous"})
    assert fitted["fills"]["asa"] == 3.0


def test_string_column_gets_the_most_frequent_value():
    train = pd.DataFrame({"sex": ["M", "M", "F", None]})
    assert stats.fit_missing_handling(train, {"sex": "impute"})["fills"] == {"sex": "M"}


def test_impute_indicator_adds_a_binary_missing_column_to_every_partition():
    train = pd.DataFrame({"alb": [3.0, np.nan, 4.0]})
    val = pd.DataFrame({"alb": [np.nan, 2.0]})
    fitted = stats.fit_missing_handling(train, {"alb": "impute_indicator"})
    assert fitted["indicators"] == {"alb": "alb_missing"}
    for part, expect in ((train, [0, 1, 0]), (val, [1, 0])):
        out = stats.apply_fitted_missing_handling(part, {"alb": "impute_indicator"}, fitted)
        assert list(out["alb_missing"]) == expect
        assert out["alb"].isna().sum() == 0
        assert out["alb"].iloc[expect.index(1)] == 3.5


def test_fitted_handling_can_run_twice_without_losing_the_indicator():
    train = pd.DataFrame({"alb": [3.0, np.nan, 4.0]})
    decisions = {"alb": "impute_indicator"}
    fitted = stats.fit_missing_handling(train, decisions)
    once = stats.apply_fitted_missing_handling(train, decisions, fitted)
    twice = stats.apply_fitted_missing_handling(once, decisions, fitted)
    assert list(twice["alb_missing"]) == [0, 1, 0]


def test_indicator_name_does_not_clash_with_an_existing_column():
    train = pd.DataFrame({"alb": [1.0, np.nan], "alb_missing": [0, 1]})
    fitted = stats.fit_missing_handling(train, {"alb": "impute_indicator"})
    assert fitted["indicators"]["alb"] == "alb_missing_"


def test_include_zero_and_remove_with_fitted_handling():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0], "c": [np.nan, 1.0]})
    decisions = {"a": "include", "b": "zero", "c": "remove"}
    fitted = stats.fit_missing_handling(df, decisions)
    assert fitted == {"fills": {}, "indicators": {}}
    out = stats.apply_fitted_missing_handling(df, decisions, fitted)
    assert out["a"].isna().sum() == 1
    assert list(out["b"]) == [0.0, 2.0]
    assert "c" not in out.columns


def test_column_empty_in_training_is_left_missing():
    train = pd.DataFrame({"x": [np.nan, np.nan]})
    fitted = stats.fit_missing_handling(train, {"x": "impute"})
    assert fitted["fills"] == {"x": None}
    out = stats.apply_fitted_missing_handling(pd.DataFrame({"x": [np.nan, 5.0]}),
                                              {"x": "impute"}, fitted)
    assert out["x"].isna().sum() == 1


def test_fill_values_are_plain_python_for_json():
    import json
    train = pd.DataFrame({"a": np.array([1, 2, 3, np.nan]), "b": ["x", "x", "y", None]})
    fitted = stats.fit_missing_handling(train, {"a": "impute", "b": "impute"})
    json.dumps(fitted)
    assert type(fitted["fills"]["a"]) is float
