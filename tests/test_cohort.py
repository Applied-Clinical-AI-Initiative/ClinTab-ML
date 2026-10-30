import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from clintab import cohort


# ----------------------------------------------------------------------------
# parse_code_list
# ----------------------------------------------------------------------------
def test_parse_code_list_separators_and_normalising():
    text = " 27130, 27447;m16.1\n\tM17.0  27447 "
    assert cohort.parse_code_list(text) == ["27130", "27447", "M16.1", "M17.0"]


def test_parse_code_list_expands_integer_ranges():
    assert cohort.parse_code_list("27130-27132, 27447") == ["27130", "27131", "27132", "27447"]
    # spaces around the hyphen and dashes pasted from a word processor
    assert cohort.parse_code_list("27130 - 27131") == ["27130", "27131"]
    assert cohort.parse_code_list("27130\u201327131") == ["27130", "27131"]


def test_parse_code_list_range_keeps_zero_padding():
    assert cohort.parse_code_list("00100-00102") == ["00100", "00101", "00102"]


def test_parse_code_list_wildcards_and_artefacts():
    assert cohort.parse_code_list("m16*, 2744*") == ["M16*", "2744*"]
    # a float artefact from a numeric export is the integer code
    assert cohort.parse_code_list("27447.0") == ["27447"]
    # but short decimal codes (ICD-9) keep their digits
    assert cohort.parse_code_list("250.00, 81.0") == ["250.00", "81.0"]
    assert cohort.parse_code_list("'27447', \"27130\"") == ["27447", "27130"]


def test_parse_code_list_empty_and_list_input():
    assert cohort.parse_code_list("") == []
    assert cohort.parse_code_list(None) == []
    assert cohort.parse_code_list(["27130-27131", 27447, "M16.1 M17"]) == [
        "27130", "27131", "27447", "M16.1", "M17"]


def test_parse_code_list_hyphenated_codes_without_range_meaning():
    # CCI codes carry two hyphens, words carry hyphens without digits
    assert cohort.parse_code_list("1.VA.53.LA-PN-N Non-Hispanic") == [
        "1.VA.53.LA-PN-N", "NON-HISPANIC"]


@pytest.mark.parametrize("bad, fragment", [
    ("M16.0-M16.9", "only whole-number ranges"),
    ("27447-27130", "reversed"),
    ("1-20000", "the limit is 10,000"),
    ("27130-", "incomplete range"),
    ("-27130", "incomplete range"),
    ("M*16", "only a trailing *"),
    ("*", "would match every code"),
])
def test_parse_code_list_rejects_bad_tokens(bad, fragment):
    with pytest.raises(ValueError, match=fragment):
        cohort.parse_code_list(bad)


def test_parse_code_list_range_limit_is_inclusive():
    assert len(cohort.parse_code_list("10000-19999")) == 10000
    with pytest.raises(ValueError):
        cohort.parse_code_list("10000-20000")


def test_parse_code_list_lists_every_bad_token():
    with pytest.raises(ValueError) as ei:
        cohort.parse_code_list("27130, M16.0-M16.9, 27447-27130")
    msg = str(ei.value)
    assert "M16.0-M16.9" in msg and "27447-27130" in msg


# ----------------------------------------------------------------------------
# code matching
# ----------------------------------------------------------------------------
def _cpt_df():
    return pd.DataFrame({
        "CPT": pd.Series([27447, 27447.0, "27447", " 27447 ", "27130", 27130.0,
                          29881, "29881", None, np.nan], dtype=object),
        "id": range(10),
    })


def test_cpt_codes_stored_as_int_float_and_str_match():
    df = _cpt_df()
    for values in ([27447], ["27447"], [27447.0], "27447", ["27447.0"], [" 27447 "]):
        out, flow = cohort.apply_filters(df, [{"column": "CPT", "op": "in", "values": values}])
        assert out["id"].tolist() == [0, 1, 2, 3], values
        assert flow[0]["n_after"] == 4


def test_in_keeps_hip_and_knee_codes():
    df = _cpt_df()
    out, flow = cohort.apply_filters(df, [{"column": "CPT", "op": "in", "values": "27130, 27447"}])
    assert out["id"].tolist() == [0, 1, 2, 3, 4, 5]
    assert flow[0]["n_missing"] == 2
    assert flow[0]["label"] == "Included CPT 27130, 27447"


def test_numeric_float_column_and_ranges():
    df = pd.DataFrame({"CPT": [27130.0, 27131.0, 27132.0, 27133.0, np.nan]})
    out, flow = cohort.apply_filters(df, [{"column": "CPT", "op": "in", "values": "27130-27132"}])
    assert len(out) == 3
    assert flow[0]["values"] == ["27130-27132"]


def test_numeric_column_matches_code_written_with_leading_zeros():
    df = pd.DataFrame({"CPT": [100, 27447, 1999]})
    out, _ = cohort.apply_filters(df, [{"column": "CPT", "op": "in", "values": "00100, 01999"}])
    assert out["CPT"].tolist() == [100, 1999]


def test_leading_zeros_in_text_are_significant():
    # undotted ICD-9: 0389 (septicaemia) is not 389 (hearing loss)
    df = pd.DataFrame({"DX": ["0389", "389", "3890"]})
    out, _ = cohort.apply_filters(df, [{"column": "DX", "op": "in", "values": "389"}])
    assert out["DX"].tolist() == ["389"]


def test_icd9_decimals_are_not_float_artefacts():
    df = pd.DataFrame({"DX": ["250.00", "25000", "250", "81.0", "810"]})
    out, _ = cohort.apply_filters(df, [{"column": "DX", "op": "in", "values": "250.00, 81.0"}])
    assert out["DX"].tolist() == ["250.00", "25000", "81.0", "810"]


def test_multi_column_any_match():
    df = pd.DataFrame({
        "CPT": [27130, 44950, 44950, 47562, 27447],
        "OTHERCPT1": [np.nan, 27447.0, np.nan, np.nan, np.nan],
        "OTHERCPT2": [np.nan, np.nan, "27130", np.nan, np.nan],
        "OTHERCPT3": [np.nan, np.nan, np.nan, np.nan, 20680.0],
    })
    cols = ["CPT", "OTHERCPT1", "OTHERCPT2", "OTHERCPT3"]
    out, flow = cohort.apply_filters(df, [{"columns": cols, "op": "in", "values": [27130, 27447]}])
    assert out.index.tolist() == [0, 1, 2, 4]
    assert flow[0]["columns"] == cols
    assert flow[0]["label"] == "Included 27130, 27447 in any of CPT, OTHERCPT1, OTHERCPT2, OTHERCPT3"


def test_not_in_multi_column_drops_any_match_and_keeps_blanks():
    df = pd.DataFrame({
        "DX1": ["M16.1", "M17.0", "S72.0", None],
        "DX2": [None, "T84.0", None, None],
    })
    out, flow = cohort.apply_filters(df, [{"columns": ["DX1", "DX2"], "op": "not_in",
                                           "values": "S72*, T84*"}])
    assert out.index.tolist() == [0, 3]
    assert flow[0]["n_removed"] == 2
    assert flow[0]["n_missing"] == 1
    assert flow[0]["kind"] == "exclude"


def test_icd_with_and_without_dots_and_case():
    df = pd.DataFrame({"DX": ["M16.1", "M161", "m16.1", " m161 ", "M16.10", "M17.0"]})
    for code in ("M16.1", "M161", "m16.1"):
        out, _ = cohort.apply_filters(df, [{"column": "DX", "op": "in", "values": code}])
        assert out.index.tolist() == [0, 1, 2, 3], code


def test_wildcards_match_prefixes_only():
    df = pd.DataFrame({"DX": ["M16.0", "M161", "M16.12", "m16", "M17.0", "XM16", None]})
    out, flow = cohort.apply_filters(df, [{"column": "DX", "op": "in", "values": "M16*"}])
    assert out.index.tolist() == [0, 1, 2, 3]
    assert flow[0]["values"] == ["M16*"]
    # the dot before the * is ignored like any other dot
    out2, _ = cohort.apply_filters(df, [{"column": "DX", "op": "in", "values": "M16.1*"}])
    assert out2.index.tolist() == [1, 2]


def test_wildcards_on_numeric_columns():
    df = pd.DataFrame({"CPT": [27440, 27447, 27130, 2744], "OTHER": [np.nan, np.nan, 27446.0, np.nan]})
    out, _ = cohort.apply_filters(df, [{"columns": ["CPT", "OTHER"], "op": "in", "values": "2744*"}])
    assert out.index.tolist() == [0, 1, 2, 3]
    out, _ = cohort.apply_filters(df, [{"columns": ["CPT", "OTHER"], "op": "in", "values": "27447*"}])
    assert out.index.tolist() == [1]


def test_cci_codes_match_with_or_without_punctuation():
    df = pd.DataFrame({"PROC": ["1VA53LAPNN", "1.VA.53.LA-PN-N", "1VB53LAPNN"]})
    out, _ = cohort.apply_filters(df, [{"column": "PROC", "op": "in", "values": "1.VA.53.LA-PN-N"}])
    assert out.index.tolist() == [0, 1]
    out, _ = cohort.apply_filters(df, [{"column": "PROC", "op": "in", "values": "1VA53*"}])
    assert out.index.tolist() == [0, 1]


def test_code_matching_on_pandas_extension_dtypes():
    df = pd.DataFrame({
        "cat": pd.Categorical(["M16.1", "M17.0", None]),
        "i64": pd.array([27447, None, 27130], dtype="Int64"),
        "st": pd.array([" 27447", None, "m16.1"], dtype="string"),
    })
    out, flow = cohort.apply_filters(df, [{"columns": ["cat", "i64", "st"], "op": "in",
                                           "values": "27447, M16*"}])
    assert out.index.tolist() == [0, 2]
    assert flow[0]["n_missing"] == 0


def test_nested_values_are_rejected():
    with pytest.raises(ValueError, match="could not read"):
        cohort.apply_filters(_cpt_df(), [{"column": "CPT", "op": "in", "values": [["27447"]]}])


def test_list_values_keep_category_values_with_spaces():
    df = pd.DataFrame({"RACE": ["White", "Black or African American", "black or african american",
                                "Asian", None]})
    out, _ = cohort.apply_filters(df, [{"column": "RACE", "op": "in",
                                        "values": ["Black or African American", "Asian"]}])
    assert out.index.tolist() == [1, 2, 3]


# ----------------------------------------------------------------------------
# numeric and other ops
# ----------------------------------------------------------------------------
def test_between_with_missing_and_non_numeric_values():
    df = pd.DataFrame({"AGE": pd.Series([17, 18, None, 65, 66, "", "abc", " 40 ", 50.5],
                                        dtype=object)})
    out, flow = cohort.apply_filters(df, [{"column": "AGE", "op": "between", "values": [18, 65]}])
    assert out.index.tolist() == [1, 3, 7, 8]
    step = flow[0]
    assert step["n_removed"] == 5
    assert step["n_missing"] == 2
    assert step["n_non_numeric"] == 1
    assert step["non_numeric_examples"] == ["abc"]
    assert step["label"] == "Excluded AGE < 18 or > 65"


def test_between_on_numeric_column_with_nan():
    df = pd.DataFrame({"BMI": [17.9, 18.5, np.nan, 40.0, 40.1]})
    out, flow = cohort.apply_filters(df, [{"column": "BMI", "op": "between", "values": ["18.5", 40]}])
    assert out.index.tolist() == [1, 3]
    assert flow[0]["n_missing"] == 1
    assert flow[0]["values"] == [18.5, 40]


def test_top_coded_ages_are_read_as_their_bound():
    df = pd.DataFrame({"AGE": ["17", "45", "90+", "90 +", "89"]})
    out, flow = cohort.apply_filters(df, [{"column": "AGE", "op": "gte", "values": 18}])
    assert out.index.tolist() == [1, 2, 3, 4]
    assert flow[0]["n_top_coded"] == 2
    assert flow[0]["n_non_numeric"] == 0
    out, _ = cohort.apply_filters(df, [{"column": "AGE", "op": "lte", "values": [89]}])
    assert out.index.tolist() == [0, 1, 4]


def test_eq_and_neq_text_and_numbers():
    df = pd.DataFrame({
        "SEX": ["male", "Female", " FEMALE ", None, "female"],
        "EMERG": [1, 0, 1.0, np.nan, 0],
    })
    out, flow = cohort.apply_filters(df, [{"column": "SEX", "op": "eq", "values": ["female"]}])
    assert out.index.tolist() == [1, 2, 4]
    assert flow[0]["label"] == "Included SEX = female"
    assert flow[0]["kind"] == "include"
    # neq drops the matches and keeps blanks
    out, flow = cohort.apply_filters(df, [{"column": "EMERG", "op": "neq", "values": ["1"]}])
    assert out.index.tolist() == [1, 3, 4]
    assert flow[0]["label"] == "Excluded EMERG = 1"
    out, _ = cohort.apply_filters(df, [{"column": "EMERG", "op": "eq", "values": 1}])
    assert out.index.tolist() == [0, 2]


def test_not_missing_needs_every_column():
    df = pd.DataFrame({"A": [1, np.nan, 3, 4], "B": ["x", "y", " ", None]})
    out, flow = cohort.apply_filters(df, [{"columns": ["A", "B"], "op": "not_missing"}])
    assert out.index.tolist() == [0]
    assert flow[0]["n_missing"] == 3
    assert flow[0]["label"] == "Excluded missing any of A, B"


def test_ops_are_case_insensitive_and_custom_label_is_used():
    df = pd.DataFrame({"AGE": [10, 20, 30]})
    out, flow = cohort.apply_filters(df, [{"column": "AGE", "op": "GTE", "values": [18],
                                           "label": "  Adults only "}])
    assert len(out) == 2
    assert flow[0]["label"] == "Adults only"
    assert flow[0]["op"] == "gte"


# ----------------------------------------------------------------------------
# flow
# ----------------------------------------------------------------------------
def _cohort_df():
    return pd.DataFrame({
        "CPT": [27130, 27447, 27447, 29881, 27130, 27447, 44950, 27130],
        "AGE": [70, 16, 65, 40, None, 80, 55, 45],
        "OUTCOME": [0, 1, None, 0, 1, 0, 1, 1],
    })


def test_steps_run_in_order_with_chained_counts():
    df = _cohort_df()
    filters = [
        {"column": "CPT", "op": "in", "values": "27130, 27447"},
        {"column": "AGE", "op": "gte", "values": [18]},
        {"column": "OUTCOME", "op": "not_missing"},
    ]
    out, flow = cohort.apply_filters(df, filters)
    assert [s["step"] for s in flow] == [1, 2, 3]
    assert [(s["n_before"], s["n_after"], s["n_removed"]) for s in flow] == [
        (8, 6, 2), (6, 4, 2), (4, 3, 1)]
    assert [s["label"] for s in flow] == [
        "Included CPT 27130, 27447", "Excluded AGE < 18", "Excluded missing OUTCOME"]
    assert out.index.tolist() == [0, 5, 7]
    assert flow[1]["n_missing"] == 1      # counted among the 6 rows left, not all 8

    # same cohort in a different order, different counts per step
    out2, flow2 = cohort.apply_filters(df, filters[::-1])
    assert out2.index.tolist() == out.index.tolist()
    assert [s["n_removed"] for s in flow2] == [1, 2, 2]


def test_flow_is_json_safe():
    _, flow = cohort.apply_filters(_cohort_df(), [
        {"column": "CPT", "op": "in", "values": "27130-27132"},
        {"column": "AGE", "op": "between", "values": [18, 90]},
    ])
    back = json.loads(json.dumps(flow))
    assert back == flow


def test_no_filters_returns_a_copy():
    df = _cohort_df()
    out, flow = cohort.apply_filters(df, [])
    assert flow == []
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_summarize_flow_lines():
    flow = [
        {"step": 1, "label": "Included CPT 27130, 27447", "kind": "include", "op": "in",
         "columns": ["CPT"], "values": ["27130", "27447"],
         "n_before": 120345, "n_after": 41220, "n_removed": 79125, "n_missing": 0},
        {"step": 2, "label": "Excluded AGE < 18", "kind": "exclude", "op": "gte",
         "columns": ["AGE"], "values": [18],
         "n_before": 41220, "n_after": 40908, "n_removed": 312, "n_missing": 15,
         "n_non_numeric": 2, "non_numeric_examples": ["unknown"], "n_top_coded": 1200,
         "top_coded_examples": ["90+"]},
    ]
    assert cohort.summarize_flow(flow, 120345) == [
        "Records in file: 120,345",
        "Included CPT 27130, 27447: 41,220",
        "Excluded AGE < 18: 312 removed, 40,908 remaining (removed rows include 15 missing "
        "and 2 not numeric, e.g. 'unknown'; 1,200 top-coded values such as '90+' read as 90)",
        "Final cohort: 40,908",
    ]


def test_summarize_flow_from_real_run_and_empty_flow():
    df = _cohort_df()
    _, flow = cohort.apply_filters(df, [{"column": "AGE", "op": "gte", "values": [18]}])
    lines = cohort.summarize_flow(flow)
    assert lines[0] == "Records in file: 8"
    assert lines[1] == "Excluded AGE < 18: 2 removed, 6 remaining (removed rows include 1 missing)"
    assert lines[-1] == "Final cohort: 6"
    assert cohort.summarize_flow([], 500) == ["Records in file: 500", "Final cohort: 500"]


# ----------------------------------------------------------------------------
# validation and safety
# ----------------------------------------------------------------------------
def test_validation_lists_every_problem_at_once():
    df = _cohort_df()
    filters = [
        {"column": "age", "op": "gte", "values": [18]},
        {"column": "AGE", "op": "greater", "values": [18]},
        {"column": "CPT", "op": "in", "values": []},
        {"column": "CPT", "op": "in", "values": "M16.0-M16.9"},
        {"columns": ["AGE", "CPT"], "op": "gte", "values": [18]},
        {"column": "AGE", "op": "between", "values": [65, 18], "label": "Adults"},
        {"column": "AGE", "columns": ["CPT"], "op": "in", "values": "1"},
        {"op": "not_missing"},
        {"column": "AGE", "op": "lte", "values": "old"},
        "not a dict",
    ]
    with pytest.raises(ValueError) as ei:
        cohort.apply_filters(df, filters)
    err = ei.value
    assert isinstance(err, cohort.CohortFilterError)
    problems = err.problems
    assert len(problems) == 10
    msg = str(err)
    assert msg.startswith("10 problems with the cohort filters")
    assert "filter 1: unknown column 'age' (did you mean 'AGE'?)" in problems
    assert any(p.startswith("filter 2: unknown op 'greater'") for p in problems)
    assert "filter 3: 'in' needs at least one code in values" in problems
    assert any(p.startswith("filter 4:") and "whole-number ranges" in p for p in problems)
    assert "filter 5: 'gte' works on one column, got 2" in problems
    assert "filter 6 (Adults): 'between' low value 65 is above high value 18" in problems
    assert "filter 7: give either 'column' or 'columns', not both" in problems
    assert any(p.startswith("filter 8: no column given") for p in problems)
    assert any(p.startswith("filter 9: 'lte' needs one number") for p in problems)
    assert "filter 10: expected a dict, got str" in problems


def test_validation_single_problem_message():
    with pytest.raises(ValueError, match="Cohort filter problem: filter 1: unknown column 'DX26'"):
        cohort.apply_filters(_cohort_df(), [{"column": "DX26", "op": "not_missing"}])


def test_filters_must_be_a_list():
    with pytest.raises(ValueError, match="list of filter dicts"):
        cohort.apply_filters(_cohort_df(), {"column": "AGE", "op": "not_missing"})


def test_input_frame_is_not_modified():
    df = pd.DataFrame({
        "CPT": pd.Series([27447, 27447.0, " 27447 ", "M16.1", None], dtype=object),
        "AGE": pd.Series(["17", "90+", None, "40", "abc"], dtype=object),
        "SEX": ["male", "female", "female", None, "male"],
    })
    before = df.copy(deep=True)
    out, _ = cohort.apply_filters(df, [
        {"column": "CPT", "op": "in", "values": "27447, M16*"},
        {"column": "AGE", "op": "gte", "values": [18]},
        {"column": "SEX", "op": "neq", "values": ["MALE"]},
        {"columns": ["CPT", "SEX"], "op": "not_missing"},
    ])
    pd.testing.assert_frame_equal(df, before)
    # the result is independent and can be edited without pandas warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out["new"] = 1
    assert "new" not in df.columns
    pd.testing.assert_frame_equal(df, before)


def test_performance_300k_rows_11_code_columns():
    rng = np.random.default_rng(0)
    n = 300_000
    codes = np.arange(10000, 69999)
    df = pd.DataFrame({"CPT": rng.choice(np.r_[codes[:3000], [27130, 27447] * 500], n)})
    for k in range(1, 11):
        col = rng.choice(codes, n).astype(float)
        col[rng.random(n) < 0.6 + 0.03 * k] = np.nan
        df[f"OTHERCPT{k}"] = col
    df["OTHERCPT10"] = df["OTHERCPT10"].astype(object)
    df.loc[::7, "OTHERCPT10"] = "0275T"
    ages = rng.integers(10, 95, n).astype(object)
    ages[ages >= 90] = "90+"
    df["AGE"] = ages
    code_cols = ["CPT"] + [f"OTHERCPT{k}" for k in range(1, 11)]

    t0 = time.perf_counter()
    out, flow = cohort.apply_filters(df, [
        {"columns": code_cols, "op": "in", "values": "27130, 27447, 2913*"},
        {"columns": code_cols[1:], "op": "not_in", "values": "27132, 0275T"},
        {"column": "AGE", "op": "between", "values": [18, 100]},
    ])
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"apply_filters took {elapsed:.2f}s"
    assert flow[0]["n_before"] == n
    assert flow[-1]["n_after"] == len(out)
    # spot check the any-match against a plain pandas version
    as_num = df[code_cols].apply(pd.to_numeric, errors="coerce")
    expect = (as_num.isin([27130, 27447]) | ((as_num >= 29130) & (as_num <= 29139))).any(axis=1)
    assert flow[0]["n_after"] == int(expect.sum())
