import json

import numpy as np
import pandas as pd
import pytest

from clintab import plots, teaching


FEATS = ["age", "bmi", "albumin", "creatinine", "asa", "sex", "smoker"]
COLTYPES = {"asa": "categorical"}


def _synthetic(n=4000, seed=0):
    """A few numeric and categorical predictors, about 10% prevalence and
    some missing values."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.normal(65, 10, n),
        "bmi": rng.normal(30, 6, n),
        "albumin": rng.normal(3.8, 0.5, n),
        "creatinine": rng.lognormal(0, 0.4, n),
        "asa": rng.choice([1, 2, 3, 4], n, p=[0.1, 0.4, 0.4, 0.1]),
        "sex": rng.choice(["M", "F"], n),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]).astype(object),
    })
    lp = (-2.9 + 0.06 * (df.age - 65) + 0.05 * (df.bmi - 30)
          - 1.2 * (df.albumin - 3.8) + 0.8 * np.log(df.creatinine)
          + 0.5 * (df.asa - 2.5) + 0.7 * (df.smoker == "yes"))
    df["complication"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-lp))).astype(int)
    for c in ("bmi", "albumin", "smoker"):
        df.loc[rng.uniform(size=n) < 0.1, c] = np.nan
    return df


def _arm(result, name):
    return next(a for a in result["arms"] if a["arm"] == name)


@pytest.fixture(scope="module")
def logistic_result():
    return teaching.leakage_demo(_synthetic(6000), "complication", FEATS,
                                 coltypes=COLTYPES, model="logistic",
                                 seeds=(1, 2, 3))


@pytest.fixture(scope="module")
def forest_result():
    return teaching.leakage_demo(_synthetic(2000), "complication", FEATS,
                                 coltypes=COLTYPES, model="random_forest",
                                 seeds=(1,),
                                 arms=["correct", "apparent", "smote_before_split"])


def test_synthetic_data_has_about_ten_percent_prevalence():
    df = _synthetic(6000)
    assert 0.07 < df["complication"].mean() < 0.13
    assert df["bmi"].isna().any() and df["smoker"].isna().any()


def test_result_shape(logistic_result):
    r = logistic_result
    for key in ("outcome", "model", "n_used", "subsampled", "prevalence",
                "lockbox_n", "seeds", "arms", "per_seed", "runtime_s"):
        assert key in r
    assert r["outcome"] == "complication" and r["model"] == "logistic"
    assert r["subsampled"] is False and r["n_used"] == 6000
    assert r["lockbox_n"] == 1200
    assert [a["arm"] for a in r["arms"]] == teaching.ARM_NAMES
    for a in r["arms"]:
        assert a["label"] and a["explanation"]
        for q in ("reported_auroc", "lockbox_auroc", "optimism",
                  "calibration_slope", "citl"):
            assert set(a[q]) == {"mean", "min", "max"}
            assert a[q]["min"] <= a[q]["mean"] <= a[q]["max"]
    assert len(r["per_seed"]) == len(teaching.ARM_NAMES) * 3
    # valid json without nan, ready for the web layer
    json.dumps(r, allow_nan=False)


def test_user_facing_text_has_no_long_dashes(logistic_result):
    text = json.dumps(logistic_result, ensure_ascii=False)
    assert "\u2013" not in text and "\u2014" not in text


def test_correct_arm_optimism_is_near_zero(logistic_result):
    opt = _arm(logistic_result, "correct")["optimism"]["mean"]
    assert abs(opt) < 0.06


def test_preprocessing_before_split_changes_little(logistic_result):
    correct = _arm(logistic_result, "correct")["optimism"]["mean"]
    pre = _arm(logistic_result, "preprocess_before_split")["optimism"]["mean"]
    assert abs(pre - correct) < 0.02


def test_smote_in_training_fold_keeps_auroc_but_overpredicts(logistic_result):
    arm = _arm(logistic_result, "smote_in_training_fold")
    assert abs(arm["optimism"]["mean"]) < 0.06
    # risks run too high on the lockbox, the correct arm stays near 0
    assert arm["citl"]["mean"] < -0.5
    assert abs(_arm(logistic_result, "correct")["citl"]["mean"]) < 0.3


def test_smote_before_split_random_forest_is_clearly_optimistic(forest_result):
    smote = _arm(forest_result, "smote_before_split")["optimism"]["mean"]
    correct = _arm(forest_result, "correct")["optimism"]["mean"]
    assert smote > 0.15
    assert smote > correct + 0.1


def test_apparent_beats_lockbox_for_random_forest(forest_result):
    arm = _arm(forest_result, "apparent")
    assert arm["reported_auroc"]["mean"] >= arm["lockbox_auroc"]["mean"]
    assert arm["optimism"]["min"] > 0


@pytest.mark.parametrize("model,arms", [
    ("logistic", None),
    ("random_forest", ["smote_before_split", "tune_on_test"]),
])
def test_same_seeds_give_identical_results(model, arms):
    df = _synthetic(1500, seed=3)
    runs = [teaching.leakage_demo(df, "complication", FEATS, coltypes=COLTYPES,
                                  model=model, seeds=(4,), arms=arms)
            for _ in range(2)]
    for r in runs:
        r.pop("runtime_s")
    assert runs[0] == runs[1]


@pytest.mark.parametrize("values", [
    lambda rng, n: rng.choice(["none", "minor", "major"], n),
    lambda rng, n: rng.normal(5, 2, n),
    lambda rng, n: np.ones(n),
])
def test_rejects_non_binary_outcome(values):
    df = _synthetic(600)
    df["complication"] = values(np.random.RandomState(0), len(df))
    with pytest.raises(ValueError, match="binary outcome"):
        teaching.leakage_demo(df, "complication", FEATS, seeds=(1,))


def test_rejects_bad_arguments():
    df = _synthetic(600)
    with pytest.raises(ValueError, match="Unknown arm"):
        teaching.leakage_demo(df, "complication", FEATS, arms=["peeking"])
    with pytest.raises(ValueError, match="Unknown model"):
        teaching.leakage_demo(df, "complication", FEATS, model="svm")
    with pytest.raises(ValueError, match="not in the data"):
        teaching.leakage_demo(df, "missing_col", FEATS)
    with pytest.raises(ValueError, match="at least 50"):
        teaching.leakage_demo(df.iloc[:200], "complication", FEATS)


def test_subsample_is_stratified_and_logged():
    df = _synthetic(3000)
    df.loc[:9, "complication"] = np.nan
    msgs = []
    r = teaching.leakage_demo(df, "complication", FEATS, coltypes=COLTYPES,
                              seeds=(1,), max_rows=1500, arms=["correct"],
                              progress=msgs.append)
    assert r["subsampled"] is True
    assert r["n_input"] == 2990 and r["n_used"] == 1500
    full_prev = df["complication"].mean()
    assert abs(r["prevalence"] - full_prev) < 0.005
    assert any("1,500" in n for n in r["notes"])
    assert any("missing outcome" in n for n in r["notes"])
    assert msgs and all(isinstance(m, str) for m in msgs) and msgs[-1] == "Done"
    assert [a["arm"] for a in r["arms"]] == ["correct"]


def test_calibration_slope_and_citl():
    rng = np.random.RandomState(0)
    lp = rng.normal(-2, 1, 20000)
    p = 1 / (1 + np.exp(-lp))
    y = (rng.uniform(size=len(p)) < p).astype(int)
    slope, citl = teaching.calibration_slope_citl(y, p)
    assert abs(slope - 1) < 0.1 and abs(citl) < 0.1
    # too extreme predictions give a slope near 0.5
    slope, _ = teaching.calibration_slope_citl(y, 1 / (1 + np.exp(-2 * lp)))
    assert abs(slope - 0.5) < 0.06
    # predicting too high gives a negative citl
    _, citl = teaching.calibration_slope_citl(y, 1 / (1 + np.exp(-(lp + 1))))
    assert abs(citl + 1) < 0.1


def test_teaching_plot_renders(logistic_result, forest_result):
    for r in (logistic_result, forest_result):
        png = plots.fig_to_png(plots.teaching_plot(r))
        assert png.startswith("data:image/png;base64,")


def test_teaching_plot_handles_missing_numbers():
    empty = {"mean": None, "min": None, "max": None}
    r = {"model": "logistic", "seeds": [1], "arms": [
        {"arm": "correct", "label": "Done correctly", "reported_auroc": empty,
         "lockbox_auroc": empty, "optimism": empty,
         "calibration_slope": empty, "citl": empty},
        {"arm": "apparent", "label": "Scored on the training patients",
         "reported_auroc": {"mean": 0.9, "min": 0.88, "max": 0.92},
         "lockbox_auroc": {"mean": 0.7, "min": 0.69, "max": 0.71},
         "optimism": {"mean": 0.2, "min": 0.19, "max": 0.21},
         "calibration_slope": {"mean": 9.0, "min": 8.0, "max": 10.0},
         "citl": empty},
    ]}
    assert plots.fig_to_png(plots.teaching_plot(r)).startswith("data:image/png")
