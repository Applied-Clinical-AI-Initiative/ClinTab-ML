import numpy as np
import pandas as pd
import pytest

from clintab import spline


def _binary_outcome_frame(n=300, seed=0):
    rng = np.random.RandomState(seed)
    age = rng.uniform(20, 90, size=n)
    logit = (age - 55) / 15
    p = 1 / (1 + np.exp(-logit))
    outcome = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"age": age, "outcome": outcome})


def test_fit_rcs_returns_curve_with_expected_shape():
    df = _binary_outcome_frame()
    res = spline.fit_rcs(df, "age", "outcome", n_knots=4)

    assert res["n_knots"] == 4
    assert len(res["knots"]) <= 4
    assert len(res["x"]) == 200
    assert len(res["logodds"]) == len(res["x"])
    assert len(res["lower"]) == len(res["x"])
    assert len(res["upper"]) == len(res["x"])
    # ci should bracket the point estimate everywhere
    assert all(lo <= m <= hi for lo, m, hi in
               zip(res["lower"], res["logodds"], res["upper"]))


def test_fit_rcs_defaults_to_4_knots_on_invalid_input():
    df = _binary_outcome_frame()
    res = spline.fit_rcs(df, "age", "outcome", n_knots=99)
    assert res["n_knots"] == 4


def test_fit_rcs_rejects_non_binary_outcome():
    df = _binary_outcome_frame()
    df["outcome"] = pd.Series(["a", "b", "c"] * (len(df) // 3 + 1))[:len(df)]
    with pytest.raises(ValueError):
        spline.fit_rcs(df, "age", "outcome", n_knots=4)


def test_fit_rcs_curve_is_on_the_log_odds_scale_not_probability():
    df = _binary_outcome_frame(n=600)
    res = spline.fit_rcs(df, "age", "outcome", n_knots=4)
    logodds = np.array(res["logodds"])
    # a probability curve would sit inside [0, 1] everywhere, log-odds here
    # runs well below 0 at young ages and above 0 at old ages
    assert logodds.min() < -0.5 and logodds.max() > 0.5
    # and it should be the logit of the model's own fitted probabilities
    assert np.all(np.diff(np.sign(logodds)) >= 0)    # crosses zero once, going up


def test_fit_rcs_reports_the_number_of_knots_actually_used():
    rng = np.random.RandomState(0)
    n = 400
    # 70% of the predictor sits on one value, so several quantile knots tie
    x = np.where(rng.uniform(size=n) < 0.7, 5.0, rng.uniform(0, 10, size=n))
    df = pd.DataFrame({"x": x, "y": (rng.uniform(size=n) < 1 / (1 + np.exp(-(x - 5) / 2))).astype(int)})
    res = spline.fit_rcs(df, "x", "y", n_knots=5)
    assert len(res["knots"]) == 3
    assert res["n_knots"] == 3
