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
    # CI should bracket the point estimate everywhere
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
