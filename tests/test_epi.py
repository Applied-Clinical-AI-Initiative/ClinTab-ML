import numpy as np
import pandas as pd
import pytest

from clintab import epi


def test_two_by_two_odds_and_risk_ratio():
    # 100 exposed (20 events), 100 unexposed (10 events)
    res = epi.two_by_two(20, 80, 10, 90)
    assert res["corrected"] is False
    assert res["risk_exposed"] == pytest.approx(0.20)
    assert res["risk_unexposed"] == pytest.approx(0.10)
    assert res["risk_ratio"] == pytest.approx(2.0)
    assert res["odds_ratio"] == pytest.approx(2.25)
    assert res["nnt_type"] == "NNH (harm)"


def test_two_by_two_applies_haldane_anscombe_correction_on_zero_cell():
    res = epi.two_by_two(0, 10, 5, 10)
    assert res["corrected"] is True
    # values should reflect the +0.5 continuity correction, not raw zero
    assert res["table"]["a"] == 0.5


def test_nnt_nnh_benefit_case():
    # exposed group has LOWER risk than unexposed -> NNT (benefit)
    out = epi.nnt_nnh(10, 90, 20, 80)
    assert out["measure"] == "NNT"
    assert out["value"] == pytest.approx(10.0)


def test_nnt_nnh_no_difference():
    out = epi.nnt_nnh(10, 90, 10, 90)
    assert out["value"] is None
    assert "No difference" in out["interpretation"]


def test_hosmer_lemeshow_structure_and_dof():
    rng = np.random.RandomState(0)
    y_prob = np.sort(rng.uniform(0, 1, size=200))
    y_true = (rng.uniform(0, 1, size=200) < y_prob).astype(int)

    res = epi.hosmer_lemeshow(y_true, y_prob, g=10)
    assert res["dof"] == 8
    assert len(res["deciles"]) == 10
    assert 0.0 <= res["p_value"] <= 1.0


def test_kaplan_meier_overall_and_grouped():
    df = pd.DataFrame({
        "time": [5, 10, 15, 20, 25, 30, 8, 12, 18, 22],
        "event": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        "arm": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    })

    overall = epi.kaplan_meier(df, "time", "event")
    assert overall["curves"][0]["n"] == 10
    assert overall["logrank"] is None

    grouped = epi.kaplan_meier(df, "time", "event", "arm")
    assert {c["name"] for c in grouped["curves"]} == {"A", "B"}
    assert grouped["logrank"] is not None
    assert "p_value" in grouped["logrank"]


def test_cox_ph_returns_hazard_ratio_rows():
    rng = np.random.RandomState(1)
    n = 100
    df = pd.DataFrame({
        "time": rng.exponential(10, size=n),
        "event": rng.binomial(1, 0.7, size=n),
        "age": rng.normal(60, 10, size=n),
    })

    res = epi.cox_ph(df, "time", "event", ["age"])
    assert res["n"] == n
    assert res["rows"][0]["covariate"] == "age"
    assert "hazard_ratio" in res["rows"][0]
