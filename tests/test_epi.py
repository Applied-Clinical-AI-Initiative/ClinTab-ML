import numpy as np
import pandas as pd
import pytest

from clintab import epi, plots

# golden values below come from R 4.3.3 with survival 3.5-8 (survfit,
# survdiff, coxph, cox.zph), epitools 0.5-10.1 and base R prop.test,
# chisq.test and fisher.test, run on exactly the data built here.


# ---------------------------------------------------------------------------
# test data (numpy RandomState streams are frozen, so these never change)
# ---------------------------------------------------------------------------
def surv_data(seed=20261030, n=180):
    rng = np.random.RandomState(seed)
    age = np.round(rng.normal(68, 9, n), 1)
    sex = np.where(rng.uniform(size=n) < 0.55, "Female", "Male")
    asa = rng.choice([1, 2, 3, 4], size=n, p=[0.15, 0.45, 0.3, 0.1])
    implant = rng.choice(["Cemented", "Hybrid", "Uncemented"], size=n)
    lin = 0.03 * (age - 68) + 0.4 * (sex == "Male") + 0.35 * (asa - 2) + 0.3 * (implant == "Uncemented")
    t_ev = rng.exponential(60 * np.exp(-lin))
    t_c = rng.uniform(5, 120, n)
    time = np.ceil(np.minimum(t_ev, t_c))  # whole months, gives ties
    event = (t_ev <= t_c).astype(int)
    return pd.DataFrame({"time": time, "event": event, "age": age, "sex": sex,
                         "asa": asa, "implant": implant})


def cr_data(seed=7, n=160):
    # 1 = revision, 2 = death (competing), 0 = censored
    rng = np.random.RandomState(seed)
    group = np.where(rng.uniform(size=n) < 0.5, "A", "B")
    h1 = np.where(group == "A", 0.010, 0.020)
    t1 = rng.exponential(1 / h1)
    t2 = rng.exponential(1 / 0.015, n)
    tc = rng.uniform(10, 100, n)
    t = np.minimum(np.minimum(t1, t2), tc)
    ev = np.where(t == tc, 0, np.where(t == t1, 1, 2))
    time = np.ceil(t / 2) * 2
    return pd.DataFrame({"time": time, "event": ev, "group": group})


def nonph_data(seed=11, n=200):
    # treatment helps early and harms late (crossing hazards)
    rng = np.random.RandomState(seed)
    trt = np.where(rng.uniform(size=n) < 0.5, "Drug", "Placebo")
    age = np.round(rng.normal(65, 8, n), 0)
    t_early = rng.exponential(np.where(trt == "Drug", 40.0, 8.0))
    t = np.where(t_early < 10, t_early, 10 + rng.exponential(np.where(trt == "Drug", 4.0, 30.0)))
    c = rng.uniform(5, 40, n)
    time = np.round(np.minimum(t, c), 1)
    event = (t <= c).astype(int)
    return pd.DataFrame({"time": time, "event": event, "trt": trt, "age": age})


def test_test_data_is_the_data_the_golden_values_came_from():
    s = surv_data()
    assert (len(s), int(s["event"].sum()), s["time"].nunique()) == (180, 118, 72)
    c = cr_data()
    assert c["event"].value_counts().to_dict() == {2: 64, 1: 61, 0: 35}


# ---------------------------------------------------------------------------
# 2x2 tables
# ---------------------------------------------------------------------------
def test_two_by_two_odds_and_risk_ratio():
    # 100 exposed (20 events), 100 unexposed (10 events)
    res = epi.two_by_two(20, 80, 10, 90)
    assert res["corrected"] is False
    assert res["correction_applied"] is False
    assert res["risk_exposed"] == pytest.approx(0.20)
    assert res["risk_unexposed"] == pytest.approx(0.10)
    assert res["risk_ratio"] == pytest.approx(2.0)
    assert res["odds_ratio"] == pytest.approx(2.25)
    assert res["nnt_type"] == "NNH (harm)"


# a, b, c, d: risk difference ci (prop.test correct = FALSE), or and rr wald
# ci (epitools oddsratio.wald / riskratio.wald, on the +0.5 table when a cell
# is zero), chisq.test default (yates) and fisher.test
GOLDEN_2X2 = [
    ((20, 80, 10, 90), dict(rd=0.1, rd_ci=(0.0020018008, 0.1979981992),
                            orr=(2.25, 0.9942948923, 5.0915478288), rr=(2.0, 0.9865632067, 4.0544791988),
                            chi2=3.1764705882, p=0.07470593331, fisher=0.07342758972)),
    ((0, 10, 5, 10), dict(rd=-0.3333333333, rd_ci=(-0.5718925525, -0.0947741142),
                          orr=(0.0909090909, 0.0044418619, 1.8605852638), rr=(0.1322314050, 0.0081102532, 2.1559307723),
                          chi2=2.34375, p=0.1257864246, fisher=0.06126482213)),
    ((10, 90, 20, 80), dict(rd=-0.1, rd_ci=(-0.1979981992, -0.0020018008),
                            orr=(0.4444444444, 0.1964039293, 1.0057378427), rr=(0.5, 0.2466408017, 1.0136197997),
                            chi2=3.1764705882, p=0.07470593331, fisher=0.07342758972)),
    ((15, 85, 20, 80), dict(rd=-0.05, rd_ci=(-0.1550913608, 0.0550913608),
                            orr=(0.7058823529, 0.3382082694, 1.4732634928), rr=(0.75, 0.4077668803, 1.3794646577),
                            chi2=0.5541125541, p=0.4566421587, fisher=0.4570106303)),
    ((3, 47, 0, 50), dict(rd=0.06, rd_ci=(-0.0058267844, 0.1258267844),
                          orr=(7.4421052632, 0.3744122561, 147.9249940340), rr=(7.0, 0.3709279658, 132.1011207444),
                          chi2=1.3745704467, p=0.2410281673, fisher=0.2424242424)),
    ((20, 133, 10, 87), dict(rd=0.0276261707, rd_ci=(-0.0530885048, 0.1083408462),
                             orr=(1.3082706767, 0.5844649422, 2.9284428200), rr=(1.2679738562, 0.6201598304, 2.5924892602),
                             chi2=0.2073111344, p=0.6488831221, fisher=0.5555080671)),
    ((12, 38, 4, 46), dict(rd=0.16, rd_ci=(0.0197563935, 0.3002436065),
                           orr=(3.6315789474, 1.0824853330, 12.1834127904), rr=(3.0, 1.0377872233, 8.6722979413),
                           chi2=3.6458333333, p=0.05620966783, fisher=0.05376475265)),
]


@pytest.mark.parametrize("cells,g", GOLDEN_2X2)
def test_two_by_two_matches_r(cells, g):
    res = epi.two_by_two(*cells)
    a, b, c, d = cells
    assert res["risk_exposed"] == pytest.approx(round(a / (a + b), 4))
    assert res["abs_risk_diff"] == pytest.approx(g["rd"], abs=5e-5)
    assert res["rd_ci"] == pytest.approx(list(g["rd_ci"]), abs=5e-5)
    assert res["odds_ratio"] == pytest.approx(g["orr"][0], abs=5e-5)
    assert res["or_ci"] == pytest.approx(list(g["orr"][1:]), abs=5e-5)
    assert res["risk_ratio"] == pytest.approx(g["rr"][0], abs=5e-5)
    assert res["rr_ci"] == pytest.approx(list(g["rr"][1:]), abs=5e-5)
    assert res["chi2"] == pytest.approx(g["chi2"], abs=5e-5)
    assert res["p_value"] == pytest.approx(g["p"], rel=1e-5)
    assert res["fisher_p_value"] == pytest.approx(g["fisher"], rel=1e-5)
    assert res["correction_applied"] is (min(cells) == 0)


def test_two_by_two_zero_cell_keeps_observed_risks():
    # the audit case: with a = 0 the exposed risk is 0, not 0.5 / 10.5
    res = epi.two_by_two(0, 10, 5, 10)
    assert res["corrected"] is True and res["correction_applied"] is True
    assert res["risk_exposed"] == 0.0
    assert res["risk_unexposed"] == pytest.approx(0.3333, abs=1e-4)
    assert res["table"] == {"a": 0, "b": 10, "c": 5, "d": 10}
    assert res["table_or_rr"]["a"] == 0.5
    assert res["nnt"] == pytest.approx(3.0)
    assert any("Haldane" in n for n in res["notes"])
    assert any("Fisher" in n for n in res["notes"])  # expected count of 2


def test_two_by_two_rejects_empty_row_and_undefined_chi_square():
    with pytest.raises(ValueError):
        epi.two_by_two(0, 0, 3, 4)
    res = epi.two_by_two(0, 10, 0, 12)  # no events at all
    assert res["chi2"] is None and res["p_value"] is None
    assert res["nnt"] is None


def test_nnt_ci_excluding_zero_follows_altman():
    # rd 0.1 (0.0020018008 to 0.1979981992): nnh 10, ci 1/0.198 to 1/0.002
    res = epi.two_by_two(20, 80, 10, 90)
    ci = res["nnt_ci"]
    assert ci["crosses_zero"] is False
    assert ci["measure"] == "NNH (harm)"
    assert ci["lower"] == pytest.approx(1 / 0.1979981992, abs=0.005)
    assert ci["upper"] == pytest.approx(1 / 0.0020018008, abs=0.005)
    out = epi.nnt_nnh(10, 90, 20, 80)
    assert out["measure"] == "NNT"
    assert out["nnt_ci"]["measure"] == "NNT (benefit)"
    assert out["nnt_ci"]["lower"] == pytest.approx(5.05, abs=0.005)
    assert out["nnt_ci"]["upper"] == pytest.approx(499.55, abs=0.005)


def test_nnt_ci_crossing_zero_runs_through_infinity():
    # rd -0.05 (-0.1550913608 to 0.0550913608), R prop.test
    out = epi.nnt_nnh(15, 85, 20, 80)
    ci = out["nnt_ci"]
    assert out["measure"] == "NNT" and out["value"] == pytest.approx(20.0)
    assert ci["crosses_zero"] is True
    assert ci["lower"] is None and ci["upper"] is None
    assert ci["benefit_bound"] == pytest.approx(1 / 0.1550913608, abs=0.005)
    assert ci["harm_bound"] == pytest.approx(1 / 0.0550913608, abs=0.005)
    assert ci["text"].startswith("95% CI NNT (benefit) 6.45 to infinity to NNH (harm) 18.15")
    assert out["value_rounded_up"] == 20


def test_nnt_nnh_benefit_case():
    # exposed group has lower risk than unexposed -> nnt (benefit)
    out = epi.nnt_nnh(10, 90, 20, 80)
    assert out["measure"] == "NNT"
    assert out["value"] == pytest.approx(10.0)


def test_nnt_nnh_no_difference():
    out = epi.nnt_nnh(10, 90, 10, 90)
    assert out["value"] is None
    assert "No difference" in out["interpretation"]
    assert out["nnt_ci"]["crosses_zero"] is True


def test_nnt_nnh_uses_the_unrounded_risk_difference():
    # ard = 20/153 - 10/97 = 0.0276..., 1 / round(ard, 4) would give 36.2
    out = epi.nnt_nnh(20, 133, 10, 87)
    exact = 1 / abs(20 / 153 - 10 / 97)
    assert out["value"] == pytest.approx(exact, abs=0.01)
    assert out["value"] == epi.two_by_two(20, 133, 10, 87)["nnt"]
    assert out["value_rounded_up"] == 37


def test_two_by_two_rejects_negative_and_non_finite_counts():
    with pytest.raises(ValueError):
        epi.two_by_two(-1, 2, 3, 4)
    with pytest.raises(ValueError):
        epi.two_by_two(float("nan"), 2, 3, 4)


def test_hosmer_lemeshow_structure_and_dof():
    rng = np.random.RandomState(0)
    y_prob = np.sort(rng.uniform(0, 1, size=200))
    y_true = (rng.uniform(0, 1, size=200) < y_prob).astype(int)

    res = epi.hosmer_lemeshow(y_true, y_prob, g=10)
    assert res["dof"] == 8
    assert len(res["deciles"]) == 10
    assert 0.0 <= res["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# kaplan-meier
# ---------------------------------------------------------------------------
def test_kaplan_meier_overall_and_grouped():
    df = pd.DataFrame({
        "time": [5, 10, 15, 20, 25, 30, 8, 12, 18, 22],
        "event": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        "arm": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    })

    overall = epi.kaplan_meier(df, "time", "event")
    assert overall["curves"][0]["n"] == 10
    assert overall["curves"][0]["name"] == "Overall"
    assert overall["logrank"] is None

    grouped = epi.kaplan_meier(df, "time", "event", "arm")
    assert {c["name"] for c in grouped["curves"]} == {"A", "B"}
    assert grouped["logrank"] is not None
    assert "p_value" in grouped["logrank"]


def test_kaplan_meier_overall_matches_r_survfit():
    # survfit(Surv(time, event) ~ 1, conf.type = "log-log"), summary(times =)
    res = epi.kaplan_meier(surv_data(), "time", "event",
                           time_points=[0, 20, 40, 60, 80, 100])
    c = res["curves"][0]
    assert c["median_survival"] == 30
    assert c["median_ci"] == [23, 40]
    assert c["events"] == 118
    golden = {20: (101, 0.6407832542, 0.5638477036, 0.7077175202),
              40: (52, 0.4209822824, 0.3419532124, 0.4978118890),
              60: (24, 0.2575963382, 0.1849125113, 0.3362293385),
              80: (13, 0.1926303149, 0.1234134971, 0.2734880417),
              100: (6, 0.1765777887, 0.1084713370, 0.2583058536)}
    for row in c["survival_at"]:
        if row["time"] == 0:
            assert (row["at_risk"], row["survival"]) == (180, 1.0)
            continue
        nr, s, lo, hi = golden[row["time"]]
        assert row["at_risk"] == nr
        assert row["survival"] == pytest.approx(s, abs=5e-5)
        assert row["lower"] == pytest.approx(lo, abs=5e-5)
        assert row["upper"] == pytest.approx(hi, abs=5e-5)
    assert c["at_risk"] == [180, 101, 52, 24, 13, 6]
    assert res["risk_table"]["groups"][0]["at_risk"] == c["at_risk"]


def test_kaplan_meier_groups_match_r_medians_and_logrank():
    res = epi.kaplan_meier(surv_data(), "time", "event", "implant",
                           time_points=[40, 80])
    by = {c["name"]: c for c in res["curves"]}
    assert [c["name"] for c in res["curves"]] == ["Cemented", "Hybrid", "Uncemented"]
    # print(survfit(..., conf.type = "log-log")): median, 0.95LCL, 0.95UCL
    assert (by["Cemented"]["median_survival"], by["Cemented"]["median_ci"]) == (26, [19, 50])
    assert (by["Hybrid"]["median_survival"], by["Hybrid"]["median_ci"]) == (48, [34, 53])
    assert (by["Uncemented"]["median_survival"], by["Uncemented"]["median_ci"]) == (21, [13, 25])
    hy40 = by["Hybrid"]["survival_at"][0]
    assert hy40["at_risk"] == 26
    assert hy40["survival"] == pytest.approx(0.60271417906, abs=5e-5)
    assert [hy40["lower"], hy40["upper"]] == pytest.approx([0.462177698046, 0.7173827483], abs=5e-5)
    un80 = by["Uncemented"]["survival_at"][1]
    assert un80["at_risk"] == 1
    assert [un80["survival"], un80["lower"], un80["upper"]] == pytest.approx(
        [0.04450400994, 0.003988712862, 0.1732043350], abs=5e-5)
    # survdiff(Surv(time, event) ~ implant): 3 groups, 2 df
    lr = res["logrank"]
    assert lr["df"] == 2
    assert lr["test_statistic"] == pytest.approx(16.0362125234, abs=5e-5)
    assert lr["p_value"] == pytest.approx(0.0003294433119, rel=1e-5)
    assert res["risk_table"]["time_points"] == [40, 80]
    assert [g["at_risk"] for g in res["risk_table"]["groups"]] == [[17, 7], [26, 5], [9, 1]]


@pytest.mark.parametrize("group,chisq,df,p", [
    ("sex", 6.2266052156, 1, 0.01258447707),
    ("asa", 15.4567256058, 3, 0.001465161043),
])
def test_logrank_two_and_four_groups_match_survdiff(group, chisq, df, p):
    lr = epi.kaplan_meier(surv_data(), "time", "event", group)["logrank"]
    assert lr["df"] == df
    assert lr["test_statistic"] == pytest.approx(chisq, abs=5e-5)
    assert lr["p_value"] == pytest.approx(p, rel=1e-5)


@pytest.mark.parametrize("time,event,median,ci", [
    # survival is exactly 0.5 between t = 2 and 3, R takes the midpoint
    ([1, 2, 3, 4], [1, 1, 1, 1], 2.5, [1, None]),
    ([2, 3, 5, 7, 8, 10, 12, 15], [1, 0, 1, 1, 0, 1, 1, 0], 10, [2, None]),
    # curve ends at exactly 0.5, R averages with the last follow-up time
    ([1, 2, 3, 4], [1, 1, 0, 0], 3, [1, None]),
])
def test_median_follows_r_quantile_rule(time, event, median, ci):
    c = epi.kaplan_meier(pd.DataFrame({"time": time, "event": event}),
                         "time", "event")["curves"][0]
    assert c["median_survival"] == median
    assert c["median_ci"] == ci


def test_kaplan_meier_default_time_points_and_event_coding():
    res = epi.kaplan_meier(surv_data(), "time", "event")
    assert res["time_points"] == [0, 25, 50, 75, 100]
    bad = pd.DataFrame({"time": [1, 2, 3], "event": [0, 1, 2]})
    with pytest.raises(ValueError, match="coded 0"):
        epi.kaplan_meier(bad, "time", "event")


# ---------------------------------------------------------------------------
# cox
# ---------------------------------------------------------------------------
# coxph(Surv(time, event) ~ age + sex + factor(asa) + implant), efron ties
GOLDEN_COX = {
    "age": (1.030397428736, 1.010341336224, 1.05085165090, 0.002828202081990),
    "sex: Male vs Female": (1.586895529802, 1.090084013681, 2.31013150445, 0.015945938428246),
    "asa: 2 vs 1": (1.598430457603, 0.913389081891, 2.79725253832, 0.100450130535769),
    "asa: 3 vs 1": (1.717352110853, 0.926683666827, 3.18263759061, 0.085785835490159),
    "asa: 4 vs 1": (3.873657727016, 1.847314214689, 8.12272436749, 0.000337759752896),
    "implant: Hybrid vs Cemented": (0.728836614673, 0.443324232361, 1.19822642687, 0.212394555921373),
    "implant: Uncemented vs Cemented": (1.821288474633, 1.128612753765, 2.93908756282, 0.014069423215517),
}


def test_cox_ph_matches_r_coxph_with_categorical_terms():
    res = epi.cox_ph(surv_data(), "time", "event", ["age", "sex", "asa", "implant"],
                     categorical=["asa"])
    rows = {r["covariate"]: r for r in res["rows"]}
    assert list(rows) == list(GOLDEN_COX)
    for name, (hr, lo, hi, p) in GOLDEN_COX.items():
        assert rows[name]["hazard_ratio"] == pytest.approx(hr, abs=5e-5)
        assert rows[name]["hr_lower"] == pytest.approx(lo, abs=5e-5)
        assert rows[name]["hr_upper"] == pytest.approx(hi, abs=5e-5)
        assert rows[name]["p_value"] == pytest.approx(p, rel=1e-4)
    assert (res["n"], res["n_events"]) == (180, 118)
    assert res["concordance"] == pytest.approx(0.6737446915, abs=5e-5)
    assert res["reference_levels"] == {"sex": "Female", "asa": "1", "implant": "Cemented"}
    assert res["categorical"] == ["sex", "asa", "implant"]
    assert rows["asa: 3 vs 1"]["reference"] == "1" and rows["asa: 3 vs 1"]["level"] == "3"


def test_cox_ph_test_matches_r_cox_zph_rank():
    # cox.zph(fit, transform = "rank") in survival 3.5-8, terms and singles
    res = epi.cox_ph(surv_data(), "time", "event", ["age", "sex", "asa", "implant"],
                     categorical=["asa"])
    ph = res["ph_test"]
    assert ph["transform"] == "rank"
    terms = {t["variable"]: t for t in ph["terms"]}
    golden_terms = {"age": (0.001508799456, 1), "sex": (0.630339378236, 1),
                    "asa": (1.829452092826, 3), "implant": (2.768490441394, 2)}
    for name, (chisq, df) in golden_terms.items():
        assert terms[name]["df"] == df
        assert terms[name]["test_statistic"] == pytest.approx(chisq, abs=5e-5)
    assert terms["asa"]["p_value"] == pytest.approx(0.6085468989, rel=1e-5)
    assert ph["global"]["test_statistic"] == pytest.approx(4.367355956974, abs=5e-5)
    assert ph["global"]["df"] == 7
    assert ph["global"]["p_value"] == pytest.approx(0.7366253181, rel=1e-5)
    singles = {r["covariate"]: r["ph_test_statistic"] for r in res["rows"]}
    golden_singles = {"asa: 2 vs 1": 0.001783288602, "asa: 3 vs 1": 0.527408974948,
                      "asa: 4 vs 1": 0.249239205755, "implant: Hybrid vs Cemented": 2.631119384431,
                      "implant: Uncemented vs Cemented": 0.323698859610}
    for name, chisq in golden_singles.items():
        assert singles[name] == pytest.approx(chisq, abs=5e-5)
    assert res["ph_notes"] == []


def test_cox_ph_flags_non_proportional_hazards():
    df = nonph_data()
    res = epi.cox_ph(df, "time", "event", ["trt", "age"])
    row = res["rows"][0]
    assert row["covariate"] == "trt: Placebo vs Drug"
    assert row["hazard_ratio"] == pytest.approx(1.313235387459, abs=5e-5)
    assert [row["hr_lower"], row["hr_upper"]] == pytest.approx([0.95025360955, 1.81487043621], abs=5e-5)
    assert res["concordance"] == pytest.approx(0.6293960031, abs=5e-5)
    terms = {t["variable"]: t for t in res["ph_test"]["terms"]}
    assert terms["trt"]["test_statistic"] == pytest.approx(99.4044301817, abs=5e-4)
    assert terms["age"]["test_statistic"] == pytest.approx(1.4293275685, abs=5e-5)
    assert res["ph_test"]["global"]["test_statistic"] == pytest.approx(99.5461267725, abs=5e-4)
    assert len(res["ph_notes"]) == 1 and "trt" in res["ph_notes"][0]
    # R's default transform, cox.zph(fit, transform = "km")
    km = epi.cox_ph(df, "time", "event", ["trt", "age"], ph_transform="km")
    terms = {t["variable"]: t for t in km["ph_test"]["terms"]}
    assert terms["trt"]["test_statistic"] == pytest.approx(97.45676423487, abs=5e-4)
    assert terms["age"]["test_statistic"] == pytest.approx(1.43251398872, abs=5e-5)
    assert km["ph_test"]["global"]["test_statistic"] == pytest.approx(97.58176755282, abs=5e-4)


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


def test_cox_ph_single_numeric_matches_r():
    # coxph(Surv(time, event) ~ age)
    r = epi.cox_ph(surv_data(), "time", "event", ["age"])["rows"][0]
    assert r["hazard_ratio"] == pytest.approx(1.0203209244, abs=5e-5)
    assert [r["hr_lower"], r["hr_upper"]] == pytest.approx([1.00193906366, 1.03904002402], abs=5e-5)
    assert r["p_value"] == pytest.approx(0.0300971554973, rel=1e-4)


def test_cox_ph_text_covariate_no_longer_fails_and_reference_can_be_set():
    df = surv_data()
    res = epi.cox_ph(df, "time", "event", ["implant"], reference={"implant": "Hybrid"})
    rows = {r["covariate"]: r for r in res["rows"]}
    assert set(rows) == {"implant: Cemented vs Hybrid", "implant: Uncemented vs Hybrid"}
    assert res["categorical"] == ["implant"]
    # asa left numeric enters as one linear term
    lin = epi.cox_ph(df, "time", "event", ["asa"])
    assert [r["covariate"] for r in lin["rows"]] == ["asa"]
    # missing values in a text covariate are dropped, not fatal
    df2 = df.copy()
    df2.loc[:9, "sex"] = np.nan
    res2 = epi.cox_ph(df2, "time", "event", ["sex"])
    assert res2["n"] == 170 and res2["n_excluded"] == 10
    with pytest.raises(ValueError, match="only one level"):
        epi.cox_ph(df.assign(k="x"), "time", "event", ["k"])


def test_cox_reference_level_matches_r_exp_minus_coef():
    # R exp(-coef) for implantHybrid in the full model is the Cemented vs Hybrid hr
    res = epi.cox_ph(surv_data(), "time", "event", ["age", "sex", "asa", "implant"],
                     categorical=["asa"], reference={"implant": "Hybrid"})
    rows = {r["covariate"]: r for r in res["rows"]}
    assert rows["implant: Cemented vs Hybrid"]["hazard_ratio"] == pytest.approx(1.372049619721, abs=5e-5)
    # the global ph test does not depend on the coding
    assert res["ph_test"]["global"]["test_statistic"] == pytest.approx(4.367355956974, abs=5e-5)


# ---------------------------------------------------------------------------
# competing risks
# ---------------------------------------------------------------------------
def test_cumulative_incidence_matches_r_survfit_multistate():
    # survfit(Surv(time, factor(event)) ~ 1): aalen-johansen, ij se, log ci
    res = epi.cumulative_incidence(cr_data(), "time", "event", 1,
                                   time_points=[0, 20, 40, 60, 80])
    c = res["curves"][0]
    golden = {20: (79, 0.2810268155, 0.2185841807, 0.3613073496),
              40: (33, 0.3641375242, 0.2931150754, 0.4523688737),
              60: (13, 0.4191525926, 0.3423647161, 0.5131629739),
              80: (7, 0.4306980051, 0.3529754454, 0.5255344927)}
    for row in c["at_times"][1:]:
        nr, cif, lo, hi = golden[row["time"]]
        assert row["at_risk"] == nr
        assert row["cif"] == pytest.approx(cif, abs=5e-5)
        assert [row["lower"], row["upper"]] == pytest.approx([lo, hi], abs=5e-5)
    assert c["at_times"][0]["cif"] == 0.0
    # full precision on the curve itself
    tl = c["timeline"]
    i2, i94 = tl.index(2.0), tl.index(94.0)
    assert c["cif"][i2] == pytest.approx(0.0125, abs=1e-10)
    assert c["se"][i2] == pytest.approx(0.008783418327, abs=1e-9)
    assert c["cif"][i94] == pytest.approx(0.4678082597, abs=1e-9)
    assert c["se"][i94] == pytest.approx(0.049885632896, abs=1e-9)
    assert c["lower"][i94] == pytest.approx(0.379575675002, abs=1e-9)
    assert c["upper"][i94] == pytest.approx(0.57655055975, abs=1e-9)
    assert (c["events"], c["competing_events"], c["censored"]) == (61, 64, 35)
    assert res["competing_codes"] == [2]


def test_cumulative_incidence_by_group_matches_r():
    res = epi.cumulative_incidence(cr_data(), "time", "event", 1, group_col="group",
                                   time_points=[20, 40, 60, 80])
    by = {c["name"]: {r["time"]: r for r in c["at_times"]} for c in res["curves"]}
    # survfit(Surv(time, factor(event)) ~ group), also equal to cmprsk timepoints()
    assert by["A"][20]["at_risk"] == 45
    assert by["A"][20]["cif"] == pytest.approx(0.2090130202, abs=5e-5)
    assert [by["A"][20]["lower"], by["A"][20]["upper"]] == pytest.approx([0.1368102463, 0.3193214235], abs=5e-5)
    assert by["A"][60]["cif"] == pytest.approx(0.3433400680, abs=5e-5)
    assert [by["A"][60]["lower"], by["A"][60]["upper"]] == pytest.approx([0.2444105685, 0.4823130319], abs=5e-5)
    assert by["B"][40]["at_risk"] == 12
    assert by["B"][40]["cif"] == pytest.approx(0.4527380588, abs=5e-5)
    assert [by["B"][40]["lower"], by["B"][40]["upper"]] == pytest.approx([0.3501524005, 0.5853786796], abs=5e-5)
    assert by["B"][80]["cif"] == pytest.approx(0.4963843987, abs=5e-5)
    # cmprsk 2.2-12 cuminc(time, event, group)$Tests
    gray = res["gray_test"]
    assert gray["df"] == 1
    assert gray["test_statistic"] == pytest.approx(3.2093345266, abs=5e-5)
    assert gray["p_value"] == pytest.approx(0.07321925761, rel=1e-5)
    death = epi.cumulative_incidence(cr_data(), "time", "event", 2, group_col="group")
    assert death["gray_test"]["test_statistic"] == pytest.approx(0.3914175493, abs=5e-5)


def cr3_data(seed, n, bins, codes3=False):
    rng = np.random.RandomState(seed)
    group = rng.choice(["Cemented", "Hybrid", "Uncemented"], size=n, p=[0.5, 0.3, 0.2])
    h1 = np.select([group == "Cemented", group == "Hybrid"], [0.01, 0.02], 0.03)
    t1 = rng.exponential(1 / h1)
    t2 = rng.exponential(1 / 0.02, n)
    t3 = rng.exponential(1 / 0.01, n)
    tc = rng.uniform(5, 80, n)
    t = np.minimum.reduce([t1, t2, t3, tc])
    ev = np.select([t == tc, t == t1, t == t2], [0, 1, 2], 3 if codes3 else 2)
    time = np.ceil(t / bins) * bins
    return pd.DataFrame({"time": time, "event": ev, "group": group})


@pytest.mark.parametrize("args,k,stat,p", [
    # cmprsk 2.2-12 cuminc(time, event, group)$Tests, three groups
    ((3, 300, 5), 1, 7.8172402177, 0.02006817386),
    ((3, 300, 5), 2, 0.1819614047, 0.9130353302),
    ((5, 120, 1, True), 3, 2.3943694082, 0.3020433575),
    ((9, 500, 10, True), 1, 38.2596326527, 4.92069363e-09),
    ((9, 500, 10, True), 2, 6.5492825093, 0.03783043829),
])
def test_gray_test_three_groups_matches_cmprsk(args, k, stat, p):
    g = epi.cumulative_incidence(cr3_data(*args), "time", "event", k, group_col="group")["gray_test"]
    assert g["df"] == 2
    assert g["test_statistic"] == pytest.approx(stat, abs=5e-5)
    assert g["p_value"] == pytest.approx(p, rel=1e-5)


def test_cif_with_two_competing_codes_matches_r():
    # codes 2 and 3 both compete with 1; survfit(Surv(time, factor(event)) ~ group)
    df = cr3_data(9, 500, 10, True)
    res = epi.cumulative_incidence(df, "time", "event", 1, group_col="group",
                                   time_points=[20, 40, 60])
    assert res["competing_codes"] == [2, 3]
    un = {r["time"]: r for r in res["curves"][2]["at_times"]}
    assert un[40]["at_risk"] == 10
    assert [un[40]["cif"], un[40]["lower"], un[40]["upper"]] == pytest.approx(
        [0.4959734707, 0.4012935109, 0.6129919297], abs=5e-5)
    # past the last follow-up of the group there is no estimate
    assert un[60]["at_risk"] == 0 and un[60]["cif"] is None
    res3 = epi.cumulative_incidence(df, "time", "event", 3, group_col="group",
                                    time_points=[20])
    hy = res3["curves"][1]["at_times"][0]
    assert [hy["cif"], hy["lower"], hy["upper"]] == pytest.approx(
        [0.2310897436, 0.1727414202, 0.3091468713], abs=5e-5)


def test_naive_one_minus_km_overstates_and_matches_r():
    # 1 - survfit(Surv(time, event == 1) ~ 1, conf.type = "log-log")
    res = epi.cumulative_incidence(cr_data(), "time", "event", 1,
                                   time_points=[20, 40, 60, 80])
    golden = {20: 0.3189469036, 40: 0.4562819290, 60: 0.5987651913, 80: 0.6322014254}
    for row in res["curves"][0]["at_times"]:
        assert row["naive"] == pytest.approx(golden[row["time"]], abs=5e-5)
        assert row["naive"] > row["cif"]
    nv = res["naive"][0]
    i = max(j for j, t in enumerate(nv["timeline"]) if t <= 20)
    assert [nv["lower"][i], nv["upper"][i]] == pytest.approx([0.2468584758, 0.4057378285], abs=1e-8)


def test_cif_equals_one_minus_km_without_competing_events():
    df = surv_data()
    res = epi.cumulative_incidence(df, "time", "event", 1, time_points=[20, 40, 60])
    km = epi.kaplan_meier(df, "time", "event", time_points=[20, 40, 60])
    for row, s in zip(res["curves"][0]["at_times"], km["curves"][0]["survival_at"]):
        assert row["cif"] == pytest.approx(1 - s["survival"], abs=1e-4)
        assert row["naive"] == pytest.approx(row["cif"], abs=1e-4)
    assert res["competing_codes"] == []


def test_cumulative_incidence_rejects_missing_event_code():
    with pytest.raises(ValueError, match="event of interest"):
        epi.cumulative_incidence(cr_data(), "time", "event", 3)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def test_km_plot_old_call_and_risk_table_both_render():
    res = epi.kaplan_meier(surv_data(), "time", "event", "implant")
    assert plots.fig_to_png(plots.km_plot(res["curves"])).startswith("data:image/png;base64,")
    fig = plots.km_plot(res["curves"], risk_table=res["risk_table"])
    assert len(fig.axes) == 2
    texts = [t.get_text() for t in fig.axes[1].texts]
    assert "64" in texts and "Number at risk" in fig.axes[1].get_title(loc="left")
    assert plots.fig_to_png(fig).startswith("data:image/png;base64,")


def test_cif_plot_with_naive_and_risk_table_renders():
    res = epi.cumulative_incidence(cr_data(), "time", "event", 1, group_col="group")
    fig = plots.cif_plot(res["curves"], naive=res["naive"], risk_table=res["risk_table"])
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any("1 minus KM" in lab for lab in labels)
    assert plots.fig_to_png(fig).startswith("data:image/png;base64,")
    assert plots.fig_to_pdf_bytes(plots.cif_plot(res["curves"]))[:4] == b"%PDF"
