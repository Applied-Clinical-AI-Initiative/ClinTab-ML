import math

import pytest
from scipy import stats

from clintab import samplesize as ss

# golden values from R pmsampsize 1.1.3 (CRAN) on R 4.3.3, pmsampsize(type =
# "b" or "c", ...). The Python pmsampsize 0.1.0 port gives the same answers
# for csrsquared input but different ones for cstatistic input, because it
# simulates with numpy's random stream instead of R's.


def test_riley_bmj_worked_example_csrsquared():
    # pmsampsize(type = "b", csrsquared = 0.288, parameters = 24, prevalence = 0.174)
    r = ss.binary_sample_size(0.174, 24, csrsquared=0.288)
    assert [c["n"] for c in r["criteria"]] == [623, 662, 221]
    assert [c["epp"] for c in r["criteria"]] == [4.52, 4.80, 1.60]
    assert [c["shrinkage"] for c in r["criteria"]] == [0.9, 0.905, None]
    assert r["min_n"] == 662
    assert r["min_events"] == 116  # R prints ceiling(115.188)
    assert r["events_exact"] == pytest.approx(115.188)
    assert r["epp"] == 4.8
    assert r["final_shrinkage"] == 0.905
    assert round(r["assumptions"]["max_csrsquared"], 3) == 0.603
    assert round(r["assumptions"]["nagelkerke_r2"], 3) == 0.477
    assert r["binding_criterion"].startswith("Criterion 2")
    assert r["assumptions"]["csrsquared_source"] == "given"


def test_riley_bmj_worked_example_cstatistic():
    # pmsampsize(type = "b", cstatistic = 0.89, parameters = 24, prevalence = 0.174)
    r = ss.binary_sample_size(0.174, 24, cstatistic=0.89)
    assert r["assumptions"]["csrsquared"] == 0.2902
    assert [c["n"] for c in r["criteria"]] == [617, 661, 221]
    assert r["min_n"] == 661 and r["min_events"] == 116 and r["epp"] == 4.79
    assert r["final_shrinkage"] == 0.906


@pytest.mark.parametrize("cs,prev,k,shrink,ns,final,shrink_final,epp", [
    (0.05, 0.05, 10, 0.9, [1750, 590, 73], 1750, 0.9, 8.75),
    (0.1, 0.1, 20, 0.9, [1699, 784, 139], 1699, 0.9, 8.50),
    (0.15, 0.3, 15, 0.9, [823, 385, 323], 823, 0.9, 16.46),
    (0.2, 0.5, 30, 0.95, [2539, 701, 385], 2539, 0.95, 42.32),
    (0.02, 0.02, 8, 0.9, [3560, 886, 31], 3560, 0.9, 8.90),
    (0.3, 0.4, 5, 0.9, [124, 111, 369], 369, 0.9, 29.52),
    (0.08, 0.03, 12, 0.85, [810, 969, 45], 969, 0.871, 2.42),
])
def test_binary_grid_matches_r(cs, prev, k, shrink, ns, final, shrink_final, epp):
    r = ss.binary_sample_size(prev, k, csrsquared=cs, shrinkage=shrink)
    assert [c["n"] for c in r["criteria"]] == ns
    assert r["min_n"] == final
    assert r["final_shrinkage"] == shrink_final
    assert r["epp"] == epp
    assert r["min_events"] == math.ceil(round(final * prev, 9))


@pytest.mark.parametrize("c,prev,k,r2,ns,final", [
    (0.75, 0.1, 10, 0.0746, [1156, 398, 139], 1156),
    (0.8, 0.05, 20, 0.0621, [2798, 1173, 73], 2798),
    (0.7, 0.3, 8, 0.1029, [659, 211, 323], 659),
    (0.85, 0.2, 15, 0.2495, [463, 405, 246], 463),
])
def test_binary_from_cstatistic_matches_r(c, prev, k, r2, ns, final):
    r = ss.binary_sample_size(prev, k, cstatistic=c)
    assert r["assumptions"]["csrsquared"] == r2
    assert [x["n"] for x in r["criteria"]] == ns
    assert r["min_n"] == final


@pytest.mark.parametrize("c,prev,r2", [
    # pmsampsize:::cstat2rsq(c, prev)$R2.coxsnell, unrounded
    (0.7, 0.05, 0.0252201795),
    (0.8, 0.174, 0.1663113751),
    (0.9, 0.3, 0.3978345392),
    (0.85, 0.5, 0.3470754995),
    (0.75, 0.1, 0.0746216905),
])
def test_cstat_conversion_replays_r_simulation(c, prev, r2):
    assert ss.cstat_to_csrsquared(c, prev) == pytest.approx(r2, abs=1e-9)


def test_r_normal_stream_is_reproduced():
    # set.seed(123456); rnorm(3)
    z = ss._r_rnorm(123456, 3)
    assert list(z) == pytest.approx([0.83373317075513842, -0.27604777321486657,
                                     -0.35500183803340557], abs=1e-15)


def test_nagelkerke_input_matches_r():
    # pmsampsize(type = "b", nagrsquared = 0.15, parameters = 20, prevalence = 0.1)
    r = ss.binary_sample_size(0.1, 20, nagrsquared=0.15)
    assert r["assumptions"]["csrsquared"] == 0.072
    assert r["min_n"] == 2399


def test_binary_input_checks():
    with pytest.raises(ValueError):
        ss.binary_sample_size(0.1, 10)
    with pytest.raises(ValueError):
        ss.binary_sample_size(0.1, 10, csrsquared=0.1, cstatistic=0.8)
    with pytest.raises(ValueError):
        ss.binary_sample_size(1.2, 10, csrsquared=0.1)
    with pytest.raises(ValueError):
        ss.binary_sample_size(0.1, 2.5, csrsquared=0.1)
    with pytest.raises(ValueError, match="largest possible"):
        ss.binary_sample_size(0.02, 10, csrsquared=0.5, shrinkage=0.9)


def test_compare_with_data():
    r = ss.binary_sample_size(0.174, 24, csrsquared=0.288)
    short = ss.compare_with_data(r, 500, 80, 24)
    assert short["adequate"] is False
    assert (short["shortfall"], short["events_shortfall"], short["epp"]) == (162, 36, 3.33)
    ok = ss.compare_with_data(r, 800, 140, 24)
    assert ok["adequate"] is True and ok["shortfall"] == 0
    assert set(ok) >= {"adequate", "n", "events", "epp", "shortfall"}
    more = ss.compare_with_data(r, 800, 140, 30)
    assert any("30" in n for n in more["notes"])


@pytest.mark.parametrize("r2,k,intercept,sd,ns,final,shrink,spp", [
    # pmsampsize(type = "c", rsquared, parameters, intercept, sd), first is
    # the documented example
    (0.2, 25, 1.9, 0.6, [918, 401, 259, 918], 918, 0.9, 36.72),
    (0.5, 10, 50, 10, [101, 101, 244, 244], 244, 0.955, 24.40),
    (0.1, 5, 100, 15, [237, 91, 239, 239], 239, 0.901, 47.80),
    (0.3, 40, 2, 1.5, [951, 561, 274, 951], 951, 0.9, 23.77),
    (0.6, 3, 10, 8, [5, 26, 237, 237], 237, 0.995, 79.0),
])
def test_continuous_matches_r(r2, k, intercept, sd, ns, final, shrink, spp):
    r = ss.continuous_sample_size(r2, k, intercept, sd)
    assert [c["n"] for c in r["criteria"]] == ns
    assert r["min_n"] == final
    assert r["final_shrinkage"] == shrink
    assert r["spp"] == spp


def test_continuous_documented_example_intercept_ci():
    # "95% CI for intercept = (1.87, 1.93), for sample size n = 918"
    r = ss.continuous_sample_size(0.2, 25, 1.9, 0.6)
    assert r["intercept_ci"] == [1.87, 1.93]


def test_continuous_criterion_4_uses_the_correct_search():
    # R pmsampsize 1.1.3 returns 314 here (sign slip in its loop). the
    # smallest n with an intercept margin of error <= 1.1 is 4920.
    r = ss.continuous_sample_size(0.2, 10, 0.5, 2)
    n = r["min_n"]
    assert n == 4920

    def mmoe(n):
        return 1 + math.sqrt(4 * 0.8 / n) * stats.t.ppf(0.975, n - 11) / 0.5

    assert mmoe(n) <= 1.1 < mmoe(n - 1)
    assert any("314" in note for note in r["notes"])
