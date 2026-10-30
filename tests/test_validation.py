import joblib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from clintab import ml, plots
from clintab import validation as V


# ---------------------------------------------------------------------------
# fixture shared with the golden values below. arithmetic only (no libm
# calls), so numpy's legacy RandomState gives the same data everywhere. the
# golden numbers were computed on this exact data (written to csv) with
# R 4.3.3 base glm / lowess, pROC 1.19.1 and python dcurves 1.1.7.
# ---------------------------------------------------------------------------
def _cal_fixture():
    rng = np.random.RandomState(2026)
    n = 400
    p = np.round(rng.uniform(0.01, 0.95, size=n), 3)
    true = 0.05 + 0.75 * p * p
    y = (rng.uniform(size=n) < true).astype(int)
    return y, p


def test_fixture_is_the_one_the_golden_values_were_computed_on():
    y, p = _cal_fixture()
    assert len(y) == 400 and int(y.sum()) == 112
    assert len(np.unique(p)) == 334            # ties included on purpose
    assert p.sum() == pytest.approx(194.232, abs=1e-9)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def test_calibration_slope_and_citl_match_r_glm():
    # R 4.3.3: glm(y ~ qlogis(p), binomial) and glm(y ~ 1, offset = qlogis(p),
    # binomial), intervals from confint.default (Wald)
    y, p = _cal_fixture()
    s = V.calibration_stats(y, p)
    assert s["slope"] == pytest.approx(0.769199852885, abs=1e-8)
    assert s["slope_ci"] == pytest.approx([0.568269327279, 0.970130378491], abs=1e-8)
    assert s["citl"] == pytest.approx(-1.238406050057, abs=1e-8)
    assert s["citl_ci"] == pytest.approx([-1.492104637738, -0.984707462375], abs=1e-8)
    assert s["n"] == 400 and s["events"] == 112
    assert s["reason"] is None


def test_oe_ratio_and_interval():
    # R 4.3.3: sum(y) / sum(p), exp(log(oe) +/- qnorm(0.975) * sqrt(1 / O))
    y, p = _cal_fixture()
    s = V.calibration_stats(y, p)
    assert s["oe_ratio"] == pytest.approx(0.576630009473, abs=1e-10)
    assert s["oe_ci"] == pytest.approx([0.479144221482, 0.693950073730], abs=1e-10)


def test_ici_e50_e90_match_r_lowess():
    # R 4.3.3: lw <- lowess(p, y, iter = 0); d <- abs(lw$y - lw$x);
    # mean(d), median(d), quantile(d, 0.9)
    y, p = _cal_fixture()
    s = V.calibration_stats(y, p)
    assert s["ici"] == pytest.approx(0.201452429787, abs=1e-8)
    assert s["e50"] == pytest.approx(0.254042692817, abs=1e-8)
    assert s["e90"] == pytest.approx(0.281805723449, abs=1e-8)


def test_flexible_calibration_curve_matches_r_lowess():
    # R 4.3.3: approx(lw$x, lw$y, xout = c(.05, .25, .5, .75, .9))
    y, p = _cal_fixture()
    curve = V.flexible_calibration_curve(y, p, n_grid=4000)
    assert curve["method"] == "lowess"
    assert len(curve["p_grid"]) == len(curve["observed"]) == 4000
    assert curve["p_grid"][0] == pytest.approx(p.min())
    assert curve["p_grid"][-1] == pytest.approx(p.max())
    at = np.interp([0.05, 0.25, 0.5, 0.75, 0.9], curve["p_grid"], curve["observed"])
    gold = [0.040425923082, 0.122797006402, 0.241334517016, 0.468206992312, 0.618947227843]
    assert at == pytest.approx(gold, abs=1e-4)


def test_flexible_calibration_curve_default_grid_and_tiny_input():
    y, p = _cal_fixture()
    curve = V.flexible_calibration_curve(y, p)
    assert len(curve["p_grid"]) == 100
    assert V.flexible_calibration_curve([0, 1], [0.2, 0.8])["p_grid"] == []


def test_calibration_stats_one_class_gives_none_with_a_reason():
    s = V.calibration_stats([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert s["slope"] is None and s["citl"] is None
    assert s["slope_ci"] == [None, None]
    assert s["oe_ratio"] == 0.0 and s["oe_ci"] == [None, None]
    assert "one outcome class" in s["reason"]
    assert s["ici"] == pytest.approx(0.25)        # smoothed risk is 0 everywhere


def test_calibration_stats_constant_predictions():
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    s = V.calibration_stats(y, np.full(10, 0.2))
    assert s["slope"] is None
    assert "identical" in s["reason"]
    # citl has a closed form here: logit(mean y) - logit(p)
    assert s["citl"] == pytest.approx(np.log(0.3 / 0.7) - np.log(0.2 / 0.8), abs=1e-8)
    assert s["oe_ratio"] == pytest.approx(1.5)
    assert s["ici"] == pytest.approx(0.1)


def test_calibration_stats_perfect_separation_does_not_crash():
    p = np.linspace(0.05, 0.95, 40)
    y = (p > 0.5).astype(int)
    s = V.calibration_stats(y, p)
    assert s["slope"] is None
    assert "separate" in s["reason"]
    assert s["citl"] is not None


def test_calibration_stats_clips_and_drops_missing_pairs():
    s = V.calibration_stats([0, 1, 1, np.nan, 0, 1], [0.0, 1.0, 0.7, 0.5, np.nan, 0.6])
    assert s["n"] == 4 and s["events"] == 3
    with pytest.raises(ValueError):
        V.calibration_stats([0, 2, 1], [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# discrimination
# ---------------------------------------------------------------------------
def test_auroc_delong_matches_proc():
    # R pROC 1.19.1: ci.auc(roc(y, p, levels = c(0, 1), direction = "<"),
    # method = "delong")
    y, p = _cal_fixture()
    auc, lo, hi = V.auroc_ci(y, p)
    assert auc == pytest.approx(0.766198536706, abs=1e-10)
    assert lo == pytest.approx(0.715701365465, abs=1e-10)
    assert hi == pytest.approx(0.816695707947, abs=1e-10)
    assert auc == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_auroc_delong_small_sample_with_ties_matches_proc():
    # same call on the first 60 rows
    y, p = _cal_fixture()
    auc, lo, hi = V.auroc_ci(y[:60], p[:60])
    assert (auc, lo, hi) == pytest.approx((0.698863636364, 0.538666811592, 0.859060461135), abs=1e-10)


def test_auroc_ci_is_clipped_and_handles_degenerate_input():
    auc, lo, hi = V.auroc_ci([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert auc == 1.0 and hi == 1.0 and lo <= 1.0
    assert all(np.isnan(v) for v in V.auroc_ci([0, 0, 0], [0.1, 0.2, 0.3]))
    auc, lo, hi = V.auroc_ci([0, 0, 1], [0.1, 0.2, 0.3])
    assert auc == 1.0 and np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    y, p = _cal_fixture()
    a = V.bootstrap_ci(y, p, "auprc", n_boot=300, seed=7)
    b = V.bootstrap_ci(y, p, "auprc", n_boot=300, seed=7)
    c = V.bootstrap_ci(y, p, "auprc", n_boot=300, seed=8)
    assert a == b and a != c
    assert a[0] < average_precision_score(y, p) < a[1]
    lo, hi = V.bootstrap_ci(y, p, "brier", n_boot=300)
    assert lo < brier_score_loss(y, p) < hi


def test_bootstrap_fast_stats_equal_resampling_with_sklearn():
    # the weighted fast paths must give the same replicates as resampling rows
    y, p = _cal_fixture()
    fast = V.bootstrap_ci(y, p, "auprc", n_boot=200, seed=3)
    slow = V.bootstrap_ci(y, p, average_precision_score, n_boot=200, seed=3)
    assert fast == pytest.approx(slow, abs=1e-12)
    fast = V.bootstrap_ci(y, p, "brier", n_boot=200, seed=3)
    slow = V.bootstrap_ci(y, p, brier_score_loss, n_boot=200, seed=3)
    assert fast == pytest.approx(slow, abs=1e-12)


def test_bootstrap_caps_replicates_on_large_data():
    rng = np.random.RandomState(0)
    n = 200_000
    y = (rng.uniform(size=n) < 0.1).astype(int)
    p = rng.uniform(size=n)
    calls = []

    def stat(yy, pp):
        calls.append(1)
        return float(yy.mean())

    V.bootstrap_ci(y, p, stat, n_boot=500)
    assert len(calls) == max(V.BOOT_MIN, V.BOOT_BUDGET_CALLABLE // n) == 200


def test_bootstrap_rejects_unknown_stat_name():
    with pytest.raises(ValueError):
        V.bootstrap_ci([0, 1], [0.2, 0.8], "auroc")


# ---------------------------------------------------------------------------
# decision curve
# ---------------------------------------------------------------------------
def test_decision_curve_matches_dcurves():
    # python dcurves 1.1.7: dca(data, outcome="y", modelnames=["p"],
    # thresholds=...), rows model == "p" and model == "all"
    y, p = _cal_fixture()
    thr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    nb = [0.2442105263157895, 0.21000000000000002, 0.17205882352941176, 0.12875,
          0.09916666666666668, 0.06607142857142859, 0.0016666666666666774,
          -0.04749999999999999, -0.09624999999999989, -0.11833333333333323,
          -0.20000000000000004, -0.09]
    treat_all = [0.24210526315789477, 0.2, 0.15294117647058825, 0.10000000000000003,
                 0.040000000000000036, -0.02857142857142858, -0.2, -0.43999999999999995,
                 -0.7999999999999998, -1.3999999999999997, -2.6000000000000005,
                 -6.200000000000001]
    dc = V.decision_curve(y, p, thresholds=thr)
    assert dc["threshold"] == thr
    assert dc["net_benefit"] == pytest.approx(nb, abs=1e-12)
    assert dc["treat_all"] == pytest.approx(treat_all, abs=1e-12)
    assert dc["treat_none"] == [0.0] * len(thr)


def test_decision_curve_counts_p_equal_to_threshold_as_positive():
    # 0.2 is one of the predicted risks, dcurves uses p >= threshold
    dc = V.decision_curve([1, 0, 0, 1], [0.2, 0.2, 0.1, 0.9], thresholds=[0.2])
    assert dc["net_benefit"][0] == pytest.approx(2 / 4 - 1 / 4 * 0.25)


def test_decision_curve_default_grid_and_bad_thresholds():
    y, p = _cal_fixture()
    dc = V.decision_curve(y, p)
    assert len(dc["threshold"]) == 99
    assert dc["threshold"][0] == 0.01 and dc["threshold"][-1] == 0.99
    with pytest.raises(ValueError):
        V.decision_curve(y, p, thresholds=[0.0, 0.5])
    with pytest.raises(ValueError):
        V.decision_curve(y, p, thresholds=[0.5, 1.0])


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
_TY = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
_TP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def test_choose_threshold_youden_ties_go_to_the_lower_threshold():
    # j = 0.8 at both 0.5 (sens 1, spec 0.8) and 0.7 (sens 0.8, spec 1)
    r = V.choose_threshold(_TY, _TP, "youden")
    assert r == {"threshold": 0.5, "method": "youden", "target": None,
                 "sensitivity": 1.0, "specificity": 0.8}


def test_choose_threshold_sensitivity_and_specificity_targets():
    r = V.choose_threshold(_TY, _TP, "sensitivity", target=0.8)
    assert (r["threshold"], r["sensitivity"], r["specificity"]) == (0.7, 0.8, 1.0)
    r = V.choose_threshold(_TY, _TP, "specificity", target=0.8)
    assert (r["threshold"], r["sensitivity"], r["specificity"]) == (0.5, 1.0, 0.8)
    r = V.choose_threshold(_TY, _TP, "specificity", target=1.0)
    assert (r["threshold"], r["sensitivity"], r["specificity"]) == (0.7, 0.8, 1.0)
    r = V.choose_threshold(_TY, _TP, "sensitivity", target=1.0)
    assert r["threshold"] == 0.5


def test_choose_threshold_specificity_one_with_a_tie_at_the_top():
    r = V.choose_threshold([0, 1, 0], [0.2, 0.9, 0.9], "specificity", target=1.0)
    assert r["specificity"] == 1.0 and r["sensitivity"] == 0.0
    assert r["threshold"] > 0.9


def test_choose_threshold_prevalence_and_fixed():
    r = V.choose_threshold(_TY, _TP, "prevalence")
    assert (r["threshold"], r["sensitivity"], r["specificity"]) == (0.5, 1.0, 0.8)
    r = V.choose_threshold(_TY, _TP, "fixed", target=0.65)
    assert (r["threshold"], r["sensitivity"], r["specificity"]) == (0.65, 0.8, 1.0)
    assert r["target"] == 0.65


def test_choose_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        V.choose_threshold(_TY, _TP, "magic")
    with pytest.raises(ValueError):
        V.choose_threshold(_TY, _TP, "fixed")
    with pytest.raises(ValueError):
        V.choose_threshold(_TY, _TP, "sensitivity", target=1.5)
    with pytest.raises(ValueError):
        V.choose_threshold([0, 0, 0], [0.1, 0.2, 0.3], "youden")
    r = V.choose_threshold([0, 0, 0], [0.1, 0.2, 0.3], "fixed", target=0.15)
    assert r["sensitivity"] is None and r["specificity"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# logistic recalibration
# ---------------------------------------------------------------------------
class _FixedRisk(ClassifierMixin, BaseEstimator):
    """stand-in for a fitted model: its predicted risk is the "p" column."""
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.asarray(X["p"], dtype=float)
        return np.column_stack([1 - p, p])


def test_recalibration_coefficients_match_r_glm():
    # R 4.3.3: coef(glm(y ~ qlogis(p), binomial))
    y, p = _cal_fixture()
    rec = V.recalibrate(_FixedRisk(), pd.DataFrame({"p": p}), y)
    assert rec.a == pytest.approx(-1.115704688895, abs=1e-8)
    assert rec.b == pytest.approx(0.769199852885, abs=1e-8)
    q = rec.predict_proba(pd.DataFrame({"p": p}))
    assert q.shape == (400, 2)
    assert np.allclose(q.sum(axis=1), 1.0)
    # refitting on the data it was fitted on gives ideal calibration
    s = V.calibration_stats(y, q[:, 1])
    assert s["slope"] == pytest.approx(1.0, abs=1e-6)
    assert s["citl"] == pytest.approx(0.0, abs=1e-6)
    assert list(rec.classes_) == [0, 1]
    assert isinstance(rec.base, _FixedRisk)


def test_recalibrator_pickles_with_joblib_and_works_with_predict_scores(tmp_path):
    rng = np.random.RandomState(1)
    n = 300
    age = rng.uniform(40, 90, n)
    sex = np.where(rng.uniform(size=n) < 0.5, "F", "M")
    risk = 0.05 + 0.6 * (age - 40) / 50
    df = pd.DataFrame({"age": age, "sex": sex,
                       "outcome": (rng.uniform(size=n) < risk).astype(int)})
    train, val, test = df.iloc[:150], df.iloc[150:225], df.iloc[225:]
    feats = ["age", "sex"]
    pipe, info = ml.train_one_model("RandomForest", train, val, "outcome", feats,
                                    "binary", do_grid_search=False)
    yv, _ = ml.encode_outcome(val["outcome"], "binary", info["classes"])
    rec = V.recalibrate(pipe, val[feats], yv)

    path = tmp_path / "rec.pkl"
    joblib.dump({"pipe": rec}, path)
    back = joblib.load(path)["pipe"]
    assert (back.a, back.b) == (rec.a, rec.b)
    s1 = ml.predict_scores(rec, test[feats], "binary")
    s2 = ml.predict_scores(back, test[feats], "binary")
    assert s1.shape == (75,) and np.array_equal(s1, s2)

    metrics, coords = ml.evaluate(back, test, "outcome", feats, "binary",
                                  classes=info["classes"])
    assert "CalibrationSlope" in metrics
    imp = ml.feature_importance(back, val, "outcome", feats, "binary", info["classes"])
    assert len(imp) > 0


def test_recalibration_needs_both_classes():
    with pytest.raises(ValueError):
        V.recalibrate(_FixedRisk(), pd.DataFrame({"p": [0.2, 0.4, 0.6]}), [0, 0, 0])
    with pytest.raises(ValueError):
        V.LogisticRecalibrator(_FixedRisk()).predict_proba(pd.DataFrame({"p": [0.2]}))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def test_flexible_calibration_plot_renders_with_and_without_extras():
    y, p = _cal_fixture()
    curve = V.flexible_calibration_curve(y, p)
    stats = V.calibration_stats(y, p)
    fig = plots.flexible_calibration_plot(curve["p_grid"], curve["observed"],
                                          binned_pred=[0.1, 0.5], binned_obs=[0.1, None],
                                          predictions=p, stats=stats, name="LR")
    assert isinstance(fig, Figure)
    text = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "Slope 0.77 (0.57 to 0.97)" in text and "O:E 0.58" in text and "ICI 0.201" in text
    assert plots.fig_to_png(fig).startswith("data:image/png;base64,")
    fig = plots.flexible_calibration_plot([0.1, 0.5], [0.12, None],
                                          stats={"slope": None, "citl": None, "oe_ratio": 0.0})
    assert len(fig.axes) == 1
    plots.fig_to_png(fig)


def test_flexible_calibration_plot_takes_the_histogram_from_binary_metrics():
    rng = np.random.RandomState(5)                 # rare outcome, well calibrated
    p = np.round(rng.uniform(0.001, 0.08, size=400), 4)
    y = (rng.uniform(size=400) < p).astype(int)
    _, coords = ml.binary_metrics(y, p, with_ci=False)
    flex = coords["calibration_flexible"]
    fig = plots.flexible_calibration_plot(flex["p_grid"], flex["observed"],
                                          coords["calibration"]["prob_pred"],
                                          coords["calibration"]["prob_true"],
                                          predictions=coords["risk_histogram"],
                                          stats=coords["calibration_stats"])
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xlim()[1] < 0.5          # zoomed to where the risks are
    assert sum(r.get_height() for r in fig.axes[1].patches) == 400
    plots.fig_to_png(fig)


def test_dca_plot_renders_and_keeps_treat_all_from_dominating():
    y, p = _cal_fixture()
    dc = V.decision_curve(y, p)
    other = V.decision_curve(y, np.clip(p * 0.5, 0, 1))
    fig = plots.dca_plot([{"name": "A", "net_benefit": dc["net_benefit"]},
                          {"name": "B", "net_benefit": [None] + other["net_benefit"][1:]}],
                         dc["treat_all"], dc["threshold"])
    lo, hi = fig.axes[0].get_ylim()
    assert lo < 0 and lo > -0.1          # treat all falls to -6.2, the axis does not
    assert hi > max(dc["net_benefit"])
    assert fig.axes[0].get_xlim() == (0.01, 0.99)
    plots.fig_to_png(fig)
