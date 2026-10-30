"""validation.py
Checks for a fitted clinical prediction model on a binary outcome:
calibration (slope, calibration-in-the-large, O:E, ICI/E50/E90 and a
flexible lowess curve), a DeLong confidence interval for the AUROC,
bootstrap percentile intervals, decision curve analysis, threshold choice
and logistic recalibration. Everything takes plain arrays (y coded 0/1,
p = predicted probability of the outcome) and returns dicts, tuples or
floats. Knows nothing about Flask.

References
  Van Calster B et al. A calibration hierarchy for risk models was defined:
    from utopia to empirical data. J Clin Epidemiol 2016;74:167-176.
  Austin PC, Steyerberg EW. The Integrated Calibration Index (ICI) and
    related metrics for quantifying the calibration of logistic regression
    models. Stat Med 2019;38:4051-4065.
  Debray TPA et al. A guide to systematic review and meta-analysis of
    prediction model performance. BMJ 2017;356:i6460.
  DeLong ER et al. Comparing the areas under two or more correlated ROC
    curves. Biometrics 1988;44:837-845 (fast form: Sun X, Xu W. IEEE
    Signal Process Lett 2014;21:1389-1393).
  Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making
    2006;26:565-574.
"""
import warnings

import numpy as np
from scipy.special import expit
from scipy.stats import norm, rankdata

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tools import sm_exceptions as _smx

from sklearn.base import BaseEstimator, ClassifierMixin

EPS = 1e-8                      # predicted risks are clipped to [EPS, 1 - EPS]
Z95 = float(norm.ppf(0.975))
LOWESS_FRAC = 2 / 3             # smoothing span, R lowess() default

# bootstrap cost caps (n * n_boot), see bootstrap_ci
BOOT_BUDGET_CALLABLE = 25_000_000
BOOT_BUDGET_FAST = 250_000_000
BOOT_MIN = 200

# statsmodels < 0.14 raised on perfect separation, newer versions warn
_SEP_ERRORS = tuple(c for c in (getattr(_smx, "PerfectSeparationError", None),) if c)
_SEP_WARNINGS = tuple(c for c in (getattr(_smx, "PerfectSeparationWarning", None),) if c)


# ============================================================================
# helpers
# ============================================================================
def _prep(y_true, p):
    """y as float 0/1 and p as float, same length, pairs with a missing
    value dropped."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and p have different lengths ({len(y)} vs {len(p)}).")
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be coded 0/1.")
    return y, p


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _num(x):
    """float, or None when missing / not finite (keeps payloads JSON safe)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _glm_logit(y, X, offset=None, maxiter=100):
    """Binomial GLM (logit link, IRLS) of y on X, where X already holds the
    constant column. Returns (result or None, info) with info keys
    converged, separation, error. Never raises on separation or a singular
    design, the caller decides what to report."""
    info = {"converged": False, "separation": False, "error": None}
    res = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial(),
                         offset=offset).fit(maxiter=maxiter)
        except _SEP_ERRORS as e:                    # pragma: no cover
            info["separation"] = True
            info["error"] = str(e)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
            info["error"] = str(e)
    conv_warn = False
    for w in caught:
        if _SEP_WARNINGS and issubclass(w.category, _SEP_WARNINGS):
            info["separation"] = True
        elif issubclass(w.category, _smx.ConvergenceWarning):
            conv_warn = True
    if res is None:
        return None, info
    info["converged"] = bool(getattr(res, "converged", True)) and not conv_warn
    with np.errstate(all="ignore"):
        finite = np.all(np.isfinite(res.params)) and np.all(np.isfinite(res.bse))
    if not finite:
        info["error"] = "the model could not be estimated (non-finite coefficients)"
    return res, info


def _lowess_sorted(y, p, frac=LOWESS_FRAC):
    """lowess of y on p with no robustness iterations (they would treat
    events as outliers on a 0/1 outcome, as noted for rms::val.prob) and
    delta = 1% of the range of p, both as in R lowess(p, y, iter = 0).
    Returns (p sorted, smoothed observed risk at each sorted p)."""
    order = np.argsort(p, kind="mergesort")
    xs = p[order]
    if xs[-1] - xs[0] < 1e-12:                      # constant p
        return xs, np.full(len(xs), y.mean())
    # statsmodels divides by zero on windows with no spread (tiny or tied
    # validation sets); the fit is still defined, so keep the output quiet
    with np.errstate(divide="ignore", invalid="ignore"):
        fit = lowess(y, p, frac=frac, it=0, delta=0.01 * (xs[-1] - xs[0]),
                     return_sorted=True)
    return fit[:, 0], fit[:, 1]


# ============================================================================
# 1. calibration
# ============================================================================
def calibration_stats(y_true, p):
    """Calibration of predicted risks p against a 0/1 outcome.

    slope: coefficient of logit(p) in a logistic regression of y on logit(p)
      with a free intercept (1 = ideal, < 1 = predictions too extreme).
    citl: calibration-in-the-large, the intercept of a logistic regression of
      y with logit(p) as an offset and no slope term (0 = ideal, > 0 = risks
      underestimated on average).
    slope_ci / citl_ci: 95% Wald intervals from the GLM.
    oe_ratio: observed events / sum of predicted risks, 95% CI from
      log(O/E) +/- 1.96 * sqrt(1/O) (Poisson approximation treating E as
      fixed, Debray et al. 2017).
    ici / e50 / e90: mean, median and 90th percentile of |p - smoothed
      observed risk|, where the smoothed risk is a lowess of y on p
      (Austin and Steyerberg 2019; lowess span 2/3, no robustness
      iterations, as in R lowess(p, y, iter = 0)).

    p is clipped to [1e-8, 1 - 1e-8]. Quantities that cannot be estimated
    (one outcome class, identical predictions, perfect separation) are None
    and "reason" says why; otherwise "reason" is None.

    Returns {slope, slope_ci [lo, hi], citl, citl_ci, oe_ratio, oe_ci, ici,
    e50, e90, n, events, reason}.
    """
    y, p = _prep(y_true, p)
    p = _clip(p)
    n, events = int(len(y)), int(y.sum())
    out = {"slope": None, "slope_ci": [None, None],
           "citl": None, "citl_ci": [None, None],
           "oe_ratio": None, "oe_ci": [None, None],
           "ici": None, "e50": None, "e90": None,
           "n": n, "events": events, "reason": None}
    if n == 0:
        out["reason"] = "no observations"
        return out
    reasons = []
    lp = np.log(p / (1 - p))
    one_class = events in (0, n)

    # slope
    if one_class:
        reasons.append("only one outcome class, so the calibration slope and "
                       "calibration-in-the-large cannot be estimated")
    elif np.ptp(lp) < 1e-12:
        reasons.append("all predicted risks are identical, so the calibration "
                       "slope is undefined")
    else:
        res, info = _glm_logit(y, np.column_stack([np.ones(n), lp]))
        if res is None or info["error"] or info["separation"]:
            reasons.append("the predictions separate the outcome perfectly, so "
                           "the calibration slope cannot be estimated")
        else:
            b, se = float(res.params[1]), float(res.bse[1])
            out["slope"] = b
            out["slope_ci"] = [b - Z95 * se, b + Z95 * se]

    # calibration-in-the-large
    if not one_class:
        res, info = _glm_logit(y, np.ones((n, 1)), offset=lp)
        if res is None or info["error"]:
            reasons.append("calibration-in-the-large could not be estimated")
        else:
            a, se = float(res.params[0]), float(res.bse[0])
            out["citl"] = a
            out["citl_ci"] = [a - Z95 * se, a + Z95 * se]

    # observed / expected
    expected = float(p.sum())
    out["oe_ratio"] = events / expected
    if events > 0:
        se = np.sqrt(1.0 / events)
        log_oe = np.log(out["oe_ratio"])
        out["oe_ci"] = [float(np.exp(log_oe - Z95 * se)),
                        float(np.exp(log_oe + Z95 * se))]
    else:
        reasons.append("no observed events, so the O:E interval is undefined")

    # ici / e50 / e90
    if n >= 3:
        xs, smooth = _lowess_sorted(y, p)
        diff = np.abs(smooth - xs)
        out["ici"] = float(diff.mean())
        out["e50"] = float(np.median(diff))
        out["e90"] = float(np.quantile(diff, 0.9))
    else:
        reasons.append("too few observations for a smoothed calibration curve")

    out["reason"] = "; ".join(reasons) or None
    return out


def flexible_calibration_curve(y_true, p, n_grid=100):
    """Flexible (lowess) calibration curve in the style of Van Calster et al.
    2016: smoothed observed risk against predicted risk, evaluated on an
    even grid over the observed range of p (no extrapolation beyond it).
    Same smoother as the ICI in calibration_stats.

    Returns {"p_grid": [...], "observed": [...], "method": "lowess",
    "frac": span}. Empty lists when there are fewer than 3 observations.
    """
    y, p = _prep(y_true, p)
    p = _clip(p)
    out = {"p_grid": [], "observed": [], "method": "lowess", "frac": LOWESS_FRAC}
    if len(y) < 3:
        return out
    xs, smooth = _lowess_sorted(y, p)
    if xs[-1] - xs[0] < 1e-12:
        out["p_grid"], out["observed"] = [float(xs[0])], [float(smooth[0])]
        return out
    ux, first = np.unique(xs, return_index=True)    # ties share one fitted value
    grid = np.linspace(xs[0], xs[-1], int(max(n_grid, 2)))
    obs = np.interp(grid, ux, smooth[first])
    out["p_grid"] = [float(v) for v in grid]
    out["observed"] = [float(v) for v in obs]
    return out


# ============================================================================
# 2. discrimination
# ============================================================================
def auroc_ci(y_true, p, alpha=0.05):
    """AUROC with a DeLong confidence interval (midrank form of Sun and Xu
    2014), matching pROC::ci.auc(method = "delong"): the interval is
    auc +/- z * se, clipped to [0, 1]. Ties count one half.

    Returns (auc, lo, hi) as floats. The interval is NaN when there are
    fewer than 2 events or 2 non-events; everything is NaN when one class
    is missing.
    """
    y, p = _prep(y_true, p)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    nan = float("nan")
    if m == 0 or n == 0:
        return nan, nan, nan
    tz = rankdata(np.r_[pos, neg])
    auc = float((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
    if m < 2 or n < 2:
        return auc, nan, nan
    v_pos = (tz[:m] - rankdata(pos)) / n            # placement of each event
    v_neg = 1.0 - (tz[m:] - rankdata(neg)) / m      # placement of each non-event
    se = float(np.sqrt(np.var(v_pos, ddof=1) / m + np.var(v_neg, ddof=1) / n))
    z = float(norm.ppf(1 - alpha / 2))
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


# ============================================================================
# 3. bootstrap
# ============================================================================
def _fast_auprc(y, p):
    """average precision (same definition as sklearn average_precision_score)
    as a function of bootstrap counts w, sorting p only once."""
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]   # last row per distinct p

    def stat(w):
        ws = w[order]
        tp = np.cumsum(ws * ys)[ends]
        fp = np.cumsum(ws * (1 - ys))[ends]
        if tp[-1] <= 0:
            return float("nan")
        called = tp + fp
        prec = np.divide(tp, called, out=np.zeros_like(tp), where=called > 0)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return stat


def _fast_brier(y, p):
    sq = (p - y) ** 2

    def stat(w):
        # plain sum, not np.dot: a threaded blas call per replicate is
        # slower than the arithmetic on a busy machine
        return float((w * sq).sum() / w.sum())
    return stat


_FAST_STATS = {"auprc": _fast_auprc, "brier": _fast_brier}


def bootstrap_ci(y_true, p, stat, n_boot=500, seed=42, alpha=0.05):
    """Percentile bootstrap interval for any stat(y, p).

    stat: a callable stat(y, p) -> float, or "auprc" / "brier" for built-in
    versions that reuse one sort and weight rows by their bootstrap count
    (same numbers as resampling, far faster on large n).

    Resamples rows with replacement, n_boot times, from a numpy Generator
    seeded with seed, so the result is deterministic. Replicates where the
    stat is undefined (e.g. no events drawn) are skipped. To keep a full
    evaluation within a few seconds, n_boot is capped so that n * n_boot
    stays under 25 million for callables and 250 million for the built-in
    stats, never going below 200 (n = 50,000 keeps the full 500).

    Returns (lo, hi) as floats, NaN when nothing could be computed.
    """
    y, p = _prep(y_true, p)
    n = len(y)
    nan = float("nan")
    if n == 0 or n_boot < 1:
        return nan, nan
    fast = None
    if isinstance(stat, str):
        key = stat.lower()
        if key not in _FAST_STATS:
            raise ValueError(f"Unknown stat {stat!r}, use a callable or one of {sorted(_FAST_STATS)}.")
        fast = _FAST_STATS[key](y, p)
        budget = BOOT_BUDGET_FAST
    else:
        budget = BOOT_BUDGET_CALLABLE
    n_boot = int(min(n_boot, max(BOOT_MIN, budget // n)))

    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                if fast is not None:
                    vals[b] = fast(np.bincount(idx, minlength=n).astype(float))
                else:
                    vals[b] = float(stat(y[idx], p[idx]))
            except (ValueError, ZeroDivisionError, FloatingPointError):
                pass
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return nan, nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ============================================================================
# 4. decision curve analysis
# ============================================================================
def default_thresholds():
    """0.01, 0.02, ..., 0.99."""
    return np.round(np.arange(1, 100) / 100, 2)


def decision_curve(y_true, p, thresholds=None):
    """Net benefit across threshold probabilities (Vickers and Elkin 2006).

    A patient counts as test positive when p >= threshold (as in the dcurves
    package). At threshold pt:
      net_benefit = TP/n - FP/n * pt / (1 - pt)
      treat_all   = prevalence - (1 - prevalence) * pt / (1 - pt)
      treat_none  = 0
    thresholds default to 0.01 to 0.99 by 0.01; pass a clinically relevant
    range when there is one. Every threshold must lie strictly between 0
    and 1.

    Returns {"threshold", "net_benefit", "treat_all", "treat_none"}, lists
    of equal length.
    """
    y, p = _prep(y_true, p)
    t = default_thresholds() if thresholds is None else np.asarray(thresholds, dtype=float).ravel()
    if t.size and ((t <= 0) | (t >= 1)).any():
        raise ValueError("Decision curve thresholds must be strictly between 0 and 1.")
    n = len(y)
    if n == 0:
        nans = [None] * len(t)
        return {"threshold": [float(v) for v in t], "net_benefit": nans,
                "treat_all": list(nans), "treat_none": [0.0] * len(t)}
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(pos) - np.searchsorted(pos, t, side="left")      # events with p >= t
    fp = len(neg) - np.searchsorted(neg, t, side="left")
    odds = t / (1 - t)
    prev = len(pos) / n
    nb = tp / n - fp / n * odds
    treat_all = prev - (1 - prev) * odds
    return {"threshold": [float(v) for v in t],
            "net_benefit": [float(v) for v in nb],
            "treat_all": [float(v) for v in treat_all],
            "treat_none": [0.0] * len(t)}


# ============================================================================
# 5. threshold choice
# ============================================================================
_THRESHOLD_METHODS = ("youden", "sensitivity", "specificity", "prevalence", "fixed")


def choose_threshold(y_true, p, method="youden", target=None):
    """Pick a classification threshold from predicted risks. A patient is
    classed positive when p >= threshold.

    Compute this on the VALIDATION set and apply the threshold unchanged to
    the test set; choosing it on the test set would bias the test metrics.

    method
      youden       maximises sensitivity + specificity - 1 over the distinct
                   predicted risks; ties go to the lowest threshold (the
                   more sensitive rule). target is ignored.
      sensitivity  the highest threshold whose sensitivity is still >=
                   target, i.e. the most specific rule that meets the
                   sensitivity target (target in (0, 1]).
      specificity  the lowest threshold whose specificity is >= target, i.e.
                   the most sensitive rule that meets the specificity target
                   (target in (0, 1]).
      prevalence   threshold = observed event rate in the data passed.
      fixed        threshold = target (in (0, 1)).

    Returns {"threshold", "method", "target", "sensitivity", "specificity"},
    with sensitivity and specificity at that threshold on the data passed
    (None when the needed outcome class is absent).
    """
    y, p = _prep(y_true, p)
    method = str(method).lower()
    if method not in _THRESHOLD_METHODS:
        raise ValueError(f"Unknown threshold method {method!r}, use one of {list(_THRESHOLD_METHODS)}.")
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    n_pos, n_neg = len(pos), len(neg)

    def sens_at(t):
        return (n_pos - np.searchsorted(pos, t, side="left")) / n_pos

    def spec_at(t):
        return np.searchsorted(neg, t, side="left") / n_neg

    if method == "fixed":
        if target is None or not 0 < float(target) < 1:
            raise ValueError("A fixed threshold needs a target strictly between 0 and 1.")
        thr = float(target)
    else:
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Choosing a threshold this way needs both outcome classes in the data.")
        if method == "prevalence":
            thr = n_pos / (n_pos + n_neg)
        elif method == "youden":
            cand = np.unique(p)
            j = sens_at(cand) + spec_at(cand) - 1
            thr = float(cand[int(np.argmax(j))])
        else:
            if target is None or not 0 < float(target) <= 1:
                raise ValueError(f"A {method} target must be in (0, 1].")
            goal = float(target) - 1e-12
            if method == "sensitivity":
                cand = np.unique(p)
                ok = cand[sens_at(cand) >= goal]
                thr = float(ok.max())                # lowest p always reaches 1.0
            else:
                # one candidate above every prediction (specificity 1.0)
                cand = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
                ok = cand[spec_at(cand) >= goal]
                thr = float(ok.min())
    return {"threshold": thr, "method": method,
            "target": None if target is None else float(target),
            "sensitivity": float(sens_at(thr)) if n_pos else None,
            "specificity": float(spec_at(thr)) if n_neg else None}


# ============================================================================
# 6. logistic recalibration
# ============================================================================
class LogisticRecalibrator(ClassifierMixin, BaseEstimator):
    """Logistic recalibration of an already fitted binary classifier:
    recalibrated logit = a + b * logit(p), with a and b estimated on the
    validation set by maximum likelihood (y ~ a + b * logit(p)). The base
    model is left untouched and is not refit.

    Usage: rec = LogisticRecalibrator(pipe).fit(X_val, y_val), then
    rec.predict_proba(X) gives sklearn's 2-column probabilities, so it drops
    into ml.predict_scores / ml.evaluate and pickles with joblib. y must be
    coded 0/1 with 1 = classes_[1] (what ml.encode_outcome returns).
    """

    def __init__(self, base=None):
        self.base = base
        self.a = None
        self.b = None

    @property
    def classes_(self):
        return self.base.classes_

    @property
    def named_steps(self):
        # lets ml.feature_importance read the base pipeline (recalibration is
        # monotone, so the ranking of features does not change)
        return self.base.named_steps

    def __sklearn_is_fitted__(self):
        return self.a is not None and self.b is not None

    def _base_p(self, X):
        proba = np.asarray(self.base.predict_proba(X))
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError("Logistic recalibration only works for a binary outcome.")
        return proba[:, 1]

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        lp = _logit(self._base_p(X))
        keep = np.isfinite(y)
        y, lp = y[keep], lp[keep]
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y must be coded 0/1.")
        if y.sum() in (0, len(y)):
            raise ValueError("Recalibration needs both outcome classes in the validation set.")
        if np.ptp(lp) < 1e-12:
            raise ValueError("The model gives every patient the same risk, so it cannot be recalibrated.")
        res, info = _glm_logit(y, np.column_stack([np.ones(len(y)), lp]))
        if res is None or info["error"] or info["separation"]:
            raise ValueError("Recalibration failed: the predictions separate the outcome "
                             "perfectly in the validation set.")
        self.a, self.b = float(res.params[0]), float(res.params[1])
        return self

    def predict_proba(self, X):
        if not self.__sklearn_is_fitted__():
            raise ValueError("Call fit() on the validation set first.")
        q = expit(self.a + self.b * _logit(self._base_p(X)))
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return np.asarray(self.classes_)[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def recalibrate(pipe, X_val, y_val):
    """LogisticRecalibrator fitted on the validation set."""
    return LogisticRecalibrator(pipe).fit(X_val, y_val)
