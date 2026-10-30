"""epi.py
Clinical epidemiology calculators:
  - odds ratio + risk ratio from a 2x2 table (with 95% CI, p-values)
  - risk difference and NNT / NNH with a 95% CI (Altman, BMJ 1998)
  - Kaplan-Meier survival with log-rank test, risk table and median CIs
  - Cox proportional hazards with categorical covariates and a
    proportional hazards test (survival::cox.zph, survival >= 3.0)
  - cumulative incidence with competing risks (Aalen-Johansen) + Gray's test
  - Hosmer-Lemeshow goodness-of-fit + calibration deciles

2x2 convention:
                  Outcome+    Outcome-
    Exposed         a           b
    Unexposed       c           d

Reference implementations each part is checked against (tests/test_epi.py):
  2x2       R 4.3.3 prop.test, chisq.test, fisher.test, epitools 0.5-10.1
  survival  R survival 3.5-8 (survfit, survdiff, coxph, cox.zph)
  competing R survival 3.5-8 survfit (multi-state), cmprsk 2.2-12 cuminc
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test


def _pval(p):
    if p is None or not np.isfinite(p):
        return None
    return float(f"{p:.6g}")


def _r4(x):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), 4)


def _num(x):
    """whole numbers as int, so json and the ui show 20 rather than 20.0"""
    x = float(x)
    return int(x) if x.is_integer() else x


def _level_str(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


# ---------------------------------------------------------------------------
# 2x2 table measures
# ---------------------------------------------------------------------------
def _z_ci(estimate_log, se, conf=0.95):
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    lo = np.exp(estimate_log - z * se)
    hi = np.exp(estimate_log + z * se)
    return round(float(lo), 4), round(float(hi), 4)


def _check_cells(a, b, c, d):
    cells = [float(a), float(b), float(c), float(d)]
    if not all(np.isfinite(cells)) or min(cells) < 0:
        raise ValueError("2x2 table counts must be finite and not negative.")
    if cells[0] + cells[1] == 0 or cells[2] + cells[3] == 0:
        raise ValueError("Each row of the 2x2 table (exposed and unexposed) "
                         "needs at least one patient.")
    return cells


def _risk_difference(a, b, c, d, conf=0.95):
    """risks, risk difference and its wald ci from the observed table.
    same as R prop.test(c(a, c), c(a + b, c + d), correct = FALSE)."""
    n1, n0 = a + b, c + d
    r1, r0 = a / n1, c / n0
    rd = r1 - r0
    se = math.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    lo, hi = float(max(rd - z * se, -1.0)), float(min(rd + z * se, 1.0))
    return r1, r0, rd, se, lo, hi


def _fmt_nnt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".") if x is not None else None


def _nnt_ci(rd, se, lo, hi, conf=0.95):
    """95% ci for nnt / nnh by inverting the risk difference ci (Altman,
    BMJ 1998;317:1309). rd < 0 means fewer events when exposed, so benefit.
    when the rd ci includes zero the nnt ci runs through infinity and is
    reported as 'NNT (benefit) X to infinity to NNH (harm) Y'."""
    pct = f"{conf * 100:g}%"
    out = {"lower": None, "upper": None, "crosses_zero": False,
           "benefit_bound": None, "harm_bound": None, "measure": None,
           "text": None}
    if se == 0:
        out["crosses_zero"] = rd == 0
        out["text"] = "not estimable (no variation in risk within the groups)"
        return out
    if hi < 0:
        out.update(measure="NNT (benefit)", lower=round(1 / abs(lo), 2),
                   upper=round(1 / abs(hi), 2))
    elif lo > 0:
        out.update(measure="NNH (harm)", lower=round(1 / hi, 2),
                   upper=round(1 / lo, 2))
    else:
        out["crosses_zero"] = True
        if lo < 0:
            out["benefit_bound"] = round(1 / abs(lo), 2)
        if hi > 0:
            out["harm_bound"] = round(1 / hi, 2)
    if not out["crosses_zero"]:
        out["text"] = (f"{out['measure']} {pct} CI {_fmt_nnt(out['lower'])} "
                       f"to {_fmt_nnt(out['upper'])}")
    else:
        parts = []
        if out["benefit_bound"] is not None:
            parts.append(f"NNT (benefit) {_fmt_nnt(out['benefit_bound'])}")
        parts.append("infinity")
        if out["harm_bound"] is not None:
            parts.append(f"NNH (harm) {_fmt_nnt(out['harm_bound'])}")
        out["text"] = (f"{pct} CI " + " to ".join(parts)
                       + " (the risk difference CI includes zero)")
    return out


def two_by_two(a, b, c, d, conf=0.95):
    """Odds ratio + risk ratio with CIs, risk difference with NNT / NNH and a
    chi-square p-value.

    risks, risk difference, NNT and the tests use the observed table. The
    Haldane-Anscombe correction (0.5 added to every cell) is used only for
    the odds ratio and risk ratio, and only when a cell is zero."""
    obs = _check_cells(a, b, c, d)
    a, b, c, d = obs

    r1, r0, rd, se_rd, rd_lo, rd_hi = _risk_difference(a, b, c, d, conf)

    # or and rr (woolf / katz log ci), corrected only if a cell is zero
    corrected = min(obs) == 0
    ca, cb, cc, cd = [x + 0.5 for x in obs] if corrected else obs
    orr = (ca * cd) / (cb * cc)
    se_log_or = np.sqrt(1 / ca + 1 / cb + 1 / cc + 1 / cd)
    or_lo, or_hi = _z_ci(np.log(orr), se_log_or, conf)
    rr = (ca / (ca + cb)) / (cc / (cc + cd))
    se_log_rr = np.sqrt(1 / ca - 1 / (ca + cb) + 1 / cc - 1 / (cc + cd))
    rr_lo, rr_hi = _z_ci(np.log(rr), se_log_rr, conf)

    # chi-square on the observed table, yates corrected like R chisq.test
    table = np.array([[a, b], [c, d]])
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    notes = []
    if expected.min() > 0:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
    else:
        chi2, p = None, None
        notes.append("Chi-square is not defined because no patient (or every "
                     "patient) had the outcome.")
    fisher_p = stats.fisher_exact(table)[1]
    if expected.min() < 5 and chi2 is not None:
        notes.append("Some expected counts are below 5, so the chi-square "
                     "p-value may be inaccurate. Use Fisher's exact p-value.")
    if corrected:
        notes.append("A cell is zero, so 0.5 was added to every cell for the "
                     "odds ratio and risk ratio only (Haldane-Anscombe). Risks, "
                     "risk difference, NNT and p-values use the observed counts.")

    nnt_raw = 1 / abs(rd) if rd != 0 else float("inf")
    ci = _nnt_ci(rd, se_rd, rd_lo, rd_hi, conf)
    return {
        "table": {"a": _num(a), "b": _num(b), "c": _num(c), "d": _num(d)},
        "table_or_rr": {"a": ca, "b": cb, "c": cc, "d": cd},
        "corrected": corrected,
        "correction_applied": corrected,
        "conf_level": conf,
        "odds_ratio": round(float(orr), 4), "or_ci": [or_lo, or_hi],
        "risk_ratio": round(float(rr), 4), "rr_ci": [rr_lo, rr_hi],
        "risk_exposed": round(float(r1), 4),
        "risk_unexposed": round(float(r0), 4),
        "abs_risk_diff": round(float(rd), 4),
        "rd_ci": [round(rd_lo, 4), round(rd_hi, 4)],
        "nnt": round(float(nnt_raw), 2) if np.isfinite(nnt_raw) else None,
        "nnt_type": "NNT (benefit)" if rd < 0 else "NNH (harm)" if rd > 0 else "n/a",
        "nnt_ci": ci,
        "chi2": round(float(chi2), 4) if chi2 is not None else None,
        "p_value": _pval(p),
        "chi2_method": ("Pearson chi-square with Yates continuity correction "
                        "(the R chisq.test default for a 2x2 table)"),
        "fisher_p_value": _pval(fisher_p),
        "min_expected": round(float(expected.min()), 4),
        "rd_ci_method": "Wald (normal approximation), as in Altman BMJ 1998",
        "notes": notes,
    }


def nnt_nnh(a, b, c, d, conf=0.95):
    """Explicit NNT / NNH breakdown from a 2x2 table, with a CI from the risk
    difference CI (Altman, BMJ 1998). Uses the observed table."""
    a, b, c, d = _check_cells(a, b, c, d)
    r1, r0, rd, se, lo, hi = _risk_difference(a, b, c, d, conf)
    ci = _nnt_ci(rd, se, lo, hi, conf)
    out = {"abs_risk_diff": round(rd, 4), "risk_exposed": round(r1, 4),
           "risk_unexposed": round(r0, 4), "rd_ci": [round(lo, 4), round(hi, 4)],
           "nnt_ci": ci, "conf_level": conf}
    # work from the unrounded risk difference, 1 / (rounded ard) drifts
    if abs(rd) < 1e-12:
        out["measure"] = None
        out["value"] = None
        out["value_rounded_up"] = None
        out["interpretation"] = "No difference in risk."
        return out
    out["value"] = round(abs(1 / rd), 2)
    # convention: report nnt rounded up to a whole patient
    out["value_rounded_up"] = int(math.ceil(round(abs(1 / rd), 6)))
    if rd < 0:
        out["measure"] = "NNT"
        out["interpretation"] = (f"Treat {out['value']} patients to prevent one "
                                 "additional event (number needed to treat).")
    else:
        out["measure"] = "NNH"
        out["interpretation"] = (f"For every {out['value']} patients exposed, one "
                                 "additional event occurs (number needed to harm).")
    if ci["text"]:
        out["interpretation"] += f" {ci['text']}."
    return out


# ---------------------------------------------------------------------------
# survival analysis helpers
# ---------------------------------------------------------------------------
def _nice_time_points(tmax, target=6):
    """0 plus round steps up to tmax, for risk tables and plot ticks"""
    if tmax is None or not np.isfinite(tmax) or tmax <= 0:
        return [0]
    raw = tmax / (target - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = 10 * mag
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    pts = np.arange(0, tmax + step * 1e-9, step)
    return [_num(round(float(p), 10)) for p in pts]


def _clean_time_points(time_points, tmax):
    if time_points is None:
        return _nice_time_points(tmax)
    pts = sorted({float(t) for t in time_points if np.isfinite(float(t)) and float(t) >= 0})
    return [_num(t) for t in pts]


def _surv_frame(df, time_col, event_col, extra=()):
    missing = [c for c in [time_col, event_col, *extra] if c and c not in df.columns]
    if missing:
        raise ValueError(f"Column not found: {', '.join(missing)}")
    data = df[[c for c in [time_col, event_col, *extra] if c]].copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[event_col] = pd.to_numeric(data[event_col], errors="coerce")
    data = data.dropna(subset=[time_col, event_col])
    if data.empty:
        raise ValueError("No rows with a numeric time and event.")
    if (data[time_col] < 0).any():
        raise ValueError("Follow-up times must not be negative.")
    return data


def _check_binary_event(s):
    bad = set(np.unique(s)) - {0, 1}
    if bad:
        raise ValueError("The event column must be coded 0 = censored, "
                         "1 = event. For competing events use cumulative "
                         "incidence instead.")


def _step_at(times, values, t):
    """value of a right-continuous step function at t (last time <= t)"""
    i = np.searchsorted(times, t, side="right") - 1
    if i < 0:
        return None
    v = values[i]
    return None if v is None or not np.isfinite(v) else float(v)


def _findq(x, y, p=0.5, tol=np.sqrt(np.finfo(float).eps)):
    """time where the cumulative failure curve y first reaches p, using the
    same rule as R survival:::findq (survival 3.5-8). if the curve is flat at
    exactly p the midpoint of that flat stretch is returned."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any() or np.nanmax(y) < p:
        return None
    xmax = x[-1]
    keep = ~pd.Series(y).duplicated().to_numpy()
    x, y = x[keep], y[keep]
    last_ok = np.isfinite(y[-1])
    # approx() drops NA pairs (a ci is NA once survival reaches 0)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]

    def idx(yy):
        # approx(yy, 1:n, p, method = "constant", f = 1), 0 based, None if NA
        if p < yy[0] or p > yy[-1]:
            return None
        return int(np.searchsorted(yy, p, side="left"))

    i1, i2 = idx(y + tol), idx(y - tol)
    if last_ok and abs(p - y[-1]) < tol and i1 is not None:
        return float((x[i1] + xmax) / 2)
    if i1 is None or i2 is None:
        return None
    return float((x[i1] + x[i2]) / 2)


def _median_from_curve(timeline, surv, lower, upper):
    t = np.concatenate([[0.0], timeline])
    med = _findq(t, np.concatenate([[0.0], 1 - np.asarray(surv)]))
    lo = _findq(t, np.concatenate([[0.0], 1 - np.asarray(lower)]))
    hi = _findq(t, np.concatenate([[0.0], 1 - np.asarray(upper)]))
    r = lambda v: None if v is None else round(v, 3)
    return r(med), [r(lo), r(hi)]


def _km_fit(t, e, label=None):
    """lifelines km as arrays. the log(-log) ci is undefined once survival
    reaches 0, so it is nan there (R gives NA, lifelines gives 0)."""
    kmf = KaplanMeierFitter()
    kmf.fit(t, e, label=label)
    timeline = kmf.timeline.astype(float)
    surv = kmf.survival_function_.iloc[:, 0].to_numpy(dtype=float)
    ci = kmf.confidence_interval_
    lower = ci.iloc[:, 0].to_numpy(dtype=float)
    upper = ci.iloc[:, 1].to_numpy(dtype=float)
    lower[surv <= 0] = np.nan
    upper[surv <= 0] = np.nan
    return timeline, surv, lower, upper


def _km_curve(name, t, e, time_points):
    timeline, surv, lower, upper = _km_fit(t, e, name)
    med, med_ci = _median_from_curve(timeline, surv, lower, upper)
    t = np.asarray(t, dtype=float)
    at_risk = [int((t >= tp).sum()) for tp in time_points]
    surv_at = []
    for tp, nr in zip(time_points, at_risk):
        # no estimate past the last follow-up time
        seen = tp <= t.max()
        surv_at.append({"time": tp, "at_risk": nr,
                        "survival": _r4(_step_at(timeline, surv, tp)) if seen else None,
                        "lower": _r4(_step_at(timeline, lower, tp)) if seen else None,
                        "upper": _r4(_step_at(timeline, upper, tp)) if seen else None})
    return {
        "name": name,
        "timeline": timeline.tolist(),
        "survival": surv.tolist(),
        "lower": [None if not np.isfinite(v) else float(v) for v in lower],
        "upper": [None if not np.isfinite(v) else float(v) for v in upper],
        "median_survival": med,
        "median_ci": med_ci,
        "n": int(len(t)),
        "events": int(np.asarray(e).sum()),
        "at_risk": at_risk,
        "survival_at": surv_at,
    }


def _sorted_levels(values):
    levels = list(pd.unique(values))
    try:
        return sorted(levels)
    except TypeError:
        return sorted(levels, key=str)


# ---------------------------------------------------------------------------
# kaplan-meier
# ---------------------------------------------------------------------------
def kaplan_meier(df, time_col, event_col, group_col=None, time_points=None):
    """KM survival. If group_col given, returns one curve per group + a
    log-rank test (k groups, k - 1 df, same as R survdiff).

    Each curve carries the median with its 95% CI, the number at risk and
    survival (95% CI) at time_points (default: round steps up to the last
    follow-up). The pointwise CI is on the log(-log) scale (lifelines
    exponential Greenwood, same as R survfit(conf.type = "log-log"), Stata
    and SAS), and the median CI is read off the CI curves as R does."""
    data = _surv_frame(df, time_col, event_col, [group_col] if group_col else [])
    _check_binary_event(data[event_col])
    tps = _clean_time_points(time_points, float(data[time_col].max()))

    curves = []
    logrank = None
    if group_col:
        data = data.dropna(subset=[group_col])
        if data.empty:
            raise ValueError("No rows with a group value.")
        groups = _sorted_levels(data[group_col])
        for g in groups:
            sub = data[data[group_col] == g]
            curves.append(_km_curve(_level_str(g), sub[time_col], sub[event_col], tps))
        if len(groups) >= 2:
            lr = multivariate_logrank_test(data[time_col], data[group_col],
                                           data[event_col])
            logrank = {"test_statistic": round(float(lr.test_statistic), 4),
                       "p_value": _pval(lr.p_value),
                       "df": int(len(groups) - 1),
                       "n_groups": int(len(groups)),
                       "method": f"log-rank test across {len(groups)} groups"}
    else:
        curves.append(_km_curve("Overall", data[time_col], data[event_col], tps))

    risk_table = {"time_points": tps,
                  "groups": [{"name": c["name"], "at_risk": c["at_risk"]} for c in curves]}
    return {"curves": curves, "logrank": logrank, "time_points": tps,
            "risk_table": risk_table,
            "ci_method": ("pointwise 95% CI on the log(-log) scale (R survfit "
                          "conf.type = 'log-log'); median CI read off the CI curves")}


# ---------------------------------------------------------------------------
# cox proportional hazards
# ---------------------------------------------------------------------------
def _is_numeric_col(s):
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return True
    nonnull = s.dropna()
    conv = pd.to_numeric(nonnull, errors="coerce")
    return bool(len(nonnull)) and not conv.isna().any()


def _km_left(time, event):
    """1 - KM just before each subject's time, the 'km' transform of cox.zph"""
    ut, inv = np.unique(time, return_inverse=True)
    nrisk = len(time) - np.searchsorted(np.sort(time), ut, side="left")
    dead = np.bincount(inv, weights=event, minlength=len(ut))
    surv = np.cumprod(1 - dead / nrisk)
    idx = np.searchsorted(ut, time, side="left")  # number of km times < t
    return 1 - np.concatenate([[1.0], surv])[idx]


def _cox_zph(X, time, event, beta, terms, transform="rank"):
    """Proportional hazards test from Schoenfeld residuals, a port of
    survival::cox.zph (survival >= 3.0, Efron ties): a score test for adding
    g(t) * x to the fitted model, per term and global. 'rank' uses the
    average rank of follow-up time over all subjects, as R does."""
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n, p = X.shape
    if transform == "rank":
        tt = stats.rankdata(time)
    elif transform == "identity":
        tt = time.copy()
    elif transform == "log":
        tt = np.log(time)
    elif transform == "km":
        tt = _km_left(time, event)
    else:
        raise ValueError("transform must be 'rank', 'km', 'identity' or 'log'.")
    g = tt - tt[event == 1].mean()

    eta = X @ beta
    w = np.exp(eta - eta.max())
    order = np.argsort(-time, kind="stable")
    Xs, ws, ts, es, gs = X[order], w[order], time[order], event[order], g[order]
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    ends = np.r_[starts[1:], n]

    s0, s1, s2 = 0.0, np.zeros(p), np.zeros((p, p))
    u2 = np.zeros(p)
    imat = np.zeros((2 * p, 2 * p))
    for st, en in zip(starts, ends):
        xb, wb, eb = Xs[st:en], ws[st:en], es[st:en]
        wx = xb * wb[:, None]
        s0 += wb.sum()
        s1 += wx.sum(0)
        s2 += wx.T @ xb
        dmask = eb == 1
        nd = int(dmask.sum())
        if nd == 0:
            continue
        xd, wd = xb[dmask], wb[dmask]
        d0, d1 = wd.sum(), (xd * wd[:, None]).sum(0)
        d2 = (xd * wd[:, None]).T @ xd
        frac = np.arange(nd) / nd  # efron
        a = s0 - frac * d0
        m1 = (s1[None, :] - frac[:, None] * d1[None, :]) / a[:, None]
        cov = s2 * np.sum(1 / a) - d2 * np.sum(frac / a) - m1.T @ m1
        gk = gs[st]
        u2 += gk * (xd.sum(0) - m1.sum(0))
        imat[:p, :p] += cov
        imat[:p, p:] += gk * cov
        imat[p:, :p] += gk * cov
        imat[p:, p:] += gk * gk * cov

    u = np.r_[np.zeros(p), u2]

    def score_test(cols):
        kk = np.r_[np.arange(p), p + np.asarray(cols)]
        uu = u[kk]
        return float(uu @ np.linalg.solve(imat[np.ix_(kk, kk)], uu))

    term_rows = []
    for name, cols in terms:
        chi = score_test(cols)
        term_rows.append({"variable": name, "test_statistic": round(chi, 4),
                          "df": len(cols), "p_value": _pval(stats.chi2.sf(chi, len(cols)))})
    col_stats = []
    for j in range(p):
        chi = score_test([j])
        col_stats.append((round(chi, 4), _pval(stats.chi2.sf(chi, 1))))
    chi = float(u @ np.linalg.solve(imat, u))
    glob = {"test_statistic": round(chi, 4), "df": p,
            "p_value": _pval(stats.chi2.sf(chi, p))}
    return term_rows, col_stats, glob


def cox_ph(df, time_col, event_col, covariates, categorical=None,
           reference=None, ph_transform="rank"):
    """Cox proportional hazards (Efron ties, as R coxph). Returns hazard
    ratios + 95% CIs per term.

    categorical: columns to treat as categorical (e.g. an ASA class coded
    1-4). Columns that are not numeric are categorical automatically. Each
    categorical column is one-hot coded against a reference level (the
    first level in sorted order, or reference[col]) and its terms are named
    like "sex: Male vs Female".

    Also returns the proportional hazards test from Schoenfeld residuals,
    per term, per coefficient and global, with time transform ph_transform
    ('rank' by default; R cox.zph defaults to 'km')."""
    covariates = list(covariates)
    if not covariates:
        raise ValueError("Pick at least one covariate.")
    if ph_transform not in ("rank", "km", "identity", "log"):
        raise ValueError("ph_transform must be 'rank', 'km', 'identity' or 'log'.")
    cat_set = set(categorical or [])
    reference = dict(reference or {})
    data = _surv_frame(df, time_col, event_col, covariates)
    n_start = len(df)

    kinds = {}
    for c in covariates:
        if c in cat_set or not _is_numeric_col(data[c]):
            kinds[c] = "categorical"
        else:
            kinds[c] = "numeric"
            data[c] = pd.to_numeric(data[c], errors="coerce").astype(float)
    for c in covariates:
        if kinds[c] == "categorical":
            s = data[c]
            if s.dtype == object:
                s = s.where(~s.astype(str).str.strip().eq(""), np.nan)
            data[c] = s
    data = data.dropna()
    if data.empty:
        raise ValueError("No complete rows for the chosen Cox covariates.")
    _check_binary_event(data[event_col])
    if data[event_col].sum() == 0:
        raise ValueError("There are no events in the complete rows, so a Cox "
                         "model cannot be fitted.")

    design = pd.DataFrame(index=data.index)
    meta, terms, ref_levels = [], [], {}
    for c in covariates:
        if kinds[c] == "numeric":
            if data[c].nunique() < 2:
                raise ValueError(f"Covariate '{c}' has the same value for every row.")
            terms.append((c, [len(meta)]))
            design[f"__x{len(meta)}"] = data[c].to_numpy(dtype=float)
            meta.append({"covariate": c, "variable": c, "level": None,
                         "reference": None, "type": "numeric"})
            continue
        levels = _sorted_levels(data[c])
        labels = [_level_str(v) for v in levels]
        if len(levels) < 2:
            raise ValueError(f"Covariate '{c}' has only one level in the complete rows.")
        ref = levels[0]
        if c in reference:
            want = str(reference[c])
            hit = [v for v, lab in zip(levels, labels) if lab == want or v == reference[c]]
            if not hit:
                raise ValueError(f"Reference level '{want}' not found in '{c}'.")
            ref = hit[0]
        ref_lab = _level_str(ref)
        ref_levels[c] = ref_lab
        cols = []
        for v, lab in zip(levels, labels):
            if v == ref:
                continue
            cols.append(len(meta))
            design[f"__x{len(meta)}"] = (data[c] == v).astype(float).to_numpy()
            meta.append({"covariate": f"{c}: {lab} vs {ref_lab}", "variable": c,
                         "level": lab, "reference": ref_lab, "type": "categorical"})
        terms.append((c, cols))

    xcols = list(design.columns)
    fit_df = design.copy()
    fit_df["__T"] = data[time_col].to_numpy(dtype=float)
    fit_df["__E"] = data[event_col].to_numpy(dtype=float)
    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col="__T", event_col="__E",
                fit_options={"precision": 1e-9})
    except Exception as e:
        raise ValueError("The Cox model could not be fitted (often a covariate "
                         "level with no events, or covariates that duplicate "
                         f"each other): {e}")
    s = cph.summary
    beta = cph.params_.loc[xcols].to_numpy(dtype=float)

    try:
        term_ph, col_ph, glob = _cox_zph(design[xcols].to_numpy(), fit_df["__T"].to_numpy(),
                                         fit_df["__E"].to_numpy(), beta, terms, ph_transform)
    except np.linalg.LinAlgError:
        # singular information matrix, the test cannot be computed
        term_ph = [{"variable": name, "test_statistic": None, "df": len(cols),
                    "p_value": None} for name, cols in terms]
        col_ph = [(None, None)] * len(xcols)
        glob = {"test_statistic": None, "df": len(xcols), "p_value": None}
    rows = []
    for k, (col, m) in enumerate(zip(xcols, meta)):
        rows.append({
            "covariate": m["covariate"],
            "hazard_ratio": round(float(s.loc[col, "exp(coef)"]), 4),
            "hr_lower": round(float(s.loc[col, "exp(coef) lower 95%"]), 4),
            "hr_upper": round(float(s.loc[col, "exp(coef) upper 95%"]), 4),
            "coef": round(float(s.loc[col, "coef"]), 4),
            "se": round(float(s.loc[col, "se(coef)"]), 4),
            "p_value": _pval(float(s.loc[col, "p"])),
            "variable": m["variable"], "level": m["level"],
            "reference": m["reference"], "type": m["type"],
            "ph_test_statistic": col_ph[k][0], "ph_p_value": col_ph[k][1],
        })
    ph_notes = []
    for t in term_ph:
        if t["p_value"] is not None and t["p_value"] < 0.05:
            ph_notes.append(
                f"The proportional hazards assumption may not hold for "
                f"{t['variable']} (Schoenfeld residual test p = {t['p_value']}). "
                "Its hazard ratio may change over follow-up, so consider "
                "stratifying on it or adding a time interaction.")
    tname = {"rank": "rank of follow-up time", "km": "1 minus Kaplan-Meier",
             "identity": "follow-up time", "log": "log follow-up time"}[ph_transform]
    return {
        "rows": rows,
        "concordance": round(float(cph.concordance_index_), 4),
        "n": int(len(data)),
        "n_events": int(data[event_col].sum()),
        "n_excluded": int(n_start - len(data)),
        "ties": "efron",
        "categorical": [c for c in covariates if kinds[c] == "categorical"],
        "reference_levels": ref_levels,
        "ph_test": {
            "transform": ph_transform,
            "method": ("score test for a time-varying effect, g(t) x, from "
                       f"Schoenfeld residuals with g(t) = {tname}; same as R "
                       f"survival::cox.zph(fit, transform = '{ph_transform}') "
                       "in survival 3.x"),
            "terms": term_ph,
            "global": glob,
        },
        "ph_notes": ph_notes,
    }


# ---------------------------------------------------------------------------
# competing risks, cumulative incidence (aalen-johansen)
# ---------------------------------------------------------------------------
def _aalen_johansen(time, code, k):
    """Aalen-Johansen cumulative incidence for cause k, codes: 0 censored,
    k the event, other non-zero values competing. Standard error from the
    infinitesimal jackknife (the derivative of the estimate with respect to
    each case weight), which is what R survfit uses for multi-state curves."""
    time = np.asarray(time, dtype=float)
    code = np.asarray(code)
    ut, inv = np.unique(time, return_inverse=True)
    m = len(ut)
    is1 = code == k
    is_any = code != 0
    d1 = np.bincount(inv, weights=is1.astype(float), minlength=m)
    dall = np.bincount(inv, weights=is_any.astype(float), minlength=m)
    nexit = np.bincount(inv, minlength=m).astype(float)
    Y = nexit[::-1].cumsum()[::-1]
    h1, h = d1 / Y, dall / Y
    s_before = np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])
    cif = np.cumsum(s_before * h1)

    # influence of one subject, grouped by (exit time index, type)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(h < 1, 1 / (Y * (1 - h)), 0.0)
    A = np.cumsum(s_before * h1 / Y)
    B = np.cumsum(h * q)
    C = np.cumsum(h * cif * q)
    typ = np.where(is1, 1, np.where(is_any, 2, 0))
    key = inv * 3 + typ
    ukey, cnt = np.unique(key, return_counts=True)
    gp, gt = ukey // 3, ukey % 3
    var = np.zeros(m)
    jumps = np.flatnonzero(d1 > 0)
    for j in jumps:
        pj = np.minimum(gp, j)
        pj1 = np.minimum(gp, j - 1)
        before = gp < j
        t1 = np.where((gt == 1) & (gp <= j), s_before[gp] / Y[gp], 0.0)
        t2 = -A[pj]
        bj = np.where(pj1 >= 0, B[np.maximum(pj1, 0)], 0.0)
        cj = np.where(pj1 >= 0, C[np.maximum(pj1, 0)], 0.0)
        exit_ev = (gt != 0) & before
        t3 = -cif[j] * (np.where(exit_ev, q[gp], 0.0) - bj)
        t4 = np.where(exit_ev, cif[gp] * q[gp], 0.0) - cj
        infl = t1 + t2 + t3 + t4
        var[j] = np.sum(cnt * infl ** 2)
    # the estimate (and its variance) only changes at cause k event times
    last = 0.0
    for j in range(m):
        if d1[j] > 0:
            last = var[j]
        var[j] = last
    return ut, Y, cif, np.sqrt(var)


def _gray_test(time, code, group, k, rho=0.0):
    """Gray's test (Ann Stat 1988) comparing the cause k cumulative incidence
    across groups. A line by line port of subroutine crst in R cmprsk 2.2-12
    (crstm.f, Robert Gray), single stratum. Returns (statistic, df) or None
    if the variance matrix is singular."""
    order = np.argsort(np.asarray(time, dtype=float), kind="stable")
    y = np.asarray(time, dtype=float)[order]
    c0 = np.asarray(code)[order]
    m = np.where(c0 == k, 1, np.where(c0 != 0, 2, 0))
    g_sorted = pd.Series(np.asarray(group)[order])
    levels = _sorted_levels(g_sorted)
    ig = pd.Categorical(g_sorted, categories=levels).codes.astype(int)
    ng = len(levels)
    ng1 = ng - 1
    n = len(y)
    rs = np.bincount(ig, minlength=ng).astype(float)
    s = np.zeros(ng1)
    v = np.zeros((ng1, ng1))
    f1m, f1 = np.zeros(ng), np.zeros(ng)
    skmm, skm = np.ones(ng), np.ones(ng)
    v3, v2 = np.zeros(ng), np.zeros((ng1, ng))
    c = np.zeros((ng, ng))
    fm = f = 0.0
    starts = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
    ends = np.r_[starts[1:], n]
    for ll, lu in zip(starts, ends):
        d = np.zeros((3, ng))
        np.add.at(d, (m[ll:lu], ig[ll:lu]), 1)
        nd1, nd2 = d[1].sum(), d[2].sum()
        if nd1 > 0 or nd2 > 0:
            tr = tq = 0.0
            for i in range(ng):
                if rs[i] <= 0:
                    continue
                skm[i] = skmm[i] * (rs[i] - d[1, i] - d[2, i]) / rs[i]
                f1[i] = f1m[i] + skmm[i] * d[1, i] / rs[i]
                tr += rs[i] / skmm[i]
                tq += rs[i] * (1 - f1m[i]) / skmm[i]
            f = fm + nd1 / tr
            fb = (1 - fm) ** rho
            a = np.zeros((ng, ng))
            for i in range(ng):
                if rs[i] <= 0:
                    continue
                t1 = rs[i] / skmm[i]
                a[i, i] = fb * t1 * (1 - t1 / tr)
                if a[i, i] != 0:
                    c[i, i] += a[i, i] * nd1 / (tr * (1 - fm))
                for j in range(i + 1, ng):
                    if rs[j] <= 0:
                        continue
                    a[i, j] = -fb * t1 * rs[j] / (skmm[j] * tr)
                    if a[i, j] != 0:
                        c[i, j] += a[i, j] * nd1 / (tr * (1 - fm))
            for i in range(1, ng):
                for j in range(i):
                    a[i, j] = a[j, i]
                    c[i, j] = c[j, i]
            for i in range(ng1):
                if rs[i] > 0:
                    s[i] += fb * (d[1, i] - nd1 * rs[i] * (1 - f1m[i]) / (skmm[i] * tq))
            if nd1 > 0:
                for kk in range(ng):
                    if rs[kk] <= 0:
                        continue
                    t4 = 1 - (1 - f) / skm[kk] if skm[kk] > 0 else 1.0
                    t5 = 1 - (nd1 - 1) / (tr * skmm[kk] - 1) if nd1 > 1 else 1.0
                    t3 = t5 * skmm[kk] * nd1 / (tr * rs[kk])
                    v3[kk] += t4 * t4 * t3
                    for i in range(ng1):
                        t1 = a[i, kk] - t4 * c[i, kk]
                        v2[i, kk] += t1 * t4 * t3
                        for j in range(i + 1):
                            v[i, j] += t1 * (a[j, kk] - t4 * c[j, kk]) * t3
            if nd2 > 0:
                for kk in range(ng):
                    if skm[kk] <= 0 or d[2, kk] <= 0:
                        continue
                    t4 = (1 - f) / skm[kk]
                    t5 = 1 - (d[2, kk] - 1) / (rs[kk] - 1) if d[2, kk] > 1 else 1.0
                    t3 = t5 * (skmm[kk] ** 2 * d[2, kk]) / rs[kk] ** 2
                    v3[kk] += t4 * t4 * t3
                    for i in range(ng1):
                        t1 = t4 * c[i, kk]
                        v2[i, kk] -= t1 * t4 * t3
                        for j in range(i + 1):
                            v[i, j] += t1 * (t4 * c[j, kk]) * t3
        if lu >= n:
            break
        np.subtract.at(rs, ig[ll:lu], 1)
        fm = f
        f1m, skmm = f1.copy(), skm.copy()
    for i in range(ng1):
        for j in range(i + 1):
            v[i, j] += np.sum(c[i] * c[j] * v3 + c[i] * v2[j] + c[j] * v2[i])
            v[j, i] = v[i, j]
    if np.linalg.matrix_rank(v) < ng1:
        return None
    return float(s @ np.linalg.solve(v, s)), ng1


def _cif_ci(cif, se, conf=0.95):
    """log scale ci, the R survfit default (conf.type = 'log')"""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(cif > 0, cif * np.exp(-z * se / cif), 0.0)
        hi = np.where(cif > 0, np.minimum(cif * np.exp(z * se / cif), 1.0), 0.0)
    return lo, hi


def cumulative_incidence(df, time_col, event_col, event_of_interest=1,
                         group_col=None, time_points=None):
    """Cumulative incidence function (CIF) with competing risks.

    event coding: 0 = censored, event_of_interest = the event, any other
    non-zero code = a competing event (e.g. death when the event is
    revision). Returns the Aalen-Johansen CIF per group with a 95% CI (log
    scale, infinitesimal jackknife SE, both as R survfit for multi-state
    data), the CIF and number at risk at time_points, and the naive
    1 minus Kaplan-Meier estimate that treats competing events as censored.
    The naive estimate overstates the incidence when competing events are
    common (Lacny et al., Clin Orthop Relat Res 2015)."""
    data = _surv_frame(df, time_col, event_col, [group_col] if group_col else [])
    k = float(event_of_interest)
    codes = set(np.unique(data[event_col]))
    if k not in codes:
        raise ValueError(f"No rows have the event of interest (code {_level_str(k)}).")
    competing = sorted(c for c in codes if c not in (0, k))
    tps = _clean_time_points(time_points, float(data[time_col].max()))

    if group_col:
        data = data.dropna(subset=[group_col])
        groups = [(_level_str(g), data[data[group_col] == g]) for g in _sorted_levels(data[group_col])]
    else:
        groups = [("Overall", data)]

    curves, naive = [], []
    for name, sub in groups:
        t = sub[time_col].to_numpy(dtype=float)
        e = sub[event_col].to_numpy(dtype=float)
        ut, Y, cif, se = _aalen_johansen(t, e, k)
        lo, hi = _cif_ci(cif, se)
        ktl, ks, kl, ku = _km_fit(t, (e == k).astype(int))
        klo, khi = 1 - ku, 1 - kl
        at = []
        for tp in tps:
            if tp > t.max():  # no estimate past the last follow-up time
                at.append({"time": tp, "at_risk": 0, "cif": None, "lower": None,
                           "upper": None, "naive": None})
                continue
            at.append({"time": tp, "at_risk": int((t >= tp).sum()),
                       "cif": _r4(_step_at(ut, cif, tp) or 0.0),
                       "lower": _r4(_step_at(ut, lo, tp) or 0.0),
                       "upper": _r4(_step_at(ut, hi, tp) or 0.0),
                       "naive": _r4(1 - _step_at(ktl, ks, tp))})
        curves.append({
            "name": name,
            "timeline": [0.0] + ut.tolist(),
            "cif": [0.0] + cif.tolist(),
            "lower": [0.0] + lo.tolist(),
            "upper": [0.0] + hi.tolist(),
            "se": [0.0] + se.tolist(),
            "n": int(len(t)),
            "events": int((e == k).sum()),
            "competing_events": int(np.isin(e, competing).sum()),
            "censored": int((e == 0).sum()),
            "at_risk": [a["at_risk"] for a in at],
            "at_times": at,
        })
        naive.append({"name": name, "timeline": ktl.tolist(),
                      "risk": (1 - ks).tolist(),
                      "lower": [None if not np.isfinite(v) else float(v) for v in klo],
                      "upper": [None if not np.isfinite(v) else float(v) for v in khi]})

    gray = None
    if len(groups) >= 2:
        g = _gray_test(data[time_col].to_numpy(dtype=float),
                       data[event_col].to_numpy(dtype=float),
                       data[group_col].to_numpy(), k)
        if g is not None:
            gray = {"test_statistic": round(g[0], 4), "df": g[1],
                    "p_value": _pval(stats.chi2.sf(g[0], g[1])), "rho": 0,
                    "method": ("Gray's test for equal cumulative incidence "
                               "across groups (R cmprsk cuminc, rho = 0)")}

    notes = []
    if competing:
        notes.append("1 minus Kaplan-Meier treats competing events as censored "
                     "and so overstates the cumulative incidence; report the "
                     "Aalen-Johansen CIF.")
    else:
        notes.append("There are no competing events, so the CIF equals "
                     "1 minus Kaplan-Meier.")
    return {
        "curves": curves,
        "naive": naive,
        "time_points": tps,
        "risk_table": {"time_points": tps,
                       "groups": [{"name": c["name"], "at_risk": c["at_risk"]} for c in curves]},
        "event_of_interest": _num(k),
        "competing_codes": [_num(c) for c in competing],
        "method": ("Aalen-Johansen estimator; 95% CI on the log scale with an "
                   "infinitesimal jackknife SE (R survfit for multi-state data)"),
        "gray_test": gray,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# hosmer-lemeshow goodness of fit
# ---------------------------------------------------------------------------
def hosmer_lemeshow(y_true, y_prob, g=10):
    """Hosmer-Lemeshow C-statistic + per-decile observed/expected for calibration."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob)
    y_true, y_prob = y_true[order], y_prob[order]

    # split into g groups of (aproximately) equal size
    groups = np.array_split(np.arange(len(y_prob)), g)
    hl_stat, deciles = 0.0, []
    for grp in groups:
        if len(grp) == 0:
            continue
        obs = y_true[grp].sum()
        exp = y_prob[grp].sum()
        n = len(grp)
        if 0 < exp < n:
            hl_stat += (obs - exp) ** 2 / (exp * (1 - exp / n))
        deciles.append({
            "n": int(n),
            "mean_pred": round(float(y_prob[grp].mean()), 4),
            "observed_rate": round(float(y_true[grp].mean()), 4),
            "observed": int(obs), "expected": round(float(exp), 2),
        })
    dof = max(g - 2, 1)
    p = float(1 - stats.chi2.cdf(hl_stat, dof))
    return {
        "hl_statistic": round(float(hl_stat), 4),
        "dof": dof,
        "p_value": float(f"{p:.6g}"),
        "well_calibrated": p > 0.05,
        "deciles": deciles,
        "mean_pred": [d["mean_pred"] for d in deciles],
        "observed_rate": [d["observed_rate"] for d in deciles],
    }
