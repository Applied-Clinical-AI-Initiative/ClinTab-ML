"""epi.py
Clinical epidemiology calculators:
  - odds ratio + risk ratio from a 2x2 table (with 95% CI, p-value)
  - NNT / NNH from a 2x2 table
  - Kaplan-Meier survival (lifelines)
  - Cox proportional hazards (lifelines)
  - Hosmer-Lemeshow goodness-of-fit + calibration deciles

2x2 convention:
                  Outcome+    Outcome-
    Exposed         a           b
    Unexposed       c           d
"""
import numpy as np
import pandas as pd
from scipy import stats

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


# ---------------------------------------------------------------------------
# 2x2 table measures
# ---------------------------------------------------------------------------
def _z_ci(estimate_log, se, conf=0.95):
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    lo = np.exp(estimate_log - z * se)
    hi = np.exp(estimate_log + z * se)
    return round(float(lo), 4), round(float(hi), 4)


def two_by_two(a, b, c, d, conf=0.95):
    """Odds ratio + risk ratio with CIs and a chi-square p-value.
    Adds Haldane-Anscombe 0.5 correction if any cell is zero."""
    a, b, c, d = float(a), float(b), float(c), float(d)
    cells = [a, b, c, d]
    corrected = False
    if 0 in cells:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        corrected = True

    # Odds ratio
    orr = (a * d) / (b * c)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    or_lo, or_hi = _z_ci(np.log(orr), se_log_or, conf)

    # Risk ratio
    risk_exp = a / (a + b)
    risk_unexp = c / (c + d)
    rr = risk_exp / risk_unexp
    se_log_rr = np.sqrt((1 / a) - 1 / (a + b) + (1 / c) - 1 / (c + d))
    rr_lo, rr_hi = _z_ci(np.log(rr), se_log_rr, conf)

    # chi-square test on the (original) table
    table = np.array([[a, b], [c, d]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=True)

    # Absolute risk difference, NNT/NNH
    ard = risk_exp - risk_unexp
    nnt_raw = 1 / ard if ard != 0 else float("inf")
    return {
        "table": {"a": a, "b": b, "c": c, "d": d},
        "corrected": corrected,
        "odds_ratio": round(float(orr), 4), "or_ci": [or_lo, or_hi],
        "risk_ratio": round(float(rr), 4), "rr_ci": [rr_lo, rr_hi],
        "risk_exposed": round(float(risk_exp), 4),
        "risk_unexposed": round(float(risk_unexp), 4),
        "abs_risk_diff": round(float(ard), 4),
        "nnt": round(abs(float(nnt_raw)), 2) if np.isfinite(nnt_raw) else None,
        "nnt_type": "NNT (benefit)" if ard < 0 else "NNH (harm)" if ard > 0 else "n/a",
        "chi2": round(float(chi2), 4),
        "p_value": float(f"{p:.6g}"),
    }


def nnt_nnh(a, b, c, d):
    """Explicit NNT / NNH breakdown from a 2x2 table."""
    res = two_by_two(a, b, c, d)
    ard = res["abs_risk_diff"]
    out = {"abs_risk_diff": ard, "risk_exposed": res["risk_exposed"],
           "risk_unexposed": res["risk_unexposed"]}
    if ard == 0:
        out["interpretation"] = "No difference in risk."
        out["value"] = None
    elif ard < 0:
        out["measure"] = "NNT"
        out["value"] = round(abs(1 / ard), 2)
        out["interpretation"] = (f"Treat {out['value']} patients to prevent one "
                                 "additional event (number needed to treat).")
    else:
        out["measure"] = "NNH"
        out["value"] = round(abs(1 / ard), 2)
        out["interpretation"] = (f"For every {out['value']} patients exposed, one "
                                 "additional event occurs (number needed to harm).")
    return out


# ---------------------------------------------------------------------------
# Survival analysis
# ---------------------------------------------------------------------------
def kaplan_meier(df, time_col, event_col, group_col=None):
    """KM survival. If group_col given, returns one curve per group + log-rank."""
    data = df[[c for c in [time_col, event_col, group_col] if c]].copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[event_col] = pd.to_numeric(data[event_col], errors="coerce")
    data = data.dropna(subset=[time_col, event_col])

    curves = []
    if group_col:
        groups = list(data[group_col].dropna().unique())
        for g in groups:
            sub = data[data[group_col] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(sub[time_col], sub[event_col], label=str(g))
            ci = kmf.confidence_interval_
            curves.append({
                "name": str(g),
                "timeline": kmf.timeline.tolist(),
                "survival": kmf.survival_function_.iloc[:, 0].tolist(),
                "lower": ci.iloc[:, 0].tolist(),
                "upper": ci.iloc[:, 1].tolist(),
                "median_survival": _median(kmf),
                "n": int(len(sub)),
            })
        logrank = None
        if len(groups) == 2:
            g0, g1 = groups
            s0, s1 = data[data[group_col] == g0], data[data[group_col] == g1]
            lr = logrank_test(s0[time_col], s1[time_col],
                              s0[event_col], s1[event_col])
            logrank = {"test_statistic": round(float(lr.test_statistic), 4),
                       "p_value": float(f"{lr.p_value:.6g}")}
        return {"curves": curves, "logrank": logrank}

    kmf = KaplanMeierFitter()
    kmf.fit(data[time_col], data[event_col], label="overall")
    ci = kmf.confidence_interval_
    curves.append({
        "name": "Overall",
        "timeline": kmf.timeline.tolist(),
        "survival": kmf.survival_function_.iloc[:, 0].tolist(),
        "lower": ci.iloc[:, 0].tolist(),
        "upper": ci.iloc[:, 1].tolist(),
        "median_survival": _median(kmf),
        "n": int(len(data)),
    })
    return {"curves": curves, "logrank": None}


def _median(kmf):
    m = kmf.median_survival_time_
    return None if (m is None or np.isinf(m) or pd.isna(m)) else round(float(m), 3)


def cox_ph(df, time_col, event_col, covariates):
    """Cox proportional hazards. Returns hazard ratios + CIs per covariate."""
    cols = [time_col, event_col] + list(covariates)
    data = df[cols].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    if data.empty:
        raise ValueError("No complete rows for the chosen Cox covariates.")

    cph = CoxPHFitter()
    cph.fit(data, duration_col=time_col, event_col=event_col)
    s = cph.summary
    rows = []
    for cov in s.index:
        rows.append({
            "covariate": str(cov),
            "hazard_ratio": round(float(s.loc[cov, "exp(coef)"]), 4),
            "hr_lower": round(float(s.loc[cov, "exp(coef) lower 95%"]), 4),
            "hr_upper": round(float(s.loc[cov, "exp(coef) upper 95%"]), 4),
            "coef": round(float(s.loc[cov, "coef"]), 4),
            "p_value": float(f"{s.loc[cov, 'p']:.6g}"),
        })
    return {
        "rows": rows,
        "concordance": round(float(cph.concordance_index_), 4),
        "n": int(len(data)),
        "n_events": int(data[event_col].sum()),
    }


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow goodness of fit
# ---------------------------------------------------------------------------
def hosmer_lemeshow(y_true, y_prob, g=10):
    """Hosmer-Lemeshow C-statistic + per-decile observed/expected for calibration."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob)
    y_true, y_prob = y_true[order], y_prob[order]

    # split into g groups of (approximately) equal size
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
