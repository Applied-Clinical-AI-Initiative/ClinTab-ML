"""samplesize.py
Minimum sample size for developing a clinical prediction model, following
Riley et al. (BMJ 2020;368:m441; Stat Med 2019;38:1276 for a binary
outcome, Stat Med 2019;38:1262 for a continuous outcome), as implemented
in the pmsampsize package. Every number is checked against R pmsampsize
1.1.3 (CRAN) in tests/test_samplesize.py.

Binary outcome criteria (pmsampsize 1.1.3 has these three, no fourth):
  1. expected uniform shrinkage of at least 0.9
  2. small optimism (default 0.05) in the apparent Nagelkerke R2
  3. overall risk estimated within a margin of 0.05

The C statistic to Cox-Snell R2 conversion (Riley, Van Calster and Collins,
Stat Med 2021;40:859) simulates one million linear predictor values. It is
replayed here with R's own random number stream (Mersenne-Twister, inversion
normals, set.seed(123456)) so the R2 and the sample sizes equal what R
pmsampsize prints.
"""
import math

import numpy as np
from scipy import special, stats

PMSAMPSIZE_SEED = 123456


# ---------------------------------------------------------------------------
# c statistic -> cox-snell r2, same simulation as pmsampsize:::cstat2rsq
# ---------------------------------------------------------------------------
def _r_mt_key(seed):
    """the 624 word mersenne-twister state R builds in set.seed(seed)"""
    s = int(seed) & 0xFFFFFFFF
    for _ in range(50):
        s = (69069 * s + 1) & 0xFFFFFFFF
    key = np.empty(625, dtype=np.uint32)
    for j in range(625):
        s = (69069 * s + 1) & 0xFFFFFFFF
        key[j] = s
    return key[1:]  # the first word is R's position counter


def _r_rnorm(seed, n):
    """n standard normals, identical to set.seed(seed); rnorm(n) in R
    (default Mersenne-Twister with inversion)"""
    bg = np.random.MT19937()
    bg.state = {"bit_generator": "MT19937",
                "state": {"key": _r_mt_key(seed), "pos": 624}}
    u = bg.random_raw(2 * n).astype(np.float64) * 2.3283064365386963e-10
    tiny = 0.5 * 2.328306437080797e-10
    u = np.where(u <= 0, tiny, u)
    u = np.where(1 - u <= 0, 1 - tiny, u)
    big = 134217728.0  # 2^27
    return special.ndtri((np.floor(big * u[0::2]) + u[1::2]) / big)


def _logistic_lr(x, y):
    """likelihood ratio statistic of glm(y ~ x, binomial) against the null"""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = X @ beta
        p = special.expit(eta)
        w = p * (1 - p)
        step = np.linalg.solve((X * w[:, None]).T @ X, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    eta = X @ beta
    ll = np.sum(y * eta - np.logaddexp(0, eta))
    ybar = y.mean()
    ll0 = len(y) * (ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar))
    return 2 * (ll - ll0)


def cstat_to_csrsquared(cstatistic, prevalence, seed=PMSAMPSIZE_SEED):
    """Approximate Cox-Snell R2 from a C statistic and outcome prevalence
    (Riley et al. Stat Med 2021;40:859): the linear predictor is normal with
    equal variance in events and non-events, a large data set is simulated
    and a logistic model refitted. Unrounded; pmsampsize rounds it to 4
    decimals before use, as binary_sample_size does."""
    c, p = float(cstatistic), float(prevalence)
    if not 0.5 < c < 1:
        raise ValueError("C statistic must be between 0.5 and 1.")
    if not 0 < p < 1:
        raise ValueError("Prevalence must be between 0 and 1.")
    n = 1000000
    mu = math.sqrt(2) * special.ndtri(c)
    n0, n1 = int(p * n), int((1 - p) * n)
    lp = _r_rnorm(seed, n0 + n1)
    lp[n0:] = mu + lp[n0:]
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    return float(1 - math.exp(-_logistic_lr(lp, y) / n))


# ---------------------------------------------------------------------------
# binary outcome
# ---------------------------------------------------------------------------
def _ceil(x):
    # guard against float noise like 85.00000000000001
    return int(math.ceil(round(x, 9)))


def _check_parameters(n_parameters):
    try:
        k = float(n_parameters)
    except (TypeError, ValueError):
        raise ValueError("Number of parameters must be a whole number.")
    if not k.is_integer() or k < 1:
        raise ValueError("Number of parameters must be a whole number of at least 1.")
    return int(k)


def _n_for_shrinkage(k, r2, s):
    """riley et al. stat med 2019 eq 11: n = k / ((S - 1) ln(1 - R2cs / S))"""
    return int(math.ceil(k / ((s - 1) * math.log(1 - r2 / s))))


def binary_sample_size(prevalence, n_parameters, csrsquared=None, cstatistic=None,
                       nagrsquared=None, shrinkage=0.9, delta=0.05, margin=0.05,
                       seed=PMSAMPSIZE_SEED):
    """Minimum sample size to develop a prediction model for a binary outcome
    (Riley et al. BMJ 2020;368:m441), same as R pmsampsize(type = "b").

    prevalence: anticipated outcome proportion. n_parameters: candidate
    predictor parameters (degrees of freedom, not variables). Give one of
    csrsquared (anticipated Cox-Snell R2), cstatistic (converted as in
    pmsampsize) or nagrsquared (Nagelkerke R2)."""
    p = float(prevalence)
    if not 0 < p < 1:
        raise ValueError("Prevalence must be between 0 and 1.")
    k = _check_parameters(n_parameters)
    s = float(shrinkage)
    if not 0 < s < 1:
        raise ValueError("Shrinkage must be between 0 and 1.")
    if not 0 < float(delta) < 1:
        raise ValueError("delta must be between 0 and 1.")
    if not 0 < float(margin) < 1:
        raise ValueError("margin must be between 0 and 1.")
    given = [v is not None for v in (csrsquared, cstatistic, nagrsquared)]
    if sum(given) != 1:
        raise ValueError("Give exactly one of the Cox-Snell R2, the C statistic "
                         "or the Nagelkerke R2.")

    # max cox-snell r2 for this prevalence (riley et al. 2018 eq 23)
    def max_r2_at(n):
        e = n * p
        lnl0 = e * math.log(e / n) + (n - e) * math.log(1 - e / n)
        return 1 - math.exp(2 * lnl0 / n)

    notes = []
    if csrsquared is not None:
        r2 = float(csrsquared)
        if not 0 < r2 < 1:
            raise ValueError("Cox-Snell R2 must be between 0 and 1.")
        source = "given"
    elif cstatistic is not None:
        raw = cstat_to_csrsquared(cstatistic, p, seed)
        r2 = round(raw, 4)
        source = "from C statistic"
        notes.append(f"Cox-Snell R2 = {r2} was approximated from C statistic "
                     f"{cstatistic} and prevalence {p} (Riley et al. Stat Med "
                     "2021;40:859), with the same simulation and seed as "
                     "pmsampsize.")
    else:
        nag = float(nagrsquared)
        if not 0 < nag < 1:
            raise ValueError("Nagelkerke R2 must be between 0 and 1.")
        r2 = round(nag * max_r2_at(k), 3)
        source = "from Nagelkerke R2"
    if r2 <= 0:
        raise ValueError("The anticipated Cox-Snell R2 must be above 0.")
    if s < r2:
        raise ValueError("Shrinkage target is lower than the Cox-Snell R2, so "
                         "criterion 1 cannot be computed.")

    # criterion 1, expected shrinkage >= s
    n1 = _n_for_shrinkage(k, r2, s)
    max_r2 = max_r2_at(n1)
    if r2 > max_r2:
        raise ValueError(f"Cox-Snell R2 {r2} is above the largest possible "
                         f"value for this prevalence ({max_r2:.3f}).")
    nag_r2 = r2 / max_r2
    # criterion 2, apparent minus adjusted nagelkerke r2 <= delta
    s2 = r2 / (r2 + float(delta) * max_r2)
    n2 = _n_for_shrinkage(k, r2, s2)
    # criterion 3, overall risk within +/- margin (pmsampsize uses z = 1.96)
    n3 = int(math.ceil((1.96 / float(margin)) ** 2 * p * (1 - p)))

    def row(name, key, n, shrink, text):
        e = n * p
        return {"name": name, "key": key, "n": n, "events": _ceil(e),
                "events_exact": round(e, 4), "epp": round(e / k, 2),
                "shrinkage": None if shrink is None else round(shrink, 3),
                "description": text}

    criteria = [
        row(f"Criterion 1: expected shrinkage of at least {s:g}", "shrinkage", n1, s,
            "Limits overfitting so predictions need little shrinkage."),
        row(f"Criterion 2: optimism in Nagelkerke R2 of at most {float(delta):g}",
            "optimism", n2, s2,
            "Keeps the apparent and the optimism-adjusted R2 close."),
        row(f"Criterion 3: overall risk within {float(margin):g}", "overall_risk",
            n3, None, "Estimates the average outcome risk (intercept) precisely."),
    ]
    nfinal = max(n1, n2, n3)
    binding = [c["name"] for c in criteria if c["n"] == nfinal]
    e_final = nfinal * p
    notes += [
        f"The minimum is {nfinal} patients with {_ceil(e_final)} events "
        f"(EPP {round(e_final / k, 2)}), set by {binding[0].split(':')[0].lower()}.",
        "Parameters means candidate predictor degrees of freedom, not variables: "
        "a 4 level categorical predictor counts as 3, a spline with 4 knots as 3, "
        "and every predictor screened before modelling counts.",
        "If the R2 is unknown, Riley et al. suggest a conservative Cox-Snell R2 "
        f"of 15% of the maximum ({round(0.15 * max_r2, 4)} here).",
    ]
    return {
        "outcome": "binary",
        "criteria": criteria,
        "min_n": nfinal,
        "min_events": _ceil(e_final),
        "events_exact": round(e_final, 4),
        "epp": round(e_final / k, 2),
        "binding_criterion": binding[0],
        "final_shrinkage": round(max(s, s2), 3),
        "assumptions": {
            "prevalence": p,
            "n_parameters": k,
            "csrsquared": r2,
            "csrsquared_source": source,
            "cstatistic": None if cstatistic is None else float(cstatistic),
            "nagrsquared_input": None if nagrsquared is None else float(nagrsquared),
            "max_csrsquared": round(max_r2, 4),
            "nagelkerke_r2": round(nag_r2, 4),
            "shrinkage": s,
            "delta": float(delta),
            "margin": float(margin),
            "seed": int(seed) if cstatistic is not None else None,
        },
        "notes": notes,
        "reference": ("Riley et al. BMJ 2020;368:m441; Stat Med 2019;38:1276; "
                      "R pmsampsize 1.1.3"),
    }


def compare_with_data(result, n, events, n_parameters):
    """Check a data set against a sample size result. n and events are what
    the data actually has, n_parameters what the model will actually use."""
    n, events = int(n), int(events)
    k = _check_parameters(n_parameters)
    need_n = int(result["min_n"])
    need_e = result.get("min_events")
    adequate = n >= need_n and (need_e is None or events >= need_e)
    notes = []
    planned_k = result.get("assumptions", {}).get("n_parameters")
    if planned_k is not None and planned_k != k:
        notes.append(f"The requirement was worked out for {planned_k} parameters "
                     f"but the model uses {k}. Rerun the calculation with {k}.")
    prev = result.get("assumptions", {}).get("prevalence")
    if prev and n > 0:
        obs = events / n
        if abs(obs - prev) > 0.25 * prev:
            notes.append(f"The observed prevalence ({obs:.3f}) differs from the "
                         f"assumed {prev:g}. Rerun with the observed value.")
    if adequate:
        notes.append("The data meet the minimum sample size.")
    else:
        notes.append(f"The data fall short of the minimum ({need_n} patients"
                     + (f", {need_e} events" if need_e is not None else "")
                     + "). Expect overfitting; use fewer parameters, penalised "
                     "regression with care, or more data.")
    return {
        "adequate": bool(adequate),
        "n": n,
        "events": events,
        "epp": round(events / k, 2),
        "shortfall": max(0, need_n - n),
        "events_shortfall": max(0, need_e - events) if need_e is not None else None,
        "required_n": need_n,
        "required_events": need_e,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# continuous outcome
# ---------------------------------------------------------------------------
def continuous_sample_size(rsquared, n_parameters, intercept, sd, shrinkage=0.9,
                           delta=0.05, mmoe=1.1):
    """Minimum sample size for a continuous outcome (Riley et al. Stat Med
    2019;38:1262), as R pmsampsize(type = "c"): 1 shrinkage, 2 optimism in
    R2, 3 residual SD within a multiplicative margin of error of 1.1,
    4 mean outcome (intercept) within the same margin.

    note: when criterion 4 needs more than the other three, pmsampsize
    1.1.3 (R and Python) flips the sign of the t quantile inside its search
    loop and stops one patient later. Here the search uses the correct
    sign, so the two only agree when criterion 4 does not bind on its own."""
    r2 = float(rsquared)
    if not 0 < r2 < 1:
        raise ValueError("R2 must be between 0 and 1.")
    k = _check_parameters(n_parameters)
    s = float(shrinkage)
    if not 0 < s < 1:
        raise ValueError("Shrinkage must be between 0 and 1.")
    intercept, sd = float(intercept), float(sd)
    if sd <= 0:
        raise ValueError("SD must be above 0.")
    if intercept == 0:
        raise ValueError("Mean outcome (intercept) must not be 0 for a "
                         "multiplicative margin of error.")

    def shrink_at(n):
        return 1 + (k - 2) / (n * math.log(1 - (r2 * (n - k - 1) + k) / (n - 1)))

    # criterion 1, search upward from k + 2 as pmsampsize does
    n1 = k + 2
    with np.errstate(all="ignore"):
        es = _safe(shrink_at, n1)
        while es is None or es < s:
            n1 += 1
            es = _safe(shrink_at, n1)
    n2 = int(math.ceil(1 + k * (1 - r2) / float(delta)))
    n3 = 234 + k

    def int_mmoe(n):
        half = math.sqrt(sd ** 2 * (1 - r2) / n) * -stats.t.ppf(0.025, n - k - 1)
        return (intercept + half) / intercept, intercept - half, intercept + half

    n4 = max(n1, n2, n3)
    while int_mmoe(n4)[0] > mmoe:
        n4 += 1
    nfinal = max(n1, n2, n3, n4)
    _, lci, uci = int_mmoe(n4)

    def row(name, key, n):
        return {"name": name, "key": key, "n": n, "spp": round(n / k, 2),
                "shrinkage": round(shrink_at(n), 3)}

    notes = [f"The minimum is {nfinal} patients ({round(nfinal / k, 2)} per parameter)."]
    if n4 > max(n1, n2, n3):
        notes.append(f"Criterion 4 sets the minimum here. pmsampsize 1.1.3 would "
                     f"report {max(n1, n2, n3) + 1} because of a sign slip in its "
                     "search loop; this value is the corrected one.")
    criteria = [
        row(f"Criterion 1: expected shrinkage of at least {s:g}", "shrinkage", n1),
        row(f"Criterion 2: optimism in R2 of at most {float(delta):g}", "optimism", n2),
        row(f"Criterion 3: residual SD within a margin of {mmoe:g}", "residual_sd", n3),
        row(f"Criterion 4: mean outcome within a margin of {mmoe:g}", "intercept", n4),
    ]
    return {
        "outcome": "continuous",
        "criteria": criteria,
        "min_n": nfinal,
        "spp": round(nfinal / k, 2),
        "final_shrinkage": round(shrink_at(nfinal), 3),
        "intercept_ci": [round(float(lci), 2), round(float(uci), 2)],
        "assumptions": {"rsquared": r2, "n_parameters": k, "intercept": intercept,
                        "sd": sd, "shrinkage": s, "delta": float(delta), "mmoe": mmoe},
        "notes": notes,
        "reference": "Riley et al. Stat Med 2019;38:1262; R pmsampsize 1.1.3",
    }


def _safe(f, n):
    try:
        v = f(n)
    except (ValueError, ZeroDivisionError):
        return None
    return v if np.isfinite(v) else None
