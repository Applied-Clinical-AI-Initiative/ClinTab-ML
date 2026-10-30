"""teaching.py
Teaching mode: shows, on the user's own data, how common leakage mistakes make
a model look better than it really is. Every arm fits the same model family to
the same patients. In every repeat a lockbox of patients is set aside before
anything else happens and no arm ever sees it, so the gap between what an
analyst would report and the lockbox AUROC is the optimism the mistake
creates. Knows nothing about Flask.
"""
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline

from clintab import ml

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except Exception:                                   # pragma: no cover
    HAS_SMOTE = False


# the subsample uses a fixed seed so every repeat works on the same patients;
# `seeds` drive the lockbox, the 70/15/15 split, smote and model randomness
_SUBSAMPLE_SEED = 0
_MIN_CLASS_COUNT = 50
_SPLIT = (0.70, 0.15, 0.15)


# ============================================================================
# 1. model families
# ============================================================================
MODELS = {
    "logistic": {
        "label": "logistic regression",
        "grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "random_forest": {
        "label": "random forest",
        "grid": {"max_depth": [None, 10], "min_samples_leaf": [1, 5]},
    },
    "gradient_boosting": {
        "label": "gradient boosting",
        "grid": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
    },
}


def _make_estimator(model, seed, n_jobs):
    if model == "logistic":
        return LogisticRegression(max_iter=2000, solver="lbfgs")
    if model == "random_forest":
        # forests fit their trees on joblib threads, never processes, so this
        # is safe in the packaged desktop app too
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=n_jobs)
    # histogram gradient boosting uses openmp threads and is far faster than
    # the classic one on 10,000+ rows
    return HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                          random_state=seed)


# ============================================================================
# 2. the arms, in order from honest to (usually) worst
# ============================================================================
ARMS = [
    {
        "arm": "correct",
        "label": "Done correctly",
        "reported_on": "internal test split",
        "explanation": (
            "Filling in missing values, scaling and choosing the model settings "
            "are all learned from the training patients only, settings are "
            "picked on the validation patients, and the score is reported on "
            "test patients the model never saw. This number should land close "
            "to the lockbox."
        ),
    },
    {
        "arm": "smote_in_training_fold",
        "label": "SMOTE used correctly (training patients only)",
        "reported_on": "internal test split",
        "explanation": (
            "Synthetic patients are added to the training set only, so the test "
            "set still holds real patients and the AUROC stays honest. The cost "
            "is that the model now thinks the outcome is far more common than "
            "it is, so its predicted risks run too high (see the calibration "
            "columns)."
        ),
    },
    {
        "arm": "preprocess_before_split",
        "label": "Missing values and scaling done before splitting",
        "reported_on": "internal test split",
        "explanation": (
            "The medians used to fill in missing values and the scaling were "
            "worked out on all patients, test patients included, before the "
            "split. That leaks a little information about the test set, but "
            "usually so little that the gap stays small."
        ),
    },
    {
        "arm": "tune_on_test",
        "label": "Settings picked on the test set",
        "reported_on": "internal test split (also used to pick settings)",
        "explanation": (
            "The model settings were chosen because they scored best on the "
            "test patients, and then that same score was reported. Keeping the "
            "best of several tries on the same patients flatters the result, "
            "most of all in small datasets."
        ),
    },
    {
        "arm": "apparent",
        "label": "Scored on the training patients",
        "reported_on": "training split",
        "explanation": (
            "The model is scored on the same patients it learned from, so it "
            "has already seen their answers. Flexible models such as random "
            "forests can memorise patients and look close to perfect here."
        ),
    },
    {
        "arm": "smote_before_split",
        "label": "SMOTE applied before splitting",
        "reported_on": "internal test split (includes synthetic patients)",
        "explanation": (
            "SMOTE invents synthetic patients by blending real patients who had "
            "the outcome. Done before the split, synthetic patients end up in "
            "the test set while the real patients they were blended from sit in "
            "the training set, so the model is tested on near copies of "
            "patients it already knows and the score looks much better than it "
            "is. (SMOTE needs complete numbers, so missing values were filled "
            "in on all patients first, as an analyst would have to.)"
        ),
    },
]
ARM_NAMES = [a["arm"] for a in ARMS]
_SMOTE_ARMS = {"smote_in_training_fold", "smote_before_split"}
_QUANTITIES = ("reported_auroc", "lockbox_auroc", "optimism",
               "calibration_slope", "citl")


# ============================================================================
# 3. small helpers
# ============================================================================
def calibration_slope_citl(y_true, p, eps=1e-6):
    """Calibration slope and calibration-in-the-large on (y, p).

    slope = coefficient of logit(p) in a logistic regression of y on logit(p)
    (1 is ideal, below 1 means predictions are too extreme). citl = intercept
    of a logistic regression of y with logit(p) as an offset (0 is ideal,
    negative means the model predicts too high on average). Returns
    (slope, citl), either NaN if statsmodels can't fit it.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    ones = np.ones_like(lp)
    slope = citl = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.GLM(y, np.column_stack([ones, lp]),
                         family=sm.families.Binomial()).fit()
            slope = float(fit.params[1])
        except Exception:
            pass
        try:
            fit = sm.GLM(y, ones[:, None], family=sm.families.Binomial(),
                         offset=lp).fit()
            citl = float(fit.params[0])
        except Exception:
            pass
    return slope, citl


def _auroc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _auroc_se(auc, n_pos, n_neg):
    """Hanley and McNeil standard error of an AUROC."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _say(progress, message):
    if progress is not None:
        progress(message)


def _clean_features(df, feat_cols, coltypes):
    """Categoricals become str (missing stays missing) and numerics become
    float, so mixed-type registry columns can't break the encoders."""
    _, num, cat = ml.build_preprocessor(df, feat_cols, coltypes=coltypes)
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        if c in cat:
            X[c] = df[c].where(df[c].isna(), df[c].astype(str))
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return X


def _stratified_take(y, n_take, seed):
    """Positions of a stratified random subset of size n_take."""
    pos = np.arange(len(y))
    keep, _ = train_test_split(pos, train_size=n_take, stratify=y,
                               random_state=seed)
    return np.sort(keep)


def _split_positions(y, seed):
    """Stratified 70/15/15 split, returns (train, val, test) positions."""
    pos = np.arange(len(y))
    tr, rest = train_test_split(pos, train_size=_SPLIT[0], stratify=y,
                                random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest],
                              random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def _fit_grid(template, grid, X_fit, y_fit):
    """One fitted copy of template per grid point (params use the clf__ prefix
    when template is a pipeline)."""
    fitted = []
    for params in ParameterGrid(grid):
        est = clone(template)
        est.set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_fit, y_fit)
        fitted.append((params, est))
    return fitted


def _pick(fitted, X_sel, y_sel):
    """Grid point with the best AUROC on the selection set (first wins ties)."""
    best, best_auc = None, -np.inf
    for params, est in fitted:
        auc = _auroc(y_sel, est.predict_proba(X_sel)[:, 1])
        if auc > best_auc:
            best, best_auc = (params, est), auc
    return best if best is not None else fitted[0]


def _params_out(params):
    return {k.replace("clf__", ""): (v if isinstance(v, (int, float, str, type(None)))
                                     else str(v))
            for k, v in params.items()}


def _arm_record(seed, arm, params, reported, p_lock, y_lock):
    lock = _auroc(y_lock, p_lock)
    slope, citl = calibration_slope_citl(y_lock, p_lock)
    return {"seed": int(seed), "arm": arm,
            "reported_auroc": reported, "lockbox_auroc": lock,
            "optimism": reported - lock,
            "calibration_slope": slope, "citl": citl,
            "best_params": _params_out(params)}


def _num(x, nd=4):
    """Rounded float, or None for NaN so the result is valid json."""
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def _summary(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if not len(v):
        return {"mean": None, "min": None, "max": None}
    return {"mean": _num(v.mean()), "min": _num(v.min()), "max": _num(v.max())}


# ============================================================================
# 4. one seed
# ============================================================================
def _run_seed(seed, X_dev, y_dev, X_lock, y_lock, coltypes, model, arms,
              n_jobs, progress, seed_tag):
    feat_cols = list(X_dev.columns)
    grid = MODELS[model]["grid"]
    pipe_grid = {f"clf__{k}": v for k, v in grid.items()}
    est = _make_estimator(model, seed, n_jobs)
    tr, va, te = _split_positions(y_dev, seed)
    Xtr, Xva, Xte = X_dev.iloc[tr], X_dev.iloc[va], X_dev.iloc[te]
    ytr, yva, yte = y_dev[tr], y_dev[va], y_dev[te]

    def pre():
        return ml.build_preprocessor(X_dev, feat_cols, coltypes=coltypes)[0]

    records = []

    # correct, tune_on_test and apparent share the same candidate fits (each
    # grid point fitted on the training split only), they just pick or score
    # on different patients
    if {"correct", "tune_on_test", "apparent"} & set(arms):
        _say(progress, f"{seed_tag}: fitting the model the correct way")
        fitted = _fit_grid(Pipeline([("pre", pre()), ("clf", est)]),
                           pipe_grid, Xtr, ytr)
        params, m = _pick(fitted, Xva, yva)
        p_lock = m.predict_proba(X_lock)[:, 1]
        if "correct" in arms:
            rep = _auroc(yte, m.predict_proba(Xte)[:, 1])
            records.append(_arm_record(seed, "correct", params, rep, p_lock, y_lock))
        if "apparent" in arms:
            rep = _auroc(ytr, m.predict_proba(Xtr)[:, 1])
            records.append(_arm_record(seed, "apparent", params, rep, p_lock, y_lock))
        if "tune_on_test" in arms:
            params_t, m_t = _pick(fitted, Xte, yte)
            rep = _auroc(yte, m_t.predict_proba(Xte)[:, 1])
            records.append(_arm_record(seed, "tune_on_test", params_t, rep,
                                       m_t.predict_proba(X_lock)[:, 1], y_lock))

    if "smote_in_training_fold" in arms:
        _say(progress, f"{seed_tag}: SMOTE inside the training patients only")
        template = ImbPipeline([("pre", pre()),
                                ("smote", SMOTE(random_state=seed)),
                                ("clf", est)])
        params, m = _pick(_fit_grid(template, pipe_grid, Xtr, ytr), Xva, yva)
        rep = _auroc(yte, m.predict_proba(Xte)[:, 1])
        records.append(_arm_record(seed, "smote_in_training_fold", params, rep,
                                   m.predict_proba(X_lock)[:, 1], y_lock))

    # both "before split" arms fit the preprocessing on every development
    # patient (train, validation and test together); the lockbox stays out
    if {"preprocess_before_split", "smote_before_split"} & set(arms):
        p_all = pre().fit(X_dev)
        Z_dev = p_all.transform(X_dev)
        Z_lock = p_all.transform(X_lock)

        if "preprocess_before_split" in arms:
            _say(progress, f"{seed_tag}: preprocessing before the split")
            params, m = _pick(_fit_grid(est, grid, Z_dev[tr], ytr), Z_dev[va], yva)
            rep = _auroc(yte, m.predict_proba(Z_dev[te])[:, 1])
            records.append(_arm_record(seed, "preprocess_before_split", params,
                                       rep, m.predict_proba(Z_lock)[:, 1], y_lock))

        if "smote_before_split" in arms:
            _say(progress, f"{seed_tag}: SMOTE before the split")
            Z_res, y_res = SMOTE(random_state=seed).fit_resample(Z_dev, y_dev)
            n = len(y_dev)
            # real patients keep their place in the shared 70/15/15 split and
            # the synthetic ones are scattered 70/15/15 across the same three
            # parts, the same as splitting the oversampled data at random
            if np.allclose(Z_res[:n], Z_dev) and np.array_equal(y_res[:n], y_dev):
                syn = np.arange(n, len(y_res))
                syn = syn[np.random.RandomState(seed).permutation(len(syn))]
                k1 = int(round(_SPLIT[0] * len(syn)))
                k2 = k1 + int(round(_SPLIT[1] * len(syn)))
                rtr = np.r_[tr, syn[:k1]]
                rva = np.r_[va, syn[k1:k2]]
                rte = np.r_[te, syn[k2:]]
            else:                                   # pragma: no cover
                rtr, rva, rte = _split_positions(y_res, seed)
            params, m = _pick(_fit_grid(est, grid, Z_res[rtr], y_res[rtr]),
                              Z_res[rva], y_res[rva])
            rep = _auroc(y_res[rte], m.predict_proba(Z_res[rte])[:, 1])
            records.append(_arm_record(seed, "smote_before_split", params, rep,
                                       m.predict_proba(Z_lock)[:, 1], y_lock))

    order = {a: i for i, a in enumerate(ARM_NAMES)}
    return sorted(records, key=lambda r: order[r["arm"]])


# ============================================================================
# 5. public entry point
# ============================================================================
def leakage_demo(df, outcome, feat_cols, coltypes=None, model="logistic",
                 seeds=(1, 2, 3), max_rows=20000, lockbox_frac=0.2, arms=None,
                 progress=None, n_jobs=-1):
    """Run the same model under each leakage arm and compare what an analyst
    would report with the honest lockbox AUROC. Binary outcomes only.

    df: the dataset. outcome: binary outcome column. feat_cols: predictors.
    coltypes: optional {col: type} as for ml.build_preprocessor (date columns
    are dropped). model: "logistic" | "random_forest" | "gradient_boosting".
    seeds: one repeat per seed; each draws its own lockbox, 70/15/15 split,
    SMOTE and model randomness. Same seeds give the same result.
    max_rows: stratified subsample above this size (None = use every row).
    lockbox_frac: share of patients locked away first in each repeat.
    arms: subset of ARM_NAMES (None = all). progress: optional callback(str).
    n_jobs: threads for random forests (-1 = all cores).

    Returns {"outcome", "model", "model_label", "positive_class", "n_input",
    "n_used", "subsampled", "prevalence", "lockbox_n", "lockbox_events",
    "chance_margin", "development_n", "seeds", "notes", "arms", "per_seed",
    "runtime_s"}. chance_margin is a rough 95% margin (2 standard errors) of
    one lockbox AUROC. Each entry of "arms" is
    {"arm", "label", "explanation", "reported_on", "reported_auroc",
    "lockbox_auroc", "optimism", "calibration_slope", "citl"} where every
    numeric is {"mean", "min", "max"} across seeds (None if it could not be
    computed). "per_seed" holds one flat record per arm and seed.
    """
    t0 = time.time()

    # ---- check the inputs ---------------------------------------------------
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose one of: "
                         + ", ".join(MODELS) + ".")
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' is not in the data.")
    coltypes = dict(coltypes or {})
    dropped_dates = [c for c in (feat_cols or []) if coltypes.get(c) == "date"]
    feat_cols = [c for c in (feat_cols or [])
                 if c != outcome and c not in dropped_dates]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError("These predictor columns are not in the data: "
                         + ", ".join(map(str, missing)) + ".")
    if not feat_cols:
        raise ValueError("Pick at least one predictor (other than the outcome).")
    if arms is None:
        arms = list(ARM_NAMES)
    else:
        arms = [arms] if isinstance(arms, str) else list(dict.fromkeys(arms))
        unknown = [a for a in arms if a not in ARM_NAMES]
        if unknown:
            raise ValueError("Unknown arm(s): " + ", ".join(map(str, unknown))
                             + ". Choose from: " + ", ".join(ARM_NAMES) + ".")
        if not arms:
            raise ValueError("Pick at least one arm.")
    if not HAS_SMOTE and _SMOTE_ARMS & set(arms):    # pragma: no cover
        raise ValueError("The SMOTE arms need the imbalanced-learn package.")
    seeds = [int(s) for s in (seeds or ())]
    if not seeds:
        raise ValueError("Give at least one seed.")
    if not 0 < float(lockbox_frac) < 0.5:
        raise ValueError("lockbox_frac must be between 0 and 0.5.")

    data = df.loc[df[outcome].notna(), feat_cols + [outcome]]
    n_missing_outcome = int(len(df) - len(data))
    n_values = data[outcome].nunique()
    if n_values != 2:
        raise ValueError(
            f"Teaching mode needs a binary outcome (exactly 2 values), but "
            f"'{outcome}' has {n_values} distinct value"
            f"{'' if n_values == 1 else 's'}. Pick a yes/no outcome such as "
            f"30-day readmission or surgical site infection.")
    y_all, classes = ml.encode_outcome(data[outcome], "binary")
    y_all = np.asarray(y_all, dtype=int)
    n_input = int(len(data))
    minority = int(min(y_all.sum(), n_input - y_all.sum()))
    if minority < _MIN_CLASS_COUNT:
        raise ValueError(
            f"Only {minority} patients are in the smaller outcome group; "
            f"teaching mode needs at least {_MIN_CLASS_COUNT} so every split "
            f"has enough events to score.")

    notes = []
    if n_missing_outcome:
        notes.append(f"{n_missing_outcome:,} rows with a missing outcome were "
                     f"left out.")
    if dropped_dates:
        notes.append("Date columns are not used as predictors: "
                     + ", ".join(dropped_dates) + ".")

    # ---- subsample once -------------------------------------------------------
    subsampled = bool(max_rows and n_input > int(max_rows))
    if subsampled:
        keep = _stratified_take(y_all, int(max_rows), _SUBSAMPLE_SEED)
        data, y_all = data.iloc[keep], y_all[keep]
        notes.append(f"The data had {n_input:,} patients; a random "
                     f"{len(data):,} with the same outcome rate were used to "
                     f"keep this quick.")
        _say(progress, f"Using a random {len(data):,} of {n_input:,} patients")
    X = _clean_features(data, feat_cols, coltypes)
    pos = np.arange(len(y_all))

    # ---- one repeat per seed --------------------------------------------------
    # each repeat locks away a fresh stratified lockbox before anything else,
    # so the lockbox's own chance error averages out over the repeats instead
    # of shifting every arm by the same amount
    per_seed = []
    lock_n = dev_n = lock_events = 0
    for i, seed in enumerate(seeds):
        tag = f"Repeat {i + 1} of {len(seeds)}"
        dev, lock = train_test_split(pos, test_size=float(lockbox_frac),
                                     stratify=y_all, random_state=seed)
        dev, lock = np.sort(dev), np.sort(lock)
        lock_n, dev_n = len(lock), len(dev)
        lock_events = int(y_all[lock].sum())
        per_seed.extend(_run_seed(seed, X.iloc[dev], y_all[dev],
                                  X.iloc[lock], y_all[lock], coltypes,
                                  model, arms, n_jobs, progress, tag))
    _say(progress, "Done")
    notes.append(f"In each repeat {lock_n:,} patients ({lockbox_frac:.0%}) were "
                 f"locked away before anything else happened. No arm ever sees "
                 f"them, so the lockbox AUROC is the honest one.")

    # ---- aggregate ------------------------------------------------------------
    arm_out = []
    for spec in ARMS:
        if spec["arm"] not in arms:
            continue
        rows = [r for r in per_seed if r["arm"] == spec["arm"]]
        entry = dict(spec)
        for q in _QUANTITIES:
            entry[q] = _summary([r[q] for r in rows])
        arm_out.append(entry)
    for r in per_seed:
        for q in _QUANTITIES:
            r[q] = _num(r[q])

    # rough 95% margin of one lockbox auroc (hanley and mcneil), so small
    # datasets come with a warning that chance alone moves these numbers
    ref_auc = arm_out[0]["lockbox_auroc"]["mean"] if arm_out else None
    margin = 2 * _auroc_se(ref_auc if ref_auc is not None else 0.75,
                           lock_events, lock_n - lock_events)
    if margin >= 0.05:
        notes.append(f"The lockbox holds only {lock_events:,} patients with the "
                     f"outcome, so any single AUROC here can be off by about "
                     f"{margin:.2f} either way by chance alone. Look at the "
                     f"min to max range, and add repeats or patients before "
                     f"trusting gaps smaller than that.")

    return {
        "outcome": outcome,
        "model": model,
        "model_label": MODELS[model]["label"],
        "positive_class": classes[1],
        "n_input": n_input,
        "n_used": int(len(y_all)),
        "subsampled": subsampled,
        "prevalence": _num(float(y_all.mean())),
        "lockbox_n": int(lock_n),
        "lockbox_events": lock_events,
        "chance_margin": _num(margin, 3),
        "development_n": int(dev_n),
        "seeds": seeds,
        "notes": notes,
        "arms": arm_out,
        "per_seed": per_seed,
        "runtime_s": round(time.time() - t0, 2),
    }
