"""ml.py
All the sklearn: preprocessing, the 15 models, validation-set grid search,
optional SMOTE on the training fold only, metrics (with calibration,
confidence intervals and decision curves from validation.py), adjusted odds
ratios, feature importance, and single-row prediction. Knows nothing about
Flask, everything takes/returns plain DataFrames, dicts and numpy arrays so
it can be tested in isolation.
"""
import re

import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, label_binarize,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.inspection import permutation_importance

from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, fbeta_score, brier_score_loss, make_scorer,
    accuracy_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, auc,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from clintab import validation

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except Exception:                                   # pragma: no cover
    HAS_SMOTE = False


# ============================================================================
# 1. task detection + model catalogue
# ============================================================================
def determine_task(series):
    """binary | multiclass | continuous, from the outcome column."""
    s = series.dropna()
    n_unique = s.nunique()
    if n_unique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s) and n_unique > 10:
        return "continuous"
    return "multiclass"


def _sklearn_version():
    return tuple(int(x) for x in re.findall(r"\d+", sklearn.__version__)[:2])


# scikit-learn 1.8 deprecated LogisticRegression(penalty=...) (removal in
# 1.10): l1_ratio now picks the penalty (0 = l2, 1 = l1, in between =
# elastic net) and C=np.inf means no penalty. releases before 1.8 only
# understand penalty=, so spell it the way the installed version expects.
_LOGREG_L1_RATIO_API = _sklearn_version() >= (1, 8)

# grid keys that would quietly turn the unpenalized reference model into a
# penalized one
_PENALTY_KEYS = {"clf__C", "clf__penalty", "clf__l1_ratio"}


def _logistic(penalty, **kw):
    """LogisticRegression with penalty "none" | "l2" | "l1" | "elasticnet",
    built without deprecation warnings on scikit-learn 1.3 through 1.9+."""
    if _LOGREG_L1_RATIO_API:
        if penalty == "none":
            kw["C"] = np.inf
            kw["l1_ratio"] = 0.0
        elif penalty == "l2":
            kw["l1_ratio"] = 0.0
        elif penalty == "l1":
            kw["l1_ratio"] = 1.0
        else:
            kw.setdefault("l1_ratio", 0.5)
    else:
        kw["penalty"] = None if penalty == "none" else penalty
        if penalty == "elasticnet":
            kw.setdefault("l1_ratio", 0.5)
    return LogisticRegression(**kw)


def _svm():
    """SVM with Platt-scaled (sigmoid) probabilities. scikit-learn 1.9
    deprecated SVC(probability=True) (removal in 1.11); the documented
    replacement CalibratedClassifierCV(SVC(), ensemble=False) does the same
    thing (5-fold sigmoid calibration, then one SVC refit on all the
    training data) and works on 1.3+ too, so grids name clf__estimator__C."""
    return CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False)


def model_catalogue(task):
    """The 15 models + sensible default grids, filtered by task.

    Classification (10): LogisticRegression is plain unpenalized maximum
    likelihood with no grid, the reference model clinicians expect.
    RidgeLogistic (L2), LassoLogistic (L1) and ElasticNetLogistic are the
    penalized versions, tuned over C (and l1_ratio). Then KNN, DecisionTree,
    RandomForest, GradientBoosting, SVM (Platt-scaled probabilities, see
    _svm) and NaiveBayes.
    Regression (5): LinearRegression, Ridge, Lasso, ElasticNet,
    RandomForestRegressor.
    """
    if task in ("binary", "multiclass"):
        # liblinear only fits an ovr scheme and raises on n_classes >= 3 in
        # current sklearn, fall back to saga (also supports l1) there.
        lasso_solver = "saga" if task == "multiclass" else "liblinear"
        lasso_max_iter = 3000 if task == "multiclass" else 2000
        return {
            "LogisticRegression": {
                # tighter tol than the 1e-4 default: odds ratios then agree
                # with R's glm to about 7 significant figures on recent
                # scikit-learn (about 5 on 1.3, whose lbfgs stops earlier)
                "model": _logistic("none", max_iter=5000, solver="lbfgs", tol=1e-6),
                "params": {},
            },
            "RidgeLogistic": {
                "model": _logistic("l2", C=1.0, max_iter=2000, solver="lbfgs"),
                "params": {"clf__C": [0.01, 0.1, 1, 10]},
            },
            "LassoLogistic": {
                "model": _logistic("l1", solver=lasso_solver, max_iter=lasso_max_iter),
                "params": {"clf__C": [0.01, 0.1, 1, 10]},
            },
            "ElasticNetLogistic": {
                "model": _logistic("elasticnet", solver="saga", l1_ratio=0.5, max_iter=3000),
                "params": {"clf__C": [0.1, 1, 10], "clf__l1_ratio": [0.2, 0.5, 0.8]},
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {"clf__n_neighbors": [3, 5, 7, 11]},
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {"clf__max_depth": [3, 5, 10, None],
                           "clf__min_samples_leaf": [1, 5, 20]},
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42, n_jobs=-1),
                "params": {"clf__n_estimators": [200, 400],
                           "clf__max_depth": [None, 10, 20]},
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {"clf__n_estimators": [100, 200],
                           "clf__learning_rate": [0.05, 0.1],
                           "clf__max_depth": [2, 3]},
            },
            "SVM": {
                "model": _svm(),
                "params": {"clf__estimator__C": [0.1, 1, 10],
                           "clf__estimator__kernel": ["rbf", "linear"]},
            },
            "NaiveBayes": {
                "model": GaussianNB(),
                "params": {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
            },
        }
    # regression
    return {
        "LinearRegression": {"model": LinearRegression(), "params": {}},
        "Ridge": {"model": Ridge(random_state=42),
                  "params": {"clf__alpha": [0.1, 1.0, 10.0]}},
        "Lasso": {"model": Lasso(random_state=42, max_iter=5000),
                  "params": {"clf__alpha": [0.001, 0.01, 0.1, 1.0]}},
        "ElasticNet": {"model": ElasticNet(random_state=42, max_iter=5000),
                       "params": {"clf__alpha": [0.01, 0.1, 1.0],
                                  "clf__l1_ratio": [0.2, 0.5, 0.8]}},
        "RandomForestRegressor": {"model": RandomForestRegressor(random_state=42, n_jobs=-1),
                                  "params": {"clf__n_estimators": [200, 400],
                                             "clf__max_depth": [None, 10, 20]}},
    }


def default_grids(task):
    return {name: cfg["params"] for name, cfg in model_catalogue(task).items()}


# ============================================================================
# 2. preprocessing
# ============================================================================
def feature_columns(df, outcome, exclude):
    exclude = set(exclude or [])
    return [c for c in df.columns if c != outcome and c not in exclude]


def _is_categorical(df, c, coltypes=None):
    """Tagged binary/categorical in coltypes, or (untagged) a non-numeric
    dtype. Shared by the sklearn preprocessor and the odds-ratio design."""
    t = (coltypes or {}).get(c)
    if t in ("binary", "categorical"):
        return True
    if t == "continuous":
        return False
    return not pd.api.types.is_numeric_dtype(df[c])


def build_preprocessor(df, feat_cols, coltypes=None, scale=True):
    """ColumnTransformer: impute+scale numerics, impute+one-hot categoricals.

    coltypes: optional {col: 'binary'|'categorical'|'continuous'|'date'|...},
    e.g. from stats.detect_column_types / meta['coltypes']. Columns tagged
    'binary' or 'categorical' there are one-hot encoded even if their pandas
    dtype is numeric (integer-coded categoricals like ASA class). Columns not
    present in coltypes fall back to dtype-based inference: object/category
    dtype -> categorical, otherwise numeric.
    """
    cat = [c for c in feat_cols if _is_categorical(df, c, coltypes)]
    num = [c for c in feat_cols if c not in cat]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        remainder="drop",
    )
    return pre, num, cat


def _norm_label(value):
    """Label as a string, with 1.0 and 1 treated as the same class (a column
    reads as float when it has missing values and as int when it doesn't).
    Other strings, "01" for one, are left alone."""
    text = str(value)
    return re.sub(r"^(-?\d+)\.0+$", r"\1", text)


def encode_outcome(series, task, classes=None):
    """Returns (y, classes) for classification or (y, None) for regression.

    Missing outcomes stay NaN (y is then a float array) instead of turning
    into an extra "nan" class. Pass the classes a model was trained with to
    get the same label coding on a validation/test set that happens to be
    missing one of them; labels that were never seen in training become NaN.
    """
    if task == "continuous":
        return pd.to_numeric(series, errors="coerce").values, None
    labels = {v: _norm_label(v) for v in series.dropna().unique()}
    if classes is None:
        classes = sorted(set(labels.values()))
    index = {_norm_label(c): i for i, c in enumerate(classes)}
    code = {v: index.get(lab, np.nan) for v, lab in labels.items()}
    y = series.map(code).to_numpy(dtype=float)
    if not np.isnan(y).any():
        y = y.astype(int)
    return y, list(classes)


def _drop_missing_outcome(X, y, task):
    """Rows with a missing outcome can't be scored or fit; classification
    labels go back to int once the NaNs are gone."""
    keep = ~pd.isna(y)
    y = y[keep]
    if task != "continuous":
        y = y.astype(int)
    return X[keep], y


# ============================================================================
# 3. scorers
# ============================================================================
def make_scoring(metric, task):
    if task == "continuous":
        return {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }.get(metric, "r2")
    # sklearn's plain "roc_auc"/"recall"/"precision" scorers assume a binary
    # outcome (average="binary") and raise valueerror on a multiclass one,
    # route those to their multiclass-safe equivalents instead.
    average = "macro" if task == "multiclass" else "binary"
    if metric == "f2":
        return make_scorer(fbeta_score, beta=2, average=average, zero_division=0)
    if metric == "f1":
        return make_scorer(f1_score, average=average, zero_division=0)
    if task == "multiclass":
        return {
            "roc": "roc_auc_ovr", "roc_auc": "roc_auc_ovr",
            "auprc": "average_precision", "average_precision": "average_precision",
            "recall": "recall_macro", "sensitivity": "recall_macro",
            "precision": "precision_macro", "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
        }.get(metric, "roc_auc_ovr")
    return {
        "roc": "roc_auc", "roc_auc": "roc_auc",
        "auprc": "average_precision", "average_precision": "average_precision",
        "recall": "recall", "sensitivity": "recall",
        "precision": "precision", "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }.get(metric, "roc_auc")


# ============================================================================
# 4. train one model
# ============================================================================
def train_one_model(name, df_train, df_val, outcome, feat_cols, task,
                    param_grid=None, scoring="roc", use_smote=False,
                    do_grid_search=True, coltypes=None, cv_folds=None):
    """Fit one model. By default, grid search is evaluated on the VALIDATION
    set only (via PredefinedSplit), then the best params are refit on the
    training set alone. coltypes: optional {col: type}, passed to
    build_preprocessor so integer-coded categoricals get one-hot encoded
    instead of scaled.

    cv_folds: if set (e.g. 5), grid search instead uses k-fold CV within the
    TRAINING set only, refitting on the full training set with the winning
    params. Useful on small datasets, where scoring candidates against a
    single validation split makes hyperparameter selection noisy. Either
    way, df_val is never touched by the search itself, only by the
    evaluate() call afterwards, so the validation metrics stay leakage-free
    regardless of which path is used.

    LogisticRegression is the unpenalized reference model: C, penalty and
    l1_ratio entries in its grid are ignored, so it is never tuned on them.

    Returns (fitted_pipeline, info_dict).
    """
    cfg = model_catalogue(task)[name]
    estimator = cfg["model"]
    grid = param_grid if param_grid is not None else cfg["params"]
    if name == "LogisticRegression" and grid:
        # the reference model stays unpenalized, a C or penalty grid would
        # turn it into a ridge/lasso fit under the same name
        grid = {k: v for k, v in grid.items() if k not in _PENALTY_KEYS}
    if isinstance(estimator, CalibratedClassifierCV) and grid:
        # older or hand-edited svm grids name the svc's own parameters
        # (clf__C, clf__kernel), route those to the wrapped estimator
        own = set(estimator.get_params(deep=False))
        grid = {(k if not k.startswith("clf__") or k[5:].split("__")[0] in own
                 else "clf__estimator__" + k[5:]): v for k, v in grid.items()}

    pre, num, cat = build_preprocessor(df_train, feat_cols, coltypes=coltypes)

    steps = [("pre", pre)]
    if use_smote and task in ("binary", "multiclass") and HAS_SMOTE:
        steps.append(("smote", SMOTE(random_state=42)))
        pipe = ImbPipeline(steps + [("clf", estimator)])
    else:
        pipe = Pipeline(steps + [("clf", estimator)])

    Xtr, Xva = df_train[feat_cols], df_val[feat_cols]
    ytr, classes = encode_outcome(df_train[outcome], task)
    yva, _ = encode_outcome(df_val[outcome], task, classes)

    # drop rows with missing outcome
    Xtr, ytr = _drop_missing_outcome(Xtr, ytr, task)
    Xva, yva = _drop_missing_outcome(Xva, yva, task)

    score = make_scoring(scoring, task)
    best_params = "default"

    if do_grid_search and grid and cv_folds:
        # k-fold cv within the training set only, df_val is never part of
        # the search, so it stays a clean held-out set for evaluate() later.
        gs = GridSearchCV(pipe, grid, scoring=score, cv=cv_folds, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        best_params = gs.best_params_
        fitted = gs.best_estimator_
    elif do_grid_search and grid:
        # grid search scores each candidate on the validation set only
        # (predefinedsplit: fold -1 = train, fold 0 = val). refit off so the
        # winning params are then re-fit on the training set alone, this keeps
        # the validation metrics we report afterwards honest (no leakage).
        X_all = pd.concat([Xtr, Xva], axis=0)
        y_all = np.concatenate([ytr, yva])
        test_fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xva))]
        ps = PredefinedSplit(test_fold)
        gs = GridSearchCV(pipe, grid, scoring=score, cv=ps, n_jobs=-1, refit=False)
        gs.fit(X_all, y_all)
        best_params = gs.best_params_
        pipe.set_params(**best_params)
        fitted = pipe.fit(Xtr, ytr)
    else:
        # train on train only, no tuning
        fitted = pipe.fit(Xtr, ytr)

    info = {"name": name, "best_params": _jsonable(best_params),
            "classes": classes, "task": task,
            "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
            "smote": bool(use_smote and task != "continuous" and HAS_SMOTE),
            "tuned": bool(do_grid_search and grid),
            "cv_folds": int(cv_folds) if (do_grid_search and grid and cv_folds) else None}
    return fitted, info


def _jsonable(p):
    if isinstance(p, dict):
        return {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                for k, v in p.items()}
    return str(p)


# ============================================================================
# 5. metrics + plot coordinates
# ============================================================================
def predict_scores(pipe, X, task):
    """Probability of positive class (binary) / predicted value (regression)."""
    if task == "continuous":
        return pipe.predict(X)
    proba = pipe.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba  # multiclass


def _r4(x):
    """round to 4 places, None when missing or not finite."""
    x = validation._num(x)
    return None if x is None else round(x, 4)


def _r6_list(values):
    return [None if v is None else round(float(v), 6) for v in values]


def binary_metrics(y_true, y_score, threshold=0.5, with_ci=True, n_boot=500):
    """Metrics + plot coordinates for a binary outcome.

    Threshold-dependent metrics (Sensitivity, Specificity, PPV, NPV, F1 and
    the confusion matrix) use the given threshold. Calibration comes from
    validation.calibration_stats: CalibrationSlope (1 = ideal),
    CalibrationInTheLarge (0 = ideal), OE_ratio (observed / expected events,
    1 = ideal) and ICI (mean absolute calibration error). Undefined values
    are None.

    with_ci adds metrics["CI"] = {name: [lo, hi]} with 95% intervals:
    AUROC by DeLong, AUPRC and Brier by percentile bootstrap (n_boot
    resamples, seed 42), calibration slope and intercept by Wald, O:E by
    log(O/E) +/- 1.96 sqrt(1/O).

    coords adds calibration_flexible (lowess curve), calibration_stats (the
    full calibration_stats dict incl. E50/E90 and a reason when something
    is undefined), dca (net benefit at thresholds 0.01 to 0.99, the same
    grid for every model so curves can be overlaid) and risk_histogram
    ({edges, counts} of the predicted risks, for the calibration plot).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    auprc = float(average_precision_score(y_true, y_score))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    cal = validation.calibration_stats(y_true, y_score)
    metrics = {
        "AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
        "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
        "PPV": round(ppv, 4), "NPV": round(npv, 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "CalibrationSlope": _r4(cal["slope"]),
        "CalibrationInTheLarge": _r4(cal["citl"]),
        "OE_ratio": _r4(cal["oe_ratio"]),
        "ICI": _r4(cal["ici"]),
        "Threshold": threshold,
        "N": int(len(y_true)),
    }
    if with_ci:
        _, auc_lo, auc_hi = validation.auroc_ci(y_true, y_score)
        metrics["CI"] = {
            "AUROC": [_r4(auc_lo), _r4(auc_hi)],
            "AUPRC": [_r4(v) for v in validation.bootstrap_ci(
                y_true, y_score, "auprc", n_boot=n_boot)],
            "Brier": [_r4(v) for v in validation.bootstrap_ci(
                y_true, y_score, "brier", n_boot=n_boot)],
            "CalibrationSlope": [_r4(v) for v in cal["slope_ci"]],
            "CalibrationInTheLarge": [_r4(v) for v in cal["citl_ci"]],
            "OE_ratio": [_r4(v) for v in cal["oe_ci"]],
        }
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_score)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])
    coords = {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics["AUROC"]},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist(), "ap": metrics["AUPRC"]},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration": {"prob_true": frac_pos.tolist(), "prob_pred": mean_pred.tolist()},
    }
    flex = validation.flexible_calibration_curve(y_true, y_score)
    coords["calibration_flexible"] = {**flex, "p_grid": _r6_list(flex["p_grid"]),
                                      "observed": _r6_list(flex["observed"])}
    coords["calibration_stats"] = {
        k: ([_r4(x) for x in v] if isinstance(v, list)
            else _r4(v) if isinstance(v, float) else v)
        for k, v in cal.items()}
    dca = validation.decision_curve(y_true, y_score)
    coords["dca"] = {k: _r6_list(v) for k, v in dca.items()}
    # 40 bins from 0 to the highest predicted risk, for the histogram under
    # the calibration curve without shipping every prediction
    finite = y_score[np.isfinite(y_score)]
    edges = np.linspace(0.0, max(float(finite.max()) if finite.size else 1.0, 1e-6), 41)
    coords["risk_histogram"] = {"edges": _r6_list(edges),
                                "counts": np.histogram(finite, bins=edges)[0].astype(int).tolist()}
    return metrics, coords


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(rmse, 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "N": int(len(y_true)),
    }
    coords = {
        "residuals": {"pred": y_pred.tolist(), "resid": (y_true - y_pred).tolist()},
        "pred_vs_actual": {"actual": y_true.tolist(), "pred": y_pred.tolist()},
    }
    return metrics, coords


def multiclass_metrics(y_true, y_pred, y_proba=None):
    """Accuracy/BalancedAccuracy/MacroF1, plus one-vs-rest ROC/PR curve
    coordinates per class when y_proba (from predict_scores) is given."""
    y_true = np.asarray(y_true)
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "BalancedAccuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "MacroF1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "N": int(len(y_true)),
    }

    coords = {}
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        # one column of y_proba per class the model was trained on, whether or
        # not every class shows up in y_true
        classes = np.arange(y_proba.shape[1])
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        roc_series, pr_series = [], []
        for i, cls in enumerate(classes):
            if y_bin[:, i].sum() in (0, len(y_bin)):
                continue                      # class absent (or alone) in this data
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            roc_series.append({"name": f"class {cls}", "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                                "auc": round(float(auc(fpr, tpr)), 4)})
            pr_series.append({"name": f"class {cls}", "recall": rec.tolist(),
                               "precision": prec.tolist(),
                               "ap": round(float(average_precision_score(y_bin[:, i], y_proba[:, i])), 4)})
        coords = {"roc": roc_series, "pr": pr_series}

    return metrics, coords


def evaluate(pipe, df, outcome, feat_cols, task, threshold=0.5, classes=None,
             with_ci=True):
    """Metrics + plot coordinates on an arbitrary dataframe. classes: the
    class list the model was trained with (info["classes"]). with_ci is
    passed to binary_metrics (ignored for other tasks)."""
    X = df[feat_cols]
    y, _ = encode_outcome(df[outcome], task, classes)
    X, y = _drop_missing_outcome(X, y, task)
    if task == "binary":
        scores = predict_scores(pipe, X, task)
        return binary_metrics(y, scores, threshold, with_ci=with_ci)
    if task == "continuous":
        preds = pipe.predict(X)
        return regression_metrics(y, preds)
    preds = pipe.predict(X)
    proba = predict_scores(pipe, X, task)
    return multiclass_metrics(y, preds, proba)


# ============================================================================
# 6. feature importance
# ============================================================================
def _feature_names(pipe, feat_cols):
    try:
        return list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        return list(feat_cols)


def feature_importance(pipe, df_val, outcome, feat_cols, task, classes=None):
    """Return a DataFrame [feature, importance] sorted descending.
    Uses coef_ / feature_importances_ when available, else permutation."""
    clf = pipe.named_steps["clf"]
    names = _feature_names(pipe, feat_cols)
    # the svm sits inside CalibratedClassifierCV(ensemble=False), read the
    # single refit svc (sigmoid calibration keeps its ranking)
    inner = getattr(clf, "calibrated_classifiers_", None)
    if inner is not None and len(inner) == 1:
        clf = inner[0].estimator

    if hasattr(clf, "coef_"):
        coef = np.ravel(clf.coef_)
        if len(coef) == len(names):
            imp = np.abs(coef)
            return (pd.DataFrame({"feature": names, "importance": imp,
                                  "signed": coef})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        if len(fi) == len(names):
            return (pd.DataFrame({"feature": names, "importance": fi})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))

    # fallback: permutation importance on the validation set
    X = df_val[feat_cols]
    y, _ = encode_outcome(df_val[outcome], task, classes)
    X, y = _drop_missing_outcome(X, y, task)
    scoring = "r2" if task == "continuous" else "roc_auc" if task == "binary" else "accuracy"
    r = permutation_importance(pipe, X, y, n_repeats=5,
                               random_state=42, scoring=scoring)
    return (pd.DataFrame({"feature": feat_cols, "importance": r.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def feature_outcome_correlations(df, outcome, feat_cols, task, classes=None):
    """Pearson correlation coefficient between each numeric feature and the
    outcome, on this dataframe. Only meaningful for binary (0/1-encoded, so
    this is really a point-biserial correlation) or continuous outcomes,     multiclass labels have no inherent order, so a Pearson r against them
    isn't statistically meaningful and callers should skip this for
    multiclass. Non-numeric or constant-valued features are skipped, since
    Pearson only applies to a linear numeric relationship.
    Returns a DataFrame [feature, pearson_r] sorted by |pearson_r| descending.
    """
    y, _ = encode_outcome(df[outcome], task, classes)
    y = pd.Series(y, index=df.index)

    rows = []
    for c in feat_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 2 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
            continue
        r = x[mask].corr(y[mask])
        if pd.notna(r):
            rows.append({"feature": c, "pearson_r": round(float(r), 4)})

    if not rows:
        return pd.DataFrame(columns=["feature", "pearson_r"])
    out = pd.DataFrame(rows)
    return (out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


# ============================================================================
# 7. single-row prediction
# ============================================================================
def predict_single(pipe, row_dict, feat_cols, task, classes=None):
    """row_dict: {feature: value}. Returns probability/value + explanation."""
    X = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in feat_cols}])
    if task == "continuous":
        val = float(pipe.predict(X)[0])
        return {"prediction": round(val, 4),
                "explanation": f"Predicted {task} outcome value of {val:.4f}."}
    proba = pipe.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = classes[idx] if classes else idx
    if len(proba) == 2:
        p_pos = float(proba[1])
        pos_label = classes[1] if classes else 1
        return {"prediction": label, "probability": round(p_pos, 4),
                "classes": classes, "proba": [round(float(p), 4) for p in proba],
                "explanation": (f"Predicted outcome '{label}'. Probability of "
                                f"'{pos_label}' = {p_pos:.1%}.")}
    return {"prediction": label, "classes": classes,
            "proba": [round(float(p), 4) for p in proba],
            "explanation": f"Predicted class '{label}' (p = {proba[idx]:.1%})."}


# ============================================================================
# 8. adjusted odds ratios + parameter counts
# ============================================================================
MAX_OR_PARAMETERS = 300     # above this an odds ratio table is not readable anyway


def _sorted_levels(labels):
    """numeric order when every label is a number (ASA 1 < 2 < 10), else
    text order."""
    try:
        return sorted(labels, key=float)
    except ValueError:
        return sorted(labels)


def _design_specs(df, feat_cols, coltypes=None, reference=None):
    """One spec per feature, in feat_cols order. Categorical columns (the
    build_preprocessor rule) get their sorted levels, a reference level
    (the first, unless reference={col: level} says otherwise) and the mode
    used to fill missing values (ties go to the first sorted level, like
    sklearn's SimpleImputer). Numeric columns get the median. Values read as
    1.0 and 1 count as the same level."""
    reference = {c: _norm_label(v) for c, v in (reference or {}).items()}
    specs = []
    for c in feat_cols:
        raw = df[c]
        n_missing = int(raw.isna().sum())
        if _is_categorical(df, c, coltypes):
            lab = pd.Series([None if pd.isna(v) else _norm_label(v) for v in raw],
                            index=raw.index, dtype=object)
            counts = lab.value_counts()
            levels = _sorted_levels(list(counts.index))
            fill = (next(lv for lv in levels if counts[lv] == counts.max())
                    if levels else None)
            ref, note = reference.get(c), None
            if ref is not None and ref not in levels:
                if levels:
                    note = f"reference level {ref!r} not found for {c}, used {levels[0]!r}"
                ref = None
            if ref is None and levels:
                ref = levels[0]
            specs.append({"col": c, "kind": "categorical", "labels": lab,
                          "levels": levels, "reference": ref, "fill": fill,
                          "n_missing": n_missing, "note": note})
        else:
            x = pd.to_numeric(raw, errors="coerce")
            med = x.median() if x.notna().any() else np.nan
            specs.append({"col": c, "kind": "numeric", "values": x,
                          "fill": None if pd.isna(med) else float(med),
                          "n_missing": int(x.isna().sum()), "note": None})
    return specs


def _spec_parameters(spec):
    if spec["kind"] == "numeric":
        return 0 if spec["fill"] is None else 1
    return max(len(spec["levels"]) - 1, 0)


def count_parameters(df, feat_cols, coltypes=None):
    """Number of candidate predictor parameters after one-hot encoding with a
    reference level dropped: 1 per numeric column, (levels - 1) per
    binary/categorical column (same column rule as build_preprocessor).
    Empty columns count 0. Use it for events per parameter and sample size
    checks."""
    return int(sum(_spec_parameters(s) for s in _design_specs(df, feat_cols, coltypes)))


def _independent_columns(X, tol=1e-10):
    """Scan columns left to right and keep each one unless it is constant or
    (numerically) a linear combination of the intercept and the columns
    already kept, the way R's glm reports aliased terms as NA. Works on the
    centred, unit-length Gram matrix with a running Cholesky factor.
    Returns (kept indices, dropped indices)."""
    from scipy.linalg import solve_triangular
    k = X.shape[1]
    Xc = X - X.mean(axis=0)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    constant = (np.ptp(X, axis=0) == 0) | (norms == 0)
    Zs = Xc / np.where(constant, 1.0, norms)
    G = Zs.T @ Zs
    L = np.zeros((k, k))
    keep, dropped = [], []
    for j in range(k):
        if constant[j]:
            dropped.append(j)
            continue
        m = len(keep)
        v = solve_triangular(L[:m, :m], G[keep, j], lower=True) if m else np.zeros(0)
        r = 1.0 - float(v @ v)
        if r <= tol:
            dropped.append(j)
            continue
        L[m, :m] = v
        L[m, m] = np.sqrt(r)
        keep.append(j)
    return keep, dropped


def _sig(x, digits=8):
    x = validation._num(x)
    return None if x is None else float(f"{x:.{digits}g}")


def adjusted_odds_ratios(df_train, outcome, feat_cols, coltypes=None, reference=None):
    """Multivariable logistic regression (statsmodels binomial GLM, no
    penalty) of a binary outcome on feat_cols, fitted on the TRAINING data,
    reported as adjusted odds ratios with 95% Wald intervals and Wald p
    values (the same numbers as R's glm(family = binomial) with
    confint.default).

    Numeric features are not scaled, so each OR is per one unit. Features
    typed binary/categorical in coltypes (or with a non-numeric dtype) are
    one-hot encoded with the first sorted level as reference (override with
    reference={col: level}); terms are named "col=level". Missing values are
    filled with the training median (numeric) or mode (categorical) and
    listed in "imputed". Rows with a missing outcome are dropped. The odds
    are for the second outcome class (outcome_level, as encode_outcome
    codes it).

    Constant or perfectly collinear terms are dropped and named in
    "warning"; so is (quasi-)complete separation, flagged when statsmodels
    detects it or a term's standard error per SD of the predictor exceeds
    10 on the log-odds scale. Nothing here raises on a bad design.

    epp = events per parameter, using the smaller outcome class.

    Returns {"rows": [{"term", "or", "ci_low", "ci_high", "p", "variable",
    "level", "coef", "se"}], "n", "events", "n_parameters", "epp",
    "reference_levels": {col: level}, "imputed": {col: value},
    "n_missing": {col: count}, "outcome_level", "converged": bool,
    "warning": str or None}.
    """
    y, classes = encode_outcome(df_train[outcome], "binary")
    if not classes or len(classes) != 2:
        raise ValueError(f"Adjusted odds ratios need a binary outcome; {outcome} has "
                         f"{len(classes or [])} classes.")
    keep = ~pd.isna(y)
    df = df_train.loc[keep]
    y = np.asarray(y, dtype=float)[keep]
    specs = _design_specs(df, feat_cols, coltypes, reference)

    cols, names, terms = [], [], []
    for s in specs:
        if s["kind"] == "numeric":
            if s["fill"] is None:
                continue
            cols.append(s["values"].fillna(s["fill"]).to_numpy(dtype=float))
            names.append(s["col"])
            terms.append((s["col"], None))
        else:
            lab = s["labels"].where(s["labels"].notna(), s["fill"])
            for lv in s["levels"]:
                if lv == s["reference"]:
                    continue
                cols.append((lab == lv).to_numpy(dtype=float))
                names.append(f"{s['col']}={lv}")
                terms.append((s["col"], lv))

    n, events = int(len(y)), int(y.sum())
    n_par = len(cols)
    notes = [s["note"] for s in specs if s["note"]]
    empty = [s["col"] for s in specs if s["kind"] == "numeric" and s["fill"] is None]
    if empty:
        notes.append("left out because every value is missing: " + ", ".join(empty))
    out = {
        "rows": [], "n": n, "events": events, "n_parameters": n_par,
        "epp": round(min(events, n - events) / n_par, 2) if n_par else None,
        "reference_levels": {s["col"]: s["reference"] for s in specs
                             if s["kind"] == "categorical" and s["reference"] is not None},
        "imputed": {s["col"]: s["fill"] for s in specs
                    if s["n_missing"] and s["fill"] is not None},
        "n_missing": {s["col"]: s["n_missing"] for s in specs if s["n_missing"]},
        "outcome_level": classes[1], "converged": False, "warning": None,
    }

    def _done():
        out["warning"] = "; ".join(notes) or None
        return out

    if events in (0, n):
        notes.insert(0, "only one outcome class in the data, so no model can be fitted")
        return _done()
    if n_par == 0:
        notes.insert(0, "no usable predictors")
        return _done()
    if n_par > MAX_OR_PARAMETERS:
        notes.insert(0, f"{n_par} parameters is too many for an odds ratio table "
                        f"(limit {MAX_OR_PARAMETERS}), group or drop high-cardinality columns")
        return _done()

    X = np.column_stack(cols)
    kept, dropped = _independent_columns(X)
    if dropped:
        notes.append("dropped because they are constant or perfectly collinear with "
                     "other predictors: " + ", ".join(names[j] for j in dropped))
    if not kept:
        notes.insert(0, "no usable predictors")
        return _done()
    Xk = X[:, kept]
    res, info = validation._glm_logit(y, np.column_stack([np.ones(n), Xk]))
    if res is None or info["error"]:
        notes.insert(0, "the model could not be fitted"
                        + (f" ({info['error']})" if info["error"] else ""))
        return _done()
    out["converged"] = bool(info["converged"])

    z = validation.Z95
    sd = Xk.std(axis=0)
    unstable = []
    with np.errstate(over="ignore", invalid="ignore"):
        for i, j in enumerate(kept):
            coef, se = float(res.params[i + 1]), float(res.bse[i + 1])
            if not np.isfinite(se) or se * sd[i] > 10:
                unstable.append(names[j])
            out["rows"].append({
                "term": names[j], "or": _sig(np.exp(coef)),
                "ci_low": _sig(np.exp(coef - z * se)),
                "ci_high": _sig(np.exp(coef + z * se)),
                "p": _sig(res.pvalues[i + 1]),
                "variable": terms[j][0], "level": terms[j][1],
                "coef": _sig(coef), "se": _sig(se),
            })
    if info["separation"] or unstable:
        who = ", ".join(unstable) if unstable else "the predictors"
        notes.insert(0, f"separation: the outcome is (almost) perfectly predicted by {who}, "
                        "so those odds ratios are not reliable (combine sparse categories "
                        "or use a penalized or Firth model)")
    if not out["converged"]:
        notes.append("the fit did not converge")
    return _done()
