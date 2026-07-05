"""ml.py
All the sklearn: preprocessing, the 15 models, validation-set grid search,
optional SMOTE on the training fold only, metrics, feature importance, and
single-row prediction. Knows nothing about Flask -- everything takes/returns
plain DataFrames, dicts and numpy arrays so it can be tested in isolation.
"""
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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
    recall_score, precision_score, accuracy_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.calibration import calibration_curve

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except Exception:                                   # pragma: no cover
    HAS_SMOTE = False


# ============================================================================
# 1. Task detection + model catalogue
# ============================================================================
def determine_task(series):
    """binary | multiclass | continuous, from the outcome column."""
    s = series.dropna()
    n_unique = s.nunique()
    if n_unique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s) and n_unique > 10:
        return "continuous"
    if n_unique <= 10:
        return "multiclass"
    return "continuous"


def model_catalogue(task):
    """The 15 models + sensible default grids, filtered by task."""
    if task in ("binary", "multiclass"):
        return {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=2000, solver="lbfgs"),
                "params": {"clf__C": [0.01, 0.1, 1, 10]},
            },
            "RidgeLogistic": {
                "model": LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs"),
                "params": {"clf__C": [0.01, 0.1, 1, 10]},
            },
            "LassoLogistic": {
                "model": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000),
                "params": {"clf__C": [0.01, 0.1, 1, 10]},
            },
            "ElasticNetLogistic": {
                "model": LogisticRegression(penalty="elasticnet", solver="saga",
                                            l1_ratio=0.5, max_iter=3000),
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
                "model": SVC(probability=True, random_state=42),
                "params": {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
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
# 2. Preprocessing
# ============================================================================
def feature_columns(df, outcome, exclude):
    exclude = set(exclude or [])
    return [c for c in df.columns if c != outcome and c not in exclude]


def build_preprocessor(df, feat_cols, scale=True):
    """ColumnTransformer: impute+scale numerics, impute+one-hot categoricals.
    object/category dtype -> categorical; otherwise numeric.
    """
    num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feat_cols if c not in num]

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


def encode_outcome(series, task):
    """Returns (y, classes) for classification or (y, None) for regression."""
    if task == "continuous":
        return pd.to_numeric(series, errors="coerce").values, None
    le = LabelEncoder()
    y = le.fit_transform(series.astype(str))
    return y, list(le.classes_)


# ============================================================================
# 3. Scorers
# ============================================================================
def make_scoring(metric, task):
    if task == "continuous":
        return {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }.get(metric, "r2")
    if metric == "f2":
        return make_scorer(fbeta_score, beta=2, zero_division=0)
    if metric == "f1":
        return make_scorer(f1_score, zero_division=0)
    return {
        "roc": "roc_auc", "roc_auc": "roc_auc",
        "auprc": "average_precision", "average_precision": "average_precision",
        "recall": "recall", "sensitivity": "recall",
        "precision": "precision", "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }.get(metric, "roc_auc")


# ============================================================================
# 4. Train one model
# ============================================================================
def train_one_model(name, df_train, df_val, outcome, feat_cols, task,
                    param_grid=None, scoring="roc", use_smote=False,
                    do_grid_search=True):
    """Fit one model. Grid search is evaluated on the VALIDATION set only
    (via PredefinedSplit), then the best params are refit on train+val.
    Returns (fitted_pipeline, info_dict).
    """
    cfg = model_catalogue(task)[name]
    estimator = cfg["model"]
    grid = param_grid if param_grid is not None else cfg["params"]

    pre, num, cat = build_preprocessor(df_train, feat_cols)

    steps = [("pre", pre)]
    if use_smote and task in ("binary", "multiclass") and HAS_SMOTE:
        steps.append(("smote", SMOTE(random_state=42)))
        pipe = ImbPipeline(steps + [("clf", estimator)])
    else:
        pipe = Pipeline(steps + [("clf", estimator)])

    Xtr, Xva = df_train[feat_cols], df_val[feat_cols]
    ytr, classes = encode_outcome(df_train[outcome], task)
    yva, _ = encode_outcome(df_val[outcome], task)

    # drop rows with missing outcome
    mtr, mva = ~pd.isna(ytr), ~pd.isna(yva)
    Xtr, ytr = Xtr[mtr], ytr[mtr]
    Xva, yva = Xva[mva], yva[mva]

    score = make_scoring(scoring, task)
    best_params = "default"

    if do_grid_search and grid:
        # Grid search scores each candidate on the VALIDATION set only
        # (PredefinedSplit: fold -1 = train, fold 0 = val). refit=False so the
        # winning params are then re-fit on the TRAINING set alone -- this keeps
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
            "smote": bool(use_smote and task != "continuous" and HAS_SMOTE)}
    return fitted, info


def _jsonable(p):
    if isinstance(p, dict):
        return {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                for k, v in p.items()}
    return str(p)


# ============================================================================
# 5. Metrics + plot coordinates
# ============================================================================
def predict_scores(pipe, X, task):
    """Probability of positive class (binary) / predicted value (regression)."""
    if task == "continuous":
        return pipe.predict(X)
    proba = pipe.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba  # multiclass


def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
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
    metrics = {
        "AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
        "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
        "PPV": round(ppv, 4), "NPV": round(npv, 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "Threshold": threshold,
        "N": int(len(y_true)),
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


def multiclass_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "BalancedAccuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "MacroF1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "N": int(len(y_true)),
    }, {}


def evaluate(pipe, df, outcome, feat_cols, task, threshold=0.5):
    """Metrics + plot coordinates on an arbitrary dataframe."""
    X = df[feat_cols]
    y, _ = encode_outcome(df[outcome], task)
    mask = ~pd.isna(y)
    X, y = X[mask], y[mask]
    if task == "binary":
        scores = predict_scores(pipe, X, task)
        return binary_metrics(y, scores, threshold)
    if task == "continuous":
        preds = pipe.predict(X)
        return regression_metrics(y, preds)
    preds = pipe.predict(X)
    return multiclass_metrics(y, preds)


# ============================================================================
# 6. Feature importance
# ============================================================================
def _feature_names(pipe, feat_cols):
    try:
        return list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        return list(feat_cols)


def feature_importance(pipe, df_val, outcome, feat_cols, task):
    """Return a DataFrame [feature, importance] sorted descending.
    Uses coef_ / feature_importances_ when available, else permutation."""
    clf = pipe.named_steps["clf"]
    names = _feature_names(pipe, feat_cols)

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
    y, _ = encode_outcome(df_val[outcome], task)
    mask = ~pd.isna(y)
    scoring = "r2" if task == "continuous" else "roc_auc" if task == "binary" else "accuracy"
    r = permutation_importance(pipe, X[mask], y[mask], n_repeats=5,
                               random_state=42, scoring=scoring)
    return (pd.DataFrame({"feature": feat_cols, "importance": r.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


# ============================================================================
# 7. Single-row prediction
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
