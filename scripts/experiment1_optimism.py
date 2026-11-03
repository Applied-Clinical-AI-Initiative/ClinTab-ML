"""experiment1_optimism.py
Protocol Section 2: compares four training protocols across four cohort sizes,
three models, and 20 seeds. Arm D uses ClinTAB-ML training components. For
each fit, records the reported metric and the same fitted model's metric on
a separate held-out partition.

Run entirely on your own machine against your own real_cohort.csv (output of
build_cohort.py). Only optimism_results.csv -- aggregate metric values,
no patient-level rows -- is meant to leave your machine.

Usage:
    python scripts/experiment1_optimism.py \
        --csv /path/outside/repo/real_cohort.csv \
        --outcome Y1_readmission \
        --features age bmi sex asa_class diabetes hypertension_med copd chf \
                   current_smoker dialysis pre_albumin pre_hct pre_creatinine \
                   operative_time anesthesia_type \
        --out optimism_results.csv
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import statsmodels.api as sm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from clintab import ml as clintab_ml  # noqa: E402

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except Exception:  # pragma: no cover
    HAS_SMOTE = False

SIZES = [1_000, 5_000, 25_000, 100_000]
SEEDS_PER_CELL = 20
MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting"]
GRID = clintab_ml.model_catalogue("binary")


def hosmer_lemeshow_p(y_true, y_prob, g=10):
    from scipy import stats
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob)
    y_true, y_prob = y_true[order], y_prob[order]
    groups = np.array_split(np.arange(len(y_prob)), g)
    hl = 0.0
    for grp in groups:
        if len(grp) == 0:
            continue
        obs, exp, n = y_true[grp].sum(), y_prob[grp].sum(), len(grp)
        if 0 < exp < n:
            hl += (obs - exp) ** 2 / (exp * (1 - exp / n))
    dof = max(g - 2, 1)
    return float(1 - stats.chi2.cdf(hl, dof))


def calibration_slope(y_true, y_prob):
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p)
    try:
        res = sm.GLM(y_true, X, family=sm.families.Binomial()).fit()
        return float(res.params[1])
    except Exception:
        return float("nan")


def metrics_dict(y_true, y_prob):
    y_true = np.asarray(y_true)
    return {
        "AUROC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "calibration_slope": calibration_slope(y_true, y_prob),
        "HL_pvalue": hosmer_lemeshow_p(y_true, y_prob),
    }


def make_pipeline(model_name, feat_cols, df, use_smote):
    estimator = GRID[model_name]["model"]
    pre, _, _ = clintab_ml.build_preprocessor(df, feat_cols)
    steps = [("pre", pre)]
    if use_smote and HAS_SMOTE:
        steps.append(("smote", SMOTE(random_state=42)))
        return ImbPipeline(steps + [("clf", estimator)])
    return Pipeline(steps + [("clf", estimator)])


def _tune_on_val_report_on_test(pipe, param_grid, Xtr, ytr, Xva, yva, Xte, yte, stack):
    """Shared mechanics for A, B, D: grid search scored on val only
    (val is NEVER the partition reported on), refit on train, report on a
    THIRD partition (test) that tuning never touched. `stack` is np.vstack
    for array inputs (arrays after preprocessing) or pd.concat for
    DataFrame inputs."""
    X_all2 = stack([Xtr, Xva])
    y_all2 = np.concatenate([ytr, yva])
    test_fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xva))]
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=PredefinedSplit(test_fold),
                       n_jobs=-1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_all2, y_all2)
    pipe.set_params(**gs.best_params_)
    fitted = pipe.fit(Xtr, ytr)
    reported = metrics_dict(yte, fitted.predict_proba(Xte)[:, 1])
    return fitted, reported


def run_arm(arm, model_name, df, feat_cols, outcome, seed):
    """Returns (reported_metrics, honest_metrics) dicts for one arm/model/seed."""
    y_all, _ = clintab_ml.encode_outcome(df[outcome], "binary")
    X_all = df[feat_cols]
    mask = ~pd.isna(y_all)
    X_all, y_all = X_all[mask], y_all[mask]

    # Reserve the held-out partition before arm-specific processing.
    X_work, X_pristine, y_work, y_pristine = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=seed)

    param_grid = GRID[model_name]["params"]

    if arm in ("A", "B", "D"):
        # Arms A, B, and D tune on validation and report on the test partition.
        Xtr, Xrest, ytr, yrest = train_test_split(
            X_work, y_work, test_size=0.30, stratify=y_work, random_state=seed)
        Xva, Xte, yva, yte = train_test_split(
            Xrest, yrest, test_size=0.5, stratify=yrest, random_state=seed)

        if arm == "A":
            # Leak: preprocessing fit on the FULL working set before the
            # three-way split even happens.
            pre, _, _ = clintab_ml.build_preprocessor(X_work.assign(**{outcome: 0}), feat_cols)
            pre.fit(X_work)
            Xtr_t, Xva_t, Xte_t = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)
            X_pristine_t = pre.transform(X_pristine)
            pipe = Pipeline([("clf", GRID[model_name]["model"])])
            fitted, reported = _tune_on_val_report_on_test(
                pipe, param_grid, Xtr_t, ytr, Xva_t, yva, Xte_t, yte, np.vstack)
            honest = metrics_dict(y_pristine, fitted.predict_proba(X_pristine_t)[:, 1])
            return reported, honest

        if arm == "B":
            # Leak: SMOTE applied to the FULL working set before the
            # three-way split even happens.
            pre, _, _ = clintab_ml.build_preprocessor(X_work.assign(**{outcome: 0}), feat_cols)
            X_work_t = pre.fit_transform(X_work)
            X_pristine_t = pre.transform(X_pristine)
            if HAS_SMOTE:
                X_res, y_res = SMOTE(random_state=seed).fit_resample(X_work_t, y_work)
            else:
                X_res, y_res = X_work_t, y_work
            Xtr, Xrest, ytr, yrest = train_test_split(
                X_res, y_res, test_size=0.30, stratify=y_res, random_state=seed)
            Xva, Xte, yva, yte = train_test_split(
                Xrest, yrest, test_size=0.5, stratify=yrest, random_state=seed)
            pipe = Pipeline([("clf", GRID[model_name]["model"])])
            fitted, reported = _tune_on_val_report_on_test(
                pipe, param_grid, Xtr, ytr, Xva, yva, Xte, yte, np.vstack)
            honest = metrics_dict(y_pristine, fitted.predict_proba(X_pristine_t)[:, 1])
            return reported, honest

        # Arm D keeps preprocessing and SMOTE inside the training pipeline.
        pipe = make_pipeline(model_name, feat_cols, df, use_smote=True)
        fitted, reported = _tune_on_val_report_on_test(
            pipe, param_grid, Xtr, ytr, Xva, yva, Xte, yte, pd.concat)
        honest = metrics_dict(y_pristine, fitted.predict_proba(X_pristine)[:, 1])
        return reported, honest

    if arm == "C":
        # Arm C scores hyperparameters and reports performance on the same partition.
        Xtr, Xte, ytr, yte = train_test_split(X_work, y_work, test_size=0.1765,
                                              stratify=y_work, random_state=seed)
        pipe = make_pipeline(model_name, feat_cols, df, use_smote=False)
        X_all2 = pd.concat([Xtr, Xte], axis=0)
        y_all2 = np.concatenate([ytr, yte])
        test_fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xte))]
        gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=PredefinedSplit(test_fold), n_jobs=-1, refit=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_all2, y_all2)
        pipe.set_params(**gs.best_params_)
        fitted = pipe.fit(Xtr, ytr)
        reported = metrics_dict(yte, fitted.predict_proba(Xte)[:, 1])
        honest = metrics_dict(y_pristine, fitted.predict_proba(X_pristine)[:, 1])
        return reported, honest

    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outcome", default="Y1_readmission")
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", default="optimism_results.csv")
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--seeds", type=int, default=SEEDS_PER_CELL)
    args = ap.parse_args()

    df_full = pd.read_csv(args.csv)
    rows = []
    total_cells = len(args.sizes) * len(MODELS) * args.seeds * 4
    done = 0

    for n in args.sizes:
        if n > len(df_full):
            print(f"skip n={n}: cohort only has {len(df_full)} rows", file=sys.stderr)
            continue
        for seed in range(args.seeds):
            sub, _ = train_test_split(df_full, train_size=n, stratify=df_full[args.outcome],
                                      random_state=seed)
            for model_name in MODELS:
                for arm in ["A", "B", "C", "D"]:
                    try:
                        reported, honest = run_arm(arm, model_name, sub, args.features, args.outcome, seed)
                    except Exception as e:
                        print(f"FAILED arm={arm} n={n} seed={seed} model={model_name}: {e}", file=sys.stderr)
                        done += 1
                        continue
                    for metric in reported:
                        rows.append({
                            "arm": arm, "n": n, "seed": seed, "model": model_name,
                            "metric": metric,
                            "reported": reported[metric], "honest": honest[metric],
                            "optimism": reported[metric] - honest[metric],
                        })
                    done += 1
                    if done % 20 == 0:
                        print(f"  {done}/{total_cells} cells done", file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows to {args.out} (aggregate metrics only -- safe to share)")


if __name__ == "__main__":
    main()
