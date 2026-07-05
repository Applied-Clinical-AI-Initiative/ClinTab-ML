#!/usr/bin/env python3
"""clintab_cli.py
Terminal access to the full ClinTAB-ML-Foundry feature set, built directly on
the same pure modules the web app uses (ml.py / stats.py / spline.py / epi.py /
plots.py / store.py). No Flask -- this is the scriptable path for batch runs.

    python clintab_cli.py summarize --csv Data/sample_clinical.csv
    python clintab_cli.py models
    python clintab_cli.py train --csv Data/sample_clinical.csv \
        --outcome mortality_30d --exclude surv_time death_event admit_date \
        --models LogisticRegression RandomForest --scoring roc --smote
    python clintab_cli.py test  --models <saved_name> ... --csv test.csv
    python clintab_cli.py spline --csv Data/sample_clinical.csv \
        --predictor age --outcome mortality_30d --knots 4 --out spline.png
    python clintab_cli.py epi or  --a 20 --b 80 --c 10 --d 90
    python clintab_cli.py epi km  --csv Data/sample_clinical.csv --time surv_time --event death_event --group sex
    python clintab_cli.py epi cox --csv Data/sample_clinical.csv --time surv_time --event death_event --cov age diabetes
    python clintab_cli.py epi hl  --csv Data/sample_clinical.csv --model <saved_name>
"""
import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import store
import stats
import ml
import spline as spline_mod
import epi
import plots


# --------------------------------------------------------------------------
# pretty printing
# --------------------------------------------------------------------------
def table(rows, headers):
    if not rows:
        print("  (no rows)"); return
    cols = list(zip(*([headers] + [[str(r.get(h, "")) for h in headers] for r in rows])))
    widths = [max(len(c) for c in col) for col in cols]
    line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line); print("  " + "-" * (len(line) - 2))
    for r in rows:
        print("  " + "  ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


def fig_save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved plot -> {path}")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_summarize(a):
    df = pd.read_csv(a.csv)
    cols = stats.detect_column_types(df)
    print(f"\n{a.csv}: {len(df)} rows x {df.shape[1]} cols\n")
    print("Detected column types:")
    table(cols, ["name", "type", "n_unique", "pct_missing", "high_missing"])
    ct = {c["name"]: c["type"] for c in cols}
    s = stats.summarize(df, ct)
    print("\nCard:", s["card"])
    print("\nContinuous variables:")
    table(s["continuous"], ["variable", "n", "pct_missing", "mean", "median", "sd", "iqr", "min", "max"])
    print("\nBinary / categorical variables:")
    for v in s["categorical"]:
        print(f"  {v['variable']} ({v['type']}, N={v['n']}, missing {v['pct_missing']}%)")
        table(v["categories"], ["category", "count", "pct"])


def cmd_models(a):
    print("\nClassifiers (binary/categorical outcomes):")
    for m in ml.model_catalogue("binary"): print("  -", m)
    print("\nRegressors (continuous outcomes):")
    for m in ml.model_catalogue("continuous"): print("  -", m)


def _split(df, ratios, seed, stratify=None):
    r = [x / sum(ratios) for x in ratios]
    strat = df[stratify] if stratify else None
    tr, tmp = train_test_split(df, test_size=(r[1] + r[2]), random_state=seed, stratify=strat)
    strat2 = tmp[stratify] if stratify else None
    va, te = train_test_split(tmp, test_size=(r[2] / (r[1] + r[2])), random_state=seed, stratify=strat2)
    return tr, va, te


def cmd_train(a):
    store.ensure_dirs()
    df = pd.read_csv(a.csv)
    cols = stats.detect_column_types(df)
    coltypes = {c["name"]: c["type"] for c in cols}
    task = ml.determine_task(df[a.outcome])
    exclude = list(a.exclude or [])
    for c, t in coltypes.items():
        if t == "date" and c not in exclude:
            exclude.append(c)

    catalogue = ml.model_catalogue(task)
    models = a.models or list(catalogue)
    bad = [m for m in models if m not in catalogue]
    if bad:
        sys.exit(f"Unknown model(s) for task '{task}': {bad}\nAvailable: {list(catalogue)}")

    strat = a.outcome if (a.stratify and task != "continuous") else None
    try:
        tr, va, te = _split(df, a.split, a.seed, strat)
    except ValueError:
        tr, va, te = _split(df, a.split, a.seed, None)
        print("  (stratified split fell back to random)")
    feat_cols = ml.feature_columns(tr, a.outcome, exclude)

    print(f"\nTraining {len(models)} model(s) | task={task} | features={len(feat_cols)} "
          f"| split {len(tr)}/{len(va)}/{len(te)} | scoring={a.scoring} | smote={a.smote}\n")

    rows = []
    for name in models:
        print(f"  … {name}")
        pipe, info = ml.train_one_model(name, tr, va, a.outcome, feat_cols, task,
                                        scoring=a.scoring, use_smote=a.smote,
                                        do_grid_search=not a.no_grid)
        vmetrics, _ = ml.evaluate(pipe, va, a.outcome, feat_cols, task, threshold=a.threshold)
        ts = store.timestamp()
        base = f"{name}_{a.outcome}_{ts}".replace(" ", "_")
        joblib.dump({"pipe": pipe, "feat_cols": feat_cols, "outcome": a.outcome,
                     "task": task, "classes": info.get("classes"), "threshold": a.threshold},
                    store.model_path(base))
        with open(store.model_meta_path(base), "w") as fh:
            json.dump({"model": name, "outcome": a.outcome, "task": task,
                       "scoring": a.scoring, "best_params": info["best_params"],
                       "val_metrics": vmetrics, "features": feat_cols, "created": ts},
                      fh, indent=2, default=str)
        imp = ml.feature_importance(pipe, va, a.outcome, feat_cols, task)
        imp.to_csv(os.path.join(store.EXPORTS, f"{base}_importance.csv"), index=False)
        rows.append({"model": name, **vmetrics})
        print(f"    saved -> models/{base}.pkl  (top feature: {imp.iloc[0]['feature']})")

    print("\nValidation metrics:")
    headers = ["model"] + [k for k in rows[0] if k != "model"]
    table(rows, headers)

    if a.test:
        print("\nHeld-out test-set metrics:")
        trows = []
        for name in models:
            mp = [p for p in os.listdir(store.MODELS) if p.startswith(name + "_" + a.outcome) and p.endswith(".pkl")]
            mp.sort()
            mdl = joblib.load(store.model_path(mp[-1][:-4]))
            m, _ = ml.evaluate(mdl["pipe"], te, a.outcome, feat_cols, task, threshold=a.threshold)
            trows.append({"model": name, **m})
        table(trows, headers)


def cmd_test(a):
    df = pd.read_csv(a.csv)
    rows, roc_series, pr_series = [], [], []
    headers = None
    for name in a.models:
        mdl = joblib.load(store.model_path(name))
        if not isinstance(mdl, dict):
            mdl = {"pipe": mdl, "feat_cols": [c for c in df.columns], "outcome": None, "task": "binary"}
        m, c = ml.evaluate(mdl["pipe"], df, mdl["outcome"], mdl["feat_cols"], mdl["task"], a.threshold)
        headers = headers or (["model"] + list(m))
        rows.append({"model": name, **m})
        if mdl["task"] == "binary":
            roc_series.append({"name": name, **c["roc"]})
            pr_series.append({"name": name, **c["pr"]})
    print(f"\nTest metrics on {a.csv}:")
    table(rows, headers)
    if roc_series and a.out:
        fig_save(plots.roc_overlay(roc_series), a.out)
        fig_save(plots.pr_overlay(pr_series), a.out.replace(".png", "_pr.png"))


def cmd_spline(a):
    df = pd.read_csv(a.csv)
    r = spline_mod.fit_rcs(df, a.predictor, a.outcome, n_knots=a.knots)
    print(f"\nRCS {a.outcome} ~ {a.predictor}: N={r['n']}, knots={r['knots']}, AIC={r['aic']}")
    if a.out:
        fig_save(plots.spline_plot(r["x"], r["logodds"], r["lower"], r["upper"],
                                   r["knots"], predictor=a.predictor), a.out)


def cmd_epi(a):
    if a.epi_cmd == "or":
        print(json.dumps(epi.two_by_two(a.a, a.b, a.c, a.d), indent=2))
        print(epi.nnt_nnh(a.a, a.b, a.c, a.d)["interpretation"])
    elif a.epi_cmd == "km":
        df = pd.read_csv(a.csv)
        res = epi.kaplan_meier(df, a.time, a.event, a.group)
        for c in res["curves"]:
            print(f"  {c['name']}: N={c['n']}, median survival={c['median_survival']}")
        if res["logrank"]:
            print("  log-rank:", res["logrank"])
        if a.out:
            fig_save(plots.km_plot(res["curves"]), a.out)
    elif a.epi_cmd == "cox":
        df = pd.read_csv(a.csv)
        res = epi.cox_ph(df, a.time, a.event, a.cov)
        print(f"  N={res['n']}, events={res['n_events']}, concordance={res['concordance']}")
        table(res["rows"], ["covariate", "hazard_ratio", "hr_lower", "hr_upper", "p_value"])
    elif a.epi_cmd == "hl":
        df = pd.read_csv(a.csv)
        mdl = joblib.load(store.model_path(a.model))
        y, _ = ml.encode_outcome(df[mdl["outcome"]], "binary")
        mask = ~pd.isna(y)
        sc = ml.predict_scores(mdl["pipe"], df[mdl["feat_cols"]][mask], "binary")
        res = epi.hosmer_lemeshow(np.asarray(y)[mask], sc)
        print(f"  HL statistic={res['hl_statistic']} (df {res['dof']}), p={res['p_value']}, "
              f"well_calibrated={res['well_calibrated']}")
        table(res["deciles"], ["n", "mean_pred", "observed_rate", "observed", "expected"])


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="clintab_cli",
                                description="ClinTAB-ML-Foundry terminal interface.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="profile + descriptive stats for a CSV")
    s.add_argument("--csv", required=True); s.set_defaults(func=cmd_summarize)

    sub.add_parser("models", help="list the 15 available models").set_defaults(func=cmd_models)

    t = sub.add_parser("train", help="split, train, tune, save model(s)")
    t.add_argument("--csv", required=True)
    t.add_argument("--outcome", required=True)
    t.add_argument("--models", nargs="*", help="default: all for the task")
    t.add_argument("--exclude", nargs="*", default=[])
    t.add_argument("--scoring", default="roc")
    t.add_argument("--split", nargs=3, type=float, default=[70, 15, 15])
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--threshold", type=float, default=0.5)
    t.add_argument("--smote", action="store_true")
    t.add_argument("--stratify", action="store_true", help="stratify split on the outcome")
    t.add_argument("--no-grid", action="store_true", help="skip hyperparameter tuning")
    t.add_argument("--test", action="store_true", help="also report held-out test metrics")
    t.set_defaults(func=cmd_train)

    te = sub.add_parser("test", help="evaluate saved model(s) on a CSV")
    te.add_argument("--models", nargs="+", required=True)
    te.add_argument("--csv", required=True)
    te.add_argument("--threshold", type=float, default=0.5)
    te.add_argument("--out", help="path for ROC/PR overlay PNG")
    te.set_defaults(func=cmd_test)

    sp = sub.add_parser("spline", help="fit a restricted cubic spline")
    sp.add_argument("--csv", required=True); sp.add_argument("--predictor", required=True)
    sp.add_argument("--outcome", required=True); sp.add_argument("--knots", type=int, default=4)
    sp.add_argument("--out", help="path for PNG"); sp.set_defaults(func=cmd_spline)

    e = sub.add_parser("epi", help="clinical epidemiology calculators")
    esub = e.add_subparsers(dest="epi_cmd", required=True)
    eo = esub.add_parser("or", help="2x2 OR/RR/NNT")
    for x in "abcd": eo.add_argument(f"--{x}", type=float, required=True)
    ek = esub.add_parser("km", help="Kaplan-Meier")
    ek.add_argument("--csv", required=True); ek.add_argument("--time", required=True)
    ek.add_argument("--event", required=True); ek.add_argument("--group", default=None)
    ek.add_argument("--out")
    ec = esub.add_parser("cox", help="Cox proportional hazards")
    ec.add_argument("--csv", required=True); ec.add_argument("--time", required=True)
    ec.add_argument("--event", required=True); ec.add_argument("--cov", nargs="+", required=True)
    eh = esub.add_parser("hl", help="Hosmer-Lemeshow from a saved model")
    eh.add_argument("--csv", required=True); eh.add_argument("--model", required=True)
    e.set_defaults(func=cmd_epi)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
