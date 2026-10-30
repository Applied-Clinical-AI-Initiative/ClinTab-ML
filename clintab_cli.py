#!/usr/bin/env python3
"""clintab_cli.py
Terminal access to the full ClinTAB-ML feature set, built directly on
the same pure modules the web app uses (clintab/ml.py / stats.py / spline.py /
epi.py / plots.py / store.py / validation.py / cohort.py / samplesize.py /
teaching.py). No Flask, this is the scriptable path for batch runs. Unlike
the web app it does not write an analysis log.

    python clintab_cli.py summarize --csv Data/sample_clinical.csv
    python clintab_cli.py models
    python clintab_cli.py cohort --csv Data/sample_clinical.csv --filters filters.json \
        --out cohort.csv --flow flow.json
    python clintab_cli.py train --csv Data/sample_clinical.csv \
        --outcome mortality_30d --exclude surv_time death_event admit_date \
        --models LogisticRegression RandomForest --scoring roc \
        --threshold-method youden --recalibrate
    python clintab_cli.py test  --models <saved_name> ... --csv test.csv [--threshold 0.3]
    python clintab_cli.py spline --csv Data/sample_clinical.csv \
        --predictor age --outcome mortality_30d --knots 4 --out spline.png
    python clintab_cli.py epi or  --a 20 --b 80 --c 10 --d 90
    python clintab_cli.py epi km  --csv Data/sample_clinical.csv --time surv_time --event death_event \
        --group sex --time-points 0 12 24 36
    python clintab_cli.py epi cox --csv Data/sample_clinical.csv --time surv_time --event death_event \
        --cov age asa_class --categorical asa_class --ph-transform rank
    python clintab_cli.py epi cif --csv Data/sample_clinical.csv --time surv_time --event death_event \
        --event-code 1 --group sex --out cif.png
    python clintab_cli.py epi hl  --csv Data/sample_clinical.csv --model <saved_name>
    python clintab_cli.py samplesize --prevalence 0.2 --parameters 10 --cstat 0.75 \
        [--n 600 --events 114]
    python clintab_cli.py teach --csv Data/sample_clinical.csv --outcome mortality_30d \
        --exclude surv_time death_event --model logistic --seeds 1,2,3 --png teach.png
"""
import argparse
import json
import math
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from clintab import store
from clintab import stats
from clintab import ml
from clintab import spline as spline_mod
from clintab import epi
from clintab import plots
from clintab import validation
from clintab import cohort as cohort_mod
from clintab import samplesize
from clintab import teaching

THRESHOLD_METHODS = ("youden", "sensitivity", "specificity", "prevalence", "fixed")


# --------------------------------------------------------------------------
# pretty printing
# --------------------------------------------------------------------------
def _ok(v):
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def _with_ci(row):
    """a metrics row with its nested "CI" dict folded into 'est (lo to hi)'."""
    ci = row.get("CI")
    if not isinstance(ci, dict):
        return row
    out = {k: v for k, v in row.items() if k != "CI"}
    for k, pair in ci.items():
        if (k in out and isinstance(pair, (list, tuple)) and len(pair) == 2
                and _ok(out[k]) and all(_ok(x) for x in pair)):
            out[k] = f"{out[k]} ({pair[0]} to {pair[1]})"
    return out


def table(rows, headers):
    if not rows:
        print("  (no rows)"); return
    rows = [_with_ci(r) for r in rows]
    headers = [h for h in headers if h != "CI"]
    cols = list(zip(*([headers] + [[str(r.get(h, "")) for h in headers] for r in rows])))
    widths = [max(len(c) for c in col) for col in cols]
    line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line); print("  " + "-" * (len(line) - 2))
    for r in rows:
        print("  " + "  ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


def fig_save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved plot -> {path}")


def _plain(o):
    """NaN/inf -> None and numpy -> python, for json files."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def _write_json(obj, path):
    with open(path, "w") as fh:
        json.dump(_plain(obj), fh, indent=2, default=str)
    print(f"  saved json -> {path}")


def _tp(t):
    return f"{t:g}" if isinstance(t, (int, float)) else str(t)


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


def _threshold_settings(a, task):
    """(method, target) for a binary outcome. A bare --threshold keeps its old
    meaning, a fixed threshold."""
    if task != "binary":
        return None, None
    method = a.threshold_method or ("fixed" if a.threshold is not None else "youden")
    target = a.threshold_target
    if method == "fixed" and target is None:
        target = a.threshold if a.threshold is not None else 0.5
    if method in ("fixed", "sensitivity", "specificity") and target is None:
        sys.exit(f"--threshold-method {method} needs --threshold-target.")
    if method == "fixed" and not 0 < target < 1:
        sys.exit("A fixed threshold must be between 0 and 1.")
    if method in ("sensitivity", "specificity") and not 0 < target <= 1:
        sys.exit(f"A {method} target must be above 0 and at most 1.")
    if method not in ("fixed", "sensitivity", "specificity"):
        target = None
    return method, target


def _print_odds_ratios(res):
    print(f"\n  Adjusted odds ratios (training data, n={res['n']}, events={res['events']}, "
          f"parameters={res['n_parameters']}, EPP={res['epp']}, "
          f"odds of {res['outcome']} = {res['outcome_level']}):")
    table(res["rows"], ["term", "or", "ci_low", "ci_high", "p"])
    if res.get("reference_levels"):
        print("  reference levels:", ", ".join(f"{k}={v}" for k, v in res["reference_levels"].items()))
    if res.get("warning"):
        print("  note:", res["warning"])


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
    thr_method, thr_target = _threshold_settings(a, task)

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

    how = f" | threshold={thr_method}" if thr_method else ""
    how += " | recalibrate" if (a.recalibrate and task == "binary") else ""
    print(f"\nTraining {len(models)} model(s) | task={task} | features={len(feat_cols)} "
          f"| split {len(tr)}/{len(va)}/{len(te)} | scoring={a.scoring} | smote={a.smote}{how}\n")

    rows, trained = [], {}
    for name in models:
        print(f"  ... {name}")
        pipe, info = ml.train_one_model(name, tr, va, a.outcome, feat_cols, task,
                                        scoring=a.scoring, use_smote=a.smote,
                                        do_grid_search=not a.no_grid,
                                        coltypes=coltypes, cv_folds=a.cv_folds)
        classes = info.get("classes")
        threshold = a.threshold if a.threshold is not None else 0.5
        recal = None
        if task == "binary":
            # recalibration and the threshold both come from the validation rows
            y_va, _ = ml.encode_outcome(va[a.outcome], task, classes)
            keep = ~pd.isna(y_va)
            X_va = va[feat_cols][keep]
            y_va = np.asarray(y_va, dtype=float)[keep].astype(int)
            try:
                if a.recalibrate:
                    pipe = validation.recalibrate(pipe, X_va, y_va)
                    recal = {"a": pipe.a, "b": pipe.b}
                if thr_method == "fixed":
                    threshold = thr_target
                else:
                    threshold = validation.choose_threshold(
                        y_va, ml.predict_scores(pipe, X_va, task), thr_method, thr_target)["threshold"]
            except ValueError as e:
                print(f"    ! {name}: {e} Skipped.")
                continue
        trained[name] = (pipe, classes, threshold)
        vmetrics, _ = ml.evaluate(pipe, va, a.outcome, feat_cols, task,
                                  threshold=threshold, classes=classes)
        thr_fields = {"threshold": threshold, "threshold_method": thr_method,
                      "threshold_target": thr_target, "recalibrated": recal is not None}
        if recal is not None:
            thr_fields["recalibration"] = recal
        odds = None
        if name == "LogisticRegression" and task == "binary":
            odds = ml.adjusted_odds_ratios(tr, a.outcome, feat_cols, coltypes)
            odds["outcome"] = a.outcome
        ts = store.timestamp()
        base = store.model_base_name(name, a.outcome, ts)
        joblib.dump({"pipe": pipe, "feat_cols": feat_cols, "outcome": a.outcome,
                     "task": task, "classes": classes, **thr_fields},
                    store.model_path(base))
        meta = {"model": name, "outcome": a.outcome, "task": task,
                "scoring": a.scoring, "best_params": info["best_params"],
                "val_metrics": vmetrics, "features": feat_cols, "created": ts, **thr_fields}
        if odds is not None:
            meta["odds_ratios"] = odds
        with open(store.model_meta_path(base), "w") as fh:
            json.dump(_plain(meta), fh, indent=2, default=str)
        imp = ml.feature_importance(pipe, va, a.outcome, feat_cols, task, classes=classes)
        imp.to_csv(os.path.join(store.EXPORTS, f"{base}_importance.csv"), index=False)
        rows.append({"model": name, **vmetrics})
        print(f"    saved -> models/{base}.pkl  (top feature: {imp.iloc[0]['feature']})")
        if task == "binary":
            extra = f", recalibrated a={recal['a']:.3f} b={recal['b']:.3f}" if recal else ""
            print(f"    threshold {threshold:.4g} ({thr_method}, on the validation set){extra}")
        if odds is not None:
            _print_odds_ratios(odds)

    if not rows:
        sys.exit("No model could be trained.")
    print("\nValidation metrics:")
    if task == "binary" and (thr_method != "fixed" or a.recalibrate):
        print("  (the threshold and any recalibration were fitted on the validation set, "
              "so these numbers are optimistic)")
    headers = ["model"] + [k for k in rows[0] if k != "model"]
    table(rows, headers)

    if a.test:
        print("\nHeld-out test-set metrics:")
        trows = []
        for name, (pipe, classes, threshold) in trained.items():
            m, _ = ml.evaluate(pipe, te, a.outcome, feat_cols, task,
                               threshold=threshold, classes=classes)
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
        if mdl["outcome"] is None or mdl["outcome"] not in df.columns:
            print(f"  ! {name}: no outcome column recorded for this model, skipping.")
            continue
        # each model's own saved threshold unless --threshold overrides it
        if a.threshold is not None:
            threshold, source = a.threshold, "user"
        else:
            saved = mdl.get("threshold")
            threshold, source = (0.5 if saved is None else float(saved)), "saved"
        m, c = ml.evaluate(mdl["pipe"], df, mdl["outcome"], mdl["feat_cols"], mdl["task"],
                           threshold, mdl.get("classes"))
        headers = headers or (["model", "threshold_source"] + list(m))
        rows.append({"model": name, "threshold_source": source, **m})
        if mdl["task"] == "binary":
            roc_series.append({"name": name, **c["roc"]})
            pr_series.append({"name": name, **c["pr"]})
    print(f"\nTest metrics on {a.csv}:")
    table(rows, headers or ["model"])
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


def _print_risk_table(rt, title="Number at risk"):
    if not rt or not rt.get("groups"):
        return
    tps = [_tp(t) for t in rt["time_points"]]
    print(f"  {title}:")
    table([{"group": g["name"], **dict(zip(tps, g["at_risk"]))} for g in rt["groups"]],
          ["group"] + tps)


def cmd_epi(a):
    if a.epi_cmd == "or":
        print(json.dumps(_plain(epi.two_by_two(a.a, a.b, a.c, a.d)), indent=2))
        print(epi.nnt_nnh(a.a, a.b, a.c, a.d)["interpretation"])
    elif a.epi_cmd == "km":
        df = pd.read_csv(a.csv)
        res = epi.kaplan_meier(df, a.time, a.event, a.group, time_points=a.time_points)
        for c in res["curves"]:
            print(f"  {c['name']}: N={c['n']}, median survival={c['median_survival']}")
        if res["logrank"]:
            print("  log-rank:", res["logrank"])
        _print_risk_table(res["risk_table"])
        if a.out:
            fig_save(plots.km_plot(res["curves"], risk_table=res["risk_table"]), a.out)
    elif a.epi_cmd == "cox":
        df = pd.read_csv(a.csv)
        res = epi.cox_ph(df, a.time, a.event, a.cov, categorical=a.categorical,
                         ph_transform=a.ph_transform)
        print(f"  N={res['n']}, events={res['n_events']}, concordance={res['concordance']}")
        table(res["rows"], ["covariate", "hazard_ratio", "hr_lower", "hr_upper", "p_value",
                            "ph_p_value"])
        g = res["ph_test"]["global"]
        print(f"  proportional hazards test ({res['ph_test']['transform']}): "
              f"global chi2={g['test_statistic']} (df {g['df']}), p={g['p_value']}")
        for note in res.get("ph_notes") or []:
            print("  note:", note)
    elif a.epi_cmd == "cif":
        df = pd.read_csv(a.csv)
        code = int(a.event_code) if float(a.event_code).is_integer() else a.event_code
        res = epi.cumulative_incidence(df, a.time, a.event, event_of_interest=code,
                                       group_col=a.group, time_points=a.time_points)
        for c in res["curves"]:
            print(f"  {c['name']}: N={c['n']}, events={c['events']}, competing="
                  f"{c['competing_events']}, censored={c['censored']}")
            table([{"time": _tp(t["time"]), "at_risk": t["at_risk"],
                    "cif": t["cif"], "lower": t["lower"], "upper": t["upper"],
                    "1 - KM": t["naive"]} for t in c["at_times"]],
                  ["time", "at_risk", "cif", "lower", "upper", "1 - KM"])
        if res["gray_test"]:
            g = res["gray_test"]
            print(f"  Gray's test: chi2={g['test_statistic']} (df {g['df']}), p={g['p_value']}")
        for note in res.get("notes") or []:
            print("  note:", note)
        if a.out:
            fig_save(plots.cif_plot(res["curves"], naive=res["naive"],
                                    risk_table=res["risk_table"]), a.out)
    elif a.epi_cmd == "hl":
        df = pd.read_csv(a.csv)
        mdl = joblib.load(store.model_path(a.model))
        if not isinstance(mdl, dict) or mdl.get("task") != "binary":
            sys.exit("Hosmer-Lemeshow needs a binary model trained with ClinTAB-ML.")
        y, _ = ml.encode_outcome(df[mdl["outcome"]], "binary", mdl.get("classes"))
        mask = ~pd.isna(y)
        sc = ml.predict_scores(mdl["pipe"], df[mdl["feat_cols"]][mask], "binary")
        res = epi.hosmer_lemeshow(np.asarray(y)[mask], sc)
        print(f"  HL statistic={res['hl_statistic']} (df {res['dof']}), p={res['p_value']}, "
              f"well_calibrated={res['well_calibrated']}")
        table(res["deciles"], ["n", "mean_pred", "observed_rate", "observed", "expected"])


def cmd_samplesize(a):
    if (a.n is None) != (a.events is None):
        sys.exit("Give both --n and --events to compare with a dataset.")
    opts = {"shrinkage": a.shrinkage, "delta": a.delta, "margin": a.margin}
    if a.cstat is not None:
        opts["cstatistic"] = a.cstat
    elif a.csr2 is not None:
        opts["csrsquared"] = a.csr2
    else:
        opts["nagrsquared"] = a.nagr2
    try:
        res = samplesize.binary_sample_size(a.prevalence, a.parameters, **opts)
    except ValueError as e:
        sys.exit(str(e))
    print(f"\nMinimum sample size, binary outcome (Riley et al.), prevalence {a.prevalence:g}, "
          f"{a.parameters} parameters:")
    table(res["criteria"], ["name", "n", "events", "epp", "shrinkage"])
    print(f"\n  minimum n = {res['min_n']}, events = {res['min_events']}, EPP = {res['epp']}")
    for note in res["notes"]:
        print("  note:", note)
    if a.n is not None:
        cmp = samplesize.compare_with_data(res, a.n, a.events, a.parameters)
        print(f"\n  your data: n = {cmp['n']}, events = {cmp['events']}, EPP = {cmp['epp']}, "
              f"adequate = {cmp['adequate']}")
        for note in cmp["notes"]:
            print("  note:", note)


def _seeds(text):
    try:
        seeds = [int(s) for s in str(text).replace(" ", "").split(",") if s]
    except ValueError:
        sys.exit("--seeds must be whole numbers separated by commas, e.g. 1,2,3.")
    if not seeds:
        sys.exit("Give at least one seed.")
    return seeds


def cmd_teach(a):
    df = pd.read_csv(a.csv)
    coltypes = {c["name"]: c["type"] for c in stats.detect_column_types(df)}
    exclude = list(a.exclude or [])
    for c, t in coltypes.items():
        if t == "date" and c not in exclude:
            exclude.append(c)
    feat_cols = ml.feature_columns(df, a.outcome, exclude)
    try:
        res = teaching.leakage_demo(df, a.outcome, feat_cols, coltypes=coltypes,
                                    model=a.model, seeds=_seeds(a.seeds), max_rows=a.max_rows,
                                    progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    except ValueError as e:
        sys.exit(str(e))

    def mean(arm, key):
        return (arm.get(key) or {}).get("mean")

    print(f"\nLeakage demo, {res['model_label']}, outcome {res['outcome']} "
          f"({res['n_used']} patients, {len(res['seeds'])} repeat(s)):")
    table([{"arm": arm["label"], "reported": mean(arm, "reported_auroc"),
            "lockbox": mean(arm, "lockbox_auroc"), "optimism": mean(arm, "optimism"),
            "slope": mean(arm, "calibration_slope")} for arm in res["arms"]],
          ["arm", "reported", "lockbox", "optimism", "slope"])
    for note in res["notes"]:
        print("  note:", note)
    if a.json:
        _write_json(res, a.json)
    if a.png:
        fig_save(plots.teaching_plot(res), a.png)


def _read_filters(path):
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "filters" in data:
        data = data["filters"]
    return data


def cmd_cohort(a):
    df = pd.read_csv(a.csv)
    try:
        filters = _read_filters(a.filters)
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read the filters file: {e}")
    try:
        out, flow = cohort_mod.apply_filters(df, filters)
    except cohort_mod.CohortFilterError as e:
        sys.exit(str(e))
    lines = cohort_mod.summarize_flow(flow, len(df))
    print()
    for line in lines:
        print(f"  {line}")
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"  saved cohort -> {a.out} ({len(out)} rows)")
    if a.flow:
        _write_json({"filters": filters, "flow": flow, "n_start": len(df),
                     "n_final": len(out), "lines": lines}, a.flow)


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="clintab_cli",
                                description="ClinTAB-ML terminal interface.")
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
    t.add_argument("--threshold", type=float, default=None,
                   help="a fixed classification threshold (same as --threshold-method "
                        "fixed --threshold-target X)")
    t.add_argument("--threshold-method", choices=THRESHOLD_METHODS, default=None,
                   help="how the threshold is chosen on the validation set "
                        "(default youden for a binary outcome)")
    t.add_argument("--threshold-target", type=float, default=None,
                   help="target for fixed / sensitivity / specificity")
    t.add_argument("--recalibrate", action="store_true",
                   help="logistic recalibration on the validation set (binary outcome)")
    t.add_argument("--smote", action="store_true")
    t.add_argument("--stratify", action="store_true", help="stratify split on the outcome")
    t.add_argument("--no-grid", action="store_true", help="skip hyperparameter tuning")
    t.add_argument("--cv-folds", type=int, default=None,
                    help="k-fold CV on the training set for grid search, instead of "
                         "the single validation-set split (useful on small datasets)")
    t.add_argument("--test", action="store_true", help="also report held-out test metrics")
    t.set_defaults(func=cmd_train)

    te = sub.add_parser("test", help="evaluate saved model(s) on a CSV")
    te.add_argument("--models", nargs="+", required=True)
    te.add_argument("--csv", required=True)
    te.add_argument("--threshold", type=float, default=None,
                    help="override every model's saved threshold")
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
    ek.add_argument("--time-points", nargs="+", type=float, default=None,
                    help="times for the number at risk table")
    ek.add_argument("--out")
    ec = esub.add_parser("cox", help="Cox proportional hazards")
    ec.add_argument("--csv", required=True); ec.add_argument("--time", required=True)
    ec.add_argument("--event", required=True); ec.add_argument("--cov", nargs="+", required=True)
    ec.add_argument("--categorical", nargs="*", default=[],
                    help="covariates to treat as categorical (e.g. ASA class)")
    ec.add_argument("--ph-transform", choices=("rank", "km", "identity", "log"), default="rank",
                    help="time transform for the proportional hazards test")
    ef = esub.add_parser("cif", help="cumulative incidence with competing risks")
    ef.add_argument("--csv", required=True); ef.add_argument("--time", required=True)
    ef.add_argument("--event", required=True,
                    help="0 = censored, --event-code = the event, other codes = competing")
    ef.add_argument("--event-code", type=float, default=1)
    ef.add_argument("--group", default=None)
    ef.add_argument("--time-points", nargs="+", type=float, default=None)
    ef.add_argument("--out")
    eh = esub.add_parser("hl", help="Hosmer-Lemeshow from a saved model")
    eh.add_argument("--csv", required=True); eh.add_argument("--model", required=True)
    e.set_defaults(func=cmd_epi)

    ss = sub.add_parser("samplesize", help="minimum sample size, binary outcome (Riley et al.)")
    ss.add_argument("--prevalence", type=float, required=True)
    ss.add_argument("--parameters", type=int, required=True,
                    help="candidate predictor parameters (degrees of freedom)")
    r2 = ss.add_mutually_exclusive_group(required=True)
    r2.add_argument("--cstat", type=float, help="anticipated C statistic")
    r2.add_argument("--csr2", type=float, help="anticipated Cox-Snell R2")
    r2.add_argument("--nagr2", type=float, help="anticipated Nagelkerke R2")
    ss.add_argument("--shrinkage", type=float, default=0.9)
    ss.add_argument("--delta", type=float, default=0.05)
    ss.add_argument("--margin", type=float, default=0.05)
    ss.add_argument("--n", type=int, default=None, help="patients in your data")
    ss.add_argument("--events", type=int, default=None, help="events in your data")
    ss.set_defaults(func=cmd_samplesize)

    tc = sub.add_parser("teach", help="leakage demo: reported vs honest lockbox AUROC")
    tc.add_argument("--csv", required=True)
    tc.add_argument("--outcome", required=True)
    tc.add_argument("--exclude", nargs="*", default=[])
    tc.add_argument("--model", choices=list(teaching.MODELS), default="logistic")
    tc.add_argument("--seeds", default="1,2,3", help="comma separated, one repeat each")
    tc.add_argument("--max-rows", type=int, default=20000)
    tc.add_argument("--json", help="path for the full result as JSON")
    tc.add_argument("--png", help="path for the plot")
    tc.set_defaults(func=cmd_teach)

    co = sub.add_parser("cohort", help="apply cohort filters and print the flow")
    co.add_argument("--csv", required=True)
    co.add_argument("--filters", required=True,
                    help="JSON file: a list of filters as cohort.apply_filters takes")
    co.add_argument("--out", help="path for the cohort CSV")
    co.add_argument("--flow", help="path for the flow as JSON")
    co.set_defaults(func=cmd_cohort)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
