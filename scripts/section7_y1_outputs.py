"""section7_y1_outputs.py
Protocol Section 7 outputs for the primary outcome (Y1), built from the Arm D
session that experiment4_arm_d_session.py created:
  - ROC and precision-recall overlay (three models, held-out test set)
  - calibration curve overlay
  - restricted cubic spline of age against Y1
  - Y1 model comparison table (validation and test metrics)
  - Table 1 (cohort characteristics with per-column missingness)

Reads the session's own splits and the saved models. Writes only figures and
aggregate tables; nothing patient-level is printed or saved.

Usage:
    python scripts/section7_y1_outputs.py --session-id <id> --out-dir /path
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clintab import ml, plots, spline  # noqa: E402  (plots selects the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["LogisticRegression", "RandomForest", "GradientBoosting"]
OUTCOME = "Y1_readmission"
DPI = 600
METRICS = ["AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "Brier"]


def load_models(sid):
    found = {}
    for meta_path in (ROOT / "models").glob("*.meta.json"):
        meta = json.load(open(meta_path))
        if meta.get("session") == sid and meta.get("outcome") == OUTCOME:
            found[meta["model"]] = (meta_path.name[: -len(".meta.json")], meta)
    missing = [m for m in ORDER if m not in found]
    if missing:
        raise SystemExit(f"models not found for this session: {missing}")
    return found


def fig_roc_pr(series, prevalence, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for s in series:
        ax.plot(s["fpr"], s["tpr"], lw=2, label=f"{s['name']} (AUC={s['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax = axes[1]
    for s in series:
        ax.plot(s["recall"], s["precision"], lw=2, label=f"{s['name']} (AP={s['ap']:.3f})")
    ax.axhline(prevalence, color="k", ls="--", lw=1, alpha=0.6,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_calibration(series, out):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
    for s in series:
        ax.plot(s["prob_pred"], s["prob_true"], "o-", lw=2, ms=4, label=s["name"])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Calibration (deciles, held-out test set)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_spline(df, out):
    res = spline.fit_rcs(df, "age", OUTCOME, n_knots=4)
    fig = plots.spline_plot(res["x"], res["logodds"], res["lower"], res["upper"],
                            res["knots"], predictor="Age (years)",
                            title="Restricted cubic spline: age vs 30-day unplanned readmission (4 knots)")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return res["knots"], res["aic"]


def comparison_table(models, test_metrics, out):
    rows = []
    for name in ORDER:
        _, meta = models[name]
        row = {"Model": name}
        for m in METRICS:
            row[f"Val {m}"] = meta["val_metrics"].get(m)
        row["Val N"] = meta["val_metrics"].get("N")
        for m in METRICS:
            row[f"Test {m}"] = test_metrics[name].get(m)
        row["Test N"] = test_metrics[name].get("N")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)


# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------
CONT = [("age", "Age, years"), ("bmi", "BMI"), ("operative_time", "Operative time"),
        ("pre_albumin", "Preoperative albumin"), ("pre_hct", "Preoperative hematocrit"),
        ("pre_creatinine", "Preoperative creatinine"),
        ("Y3_length_of_stay", "Total hospital length of stay, days")]
CAT = [("_procedure", "Procedure"), ("sex", "Sex"), ("race", "Race"),
       ("ethnicity_hispanic", "Hispanic ethnicity"), ("asa_class", "ASA class"),
       ("functional_status", "Functional status"), ("diabetes", "Diabetes"),
       ("hypertension_med", "Hypertension requiring medication"), ("copd", "COPD"),
       ("chf", "Congestive heart failure"), ("current_smoker", "Current smoker"),
       ("dialysis", "Dialysis"), ("anesthesia_type", "Anesthesia type"),
       ("operyr", "Operative year"), (OUTCOME, "30-day unplanned readmission")]
MAX_LEVELS = 8


def _num(x):
    return f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}"


def _level(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def table1(df, out):
    n = len(df)
    rows = [{"Variable": "N", "Level": "", "Overall": f"{n}", "Missing n": "", "Missing %": ""}]
    for col, label in CONT:
        if col not in df:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        miss = int(s.isna().sum())
        v = s.dropna()
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        rows.append({"Variable": label, "Level": "mean (SD)",
                     "Overall": f"{_num(v.mean())} ({_num(v.std())})",
                     "Missing n": miss, "Missing %": f"{100 * miss / n:.1f}"})
        rows.append({"Variable": "", "Level": "median [IQR]",
                     "Overall": f"{_num(med)} [{_num(q1)}-{_num(q3)}]",
                     "Missing n": "", "Missing %": ""})
    for col, label in CAT:
        if col not in df:
            continue
        s = df[col].dropna()
        miss = n - len(s)
        counts = s.map(_level).value_counts()
        shown = counts.head(MAX_LEVELS)
        rest = counts.iloc[MAX_LEVELS:]
        first = True
        for lvl, c in shown.items():
            name = lvl
            if col == OUTCOME:
                name = {"1": "Yes", "0": "No"}.get(lvl, lvl)
            rows.append({"Variable": label if first else "", "Level": name,
                         "Overall": f"{c} ({100 * c / len(s):.1f}%)",
                         "Missing n": miss if first else "",
                         "Missing %": f"{100 * miss / n:.1f}" if first else ""})
            first = False
        if len(rest):
            rows.append({"Variable": "", "Level": f"Other ({len(rest)} categories)",
                         "Overall": f"{int(rest.sum())} ({100 * rest.sum() / len(s):.1f}%)",
                         "Missing n": "", "Missing %": ""})
    pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sess = ROOT / "runtime" / "sessions" / args.session_id

    models = load_models(args.session_id)
    test = pd.read_csv(sess / "test.csv")
    y_test, _ = ml.encode_outcome(test[OUTCOME], "binary")
    prevalence = float(np.nanmean(y_test))

    series, calib, test_metrics = [], [], {}
    for name in ORDER:
        saved, _ = models[name]
        obj = joblib.load(ROOT / "models" / f"{saved}.pkl")
        metrics, coords = ml.evaluate(obj["pipe"], test, OUTCOME, obj["feat_cols"],
                                      "binary", obj["threshold"])
        test_metrics[name] = metrics
        series.append({"name": name, **coords["roc"], **{
            "precision": coords["pr"]["precision"], "recall": coords["pr"]["recall"],
            "ap": coords["pr"]["ap"]}})
        calib.append({"name": name, **coords["calibration"]})

    fig_roc_pr(series, prevalence, out / "figure_roc_pr_overlay.png")
    fig_calibration(calib, out / "figure_calibration_curve.png")
    comparison_table(models, test_metrics, out / "table_y1_model_comparison.csv")

    full = pd.read_csv(sess / "full.csv")
    knots, aic = fig_spline(full, out / "figure_rcs_age_vs_y1.png")
    table1(full, out / "table1_cohort_characteristics.csv")

    print("wrote:")
    for p in sorted(out.iterdir()):
        print("  ", p.name)
    print(f"RCS knots (age): {[round(k, 1) for k in knots]}, AIC {aic}")


if __name__ == "__main__":
    main()
