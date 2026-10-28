"""section7_y2_y4.py
Protocol Section 7: Y2 (serious morbidity) 2x2 panel and Y4 (30-day death)
Kaplan-Meier plus Cox model, using clintab.epi (the same code the app runs).
Runs locally on the cohort CSV; prints only status lines and saves aggregates.

Usage:
    python scripts/section7_y2_y4.py --csv cohort.csv --out-dir DIR \
        [--exposure diabetes] [--cox-covariates age sex asa_class diabetes ...]

Exposure: a binary column, or a column plus --threshold (for example
asa_class with --threshold 3 means asa_class >= 3 versus below).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clintab import epi  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 600
SUB = [False]
DEFAULT_COX = ["age", "sex", "asa_class", "diabetes", "copd", "chf", "current_smoker",
               "dialysis", "bmi"]
YES = {"YES", "Y", "1", "1.0", "TRUE", "MALE"}


def binarise(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return num
    return s.astype(str).str.strip().str.upper().isin(YES).astype(float).where(s.notna())


def exposure_flag(df, col, threshold):
    if threshold is not None:
        num = pd.to_numeric(df[col], errors="coerce")
        return (num >= threshold).astype(float).where(num.notna())
    return binarise(df[col])


def y2_panel(df, exposure, threshold, out):
    x = exposure_flag(df, exposure, threshold)
    y = pd.to_numeric(df["Y2_serious_morbidity"], errors="coerce")
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    a, b = int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    c, d = int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())
    res = epi.two_by_two(a, b, c, d)
    nnt = epi.nnt_nnh(a, b, c, d)
    pd.DataFrame([{
        "exposure": exposure if threshold is None else f"{exposure}>={threshold}",
        "a_exposed_event": a, "b_exposed_no_event": b, "c_unexposed_event": c,
        "d_unexposed_no_event": d, "odds_ratio": res["odds_ratio"], "or_ci": res["or_ci"],
        "risk_ratio": res["risk_ratio"], "rr_ci": res["rr_ci"],
        "risk_exposed": res["risk_exposed"], "risk_unexposed": res["risk_unexposed"],
        "abs_risk_diff": res["abs_risk_diff"], "nnt_or_nnh": nnt.get("value"),
        "measure": nnt.get("measure"), "chi2": res["chi2"], "p_value": res["p_value"],
    }]).to_csv(out / "table_y2_2x2.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.2]})
    ax1.axis("off")
    cells = [["", "Morbidity", "No morbidity"],
             ["Exposed", f"{a:,}", f"{b:,}"], ["Unexposed", f"{c:,}", f"{d:,}"]]
    t = ax1.table(cellText=cells, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 2.2)
    ax1.set_title("2x2 table", fontsize=11)
    ax2.axis("off")
    lines = [f"Odds ratio {res['odds_ratio']} (95% CI {res['or_ci'][0]} to {res['or_ci'][1]})",
             f"Risk ratio {res['risk_ratio']} (95% CI {res['rr_ci'][0]} to {res['rr_ci'][1]})",
             f"Risk exposed {res['risk_exposed']:.4f}; unexposed {res['risk_unexposed']:.4f}",
             f"Absolute risk difference {res['abs_risk_diff']:.4f}",
             f"{nnt.get('measure', 'NNT/NNH')} {nnt.get('value')}",
             f"Chi-square {res['chi2']}, p = {res['p_value']}"]
    ax2.text(0, 0.5, "\n".join(lines), va="center", fontsize=11)
    ax2.set_title("Effect measures", fontsize=11)
    label = exposure if threshold is None else f"{exposure} >= {threshold}"
    fig.suptitle(f"{'Extended LOS (Y2 substitute)' if SUB[0] else 'Serious morbidity (Y2)'} by {label}", fontsize=13)
    fig.savefig(out / "figure_y2_2x2_panel.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Y2: wrote figure_y2_2x2_panel.png and table_y2_2x2.csv")


def y4_km_cox(df, exposure, threshold, covariates, out):
    df = df.copy()
    df["_grp"] = exposure_flag(df, exposure, threshold).map({0.0: "Unexposed", 1.0: "Exposed"})
    km = epi.kaplan_meier(df, "Y4_time", "Y4_event", "_grp")
    fig, ax = plt.subplots(figsize=(7, 5))
    for cv, col in zip(km["curves"], ["#3b7dd8", "#d8763b"]):
        ax.step(cv["timeline"], cv["survival"], where="post", label=f"{cv['name']} (n={cv['n']:,})",
                color=col, lw=1.8)
        ax.fill_between(cv["timeline"], cv["lower"], cv["upper"], step="post", alpha=0.15, color=col)
    lr = km["logrank"]
    ttl = "Remaining in hospital (Y4 substitute)" if SUB[0] else "Survival to 30 days after surgery"
    if lr:
        ttl += f" (log-rank p = {lr.get('p_value')})"
    ax.set_title(ttl, fontsize=12)
    ax.set_xlabel("Days from operation")
    ax.set_ylabel("Proportion still in hospital" if SUB[0] else "Survival probability")
    ax.set_xlim(0, 30)
    ax.legend(frameon=False)
    ymin = min(min(cv["lower"]) for cv in km["curves"])
    ax.set_ylim(max(0, ymin - 0.001), 1.0005) if not SUB[0] else ax.set_ylim(0, 1.02)
    fig.savefig(out / "figure_y4_km.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    covs = [c for c in covariates if c in df.columns]
    d2 = df.copy()
    for c in covs:
        d2[c] = binarise(d2[c]) if d2[c].dtype == object else d2[c]
    cox = epi.cox_ph(d2, "Y4_time", "Y4_event", covs)
    tbl = pd.DataFrame(cox["rows"])
    tbl["n"], tbl["n_events"], tbl["concordance"] = cox["n"], cox["n_events"], cox["concordance"]
    tbl.to_csv(out / "table_y4_cox.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(tbl) + 1.5))
    yy = range(len(tbl))
    ax.errorbar(tbl.hazard_ratio, list(yy), xerr=[tbl.hazard_ratio - tbl.hr_lower,
                tbl.hr_upper - tbl.hazard_ratio], fmt="o", color="#3b7dd8", capsize=3)
    ax.axvline(1, color="#555555", ls="--", lw=0.8)
    ax.set_yticks(list(yy))
    ax.set_yticklabels(tbl.covariate)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_title(f"Cox model, {'time to discharge (HR<1 = longer stay)' if SUB[0] else '30-day mortality'} (n={cox['n']:,})",
                 fontsize=11)
    ax.invert_yaxis()
    fig.savefig(out / "figure_y4_cox_forest.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Y4: wrote KM figure, Cox forest and table (events={cox['n_events']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exposure", default="diabetes")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--cox-covariates", nargs="*", default=DEFAULT_COX)
    ap.add_argument("--substitute", action="store_true",
                    help="no complication/death columns: use extended LOS (Y2) and time to discharge (Y4)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    if args.exposure not in df.columns:
        raise SystemExit(f"exposure column {args.exposure} not in cohort")
    SUB[0] = args.substitute
    if args.substitute:
        los = pd.to_numeric(df["Y3_length_of_stay"], errors="coerce")
        df["Y2_serious_morbidity"] = (los > los.quantile(0.75)).astype(int).where(los.notna())
        days = pd.to_numeric(df["days_op_to_discharge"], errors="coerce")
        df["Y4_time"] = days.clip(lower=0.5, upper=30)
        df["Y4_event"] = (days <= 30).astype(int).where(days.notna())
        df = df[df["Y4_time"].notna()]
        print("SUBSTITUTE mode: Y2 = LOS above 75th percentile; Y4 = time to discharge")
    if "Y2_serious_morbidity" in df.columns:
        y2_panel(df, args.exposure, args.threshold, out)
    else:
        print("Y2 skipped: cohort has no Y2_serious_morbidity column")
    if {"Y4_time", "Y4_event"} <= set(df.columns):
        y4_km_cox(df, args.exposure, args.threshold, args.cox_covariates, out)
    else:
        print("Y4 skipped: cohort has no Y4_time / Y4_event columns")


if __name__ == "__main__":
    main()
