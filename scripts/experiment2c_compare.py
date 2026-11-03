"""experiment2c_compare.py
Protocol Section 3: joins the Python-side (ClinTAB-ML) and R-side (reference)
results, reports max absolute deviation per row, and writes
equivalence_table.csv.

Usage:
    python scripts/experiment2c_compare.py --dir /path/to/intermediate/files --out equivalence_table.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="equivalence_table.csv")
    args = ap.parse_args()
    d = Path(args.dir)

    py = json.load(open(d / "python_side_results.json"))
    r = json.load(open(d / "r_side_results.json"))

    rows = []

    # ml.py metrics vs sklearn -- these call the same sklearn functions
    # under the hood, so this checks consistency rather than independent equivalence.
    for metric in ["AUROC", "AUPRC", "Brier"]:
        c = py["ml_metrics"]["clintab"][metric]
        s = py["ml_metrics"]["reference_sklearn"][metric]
        rows.append({"component": "ml.py metrics", "quantity": metric,
                     "clintab": c, "reference": s, "abs_deviation": abs(c - s)})

    # 2x2 panel vs epitools
    for label, key in [("2x2 panel", "twobytwo"), ("2x2 zero-cell (Haldane-Anscombe)", "twobytwo_zero_cell")]:
        pc, rc = py[key], r[key]
        rows.append({"component": label, "quantity": "odds_ratio",
                     "clintab": pc["odds_ratio"], "reference": rc["odds_ratio"],
                     "abs_deviation": abs(pc["odds_ratio"] - rc["odds_ratio"])})
        rows.append({"component": label, "quantity": "OR_CI_lower",
                     "clintab": pc["or_ci"][0], "reference": rc["or_ci"][0],
                     "abs_deviation": abs(pc["or_ci"][0] - rc["or_ci"][0])})
        rows.append({"component": label, "quantity": "OR_CI_upper",
                     "clintab": pc["or_ci"][1], "reference": rc["or_ci"][1],
                     "abs_deviation": abs(pc["or_ci"][1] - rc["or_ci"][1])})
        rows.append({"component": label, "quantity": "risk_ratio",
                     "clintab": pc["risk_ratio"], "reference": rc["risk_ratio"],
                     "abs_deviation": abs(pc["risk_ratio"] - rc["risk_ratio"])})
        rows.append({"component": label, "quantity": "chi2",
                     "clintab": pc["chi2"], "reference": rc["chi2"],
                     "abs_deviation": abs(pc["chi2"] - rc["chi2"])})
        rows.append({"component": label, "quantity": "chi2_p_value",
                     "clintab": pc["p_value"], "reference": rc["chi2_p_value"],
                     "abs_deviation": abs(pc["p_value"] - rc["chi2_p_value"])})

    # Cox vs survival::coxph
    age_row = next(row for row in py["cox"]["rows"] if row["covariate"] == "age")
    dia_row = next(row for row in py["cox"]["rows"] if row["covariate"] == "diabetes")
    rows.append({"component": "Cox PH", "quantity": "HR_age",
                 "clintab": age_row["hazard_ratio"], "reference": r["cox"]["hazard_ratio_age"],
                 "abs_deviation": abs(age_row["hazard_ratio"] - r["cox"]["hazard_ratio_age"])})
    rows.append({"component": "Cox PH", "quantity": "HR_diabetes",
                 "clintab": dia_row["hazard_ratio"], "reference": r["cox"]["hazard_ratio_diabetes"],
                 "abs_deviation": abs(dia_row["hazard_ratio"] - r["cox"]["hazard_ratio_diabetes"])})
    rows.append({"component": "Cox PH", "quantity": "concordance",
                 "clintab": py["cox"]["concordance"], "reference": r["cox"]["concordance"],
                 "abs_deviation": abs(py["cox"]["concordance"] - r["cox"]["concordance"])})

    # Hosmer-Lemeshow vs ResourceSelection::hoslem.test
    rows.append({"component": "Hosmer-Lemeshow", "quantity": "statistic",
                 "clintab": py["hosmer_lemeshow"]["hl_statistic"],
                 "reference": r["hosmer_lemeshow"]["statistic"],
                 "abs_deviation": abs(py["hosmer_lemeshow"]["hl_statistic"] - r["hosmer_lemeshow"]["statistic"])})
    rows.append({"component": "Hosmer-Lemeshow", "quantity": "p_value",
                 "clintab": py["hosmer_lemeshow"]["p_value"],
                 "reference": r["hosmer_lemeshow"]["p_value"],
                 "abs_deviation": abs(py["hosmer_lemeshow"]["p_value"] - r["hosmer_lemeshow"]["p_value"])})

    # RCS vs rms::lrm -- AIC and max deviation across the 100-point grid
    rows.append({"component": "RCS", "quantity": "AIC",
                 "clintab": py["rcs"]["aic"], "reference": r["rcs"]["aic"],
                 "abs_deviation": abs(py["rcs"]["aic"] - r["rcs"]["aic"])})
    py_curve = py["rcs"]["logodds"]
    r_curve = r["rcs"]["logodds_at_grid"]
    max_dev = max(abs(a - b) for a, b in zip(py_curve, r_curve))
    rows.append({"component": "RCS", "quantity": "logodds_curve_max_abs_dev",
                 "clintab": None, "reference": None, "abs_deviation": max_dev})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {args.out}")
    print(f"Max abs deviation across every row: {out['abs_deviation'].max():.6g}")


if __name__ == "__main__":
    main()
