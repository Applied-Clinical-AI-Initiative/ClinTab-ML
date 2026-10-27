"""Build manuscript Tables 1 to 4 and the supplementary tables as CSV.

Run: python paper/analysis/tables.py
"""
import json
import re

import pandas as pd

from common import (ARMS, F_EQUIV, F_EX4_CLAUSES, F_EX4_RECOVER, F_METHODS,
                    F_TABLE1, F_TRIPOD, F_Y1, MODEL_LABEL, MODELS, PLACEHOLDER,
                    TABLES, ensure_dirs, has_content, optimism_summary)

# protocol section 3: which reference implementation each component was
# checked against.
REFERENCE_IMPL = {
    "ml.py metrics": "scikit-learn, called directly",
    "2x2 panel": "R epitools",
    "2x2 zero-cell (Haldane-Anscombe)": "R epitools",
    "Cox PH": "R survival::coxph",
    "Hosmer-Lemeshow": "R ResourceSelection::hoslem.test",
    "RCS": "R rms::lrm at matched knots",
}
COMPONENT_LABEL = {
    "ml.py metrics": "Discrimination and accuracy metrics",
    "2x2 panel": "2x2 table",
    "2x2 zero-cell (Haldane-Anscombe)": "2x2 table with zero cell",
    "Cox PH": "Cox proportional hazards",
    "Hosmer-Lemeshow": "Hosmer-Lemeshow test",
    "RCS": "Restricted cubic spline",
}
QUANTITY_LABEL = {
    "odds_ratio": "Odds ratio", "OR_CI_lower": "OR 95% CI, lower",
    "OR_CI_upper": "OR 95% CI, upper", "risk_ratio": "Risk ratio",
    "chi2": "Chi-square", "chi2_p_value": "Chi-square P value",
    "HR_age": "HR, age", "HR_diabetes": "HR, diabetes",
    "concordance": "Concordance", "statistic": "Statistic",
    "p_value": "P value", "AIC": "AIC",
    "logodds_curve_max_abs_dev": "Log-odds curve (100-point grid)",
}
# clintab-ml reports to 4 decimal places, so agreement to half a unit in the
# fourth decimal is the most that can be asked of it.
TOLERANCE = 5e-5


def fmt_ci(m, lo, hi, dp=3):
    return f"{m:+.{dp}f} ({lo:+.{dp}f} to {hi:+.{dp}f})".replace("-", "\u2212")


def table1():
    t = pd.read_csv(F_TABLE1, dtype=str).fillna("")
    t = t.rename(columns={"Overall": "Overall, n (%) or summary",
                          "Missing %": "Missing, %", "Missing n": "Missing, n"})
    # carry the variable name down so each level row is self-describing.
    t["Variable"] = t["Variable"].replace("", pd.NA).ffill()
    t.to_csv(TABLES / "table1_cohort.csv", index=False)
    return t


def table2():
    pooled = optimism_summary("AUROC", pool_models=True)
    per = optimism_summary("AUROC", pool_models=False)
    sizes = sorted(pooled.n.unique())
    rows = []
    for arm, label in ARMS.items():
        r = {"Arm": arm, "Protocol": label, "Model": "All models (mean)"}
        for n in sizes:
            x = pooled[(pooled.arm == arm) & (pooled.n == n)].iloc[0]
            r[f"n = {n:,}"] = fmt_ci(x["mean"], x.lo, x.hi)
        rows.append(r)
        for m in MODELS:
            r = {"Arm": arm, "Protocol": label, "Model": MODEL_LABEL[m]}
            for n in sizes:
                x = per[(per.arm == arm) & (per.model == m) & (per.n == n)].iloc[0]
                r[f"n = {n:,}"] = fmt_ci(x["mean"], x.lo, x.hi)
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "table2_optimism_auroc.csv", index=False)

    # supplement: every metric, arm, model and size.
    sup = []
    for metric in ["AUROC", "AUPRC", "Brier", "calibration_slope", "HL_pvalue"]:
        s = optimism_summary(metric, pool_models=False)
        s.insert(0, "metric", metric)
        sup.append(s)
    pd.concat(sup).round(4).to_csv(TABLES / "tableS1_optimism_all_metrics.csv", index=False)
    return t


def table3():
    e = pd.read_csv(F_EQUIV)
    e["Reference implementation"] = e.component.map(REFERENCE_IMPL)
    e["Component"] = e.component.map(COMPONENT_LABEL)
    e["Quantity"] = e.quantity.map(QUANTITY_LABEL).fillna(e.quantity)
    e["Within tolerance"] = (e.abs_deviation <= TOLERANCE).map({True: "Yes", False: "No"})

    def num(v):
        return "" if pd.isna(v) else f"{v:g}"

    def dev(v):
        return "0" if v == 0 else (f"{v:.2e}" if v < 1e-3 else f"{v:.2f}")

    t = pd.DataFrame({
        "Component": e["Component"],
        "Reference implementation": e["Reference implementation"],
        "Quantity": e["Quantity"],
        "ClinTAB-ML": e.clintab.map(num),
        "Reference": e.reference.map(num),
        "Absolute deviation": e.abs_deviation.map(dev),
        "Within tolerance": e["Within tolerance"],
    })
    t.to_csv(TABLES / "table3_equivalence.csv", index=False)
    return t


STATUS_LABEL = {"found": "Found", "manual_review": "Manual review",
                "not_applicable": "Not applicable", "not_found": "Not found"}


def table4():
    r = json.load(open(F_TRIPOD))
    t = pd.DataFrame([{
        "Section": i["section"],
        "Item": i["description"],
        "item_id": i["id"],
        "Automated report (ClinTAB-ML)": STATUS_LABEL[i["status"]],
    } for i in r["items"]])
    if has_content(F_EX4_RECOVER):
        s = pd.read_csv(F_EX4_RECOVER)  # item_id, clintab_artifacts, comparator_artifacts
        t = t.merge(s, on="item_id", how="left")
        t = t.rename(columns={"clintab_artifacts": "Recoverable, ClinTAB-ML artifacts",
                              "comparator_artifacts": "Recoverable, comparator artifacts"})
    else:
        t["Recoverable, ClinTAB-ML artifacts"] = PLACEHOLDER
        t["Recoverable, comparator artifacts"] = PLACEHOLDER
    t.drop(columns="item_id").to_csv(TABLES / "table4_tripod.csv", index=False)
    return t


def split_clauses(text):
    """Sentence-level clauses of the generated methods paragraph."""
    parts = re.split(r"(?<=[.)])\s+(?=[A-Z])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def table_s2_clauses():
    clauses = split_clauses(open(F_METHODS).read())
    t = pd.DataFrame({"Clause": range(1, len(clauses) + 1), "Text": clauses})
    if has_content(F_EX4_CLAUSES):
        s = pd.read_csv(F_EX4_CLAUSES)  # clause, rater1, rater2, adjudicated
        t = t.merge(s, on="Clause", how="left")
    else:
        for c in ("Rater 1", "Rater 2", "Adjudicated"):
            t[c] = PLACEHOLDER
    t.to_csv(TABLES / "tableS2_methods_clauses.csv", index=False)
    return t


def table_s3_y1():
    y = pd.read_csv(F_Y1)
    rows = []
    for _, r in y.iterrows():
        for part in ("Val", "Test"):
            rows.append({
                "Model": MODEL_LABEL[r.Model],
                "Partition": "Validation" if part == "Val" else "Test",
                "n": f"{int(r[f'{part} N']):,}",
                "AUROC": f"{r[f'{part} AUROC']:.3f}",
                "AUPRC": f"{r[f'{part} AUPRC']:.3f}",
                "Sensitivity": f"{r[f'{part} Sensitivity']:.3f}",
                "Specificity": f"{r[f'{part} Specificity']:.3f}",
                "PPV": f"{r[f'{part} PPV']:.3f}",
                "NPV": f"{r[f'{part} NPV']:.3f}",
                "Brier": f"{r[f'{part} Brier']:.3f}",
            })
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "tableS3_y1_models.csv", index=False)
    return t


def main():
    ensure_dirs()
    table1()
    table2()
    table3()
    table4()
    table_s2_clauses()
    table_s3_y1()
    print("Tables written to", TABLES)


if __name__ == "__main__":
    main()
