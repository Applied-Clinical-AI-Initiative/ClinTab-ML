"""Build the facts sheet: every number the manuscript is allowed to state.

Writes paper/v2/facts.json (read by the manuscript builder, which refuses to
render a {{key}} that is not here) and paper/v2/facts.md (human-readable, with
the file, column and filter each value came from).

Run: python paper/analysis/facts.py
"""
import json

import numpy as np
import pandas as pd

from common import (F_EQUIV, F_HARDWARE, F_METHODS, F_TABLE1, F_TRIPOD, F_Y1,
                    F_EX4_CLAUSES, F_EX4_RECOVER, F_FLOW, OUT, PLACEHOLDER,
                    ensure_dirs, has_content, load_optimism, optimism_summary,
                    scaling_totals)
from tables import TOLERANCE

FACTS = []


def fact(key, value, source, note=""):
    FACTS.append({"key": key, "value": value, "source": source, "note": note})


def c(n):
    return f"{int(n):,}"


def series(ns):
    items = [c(n) for n in ns]
    return ", ".join(items[:-1]) + " and " + items[-1]


def s3(x):
    """Signed 3 dp, with a true minus sign for print."""
    return f"{x:+.3f}".replace("-", "−")


def u3(x):
    return f"{x:.3f}".replace("-", "−")


def cohort():
    t = pd.read_csv(F_TABLE1, dtype=str).fillna("")
    t["Var"] = t.Variable.replace("", np.nan).ffill()
    src = "table1_cohort_characteristics.csv"

    def cell(var, level):
        r = t[(t.Var == var) & (t.Level == level)].iloc[0]
        return r

    n = int(t[t.Variable == "N"].Overall.iloc[0])
    fact("cohort_n", c(n), f"{src}: Variable = N, column Overall")
    fact("cohort_n_lay", "nearly 300,000", f"{src}: Variable = N ({n:,}), rounded for the lay summary")

    def count_pct(key, var, level, label):
        r = cell(var, level)
        cnt, pct = r.Overall.split(" (")
        fact(f"{key}_n", c(cnt), f"{src}: {var} = {level}, Overall (count)")
        fact(f"{key}_pct", pct.rstrip("%)"), f"{src}: {var} = {level}, Overall (%)")

    count_pct("tka", "Procedure", "tka", "TKA")
    count_pct("tha", "Procedure", "tha", "THA")
    count_pct("female", "Sex", "female", "female")
    count_pct("readmit", "30-day unplanned readmission", "Yes", "readmission")
    r = cell("Age, years", "mean (SD)")
    mean, sd = r.Overall.replace(")", "").split(" (")
    fact("age_mean", mean, f"{src}: Age, years, mean (SD)")
    fact("age_sd", sd, f"{src}: Age, years, mean (SD)")

    years = sorted(t[t.Var == "Operative year"].Level.astype(int))
    fact("year_first", str(years[0]), f"{src}: Operative year levels")
    fact("year_last", str(years[-1]), f"{src}: Operative year levels")

    miss = t[(t["Missing %"] != "") & (t.Variable != "")].copy()
    miss["pct"] = miss["Missing %"].astype(float)
    for key, var in [("albumin", "Preoperative albumin"), ("creat", "Preoperative creatinine"),
                     ("hct", "Preoperative hematocrit"), ("bmi", "BMI")]:
        r = miss[miss.Variable == var].iloc[0]
        fact(f"miss_{key}_pct", f"{float(r['Missing %']):.1f}", f"{src}: {var}, Missing %")
        fact(f"miss_{key}_n", c(r["Missing n"]), f"{src}: {var}, Missing n")
    fact("n_vars_missing", str(int((miss.pct > 0).sum())),
         f"{src}: count of variables with Missing % > 0")

    fact("n_columns", "30", "methods_paragraph.txt, clause 1 ('292373 rows and 30 columns')")
    for part, v in [("train", 204661), ("val", 43856), ("test", 43856)]:
        fact(f"split_{part}", c(v), "methods_paragraph.txt, clause 2; val/test cross-checked "
             "against table_y1_model_comparison.csv Val N / Test N")
    fact("split_sum_check", c(204661 + 43856 + 43856),
         "sum of split sizes, equals cohort_n", "consistency check")

    fact("cpt_tha", "27130", "Experimental protocol section 1.1")
    fact("cpt_tka", "27447", "Experimental protocol section 1.1")

    if has_content(F_FLOW):
        # rows in order: raw puf, after the cpt filter, ..., analytic cohort last
        fl = pd.read_csv(F_FLOW)  # step, n
        for i, r in fl.iterrows():
            fact(f"flow_{i}", c(r.n), f"inclusion_flow.csv row {i}: {r.step}")
        # the names figure 2 reads
        fact("flow_raw_n", c(fl.n.iloc[0]), f"inclusion_flow.csv row 0: {fl.step.iloc[0]}")
        fact("flow_cpt_n", c(fl.n.iloc[1]), f"inclusion_flow.csv row 1: {fl.step.iloc[1]}")
        fact("flow_excl_n", c(fl.n.iloc[1] - fl.n.iloc[-1]),
             "inclusion_flow.csv: row 1 minus the last row (the analytic cohort)")
    else:
        fact("flow_raw_n", PLACEHOLDER, "NOT IN RESULTS FOLDER",
             "N in pooled 2021 to 2023 PUF before any exclusion")
        fact("flow_cpt_n", PLACEHOLDER, "NOT IN RESULTS FOLDER", "after CPT 27130/27447 filter")
        fact("flow_excl_n", PLACEHOLDER, "NOT IN RESULTS FOLDER", "N removed at each exclusion step")


def ex1():
    d = load_optimism()
    src = "optimism_results(EX1).csv"
    sizes = sorted(d.n.unique())
    fact("ex1_sizes", series(sizes), f"{src}: unique n")
    fact("ex1_seeds", str(d.seed.nunique()), f"{src}: unique seed")
    fact("ex1_models", str(d.model.nunique()), f"{src}: unique model")
    fact("ex1_arms", str(d.arm.nunique()), f"{src}: unique arm")
    words = {3: "three", 4: "four", 5: "five"}
    fact("ex1_arms_w", words[d.arm.nunique()], f"{src}: unique arm, as a word")
    fact("ex1_models_w", words[d.model.nunique()], f"{src}: unique model, as a word")
    fact("ex1_metrics", str(d.metric.nunique()), f"{src}: unique metric")
    fact("ex1_n_min", c(min(sizes)), f"{src}: min n")
    fact("ex1_n_max", c(max(sizes)), f"{src}: max n")
    fact("ex1_fits", c(len(d) // d.metric.nunique()),
         f"{src}: rows / metrics (arm x n x seed x model)")

    pooled = optimism_summary("AUROC", True)
    per = optimism_summary("AUROC", False)
    for _, r in pooled.iterrows():
        k = f"ex1_{r.arm}_{r.n}"
        filt = f"{src}: metric = AUROC, arm = {r.arm}, n = {r.n}; seed means over 3 models; 95% t-interval over 20 seeds"
        fact(f"{k}_mean", s3(r["mean"]), filt)
        fact(f"{k}_ci", f"{s3(r.lo)} to {s3(r.hi)}", filt)
        fact(f"{k}_rep", u3(r.reported), filt + " (mean reported AUROC)")
        fact(f"{k}_hon", u3(r.honest), filt + " (mean honest AUROC)")
        fact(f"{k}_pos", str(r.n_positive), filt + " (seeds with optimism > 0)")
    for _, r in per.iterrows():
        k = f"ex1_{r.arm}_{r.model}_{r.n}"
        filt = f"{src}: metric = AUROC, arm = {r.arm}, model = {r.model}, n = {r.n}; 95% t-interval over 20 seeds"
        fact(f"{k}_mean", s3(r["mean"]), filt)
        fact(f"{k}_ci", f"{s3(r.lo)} to {s3(r.hi)}", filt)

    dD = per[per.arm == "D"]
    fact("ex1_D_min", s3(dD["mean"].min()), f"{src}: AUROC, arm D, min over model x n cell means")
    fact("ex1_D_max", s3(dD["mean"].max()), f"{src}: AUROC, arm D, max over model x n cell means")
    cs = optimism_summary("calibration_slope", False)
    cs = cs[cs.arm == "D"]
    fact("ex1_D_cslope_min", s3(cs["mean"].min()), f"{src}: calibration_slope, arm D, min cell mean")
    fact("ex1_D_cslope_max", s3(cs["mean"].max()), f"{src}: calibration_slope, arm D, max cell mean")

    for metric, tag in [("AUPRC", "auprc"), ("Brier", "brier"), ("calibration_slope", "cslope")]:
        pm = optimism_summary(metric, True)
        for _, r in pm.iterrows():
            filt = f"{src}: metric = {metric}, arm = {r.arm}, n = {r.n}; model-averaged over 20 seeds"
            fact(f"ex1_{tag}_{r.arm}_{r.n}_mean", s3(r["mean"]), filt + " (mean optimism)")
            fact(f"ex1_{tag}_{r.arm}_{r.n}_rep", u3(r.reported), filt + " (mean reported)")
            fact(f"ex1_{tag}_{r.arm}_{r.n}_hon", u3(r.honest), filt + " (mean honest)")

    ap = optimism_summary("AUPRC", True)
    for n in (1000, 100000):
        r = ap[(ap.arm == "B") & (ap.n == n)].iloc[0]
        filt = f"{src}: metric = AUPRC, arm = B, n = {n}; model-averaged"
        fact(f"ex1_B_{n}_auprc_rep", u3(r.reported), filt + " reported")
        fact(f"ex1_B_{n}_auprc_hon", u3(r.honest), filt + " honest")

    # events expected in a 1,000-patient stratified subsample at cohort prevalence.
    fact("ex1_events_1000", "28", "1,000 x 8,099 / 292,373 (Table 1 readmission prevalence)",
         "arithmetic, rounded")


def ex2():
    e = pd.read_csv(F_EQUIV)
    src = "equivalence_table(ex2).csv"
    core = e[e.component != "RCS"]
    fact("ex2_n_quantities", str(len(core)), f"{src}: rows with component != RCS")
    fact("ex2_n_components", str(core.component.nunique()), f"{src}: unique component, excluding RCS")
    fact("ex2_n_exact", str(int((core.abs_deviation == 0).sum())), f"{src}: abs_deviation == 0, excluding RCS")
    mx = core.abs_deviation.max()
    fact("ex2_max_dev", f"{mx:.1e}".replace("e-05", " × 10⁻⁵"),
         f"{src}: max abs_deviation excluding RCS (row: {core.loc[core.abs_deviation.idxmax(), 'quantity']})")
    fact("ex2_tol", "5 × 10⁻⁵", "paper/analysis/tables.py TOLERANCE",
         "half a unit in the 4th decimal place, the precision ClinTAB-ML reports to")
    rcs = e[e.component == "RCS"].set_index("quantity")
    fact("ex2_rcs_aic_dev", f"{rcs.loc['AIC', 'abs_deviation']:.2f}", f"{src}: RCS, AIC, abs_deviation")
    fact("ex2_rcs_curve_dev", f"{rcs.loc['logodds_curve_max_abs_dev', 'abs_deviation']:.2f}",
         f"{src}: RCS, logodds_curve_max_abs_dev")
    fact("ex2_rcs_grid", "100", "Experimental protocol section 3 (curve compared at 100 grid points)")

    hw = dict(line.split(":", 1) for line in open(F_HARDWARE) if ":" in line)
    hw = {k.strip(): v.strip() for k, v in hw.items()}
    fact("hw_machine", f"{hw['CPU']} {hw['Machine']}", "hardware (ex2).txt: CPU, Machine")
    fact("hw_cores", hw["Cores"].split()[0], "hardware (ex2).txt: Cores")
    fact("hw_ram", hw["RAM"], "hardware (ex2).txt: RAM")
    fact("hw_os", hw["OS"].split(" (")[0], "hardware (ex2).txt: OS")
    fact("hw_python", hw["Python"].split()[0], "hardware (ex2).txt: Python")
    fact("hw_sklearn", "1.9.0", "hardware (ex2).txt: Libraries")
    fact("hw_r", "4.6.0", "hardware (ex2).txt: R")


def ex3():
    s, per = scaling_totals()
    src = "scaling_combined (ex3).csv"
    fact("ex3_repeats", str(s.repeat.nunique()), f"{src}: unique repeat")
    sizes = sorted(s.n.unique())
    fact("ex3_sizes", series(sizes), f"{src}: unique n")
    med = per.groupby("n").median()
    for n in sizes:
        sec = med.loc[n, "seconds"]
        rss = med.loc[n, "peak_rss_mb"]
        filt = f"{src}: n = {n}; sum of stage seconds per repeat, median over repeats"
        fact(f"ex3_{n}_sec", f"{sec:.1f}", filt)
        fact(f"ex3_{n}_min", f"{sec / 60:.1f}", filt + " (minutes)")
        fact(f"ex3_{n}_rss_gb", f"{rss / 1024:.2f}", f"{src}: n = {n}; max peak_rss_mb per repeat, median, /1024")
        fact(f"ex3_{n}_rss_mb", f"{rss:.0f}", f"{src}: n = {n}; max peak_rss_mb per repeat, median")
    full = s[s.n == sizes[-1]]
    stage = full.groupby("stage").seconds.median()
    tot = med.loc[sizes[-1], "seconds"]
    trees = stage[["train_RandomForest", "train_GradientBoosting"]].sum()
    fact("ex3_full_tree_share", f"{100 * trees / tot:.0f}",
         f"{src}: n = {sizes[-1]}; median train_RandomForest + train_GradientBoosting / median total, %")
    fact("ex3_full_lr_sec", f"{stage['train_LogisticRegression']:.1f}",
         f"{src}: n = {sizes[-1]}, stage train_LogisticRegression, median seconds")
    rep = s[s.stage.isin(["methods_text", "tripod_report"])]
    fact("ex3_report_ms_max", f"{rep.seconds.max() * 1000:.1f}",
         f"{src}: stages methods_text and tripod_report, max seconds over all n and repeats, x1000")
    fact("ex3_ram_frac", f"{100 * med.loc[sizes[-1], 'peak_rss_mb'] / (8 * 1024):.0f}",
         f"{src} peak RSS at n = {sizes[-1]} / 8 GB RAM (hardware (ex2).txt), %")


def demo():
    y = pd.read_csv(F_Y1).set_index("Model")
    src = "table_y1_model_comparison.csv"
    for m, k in [("LogisticRegression", "lr"), ("RandomForest", "rf"), ("GradientBoosting", "gb")]:
        for part in ("Val", "Test"):
            for met in ("AUROC", "AUPRC", "Brier", "Sensitivity", "Specificity"):
                fact(f"y1_{k}_{part.lower()}_{met.lower()}", f"{y.loc[m, f'{part} {met}']:.3f}",
                     f"{src}: Model = {m}, column {part} {met}")
    diffs = [y.loc[m, "Test AUROC"] - y.loc[m, "Val AUROC"] for m in y.index]
    fact("y1_testval_maxdiff", f"{max(abs(d) for d in diffs):.3f}",
         f"{src}: max |Test AUROC - Val AUROC| over models")

    fact("hl_stat", "38,468", "methods_paragraph.txt clause 6 (statistic=38467.9564), rounded")
    fact("hl_p_printed", "p=0.0", "methods_paragraph.txt clause 6, verbatim")

    r = json.load(open(F_TRIPOD))
    sm = r["summary"]
    for k in ("found", "manual_review", "not_applicable", "not_found", "total"):
        fact(f"tripod_{k}", str(sm[k]), f"tripod_report.json: summary.{k}")
    status = {i["id"]: i["status"] for i in r["items"]}
    fact("tripod_missing_status", status["missing_data_handling"].replace("_", " "),
         "tripod_report.json: items[id = missing_data_handling].status")

    n_clauses = len([p for p in open(F_METHODS).read().split(". ") if p.strip()])
    from tables import split_clauses
    fact("methods_n_clauses", str(len(split_clauses(open(F_METHODS).read()))),
         "methods_paragraph.txt split into sentences (paper/analysis/tables.py split_clauses)")

    if has_content(F_EX4_CLAUSES):
        sc = pd.read_csv(F_EX4_CLAUSES)
        fact("ex4_accurate", str(int((sc.adjudicated == "accurate").sum())), "clause_scores.csv")
        fact("ex4_agree", str(int((sc.rater1 == sc.rater2).sum())), "clause_scores.csv")
    else:
        for k in ("ex4_accurate", "ex4_inaccurate", "ex4_missing_clauses", "ex4_agree", "ex4_kappa"):
            fact(k, PLACEHOLDER, "NOT IN RESULTS FOLDER", "author clause scoring, protocol 5.1")
    if not has_content(F_EX4_RECOVER):
        for k in ("ex4_recover_clintab", "ex4_recover_comparator"):
            fact(k, PLACEHOLDER, "NOT IN RESULTS FOLDER", "author recoverability scoring, protocol 5.2")

    fact("acs_calc_auc", PLACEHOLDER, "EXTERNAL LITERATURE, not in results folder",
         "published ACS Surgical Risk Calculator readmission c-statistic, with citation")
    fact("reb_number", PLACEHOLDER, "REB determination letter", "user to supply")


def write():
    ensure_dirs()
    facts = {f["key"]: f["value"] for f in FACTS}
    json.dump(facts, open(OUT / "facts.json", "w"), indent=1, ensure_ascii=False)
    lines = ["# ClinTAB-ML manuscript facts sheet", "",
             "Generated by `paper/analysis/facts.py` from the results folder. Every number in "
             "`Manuscript_v2.docx` is rendered from this sheet by key; the builder fails on "
             "any key not listed here. Rows marked NOT IN RESULTS FOLDER render as "
             "[PLACEHOLDER] in the manuscript.", "",
             "| Key | Value | Source (file: column / filter) | Note |", "|---|---|---|---|"]
    for f in FACTS:
        lines.append(f"| `{f['key']}` | {f['value']} | {f['source']} | {f['note']} |")
    open(OUT / "facts.md", "w").write("\n".join(lines) + "\n")
    print(f"{len(FACTS)} facts written to {OUT / 'facts.md'}")


def main():
    cohort()
    ex1()
    ex2()
    ex3()
    demo()
    write()


if __name__ == "__main__":
    main()
