"""Shared paths, loaders and plotting style for the manuscript analysis.

Everything here reads from the results folder produced by the experiment runs
and writes into paper/v2/. Nothing reads the NSQIP PUF directly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
OUT = REPO / "paper" / "v2"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

F_OPTIMISM = RESULTS / "optimism_results(EX1).csv"
F_EQUIV = RESULTS / "equivalence_table(ex2).csv"
F_SCALING = RESULTS / "scaling_combined (ex3).csv"
F_HARDWARE = RESULTS / "hardware (ex2).txt"
F_TABLE1 = RESULTS / "section7_outputs" / "table1_cohort_characteristics.csv"
F_Y1 = RESULTS / "section7_outputs" / "table_y1_model_comparison.csv"
F_ROC_PNG = RESULTS / "section7_outputs" / "figure_roc_pr_overlay.png"
F_CAL_PNG = RESULTS / "section7_outputs" / "figure_calibration_curve.png"
F_METHODS = RESULTS / "experiment4" / "methods_paragraph.txt"
F_TRIPOD = RESULTS / "experiment4" / "tripod_report.json"
# inputs not yet in the results folder. scripts fall back to placeholders
# until these files exist, then pick them up with no code change.
F_EX4_CLAUSES = RESULTS / "experiment4" / "clause_scores.csv"
F_EX4_RECOVER = RESULTS / "experiment4" / "recoverability_scores.csv"
F_FLOW = RESULTS / "section7_outputs" / "inclusion_flow.csv"
F_Y1_PRED = RESULTS / "section7_outputs" / "y1_test_predictions.csv"

ARMS = {
    "A": "Preprocessing fitted before splitting",
    "B": "SMOTE applied before splitting",
    "C": "Tuning scored on the test partition",
    "D": "ClinTAB-ML default pipeline",
}
MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting"]
MODEL_LABEL = {"LogisticRegression": "Logistic regression",
               "RandomForest": "Random forest",
               "GradientBoosting": "Gradient boosting"}
PLACEHOLDER = "[PLACEHOLDER]"


def has_content(path):
    """True for a file that exists and isn't an empty stub."""
    return path.exists() and path.stat().st_size > 0


def ensure_dirs():
    for d in (OUT, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def t_ci(x, level=0.95):
    """Mean and t-based confidence interval across seeds."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def load_optimism():
    return pd.read_csv(F_OPTIMISM)


def optimism_summary(metric="AUROC", pool_models=True):
    """Mean optimism with 95% t-interval across seeds.

    pool_models=True averages the three models within each seed first, so the
    interval reflects seed-to-seed variation of the arm as a whole.
    """
    d = load_optimism()
    d = d[d.metric == metric]
    keys = ["arm", "n"] if pool_models else ["arm", "model", "n"]
    if pool_models:
        d = d.groupby(["arm", "n", "seed"], as_index=False)[["reported", "honest", "optimism"]].mean()
    rows = []
    for k, g in d.groupby(keys):
        m, lo, hi = t_ci(g.optimism)
        rec = dict(zip(keys, k))
        rec.update(mean=m, lo=lo, hi=hi, reported=g.reported.mean(),
                   honest=g.honest.mean(), n_seeds=len(g),
                   n_positive=int((g.optimism > 0).sum()))
        rows.append(rec)
    return pd.DataFrame(rows)


def scaling_totals():
    """Per-repeat end-to-end wall clock and peak RSS."""
    s = pd.read_csv(F_SCALING)
    per = s.groupby(["n", "repeat"]).agg(seconds=("seconds", "sum"),
                                         peak_rss_mb=("peak_rss_mb", "max")).reset_index()
    return s, per


def mpl_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_fig(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
