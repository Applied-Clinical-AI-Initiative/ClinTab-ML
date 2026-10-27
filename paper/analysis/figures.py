"""Manuscript figures, black and white safe, sized for journal column width.

Figure 1  Architecture and enforcement points
Figure 2  Inclusion flow
Figure 3  AUROC optimism versus cohort size, faceted by arm
Figure 4  Runtime and peak memory versus cohort size
Figure 5  ROC with precision-recall inset, and calibration (full cohort, Y1)

Run: python paper/analysis/figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from common import (ARMS, F_CAL_PNG, F_ROC_PNG, F_Y1_PRED, FIGURES,
                    MODEL_LABEL, MODELS, OUT, PLACEHOLDER, ensure_dirs, has_content,
                    mpl_style, optimism_summary, save_fig, scaling_totals)

SINGLE = 3.5   # inches, one journal column
DOUBLE = 7.0   # inches, full page width
MARKERS = {"LogisticRegression": "o", "RandomForest": "s", "GradientBoosting": "^"}
FILLS = {"LogisticRegression": "black", "RandomForest": "white", "GradientBoosting": "0.6"}
LINES = {"LogisticRegression": "-", "RandomForest": "--", "GradientBoosting": ":"}


def box(ax, x, y, w, h, text, lw=0.8, ls="-", fc="white", fs=7, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                       lw=lw, ls=ls, ec="black", fc=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, wrap=True)
    return p


def arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=7))


def badge(ax, x, y, label):
    ax.add_patch(plt.Circle((x, y), 0.022, fc="black", ec="black", zorder=5))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=6,
            weight="bold", zorder=6)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # local machine boundary.
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.94, boxstyle="round,pad=0,rounding_size=0.02",
                                lw=0.8, ls=(0, (4, 3)), ec="black", fc="none"))
    ax.text(0.03, 0.915, "Researcher's own machine: no data leave the device",
            fontsize=7, style="italic", va="center")

    x, w, h = 0.05, 0.50, 0.085
    steps = [
        (0.79, "1  Upload CSV and profile columns"),
        (0.675, "2  Confirm column types and per-column missing-data handling"),
        (0.56, "3  Stratified split into training, validation and test partitions"),
        (0.445, "4  Pipeline fitted on the training fold only:\nimputation, scaling, encoding, then SMOTE"),
        (0.33, "5  Grid search scored on the validation partition;\nwinning settings refit on training alone"),
        (0.215, "6  Evaluate on the held-out test partition"),
        (0.06, "Discrimination, calibration, splines, epidemiological measures"),
    ]
    for y, t in steps:
        box(ax, x, y, w, h, t, fs=6.8)
    for (y0, _), (y1, _) in zip(steps[:-1], steps[1:]):
        arrow(ax, x + w / 2, y0, x + w / 2, y1 + h)

    # enforcement points, keyed to the experiment 1 comparator arms.
    badge(ax, x - 0.012, 0.445 + h / 2 + 0.02, "E1")
    badge(ax, x - 0.012, 0.445 + h / 2 - 0.025, "E2")
    badge(ax, x - 0.012, 0.33 + h / 2, "E3")
    badge(ax, x - 0.012, 0.215 + h / 2, "E4")

    # provenance column.
    lx, lw_ = 0.63, 0.34
    box(ax, lx, 0.35, lw_, 0.25,
        "Append-only analysis log (JSON Lines)\n\none entry per action:\ninputs, outputs, timestamp\n\n"
        "entries are never rewritten", lw=1.4)
    for y, _ in steps[:6]:
        arrow(ax, x + w, y + h / 2, lx, min(max(y + h / 2, 0.37), 0.58), ls=(0, (2, 2)))
    hw = (lw_ - 0.02) / 2
    box(ax, lx, 0.18, hw, 0.12, "Methods\nparagraph\n(fixed templates)", fs=6.5)
    box(ax, lx + hw + 0.02, 0.18, hw, 0.12, "TRIPOD+AI\ncoverage report", fs=6.5)
    arrow(ax, lx + hw / 2, 0.35, lx + hw / 2, 0.30)
    arrow(ax, lx + hw * 1.5 + 0.02, 0.35, lx + hw * 1.5 + 0.02, 0.30)

    legend = ("E1  preprocessing fitted inside the\n       training fold (prevents Arm A)\n"
              "E2  SMOTE after splitting, training\n       fold only (prevents Arm B)\n"
              "E3  tuning scored on validation,\n       never test (prevents Arm C)\n"
              "E4  test partition excluded from fitting\n       and tuning; each use is logged")
    ax.text(lx, 0.895, legend, fontsize=6.0, va="top", linespacing=1.35)
    box(ax, lx, 0.04, lw_, 0.10, "Same functions callable from a\ncommand-line interface", fs=6.5, ls=(0, (1, 1)))
    save_fig(fig, "figure1_architecture")
    plt.close(fig)


def fig2_flow():
    facts = json.load(open(OUT / "facts.json"))      # flow counts come from facts.py
    ph = PLACEHOLDER
    fig, ax = plt.subplots(figsize=(SINGLE + 0.6, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x, w, h = 0.04, 0.52, 0.11
    n = lambda k: facts.get(k, ph)
    main = [
        (0.86, f"ACS NSQIP Participant Use Data Files\n{facts['year_first']} to {facts['year_last']}, pooled\nn = {n('flow_raw_n')}"),
        (0.62, f"Primary CPT {facts['cpt_tha']} (THA)\nor {facts['cpt_tka']} (TKA)\nn = {n('flow_cpt_n')}"),
        (0.36, f"Analytic cohort\nn = {facts['cohort_n']}\nTKA {facts['tka_n']}; THA {facts['tha_n']}"),
    ]
    for y, t in main:
        box(ax, x, y, w, h, t)
    arrow(ax, x + w / 2, 0.86, x + w / 2, 0.62 + h)
    arrow(ax, x + w / 2, 0.62, x + w / 2, 0.36 + h)
    ex_y = 0.555
    ax.plot([x + w / 2, 0.61], [ex_y, ex_y], color="black", lw=0.7)
    box(ax, 0.61, ex_y - 0.10, 0.38, 0.20,
        f"Excluded\nn = {n('flow_excl_n')}\nage < 18 y\nnon-elective or emergent\n"
        f"missing outcome\nrevision or bilateral", fs=6.0)
    # partition row.
    parts = [("Training", facts["split_train"]), ("Validation", facts["split_val"]),
             ("Test", facts["split_test"])]
    pw = 0.30
    for i, (lab, v) in enumerate(parts):
        px = 0.02 + i * (pw + 0.03)
        box(ax, px, 0.08, pw, 0.10, f"{lab}\nn = {v}", fs=6.5)
        arrow(ax, x + w / 2, 0.36, px + pw / 2, 0.18)
    ax.text(0.5, 0.025, "Split stratified on 30-day unplanned readmission", ha="center",
            fontsize=6.3, style="italic")
    save_fig(fig, "figure2_flow")
    plt.close(fig)


def fig3_optimism():
    per = optimism_summary("AUROC", pool_models=False)
    sizes = sorted(per.n.unique())
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6), sharex=True, sharey=True)
    offs = {"LogisticRegression": 0.92, "RandomForest": 1.0, "GradientBoosting": 1.08}
    for ax, (arm, label) in zip(axes.flat, ARMS.items()):
        ax.axhline(0, color="black", lw=0.6)
        for m in MODELS:
            d = per[(per.arm == arm) & (per.model == m)].sort_values("n")
            xs = d.n.values * offs[m]
            ax.errorbar(xs, d["mean"], yerr=[d["mean"] - d.lo, d.hi - d["mean"]],
                        fmt=MARKERS[m], ls=LINES[m], color="black", mfc=FILLS[m],
                        mec="black", ms=4, lw=0.8, elinewidth=0.7, capsize=2,
                        label=MODEL_LABEL[m])
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes])
        ax.minorticks_off()
        ax.set_title(f"Arm {arm}: {label}", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Cohort size, n (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC optimism\n(reported minus held-out)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "figure3_optimism")
    plt.close(fig)


def fig4_scaling():
    s, per = scaling_totals()
    sizes = sorted(s.n.unique())
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))
    series = [
        ("End-to-end", per.rename(columns={"seconds": "v"}), "o", "-", "black"),
    ]
    stage_groups = {
        "Train random forest": ["train_RandomForest"],
        "Train gradient boosting": ["train_GradientBoosting"],
        "Train logistic regression": ["train_LogisticRegression"],
        "Methods text + TRIPOD report": ["methods_text", "tripod_report"],
    }
    styles = [("s", "--", "white"), ("^", ":", "0.6"), ("D", "-.", "white"), ("v", (0, (1, 1)), "black")]
    for (lab, stages), (mk, ls, fc) in zip(stage_groups.items(), styles):
        g = s[s.stage.isin(stages)].groupby(["n", "repeat"]).seconds.sum().reset_index()
        g = g.rename(columns={"seconds": "v"})
        series.append((lab, g, mk, ls, fc))
    for lab, g, mk, ls, fc in series:
        agg = g.groupby("n").v.agg(["median", "min", "max"]).reindex(sizes)
        a.errorbar(agg.index, agg["median"], yerr=[agg["median"] - agg["min"], agg["max"] - agg["median"]],
                   fmt=mk, ls=ls, color="black", mfc=fc, mec="black", ms=3.5, lw=0.8,
                   elinewidth=0.6, capsize=1.5, label=lab)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Cohort size, n (log scale)")
    a.set_ylabel("Wall-clock time, s (log scale)")
    a.set_title("A  Runtime", loc="left")

    agg = per.groupby("n").peak_rss_mb.agg(["median", "min", "max"]).reindex(sizes) / 1024
    b.errorbar(agg.index, agg["median"], yerr=[agg["median"] - agg["min"], agg["max"] - agg["median"]],
               fmt="o", ls="-", color="black", mfc="black", ms=3.5, lw=0.8, capsize=1.5,
               label="Peak resident memory")
    b.axhline(8, color="black", ls="--", lw=0.8)
    b.text(sizes[0], 8 * 1.12, "Installed RAM, 8 GB", fontsize=6.5, va="bottom")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_ylim(0.3, 16)
    b.set_yticks([0.5, 1, 2, 4, 8])
    b.set_yticklabels(["0.5", "1", "2", "4", "8"])
    b.set_xlabel("Cohort size, n (log scale)")
    b.set_ylabel("Peak resident memory, GB (log scale)")
    b.set_title("B  Memory", loc="left")
    for ax in (a, b):
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{x:,}" for x in sizes], fontsize=6)
        ax.minorticks_off() if ax is b else None
        ax.spines[["top", "right"]].set_visible(False)
    h_, l_ = a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
    save_fig(fig, "figure4_scaling")
    plt.close(fig)


def fig5_roc_calibration():
    """Redrawn from test-set predictions when they exist. Until then, the two
    application-exported PNGs are placed side by side as an interim figure."""
    if has_content(F_Y1_PRED):
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
        p = pd.read_csv(F_Y1_PRED)  # columns: y_true, logisticregression, randomforest, gradientboosting
        fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 3.3))
        ins = a.inset_axes([0.52, 0.08, 0.45, 0.42])
        for m in MODELS:
            fpr, tpr, _ = roc_curve(p.y_true, p[m])
            a.plot(fpr, tpr, ls=LINES[m], color="black", label=MODEL_LABEL[m])
            pr, rc, _ = precision_recall_curve(p.y_true, p[m])
            ins.plot(rc, pr, ls=LINES[m], color="black", lw=0.7)
            fr, mp = calibration_curve(p.y_true, p[m], n_bins=10, strategy="quantile")
            b.plot(mp, fr, ls=LINES[m], marker=MARKERS[m], mfc=FILLS[m], mec="black",
                   color="black", ms=3.5, label=MODEL_LABEL[m])
        a.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls=(0, (1, 2)))
        ins.axhline(p.y_true.mean(), color="0.5", lw=0.6, ls=(0, (1, 2)))
        ins.set_xlabel("Recall", fontsize=6)
        ins.set_ylabel("Precision", fontsize=6)
        ins.tick_params(labelsize=5.5)
        a.set_xlabel("1 minus specificity")
        a.set_ylabel("Sensitivity")
        a.set_title("A  ROC, with precision-recall inset", loc="left")
        a.legend(frameon=False, loc="upper left")
        lim = max(b.get_xlim()[1], b.get_ylim()[1])
        b.plot([0, lim], [0, lim], color="0.5", lw=0.6, ls=(0, (1, 2)))
        b.set_xlabel("Mean predicted risk (decile)")
        b.set_ylabel("Observed readmission proportion")
        b.set_title("B  Calibration, test partition", loc="left")
        for ax in (a, b):
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_fig(fig, "figure5_roc_calibration")
        plt.close(fig)
        return "redrawn"
    import matplotlib.image as mpimg
    roc, cal = mpimg.imread(F_ROC_PNG), mpimg.imread(F_CAL_PNG)
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * roc.shape[0] / (roc.shape[1] + cal.shape[1] * roc.shape[0] / cal.shape[0])))
    wr = [roc.shape[1] / roc.shape[0], cal.shape[1] / cal.shape[0]]
    gs = fig.add_gridspec(1, 2, width_ratios=wr, wspace=0.02)
    for i, (img, lab) in enumerate([(roc, "A"), (cal, "B")]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")
        ax.text(0, -0.02, lab, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")
    fig.savefig(FIGURES / "figure5_roc_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return "interim composite of exported PNGs"


def main():
    ensure_dirs()
    mpl_style()
    fig1_architecture()
    fig2_flow()
    fig3_optimism()
    fig4_scaling()
    status = fig5_roc_calibration()
    print("Figures written to", FIGURES, "| Figure 5:", status)


if __name__ == "__main__":
    main()
