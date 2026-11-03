"""experiment1_figure.py
Optimism versus n: AUROC (reported minus honest), one panel per arm, one line
per model, mean over seeds with a 95% interval. Reads aggregate metrics only.

Usage:
    python scripts/experiment1_figure.py --csv optimism_results.csv --out figure_optimism_vs_n.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = {"A": "A: preprocessing fit on all data", "B": "B: SMOTE before split",
        "C": "C: tuned and reported on same partition", "D": "D: ClinTAB-ML pipeline"}
COLORS = {"LogisticRegression": "#3b7dd8", "RandomForest": "#d8763b", "GradientBoosting": "#3ba86b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figure_optimism_vs_n.png")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d = d[d.metric == "AUROC"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (arm, title) in zip(axes, ARMS.items()):
        for model, col in COLORS.items():
            s = d[(d.arm == arm) & (d.model == model)].groupby("n").optimism
            m, se = s.mean(), s.sem()
            ax.errorbar(m.index, m.values, yerr=1.96 * se.values, marker="o", ms=4,
                        capsize=3, lw=1.5, color=col, label=model)
        ax.axhline(0, color="#555555", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([1000, 5000, 25000, 100000])
        ax.set_xticklabels(["1k", "5k", "25k", "100k"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cohort size (n)")
    axes[0].set_ylabel("AUROC optimism (reported minus honest)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=600, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
