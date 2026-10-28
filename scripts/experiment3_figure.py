"""experiment3_figure.py
Runtime and peak memory versus cohort size from the scaling CSVs (timings
only, no patient data). Later files override earlier ones for the same n.

Usage:
    python scripts/experiment3_figure.py --csv old.csv new1.csv new2.csv \
        --out-fig figure_scaling.png --out-csv scaling_combined.csv
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", nargs="+", required=True)
ap.add_argument("--out-fig", required=True)
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()

frames = {}
for f in a.csv:
    d = pd.read_csv(f)
    if "resampled" not in d:
        d["resampled"] = False
    for n, g in d.groupby("n"):
        frames[n] = g
d = pd.concat(frames.values(), ignore_index=True).sort_values(["n", "repeat"])
d.to_csv(a.out_csv, index=False)

per_run = d.groupby(["n", "repeat"]).seconds.sum().groupby("n").agg(["mean", "std", "count"])
train = d[d.stage.str.startswith("train_")].groupby(["n", "repeat", "stage"]).seconds.sum() \
    .groupby(["n", "stage"]).mean().unstack("stage")
rss = d.groupby("n").peak_rss_mb.max()
resampled = set(d[d.resampled.astype(bool)].n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.errorbar(per_run.index, per_run["mean"], yerr=per_run["std"].fillna(0), marker="o",
             color="black", capsize=3, lw=2, label="Total pipeline")
colors = {"train_LogisticRegression": "#3b7dd8", "train_RandomForest": "#d8763b",
          "train_GradientBoosting": "#3ba86b"}
for col, c in colors.items():
    ax1.plot(train.index, train[col], marker="s", ms=4, lw=1.3, color=c,
             label=col.replace("train_", "Train "))
for n in resampled:
    ax1.annotate("bootstrap-resampled\ncohort", (n, per_run.loc[n, "mean"]),
                 textcoords="offset points", xytext=(-70, -28), fontsize=8)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Cohort size (n)")
ax1.set_ylabel("Wall-clock seconds")
ax1.set_title("Runtime versus cohort size")
ax1.legend(frameon=False, fontsize=8)
ax2.plot(rss.index, rss.values / 1024, marker="o", color="#8a3b3b")
ax2.set_xscale("log")
ax2.set_xlabel("Cohort size (n)")
ax2.set_ylabel("Peak resident memory (GB)")
ax2.set_title("Peak memory versus cohort size")
fig.tight_layout()
fig.savefig(a.out_fig, dpi=600, bbox_inches="tight", facecolor="white")
print(per_run.round(1).to_string())
print((rss / 1024).round(2).to_string())
