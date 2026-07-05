"""plots.py
Server-side matplotlib rendering. Returns base64 PNGs for inline <img> display
and can write PNG/PDF files for download. No SVG / canvas / client charting --
all figures are produced here in Python (matplotlib 'Agg' backend).

(Not in the original file list, but pulled out of routes.py to keep HTTP and
drawing code separate -- noted in the README.)
"""
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})


def _encode(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fig_to_png(fig):
    return "data:image/png;base64," + _encode(fig, "png")


def fig_to_pdf_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Curves that can overlay several models
# ---------------------------------------------------------------------------
def roc_overlay(series, title="ROC Curve"):
    """series: list of {name, fpr, tpr, auc}."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for s in series:
        ax.plot(s["fpr"], s["tpr"], lw=2,
                label=f"{s['name']} (AUC={s['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    return fig


def pr_overlay(series, title="Precision–Recall Curve"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for s in series:
        ax.plot(s["recall"], s["precision"], lw=2,
                label=f"{s['name']} (AP={s['ap']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    return fig


def confusion_plot(cm, labels=("0", "1"), title="Confusion Matrix"):
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], labels=[f"Pred {labels[0]}", f"Pred {labels[1]}"])
    ax.set_yticks([0, 1], labels=[f"True {labels[0]}", f"True {labels[1]}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black",
                    fontsize=14)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    return fig


def calibration_plot(prob_pred, prob_true, name="model", title="Calibration"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
    ax.plot(prob_pred, prob_true, "o-", lw=2, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig


def residual_plot(pred, resid, title="Residuals"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(pred, resid, s=14, alpha=0.6)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title(title)
    return fig


def pred_vs_actual_plot(actual, pred, title="Predicted vs Actual"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, pred, s=14, alpha=0.6)
    lo = min(min(actual), min(pred)); hi = max(max(actual), max(pred))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig


def importance_plot(names, values, title="Feature Importance (top 20)"):
    names, values = list(names)[:20][::-1], list(values)[:20][::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.32 * len(names))))
    ax.barh(names, values, color="#3b7dd8")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0)
    return fig


def spline_plot(x, logodds, lower, upper, knots, predictor="x",
                title="Restricted Cubic Spline"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, logodds, color="#1f4e96", lw=2, label="Log-odds")
    if len(lower):
        ax.fill_between(x, lower, upper, color="#1f4e96", alpha=0.15, label="95% CI")
    for k in knots:
        ax.axvline(k, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel(predictor)
    ax.set_ylabel("Log-odds of outcome")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def km_plot(curves, title="Kaplan–Meier"):
    """curves: list of {name, timeline, survival, lower, upper}."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in curves:
        ax.step(c["timeline"], c["survival"], where="post", lw=2, label=c["name"])
        if c.get("lower"):
            ax.fill_between(c["timeline"], c["lower"], c["upper"],
                            step="post", alpha=0.15)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def calibration_hl_plot(prob_pred, obs, expected=None, title="Calibration (Hosmer–Lemeshow)"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
    ax.plot(prob_pred, obs, "o-", lw=2, label="Observed")
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig
