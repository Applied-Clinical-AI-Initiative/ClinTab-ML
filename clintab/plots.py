"""plots.py
Server-side matplotlib rendering. Returns base64 PNGs for inline <img> display
and can write PNG/PDF files for download. No SVG / canvas / client charting,
all figures are produced here in Python (matplotlib 'Agg' backend).
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
# curves that can overlay several models
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


def pr_overlay(series, title="Precision-Recall Curve"):
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


def _finite(values):
    """list -> float array with None (sanitized NaN) turned back into NaN."""
    if values is None:
        return np.array([], dtype=float)
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _fmt_ci(est, ci, digits=2):
    if est is None:
        return "n/a"
    text = f"{est:.{digits}f}"
    if ci and ci[0] is not None and ci[1] is not None:
        text += f" ({ci[0]:.{digits}f} to {ci[1]:.{digits}f})"
    return text


def _calibration_text(stats):
    """annotation lines from a validation.calibration_stats dict."""
    lines = [f"Slope {_fmt_ci(stats.get('slope'), stats.get('slope_ci'))}",
             f"CITL {_fmt_ci(stats.get('citl'), stats.get('citl_ci'))}",
             f"O:E {_fmt_ci(stats.get('oe_ratio'), stats.get('oe_ci'))}"]
    if stats.get("ici") is not None:
        lines.append(f"ICI {stats['ici']:.3f}")
    return "\n".join(lines)


def flexible_calibration_plot(p_grid, observed, binned_pred=None, binned_obs=None,
                              predictions=None, stats=None, name="model",
                              title="Calibration"):
    """Flexible (lowess) calibration curve against the ideal diagonal, with
    optional binned points (e.g. deciles), a histogram of the predicted
    risks underneath, and a box with slope, calibration-in-the-large, O:E
    and ICI (with 95% CIs when stats has them). predictions: the predicted
    risks, or a precomputed {edges, counts} (ml.binary_metrics
    coords["risk_histogram"]). stats: a validation.calibration_stats dict.
    Axes zoom to the range of the predictions so a rare outcome is not
    squeezed into a corner."""
    grid, obs = _finite(p_grid), _finite(observed)
    bx, by = _finite(binned_pred), _finite(binned_obs)
    hist = predictions if isinstance(predictions, dict) else None
    if hist is not None:
        edges, counts = _finite(hist.get("edges")), _finite(hist.get("counts"))
        preds = edges[-1:] if counts.size and len(edges) == counts.size + 1 else np.array([])
    else:
        preds = _finite(predictions)
        preds = preds[np.isfinite(preds)]

    seen = np.concatenate([grid, obs, bx, by, preds])
    seen = seen[np.isfinite(seen)]
    hi = float(min(1.0, max(0.05, seen.max() * 1.1))) if seen.size else 1.0

    if preds.size:
        fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        axh = None
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Ideal")
    if grid.size:
        ax.plot(grid, obs, lw=2, color="#1f4e96", label=f"{name} (lowess)")
    if bx.size:
        ax.plot(bx, by, "o", ms=5, color="#d9822b", label="Grouped (observed)")
    ax.set_ylabel("Observed proportion")
    ax.set_title(title)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.legend(loc="lower right", fontsize=8)
    if stats:
        ax.text(0.03, 0.97, _calibration_text(stats), transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9})
    if axh is not None:
        if hist is not None:
            axh.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="0.55")
        else:
            axh.hist(preds, bins=np.linspace(0, hi, 41), color="0.55")
        axh.set_xlabel("Predicted probability")
        axh.set_ylabel("Count")
        axh.grid(axis="x", alpha=0)
    else:
        ax.set_xlabel("Predicted probability")
    return fig


def dca_plot(curves, treat_all, thresholds, title="Decision curve"):
    """Net benefit against threshold probability. curves: list of
    {name, net_benefit} on the same thresholds; treat_all from
    validation.decision_curve; treat none is the line at 0. The y axis
    stops a little below 0 so the steep fall of treat all does not flatten
    the curves that matter."""
    t = _finite(thresholds)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    tops = [0.0]
    for c in curves:
        nb = _finite(c["net_benefit"])
        ax.plot(t, nb, lw=2, label=c["name"])
        if np.isfinite(nb).any():
            tops.append(float(np.nanmax(nb)))
    ta = _finite(treat_all)
    ax.plot(t, ta, color="0.45", lw=1.5, ls="--", label="Treat all")
    if np.isfinite(ta).any():
        tops.append(float(np.nanmax(ta)))
    ax.axhline(0, color="k", lw=1.2, label="Treat none")
    top = max(tops)
    top = top if top > 0 else 0.05
    ax.set_ylim(-0.1 * top, top * 1.08)
    if t.size:
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(fontsize=8)
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


def _short_label(name, max_len=42):
    """Display-only cleanup for a feature name: drop the ColumnTransformer's
    num__/cat__ prefix and truncate very long one-hot category labels (e.g. a
    multi-value categorical column whose raw values are comma-joined strings)
    so a handful of outlier labels don't blow up the whole plot's width. The
    underlying CSV export is untouched, this only affects the PNG."""
    if name.startswith("num__") or name.startswith("cat__"):
        name = name.split("__", 1)[1]
    if len(name) > max_len:
        name = name[:max_len - 1].rstrip() + "…"
    return name


def importance_plot(names, values, title="Feature Importance (top 20)"):
    names = [_short_label(n) for n in names][:20][::-1]
    values = list(values)[:20][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(names))))
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


def km_plot(curves, title="Kaplan-Meier", risk_table=None):
    """curves: list of {name, timeline, survival, lower, upper}.
    risk_table: optional {time_points, groups: [{name, at_risk}]} (as returned
    by epi.kaplan_meier) drawn as a number at risk table under the axes."""
    fig, ax, tab = _curve_axes(risk_table)
    colors = _series_colors()
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        ax.step(c["timeline"], c["survival"], where="post", lw=2, label=c["name"],
                color=col)
        if c.get("lower"):
            ax.fill_between(c["timeline"], _floats(c["lower"]), _floats(c["upper"]),
                            step="post", alpha=0.15, color=col)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if tab is not None:
        _draw_risk_table(ax, tab, risk_table, [c["name"] for c in curves])
    return fig


def _series_colors():
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _floats(values):
    """list that may hold None (no estimate) -> float array with nan"""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _curve_axes(risk_table, base_height=4.5):
    """main axes, plus an at-risk table axes under it when risk_table is given"""
    groups = (risk_table or {}).get("groups") or []
    tps = (risk_table or {}).get("time_points") or []
    if not groups or not tps:
        fig, ax = plt.subplots(figsize=(6, base_height))
        return fig, ax, None
    tab_h = 0.28 * len(groups) + 0.5
    fig, (ax, tab) = plt.subplots(
        2, 1, figsize=(6, base_height + tab_h),
        gridspec_kw={"height_ratios": [base_height, tab_h]})
    return fig, ax, tab


def _draw_risk_table(ax, tab, risk_table, names):
    """number at risk per group at each time point, columns under the x ticks"""
    from matplotlib import transforms
    tps = list(risk_table["time_points"])
    groups = risk_table["groups"]
    colors = _series_colors()
    ax.set_xticks(tps)
    tab.set_xlim(ax.get_xlim())
    n = len(groups)
    # marker just left of the axes, in points so it never hits the labels
    mark = transforms.offset_copy(
        transforms.blended_transform_factory(tab.transAxes, tab.transData),
        fig=tab.figure, x=-18, units="points")
    for row, g in enumerate(groups):
        y = n - 1 - row
        for t, v in zip(tps, g["at_risk"]):
            tab.text(t, y, str(v), ha="center", va="center", fontsize=8)
        if g["name"] in names:
            col = colors[names.index(g["name"]) % len(colors)]
            tab.plot([0], [y], marker="s", ms=6, color=col, transform=mark,
                     clip_on=False)
    tab.set_ylim(-0.6, n - 0.4)
    tab.set_yticks(range(n))
    tab.set_yticklabels([g["name"] for g in groups][::-1], fontsize=8)
    tab.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    tab.tick_params(axis="y", length=0, pad=26)
    tab.grid(False)
    for side in ("top", "right", "bottom", "left"):
        tab.spines[side].set_visible(False)
    tab.set_title("Number at risk", fontsize=8, loc="left")


def cif_plot(curves, naive=None, title="Cumulative incidence", risk_table=None):
    """curves: list of {name, timeline, cif, lower, upper} (epi.cumulative_incidence).
    naive: optional list of {name, timeline, risk}, the 1 minus Kaplan-Meier
    estimate that censors competing events, drawn dashed in the same colour
    so the overestimate is visible. risk_table as in km_plot."""
    fig, ax, tab = _curve_axes(risk_table)
    colors = _series_colors()
    names = [c["name"] for c in curves]
    ymax = 0.0
    for i, c in enumerate(curves):
        col = colors[i % len(colors)]
        label = c["name"] if len(curves) > 1 or naive else "CIF"
        ax.step(c["timeline"], c["cif"], where="post", lw=2, color=col,
                label=f"{label} (Aalen-Johansen)" if naive else label)
        ymax = max(ymax, np.nanmax(_floats(c["cif"])))
        if c.get("lower"):
            up = _floats(c["upper"])
            ax.fill_between(c["timeline"], _floats(c["lower"]), up,
                            step="post", alpha=0.15, color=col)
            ymax = max(ymax, np.nanmax(up))
    for j, nv in enumerate(naive or []):
        i = names.index(nv["name"]) if nv["name"] in names else j
        col = colors[i % len(colors)]
        ax.step(nv["timeline"], nv["risk"], where="post", lw=1.5, ls="--", color=col,
                label=f"{nv['name']} (1 minus KM, competing events censored)")
        ymax = max(ymax, np.nanmax(_floats(nv["risk"])))
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative incidence")
    ax.set_ylim(0, min(1.02, max(ymax * 1.1, 0.05)))
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    if tab is not None:
        _draw_risk_table(ax, tab, risk_table, names)
    return fig


def calibration_hl_plot(prob_pred, obs, expected=None, title="Calibration (Hosmer-Lemeshow)"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
    ax.plot(prob_pred, obs, "o-", lw=2, label="Observed")
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig


def teaching_plot(result, title=None):
    """Teaching mode (teaching.leakage_demo result): per arm, the AUROC an
    analyst would report next to the honest lockbox AUROC, joined by a line so
    the gap stands out. Dots are means over the repeats, bars span min to max.
    Arms run from the correct one at the top to the most optimistic at the
    bottom. The small right panel is the lockbox calibration slope (1 is
    ideal)."""
    import textwrap
    from matplotlib.lines import Line2D

    reported_c, lockbox_c, link_c = "#eb6834", "#2a78d6", "#9a9a96"

    def _stat(a, key, which="mean"):
        v = (a.get(key) or {}).get(which)
        return None if v is None else float(v)

    def _xerr(a, key):
        m, lo, hi = (_stat(a, key, w) for w in ("mean", "min", "max"))
        if m is None or lo is None or hi is None:
            return None
        return [[max(m - lo, 0.0)], [max(hi - m, 0.0)]]

    arms = list(result.get("arms") or [])
    first = [a for a in arms if a.get("arm") == "correct"]
    rest = [a for a in arms if a.get("arm") != "correct"]
    rest.sort(key=lambda a: -np.inf if _stat(a, "optimism") is None
              else _stat(a, "optimism"))
    arms = first + rest
    ys = np.arange(len(arms))[::-1]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 1.4 + 0.62 * max(len(arms), 1)), sharey=True,
        gridspec_kw={"width_ratios": [3, 1.2]})

    lows = [0.5]
    for y, a in zip(ys, arms):
        rep, lock = _stat(a, "reported_auroc"), _stat(a, "lockbox_auroc")
        if rep is not None and lock is not None:
            ax.plot([lock, rep], [y, y], color=link_c, lw=2, zorder=1)
        if lock is not None:
            ax.errorbar(lock, y, xerr=_xerr(a, "lockbox_auroc"), fmt="o",
                        ms=8, color=lockbox_c, ecolor=lockbox_c, capsize=3,
                        lw=1.2, zorder=3, mec="white", mew=1.5)
            lows.append(_stat(a, "lockbox_auroc", "min") or lock)
        if rep is not None:
            ax.errorbar(rep, y, xerr=_xerr(a, "reported_auroc"), fmt="o",
                        ms=8, color=reported_c, ecolor=reported_c, capsize=3,
                        lw=1.2, zorder=4, mec="white", mew=1.5)
            lows.append(_stat(a, "reported_auroc", "min") or rep)
        opt = _stat(a, "optimism")
        if opt is not None:
            ax.text(1.005, y, f"{opt:+.2f}", va="center", ha="left",
                    fontsize=9, color="#333333",
                    fontweight="bold" if opt >= 0.05 else "normal")

    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlim(max(0.0, min(lows) - 0.03), 1.06)
    ax.set_yticks(ys, labels=[textwrap.fill(str(a.get("label", a.get("arm"))), 32)
                              for a in arms])
    ax.set_ylim(-0.7, len(arms) - 0.3)
    ax.set_xlabel("AUROC (dotted line = coin toss, number = reported minus lockbox)")
    ax.grid(axis="y", alpha=0)
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=lockbox_c,
                      label="Lockbox (honest)"),
               Line2D([], [], ls="", marker="o", ms=8, color=reported_c,
                      label="Reported (what a paper would say)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=2, fontsize=8, frameon=False, borderaxespad=0.2)

    # calibration slope on the lockbox, values far off the scale are pinned
    # to the edge and labelled with their real value
    cap_lo, cap_hi = -0.5, 3.0
    slopes = [_stat(a, "calibration_slope") for a in arms]
    shown = [min(max(v, cap_lo), cap_hi) for a in arms
             for v in (_stat(a, "calibration_slope", w) for w in ("min", "max"))
             if v is not None]
    ax2.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    for y, a, s in zip(ys, arms, slopes):
        if s is None:
            continue
        pinned = min(max(s, cap_lo), cap_hi)
        err = _xerr(a, "calibration_slope") if pinned == s else None
        ax2.errorbar(pinned, y, xerr=err, fmt="D", ms=6, color="#52514e",
                     ecolor="#52514e", capsize=3, lw=1.2)
        if pinned != s:
            ax2.text(pinned, y + 0.3, f"{s:.1f}", ha="center", fontsize=8,
                     color="#333333")
    ax2.set_xlim(min([0.0] + shown) - 0.1, max([1.5] + shown) + 0.15)
    ax2.set_xlabel("Calibration slope\n(lockbox, 1 = ideal)")
    ax2.grid(axis="y", alpha=0)
    ax2.tick_params(axis="y", length=0)

    k = len(result.get("seeds") or [])
    label = result.get("model_label") or result.get("model") or "model"
    fig.suptitle(title or f"Reported vs honest AUROC, {label} "
                          f"({k} repeat{'' if k == 1 else 's'})")
    return fig


def flow_plot(flow, n_start, title="Cohort"):
    """STROBE-style cohort flow diagram from cohort.apply_filters. A start box
    with n_start, then per step a main box with the rows left (n_after) and a
    side box with the step label and the rows it removed. n_start defaults to
    the first step's n_before."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch

    flow = list(flow or [])
    if n_start is None:
        n_start = flow[0]["n_before"] if flow else 0
    line_h, pad = 0.2, 0.18          # data units per text line, box padding
    main_x, main_w = 2.6, 4.2        # centre and width of the main column
    side_x, side_w = 7.5, 4.6

    def height(text):
        return line_h * (text.count("\n") + 1) + 2 * pad

    def box(ax, x, top, w, text, fc):
        h = height(text)
        ax.add_patch(FancyBboxPatch((x - w / 2, top - h), w, h, fc=fc, ec="0.35", lw=1,
                                    boxstyle="round,pad=0,rounding_size=0.06"))
        ax.text(x, top - h / 2, text, ha="center", va="center", fontsize=9)
        return top - h

    def arrow(ax, xy_from, xy_to):
        ax.annotate("", xy=xy_to, xytext=xy_from,
                    arrowprops={"arrowstyle": "-|>", "color": "0.35", "lw": 1.1,
                                "shrinkA": 0, "shrinkB": 0})

    # lay everything out top-down first, so the figure height fits it
    items, y = [], 0.0
    start_text = f"Records in file\nn = {int(n_start):,}"
    items.append(("main", y, start_text, "#eef3fb"))
    y -= height(start_text)
    for i, step in enumerate(flow):
        label = textwrap.fill(str(step.get("label") or f"Step {step.get('step', i + 1)}"), 40)
        side_text = f"{label}\nRemoved: n = {int(step.get('n_removed', 0)):,}"
        gap = max(height(side_text) + 0.35, 0.9)
        items.append(("side", y - gap / 2, side_text, "white"))
        items.append(("link", y, y - gap, None))
        y -= gap
        last = i == len(flow) - 1
        main_text = (f"{'Final cohort' if last else 'Remaining'}\n"
                     f"n = {int(step.get('n_after', 0)):,}")
        items.append(("main", y, main_text, "#eef3fb" if not last else "#dfeadb"))
        y -= height(main_text)
    total = -y

    fig, ax = plt.subplots(figsize=(8, max(1.2, 0.85 * total + 0.6)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-total - 0.05, 0.05)
    ax.axis("off")
    for kind, y_at, what, fc in items:
        if kind == "main":                  # y_at = top of the box
            box(ax, main_x, y_at, main_w, what, fc)
        elif kind == "side":                # y_at = centre of the box
            box(ax, side_x, y_at + height(what) / 2, side_w, what, fc)
            arrow(ax, (main_x, y_at), (side_x - side_w / 2, y_at))
        else:                               # link from y_at down to what
            arrow(ax, (main_x, y_at), (main_x, what))
    ax.set_title(title)
    return fig
