"""make_inclusion_flow.py
Draws the cohort inclusion flow figure (protocol Section 1.1 / Section 7)
from the "Inclusion flow" aggregate-count output of build_cohort.py.
Save that output to a text file and pass its path with --log.

Usage:
    python scripts/make_inclusion_flow.py --log counts.txt --out inclusion_flow.png

counts.txt can be the pasted Terminal block as-is; only lines of the form
"  key: value" are read.
"""
import argparse
import ast
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DPI = 600


def parse(path):
    vals = {}
    for line in open(path):
        m = re.match(r"^\s*([a-z_0-9]+):\s*(.+?)\s*$", line)
        if m:
            key, raw = m.groups()
            try:
                vals[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                pass
    return vals


def build_steps(v):
    """[(box label, n_remaining, exclusion label or None, n_excluded or None)]"""
    steps = []
    n = v["pooled_all_files"]
    steps.append(("Records in the four NSQIP extracts\n(TKA, THA, TKA revision, THA revision)", n, None, None))
    chain = [
        ("after_excluding_revision_files", "Revision cases", "revision_rows_excluded"),
        ("after_excluding_bilateral", "Bilateral cases (concurrent CPT)", "bilateral_rows_excluded"),
        ("after_year_filter_2021_2023", "Operative year outside 2021–2023", "dropped_rows_outside_2021_2023"),
        ("after_age_filter", "Age under 18", None),
        ("after_elective_filter", "Non-elective", None),
        ("after_emergency_exclusion", "Emergent", None),
        ("after_missing_outcome_exclusion", "Missing primary outcome", None),
    ]
    prev = n
    for key, label, excl_key in chain:
        if key not in v:
            continue
        excluded = v[excl_key] if excl_key and excl_key in v else prev - v[key]
        steps.append((f"Remaining", v[key], f"{label}", excluded))
        prev = v[key]
    return steps


def draw(steps, out):
    n_boxes = len(steps)
    fig_h = 1.15 * n_boxes + 1
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_boxes)
    ax.axis("off")
    bw, bh = 0.46, 0.62
    main_x, side_x = 0.26, 0.755
    for i, (label, n, excl_label, excl_n) in enumerate(steps):
        y = n_boxes - 1 - i + 0.19
        if i == len(steps) - 1:
            text = f"Final analytic cohort\nN = {n:,}"
            fc = "#dbe9f6"
        else:
            text = f"{label}\nN = {n:,}"
            fc = "#f4f6f9"
        ax.add_patch(FancyBboxPatch((main_x - bw / 2, y), bw, bh, boxstyle="round,pad=0.01",
                                    fc=fc, ec="#333333", lw=1.2))
        ax.text(main_x, y + bh / 2, text, ha="center", va="center", fontsize=10)
        if i > 0:
            ax.annotate("", xy=(main_x, y + bh), xytext=(main_x, y + bh + 0.19 * 2 + 0.0),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"))
            ex_y = y + bh + 0.19 - bh / 2 + 0.02
            ax.add_patch(FancyBboxPatch((side_x - 0.235, ex_y), 0.47, bh * 0.82,
                                        boxstyle="round,pad=0.01", fc="#fbeeee",
                                        ec="#8a3b3b", lw=1.0))
            ax.text(side_x, ex_y + bh * 0.41,
                    f"Excluded: {excl_label}\nn = {excl_n:,}", ha="center", va="center", fontsize=9)
            ax.annotate("", xy=(side_x - 0.235, ex_y + bh * 0.41),
                        xytext=(main_x, ex_y + bh * 0.41),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#8a3b3b"))
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="inclusion_flow.png")
    args = ap.parse_args()
    vals = parse(args.log)
    if "pooled_all_files" not in vals:
        raise SystemExit("could not find 'pooled_all_files' in the text; paste the whole Inclusion flow block")
    draw(build_steps(vals), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
