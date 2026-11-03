"""experiment2a_python_side.py
Protocol Section 3: computes ClinTAB-ML's own numbers for the numerical
equivalence table, and exports the underlying (synthetic, bundled) data so
experiment2b_r_side.R can compute the same quantities independently in R.

Uses the bundled synthetic dataset at Data/sample_clinical.csv for numerical
validation against reference implementations.

Usage:
    python scripts/experiment2a_python_side.py --out-dir /path/for/intermediate/files
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clintab import ml as clintab_ml  # noqa: E402
from clintab import epi as clintab_epi  # noqa: E402
from clintab import spline as clintab_spline  # noqa: E402

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "Data" / "sample_clinical.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SAMPLE_CSV)
    results = {}

    # ------------------------------------------------------------------
    # Row 1: ml.py metrics vs scikit-learn called directly
    # ------------------------------------------------------------------
    feat_cols = ["age", "bmi", "sex", "asa_class", "length_of_stay", "diabetes"]
    outcome = "mortality_30d"
    pipe, info = clintab_ml.train_one_model(
        "LogisticRegression", df, df, outcome, feat_cols, "binary",
        do_grid_search=False)
    y, _ = clintab_ml.encode_outcome(df[outcome], "binary")
    scores = clintab_ml.predict_scores(pipe, df[feat_cols], "binary")
    clintab_metrics, _ = clintab_ml.binary_metrics(y, scores)
    sklearn_metrics = {
        "AUROC": round(float(roc_auc_score(y, scores)), 4),
        "AUPRC": round(float(average_precision_score(y, scores)), 4),
        "Brier": round(float(brier_score_loss(y, scores)), 4),
    }
    results["ml_metrics"] = {"clintab": clintab_metrics, "reference_sklearn": sklearn_metrics}

    # ------------------------------------------------------------------
    # Row 2: epi.py 2x2 panel vs R epitools -- fixed illustrative table,
    # the same one already used elsewhere in the paper (20/80/10/90).
    # ------------------------------------------------------------------
    twobytwo = clintab_epi.two_by_two(20, 80, 10, 90)
    results["twobytwo"] = twobytwo
    pd.DataFrame([{"a": 20, "b": 80, "c": 10, "d": 90}]).to_csv(out / "twobytwo_table.csv", index=False)

    # ------------------------------------------------------------------
    # Row 3: epi.py 2x2 with a zero cell -- Haldane-Anscombe correction
    # ------------------------------------------------------------------
    twobytwo_zero = clintab_epi.two_by_two(0, 80, 10, 90)
    results["twobytwo_zero_cell"] = twobytwo_zero
    pd.DataFrame([{"a": 0, "b": 80, "c": 10, "d": 90}]).to_csv(out / "twobytwo_zero_table.csv", index=False)

    # ------------------------------------------------------------------
    # Row 4: epi.py Cox vs R survival::coxph
    # ------------------------------------------------------------------
    cox = clintab_epi.cox_ph(df, "surv_time", "death_event", ["age", "diabetes"])
    results["cox"] = cox
    df[["surv_time", "death_event", "age", "diabetes"]].dropna().to_csv(
        out / "cox_data.csv", index=False)

    # ------------------------------------------------------------------
    # Row 5: epi.py Hosmer-Lemeshow vs R ResourceSelection::hoslem.test
    # ------------------------------------------------------------------
    hl = clintab_epi.hosmer_lemeshow(y, scores, g=10)
    results["hosmer_lemeshow"] = hl
    pd.DataFrame({"y_true": y, "y_prob": scores}).to_csv(out / "hl_data.csv", index=False)

    # ------------------------------------------------------------------
    # Row 6: spline.py RCS vs R rms::lrm at matched knots
    # ------------------------------------------------------------------
    rcs = clintab_spline.fit_rcs(df, "age", "sex", n_knots=4)
    results["rcs"] = {"knots": rcs["knots"], "aic": rcs["aic"],
                      "x": rcs["x"], "logodds": rcs["logodds"]}
    df[["age", "sex"]].dropna().to_csv(out / "rcs_data.csv", index=False)
    pd.DataFrame({"x": rcs["x"], "logodds": rcs["logodds"]}).to_csv(
        out / "rcs_python_curve.csv", index=False)
    with open(out / "rcs_knots.json", "w") as f:
        json.dump(rcs["knots"], f)

    with open(out / "python_side_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Wrote intermediate data + Python-side results to {out}")
    print("Now run: Rscript scripts/experiment2b_r_side.R", str(out))


if __name__ == "__main__":
    main()
