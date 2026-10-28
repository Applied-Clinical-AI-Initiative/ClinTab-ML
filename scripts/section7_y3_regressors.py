"""section7_y3_regressors.py
Protocol Section 7, regressor panel for Y3 (total hospital length of stay).
Runs the five regressors through the real app routes with default settings
(random 70/15/15 split, R2-scored grid search) in a fresh session, so the
Arm D session used for Experiment 4 is left untouched.

Uses Flask's in-process test client: no server, nothing leaves the machine.
Prints only status lines and model names. Writes a figure (feature
importance for each regressor plus a held-out test metrics panel) and an
aggregate metrics table.

Usage:
    python scripts/section7_y3_regressors.py --csv /path/to/cohort.csv --out-dir /path
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import create_app  # noqa: E402
from clintab import plots  # noqa: E402  (selects the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402

OUTCOME = "Y3_length_of_stay"
MODELS = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForestRegressor"]
# Everything outside the protocol's predictor list, plus anything that is or
# is derived from an outcome (length_of_stay and days_op_to_discharge encode
# Y3 itself; Y1 is a post-operative outcome).
EXCLUDE = ["admyr", "elective", "emergency", "operyr", "inout", "race",
           "ethnicity_hispanic", "bmi_cat", "unplanned_readmission",
           "length_of_stay", "days_op_to_discharge", "Y1_readmission"]
DPI = 600
TOP = 10


def check(resp, label):
    if resp.status_code != 200:
        msg = ""
        try:
            msg = str(resp.get_json().get("error", ""))[:150]
        except Exception:
            pass
        raise SystemExit(f"{label} failed: HTTP {resp.status_code} {msg}")
    print(f"{label}: ok", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--outcome", default=OUTCOME)
    ap.add_argument("--exclude", nargs="*", default=EXCLUDE)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = create_app().test_client()
    with open(args.csv, "rb") as fh:
        r = client.post("/api/upload", data={"file": (fh, Path(args.csv).name)},
                        content_type="multipart/form-data")
    check(r, "upload")  # body not printed: it contains a data preview

    check(client.post("/api/confirm", json={
        "types": {}, "missing": {}, "method": "random",
        "ratios": [0.7, 0.15, 0.15], "smote": False, "seed": 42}), "confirm split")

    check(client.post("/api/train", json={
        "outcome": args.outcome, "exclude": args.exclude, "confounders": [],
        "models": MODELS, "grids": {}, "scoring": "r2",
        "grid_search": True, "smote": False, "threshold": 0.5}), "train config")

    print("training five regressors (RandomForestRegressor is slow) ...", flush=True)
    stream = client.get("/api/train/stream")
    importance, saved_to_model = {}, {}
    for line in stream.get_data(as_text=True).splitlines():
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("event") == "model_done":
            importance[ev["model"]] = ev["importance"]
            saved_to_model[ev["saved_as"]] = ev["model"]
            print(f"  trained {ev['model']}", flush=True)
        elif ev.get("event") == "model_error":
            print(f"  FAILED {ev['model']}: {str(ev.get('error'))[:150]}", flush=True)
    if not saved_to_model:
        raise SystemExit("no models trained")

    r = client.post("/api/test", data={"models": list(saved_to_model),
                                       "source": "test", "threshold": "0.5"})
    check(r, "held-out test evaluation")
    rows = []
    for entry in r.get_json()["models"]:
        m = entry.get("metrics")
        if m:
            rows.append({"Model": saved_to_model.get(entry["name"], entry["name"]),
                         "Test MAE": m["MAE"], "Test RMSE": m["RMSE"],
                         "Test R2": m["R2"], "Test N": m["N"]})
    metrics = pd.DataFrame(rows).set_index("Model").reindex(
        [m for m in MODELS if m in importance]).reset_index()
    metrics.to_csv(out / "table_y3_regressor_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    flat = list(axes.ravel())
    for ax, name in zip(flat, [m for m in MODELS if m in importance]):
        recs = importance[name][:TOP][::-1]
        ax.barh([plots._short_label(x["feature"]) for x in recs],
                [x["importance"] for x in recs], color="#3b7dd8")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0)
    last = flat[5]
    last.axis("off")
    cell = [[r["Model"], f"{r['Test MAE']:.2f}", f"{r['Test RMSE']:.2f}", f"{r['Test R2']:.3f}"]
            for _, r in metrics.iterrows()]
    tbl = last.table(cellText=cell, colLabels=["Model", "MAE", "RMSE", "R²"],
                     loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    last.set_title("Held-out test set", fontsize=11)
    fig.suptitle("Total hospital length of stay: regressor panel", fontsize=13)
    fig.savefig(out / "figure_y3_regressor_panel.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote figure_y3_regressor_panel.png and table_y3_regressor_metrics.csv")


if __name__ == "__main__":
    main()
