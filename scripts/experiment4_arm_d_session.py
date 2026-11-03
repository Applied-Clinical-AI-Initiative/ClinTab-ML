"""experiment4_arm_d_session.py
Protocol Section 5: generates artifacts for reporting recoverability assessment.
Runs the Arm D pipeline end to end
through the real app routes on the full cohort, then writes out the two
artifacts Experiment 4 scores: the generated methods paragraph and the
TRIPOD+AI coverage report, both read from the session's analysis log.

Uses Flask's in-process test client, so no server is started and nothing
leaves the machine. Deliberately prints only status codes, event names and
the two aggregate reports -- never a response body that could contain rows
(the upload response includes a data preview, so it is never printed).

The run mirrors a complete default workflow: stratified 70/15/15 split,
SMOTE on the training fold, ROC-scored grid search for three models,
held-out test evaluation, and a Hosmer-Lemeshow check.

Usage:
    python scripts/experiment4_arm_d_session.py \
        --csv /path/to/cohort.csv --out-dir /path/to/outputs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import create_app  # noqa: E402

OUTCOME = "Y1_readmission"
MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting"]
# Everything outside the protocol's Section 1.3 predictor list, plus any
# column that is the outcome or derived from it / from other outcomes.
EXCLUDE = ["admyr", "elective", "emergency", "operyr", "inout", "race",
           "ethnicity_hispanic", "bmi_cat", "unplanned_readmission",
           "length_of_stay", "days_op_to_discharge", "Y3_length_of_stay"]


def check(resp, label):
    if resp.status_code != 200:
        msg = ""
        try:
            msg = str(resp.get_json().get("error", ""))[:150]
        except Exception:
            pass
        raise SystemExit(f"{label} failed: HTTP {resp.status_code} {msg}")
    print(f"{label}: ok")


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
        r = client.post("/api/upload",
                        data={"file": (fh, Path(args.csv).name)},
                        content_type="multipart/form-data")
    check(r, "upload")  # body intentionally not printed: it contains a preview

    r = client.post("/api/confirm", json={
        "types": {}, "missing": {}, "method": "stratified",
        "stratify_col": args.outcome, "ratios": [0.7, 0.15, 0.15],
        "smote": True, "seed": 42})
    check(r, "confirm split")

    r = client.post("/api/train", json={
        "outcome": args.outcome, "exclude": args.exclude, "confounders": [],
        "models": MODELS, "grids": {}, "scoring": "roc",
        "grid_search": True, "smote": True, "threshold": 0.5})
    check(r, "train config")

    print("training (this can take a while) ...", flush=True)
    stream = client.get("/api/train/stream")
    saved = []
    for line in stream.get_data(as_text=True).splitlines():
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        kind = ev.get("event")
        if kind == "model_done":
            saved.append(ev["saved_as"])
            print(f"  trained {ev['model']}")
        elif kind == "model_error":
            print(f"  FAILED {ev['model']}: {str(ev.get('error'))[:150]}")
    if not saved:
        raise SystemExit("no models trained")

    r = client.post("/api/test", data={"models": saved, "source": "test",
                                       "threshold": "0.5"})
    check(r, "held-out test evaluation")

    r = client.post("/api/epi/hl", json={"model": saved[0]})
    check(r, "Hosmer-Lemeshow check")

    methods = client.get("/api/report/methods").get_json()["text"]
    tripod = client.get("/api/report/tripod").get_json()
    (out / "methods_paragraph.txt").write_text(methods)
    (out / "tripod_report.json").write_text(json.dumps(tripod, indent=2))

    print("\n=== METHODS PARAGRAPH (verbatim) ===")
    print(methods)
    print("\n=== TRIPOD+AI COVERAGE ===")
    print(tripod["summary"])
    for it in tripod["items"]:
        print(f"  {it['status']:<15} {it['id']}")
    print(f"\nWrote methods_paragraph.txt and tripod_report.json to {out}")


if __name__ == "__main__":
    main()
