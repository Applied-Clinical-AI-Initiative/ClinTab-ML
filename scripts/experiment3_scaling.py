"""experiment3_scaling.py
Protocol Section 4: times the full local pipeline (profile, split, train
three models, evaluate, one spline fit, one epi calculation, log write,
generate both reports) at increasing cohort sizes, entirely through the
same modules the web app uses. Records wall clock per stage and peak RSS.

Run entirely on your own machine against your own real_cohort.csv. Only
scaling.csv (timings and memory numbers, no patient-level data) is meant to
leave your machine.

Usage:
    python scripts/experiment3_scaling.py \
        --csv /path/outside/repo/real_cohort.csv \
        --outcome Y1_readmission \
        --features age bmi sex asa_class diabetes ... \
        --sizes 1000 10000 100000 1000000 \
        --repeats 5 \
        --out scaling.csv
"""
import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clintab import ml as clintab_ml  # noqa: E402
from clintab import spline as clintab_spline  # noqa: E402
from clintab import epi as clintab_epi  # noqa: E402
from clintab import analysis_log  # noqa: E402
from clintab import methods_text, tripod_report  # noqa: E402


def peak_rss_mb():
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    result = fn(*a, **kw)
    return result, time.perf_counter() - t0


def run_once(df, outcome, feat_cols, spline_predictor, session_id):
    stages = {}

    (df_train, df_temp), t = timed(
        train_test_split, df, test_size=0.3, stratify=df[outcome], random_state=42)
    stages["split_1"] = t
    (df_val, df_test), t = timed(
        train_test_split, df_temp, test_size=0.5, stratify=df_temp[outcome], random_state=42)
    stages["split_2"] = t

    analysis_log.log_action(session_id, "upload", inputs={"n_rows": len(df)}, outputs={})

    fitted = {}
    for name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
        (pipe, info), t = timed(
            clintab_ml.train_one_model, name, df_train, df_val, outcome, feat_cols, "binary",
            scoring="roc", use_smote=True, do_grid_search=True)
        stages[f"train_{name}"] = t
        fitted[name] = pipe

    for name, pipe in fitted.items():
        (metrics, coords), t = timed(
            clintab_ml.evaluate, pipe, df_test, outcome, feat_cols, "binary")
        stages[f"evaluate_{name}"] = t

    (spline_result, t) = timed(
        clintab_spline.fit_rcs, df, spline_predictor, outcome, 4)
    stages["spline_fit"] = t

    y = df[outcome].values
    a = int(((y == 1)).sum() // 2)
    b = int((y == 1).sum() - a)
    c = int(((y == 0)).sum() // 2)
    d = int((y == 0).sum() - c)
    (_, t) = timed(clintab_epi.two_by_two, a, b, c, d)
    stages["epi_2x2"] = t

    scores = clintab_ml.predict_scores(fitted["GradientBoosting"], df_test[feat_cols], "binary")
    (_, t) = timed(clintab_epi.hosmer_lemeshow, df_test[outcome].values, scores)
    stages["epi_hosmer_lemeshow"] = t

    (_, t) = timed(analysis_log.read_log, session_id)
    stages["log_read"] = t

    (_, t) = timed(methods_text.generate_methods_text, session_id)
    stages["methods_text"] = t
    (_, t) = timed(tripod_report.generate_tripod_report, session_id)
    stages["tripod_report"] = t

    return stages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outcome", default="Y1_readmission")
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--spline-predictor", default="age")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 100_000, 1_000_000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="scaling.csv")
    args = ap.parse_args()

    df_full = pd.read_csv(args.csv)
    print(f"hardware: {platform.processor() or platform.machine()}, "
          f"{platform.system()} {platform.release()}, "
          f"python {platform.python_version()}")

    rows = []
    for n in args.sizes:
        resampled = n > len(df_full)
        if resampled:
            print(f"n={n} exceeds the cohort ({len(df_full)}): bootstrap resampling with "
                  "replacement, timing only", file=sys.stderr)
        for rep in range(args.repeats):
            sub = df_full.sample(n=n, random_state=rep, replace=resampled) \
                if n != len(df_full) else df_full
            sid = f"scaling_{n}_{rep}"
            stages = run_once(sub, args.outcome, args.features, args.spline_predictor, sid)
            for stage, seconds in stages.items():
                rows.append({"n": n, "repeat": rep, "stage": stage, "seconds": seconds,
                            "peak_rss_mb": peak_rss_mb(), "resampled": resampled})
            print(f"n={n} repeat={rep} total={sum(stages.values()):.1f}s "
                  f"peak_rss={peak_rss_mb():.0f}MB")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows to {args.out}")
    print("Record your exact CPU model / core count / RAM manually alongside this file, "
          "platform.processor() is not reliable on Apple Silicon.")


if __name__ == "__main__":
    main()
