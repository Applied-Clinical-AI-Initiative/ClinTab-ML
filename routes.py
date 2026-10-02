"""routes.py
Every HTTP endpoint. This file only does HTTP: parse the request, call into
clintab/ (ml.py / stats.py / spline.py / epi.py / plots.py) for the real work,
return JSON (or an SSE stream / a file download). No statistics or sklearn
logic lives here.
"""
import io
import json
import math
import re

import joblib
import numpy as np
import pandas as pd
from flask import (
    Blueprint, request, jsonify, session, Response, send_file,
    send_from_directory, current_app,
)

from clintab import store
from clintab import stats
from clintab import ml
from clintab import plots
from clintab import spline as spline_mod
from clintab import epi
from clintab import analysis_log
from clintab import methods_text
from clintab import tripod_report

bp = Blueprint("api", __name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def current_sid():
    return session.get("sid")

def require_session():
    sid = current_sid()
    if not store.session_exists(sid):
        return None
    return sid

def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))[:60]


def _sanitize(o):
    """Recursively make a payload strict-JSON safe: NaN/Infinity -> null, and
    numpy scalars/arrays -> plain Python. Browsers' JSON.parse rejects the NaN
    and Infinity tokens that Python's json module emits by default (e.g. from
    a metric that's undefined for a degenerate train/val split)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
@bp.route("/")
def index():
    return send_from_directory(current_app.static_folder, "index.html")


# ===========================================================================
# 1. DATA UPLOAD
# ===========================================================================
@bp.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400

    sid = store.new_session_id()
    store.session_dir(sid, create=True)
    raw_path = store.session_path(sid, "raw.csv")
    f.save(raw_path)

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    cols = stats.detect_column_types(df)
    meta = {
        "filename": f.filename,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns_detected": cols,
        "stage": "uploaded",
    }
    store.save_meta(sid, meta)
    session["sid"] = sid
    analysis_log.log_action(sid, "upload", inputs={"filename": f.filename},
                            outputs={"n_rows": meta["n_rows"], "n_cols": meta["n_cols"]})

    preview = df.head(20).where(pd.notnull(df.head(20)), None).to_dict(orient="records")
    return jsonify({
        "session_id": sid,
        "filename": f.filename,
        "n_rows": meta["n_rows"],
        "n_cols": meta["n_cols"],
        "columns": cols,
        "preview": preview,
        "preview_cols": [str(c) for c in df.columns],
    })


@bp.route("/api/confirm", methods=["POST"])
def confirm():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session. Upload a file first."}), 400
    body = request.get_json(force=True)

    coltypes = body.get("types", {})          # {col: binary|categorical|continuous|date|exclude}
    missing = body.get("missing", {})         # {col: include|zero|remove}
    ratios = body.get("ratios", [0.70, 0.15, 0.15])
    method = body.get("method", "random")      # random | stratified
    stratify_col = body.get("stratify_col") or None
    smote_pref = bool(body.get("smote", False))
    seed = int(body.get("seed", 42))

    df = pd.read_csv(store.session_path(sid, "raw.csv"))

    # apply missing-value handling, then drop excluded columns
    df = stats.apply_missing_handling(df, missing)
    excluded = [c for c, t in coltypes.items() if t == "exclude"]
    df = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
    kept_types = {c: t for c, t in coltypes.items()
                  if t != "exclude" and c in df.columns}

    # save cleaned full dataset
    df.to_csv(store.session_path(sid, "full.csv"), index=False)

    # split -------------------------------------------------------------
    r = [float(x) for x in ratios]
    r = [x / sum(r) for x in r]
    from sklearn.model_selection import train_test_split
    strat = df[stratify_col] if (method == "stratified" and stratify_col in df.columns) else None
    try:
        train, temp = train_test_split(df, test_size=(r[1] + r[2]),
                                       random_state=seed, stratify=strat)
        strat2 = temp[stratify_col] if strat is not None else None
        val_frac = r[1] / (r[1] + r[2])
        val, test = train_test_split(temp, test_size=(1 - val_frac),
                                     random_state=seed, stratify=strat2)
    except ValueError:
        # stratification can fail on tiny/rare classes -> fall back to random
        train, temp = train_test_split(df, test_size=(r[1] + r[2]), random_state=seed)
        val, test = train_test_split(temp, test_size=(r[2] / (r[1] + r[2])),
                                     random_state=seed)
        method = "random (stratify fell back)"

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(store.session_path(sid, f"{name}.csv"), index=False)

    # SMOTE applicability hint
    smote_hint = None
    if stratify_col and stratify_col in df.columns:
        frac = stats.minority_fraction(df[stratify_col])
        if frac is not None:
            smote_hint = {"column": stratify_col, "minority_fraction": round(frac, 4),
                          "suggest_smote": frac < 0.30}

    meta = store.load_meta(sid)
    meta.update({
        "stage": "confirmed",
        "coltypes": kept_types,
        "missing": missing,
        "excluded": excluded,
        "split": {"ratios": r, "method": method, "stratify_col": stratify_col, "seed": seed},
        "smote_pref": smote_pref,
        "n_rows_clean": int(len(df)),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "smote_hint": smote_hint,
    })
    store.save_meta(sid, meta)
    analysis_log.log_action(sid, "confirm", inputs={"method": method, "ratios": r,
                            "stratify_col": stratify_col, "smote_pref": smote_pref,
                            "missing": missing, "smote_hint": smote_hint},
                            outputs={"n_train": len(train), "n_val": len(val), "n_test": len(test)})
    return jsonify({"ok": True, "n_train": len(train), "n_val": len(val),
                    "n_test": len(test), "method": method, "smote_hint": smote_hint})


# ===========================================================================
# 2. DATA SUMMARY
# ===========================================================================
@bp.route("/api/summary")
def summary():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    meta = store.load_meta(sid)
    if meta.get("stage") != "confirmed":
        return jsonify({"error": "Confirm the dataset first."}), 400
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    return jsonify(stats.summarize(df, meta["coltypes"]))


@bp.route("/api/session")
def session_info():
    sid = current_sid()
    if not store.session_exists(sid):
        return jsonify({"active": False})
    return jsonify({"active": True, "meta": store.load_meta(sid)})


@bp.route("/api/sessions")
def list_sessions():
    sessions = sorted(store.list_sessions(), key=lambda m: m.get("updated") or 0, reverse=True)
    out = [{"session_id": m["session_id"], "filename": m.get("filename"),
            "n_rows": m.get("n_rows"), "stage": m.get("stage")} for m in sessions]
    return jsonify({"sessions": out, "active": current_sid()})


@bp.route("/api/sessions/<sid>/activate", methods=["POST"])
def activate_session(sid):
    if not store.session_exists(sid):
        return jsonify({"error": "Dataset not found."}), 404
    session["sid"] = sid
    return jsonify({"ok": True, "meta": store.load_meta(sid)})


@bp.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session_route(sid):
    if not store.session_exists(sid):
        return jsonify({"error": "Dataset not found."}), 404
    store.delete_session(sid)
    if current_sid() == sid:
        session.pop("sid", None)
    return jsonify({"ok": True})


@bp.route("/api/sessions", methods=["DELETE"])
def clear_all_sessions():
    store.delete_all_sessions()
    session.pop("sid", None)
    return jsonify({"ok": True})


# ===========================================================================
# 3. MODEL TRAINING
# ===========================================================================
@bp.route("/api/train/columns")
def train_columns():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    meta = store.load_meta(sid)
    coltypes = meta.get("coltypes", {})
    outcomes = [{"name": c, "type": t} for c, t in coltypes.items()
                if t in ("binary", "categorical", "continuous")]
    return jsonify({
        "outcomes": outcomes,
        "columns": [{"name": c, "type": t} for c, t in coltypes.items()],
        "default_grids_binary": ml.default_grids("binary"),
        "default_grids_continuous": ml.default_grids("continuous"),
        "smote_available": ml.HAS_SMOTE,
        "n_train": meta.get("n_train"),
    })


@bp.route("/api/train/upload-split", methods=["POST"])
def train_upload_split():
    """Bring-your-own train/validation split: replaces train.csv/val.csv for
    this session with user-uploaded CSVs instead of the automatic split from
    /api/confirm. The held-out test.csv and full.csv (used by Data Summary /
    Spline / Clinical Epi) are left untouched."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    if "train_file" not in request.files or "val_file" not in request.files:
        return jsonify({"error": "Both a train CSV and a validation CSV are required."}), 400

    try:
        train_df = pd.read_csv(request.files["train_file"])
        val_df = pd.read_csv(request.files["val_file"])
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    if train_df.empty or val_df.empty:
        return jsonify({"error": "Uploaded train/validation CSVs must not be empty."}), 400
    if set(train_df.columns) != set(val_df.columns):
        return jsonify({"error": "Train and validation CSVs must have the same columns."}), 400

    train_df.to_csv(store.session_path(sid, "train.csv"), index=False)
    val_df.to_csv(store.session_path(sid, "val.csv"), index=False)

    # re-detect column types from the uploaded data so outcome selection and
    # coltype-aware preprocessing downstream stay consistent with what's
    # actually in these files
    combined = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    cols = stats.detect_column_types(combined)
    coltypes = {c["name"]: c["type"] for c in cols}

    meta = store.load_meta(sid)
    meta.update({
        "coltypes": coltypes,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "custom_split": True,
    })
    store.save_meta(sid, meta)
    analysis_log.log_action(
        sid, "upload_custom_split",
        inputs={"train_filename": request.files["train_file"].filename,
                "val_filename": request.files["val_file"].filename},
        outputs={"n_train": len(train_df), "n_val": len(val_df)})

    return jsonify({"ok": True, "n_train": len(train_df), "n_val": len(val_df), "columns": cols})


@bp.route("/api/train", methods=["POST"])
def train_config():
    """Store the training configuration; execution happens on the SSE stream."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    body = request.get_json(force=True)
    meta = store.load_meta(sid)

    outcome = body["outcome"]
    # Read the whole outcome column, not a head sample: task detection counts
    # distinct values, and a few hundred rows can show <=10 for a continuous
    # outcome (e.g. length of stay), misclassifying it as multiclass.
    outcome_col = pd.read_csv(store.session_path(sid, "train.csv"), usecols=[outcome])[outcome]
    task = ml.determine_task(outcome_col)

    cfg = {
        "outcome": outcome,
        "task": task,
        "exclude": body.get("exclude", []),
        "confounders": body.get("confounders", []),
        "models": body.get("models", []),
        "grids": body.get("grids", {}),         # {model: {param: [..]}}  (optional overrides)
        "scoring": body.get("scoring", "roc"),
        "grid_search": bool(body.get("grid_search", True)),
        "smote": bool(body.get("smote", meta.get("smote_pref", False))),
        "threshold": float(body.get("threshold", 0.5)),
        # k-fold CV on the training set for grid search, instead of the
        # single validation-set split -- helps on small datasets. Leave unset
        # (None) to keep the default validation-set-only behavior.
        "cv_folds": int(body["cv_folds"]) if body.get("cv_folds") else None,
    }
    meta["train_config"] = cfg
    store.save_meta(sid, meta)
    return jsonify({"ok": True, "task": task, "n_models": len(cfg["models"])})


@bp.route("/api/train/stream")
def train_stream():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    meta = store.load_meta(sid)
    cfg = meta.get("train_config")
    if not cfg:
        return jsonify({"error": "No training configuration. Call /api/train first."}), 400

    df_train = pd.read_csv(store.session_path(sid, "train.csv"))
    df_val = pd.read_csv(store.session_path(sid, "val.csv"))

    outcome = cfg["outcome"]
    task = cfg["task"]
    exclude = list(cfg["exclude"])
    # date columns should never be features
    for c, t in meta.get("coltypes", {}).items():
        if t == "date" and c not in exclude:
            exclude.append(c)
    feat_cols = ml.feature_columns(df_train, outcome, exclude)

    def sse(obj):
        return f"data: {json.dumps(_sanitize(obj))}\n\n"

    def generate():
        models = cfg["models"]
        yield sse({"event": "start", "n_models": len(models), "task": task,
                   "features": len(feat_cols), "outcome": outcome})
        results = []
        for i, name in enumerate(models):
            yield sse({"event": "progress", "model": name, "index": i,
                       "total": len(models),
                       "pct": round(100 * i / max(len(models), 1)),
                       "message": f"Training {name}…"})
            try:
                grid = cfg["grids"].get(name)
                pipe, info = ml.train_one_model(
                    name, df_train, df_val, outcome, feat_cols, task,
                    param_grid=grid, scoring=cfg["scoring"],
                    use_smote=cfg["smote"], do_grid_search=cfg["grid_search"],
                    coltypes=meta.get("coltypes", {}), cv_folds=cfg["cv_folds"])

                vmetrics, vcoords = ml.evaluate(pipe, df_val, outcome, feat_cols,
                                                task, threshold=cfg["threshold"])

                # feature importance -> CSV + PNG
                imp = ml.feature_importance(pipe, df_val, outcome, feat_cols, task)
                ts = store.timestamp()
                base = _safe(f"{name}_{outcome}_{ts}")
                joblib.dump({"pipe": pipe, "feat_cols": feat_cols, "outcome": outcome,
                             "task": task, "classes": info.get("classes"),
                             "threshold": cfg["threshold"]},
                            store.model_path(base))
                store_meta = {
                    "model": name, "outcome": outcome, "task": task,
                    "scoring": cfg["scoring"], "smote": info["smote"],
                    "best_params": info["best_params"], "classes": info.get("classes"),
                    "features": feat_cols, "val_metrics": vmetrics,
                    "threshold": cfg["threshold"], "created": ts,
                    "session": sid,
                }
                with open(store.model_meta_path(base), "w") as fh:
                    json.dump(store_meta, fh, indent=2, default=str)
                imp.to_csv(store.EXPORTS + f"/{base}_importance.csv", index=False)

                imp_png = plots.fig_to_png(plots.importance_plot(
                    imp["feature"], imp["importance"], f"{name}: feature importance"))

                results.append({"model": name, "metrics": vmetrics})
                analysis_log.log_action(sid, "train_model",
                                        inputs={"model": name, "outcome": outcome,
                                                "scoring": cfg["scoring"],
                                                "smote": info["smote"],
                                                "cv_folds": info["cv_folds"]},
                                        outputs={"saved_as": base, "metrics": vmetrics})
                yield sse({"event": "model_done", "model": name,
                           "saved_as": base, "metrics": vmetrics,
                           "best_params": info["best_params"],
                           "coords": vcoords,
                           "importance": imp.head(20).to_dict(orient="records"),
                           "importance_png": imp_png,
                           "pct": round(100 * (i + 1) / max(len(models), 1))})
            except Exception as e:
                yield sse({"event": "model_error", "model": name, "error": str(e)})
        yield sse({"event": "complete", "results": results})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ===========================================================================
# 4. MODEL TESTING
# ===========================================================================
@bp.route("/api/models")
def list_models():
    return jsonify({"models": store.list_models()})


@bp.route("/api/model/upload", methods=["POST"])
def model_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    name = _safe(f.filename.rsplit(".", 1)[0]) + "_uploaded"
    f.save(store.model_path(name))
    return jsonify({"ok": True, "name": name})


@bp.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    name = _safe(name)
    if not store.model_exists(name):
        return jsonify({"error": "Model not found."}), 404
    store.delete_model(name)
    return jsonify({"ok": True})


def _load_model(name):
    obj = joblib.load(store.model_path(name))
    if not isinstance(obj, dict):       # raw pipeline pickle (user uploaded)
        return {"pipe": obj, "feat_cols": None, "outcome": None,
                "task": None, "classes": None, "threshold": 0.5}
    return obj


@bp.route("/api/test", methods=["POST"])
def test_models():
    """Evaluate one or more saved models on the held-out test set (or a new CSV)."""
    names = request.form.getlist("models") or (request.json or {}).get("models", [])
    source = request.form.get("source") or (request.json or {}).get("source", "test")
    threshold = float(request.form.get("threshold", 0.5)) if request.form else \
        float((request.json or {}).get("threshold", 0.5))

    # choose dataframe
    sid = None
    if source == "upload" and "file" in request.files:
        df = pd.read_csv(request.files["file"])
    else:
        sid = require_session()
        if not sid:
            return jsonify({"error": "No session test set; upload a CSV instead."}), 400
        df = pd.read_csv(store.session_path(sid, "test.csv"))

    roc_series, pr_series = [], []
    out = {"models": [], "task": None}
    for name in names:
        mdl = _load_model(name)
        pipe, feat_cols = mdl["pipe"], mdl["feat_cols"]
        outcome, task = mdl["outcome"], mdl["task"]
        if outcome is None or outcome not in df.columns:
            out["models"].append({"name": name, "error": "Outcome column not found in data."})
            continue
        if feat_cols is None:
            feat_cols = [c for c in df.columns if c != outcome]
        out["task"] = task
        metrics, coords = ml.evaluate(pipe, df, outcome, feat_cols, task, threshold)
        if sid:
            analysis_log.log_action(sid, "test_model", inputs={"model": name, "source": source},
                                    outputs={"metrics": metrics})
        entry = {"name": name, "metrics": metrics, "coords": coords}
        if task in ("binary", "continuous"):
            corr = ml.feature_outcome_correlations(df, outcome, feat_cols, task)
            entry["correlations"] = corr.head(20).to_dict(orient="records")
        if task == "binary":
            roc_series.append({"name": name, **coords["roc"]})
            pr_series.append({"name": name, **coords["pr"]})
            entry["confusion_png"] = plots.fig_to_png(
                plots.confusion_plot(coords["confusion"], title=f"{name}: confusion"))
            entry["calibration_png"] = plots.fig_to_png(plots.calibration_plot(
                coords["calibration"]["prob_pred"], coords["calibration"]["prob_true"],
                name=name, title=f"{name}: calibration"))
        elif task == "continuous":
            entry["residuals_png"] = plots.fig_to_png(plots.residual_plot(
                coords["residuals"]["pred"], coords["residuals"]["resid"],
                title=f"{name}: residuals"))
            entry["pred_vs_actual_png"] = plots.fig_to_png(plots.pred_vs_actual_plot(
                coords["pred_vs_actual"]["actual"], coords["pred_vs_actual"]["pred"],
                title=f"{name}: predicted vs actual"))
        elif task == "multiclass" and coords.get("roc"):
            entry["roc_png"] = plots.fig_to_png(plots.roc_overlay(
                coords["roc"], title=f"{name}: ROC (one-vs-rest)"))
            entry["pr_png"] = plots.fig_to_png(plots.pr_overlay(
                coords["pr"], title=f"{name}: PR (one-vs-rest)"))
        out["models"].append(entry)

    if roc_series:
        out["roc_png"] = plots.fig_to_png(plots.roc_overlay(roc_series))
        out["pr_png"] = plots.fig_to_png(plots.pr_overlay(pr_series))
    return jsonify(out)


@bp.route("/api/model/features")
def model_features():
    """Feature list for a saved model (for building a single-prediction form)."""
    name = request.args.get("name")
    mdl = _load_model(name)
    return jsonify({"name": name, "features": mdl["feat_cols"],
                    "outcome": mdl["outcome"], "task": mdl["task"],
                    "classes": mdl["classes"]})


@bp.route("/api/predict-single", methods=["POST"])
def predict_single():
    """Upload a JSON row (+ choose a model) -> single prediction with probability."""
    if "model_file" in request.files and "json_file" in request.files:
        pipe_obj = joblib.load(request.files["model_file"])
        mdl = pipe_obj if isinstance(pipe_obj, dict) else {"pipe": pipe_obj,
              "feat_cols": None, "task": "binary", "classes": None}
        row = json.load(request.files["json_file"])
    else:
        body = request.get_json(force=True)
        mdl = _load_model(body["model"])
        row = body.get("row", {})
        if isinstance(row, str):
            row = json.loads(row)

    feat_cols = mdl["feat_cols"] or list(row.keys())
    task = mdl["task"] or "binary"
    result = ml.predict_single(mdl["pipe"], row, feat_cols, task, mdl.get("classes"))
    return jsonify(result)


# ===========================================================================
# 5. SPLINE
# ===========================================================================
@bp.route("/api/spline", methods=["POST"])
def spline_fit():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    body = request.get_json(force=True)
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    try:
        res = spline_mod.fit_rcs(df, body["predictor"], body["outcome"],
                                 n_knots=body.get("n_knots", 4))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    res["png"] = plots.fig_to_png(plots.spline_plot(
        res["x"], res["logodds"], res["lower"], res["upper"], res["knots"],
        predictor=res["predictor"],
        title=f"RCS: {res['outcome']} ~ {res['predictor']} ({res['n_knots']} knots)"))
    analysis_log.log_action(sid, "fit_spline",
                            inputs={"predictor": res["predictor"], "outcome": res["outcome"],
                                    "n_knots": res["n_knots"]},
                            outputs={"aic": res["aic"], "n": res["n"]})
    return jsonify(res)


# ===========================================================================
# 6. CLINICAL EPI
# ===========================================================================
@bp.route("/api/epi/or", methods=["POST"])
def epi_or():
    b = request.get_json(force=True)
    return jsonify(epi.two_by_two(b["a"], b["b"], b["c"], b["d"]))


@bp.route("/api/epi/nnt", methods=["POST"])
def epi_nnt():
    b = request.get_json(force=True)
    return jsonify(epi.nnt_nnh(b["a"], b["b"], b["c"], b["d"]))


@bp.route("/api/epi/km", methods=["POST"])
def epi_km():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(force=True)
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    try:
        res = epi.kaplan_meier(df, b["time"], b["event"], b.get("group") or None)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    res["png"] = plots.fig_to_png(plots.km_plot(res["curves"]))
    analysis_log.log_action(sid, "epi_km",
                            inputs={"time": b["time"], "event": b["event"],
                                    "group": b.get("group")},
                            outputs={"n_curves": len(res["curves"]),
                                     "logrank": res["logrank"]})
    return jsonify(res)


@bp.route("/api/epi/cox", methods=["POST"])
def epi_cox():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(force=True)
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    try:
        res = epi.cox_ph(df, b["time"], b["event"], b["covariates"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    analysis_log.log_action(sid, "epi_cox",
                            inputs={"time": b["time"], "event": b["event"],
                                    "covariates": b["covariates"]},
                            outputs={"concordance": res["concordance"], "n": res["n"]})
    return jsonify(res)


@bp.route("/api/epi/hl", methods=["POST"])
def epi_hl():
    """Hosmer-Lemeshow from a saved model applied to the test set."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(force=True)
    try:
        mdl = _load_model(b["model"])
    except Exception as e:
        return jsonify({"error": f"Could not load model: {e}"}), 400
    if mdl["task"] != "binary":
        return jsonify({"error": "Hosmer-Lemeshow needs a binary model."}), 400
    df = pd.read_csv(store.session_path(sid, "test.csv"))
    y, _ = ml.encode_outcome(df[mdl["outcome"]], "binary")
    mask = ~pd.isna(y)
    scores = ml.predict_scores(mdl["pipe"], df[mdl["feat_cols"]][mask], "binary")
    res = epi.hosmer_lemeshow(np.asarray(y)[mask], scores)
    res["png"] = plots.fig_to_png(plots.calibration_hl_plot(
        res["mean_pred"], res["observed_rate"]))
    analysis_log.log_action(sid, "epi_hl", inputs={"model": b["model"]},
                            outputs={"hl_statistic": res["hl_statistic"],
                                     "p_value": res["p_value"]})
    return jsonify(res)


# ===========================================================================
# 7. PDF export (any chart) -- "render this data as a downloadable PDF"
# ===========================================================================
def _build_fig(kind, data):
    if kind == "roc":
        return plots.roc_overlay(data["series"])
    if kind == "pr":
        return plots.pr_overlay(data["series"])
    if kind == "confusion":
        return plots.confusion_plot(data["confusion"])
    if kind == "calibration":
        return plots.calibration_plot(data["prob_pred"], data["prob_true"])
    if kind == "residuals":
        return plots.residual_plot(data["pred"], data["resid"])
    if kind == "pred_vs_actual":
        return plots.pred_vs_actual_plot(data["actual"], data["pred"])
    if kind == "importance":
        return plots.importance_plot(data["names"], data["values"])
    if kind == "spline":
        return plots.spline_plot(data["x"], data["logodds"], data["lower"],
                                 data["upper"], data["knots"], data.get("predictor", "x"))
    if kind == "km":
        return plots.km_plot(data["curves"])
    if kind == "hl":
        return plots.calibration_hl_plot(data["mean_pred"], data["observed_rate"])
    raise ValueError(f"Unknown chart kind: {kind}")


@bp.route("/api/render/pdf", methods=["POST"])
def render_pdf():
    b = request.get_json(force=True)
    fig = _build_fig(b["kind"], b["data"])
    pdf = plots.fig_to_pdf_bytes(fig)
    return send_file(io.BytesIO(pdf), mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"{_safe(b.get('name', b['kind']))}.pdf")


# ===========================================================================
# 8. Reports (methods text + TRIPOD+AI coverage), from the analysis log
# ===========================================================================
@bp.route("/api/report/methods")
def report_methods():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    return jsonify({"text": methods_text.generate_methods_text(sid)})


@bp.route("/api/report/tripod")
def report_tripod():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    return jsonify(tripod_report.generate_tripod_report(sid))
