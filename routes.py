"""routes.py
Every HTTP endpoint. This file only does HTTP: parse the request, call into
clintab/ (ml.py / stats.py / spline.py / epi.py / plots.py / validation.py /
cohort.py / samplesize.py / teaching.py) for the real work, return JSON (or an
SSE stream / a file download). No statistics or sklearn logic lives here.
"""
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import threading
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import (
    Blueprint, request, jsonify, session, Response, send_file,
    send_from_directory, current_app,
)
from sklearn.model_selection import train_test_split

import clintab
from clintab import store
from clintab import stats
from clintab import ml
from clintab import plots
from clintab import spline as spline_mod
from clintab import epi
from clintab import validation
from clintab import cohort as cohort_mod
from clintab import samplesize
from clintab import teaching
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


_safe = store.safe_name


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


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_TS_SUFFIX = re.compile(r"_\d{8}-\d{6}$")


def _model_identity(name, mdl=None):
    """(model type, outcome) for a saved model, from its sidecar meta, else
    from the loaded object, else None. Used so logs and reports name models
    by type and outcome instead of by file id."""
    model_type, outcome = None, (mdl or {}).get("outcome")
    try:
        with open(store.model_meta_path(name)) as fh:
            m = json.load(fh)
        model_type = m.get("model") or model_type
        outcome = m.get("outcome") or outcome
    except (OSError, ValueError):
        pass
    if model_type is None and name and not name.endswith("_uploaded"):
        stem = _TS_SUFFIX.sub("", name)
        model_type = stem.split("_", 1)[0] if _TS_SUFFIX.search(name) else None
    return model_type, outcome


# ---------------------------------------------------------------------------
# test-set lock: once the internal test partition has been used, testing a
# model that wasn't in that first use needs a stated reason, because picking a
# model after seeing test results makes the test results optimistic.
# ---------------------------------------------------------------------------
LOCK_MESSAGE = ("The internal test partition has already been used. Choosing a model "
                "after seeing test results makes those results optimistic, so testing "
                "a new model on it needs a reason, which is logged.")


def _test_lock_gate(sid, names, unlock_reason):
    """Returns (error_response or None, new_models). new_models are the
    requested models outside the lock; if there are any and a reason was
    given, they are added to the lock and a test_unlock entry is logged."""
    meta = store.load_meta(sid) or {}
    lock = meta.get("test_lock")
    if not lock:
        return None, []
    new = [n for n in names if n not in lock.get("models", [])]
    if not new:
        return None, []
    if not unlock_reason:
        return (jsonify({"error": LOCK_MESSAGE, "locked": True, "new_models": new,
                         "locked_models": lock.get("models", [])}), 409), new
    lock["models"] = list(lock.get("models", [])) + new
    meta["test_lock"] = lock
    store.save_meta(sid, meta)
    analysis_log.log_action(sid, "test_unlock",
                            inputs={"reason": unlock_reason, "models": new},
                            outputs={"locked_models": lock["models"]})
    return None, new


def _record_test_use(sid, evaluated):
    """First use of the internal test partition: lock it to these models."""
    if not evaluated:
        return (store.load_meta(sid) or {}).get("test_lock")
    meta = store.load_meta(sid) or {}
    if not meta.get("test_lock"):
        meta["test_lock"] = {"locked_at": _now_iso(), "models": list(evaluated)}
        store.save_meta(sid, meta)
    return meta["test_lock"]


def _unlock_reason(body):
    return str(request.form.get("unlock_reason") or body.get("unlock_reason") or "").strip()


def _apply_model_missing(df, mdl):
    """Run new data (an external CSV, a single row) through the missing-data
    handling the model was trained with: training-fitted fill values and
    missing indicators. Session partitions already had it applied at confirm."""
    mh = (mdl or {}).get("missing_handling")
    if not mh:
        return df
    # never fill the outcome: rows without one are dropped from evaluation
    decisions = {c: d for c, d in mh.get("decisions", {}).items() if c != mdl.get("outcome")}
    return stats.apply_fitted_missing_handling(
        df, decisions,
        {"fills": mh.get("fills", {}), "indicators": mh.get("indicators", {})})


def _derived_features(mdl):
    """indicator columns the app builds itself, so a form shouldn't ask for them."""
    return list(((mdl or {}).get("missing_handling") or {}).get("indicators", {}).values())


def _feature_cols(meta, df, outcome, exclude):
    """the model features: every column but the outcome, the excluded ones
    and date columns (dates are never features)."""
    exclude = list(exclude or [])
    for c, t in (meta or {}).get("coltypes", {}).items():
        if t == "date" and c not in exclude:
            exclude.append(c)
    return ml.feature_columns(df, outcome, exclude)


def _sse(obj):
    return f"data: {json.dumps(_sanitize(obj))}\n\n"


def _num_or_none(value, what):
    """float, or None for a missing or blank value. ValueError with a plain
    message otherwise."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number.")
    if not math.isfinite(out):
        raise ValueError(f"{what} must be a number.")
    return out


def _cohort_payload(flow, n_start):
    n_final = flow[-1]["n_after"] if flow else n_start
    return {"flow": flow, "lines": cohort_mod.summarize_flow(flow, n_start),
            "n_start": int(n_start), "n_final": int(n_final)}


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
@bp.route("/")
def index():
    return send_from_directory(current_app.static_folder, "index.html")


@bp.route("/api/version")
def version():
    return jsonify({"app": "ClinTAB-ML", "version": clintab.__version__})


# ===========================================================================
# 1. data upload
# ===========================================================================
@bp.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if not (f.filename or "").lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400

    sid = store.new_session_id()
    store.session_dir(sid, create=True)
    raw_path = store.session_path(sid, "raw.csv")
    f.save(raw_path)

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        shutil.rmtree(store.session_dir(sid), ignore_errors=True)   # dont leave a half session behind
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400
    if df.empty:
        shutil.rmtree(store.session_dir(sid), ignore_errors=True)
        return jsonify({"error": "The CSV has no data rows."}), 400

    cols = stats.detect_column_types(df)
    meta = {
        "filename": f.filename,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns_detected": cols,
        "stage": "uploaded",
    }
    meta["sha256"] = _sha256_file(raw_path)
    store.save_meta(sid, meta)
    session["sid"] = sid
    analysis_log.log_action(sid, "upload",
                            inputs={"filename": f.filename, "sha256": meta["sha256"]},
                            outputs={"n_rows": meta["n_rows"], "n_cols": meta["n_cols"],
                                     "environment": analysis_log.environment()})

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


@bp.route("/api/cohort/preview", methods=["POST"])
def cohort_preview():
    """Run cohort filters on the raw upload without saving anything, so the
    flow can be checked before confirm."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session. Upload a file first."}), 400
    body = request.get_json(silent=True) or {}
    df = pd.read_csv(store.session_path(sid, "raw.csv"))
    try:
        _, flow = cohort_mod.apply_filters(df, body.get("filters") or [])
    except cohort_mod.CohortFilterError as e:
        return jsonify({"error": str(e), "problems": e.problems}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Could not apply the cohort filters: {e}",
                        "problems": [str(e)]}), 400
    out = _cohort_payload(flow, len(df))
    out["png"] = plots.fig_to_png(plots.flow_plot(flow, len(df)))
    return jsonify(out)


@bp.route("/api/confirm", methods=["POST"])
def confirm():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session. Upload a file first."}), 400
    body = request.get_json(force=True)

    coltypes = body.get("types", {})          # {col: binary|categorical|continuous|date|exclude}
    missing = body.get("missing") or {}       # {col: include|zero|impute|impute_indicator|remove}
    ratios = body.get("ratios", [0.70, 0.15, 0.15])
    method = body.get("method", "random")      # random | stratified
    stratify_col = body.get("stratify_col") or None
    smote_pref = bool(body.get("smote", False))
    try:
        seed = int(body.get("seed", 42))
        r = [float(x) for x in ratios]
    except (TypeError, ValueError):
        return jsonify({"error": "Split ratios and seed must be numbers."}), 400
    if len(r) != 3 or min(r) <= 0:
        return jsonify({"error": "Train, validation and test shares must all be above zero."}), 400
    r = [x / sum(r) for x in r]
    if not isinstance(missing, dict):
        return jsonify({"error": "Missing-data decisions must be an object of {column: decision}."}), 400
    bad = sorted({str(d) for d in missing.values() if d not in stats.MISSING_DECISIONS})
    if bad:
        return jsonify({"error": f"Unknown missing-data decision(s): {bad}. "
                                 f"Use one of {list(stats.MISSING_DECISIONS)}."}), 400

    filters = body.get("cohort") or []
    df = pd.read_csv(store.session_path(sid, "raw.csv"))
    # the cohort comes first, so missingness, fills and the split all
    # describe the study cohort, not the whole file
    cohort_info = None
    if filters:
        n_start = len(df)
        try:
            df, flow = cohort_mod.apply_filters(df, filters)
        except cohort_mod.CohortFilterError as e:
            return jsonify({"error": str(e), "problems": e.problems}), 400
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Could not apply the cohort filters: {e}",
                            "problems": [str(e)]}), 400
        if len(df) == 0:
            return jsonify({"error": "No rows are left after the cohort filters. "
                                     "Loosen a filter and try again.",
                            "cohort": _cohort_payload(flow, n_start)}), 400
        cohort_info = _cohort_payload(flow, n_start)
    # measured before anything is dropped or filled, for the log and tripod
    missing_pct = stats.missing_percentages(df)
    missing = {c: d for c, d in missing.items() if c in df.columns}

    # drop removed + excluded columns now; fills wait until after the split so
    # they can be learned from the training partition alone
    excluded = [c for c, t in coltypes.items() if t == "exclude"]
    removed = [c for c, d in missing.items() if d == "remove"]
    df = df.drop(columns=[c for c in df.columns if c in set(excluded) | set(removed)])
    kept_types = {c: t for c, t in coltypes.items()
                  if t != "exclude" and c in df.columns}

    if method == "stratified" and stratify_col not in df.columns:
        return jsonify({"error": "Choose a column to stratify by (it can't be excluded or removed)."}), 400

    # split -------------------------------------------------------------
    # stratify on string labels, so a still-missing value is just one more level
    strat = df[stratify_col].astype(str) if method == "stratified" else None
    try:
        train, temp = train_test_split(df, test_size=(r[1] + r[2]),
                                       random_state=seed, stratify=strat)
        strat2 = temp[stratify_col].astype(str) if strat is not None else None
        val_frac = r[1] / (r[1] + r[2])
        val, test = train_test_split(temp, test_size=(1 - val_frac),
                                     random_state=seed, stratify=strat2)
    except (ValueError, TypeError):
        # stratification can fail on tiny/rare classes -> fall back to random
        try:
            train, temp = train_test_split(df, test_size=(r[1] + r[2]), random_state=seed)
            val, test = train_test_split(temp, test_size=(r[2] / (r[1] + r[2])),
                                         random_state=seed)
        except ValueError:
            return jsonify({"error": "Not enough rows to make three non-empty partitions."}), 400
        if method == "stratified":
            method = "random (stratify fell back)"

    # class imbalance note: reports the minority share (observed values, before
    # any fill), never recommends resampling (it usually worsens calibration,
    # van den Goorbergh et al. 2022)
    smote_hint = None
    if stratify_col and stratify_col in df.columns:
        balance = stats.class_balance(df[stratify_col])
        if balance is not None:
            smote_hint = {"column": stratify_col, **balance}

    # missing data: fill values come from the training partition only, then
    # the same values go to validation, test and the full cleaned dataset
    fitted = stats.fit_missing_handling(train, missing, kept_types)
    train, val, test, df = (stats.apply_fitted_missing_handling(p, missing, fitted)
                            for p in (train, val, test, df))
    for ind in fitted["indicators"].values():
        kept_types[ind] = "binary"

    # save cleaned full dataset + partitions
    df.to_csv(store.session_path(sid, "full.csv"), index=False)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(store.session_path(sid, f"{name}.csv"), index=False)

    meta = store.load_meta(sid)
    meta.update({
        "stage": "confirmed",
        "custom_split": False,          # a new split replaces any uploaded one
        "coltypes": kept_types,
        "missing": missing,
        "missing_pct": missing_pct,
        "fill_values": fitted["fills"],
        "indicators": fitted["indicators"],
        "excluded": excluded,
        "split": {"ratios": r, "method": method, "stratify_col": stratify_col, "seed": seed},
        "smote_pref": smote_pref,
        "n_rows_clean": int(len(df)),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "smote_hint": smote_hint,
        "cohort": ({"filters": filters, "flow": cohort_info["flow"],
                    "n_start": cohort_info["n_start"], "n_final": cohort_info["n_final"]}
                   if cohort_info else None),
    })
    store.save_meta(sid, _sanitize(meta))
    if cohort_info:
        analysis_log.log_action(sid, "cohort_filter", inputs={"filters": filters},
                                outputs=_sanitize({"flow": cohort_info["flow"],
                                                   "n_start": cohort_info["n_start"],
                                                   "n_final": cohort_info["n_final"]}))
    analysis_log.log_action(sid, "confirm",
                            inputs={"method": method, "ratios": r, "seed": seed,
                                    "stratify_col": stratify_col, "smote_pref": smote_pref,
                                    "missing": missing, "missing_pct": missing_pct,
                                    "excluded": excluded, "smote_hint": smote_hint,
                                    # 0 tells the reports an earlier cohort no longer applies
                                    "cohort_filters": len(filters) if cohort_info else 0},
                            outputs={"n_rows": int(len(df)), "n_train": len(train),
                                     "n_val": len(val), "n_test": len(test),
                                     "fill_values": fitted["fills"],
                                     "indicators": fitted["indicators"],
                                     "removed": removed,
                                     "fill_source": "training partition"})
    return jsonify({"ok": True, "n_train": len(train), "n_val": len(val),
                    "n_test": len(test), "method": method, "smote_hint": smote_hint,
                    "missing_pct": missing_pct, "fill_values": fitted["fills"],
                    "indicators": fitted["indicators"], "cohort": cohort_info})


# ===========================================================================
# 2. data summary
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
# 3. model training
# ===========================================================================
@bp.route("/api/train/columns")
def train_columns():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    meta = store.load_meta(sid)
    coltypes = meta.get("coltypes", {})
    names = [c for c, t in coltypes.items() if t in ("binary", "categorical", "continuous")]
    # the same rule /api/train applies, so the page and the server agree
    tasks = {}
    if names:
        head = pd.read_csv(store.session_path(sid, "train.csv"), usecols=names)
        tasks = {c: ml.determine_task(head[c]) for c in names}
    outcomes = [{"name": c, "type": coltypes[c], "task": tasks.get(c)} for c in names]
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
    # coltype-aware preprocessing downstream stay consitent with what's
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


THRESHOLD_METHODS = ("youden", "sensitivity", "specificity", "prevalence", "fixed")


def _outcome_imputed(meta, outcome):
    """A message when confirm filled missing values of this outcome column
    (an imputed outcome is not an observed one), else None. A user-supplied
    split never went through confirm, so it is not checked."""
    if meta.get("custom_split"):
        return None
    filled = (outcome in (meta.get("fill_values") or {})
              or outcome in (meta.get("indicators") or {}))
    pct = meta.get("missing_pct")
    had_missing = pct is None or (pct or {}).get(outcome, 0) > 0
    if not (filled and had_missing):
        return None
    return (f"The outcome {outcome} had missing values that were filled in at confirm, "
            f"so some outcomes would be made up rather than observed. Set the "
            f"missing-data choice for {outcome} to \"Include as is\" (rows with a missing "
            f"outcome are then left out of training) and confirm again.")


@bp.route("/api/train", methods=["POST"])
def train_config():
    """Store the training configuration; execution happens on the SSE stream."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    body = request.get_json(force=True)
    meta = store.load_meta(sid)

    outcome = body.get("outcome")
    train_cols = list(pd.read_csv(store.session_path(sid, "train.csv"), nrows=0).columns)
    if outcome not in train_cols:
        return jsonify({"error": f"Outcome column {outcome!r} is not in the training data."}), 400
    # read teh whole outcome column, not a head sample: task detection counts
    # distinct values, and a few hundred rows can show <=10 for a continuous
    # outcome (e.g. length of stay) or miss a rare class completely.
    task = ml.determine_task(pd.read_csv(store.session_path(sid, "train.csv"),
                                         usecols=[outcome])[outcome])
    unknown = [m for m in body.get("models", []) if m not in ml.model_catalogue(task)]
    if unknown:
        return jsonify({"error": f"Unknown model(s) for a {task} outcome: {unknown}"}), 400
    guard = _outcome_imputed(meta, outcome)
    if guard:
        return jsonify({"error": guard}), 400
    try:
        cv_folds = int(body["cv_folds"]) if body.get("cv_folds") else None
        threshold = float(body.get("threshold", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "cv_folds must be a whole number and threshold a number."}), 400
    if cv_folds is not None and cv_folds < 2:
        return jsonify({"error": "cv_folds must be at least 2."}), 400
    thr_method, thr_target, recalibrate = None, None, False
    if task == "binary":
        thr_method = str(body.get("threshold_method") or "youden").strip().lower()
        if thr_method not in THRESHOLD_METHODS:
            return jsonify({"error": f"Unknown threshold method {thr_method!r}. "
                                     f"Use one of {list(THRESHOLD_METHODS)}."}), 400
        try:
            thr_target = _num_or_none(body.get("threshold_target"), "The threshold target")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if thr_method == "fixed" and thr_target is None and "threshold" in body:
            thr_target = threshold          # older clients send the fixed value as threshold
        if thr_method in ("fixed", "sensitivity", "specificity") and thr_target is None:
            return jsonify({"error": f"The {thr_method} threshold method needs a target "
                                     "value (threshold_target)."}), 400
        if thr_method == "fixed" and not 0 < thr_target < 1:
            return jsonify({"error": "A fixed threshold must be between 0 and 1."}), 400
        if thr_method in ("sensitivity", "specificity") and not 0 < thr_target <= 1:
            return jsonify({"error": f"A {thr_method} target must be above 0 and at most 1."}), 400
        if thr_method not in ("fixed", "sensitivity", "specificity"):
            thr_target = None
        if thr_method == "fixed":
            threshold = thr_target
        recalibrate = bool(body.get("recalibrate", False))

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
        "threshold": threshold,
        # binary only: how the threshold is picked on the validation set,
        # and whether predictions are logistically recalibrated there
        "threshold_method": thr_method,
        "threshold_target": thr_target,
        "recalibrate": recalibrate,
        # k-fold cv on the training set for grid search, instead of the
        # single validation set split, helps on small datasets. none keeps
        # the default (validation set only).
        "cv_folds": cv_folds,
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
    confounders = list(cfg.get("confounders", []))
    # confounders are forced in, so they can't also be excluded
    exclude = [c for c in cfg["exclude"] if c not in confounders]
    # date columns should never be features
    feat_cols = _feature_cols(meta, df_train, outcome, exclude)
    confounders = [c for c in confounders if c in feat_cols]
    coltypes = meta.get("coltypes", {})
    # saved with each model so external data and single rows get the same
    # training-fitted fills and indicators. not for a user-supplied split,
    # whose files never went through the confirm-step handling.
    missing_handling = None
    if meta.get("missing") and not meta.get("custom_split"):
        missing_handling = {"decisions": meta["missing"],
                            "fills": meta.get("fill_values", {}),
                            "indicators": meta.get("indicators", {})}

    sse = _sse

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
                    coltypes=coltypes, cv_folds=cfg["cv_folds"])
                classes = info.get("classes")

                # binary: optional recalibration, then the threshold, both
                # fitted on the validation rows that have an outcome
                threshold = cfg["threshold"]
                thr_method = thr_target = recal = None
                notes = []
                if task == "binary":
                    thr_method = cfg.get("threshold_method") or "fixed"
                    thr_target = cfg.get("threshold_target")
                    y_va, _ = ml.encode_outcome(df_val[outcome], task, classes)
                    keep = ~pd.isna(y_va)
                    X_va = df_val[feat_cols][keep]
                    y_va = np.asarray(y_va, dtype=float)[keep].astype(int)
                    if cfg.get("recalibrate"):
                        pipe = validation.recalibrate(pipe, X_va, y_va)
                        recal = {"a": pipe.a, "b": pipe.b}
                        notes.append("Predictions were recalibrated on the validation set, "
                                     "so the validation calibration shown here is optimistic "
                                     "(it was fitted on these patients). Judge calibration "
                                     "on the test set.")
                    if thr_method == "fixed":
                        threshold = thr_target if thr_target is not None else threshold
                    else:
                        choice = validation.choose_threshold(
                            y_va, ml.predict_scores(pipe, X_va, task), thr_method, thr_target)
                        threshold = choice["threshold"]
                        notes.append("The threshold was chosen on the validation set, so "
                                     "sensitivity and specificity there are optimistic. The "
                                     "test set gives honest values.")

                vmetrics, vcoords = ml.evaluate(pipe, df_val, outcome, feat_cols,
                                                task, threshold=threshold,
                                                classes=classes)

                # adjusted odds ratios from an unpenalized glm on the training
                # rows, next to the reference logistic model
                odds = None
                if name == "LogisticRegression" and task == "binary":
                    try:
                        odds = ml.adjusted_odds_ratios(df_train, outcome, feat_cols, coltypes)
                    except Exception as e:
                        notes.append(f"Adjusted odds ratios could not be computed: {e}")

                # feature importance -> csv + png
                imp = ml.feature_importance(pipe, df_val, outcome, feat_cols, task,
                                            classes=classes)
                ts = store.timestamp()
                base = store.model_base_name(name, outcome, ts)
                thr_fields = {"threshold": threshold, "threshold_method": thr_method,
                              "threshold_target": thr_target, "recalibrated": recal is not None}
                if recal is not None:
                    thr_fields["recalibration"] = recal
                joblib.dump({"pipe": pipe, "feat_cols": feat_cols, "outcome": outcome,
                             "task": task, "classes": classes,
                             "missing_handling": missing_handling, **thr_fields},
                            store.model_path(base))
                store_meta = {
                    "model": name, "outcome": outcome, "task": task,
                    "scoring": cfg["scoring"], "smote": info["smote"],
                    "best_params": info["best_params"], "classes": classes,
                    "features": feat_cols, "confounders": confounders,
                    "val_metrics": vmetrics,
                    **thr_fields, "created": ts,
                    "session": sid,
                }
                if odds is not None:
                    store_meta["odds_ratios"] = odds
                with open(store.model_meta_path(base), "w") as fh:
                    json.dump(_sanitize(store_meta), fh, indent=2, default=str)
                imp.to_csv(os.path.join(store.EXPORTS, f"{base}_importance.csv"), index=False)

                imp_png = plots.fig_to_png(plots.importance_plot(
                    imp["feature"], imp["importance"], f"{name}: feature importance"))

                results.append({"model": name, "metrics": vmetrics})
                outputs = {"saved_as": base, "metrics": vmetrics,
                           "n_train": info.get("n_train", int(len(df_train))), **thr_fields}
                if task == "binary":
                    # events = training rows in the positive (second) class
                    y_tr, _ = ml.encode_outcome(df_train[outcome], task, classes)
                    y_tr = np.asarray(y_tr, dtype=float)
                    outputs["events"] = int(np.sum(y_tr == 1))
                    if classes and len(classes) == 2:
                        outputs["positive_class"] = str(classes[1])
                try:
                    # candidate parameters, one-hot with a reference level dropped
                    outputs["n_parameters"] = ml.count_parameters(df_train, feat_cols, coltypes)
                except Exception:
                    pass
                analysis_log.log_action(sid, "train_model",
                                        inputs={"model": name, "outcome": outcome,
                                                "task": task,
                                                "scoring": cfg["scoring"],
                                                "smote": info["smote"],
                                                "tuned": info["tuned"],
                                                "cv_folds": info["cv_folds"],
                                                "confounders": confounders},
                                        outputs=_sanitize(outputs))
                if odds is not None:
                    analysis_log.log_action(sid, "adjusted_or",
                                            inputs={"outcome": outcome, "features": feat_cols},
                                            outputs=_sanitize({k: odds.get(k) for k in (
                                                "n", "events", "n_parameters", "epp")}))
                done = {"event": "model_done", "model": name,
                        "saved_as": base, "metrics": vmetrics,
                        "best_params": info["best_params"],
                        "coords": vcoords,
                        "importance": imp.head(20).to_dict(orient="records"),
                        "importance_png": imp_png,
                        **thr_fields, "notes": notes,
                        "pct": round(100 * (i + 1) / max(len(models), 1))}
                if odds is not None:
                    done["odds_ratios"] = odds
                yield sse(done)
            except Exception as e:
                yield sse({"event": "model_error", "model": name, "error": str(e)})
        yield sse({"event": "complete", "results": results})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ===========================================================================
# 4. model testing
# ===========================================================================
@bp.route("/api/models")
def list_models():
    return jsonify({"models": store.list_models()})


@bp.route("/api/model/upload", methods=["POST"])
def model_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    fname = f.filename or ""
    if not fname.lower().endswith((".pkl", ".joblib")):
        return jsonify({"error": "Please upload a .pkl (joblib) model file."}), 400
    store.ensure_dirs()
    name = _safe(fname.rsplit(".", 1)[0], max_len=80) + "_uploaded"
    f.save(store.model_path(name))
    return jsonify({"ok": True, "name": name})


@bp.route("/api/models/<name>", methods=["DELETE"])
def delete_model(name):
    if not store.model_exists(name):
        return jsonify({"error": "Model not found."}), 404
    store.delete_model(name)
    return jsonify({"ok": True})


def _load_model(name):
    """Load a saved model by name. Pickles run code when loaded, so this is
    only for model files you trust (see the security note in the README)."""
    if not store.model_exists(name):
        raise FileNotFoundError(f"Model not found: {name}")
    obj = joblib.load(store.model_path(name))
    if not isinstance(obj, dict):       # raw pipeline pickle (user uploaded)
        return {"pipe": obj, "feat_cols": None, "outcome": None,
                "task": None, "classes": None, "threshold": 0.5}
    return obj


def _flexible_calibration_fig(coords, name="model", title="Calibration"):
    """flexible calibration plot from ml.binary_metrics coords (or the same
    keys flattened, as the page may send them back for a pdf)."""
    flex = coords.get("calibration_flexible") or {}
    binned = coords.get("calibration") or {}
    return plots.flexible_calibration_plot(
        flex.get("p_grid", coords.get("p_grid")), flex.get("observed", coords.get("observed")),
        binned_pred=binned.get("prob_pred", coords.get("binned_pred")),
        binned_obs=binned.get("prob_true", coords.get("binned_obs")),
        predictions=coords.get("risk_histogram", coords.get("predictions")),
        stats=coords.get("calibration_stats", coords.get("stats")),
        name=coords.get("name", name), title=coords.get("title", title))


@bp.route("/api/test", methods=["POST"])
def test_models():
    """Evaluate one or more saved models on the held-out test set (or a new CSV).

    The internal test partition is locked to the models of its first use.
    Testing any other model on it returns 409 {"locked": true} unless the
    request carries a non-empty unlock_reason, which is logged. External
    CSVs are never locked.

    With no threshold in the request each model uses the threshold saved
    with it (chosen on the validation set at training); a threshold given
    here overrides it for every model."""
    body = request.get_json(silent=True) or {}
    names = request.form.getlist("models") or body.get("models", [])
    names = list(dict.fromkeys(names))          # same model twice = once
    source = request.form.get("source") or body.get("source", "test")
    raw_thr = request.form.get("threshold")
    if raw_thr is None:
        raw_thr = body.get("threshold")
    try:
        user_threshold = _num_or_none(raw_thr, "Threshold")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not names:
        return jsonify({"error": "No models selected."}), 400

    # choose dataframe
    external = source == "upload"
    ext_info = {}
    if external:
        if "file" not in request.files:
            return jsonify({"error": "Choose a CSV to test on."}), 400
        f = request.files["file"]
        raw = f.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            return jsonify({"error": f"Could not parse CSV: {e}"}), 400
        ext_info = {"filename": f.filename, "sha256": hashlib.sha256(raw).hexdigest(),
                    "n": int(len(df))}
        sid = require_session()       # optional: logged if a session is active
    else:
        sid = require_session()
        if not sid:
            return jsonify({"error": "No session test set; upload a CSV instead."}), 400
        err, _ = _test_lock_gate(sid, names, _unlock_reason(body))
        if err is not None:
            return err
        df = pd.read_csv(store.session_path(sid, "test.csv"))

    roc_series, pr_series, dca_series = [], [], []
    dca_base = None
    out = {"models": [], "task": None}
    evaluated = []
    for name in names:
        try:
            mdl = _load_model(name)
        except Exception as e:
            out["models"].append({"name": name, "error": f"Could not load model: {e}"})
            continue
        pipe, feat_cols = mdl["pipe"], mdl["feat_cols"]
        outcome, task = mdl["outcome"], mdl["task"]
        # external rows get the model's own training-fitted fills/indicators
        data = _apply_model_missing(df, mdl) if external else df
        if outcome is None or outcome not in data.columns:
            out["models"].append({"name": name, "error": "Outcome column not found in data."})
            continue
        if feat_cols is None:
            feat_cols = [c for c in data.columns if c != outcome]
        absent = [c for c in feat_cols if c not in data.columns]
        if absent:
            out["models"].append({"name": name, "error": f"Missing feature column(s): {absent[:5]}"})
            continue
        out["task"] = task
        classes = mdl.get("classes")
        if user_threshold is not None:
            threshold, thr_source, how = user_threshold, "user", "user"
        else:
            saved = mdl.get("threshold")
            threshold = 0.5 if saved is None else float(saved)
            # how the saved threshold was picked, for the methods text
            thr_source, how = "saved", mdl.get("threshold_method") or "fixed"
        try:
            metrics, coords = ml.evaluate(pipe, data, outcome, feat_cols, task, threshold, classes)
        except Exception as e:
            out["models"].append({"name": name, "error": f"Could not evaluate: {e}"})
            continue
        evaluated.append(name)
        recalibrated = bool(mdl.get("recalibrated"))
        if sid:
            model_type, model_outcome = _model_identity(name, mdl)
            if external:
                inputs = {"model": name, "source": "external", **ext_info}
            else:
                inputs = {"model": name, "source": "internal_test", "n": int(len(df))}
            inputs.update({"model_type": model_type, "outcome": model_outcome,
                           "task": task, "threshold": threshold})
            outputs = {"metrics": metrics, "threshold_method": how,
                       "threshold_source": thr_source, "recalibrated": recalibrated}
            if thr_source == "saved" and mdl.get("threshold_target") is not None:
                outputs["threshold_target"] = mdl["threshold_target"]
            if task == "binary":
                outputs["dca"] = True
            analysis_log.log_action(sid, "test_model", inputs=inputs,
                                    outputs=_sanitize(outputs))
        entry = {"name": name, "metrics": metrics, "coords": coords,
                 "threshold": threshold, "threshold_method": thr_source,
                 "recalibrated": recalibrated}
        if task in ("binary", "continuous"):
            corr = ml.feature_outcome_correlations(data, outcome, feat_cols, task, classes)
            entry["correlations"] = corr.head(20).to_dict(orient="records")
        if task == "binary":
            roc_series.append({"name": name, **coords["roc"]})
            pr_series.append({"name": name, **coords["pr"]})
            entry["confusion_png"] = plots.fig_to_png(
                plots.confusion_plot(coords["confusion"], title=f"{name}: confusion"))
            entry["calibration_png"] = plots.fig_to_png(_flexible_calibration_fig(
                coords, name=name, title=f"{name}: calibration"))
            entry["calibration_stats"] = coords["calibration_stats"]
            dca = coords["dca"]
            if dca_base is None:
                dca_base = {"thresholds": dca["threshold"], "treat_all": dca["treat_all"]}
            dca_series.append({"name": name, "net_benefit": dca["net_benefit"]})
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
    if dca_series:
        # every model is on the same 0.01 to 0.99 grid; treat all comes from
        # the first model's data (the same rows unless outcomes differ)
        out["dca"] = {**dca_base, "models": dca_series}
        out["dca_png"] = plots.fig_to_png(plots.dca_plot(
            dca_series, dca_base["treat_all"], dca_base["thresholds"]))
    out["source"] = "external" if external else "internal_test"
    if not external:
        out["test_lock"] = _record_test_use(sid, evaluated)
    return jsonify(out)


@bp.route("/api/model/features")
def model_features():
    """Feature list for a saved model (for building a single-prediction form)."""
    name = request.args.get("name")
    try:
        mdl = _load_model(name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"name": name, "features": mdl["feat_cols"],
                    "derived_features": _derived_features(mdl),
                    "outcome": mdl["outcome"], "task": mdl["task"],
                    "classes": mdl["classes"]})


@bp.route("/api/model/<name>/odds-ratios")
def model_odds_ratios(name):
    """The adjusted odds ratio table saved with a LogisticRegression model
    trained on a binary outcome (ml.adjusted_odds_ratios, fitted on the
    training partition)."""
    meta = None
    if store.model_exists(name):
        try:
            with open(store.model_meta_path(name)) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            meta = None
    if not meta or not isinstance(meta.get("odds_ratios"), dict):
        return jsonify({"error": "No adjusted odds ratios for this model. They are saved "
                                 "for a LogisticRegression model trained on a binary "
                                 "outcome."}), 404
    return jsonify({**meta["odds_ratios"], "name": name, "model": meta.get("model"),
                    "outcome": meta.get("outcome")})


@bp.route("/api/predict-single", methods=["POST"])
def predict_single():
    """Upload a JSON row (+ choose a model) -> single prediction with probability."""
    try:
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
        if not isinstance(row, dict):
            return jsonify({"error": "The row must be a JSON object of {feature: value}."}), 400

        feat_cols = mdl["feat_cols"] or list(row.keys())
        task = mdl["task"] or "binary"
        if mdl.get("missing_handling"):
            # same training-fitted fills and indicators as the model saw;
            # indicators always come from whether the value was given
            derived = set(_derived_features(mdl))
            frame = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_cols
                                   if c not in derived}])
            row = _apply_model_missing(frame, mdl).iloc[0].to_dict()
        result = ml.predict_single(mdl["pipe"], row, feat_cols, task, mdl.get("classes"))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Could not predict: {e}"}), 400
    return jsonify(result)


# ===========================================================================
# 5. spline
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
# 6. clinical epi
# ===========================================================================
def _cells(b):
    return {k: b[k] for k in ("a", "b", "c", "d")}


def _time_points(value):
    """optional list of follow-up times for risk tables, None when not given."""
    if value is None or value == "" or value == []:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("time_points must be a list of numbers.")
    try:
        pts = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError("time_points must be a list of numbers.")
    if any(not math.isfinite(v) or v < 0 for v in pts):
        raise ValueError("time_points must be numbers of 0 or more.")
    return pts


@bp.route("/api/epi/or", methods=["POST"])
def epi_or():
    b = request.get_json(force=True)
    try:
        res = epi.two_by_two(b["a"], b["b"], b["c"], b["d"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid 2x2 table: {e}"}), 400
    sid = require_session()
    if sid:     # the calculator works without a dataset, it's only logged with one
        analysis_log.log_action(sid, "epi_or", inputs=_cells(b), outputs=_sanitize({
            k: res.get(k) for k in ("odds_ratio", "or_ci", "risk_ratio", "rr_ci",
                                    "abs_risk_diff", "rd_ci", "nnt", "nnt_ci",
                                    "p_value", "fisher_p_value", "corrected")}))
    return jsonify(res)


@bp.route("/api/epi/nnt", methods=["POST"])
def epi_nnt():
    b = request.get_json(force=True)
    try:
        res = epi.nnt_nnh(b["a"], b["b"], b["c"], b["d"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid 2x2 table: {e}"}), 400
    sid = require_session()
    if sid:
        analysis_log.log_action(sid, "epi_nnt", inputs=_cells(b), outputs=_sanitize({
            k: res.get(k) for k in ("measure", "value", "abs_risk_diff", "rd_ci",
                                    "nnt_ci", "risk_exposed", "risk_unexposed")}))
    return jsonify(res)


@bp.route("/api/epi/km", methods=["POST"])
def epi_km():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(force=True)
    try:
        time_points = _time_points(b.get("time_points"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    try:
        res = epi.kaplan_meier(df, b["time"], b["event"], b.get("group") or None,
                               time_points=time_points)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    res["png"] = plots.fig_to_png(plots.km_plot(res["curves"], risk_table=res["risk_table"]))
    analysis_log.log_action(sid, "epi_km",
                            inputs={"time": b["time"], "event": b["event"],
                                    "group": b.get("group"), "time_points": time_points},
                            outputs=_sanitize({"n_curves": len(res["curves"]),
                                               "n": sum(c.get("n", 0) for c in res["curves"]),
                                               "logrank": res["logrank"]}))
    return jsonify(res)


@bp.route("/api/epi/cox", methods=["POST"])
def epi_cox():
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(force=True)
    categorical = b.get("categorical") or []
    reference = b.get("reference") or {}
    ph_transform = b.get("ph_transform") or "rank"
    if not isinstance(categorical, (list, tuple)) or not isinstance(reference, dict):
        return jsonify({"error": "categorical must be a list of columns and reference "
                                 "an object of {column: level}."}), 400
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    try:
        res = epi.cox_ph(df, b["time"], b["event"], b["covariates"],
                         categorical=list(categorical), reference=reference,
                         ph_transform=ph_transform)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    analysis_log.log_action(sid, "epi_cox",
                            inputs={"time": b["time"], "event": b["event"],
                                    "covariates": b["covariates"],
                                    "categorical": res.get("categorical"),
                                    "reference": res.get("reference_levels"),
                                    "ph_transform": ph_transform},
                            outputs=_sanitize({"concordance": res["concordance"], "n": res["n"],
                                               "n_events": res.get("n_events"),
                                               "ph_test": {"transform": ph_transform,
                                                           "global": res["ph_test"]["global"]}}))
    return jsonify(res)


@bp.route("/api/epi/cif", methods=["POST"])
def epi_cif():
    """Cumulative incidence with competing risks (Aalen-Johansen), Gray's
    test across groups and the naive 1 minus Kaplan-Meier for comparison."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(silent=True) or {}
    if not b.get("time") or not b.get("event"):
        return jsonify({"error": "Choose a time column and an event column."}), 400
    try:
        time_points = _time_points(b.get("time_points"))
        k = _num_or_none(b.get("event_of_interest"), "The event of interest")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    k = 1 if k is None else (int(k) if float(k).is_integer() else k)
    group = b.get("group") or None
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    try:
        res = epi.cumulative_incidence(df, b["time"], b["event"], event_of_interest=k,
                                       group_col=group, time_points=time_points)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    res["png"] = plots.fig_to_png(plots.cif_plot(res["curves"], naive=res["naive"],
                                                 risk_table=res["risk_table"]))
    gray = res.get("gray_test") or {}
    analysis_log.log_action(sid, "epi_cif",
                            inputs={"time": b["time"], "event": b["event"],
                                    "event_of_interest": k, "group": group},
                            outputs=_sanitize({"n": sum(c["n"] for c in res["curves"]),
                                               "n_curves": len(res["curves"]),
                                               "gray_p": gray.get("p_value")}))
    return jsonify(res)


@bp.route("/api/epi/hl", methods=["POST"])
def epi_hl():
    """Hosmer-Lemeshow from a saved model applied to the test set. This uses
    the internal test partition, so it is under the same test-set lock as
    /api/test."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    b = request.get_json(force=True)
    try:
        mdl = _load_model(b["model"])
    except Exception as e:
        return jsonify({"error": f"Could not load model: {e}"}), 400
    if mdl["task"] != "binary" or not mdl["feat_cols"]:
        return jsonify({"error": "Hosmer-Lemeshow needs a binary model trained in ClinTAB-ML."}), 400
    err, _ = _test_lock_gate(sid, [b["model"]], _unlock_reason(b))
    if err is not None:
        return err
    df = pd.read_csv(store.session_path(sid, "test.csv"))
    if mdl["outcome"] not in df.columns:
        return jsonify({"error": "Outcome column not found in the test set."}), 400
    y, _ = ml.encode_outcome(df[mdl["outcome"]], "binary", mdl.get("classes"))
    mask = ~pd.isna(y)
    try:
        scores = ml.predict_scores(mdl["pipe"], df[mdl["feat_cols"]][mask], "binary")
    except Exception as e:
        return jsonify({"error": f"Could not score the test set: {e}"}), 400
    res = epi.hosmer_lemeshow(np.asarray(y)[mask], scores)
    res["png"] = plots.fig_to_png(plots.calibration_hl_plot(
        res["mean_pred"], res["observed_rate"]))
    _record_test_use(sid, [b["model"]])
    model_type, model_outcome = _model_identity(b["model"], mdl)
    analysis_log.log_action(sid, "epi_hl",
                            inputs={"model": b["model"], "source": "internal_test",
                                    "model_type": model_type, "outcome": model_outcome},
                            outputs={"hl_statistic": res["hl_statistic"],
                                     "p_value": res["p_value"]})
    return jsonify(res)


# ===========================================================================
# 6b. sample size (riley et al. / pmsampsize), works without a dataset
# ===========================================================================
def _train_outcome_counts(sid, outcome, exclude):
    """(n, events, n_parameters, task) from the session's training
    partition: rows with an outcome, rows in the positive (second) class,
    and candidate parameters for the features training would use."""
    meta = store.load_meta(sid) or {}
    df = pd.read_csv(store.session_path(sid, "train.csv"))
    if outcome not in df.columns:
        raise ValueError(f"Outcome column {outcome!r} is not in the training data.")
    task = ml.determine_task(df[outcome])
    feat_cols = _feature_cols(meta, df, outcome, exclude)
    k = ml.count_parameters(df, feat_cols, meta.get("coltypes", {}))
    if task == "binary":
        y, _ = ml.encode_outcome(df[outcome], task)
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        return int(len(y)), int(np.sum(y == 1)), k, task
    n = int(df[outcome].notna().sum())
    return n, None, k, task


def _exclude_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [c.strip() for c in value.split(",") if c.strip()]
    return list(value)


@bp.route("/api/samplesize", methods=["POST"])
def sample_size():
    b = request.get_json(silent=True) or {}
    otype = str(b.get("outcome_type") or "binary").strip().lower()
    if otype not in ("binary", "continuous"):
        return jsonify({"error": "outcome_type must be binary or continuous."}), 400
    compare = b.get("compare")
    sid = require_session()
    if compare and not sid:
        return jsonify({"error": "Comparing with the data needs an active dataset."}), 400
    if compare is not None and not isinstance(compare, dict):
        return jsonify({"error": "compare must be an object {outcome, exclude}."}), 400

    def num(key, what):
        return _num_or_none(b.get(key), what)

    try:
        n_par = num("n_parameters", "The number of parameters")
        if n_par is None:
            raise ValueError("Give the number of candidate predictor parameters.")
        opts = {}
        if otype == "binary":
            prevalence = num("prevalence", "The prevalence")
            if prevalence is None:
                raise ValueError("Give the anticipated outcome prevalence.")
            for key, what in (("cstatistic", "The C statistic"),
                              ("csrsquared", "The Cox-Snell R2"),
                              ("nagrsquared", "The Nagelkerke R2"),
                              ("shrinkage", "Shrinkage"), ("delta", "delta"),
                              ("margin", "margin")):
                v = num(key, what)
                if v is not None:
                    opts[key] = v
            result = samplesize.binary_sample_size(prevalence, n_par, **opts)
        else:
            needed = {}
            for key, what in (("rsquared", "R2"), ("intercept", "The mean outcome (intercept)"),
                              ("sd", "The outcome SD")):
                needed[key] = num(key, what)
                if needed[key] is None:
                    raise ValueError("Give the anticipated R2, the mean outcome "
                                     "(intercept) and the outcome SD.")
            for key, what in (("shrinkage", "Shrinkage"), ("delta", "delta"),
                              ("mmoe", "mmoe")):
                v = num(key, what)
                if v is not None:
                    opts[key] = v
            result = samplesize.continuous_sample_size(
                needed["rsquared"], n_par, needed["intercept"], needed["sd"], **opts)
        comparison = None
        if compare:
            outcome = compare.get("outcome")
            n, events, k, task = _train_outcome_counts(
                sid, outcome, _exclude_list(compare.get("exclude")))
            if otype == "binary" and task != "binary":
                raise ValueError(f"{outcome} is not a binary outcome, so it can't be "
                                 "compared with a binary sample size.")
            if otype == "continuous" and task != "continuous":
                raise ValueError(f"{outcome} is not a continuous outcome, so it can't be "
                                 "compared with a continuous sample size.")
            # a continuous outcome has no events, every row counts
            comparison = samplesize.compare_with_data(
                result, n, events if events is not None else n, k)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    if sid:
        a = result.get("assumptions", {})
        inputs = {"outcome_type": otype, "n_parameters": a.get("n_parameters"),
                  "shrinkage": a.get("shrinkage")}
        if otype == "binary":
            inputs["prevalence"] = a.get("prevalence")
            for key in ("cstatistic", "csrsquared", "nagrsquared"):
                if key in opts:
                    inputs[key] = opts[key]
        else:
            inputs["rsquared"] = a.get("rsquared")
        outputs = {"min_n": result.get("min_n"), "min_events": result.get("min_events"),
                   "epp": result.get("epp"), "criteria": result.get("criteria")}
        if comparison:
            outputs["compare"] = {k: comparison.get(k) for k in ("adequate", "n", "events")}
        analysis_log.log_action(sid, "sample_size", inputs=_sanitize(inputs),
                                outputs=_sanitize(outputs))
    return jsonify({"result": result, "compare": comparison})


@bp.route("/api/samplesize/defaults")
def sample_size_defaults():
    """Prevalence, parameters, n and events from the training partition, so
    the sample size form can be prefilled. Binary outcomes only."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    outcome = request.args.get("outcome")
    if not outcome:
        return jsonify({"error": "Choose an outcome."}), 400
    try:
        n, events, k, task = _train_outcome_counts(
            sid, outcome, _exclude_list(request.args.get("exclude")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if task != "binary":
        return jsonify({"error": f"{outcome} is not a binary outcome. Defaults are only "
                                 "filled in for a binary outcome."}), 400
    return jsonify({"prevalence": round(events / n, 4) if n else None,
                    "n_parameters": k, "n": n, "events": events})


# ===========================================================================
# 6c. teaching mode: leakage demo on the user's own data
# ===========================================================================
TEACHING_MAX_SEEDS = 5


@bp.route("/api/teaching", methods=["POST"])
def teaching_config():
    """Store the leakage demo configuration; it runs on the SSE stream."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    meta = store.load_meta(sid) or {}
    if meta.get("stage") != "confirmed":
        return jsonify({"error": "Confirm the dataset first."}), 400
    b = request.get_json(silent=True) or {}
    outcome = b.get("outcome")
    exclude = _exclude_list(b.get("exclude"))
    model = b.get("model") or "logistic"
    cols = list(pd.read_csv(store.session_path(sid, "full.csv"), nrows=0).columns)
    if outcome not in cols:
        return jsonify({"error": f"Outcome column {outcome!r} is not in the data."}), 400
    unknown = [c for c in exclude if c not in cols]
    if unknown:
        return jsonify({"error": f"Unknown column(s) to exclude: {unknown}"}), 400
    if model not in teaching.MODELS:
        return jsonify({"error": f"Unknown model {model!r}. Use one of "
                                 f"{list(teaching.MODELS)}."}), 400
    try:
        n_seeds = int(b["n_seeds"]) if b.get("n_seeds") not in (None, "") else 3
        max_rows = int(b["max_rows"]) if b.get("max_rows") not in (None, "") else 20000
    except (TypeError, ValueError):
        return jsonify({"error": "n_seeds and max_rows must be whole numbers."}), 400
    if not 1 <= n_seeds <= TEACHING_MAX_SEEDS:
        return jsonify({"error": f"n_seeds must be between 1 and {TEACHING_MAX_SEEDS}."}), 400
    if max_rows < 1:
        return jsonify({"error": "max_rows must be at least 1."}), 400
    y = pd.read_csv(store.session_path(sid, "full.csv"), usecols=[outcome])[outcome]
    if y.dropna().nunique() != 2:
        return jsonify({"error": f"Teaching mode needs a binary outcome (exactly 2 values), "
                                 f"but {outcome} has {int(y.dropna().nunique())}."}), 400
    meta["teaching_config"] = {"outcome": outcome, "exclude": exclude, "model": model,
                               "seeds": list(range(1, n_seeds + 1)), "max_rows": max_rows}
    store.save_meta(sid, meta)
    return jsonify({"ok": True})


@bp.route("/api/teaching/stream")
def teaching_stream():
    """SSE: progress messages, then the leakage demo result and its plot.
    The demo runs on a background thread (joblib threads inside, never
    processes, so it also works in the packaged desktop app)."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    meta = store.load_meta(sid) or {}
    cfg = meta.get("teaching_config")
    if not cfg:
        return jsonify({"error": "No teaching configuration. Call /api/teaching first."}), 400
    df = pd.read_csv(store.session_path(sid, "full.csv"))
    coltypes = meta.get("coltypes", {})
    feat_cols = _feature_cols(meta, df, cfg["outcome"], cfg.get("exclude"))

    def generate():
        q = queue.Queue()
        box = {}

        def work():
            try:
                box["result"] = teaching.leakage_demo(
                    df, cfg["outcome"], feat_cols, coltypes=coltypes, model=cfg["model"],
                    seeds=cfg["seeds"], max_rows=cfg["max_rows"],
                    progress=lambda m: q.put(("progress", str(m))))
            except Exception as e:
                box["error"] = e
            finally:
                q.put(("done", None))

        threading.Thread(target=work, daemon=True).start()
        yield _sse({"event": "progress", "message": "Starting the leakage demo"})
        while True:
            try:
                kind, msg = q.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"           # sse comment, keeps proxies from timing out
                continue
            if kind == "done":
                break
            yield _sse({"event": "progress", "message": msg})
        if "error" in box:
            yield _sse({"event": "error", "error": str(box["error"])})
            return
        result = box["result"]
        try:
            png = plots.fig_to_png(plots.teaching_plot(result))
        except Exception:
            png = None
        arms = [{"arm": a["arm"],
                 **{q_: (a.get(q_) or {}).get("mean")
                    for q_ in ("reported_auroc", "lockbox_auroc", "optimism")}}
                for a in result.get("arms", [])]
        analysis_log.log_action(sid, "leakage_demo",
                                inputs={"outcome": cfg["outcome"], "model": cfg["model"],
                                        "seeds": cfg["seeds"], "max_rows": cfg["max_rows"]},
                                outputs=_sanitize({"arms": arms}))
        yield _sse({"event": "complete", "result": result, "png": png})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ===========================================================================
# 7. pdf export (any chart), "render this data as a downloadable pdf"
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
        return plots.km_plot(data["curves"], risk_table=data.get("risk_table"))
    if kind == "hl":
        return plots.calibration_hl_plot(data["mean_pred"], data["observed_rate"])
    if kind == "cif":
        return plots.cif_plot(data["curves"], naive=data.get("naive"),
                              risk_table=data.get("risk_table"))
    if kind == "dca":
        # the /api/test "dca" object: {thresholds, treat_all, models}
        return plots.dca_plot(data.get("models") or data["curves"], data["treat_all"],
                              data["thresholds"])
    if kind == "calibration_flexible":
        return _flexible_calibration_fig(data)
    if kind == "teaching":
        return plots.teaching_plot(data.get("result") or data)
    if kind == "flow":
        return plots.flow_plot(data["flow"], data.get("n_start"),
                               title=data.get("title") or "Cohort")
    raise ValueError(f"Unknown chart kind: {kind}")


@bp.route("/api/render/pdf", methods=["POST"])
def render_pdf():
    b = request.get_json(force=True)
    try:
        fig = _build_fig(b["kind"], b["data"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Could not render that chart: {e}"}), 400
    pdf = plots.fig_to_pdf_bytes(fig)
    return send_file(io.BytesIO(pdf), mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"{_safe(b.get('name', b['kind']))}.pdf")


# ===========================================================================
# 8. reports (methods text + tripod+ai coverage), from the analysis log
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


@bp.route("/api/report/verify-log")
def report_verify_log():
    """Check the analysis log's hash chain: {ok, n_entries, first_bad_line, legacy}."""
    sid = require_session()
    if not sid:
        return jsonify({"error": "No active session."}), 400
    return jsonify(analysis_log.verify_log(sid))
