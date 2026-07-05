"""store.py
Filesystem layout + session metadata helpers. Knows nothing about Flask or
sklearn -- just paths and JSON. Keeps the data-saving clean and in one place.

Layout (all created automatically on first use):

    ClinTAB-ML-Foundry/
      runtime/
        sessions/<session_id>/
            raw.csv      # exactly what the user uploaded
            full.csv     # cleaned full dataset (after missing-value handling)
            train.csv    # split partitions
            val.csv
            test.csv
            meta.json    # column types, split config, etc.
        exports/         # generated CSVs / plot files offered for download
      models/<name>.pkl  # trained model pickles (+ <name>.meta.json sidecar)
"""
import json
import os
import time
import uuid

BASE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(BASE, "runtime")
SESSIONS = os.path.join(RUNTIME, "sessions")
EXPORTS = os.path.join(RUNTIME, "exports")
MODELS = os.path.join(BASE, "models")


def ensure_dirs():
    for d in (RUNTIME, SESSIONS, EXPORTS, MODELS):
        os.makedirs(d, exist_ok=True)


def new_session_id():
    return uuid.uuid4().hex[:12]


def session_dir(session_id, create=False):
    d = os.path.join(SESSIONS, session_id)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def session_path(session_id, filename):
    return os.path.join(session_dir(session_id), filename)


def save_meta(session_id, meta):
    meta["updated"] = time.time()
    with open(session_path(session_id, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_meta(session_id):
    p = session_path(session_id, "meta.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def session_exists(session_id):
    return bool(session_id) and os.path.exists(session_path(session_id, "meta.json"))


def model_path(name):
    return os.path.join(MODELS, name + ".pkl")


def model_meta_path(name):
    return os.path.join(MODELS, name + ".meta.json")


def list_models():
    ensure_dirs()
    out = []
    for fn in sorted(os.listdir(MODELS)):
        if fn.endswith(".pkl"):
            name = fn[:-4]
            meta = {}
            mp = model_meta_path(name)
            if os.path.exists(mp):
                with open(mp) as f:
                    meta = json.load(f)
            out.append({"name": name, "meta": meta})
    return out


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")
