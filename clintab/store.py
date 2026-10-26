"""store.py
Filesystem layout + session metadata helpers. Knows nothing about Flask or
sklearn, just paths and JSON. Keeps the data-saving clean and in one place.

Layout (all created automatically on first use):

    ClinTAB-ML/
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
import re
import shutil
import time
import uuid

# clintab_home moves all generated data (the desktop app points it at the
# user's app data folder). default is the repo root.
BASE = os.environ.get("CLINTAB_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(BASE, "runtime")
SESSIONS = os.path.join(RUNTIME, "sessions")
EXPORTS = os.path.join(RUNTIME, "exports")
MODELS = os.path.join(BASE, "models")


def ensure_dirs():
    for d in (RUNTIME, SESSIONS, EXPORTS, MODELS):
        os.makedirs(d, exist_ok=True)


_SID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def new_session_id():
    return uuid.uuid4().hex[:12]


def safe_name(name, max_len=120):
    """File-system safe version of a model/export name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))[:max_len]


def model_base_name(model, outcome, ts):
    """<model>_<outcome>_<timestamp>. The prefix is trimmed, the timestamp never
    is, so a long outcome name can't make two runs overwrite each other."""
    return f"{safe_name(f'{model}_{outcome}', max_len=60)}_{ts}"


def valid_session_id(session_id):
    return isinstance(session_id, str) and _SID_RE.fullmatch(session_id) is not None


def session_dir(session_id, create=False):
    if not valid_session_id(session_id):
        raise ValueError(f"Invalid session id: {session_id!r}")
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
    return valid_session_id(session_id) and os.path.exists(session_path(session_id, "meta.json"))


def _session_ids():
    """Session folders only, a stray file like .DS_Store isn't one."""
    return [sid for sid in os.listdir(SESSIONS)
            if valid_session_id(sid) and os.path.isdir(os.path.join(SESSIONS, sid))]


def list_sessions():
    ensure_dirs()
    out = []
    for sid in _session_ids():
        meta = load_meta(sid)
        if meta:
            out.append({"session_id": sid, **meta})
    return out


def delete_session(session_id):
    d = session_dir(session_id)
    if os.path.exists(d):
        shutil.rmtree(d)


def delete_all_sessions():
    ensure_dirs()
    for sid in _session_ids():
        delete_session(sid)


def model_path(name):
    return os.path.join(MODELS, safe_name(name) + ".pkl")


def model_meta_path(name):
    return os.path.join(MODELS, safe_name(name) + ".meta.json")


def model_exists(name):
    return bool(name) and os.path.exists(model_path(name))


def delete_model(name):
    for p in (model_path(name), model_meta_path(name)):
        if os.path.exists(p):
            os.remove(p)


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
