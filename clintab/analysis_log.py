"""analysis_log.py
Append-only record of every analysis action taken in a session: each model
trained, each spline fit, each epi calculation, with its inputs, key
outputs, and a timestamp. meta.json (see store.py) tracks current session
state, column types, split config, but overwrites itself as things
change. This is the complement: a running history that nothing rewrites, so
it's the single source of truth for reconstructing what actually happened in
a session, for reproducibility and reporting.

Pure Python, no Flask, routes.py calls log_action() after a real action
happens; nothing here knows about HTTP.

Stored as JSON Lines (one JSON object per line) at
runtime/sessions/<session_id>/analysis_log.jsonl, appending is just
writing one more line, so entries already logged are never rewritten.

Tamper evidence: every entry carries "prev_hash", the sha256 hex digest of
the previous line exactly as stored (without its line ending), or "" for
the first line. verify_log() walks that chain, so editing, inserting or
deleting a line before the last one shows up. It is tamper evident, not
tamper proof: someone who rewrites the whole chain, or edits or cuts the
last line(s), is not detected. Logs written before the chain existed have no prev_hash and
are still read normally; verify_log() reports them as legacy.
"""
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import threading
import time

import clintab
from clintab import store

# one lock per process, so two requests logging at once can't both chain
# onto the same previous line
_LOCK = threading.Lock()

# (distribution name, import name) for the provenance record
_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("imbalanced-learn", "imblearn"),
    ("statsmodels", "statsmodels"),
    ("lifelines", "lifelines"),
    ("scipy", "scipy"),
]


def log_path(session_id):
    return store.session_path(session_id, "analysis_log.jsonl")


def line_hash(line):
    """sha256 hex of one stored line, line ending stripped."""
    return hashlib.sha256(line.rstrip("\r\n").encode("utf-8")).hexdigest()


def _last_line(path):
    """(last non-empty line, whether the file ends in a newline). Reads
    backwards from the end so long logs aren't read in full."""
    if not os.path.exists(path):
        return None, True
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None, True
        f.seek(size - 1)
        ends_nl = f.read(1) == b"\n"
        buf, pos = b"", size
        while pos > 0:
            step = min(4096, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            body = buf.rstrip(b"\r\n")
            if b"\n" in body:
                return body.rsplit(b"\n", 1)[1].decode("utf-8", "replace"), ends_nl
        body = buf.rstrip(b"\r\n")
        return (body.decode("utf-8", "replace") if body else None), ends_nl


def _pkg_version(dist, module):
    try:
        return importlib.metadata.version(dist)
    except Exception:
        pass
    try:                                    # frozen builds may lack dist metadata
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:
        return None


def environment():
    """Software versions for the provenance record (no network, nothing
    leaves the machine)."""
    env = {"python": platform.python_version(), "clintab": clintab.__version__}
    for dist, module in _PACKAGES:
        env[dist] = _pkg_version(dist, module)
    env["platform"] = platform.platform()
    return env


def log_action(session_id, action, inputs=None, outputs=None):
    """Append one structured entry and return it.

    action: short string identifying what happened, e.g. 'train_model',
    'fit_spline', 'epi_or'.
    inputs / outputs: plain-JSON-safe dicts describing what went in and what
    came out, e.g. inputs={'model': 'LogisticRegression',
    'outcome': 'mortality_30d'}, outputs={'auroc': 0.81}.
    """
    store.session_dir(session_id, create=True)
    path = log_path(session_id)
    with _LOCK:
        prev, ends_nl = _last_line(path)
        entry = {
            "timestamp": time.time(),
            "action": action,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "prev_hash": line_hash(prev) if prev is not None else "",
        }
        line = json.dumps(entry, default=str)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            if not ends_nl:
                f.write("\n")           # a half-written line stays on its own line
            f.write(line + "\n")
    return entry


def read_log(session_id):
    """Return every logged entry for a session, in the order they happened.
    Empty list if the session has no log yet. Old logs without prev_hash
    read the same way."""
    p = log_path(session_id)
    if not os.path.exists(p):
        return []
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue            # half-written last line, e.g. after a crash
            if isinstance(entry, dict) and "action" in entry:
                entry.setdefault("inputs", {})
                entry.setdefault("outputs", {})
                entries.append(entry)
    return entries


def verify_log(session_id):
    """Check the prev_hash chain.

    Returns {"ok", "n_entries", "first_bad_line", "legacy"}:
      ok             True if every chained line matches the line before it
      n_entries      non-empty lines in the log
      first_bad_line 1-based line number of the first break, or None
      legacy         True if the log starts with entries written before the
                     chain existed (those can't be checked)
    A line that isn't valid JSON, or an entry without prev_hash after the
    chain has started, counts as a break.
    """
    p = log_path(session_id)
    result = {"ok": True, "n_entries": 0, "first_bad_line": None, "legacy": False}
    if not os.path.exists(p):
        return result
    with open(p, encoding="utf-8", newline="") as f:
        raw = f.read()
    prev = None
    chained = False
    for n, line in enumerate(raw.split("\n"), start=1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        result["n_entries"] += 1
        bad = False
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            entry, bad = None, True
        if not bad:
            if not isinstance(entry, dict):
                bad = True
            elif "prev_hash" not in entry:
                if chained:
                    bad = True
                else:
                    result["legacy"] = True
            else:
                chained = True
                expected = line_hash(prev) if prev is not None else ""
                bad = entry["prev_hash"] != expected
        if bad and result["first_bad_line"] is None:
            result["ok"] = False
            result["first_bad_line"] = n
        prev = line
    return result
