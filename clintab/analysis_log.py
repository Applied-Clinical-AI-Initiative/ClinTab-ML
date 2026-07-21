"""analysis_log.py
Append-only record of every analysis action taken in a session: each model
trained, each spline fit, each epi calculation, with its inputs, key
outputs, and a timestamp. meta.json (see store.py) tracks current session
state -- column types, split config -- but overwrites itself as things
change. This is the complement: a running history that nothing rewrites, so
it's the single source of truth for reconstructing what actually happened in
a session, for reproducibility and reporting.

Pure Python, no Flask -- routes.py calls log_action() after a real action
happens; nothing here knows about HTTP.

Stored as JSON Lines (one JSON object per line) at
runtime/sessions/<session_id>/analysis_log.jsonl -- appending is just
writing one more line, so entries already logged are never rewritten.
"""
import json
import os
import time

from clintab import store


def log_path(session_id):
    return store.session_path(session_id, "analysis_log.jsonl")


def log_action(session_id, action, inputs=None, outputs=None):
    """Append one structured entry and return it.

    action: short string identifying what happened, e.g. 'train_model',
    'fit_spline', 'epi_or'.
    inputs / outputs: plain-JSON-safe dicts describing what went in and what
    came out, e.g. inputs={'model': 'LogisticRegression',
    'outcome': 'mortality_30d'}, outputs={'auroc': 0.81}.
    """
    store.session_dir(session_id, create=True)
    entry = {
        "timestamp": time.time(),
        "action": action,
        "inputs": inputs or {},
        "outputs": outputs or {},
    }
    with open(log_path(session_id), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_log(session_id):
    """Return every logged entry for a session, in the order they happened.
    Empty list if the session has no log yet."""
    p = log_path(session_id)
    if not os.path.exists(p):
        return []
    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
