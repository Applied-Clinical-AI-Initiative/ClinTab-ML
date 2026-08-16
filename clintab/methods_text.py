"""methods_text.py
Deterministic methods-section text, assembled from a session's analysis log
(clintab/analysis_log.py). Every sentence comes from a fixed template filled
in with structured log fields -- there is no generative/LLM call anywhere in
here, so the same log always produces the exact same text, and the wording
is defensible to paste into a manuscript. Pure Python, no Flask.
"""
from clintab import analysis_log

_METRIC_ORDER = ["AUROC", "AUPRC", "Accuracy", "BalancedAccuracy", "MacroF1",
                 "R2", "RMSE", "MAE"]

# Internal scoring keys (clintab/ml.py, static/app.js #scoreSel) rendered as
# the terms a manuscript would actually use, spelling out acronyms on first
# use per standard scientific-writing convention.
_SCORING_LABELS = {
    "roc": "the area under the receiver operating characteristic curve (AUROC)",
    "auprc": "the area under the precision-recall curve (AUPRC)",
    "f1": "F1 score",
    "f2": "F2 score",
    "recall": "recall (sensitivity)",
    "precision": "precision",
    "accuracy": "accuracy",
    "r2": "R-squared",
    "mae": "mean absolute error",
    "rmse": "root mean squared error",
}


def _article(word):
    """'A' or 'An', by the leading sound of word (e.g. 'An ElasticNet')."""
    return "An" if word and word[0].upper() in "AEIOU" else "A"


def _scoring_label(key):
    return _SCORING_LABELS.get(key, key or "the default metric")


def _format_metrics(metrics):
    if not metrics:
        return ""
    picked = [(k, metrics[k]) for k in _METRIC_ORDER if k in metrics]
    if not picked:
        return ""
    bits = ", ".join(f"{k}={v}" for k, v in picked)
    return f" (validation {bits})"


def _sentence_upload(e):
    i, o = e["inputs"], e["outputs"]
    filename = i.get("filename", "the uploaded file")
    return (f"A dataset comprising {o.get('n_rows', '?')} rows and {o.get('n_cols', '?')} "
            f"columns ({filename}) was used for analysis.")


def _sentence_confirm(e):
    i, o = e["inputs"], e["outputs"]
    if i.get("method") == "stratified" and i.get("stratify_col"):
        split_txt = f"a split stratified on {i['stratify_col']}"
    elif i.get("method") == "stratified":
        split_txt = "a stratified split"
    else:
        split_txt = "a random split"
    sentence = (f"The data were partitioned into training (n={o.get('n_train', '?')}), "
                f"validation (n={o.get('n_val', '?')}), and test (n={o.get('n_test', '?')}) "
                f"sets using {split_txt}.")
    if i.get("smote_pref"):
        sentence += (" The Synthetic Minority Over-sampling Technique (SMOTE) was applied "
                     "to the training fold only, to address class imbalance.")
    return sentence


def _sentence_train_model(e):
    i, o = e["inputs"], e["outputs"]
    model = i.get("model", "model")
    cv_folds = i.get("cv_folds")
    tuning = (f"{cv_folds}-fold cross-validation on the training set" if cv_folds
              else "a held-out validation set")
    smote_clause = ", with SMOTE applied to the training fold," if i.get("smote") else ""
    return (f"{_article(model)} {model} model was trained to predict "
            f"{i.get('outcome', 'the outcome')}{smote_clause} with hyperparameters "
            f"tuned by {_scoring_label(i.get('scoring'))} scoring against "
            f"{tuning}{_format_metrics(o.get('metrics'))}.")


def _train_group_key(e):
    """Entries that share this key were trained under the same protocol
    (same outcome/scoring/tuning/SMOTE). The first entry in a group states
    the full protocol; later ones in the group refer back to it instead of
    repeating the same sentence template verbatim."""
    i = e["inputs"]
    return (i.get("outcome"), i.get("scoring"), i.get("cv_folds"), i.get("smote"))


def _sentence_train_models(group):
    """Render one or more train_model entries that share a training run as
    a single cohesive passage: the first sentence spells out the full
    protocol ('An X model was trained to predict Y, with hyperparameters
    tuned by...'); subsequent models are named together as trained under
    that same protocol, rather than repeating the whole clause per model,
    each still carrying its own validation metrics."""
    first, *rest = group
    sentences = [_sentence_train_model(first)]
    if rest:
        names = []
        for idx, e in enumerate(rest):
            model = e["inputs"].get("model", "model")
            article = _article(model) if idx == 0 else _article(model).lower()
            names.append(f"{article} {model}{_format_metrics(e['outputs'].get('metrics'))}")
        if len(names) == 1:
            joined = names[0]
        elif len(names) == 2:
            joined = f"{names[0]} and {names[1]}"
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        was_were = "was" if len(names) == 1 else "were"
        sentences.append(f"{joined} {was_were} trained under the same protocol.")
    return " ".join(sentences)


def _sentence_fit_spline(e):
    i, o = e["inputs"], e["outputs"]
    return (f"A restricted cubic spline with {i.get('n_knots', '?')} knots was fit to "
            f"model the relationship between {i.get('predictor', 'the predictor')} and "
            f"{i.get('outcome', 'the outcome')} (AIC={o.get('aic', '?')}, "
            f"n={o.get('n', '?')}).")


def _sentence_epi_km(e):
    i, o = e["inputs"], e["outputs"]
    sentence = (f"Kaplan–Meier survival analysis was performed using "
                f"{i.get('time', 'time')} as the time variable and "
                f"{i.get('event', 'event')} as the event indicator")
    if i.get("group"):
        sentence += f", stratified by {i['group']}"
    sentence += "."
    logrank = o.get("logrank")
    if logrank:
        sentence += f" A log-rank test was performed (p={logrank.get('p_value', '?')})."
    return sentence


def _sentence_epi_cox(e):
    i, o = e["inputs"], e["outputs"]
    covariates = ", ".join(i.get("covariates") or []) or "the specified covariates"
    return (f"A Cox proportional-hazards regression model was fit using "
            f"{i.get('time', 'time')} as the time variable and "
            f"{i.get('event', 'event')} as the event indicator, adjusting for "
            f"{covariates} (concordance={o.get('concordance', '?')}, "
            f"n={o.get('n', '?')}).")


def _sentence_epi_hl(e):
    i, o = e["inputs"], e["outputs"]
    return (f"Model calibration was assessed using the Hosmer–Lemeshow "
            f"goodness-of-fit test on {i.get('model', 'the model')} "
            f"(statistic={o.get('hl_statistic', '?')}, p={o.get('p_value', '?')}).")


_BUILDERS = {
    "upload": _sentence_upload,
    "confirm": _sentence_confirm,
    "fit_spline": _sentence_fit_spline,
    "epi_km": _sentence_epi_km,
    "epi_cox": _sentence_epi_cox,
    "epi_hl": _sentence_epi_hl,
}

# What makes two log entries "the same analysis step" for report purposes,
# keyed by action. Re-running a step (e.g. re-fitting a spline with a tweaked
# knot count) should replace its earlier entry in the report rather than
# appending a second sentence for it.
_DEDUPE_KEY = {
    "upload": lambda i: (),
    "confirm": lambda i: (),
    "train_model": lambda i: (i.get("model"), i.get("outcome")),
    "fit_spline": lambda i: (i.get("predictor"), i.get("outcome")),
    "epi_km": lambda i: (i.get("time"), i.get("event"), i.get("group")),
    "epi_cox": lambda i: (i.get("time"), i.get("event")),
    "epi_hl": lambda i: (i.get("model"),),
}


def _dedupe_latest(entries):
    """Keep only the most recent entry for each (action, dedupe key),
    ordered by when that most recent occurrence happened."""
    latest = {}
    for entry in entries:
        action = entry["action"]
        key_fn = _DEDUPE_KEY.get(action)
        dedupe_key = (action, key_fn(entry["inputs"]) if key_fn else id(entry))
        latest.pop(dedupe_key, None)
        latest[dedupe_key] = entry
    return list(latest.values())


def generate_methods_text(session_id):
    """Build a deterministic methods paragraph from a session's analysis
    log. Returns '' if the session has no logged actions. Log actions with
    no matching sentence template (e.g. 'test_model', which belongs in
    results rather than methods) are silently skipped. If the same step was
    run more than once (e.g. a spline refit), only its most recent run is
    included.
    """
    entries = _dedupe_latest(analysis_log.read_log(session_id))
    train_groups = {}
    for entry in entries:
        if entry["action"] == "train_model":
            train_groups.setdefault(_train_group_key(entry), []).append(entry)

    sentences = []
    emitted_train_keys = set()
    for entry in entries:
        if entry["action"] == "train_model":
            key = _train_group_key(entry)
            if key in emitted_train_keys:
                continue
            emitted_train_keys.add(key)
            sentences.append(_sentence_train_models(train_groups[key]))
            continue
        builder = _BUILDERS.get(entry["action"])
        if builder:
            sentences.append(builder(entry))
    return " ".join(sentences)
