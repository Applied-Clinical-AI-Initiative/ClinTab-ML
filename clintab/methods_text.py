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
    return (f"A dataset of {o.get('n_rows', '?')} rows and {o.get('n_cols', '?')} "
            f"columns ({filename}) was used for analysis.")


def _sentence_confirm(e):
    i, o = e["inputs"], e["outputs"]
    method_txt = "a stratified" if i.get("method") == "stratified" else "a random"
    sentence = (f"The data were split into training (n={o.get('n_train', '?')}), "
                f"validation (n={o.get('n_val', '?')}), and test (n={o.get('n_test', '?')}) "
                f"sets using {method_txt} split")
    if i.get("stratify_col"):
        sentence += f" stratified on {i['stratify_col']}"
    sentence += "."
    if i.get("smote_pref"):
        sentence += (" SMOTE was applied to the training fold only, to address "
                     "class imbalance.")
    return sentence


def _sentence_train_model(e):
    i, o = e["inputs"], e["outputs"]
    cv_folds = i.get("cv_folds")
    tuning = (f"{cv_folds}-fold cross-validation on the training set" if cv_folds
              else "a held-out validation set")
    smote_txt = ", with SMOTE applied to the training fold," if i.get("smote") else ""
    return (f"A {i.get('model', 'model')} was trained to predict "
            f"{i.get('outcome', 'the outcome')}{smote_txt}, with hyperparameters "
            f"tuned by {i.get('scoring', 'the default metric')} scoring using "
            f"{tuning}{_format_metrics(o.get('metrics'))}.")


def _sentence_fit_spline(e):
    i, o = e["inputs"], e["outputs"]
    return (f"A restricted cubic spline with {i.get('n_knots', '?')} knots was fit "
            f"for {i.get('predictor', 'the predictor')} against "
            f"{i.get('outcome', 'the outcome')} (AIC={o.get('aic', '?')}, "
            f"n={o.get('n', '?')}).")


def _sentence_epi_km(e):
    i, o = e["inputs"], e["outputs"]
    sentence = (f"Kaplan-Meier survival analysis was performed on "
                f"{i.get('time', 'time')}/{i.get('event', 'event')}")
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
    return (f"A Cox proportional hazards model was fit for "
            f"{i.get('time', 'time')}/{i.get('event', 'event')}, adjusting for "
            f"{covariates} (concordance={o.get('concordance', '?')}, "
            f"n={o.get('n', '?')}).")


def _sentence_epi_hl(e):
    i, o = e["inputs"], e["outputs"]
    return (f"Model calibration was assessed with the Hosmer-Lemeshow test on "
            f"{i.get('model', 'the model')} (statistic={o.get('hl_statistic', '?')}, "
            f"p={o.get('p_value', '?')}).")


_BUILDERS = {
    "upload": _sentence_upload,
    "confirm": _sentence_confirm,
    "train_model": _sentence_train_model,
    "fit_spline": _sentence_fit_spline,
    "epi_km": _sentence_epi_km,
    "epi_cox": _sentence_epi_cox,
    "epi_hl": _sentence_epi_hl,
}


def generate_methods_text(session_id):
    """Build a deterministic methods paragraph from a session's analysis
    log. Returns '' if the session has no logged actions. Log actions with
    no matching sentence template (e.g. 'test_model', which belongs in
    results rather than methods) are silently skipped.
    """
    entries = analysis_log.read_log(session_id)
    sentences = []
    for entry in entries:
        builder = _BUILDERS.get(entry["action"])
        if builder:
            sentences.append(builder(entry))
    return " ".join(sentences)
