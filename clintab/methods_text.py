"""methods_text.py
Deterministic methods-section text, assembled from a session's analysis log
(clintab/analysis_log.py). Every sentence comes from a fixed template filled
in with structured log fields, there is no generative/LLM call anywhere in
here, so the same log always produces the exact same text, and the wording
is defensible to paste into a manuscript. Pure Python, no Flask.

Sentences are grouped by topic (data, cohort, sample size, missing data,
split, models, evaluation, other analyses, software) rather than strictly in
log order. Teaching-only actions (leakage_demo) never appear: they are not
part of the study.
"""
import math
import re

from clintab import analysis_log

_IGNORED = {"leakage_demo"}

_METRIC_ORDER = ["AUROC", "AUPRC", "Accuracy", "BalancedAccuracy", "MacroF1",
                 "R2", "RMSE", "MAE"]

# internal scoring keys (clintab/ml.py, the scoring dropdown in static/app.js) rendered as
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

# model type keys from clintab/ml.py. models are named by type and outcome
# (logged with each entry), never by saved file id; these keys also let an
# older log's file id be split back into type and outcome.
_MODEL_KEYS = {
    "LogisticRegression", "RidgeLogistic", "LassoLogistic", "ElasticNetLogistic",
    "KNN", "DecisionTree", "RandomForest", "GradientBoosting", "SVM", "NaiveBayes",
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForestRegressor",
}

# test metrics in reporting order, with the phrase that introduces each
_TEST_METRICS = [
    ("AUROC", "an AUROC of"),
    ("AUPRC", "an AUPRC of"),
    ("Brier", "a Brier score of"),
    ("CalibrationSlope", "a calibration slope of"),
    ("CalibrationInTheLarge", "a calibration-in-the-large of"),
    ("OE_ratio", "an observed-to-expected ratio of"),
    ("ICI", "an integrated calibration index of"),
    ("R2", "an R-squared of"),
    ("RMSE", "a root mean squared error of"),
    ("MAE", "a mean absolute error of"),
    ("Accuracy", "an accuracy of"),
    ("BalancedAccuracy", "a balanced accuracy of"),
    ("MacroF1", "a macro-averaged F1 score of"),
]

_DECISION_ORDER = ["impute", "impute_indicator", "include", "zero", "remove"]
_DECISION_TEXT = {
    "impute": "imputed with the median (continuous) or most frequent value (categorical)",
    "impute_indicator": ("imputed the same way with an added binary missing-value "
                         "indicator"),
    "include": ("left missing and imputed inside each model pipeline, fitted on the "
                "training data only"),
    "zero": "filled with zero",
    "remove": "removed",
}
# impute_indicator when no column was plainly imputed ("the same way" would
# have nothing to refer to)
_INDICATOR_ALONE = ("imputed with the median (continuous) or most frequent value "
                    "(categorical), with an added binary missing-value indicator")

# threshold methods logged by /api/train and /api/test, as a manuscript
# would describe them. anything else (older free-text logs) is quoted as is.
_THRESHOLD_TEXT = {
    "youden": "chosen on the validation set by maximising the Youden index",
    "sensitivity": ("chosen on the validation set as the highest threshold with a "
                    "sensitivity of at least {target}"),
    "specificity": ("chosen on the validation set as the lowest threshold with a "
                    "specificity of at least {target}"),
    "prevalence": "set to the outcome prevalence in the validation set",
    "fixed": "set in advance",
    "user": "set by the analyst at testing",
}

_INTERNAL_SOURCES = {"internal_test", "test"}
_TS_SUFFIX = re.compile(r"_\d{8}-\d{6}$")
_NAME_CAP = 8           # list column names only up to this many


# ----------------------------------------------------------------------------
# small formatting helpers
# ----------------------------------------------------------------------------
def _article(word):
    """'A' or 'An', by the leading sound of word (e.g. 'An ElasticNet').
    All-caps acronyms are read letter by letter, so 'An SVM' but 'A KNN'."""
    if not word:
        return "A"
    vowel_sounds = "AEIOUFHLMNRSX" if word.isupper() and len(word) > 1 else "AEIOU"
    return "An" if word[0].upper() in vowel_sounds else "A"


def _scoring_label(key):
    return _SCORING_LABELS.get(key, key or "the default metric")


def _display(model):
    return model or "model"


def _fmt(v):
    """numbers as logged: 20.0 -> '20', 0.6801 -> '0.6801'."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:g}" if abs(v) >= 1e-4 else str(v)
    return str(v)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _names(cols):
    cols = list(cols)
    return f" ({', '.join(map(str, cols))})" if 0 < len(cols) <= _NAME_CAP else ""


def _pct(x):
    return f"{round(float(x) * 100, 1):g}%"


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def _clean(text):
    """user-typed text with long dashes swapped for plain punctuation."""
    text = re.sub(r"\s*\u2014\s*|\s+\u2013\s+", ", ", str(text))
    return text.replace("\u2013", "-").strip()


def _format_metrics(metrics):
    if not metrics:
        return ""
    picked = [(k, metrics[k]) for k in _METRIC_ORDER if k in metrics]
    if not picked:
        return ""
    bits = ", ".join(f"{k}={v}" for k, v in picked)
    return f" (validation {bits})"


# ----------------------------------------------------------------------------
# model naming: by type and outcome, never by saved file id
# ----------------------------------------------------------------------------
def _model_lookup(entries):
    """saved file id -> (model type, outcome), from train_model entries."""
    out = {}
    for e in entries:
        if e["action"] == "train_model" and e["outputs"].get("saved_as"):
            out[e["outputs"]["saved_as"]] = (e["inputs"].get("model"), e["inputs"].get("outcome"))
    return out


def _model_label(inputs, lookup):
    """'the LogisticRegression model for mortality_30d'."""
    name = inputs.get("model") or ""
    mtype, outcome = inputs.get("model_type"), inputs.get("outcome")
    if name in lookup:
        mtype = mtype or lookup[name][0]
        outcome = outcome or lookup[name][1]
    if not mtype:
        if name.endswith("_uploaded"):
            return f"the uploaded model {name[:-len('_uploaded')]}"
        stem = _TS_SUFFIX.sub("", name)
        head, _, rest = stem.partition("_")
        if head in _MODEL_KEYS:
            mtype, outcome = head, outcome or rest or None
        else:
            return f"the model {stem or 'supplied'}"
    label = f"the {_display(mtype)} model"
    return f"{label} for {outcome}" if outcome else label


# ----------------------------------------------------------------------------
# data source, cohort, sample size
# ----------------------------------------------------------------------------
def _sentence_upload(e):
    i, o = e["inputs"], e["outputs"]
    filename = i.get("filename", "the uploaded file")
    sha = i.get("sha256")
    tail = f"; SHA-256 prefix {sha[:12]}" if sha else ""
    return (f"Data were taken from {filename} ({o.get('n_rows', '?')} rows and "
            f"{o.get('n_cols', '?')} columns{tail}).")


def _threshold_text(method, target=None):
    """how a classification threshold was chosen, or None when unknown."""
    if not method:
        return None
    text = _THRESHOLD_TEXT.get(str(method).lower())
    if text is None:
        return f"chosen by {method}"
    if "{target}" in text:
        if not _finite(target):
            return text.split(" as the ")[0] + f" to meet a {str(method).lower()} target"
        text = text.format(target=_fmt(target))
    return text


def _sentence_cohort(e):
    flow = e["outputs"].get("flow") or []
    if not flow:
        return ""
    steps = []
    for s in flow:
        label = s.get("label") or f"step {s.get('step', len(steps) + 1)}"
        steps.append(f"{label} ({_fmt(s.get('n_removed', '?'))} excluded, "
                     f"{_fmt(s.get('n_after', '?'))} remaining)")
    start = flow[0].get("n_before")
    lead = f"Starting from {_fmt(start)} records, the" if start is not None else "The"
    n = len(flow)
    return (f"{lead} cohort was defined by {n} inclusion step{'s' if n != 1 else ''}: "
            f"{'; '.join(steps)}.")


def _sentence_sample_size(e):
    i, o = e["inputs"], e["outputs"]
    assumptions = []
    if i.get("prevalence") is not None:
        assumptions.append(f"an outcome prevalence of {_fmt(i['prevalence'])}")
    if i.get("n_parameters") is not None:
        assumptions.append(f"{_fmt(i['n_parameters'])} candidate predictor parameters")
    if i.get("cstatistic") is not None:
        assumptions.append(f"an anticipated C-statistic of {_fmt(i['cstatistic'])}")
    elif i.get("csrsquared") is not None:
        assumptions.append(f"an anticipated Cox-Snell R-squared of {_fmt(i['csrsquared'])}")
    elif i.get("nagrsquared") is not None:
        assumptions.append(f"an anticipated Nagelkerke R-squared of {_fmt(i['nagrsquared'])}")
    elif i.get("rsquared") is not None:
        assumptions.append(f"an anticipated R-squared of {_fmt(i['rsquared'])}")
    if i.get("shrinkage") is not None:
        assumptions.append(f"a target shrinkage factor of {_fmt(i['shrinkage'])}")
    sentence = ("The minimum sample size for model development was calculated following "
                "the criteria of Riley et al. (BMJ 2020)")
    sentence += f", assuming {_join(assumptions)}." if assumptions else "."
    needs = []
    if o.get("min_n") is not None:
        needs.append(f"{_fmt(o['min_n'])} participants")
    if o.get("min_events") is not None:
        needs.append(f"{_fmt(o['min_events'])} events")
    if needs:
        sentence += f" At least {_join(needs)} were required"
        sentence += (f" ({_fmt(o['epp'])} events per candidate predictor parameter)."
                     if o.get("epp") is not None else ".")
    return sentence


# ----------------------------------------------------------------------------
# confirm: missing data, exclusions, split
# ----------------------------------------------------------------------------
def _sentence_missing(e):
    i, o = e["inputs"], e["outputs"]
    decisions = i.get("missing") or {}
    excluded = set(i.get("excluded") or [])
    pct = i.get("missing_pct", o.get("missing_pct"))
    fitted_on_train = "fill_values" in o

    if pct is not None:
        if not pct:
            return "There were no missing values in the dataset."
        cols = [c for c in pct if c not in excluded]
        if not cols:
            return ""
        by_type = {d: [c for c in cols if decisions.get(c) == d] for d in _DECISION_ORDER}
        undecided = [c for c in cols if decisions.get(c) not in _DECISION_TEXT]
    else:
        # older logs: decisions only, no per-column missingness
        cols = [c for c, x in decisions.items() if x in _DECISION_TEXT and c not in excluded]
        if not cols:
            return ""
        by_type = {d: [c for c in cols if decisions[c] == d] for d in _DECISION_ORDER}
        undecided = []

    if len(cols) == 1:
        c = cols[0]
        d = decisions.get(c)
        what = (_INDICATOR_ALONE if d == "impute_indicator" else _DECISION_TEXT[d]
                if d in _DECISION_TEXT else "left as recorded with no logged decision")
        span = f" ({_fmt(pct[c])}% missing)" if pct is not None else ""
        sentence = f"One column with missing values, {c}{span}, was {what}."
    else:
        clauses = []
        for d in _DECISION_ORDER:
            group = by_type[d]
            if group:
                k = len(group)
                what = _DECISION_TEXT[d]
                if d == "impute_indicator" and not by_type["impute"]:
                    what = _INDICATOR_ALONE
                clauses.append(f"{k}{_names(group)} {'was' if k == 1 else 'were'} {what}")
        if undecided:
            k = len(undecided)
            clauses.append(f"{k}{_names(undecided)} {'was' if k == 1 else 'were'} left as "
                           f"recorded with no logged decision")
        if pct is not None:
            lo, hi = min(pct[c] for c in cols), max(pct[c] for c in cols)
            span = f"{_fmt(lo)}% missing" if lo == hi else f"{_fmt(lo)}% to {_fmt(hi)}% missing"
            lead = f"Of the {len(cols)} columns with missing values ({span})"
        else:
            lead = f"Of the {len(cols)} columns with a missing-data decision"
        if len(clauses) > 1 and any("," in c for c in clauses):
            joined = "; ".join(clauses[:-1]) + f"; and {clauses[-1]}"   # clauses hold commas
        else:
            joined = _join(clauses)
        sentence = f"{lead}, {joined}."
    if by_type["impute"] or by_type["impute_indicator"]:
        if fitted_on_train:
            sentence += (" Fill values were estimated on the training partition only and "
                         "then applied to the validation and test partitions.")
        else:
            sentence += " Fill values were computed on the full dataset before it was split."
    return sentence


def _sentence_exclusions(e):
    excluded = e["inputs"].get("excluded") or []
    if not excluded:
        return ""
    k = len(excluded)
    if k == 1:
        return f"One column ({excluded[0]}) was excluded from the analysis."
    return f"A total of {k} columns{_names(excluded)} were excluded from the analysis."


def _sentence_split(e):
    i, o = e["inputs"], e["outputs"]
    method = i.get("method") or "random"
    col = i.get("stratify_col")
    if method == "stratified" and col:
        split_txt = f"a split stratified on {col}"
    elif method == "stratified":
        split_txt = "a stratified split"
    elif method.startswith("random (") and col:
        split_txt = (f"a random split (stratification on {col} was requested but was not "
                     f"possible)")
    else:
        split_txt = "a random split"
    ratios = i.get("ratios")
    share = (f" ({'/'.join(_pct(r) for r in ratios)})"
             if isinstance(ratios, list) and len(ratios) == 3 else "")
    seed = f", with random seed {i['seed']}" if i.get("seed") is not None else ""
    return (f"The data were partitioned into training (n={o.get('n_train', '?')}), "
            f"validation (n={o.get('n_val', '?')}), and test (n={o.get('n_test', '?')}) "
            f"sets{share} using {split_txt}{seed}.")


def _sentence_confirm(e):
    return " ".join(s for s in (_sentence_missing(e), _sentence_exclusions(e),
                                _sentence_split(e)) if s)


def _sentence_custom_split(e):
    o = e["outputs"]
    return (f"The training (n={o.get('n_train', '?')}) and validation "
            f"(n={o.get('n_val', '?')}) sets were then replaced by user-supplied "
            f"files; the test set was left unchanged.")


# ----------------------------------------------------------------------------
# model training
# ----------------------------------------------------------------------------
def _sentence_train_model(e):
    i, o = e["inputs"], e["outputs"]
    model = _display(i.get("model"))
    cv_folds = i.get("cv_folds")
    tuning = (f"{cv_folds}-fold cross-validation on the training set" if cv_folds
              else "a held-out validation set")
    smote_clause = ", with SMOTE applied to the training data only," if i.get("smote") else ""
    # logs from before "tuned" was recorded always came from a tuned run
    if i.get("tuned", True):
        fit_clause = (f"hyperparameters tuned by {_scoring_label(i.get('scoring'))} "
                      f"scoring against {tuning}")
    else:
        fit_clause = "default hyperparameters and no tuning"
    return (f"{_article(model)} {model} model was trained to predict "
            f"{i.get('outcome', 'the outcome')}{smote_clause} with {fit_clause}"
            f"{_format_metrics(o.get('metrics'))}.")


def _train_group_key(e):
    """Entries that share this key were trained under the same protocol
    (same outcome/scoring/tuning/SMOTE). The first entry in a group states
    the full protocol; later ones in the group refer back to it instead of
    repeating the same sentence template verbatim."""
    i = e["inputs"]
    return (i.get("outcome"), i.get("scoring"), i.get("cv_folds"), i.get("smote"),
            i.get("tuned", True))


def _sentence_training_data(group):
    """records, events and model input parameters, from the first entry in
    the group that logged them."""
    for e in group:
        o = e["outputs"]
        n = o.get("n_train")
        if not n:
            continue
        sentence = f"Training used {n} records"
        events = o.get("events")
        if events is not None:
            outcome = e["inputs"].get("outcome", "the outcome")
            pos = o.get("positive_class")
            who = f"{outcome} = {pos}" if pos is not None else "the outcome"
            sentence += f", of which {events} ({100 * events / n:.1f}%) had {who}"
        p = o.get("n_parameters")
        if p:
            sentence += (f", with {p} model input parameters after encoding of "
                         f"categorical predictors")
            if events is not None:
                sentence += f" ({events / p:.1f} events per parameter)"
        return sentence + "."
    return ""


def _sentences_threshold_recalibration(group):
    """recalibration and threshold choice for binary models, from the
    train_model outputs (logs from before these were recorded give none)."""
    out = []
    recal = [_display(e["inputs"].get("model")) for e in group if e["outputs"].get("recalibrated")]
    if recal:
        these = f"the {_join(recal)} model" + ("s" if len(recal) > 1 else "")
        out.append(f"Predicted risks from {these} were recalibrated with logistic "
                   f"recalibration (a new intercept and slope on the logit scale) fitted "
                   f"on the validation set, so validation calibration was optimistic and "
                   f"calibration was judged on the test data.")
    picks = []
    for e in group:
        o = e["outputs"]
        text = _threshold_text(o.get("threshold_method"), o.get("threshold_target"))
        if text is None:
            continue
        if str(o.get("threshold_method")).lower() == "fixed" and _finite(o.get("threshold")):
            text += f" at {_fmt(o['threshold'])}"
        picks.append((_display(e["inputs"].get("model")), text))
    if picks:
        texts = list(dict.fromkeys(t for _, t in picks))
        if len(texts) == 1:
            out.append(f"The classification threshold was {texts[0]}.")
        else:
            out.append("The classification threshold was "
                       + _join([f"{t} for the {m} model" for m, t in picks]) + ".")
    return out


def _sentence_train_models(group):
    """Render one or more train_model entries that share a training run as
    a single cohesive passage: the first sentence spells out the full
    protocol ('A LogisticRegression model was trained to predict Y, with
    hyperparameters tuned by...'); subsequent models are named together as
    trained under that same protocol, rather than repeating the whole clause
    per model, each still carrying its own validation metrics."""
    first, *rest = group
    sentences = [_sentence_train_model(first)]
    if rest:
        names = []
        for idx, e in enumerate(rest):
            model = _display(e["inputs"].get("model"))
            article = _article(model) if idx == 0 else _article(model).lower()
            names.append(f"{article} {model}{_format_metrics(e['outputs'].get('metrics'))}")
        was_were = "was" if len(names) == 1 else "were"
        sentences.append(f"{_join(names)} {was_were} trained under the same protocol.")
    data = _sentence_training_data(group)
    if data:
        sentences.append(data)
    sentences.extend(_sentences_threshold_recalibration(group))
    if first["inputs"].get("smote"):
        these = "this model" if len(group) == 1 else "these models"
        sentences.append(f"Because SMOTE changed the outcome prevalence seen during "
                         f"training, the predicted probabilities of {these} were at risk "
                         f"of miscalibration.")
    return " ".join(sentences)


# ----------------------------------------------------------------------------
# evaluation: internal test, test reuse, external, decision curves
# ----------------------------------------------------------------------------
def _ci(metrics, outputs, key):
    ci = metrics.get("CI") if isinstance(metrics.get("CI"), dict) else outputs.get("CI")
    if isinstance(ci, dict):
        pair = ci.get(key)
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(map(_finite, pair)):
            return f" (95% CI {_fmt(pair[0])} to {_fmt(pair[1])})"
    return ""


def _performance(e):
    """'had an AUROC of 0.78 (95% CI ...), and a calibration slope of 0.9'
    plus an optional threshold sentence, or ('', '') if nothing to report."""
    o = e["outputs"]
    m = o.get("metrics") or {}
    if not isinstance(m, dict):
        return "", ""
    bits = [f"{phrase} {_fmt(m[k])}{_ci(m, o, k)}"
            for k, phrase in _TEST_METRICS if _finite(m.get(k))]
    thr = ""
    if _finite(m.get("Sensitivity")) and _finite(m.get("Specificity")):
        t = m.get("Threshold", e["inputs"].get("threshold"))
        text = _threshold_text(o.get("threshold_method"), o.get("threshold_target"))
        how = f" ({text})" if text else ""
        at = f"At a classification threshold of {_fmt(t)}{how}, " if _finite(t) else "At the chosen threshold, "
        thr = (f"{at}sensitivity was {_fmt(m['Sensitivity'])}{_ci(m, o, 'Sensitivity')} and "
               f"specificity was {_fmt(m['Specificity'])}{_ci(m, o, 'Specificity')}.")
    return (f"had {_join(bits)}" if bits else ""), thr


def _subject(e, lookup):
    label = _model_label(e["inputs"], lookup)
    if e["outputs"].get("recalibrated"):
        label += ", after recalibration,"
    return label


def _sentences_internal_tests(tests, lookup):
    rows = []
    for e in tests:
        perf, thr = _performance(e)
        if not perf:
            continue
        rows.append(f"{_cap(_subject(e, lookup))} {perf}." + (f" {thr}" if thr else ""))
    if not rows:
        return ""
    ns = [(t["outputs"].get("metrics") or {}).get("N") or t["inputs"].get("n") for t in tests]
    ns = [n for n in ns if n]
    size = f" (n={ns[-1]})" if ns and len(set(ns)) == 1 else ""
    return f"Performance was evaluated on the internal test partition{size}. " + " ".join(rows)


def _sentence_unlock(e, lookup):
    models = e["inputs"].get("models") or []
    labels = [_model_label({"model": m}, lookup) for m in models] or ["further models"]
    reason = _clean(e["inputs"].get("reason") or "")
    why = f" The reason given was: \"{reason}\"." if reason else " No reason was recorded."
    these = "this model" if len(labels) == 1 else "these models"
    return (f"The internal test partition was reused after test results had been seen, "
            f"to evaluate {_join(labels)}.{why} Test results for {these} were therefore "
            f"likely to be optimistic.")


def _sentence_external(e, lookup):
    i = e["inputs"]
    perf, thr = _performance(e)
    where = i.get("filename") or "an external dataset"
    extra = []
    if i.get("n") is not None:
        extra.append(f"n={i['n']}")
    if i.get("sha256"):
        extra.append(f"SHA-256 prefix {i['sha256'][:12]}")
    detail = f" ({'; '.join(extra)})" if extra else ""
    sentence = f"{_cap(_subject(e, lookup))} was evaluated on external data from {where}{detail}"
    sentence += f" and {perf}." if perf else "."
    return sentence + (f" {thr}" if thr else "")


def _sentence_dca(tests, lookup):
    picked = [e for e in tests if e["outputs"].get("dca")]
    if not picked:
        return ""
    parts = []
    for e in picked:
        where = ("external data" if e["inputs"].get("source") == "external"
                 else "the internal test partition")
        parts.append(f"{_model_label(e['inputs'], lookup)} on {where}")
    return (f"Decision curve analysis was performed for {_join(parts)} to estimate net "
            f"benefit across a range of threshold probabilities.")


def _sentence_epi_hl(e, lookup):
    o = e["outputs"]
    return (f"Model calibration was assessed using the Hosmer-Lemeshow "
            f"goodness-of-fit test on {_model_label(e['inputs'], lookup)} "
            f"(statistic={o.get('hl_statistic', '?')}, p={o.get('p_value', '?')}).")


# ----------------------------------------------------------------------------
# other analyses
# ----------------------------------------------------------------------------
def _sentence_adjusted_or(e):
    i, o = e["inputs"], e["outputs"]
    feats = i.get("features") or []
    with_txt = f" including {_join(map(str, feats))}" if feats else ""
    stats = []
    if o.get("n") is not None:
        stats.append(f"n={o['n']}")
    if o.get("events") is not None:
        stats.append(f"{o['events']} events")
    if o.get("n_parameters") is not None:
        stats.append(f"{o['n_parameters']} parameters")
    detail = f" ({', '.join(stats)})" if stats else ""
    return (f"A multivariable logistic regression model for {i.get('outcome', 'the outcome')}"
            f"{with_txt} was fitted to estimate adjusted odds ratios{detail}.")


def _sentence_fit_spline(e):
    i, o = e["inputs"], e["outputs"]
    return (f"A restricted cubic spline with {i.get('n_knots', '?')} knots was fit to "
            f"model the relationship between {i.get('predictor', 'the predictor')} and "
            f"{i.get('outcome', 'the outcome')} (AIC={o.get('aic', '?')}, "
            f"n={o.get('n', '?')}).")


def _sentence_epi_km(e):
    i, o = e["inputs"], e["outputs"]
    sentence = (f"Kaplan-Meier survival analysis was performed using "
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
    sentence = (f"A Cox proportional-hazards regression model was fit using "
                f"{i.get('time', 'time')} as the time variable and "
                f"{i.get('event', 'event')} as the event indicator, adjusting for "
                f"{covariates} (concordance={o.get('concordance', '?')}, "
                f"n={o.get('n', '?')}).")
    ph = o.get("ph_test") or {}
    glob = ph.get("global") or {}
    if _finite(glob.get("p_value")):
        sentence += (f" The proportional hazards assumption was checked with a test based "
                     f"on scaled Schoenfeld residuals (global p={_fmt(glob['p_value'])}).")
    return sentence


def _sentence_epi_cif(e):
    i, o = e["inputs"], e["outputs"]
    what = i.get("event", "the event")
    if i.get("event_of_interest") is not None:
        what = f"{what} = {_fmt(i['event_of_interest'])}"
    sentence = (f"The cumulative incidence of {what} was estimated using "
                f"{i.get('time', 'time')} as the time variable, with other event types "
                f"treated as competing risks")
    if i.get("group"):
        sentence += f", stratified by {i['group']}"
    if o.get("n") is not None:
        sentence += f" (n={o['n']})"
    return sentence + "."


_NNT_WORDS = {"NNT": "number needed to treat", "NNH": "number needed to harm"}


def _sentences_two_by_two(entries):
    """one sentence per distinct 2x2 table, merging the OR and NNT calls."""
    tables = {}
    for e in entries:
        i = e["inputs"]
        key = tuple(_fmt(i.get(k)) for k in ("a", "b", "c", "d"))
        tables.setdefault(key, {})[e["action"]] = e["outputs"]
    out = []
    for (a, b, c, d), res in tables.items():
        parts = []
        r = res.get("epi_or")
        if r:
            if r.get("odds_ratio") is not None:
                ci = r.get("or_ci") or []
                parts.append(f"the odds ratio was {_fmt(r['odds_ratio'])}"
                             + (f" (95% CI {_fmt(ci[0])} to {_fmt(ci[1])})" if len(ci) == 2 else ""))
            if r.get("risk_ratio") is not None:
                ci = r.get("rr_ci") or []
                parts.append(f"the risk ratio was {_fmt(r['risk_ratio'])}"
                             + (f" (95% CI {_fmt(ci[0])} to {_fmt(ci[1])})" if len(ci) == 2 else ""))
        n = res.get("epi_nnt")
        if n:
            if n.get("value") is None:
                parts.append("there was no difference in risk")
            else:
                word = _NNT_WORDS.get(n.get("measure"), "number needed to treat or harm")
                parts.append(f"the {word} was {_fmt(n['value'])}")
        if not parts:
            continue
        sentence = f"From a 2x2 table (a={a}, b={b}, c={c}, d={d}), {_join(parts)}."
        if r and r.get("corrected"):
            sentence += (" A Haldane-Anscombe correction (0.5 added to every cell) was "
                         "applied because of a zero cell.")
        out.append(sentence)
    return " ".join(out)


def _sentence_software(entries):
    env = None
    for e in entries:
        if isinstance(e["outputs"].get("environment"), dict):
            env = e["outputs"]["environment"]
    if not env:
        return ""
    pkgs = [f"{k} {env[k]}" for k in ("scikit-learn", "imbalanced-learn", "statsmodels",
                                      "lifelines", "numpy", "pandas", "scipy") if env.get(k)]
    sentence = f"Analyses were performed with ClinTAB-ML {env.get('clintab', '?')}"
    if env.get("python"):
        sentence += f" in Python {env['python']}"
    if pkgs:
        sentence += f" ({', '.join(pkgs)})"
    if env.get("platform"):
        sentence += f" on {env['platform']}"
    return sentence + "."


# ----------------------------------------------------------------------------
# dedupe + assembly
# ----------------------------------------------------------------------------
def _source(i):
    s = i.get("source", "test")
    return "internal_test" if s in _INTERNAL_SOURCES else s


# what makes two log entries "the same analysis step" for report purposes,
# keyed by action. re-running a step (e.g. re-fitting a spline with a tweaked
# knot count) should replace its earlier entry in the report rather than
# appending a second sentence for it. actions without a key (test_unlock) are
# all kept.
_DEDUPE_KEY = {
    "upload": lambda i: (),
    "confirm": lambda i: (),
    "upload_custom_split": lambda i: (),
    "cohort_filter": lambda i: (),
    "sample_size": lambda i: (),
    "train_model": lambda i: (i.get("model"), i.get("outcome")),
    "test_model": lambda i: (i.get("model"), _source(i), i.get("sha256")),
    "fit_spline": lambda i: (i.get("predictor"), i.get("outcome")),
    "epi_km": lambda i: (i.get("time"), i.get("event"), i.get("group")),
    "epi_cox": lambda i: (i.get("time"), i.get("event")),
    "epi_cif": lambda i: (i.get("time"), i.get("event"), i.get("event_of_interest"),
                          i.get("group")),
    "epi_hl": lambda i: (i.get("model"),),
    "epi_or": lambda i: tuple(str(i.get(k)) for k in "abcd"),
    "epi_nnt": lambda i: tuple(str(i.get(k)) for k in "abcd"),
    "adjusted_or": lambda i: (i.get("outcome"), tuple(i.get("features") or [])),
}


def _dedupe_latest(entries):
    """Keep only the most recent entry for each (action, dedupe key),
    ordered by when that most recent occurrence happened."""
    latest = {}
    for n, entry in enumerate(entries):
        action = entry["action"]
        key_fn = _DEDUPE_KEY.get(action)
        dedupe_key = (action, key_fn(entry["inputs"]) if key_fn else n)
        latest.pop(dedupe_key, None)
        latest[dedupe_key] = entry
    return list(latest.values())


def _of(entries, *actions):
    return [e for e in entries if e["action"] in actions]


def _confirmed_without_cohort(entries):
    """True when the latest confirm logged that no cohort filters were used,
    so an earlier cohort_filter entry no longer describes the data."""
    confirms = _of(entries, "confirm")
    return bool(confirms) and confirms[-1]["inputs"].get("cohort_filters") == 0


def generate_methods_text(session_id):
    """Build a deterministic methods paragraph from a session's analysis
    log. Returns '' if the session has no logged actions. Actions with no
    template, and teaching demos, are skipped. If the same step was run more
    than once (e.g. a spline refit), only its most recent run is included.
    """
    raw = [e for e in analysis_log.read_log(session_id) if e.get("action") not in _IGNORED]
    lookup = _model_lookup(raw)
    entries = _dedupe_latest(raw)
    # a custom split only stands untill the dataset is confirmed (re-split) again
    last_confirm = max((n for n, e in enumerate(entries) if e["action"] == "confirm"),
                       default=-1)
    entries = [e for n, e in enumerate(entries)
               if not (e["action"] == "upload_custom_split" and n < last_confirm)]
    # a cohort only stands if the latest confirm used one (0 = confirmed without)
    if last_confirm >= 0 and _confirmed_without_cohort(entries):
        entries = [e for e in entries if e["action"] != "cohort_filter"]

    sentences = []
    add = sentences.append
    for e in _of(entries, "upload"):
        add(_sentence_upload(e))
    for e in _of(entries, "cohort_filter"):
        add(_sentence_cohort(e))
    for e in _of(entries, "sample_size"):
        add(_sentence_sample_size(e))
    for e in _of(entries, "confirm"):
        add(_sentence_confirm(e))
    for e in _of(entries, "upload_custom_split"):
        add(_sentence_custom_split(e))

    # models, grouped by shared protocol, in order of first appearance
    groups = {}
    for e in _of(entries, "train_model"):
        groups.setdefault(_train_group_key(e), []).append(e)
    for group in groups.values():
        add(_sentence_train_models(group))

    tests = _of(entries, "test_model")
    internal = [e for e in tests if _source(e["inputs"]) == "internal_test"]
    external = [e for e in tests if _source(e["inputs"]) == "external"]
    add(_sentences_internal_tests(internal, lookup))
    for e in _of(entries, "test_unlock"):
        add(_sentence_unlock(e, lookup))
    for e in external:
        if _performance(e)[0]:
            add(_sentence_external(e, lookup))
    add(_sentence_dca(tests, lookup))
    for e in _of(entries, "epi_hl"):
        add(_sentence_epi_hl(e, lookup))

    for e in _of(entries, "adjusted_or"):
        add(_sentence_adjusted_or(e))
    for e in _of(entries, "fit_spline"):
        add(_sentence_fit_spline(e))
    survival = {"epi_km": _sentence_epi_km, "epi_cox": _sentence_epi_cox,
                "epi_cif": _sentence_epi_cif}
    for e in _of(entries, *survival):
        add(survival[e["action"]](e))
    add(_sentences_two_by_two(_of(entries, "epi_or", "epi_nnt")))

    if any(s for s in sentences):
        add(_sentence_software(raw))
    return " ".join(s for s in sentences if s)
