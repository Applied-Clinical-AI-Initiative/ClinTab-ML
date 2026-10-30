"""tripod_report.py
Maps a session's analysis log onto the reporting domains of TRIPOD+AI
(Collins et al., BMJ 2024), the reporting guideline for studies developing
or validating clinical prediction models, including ones built with machine
learning.

Important: this is a practical subset of TRIPOD+AI's reporting domains, not
a verbatim reproduction of all 27 official checklist items, treat it as a
starting point, not a substitute for the published checklist. And for every
item, "evidence found" only means the underlying analysis step is traceable
in this session's log. It does NOT mean the manuscript actually reports it
correctly, or at all, narrative items (rationale, limitations, clinical
interpretation, and so on) can't be verified from a log no matter what, and
are flagged for manual review rather than guessed at.

Each check returns (status, note). status is one of 'found', 'not_found',
'not_applicable', 'manual_review'; note is a short plain-language reason or
None. Teaching-only actions (leakage_demo) are ignored.

Pure Python, no Flask.
"""
from clintab import analysis_log

DISCLAIMER = (
    "This report checks what this session's analysis log can prove was "
    "actually run. It is not an official TRIPOD+AI compliance check, and "
    "'evidence found' does not mean your manuscript reports an item "
    "correctly (or at all). Items marked 'manual review' need your own "
    "judgement, either because they are narrative content a log can never "
    "verify or because the log only holds partial evidence. Always check the "
    "published TRIPOD+AI checklist (Collins et al., BMJ 2024) directly before "
    "submission."
)

# actions that are teaching demos, not part of any study
_IGNORED = {"leakage_demo"}

# test_model sources; "test" is what logs before this version wrote for the
# internal partition
_INTERNAL_SOURCES = {"internal_test", "test"}


def _has_action(entries, *actions):
    return any(e["action"] in actions for e in entries)


def _of(entries, action):
    return [e for e in entries if e["action"] == action]


def _metrics(e):
    m = e["outputs"].get("metrics")
    return m if isinstance(m, dict) else {}


def _any_metric(entries, actions, keys):
    for e in entries:
        if e["action"] in actions:
            metrics = e["outputs"].get("metrics") or e["outputs"]
            if any(k in metrics for k in keys):
                return True
    return False


def _source(e):
    return e["inputs"].get("source", "test")


def _internal_tests(entries):
    return [e for e in _of(entries, "test_model") if _source(e) in _INTERNAL_SOURCES]


def _external_tests(entries):
    return [e for e in _of(entries, "test_model") if _source(e) == "external"]


def _calibration_measured(entries):
    """slope and calibration-in-the-large both logged on some test result."""
    return any("CalibrationSlope" in _metrics(e) and "CalibrationInTheLarge" in _metrics(e)
               for e in _of(entries, "test_model"))


def _smote_used(entries):
    return any(e["inputs"].get("smote") for e in _of(entries, "train_model"))


def _is_binary(e):
    """train_model / test_model entry for a binary outcome? None if the log
    can't tell (older logs without task and without metrics)."""
    task = e["inputs"].get("task")
    if task:
        return task == "binary"
    if "events" in e["outputs"]:
        return True
    m = _metrics(e)
    if "AUROC" in m:
        return True
    if any(k in m for k in ("R2", "MacroF1", "Accuracy")):
        return False
    return None


def _binary_status(entries):
    """True if any model is binary, False if all are known to be non-binary,
    None if the log can't tell."""
    flags = [_is_binary(e) for e in entries if e["action"] in ("train_model", "test_model")]
    if any(f is True for f in flags):
        return True
    if flags and all(f is False for f in flags):
        return False
    return None


def _prevalence(entries):
    """(events, n_train) from the most recent binary train_model that logged
    both, or None."""
    for e in reversed(_of(entries, "train_model")):
        o = e["outputs"]
        if o.get("events") is not None and o.get("n_train"):
            return int(o["events"]), int(o["n_train"])
    return None


def _epp(entries):
    """(events, n_parameters) from the most recent train_model with both."""
    for e in reversed(_of(entries, "train_model")):
        o = e["outputs"]
        if o.get("events") is not None and o.get("n_parameters"):
            return int(o["events"]), int(o["n_parameters"])
    return None


def _last(entries, action):
    found = _of(entries, action)
    return found[-1] if found else None


# ----------------------------------------------------------------------------
# checks: each returns (status, note)
# ----------------------------------------------------------------------------
def _check_data_source(entries):
    e = _last(entries, "upload")
    if not e:
        return "not_found", None
    sha = e["inputs"].get("sha256")
    note = f"File SHA-256 {sha[:12]}... recorded." if sha else None
    return "found", note


def _check_eligibility(entries):
    e = _last(entries, "cohort_filter")
    confirm = _last(entries, "confirm")
    if e and confirm and confirm["inputs"].get("cohort_filters") == 0:
        e = None            # the latest confirm dropped the cohort filters
    if not e:
        return "manual_review", "No cohort filter steps were logged; describe eligibility in the manuscript."
    flow = e["outputs"].get("flow") or []
    return "found", f"{len(flow)} inclusion step(s) logged with counts."


def _check_missing_data_handling(entries):
    """not_applicable when nothing was missing, found when every column with
    missing values got a logged decision (or was excluded), not_found
    otherwise, manual_review for older logs without per-column missingness."""
    e = _last(entries, "confirm")
    if not e:
        return "not_applicable", None
    i = e["inputs"]
    pct = i.get("missing_pct", e["outputs"].get("missing_pct"))
    if pct is None:
        return ("manual_review", "This log predates per-column missingness records, "
                "so it can't show whether every column with missing values was handled.")
    if not pct:
        return "not_applicable", "No column had missing values."
    decisions = i.get("missing") or {}
    excluded = set(i.get("excluded") or [])
    undecided = sorted(c for c in pct if c not in decisions and c not in excluded)
    if undecided:
        shown = ", ".join(undecided[:5]) + (", ..." if len(undecided) > 5 else "")
        return "not_found", f"No logged decision for {len(undecided)} column(s) with missing values: {shown}."
    return "found", f"{len(pct)} column(s) had missing values; each has a logged decision."


def _check_outcome_definition(entries):
    found = _has_action(entries, "train_model", "epi_km", "epi_cox", "epi_cif", "adjusted_or")
    return ("found", None) if found else ("not_found", None)


def _check_sample_size(entries):
    """found only when a sample size calculation was logged. a split
    happening is not a sample size justification."""
    epp = _epp(entries)
    epp_note = (f"Development data had {epp[0]} events for {epp[1]} model parameters "
                f"({epp[0] / epp[1]:.1f} events per parameter)." if epp else None)
    e = _last(entries, "sample_size")
    if e:
        o = e["outputs"]
        bits = []
        if o.get("min_n") is not None:
            bits.append(f"minimum n {o['min_n']}")
        if o.get("min_events") is not None:
            bits.append(f"minimum events {o['min_events']}")
        note = f"Sample size calculation logged ({', '.join(bits)})." if bits else \
            "Sample size calculation logged."
        return "found", " ".join(x for x in (note, epp_note) if x)
    note = "No sample size calculation was logged; justify the sample size in the manuscript."
    return "manual_review", " ".join(x for x in (note, epp_note) if x)


def _check_model_building(entries):
    return ("found", None) if _has_action(entries, "train_model") else ("not_found", None)


def _check_model_specification(entries):
    if _has_action(entries, "adjusted_or"):
        return "found", "An adjusted (multivariable) odds ratio model was logged."
    if _has_action(entries, "train_model"):
        return ("manual_review", "Report the final model in full (for regression, every "
                "coefficient and the intercept) or share the model object.")
    return "not_applicable", None


def _check_class_imbalance(entries):
    """never penalises not resampling. found when prevalence is known and
    either no resampling was used, or resampling was used and calibration
    was measured. not_found when SMOTE was used without any calibration."""
    if not _has_action(entries, "train_model"):
        return "not_applicable", None
    smote = _smote_used(entries)
    calibrated = _calibration_measured(entries)
    prev = _prevalence(entries)
    prev_note = (f"Outcome prevalence in training {prev[0]}/{prev[1]} "
                 f"({100 * prev[0] / prev[1]:.1f}%)." if prev else None)
    if smote and not calibrated:
        return "not_found", " ".join(x for x in (
            "SMOTE was used but calibration (slope and calibration-in-the-large) was "
            "never measured. Resampling usually distorts predicted probabilities.",
            prev_note) if x)
    if prev:
        how = ("SMOTE was used and calibration was measured." if smote
               else "No resampling was used.")
        return "found", f"{prev_note} {how}"
    if _binary_status(entries) is False:
        return "not_applicable", "No binary outcome was modelled."
    return "manual_review", "Outcome prevalence was not logged; report it in the manuscript."


def _check_discrimination(entries):
    if not _has_action(entries, "train_model", "test_model", "epi_cox"):
        return "not_applicable", None
    found = _any_metric(entries, ("train_model", "test_model"),
                        ("AUROC", "Accuracy", "MacroF1", "R2"))
    found = found or any(e["action"] == "epi_cox" and "concordance" in e["outputs"]
                         for e in entries)
    if not found:
        return "not_found", None
    has_ci = any(_metrics(e).get("CI") or e["outputs"].get("CI")
                 for e in _of(entries, "test_model"))
    note = (None if has_ci else
            "No confidence intervals were logged; report performance with 95% CIs.")
    return "found", note


def _check_calibration(entries):
    if not _has_action(entries, "train_model", "test_model"):
        return "not_applicable", None
    if _calibration_measured(entries):
        return "found", "Calibration slope and calibration-in-the-large were logged."
    if _has_action(entries, "epi_hl"):
        return ("manual_review", "Only a Hosmer-Lemeshow test was logged. HL is not "
                "recommended for assessing calibration (Van Calster et al., BMC Med 2019); "
                "report the calibration slope, calibration-in-the-large and a calibration plot.")
    return "not_found", "Report the calibration slope, calibration-in-the-large and a calibration plot."


def _check_internal_validation(entries):
    if not _has_action(entries, "train_model", "test_model"):
        return "not_applicable", None
    if not _internal_tests(entries):
        return "not_found", "The internal test partition was never evaluated."
    note = ("The internal test partition was evaluated. Bootstrap internal validation "
            "is preferred over a single random split (Collins et al., BMJ 2024).")
    unlocks = _of(entries, "test_unlock")
    if unlocks:
        models = sorted({m for e in unlocks for m in (e["inputs"].get("models") or [])})
        note += (f" The test partition was reused for {len(models)} model(s) chosen after "
                 "earlier test results were seen, so those results are optimistic.")
    return "found", note


def _check_external_validation(entries):
    if not _has_action(entries, "train_model", "test_model"):
        return "not_applicable", None
    ext = _external_tests(entries)
    if ext:
        files = sorted({e["inputs"].get("filename") or "?" for e in ext})
        return "found", f"Evaluated on external data: {', '.join(files)}."
    return ("not_found", "No external dataset was evaluated. A random split of the same "
            "dataset is internal validation, not external validation.")


def _check_clinical_utility(entries):
    if any(e["outputs"].get("dca") for e in _of(entries, "test_model")):
        return "found", "Decision curve analysis was logged."
    if not _has_action(entries, "train_model", "test_model"):
        return "not_applicable", None
    if _binary_status(entries) is False:
        return "not_applicable", "Decision curves apply to binary outcomes."
    return "not_found", "No decision curve analysis was logged."


def _check_survival_analysis(entries):
    found = _has_action(entries, "epi_km", "epi_cox", "epi_cif")
    return ("found", None) if found else ("not_applicable", None)


def _check_nonlinearity(entries):
    return ("found", None) if _has_action(entries, "fit_spline") else ("not_applicable", None)


# each item: id, section, a description in our own words (not verbatim
# official tripod+ai text), and either a check(entries) -> (status, note), or
# no check function for narrative items no log can ever verify (flagged
# 'manual_review' instead).
_CHECKLIST = [
    {"id": "title_identifies_study_type", "section": "Title & Abstract",
     "description": "Title/abstract identify this as developing and/or validating a "
                     "clinical prediction model.", "check": None},
    {"id": "structured_abstract", "section": "Title & Abstract",
     "description": "Abstract summarizes objectives, data source, methods, results, "
                     "and conclusions.", "check": None},
    {"id": "background_rationale", "section": "Introduction",
     "description": "Background, clinical context, and rationale for the model are "
                     "explained.", "check": None},
    {"id": "objectives", "section": "Introduction",
     "description": "Study objectives, including whether this is model development, "
                     "validation, or both, are stated.", "check": None},
    {"id": "data_source", "section": "Methods: Data",
     "description": "Source of the data (registry/cohort) and study setting are "
                     "described.", "check": _check_data_source},
    {"id": "eligibility_criteria", "section": "Methods: Data",
     "description": "Eligibility criteria for participants/records are described.",
     "check": _check_eligibility},
    {"id": "missing_data_handling", "section": "Methods: Data",
     "description": "How missing data were handled is documented for every variable "
                     "with missing values.", "check": _check_missing_data_handling},
    {"id": "outcome_definition", "section": "Methods: Outcome",
     "description": "The outcome being predicted is clearly defined, including how "
                     "and when it was determined.", "check": _check_outcome_definition},
    {"id": "predictors_specified", "section": "Methods: Predictors",
     "description": "Predictors used in the model are specified, including how and "
                     "when they were measured.", "check": None},
    {"id": "sample_size", "section": "Methods: Sample size",
     "description": "How the study size was arrived at is explained, with a sample "
                     "size calculation showing it was sufficient.", "check": _check_sample_size},
    {"id": "model_building_strategy", "section": "Methods: Analysis",
     "description": "The model-building and hyperparameter-tuning approach (e.g. "
                     "cross-validation vs. a held-out validation set) is described.",
     "check": _check_model_building},
    {"id": "model_specification", "section": "Methods: Analysis",
     "description": "The full model is specified (e.g. all regression coefficients "
                     "and the intercept) so others can make predictions with it.",
     "check": _check_model_specification},
    {"id": "class_imbalance", "section": "Methods: Analysis",
     "description": "Outcome prevalence is reported, and if class imbalance methods "
                     "(e.g. resampling) were used, why, how, and how calibration was "
                     "then checked.", "check": _check_class_imbalance},
    {"id": "discrimination_reported", "section": "Results: Model performance",
     "description": "Discrimination (e.g. AUROC, concordance) is reported, with "
                     "confidence intervals.", "check": _check_discrimination},
    {"id": "calibration_reported", "section": "Results: Model performance",
     "description": "Calibration is reported (calibration slope, calibration-in-the-"
                     "large and a calibration plot).", "check": _check_calibration},
    {"id": "internal_validation", "section": "Results: Model performance",
     "description": "Performance is reported on data held out from model development "
                     "(internal validation). Bootstrap internal validation is preferred "
                     "over a single random split.", "check": _check_internal_validation},
    {"id": "external_validation", "section": "Results: Model performance",
     "description": "Performance is reported on data from a different source, setting "
                     "or period (external validation). A random split of the same "
                     "dataset does not count.", "check": _check_external_validation},
    {"id": "clinical_utility", "section": "Results: Model performance",
     "description": "Clinical utility is assessed (e.g. net benefit from decision curve "
                     "analysis).", "check": _check_clinical_utility},
    {"id": "survival_analysis", "section": "Results (if applicable)",
     "description": "Survival analysis (Kaplan-Meier / Cox / competing risks) is "
                     "reported, if the study has a time-to-event outcome.",
     "check": _check_survival_analysis},
    {"id": "nonlinearity_explored", "section": "Results (if applicable)",
     "description": "Non-linear predictor-outcome relationships are explored (e.g. "
                     "restricted cubic splines), if relevant.",
     "check": _check_nonlinearity},
    {"id": "limitations", "section": "Discussion",
     "description": "Study limitations and risk of bias are discussed.", "check": None},
    {"id": "interpretation", "section": "Discussion",
     "description": "Results are interpreted in the context of clinical practice and "
                     "intended use of the model.", "check": None},
    {"id": "generalizability", "section": "Discussion",
     "description": "Generalizability of the model beyond the study population is "
                     "discussed.", "check": None},
    {"id": "code_data_availability", "section": "Other information",
     "description": "Availability of the code, model, and/or data is stated (an "
                     "AI-specific emphasis in TRIPOD+AI).", "check": None},
    {"id": "funding_and_coi", "section": "Other information",
     "description": "Funding sources and conflicts of interest are declared.",
     "check": None},
]


def generate_tripod_report(session_id):
    """Check a session's analysis log against the TRIPOD+AI reporting
    domains in _CHECKLIST. Returns {disclaimer, items, summary}; each item
    is {id, section, description, status, note}.
    """
    entries = [e for e in analysis_log.read_log(session_id)
               if e.get("action") not in _IGNORED]
    items = []
    for spec in _CHECKLIST:
        if spec["check"] is None:
            status, note = "manual_review", None
        else:
            status, note = spec["check"](entries)
        items.append({
            "id": spec["id"],
            "section": spec["section"],
            "description": spec["description"],
            "status": status,
            "note": note,
        })

    summary = {
        "total": len(items),
        "found": sum(1 for i in items if i["status"] == "found"),
        "not_found": sum(1 for i in items if i["status"] == "not_found"),
        "not_applicable": sum(1 for i in items if i["status"] == "not_applicable"),
        "manual_review": sum(1 for i in items if i["status"] == "manual_review"),
    }

    return {"disclaimer": DISCLAIMER, "items": items, "summary": summary}
