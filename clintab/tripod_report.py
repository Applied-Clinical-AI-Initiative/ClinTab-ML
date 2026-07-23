"""tripod_report.py
Maps a session's analysis log onto the reporting domains of TRIPOD+AI
(Collins et al., BMJ 2024) -- the reporting guideline for studies developing
or validating clinical prediction models, including ones built with machine
learning.

Important: this is a practical subset of TRIPOD+AI's reporting domains, not
a verbatim reproduction of all 27 official checklist items -- treat it as a
starting point, not a substitute for the published checklist. And for every
item, "evidence found" only means the underlying analysis step is traceable
in this session's log. It does NOT mean the manuscript actually reports it
correctly, or at all -- narrative items (rationale, limitations, clinical
interpretation, and so on) can't be verified from a log no matter what, and
are flagged for manual review rather than guessed at.

Pure Python, no Flask.
"""
from clintab import analysis_log

DISCLAIMER = (
    "This report checks what this session's analysis log can prove was "
    "actually run -- it is not an official TRIPOD+AI compliance check, and "
    "'evidence found' does not mean your manuscript reports an item "
    "correctly (or at all). Items marked 'manual review' are narrative "
    "content a log can never verify. Always check the published TRIPOD+AI "
    "checklist (Collins et al., BMJ 2024) directly before submission."
)


def _has_action(entries, *actions):
    return any(e["action"] in actions for e in entries)


def _any_metric(entries, actions, keys):
    for e in entries:
        if e["action"] in actions:
            metrics = e["outputs"].get("metrics") or e["outputs"]
            if any(k in metrics for k in keys):
                return True
    return False


def _check_data_source(entries):
    return "found" if _has_action(entries, "upload") else "not_found"


def _check_sample_size(entries):
    return "found" if _has_action(entries, "confirm") else "not_found"


def _check_outcome_definition(entries):
    return "found" if _has_action(entries, "train_model", "epi_km", "epi_cox") else "not_found"


def _check_model_building(entries):
    return "found" if _has_action(entries, "train_model") else "not_found"


def _check_discrimination(entries):
    if not _has_action(entries, "train_model", "test_model", "epi_cox"):
        return "not_applicable"
    found = _any_metric(entries, ("train_model", "test_model"),
                        ("AUROC", "Accuracy", "MacroF1", "R2"))
    found = found or any(e["action"] == "epi_cox" and "concordance" in e["outputs"]
                         for e in entries)
    return "found" if found else "not_found"


def _check_calibration(entries):
    if not _has_action(entries, "train_model", "test_model"):
        return "not_applicable"
    return "found" if _has_action(entries, "epi_hl") else "not_found"


def _check_external_validation(entries):
    if not _has_action(entries, "train_model"):
        return "not_applicable"
    return "found" if _has_action(entries, "test_model") else "not_found"


def _check_survival_analysis(entries):
    return "found" if _has_action(entries, "epi_km", "epi_cox") else "not_applicable"


def _check_nonlinearity(entries):
    return "found" if _has_action(entries, "fit_spline") else "not_applicable"


# Each item: id, section, a description in our own words (not verbatim
# official TRIPOD+AI text), and either a check(entries) -> 'found' |
# 'not_found' | 'not_applicable', or check=None for narrative items no log
# can ever verify (flagged 'manual_review' instead).
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
     "check": None},
    {"id": "missing_data_handling", "section": "Methods: Data",
     "description": "How missing data were handled is documented.", "check": None},
    {"id": "outcome_definition", "section": "Methods: Outcome",
     "description": "The outcome being predicted is clearly defined, including how "
                     "and when it was determined.", "check": _check_outcome_definition},
    {"id": "predictors_specified", "section": "Methods: Predictors",
     "description": "Predictors used in the model are specified, including how and "
                     "when they were measured.", "check": None},
    {"id": "sample_size", "section": "Methods: Sample size",
     "description": "Sample sizes for development/validation/test sets are reported, "
                     "with justification.", "check": _check_sample_size},
    {"id": "model_building_strategy", "section": "Methods: Analysis",
     "description": "The model-building and hyperparameter-tuning approach (e.g. "
                     "cross-validation vs. a held-out validation set) is described.",
     "check": _check_model_building},
    {"id": "class_imbalance", "section": "Methods: Analysis",
     "description": "If class imbalance was addressed (e.g. resampling), this is "
                     "documented.", "check": None},
    {"id": "discrimination_reported", "section": "Results: Model performance",
     "description": "Discrimination (e.g. AUROC, concordance) is reported.",
     "check": _check_discrimination},
    {"id": "calibration_reported", "section": "Results: Model performance",
     "description": "Calibration (e.g. Hosmer-Lemeshow, calibration plot) is reported.",
     "check": _check_calibration},
    {"id": "external_validation", "section": "Results: Model performance",
     "description": "Performance is reported on a held-out test set, separate from "
                     "model development.", "check": _check_external_validation},
    {"id": "survival_analysis", "section": "Results (if applicable)",
     "description": "Survival analysis (Kaplan-Meier / Cox) is reported, if the study "
                     "has a time-to-event outcome.", "check": _check_survival_analysis},
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
    domains in _CHECKLIST. Returns {disclaimer, items, summary}.
    """
    entries = analysis_log.read_log(session_id)
    items = []
    for spec in _CHECKLIST:
        if spec["check"] is None:
            status = "manual_review"
        else:
            status = spec["check"](entries)
        items.append({
            "id": spec["id"],
            "section": spec["section"],
            "description": spec["description"],
            "status": status,
        })

    summary = {
        "total": len(items),
        "found": sum(1 for i in items if i["status"] == "found"),
        "not_found": sum(1 for i in items if i["status"] == "not_found"),
        "not_applicable": sum(1 for i in items if i["status"] == "not_applicable"),
        "manual_review": sum(1 for i in items if i["status"] == "manual_review"),
    }

    return {"disclaimer": DISCLAIMER, "items": items, "summary": summary}
