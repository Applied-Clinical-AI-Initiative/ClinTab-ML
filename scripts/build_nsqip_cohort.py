"""build_nsqip_cohort.py
Protocol Section 1: builds the arthroplasty cohort from local ACS NSQIP PUF
extracts and writes a single clean CSV ready to feed into ClinTAB-ML.

Input files are split by procedure and revision status:
TKA (primary knee), TKA REV (revision knee), THA (primary hip), THA REV
(revision hip). Each file spans multiple operative years internally via a
year column (OPERYR). The protocol calls for 2021-2023 only, pooled, primary,
unilateral cases only -- revision files are accepted for completeness but
excluded from the cohort by construction, any row whose year falls outside
2021-2023 is dropped and counted, and bilateral cases (same-side procedure
code repeated as a concurrent CPT on the same row) are dropped and counted.

The script reads local files and writes patient-level values only to the
specified output CSV. Console output contains aggregate counts and missingness.

DO NOT commit the raw files or the output CSV. Both are covered by your PUF
data use agreement. Only this script belongs in the repository.

Usage:
    python scripts/build_nsqip_cohort.py \
        --tka /path/to/TKA.csv \
        --tha /path/to/THA.csv \
        --tka-rev /path/to/TKA_REV.csv \
        --tha-rev /path/to/THA_REV.csv \
        --out /path/outside/the/repo/real_cohort.csv

--tka-rev and --tha-rev are optional (accepted for completeness, e.g. if you
later want a revision-cohort comparison) but never contribute rows to the
primary cohort this script writes.

Accepts .csv, .csv.gz, or .sas7bdat per file (auto-detected by extension).

The column mapping below reflects the headers of the available NSQIP extract.
Its headers lack standard complication flags for Y2 and death-timing fields
for Y4. When these fields are absent, the script warns and omits those
outcomes. Add alternate field names to MORBIDITY_COMPONENTS or DEATH_FIELDS
when applicable.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# 1.1 Scope
# ============================================================================
INCLUDED_YEARS = {2021, 2022, 2023}
CPT_BY_PROCEDURE = {"tka": "27447", "tha": "27130"}

# ============================================================================
# 1.3 Variable mapping, raw PUF column -> canonical name.
# One file = one procedure/revision category, but all four share the same
# column layout, so a single mapping applies to all of them.
# ============================================================================
COLUMN_MAP = {
    # identifiers / filtering
    "CPT": "cpt",                          # may not be present -- see procedure sanity check below
    "ADMYR": "admyr",                      # admission year (distinct from OPERYR)
    "SEX": "sex",
    "AGE": "age",
    "ELECTSURG": "elective",
    "EMERGNCY": "emergency",
    "OPERYR": "operyr",
    "INOUT": "inout",                      # inpatient/outpatient, passthrough for Table 1

    # demographics carried through for Table 1 -- NOT model predictors unless
    # you decide to add them to --features later
    "RACE_NEW": "race",
    "ETHNICITY_HISPANIC": "ethnicity_hispanic",

    # predictors (1.3)
    "BMI": "bmi",                          # already computed in this extract -- do not re-derive
    "BMI_CAT": "bmi_cat",
    "ASACLAS": "asa_class",
    "FNSTATUS2": "functional_status",
    "DIABETES": "diabetes",
    "HYPERMED": "hypertension_med",
    "HXCOPD": "copd",
    "HXCHF": "chf",
    "SMOKE": "current_smoker",
    "DIALYSIS": "dialysis",
    "PRALBUM": "pre_albumin",
    "PRHCT": "pre_hct",
    "PRCREAT": "pre_creatinine",
    "OPTIME": "operative_time",
    "ANESTHES": "anesthesia_type",

    # outcomes (1.2)
    "UNPLANNEDREADMISSION1": "unplanned_readmission",  # Y1 source -- NOT READMISSION1 (that's any readmission, not specifically unplanned)
    "TOTHLOS": "length_of_stay",           # Y3 source
    "DOPTODIS": "days_op_to_discharge",    # not currently used below, kept for reference
}

# concurrent-procedure fields used to detect bilateral cases (1.1 exclusion):
# a row is bilateral if this file's own procedure CPT also shows up among its
# own concurrent-procedure codes on the same row.
CONCURRENT_CPT_FIELDS = [f"OTHERCPT{i}" for i in range(4, 11)] + [f"CONCPT{i}" for i in range(4, 11)]

# The Y2 serious-morbidity definition varies by study. Confirm these
# components against the intended definition before using Y2 in an analysis.
# None was present in the available extract's headers.
MORBIDITY_COMPONENTS = [
    "SUPINFEC", "WNDINFD", "ORGSPCSSI", "DEHIS", "OUPNEUMO", "REINTUB",
    "PULEMBOL", "FAILWEAN", "RENAINSF", "OPRENAFL", "URNINFEC", "CNSCVA",
    "CDARREST", "CDMI", "OTHBLEED", "OTHDVT", "OTHSYSEP", "OTHSESHOCK",
]

# These Y4 death-field candidates were absent from the available extract's
# headers. Add an alternate field name here if the source provides one.
DEATH_FIELDS = ["DOD", "DEATH30YN", "YRDEATH"]


def _read_any(path):
    path = Path(path)
    if path.suffix == ".sas7bdat":
        return pd.read_sas(path, encoding="latin1")
    return pd.read_csv(path, low_memory=False)


def _map_columns(df, label):
    missing = [raw for raw in COLUMN_MAP if raw not in df.columns]
    if missing:
        print(f"  [{label}] WARNING: {len(missing)} expected columns not found: {missing}",
              file=sys.stderr)
    present = {raw: canon for raw, canon in COLUMN_MAP.items() if raw in df.columns}
    out = df[list(present)].rename(columns=present)
    for c in MORBIDITY_COMPONENTS:
        if c in df.columns:
            out[c] = df[c]
    for c in DEATH_FIELDS:
        if c in df.columns:
            out[c] = df[c]
    for c in CONCURRENT_CPT_FIELDS:
        if c in df.columns:
            out[c] = df[c]
    return out


def _load_category(path, procedure, is_revision, n_at_step):
    if path is None:
        return None
    label = f"{procedure}{'_rev' if is_revision else ''}"
    print(f"Reading {label} from {path} ...")
    raw = _read_any(path)
    n_at_step.setdefault("raw_rows_by_file", {})[label] = len(raw)

    mapped = _map_columns(raw, label)
    mapped["_procedure"] = procedure
    mapped["_is_revision"] = is_revision

    # sanity check: does the file's own CPT column agree with the procedure
    # this file is supposed to be? Reported, not silently trusted. Some
    # extracts drop the CPT column once already split by procedure -- that's
    # fine, this check just no-ops if it's not there.
    expected_cpt = CPT_BY_PROCEDURE[procedure]
    if "cpt" in mapped:
        mismatched = (mapped["cpt"].apply(_norm_code) != expected_cpt).sum()
        if mismatched:
            print(f"  [{label}] NOTE: {mismatched} rows have a CPT other than "
                  f"{expected_cpt}, verify this file is what its name says", file=sys.stderr)
    return mapped


def _norm_code(x):
    """Normalize a procedure code for comparison -- CSV round-tripping turns
    a numeric-looking code like "27447" into the float 27447.0, so a plain
    string comparison silently never matches. Coerce both sides through the
    same float-then-strip path so "27447", 27447, and 27447.0 all agree."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except ValueError:
        return s


def _mark_bilateral(df):
    """A row is bilateral if its own procedure's CPT also appears among its
    own concurrent-procedure codes -- i.e. the same joint was done twice in
    the same encounter."""
    present_fields = [c for c in CONCURRENT_CPT_FIELDS if c in df.columns]
    if not present_fields:
        print("WARNING: no OTHERCPT/CONCPT concurrent-procedure columns found -- "
              "cannot detect bilateral cases, they will NOT be excluded.", file=sys.stderr)
        df["_is_bilateral"] = False
        return df

    def row_is_bilateral(row):
        expected = _norm_code(CPT_BY_PROCEDURE.get(row["_procedure"]))
        vals = [_norm_code(row[c]) for c in present_fields if pd.notna(row.get(c))]
        return expected in vals

    df["_is_bilateral"] = df.apply(row_is_bilateral, axis=1)
    return df


def _derive_outcomes(df):
    if "unplanned_readmission" in df:
        readmission = df["unplanned_readmission"].astype("string").str.strip().str.upper()
        df["Y1_readmission"] = readmission.map({"YES": 1, "NO": 0, "1": 1, "0": 0})
    else:
        print("WARNING: no UNPLANNEDREADMISSION1 column found -- Y1 cannot be derived.",
              file=sys.stderr)

    present_components = [c for c in MORBIDITY_COMPONENTS if c in df.columns]
    if present_components:
        flags = df[present_components].apply(
            lambda s: s.astype(str).str.strip().str.upper().eq("YES")
        )
        df["Y2_serious_morbidity"] = flags.any(axis=1).astype(int)
        df = df.drop(columns=present_components)
    else:
        print("WARNING: none of the standard NSQIP complication-flag columns "
              "(SUPINFEC, PULEMBOL, ...) were found -- Y2_serious_morbidity was "
              "NOT created. If your PUF pull has these under different names, "
              "add them to MORBIDITY_COMPONENTS in this script.", file=sys.stderr)

    if "length_of_stay" in df:
        df["Y3_length_of_stay"] = pd.to_numeric(df["length_of_stay"], errors="coerce")

    death_field = next((c for c in DEATH_FIELDS if c in df.columns), None)
    if death_field:
        raw = df[death_field]
        d = pd.to_numeric(raw, errors="coerce")
        if d.notna().any():
            df["Y4_time"] = d.clip(upper=30).fillna(30)
            df["Y4_event"] = (d <= 30).fillna(False).astype(int)
        else:
            yes = raw.astype(str).str.strip().str.upper().isin(["YES", "Y", "1"])
            print(f"WARNING: {death_field} is a yes/no flag with no timing; Y4_time is set to "
                  "30 for everyone, so only event counts (not KM/Cox timing) are meaningful.",
                  file=sys.stderr)
            df["Y4_time"] = 30.0
            df["Y4_event"] = yes.astype(int)
        df = df.drop(columns=[c for c in DEATH_FIELDS if c in df.columns])
    else:
        print("WARNING: no death-timing column was found -- Y4_time/Y4_event were NOT "
              "created. Run check_puf_headers.py to find the right name, then pass it "
              "with --death-field NAME.", file=sys.stderr)

    return df


def build(tka_path, tha_path, tka_rev_path, tha_rev_path, out_path):
    n_at_step = {}
    frames = []
    for procedure, path, is_rev in [
        ("tka", tka_path, False), ("tha", tha_path, False),
        ("tka", tka_rev_path, True), ("tha", tha_rev_path, True),
    ]:
        frame = _load_category(path, procedure, is_rev, n_at_step)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise SystemExit("No input files given -- pass at least --tka and/or --tha.")

    df = pd.concat(frames, ignore_index=True)
    n_at_step["pooled_all_files"] = len(df)

    # Primary cohort only -- revision files are loaded but excluded here,
    # by construction rather than by CPT-based guessing.
    n_revision = int(df["_is_revision"].sum())
    df = df[~df["_is_revision"]].drop(columns=["_is_revision"])
    n_at_step["after_excluding_revision_files"] = len(df)
    n_at_step["revision_rows_excluded"] = n_revision

    # Bilateral exclusion -- same joint done twice in one encounter.
    df = _mark_bilateral(df)
    n_bilateral = int(df["_is_bilateral"].sum())
    df = df[~df["_is_bilateral"]].drop(columns=["_is_bilateral"] + CONCURRENT_CPT_FIELDS,
                                        errors="ignore")
    n_at_step["after_excluding_bilateral"] = len(df)
    n_at_step["bilateral_rows_excluded"] = n_bilateral

    # Year filter -- 2021-2023 ONLY. Anything outside that range is dropped
    # and counted, not kept.
    if "operyr" in df:
        year_num = pd.to_numeric(df["operyr"], errors="coerce")
        n_before = len(df)
        df = df[year_num.isin(INCLUDED_YEARS)]
        n_at_step["dropped_rows_outside_2021_2023"] = n_before - len(df)
    else:
        print("WARNING: no operyr/OPERYR column found -- cannot enforce the "
              "2021-2023 filter. Check the year-column mapping.", file=sys.stderr)
    n_at_step["after_year_filter_2021_2023"] = len(df)

    if "age" in df:
        age_num = pd.to_numeric(df["age"].replace("90+", "90"), errors="coerce")
        df = df[age_num >= 18]
    n_at_step["after_age_filter"] = len(df)

    if "elective" in df:
        df = df[df["elective"].astype(str).str.strip().str.upper().eq("YES")]
    n_at_step["after_elective_filter"] = len(df)

    if "emergency" in df:
        df = df[~df["emergency"].astype(str).str.strip().str.upper().eq("YES")]
    n_at_step["after_emergency_exclusion"] = len(df)

    df = _derive_outcomes(df)

    if "Y1_readmission" in df:
        df = df[df["Y1_readmission"].notna()]
    n_at_step["after_missing_outcome_exclusion"] = len(df)

    df.to_csv(out_path, index=False)

    print("\n=== Inclusion flow (aggregate counts only -- safe to report/share) ===")
    for step, n in n_at_step.items():
        print(f"  {step}: {n}")
    print(f"\nWrote {len(df)} rows x {len(df.columns)} columns to {out_path}")
    if "Y2_serious_morbidity" not in df.columns:
        print("\n*** Y2 (serious morbidity) was NOT created -- see WARNING above. ***")
    if "Y4_time" not in df.columns:
        print("*** Y4 (time to death) was NOT created -- see WARNING above. ***")
    print("\nDo not commit this file. Add it to .gitignore if it isn't already.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tka", help="path to the primary TKA file")
    ap.add_argument("--tha", help="path to the primary THA file")
    ap.add_argument("--tka-rev", help="path to the TKA revision file (excluded from cohort)")
    ap.add_argument("--tha-rev", help="path to the THA revision file (excluded from cohort)")
    ap.add_argument("--out", required=True, help="output CSV path (outside the repo)")
    ap.add_argument("--death-field", help="raw column holding days from operation to death (Y4)")
    ap.add_argument("--morbidity-cols", nargs="*", default=[],
                    help="extra raw complication-flag columns to include in Y2")
    args = ap.parse_args()
    if args.death_field:
        DEATH_FIELDS.insert(0, args.death_field)
    MORBIDITY_COMPONENTS.extend(args.morbidity_cols)
    build(args.tka, args.tha, args.tka_rev, args.tha_rev, args.out)
