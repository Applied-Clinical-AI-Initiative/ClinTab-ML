"""stats.py
Data profiling + descriptive statistics. Pure pandas/numpy -- no Flask, no
sklearn. Two jobs:

  1. detect_column_types()  -> auto-classify each column as binary / categorical
     / continuous / date, with missingness, so the upload page can pre-fill the
     radio buttons.
  2. summarize()            -> the Data Summary section tables.
"""
import warnings

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Column type detection
# ----------------------------------------------------------------------------
# Heuristics (date / integer test) only need a sample. On large registries
# parsing every full string column with dateutil is the upload bottleneck, so we
# cap the work at SAMPLE_N non-null values per column. Exact counts that are
# cheap (missingness, nunique) are still computed on the full column.
SAMPLE_N = 4000


def _looks_like_date(sample):
    """Return True if a sample of (object) values parses cleanly as dates."""
    if sample.empty:
        return False
    if not (sample.dtype == object or str(sample.dtype).startswith("datetime")):
        return False
    # cheap reject: values with no date-ish separators are almost never dates
    head = sample.astype(str).head(50)
    if not head.str.contains(r"[-/:]|\d{4}", regex=True).any():
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        return False
    return parsed.notna().mean() >= 0.8


def _is_integer_like(sample):
    s = pd.to_numeric(sample, errors="coerce").dropna()
    if s.empty:
        return False
    return bool(np.all(np.equal(np.mod(s, 1), 0)))


def detect_column_types(df):
    """Classify every column. Rules (from spec):
       - binary      : exactly 2 unique non-null values
       - categorical : strings, OR integer-like numeric with < 10 unique values
       - continuous  : numeric with > 10 unique values
       - date        : parses as datetime (excluded from modelling by default)
    Returns a list of dicts (preserves column order).
    """
    n = len(df)
    cols = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        n_unique = int(non_null.nunique())
        pct_missing = round(100.0 * (n - len(non_null)) / n, 2) if n else 0.0
        numeric = pd.api.types.is_numeric_dtype(s)
        sample = non_null if len(non_null) <= SAMPLE_N else non_null.sample(SAMPLE_N, random_state=0)

        if not numeric and _looks_like_date(sample):
            ctype = "date"
        elif n_unique == 2:
            ctype = "binary"
        elif not numeric:
            ctype = "categorical"
        elif n_unique < 10 and _is_integer_like(sample):
            ctype = "categorical"
        elif n_unique > 10:
            ctype = "continuous"
        else:
            # numeric, 3..10 uniques -> treat as categorical (ordinal-ish)
            ctype = "categorical"

        cols.append({
            "name": str(col),
            "type": ctype,
            "n_unique": n_unique,
            "pct_missing": pct_missing,
            "high_missing": pct_missing > 50.0,
            "numeric": bool(numeric),
            "sample": [str(x) for x in non_null.unique()[:5]],
        })
    return cols


def minority_fraction(series):
    """Smallest class proportion for a binary/categorical column (for SMOTE hint)."""
    vc = series.dropna().value_counts(normalize=True)
    if vc.empty:
        return None
    return float(vc.min())


# ----------------------------------------------------------------------------
# Descriptive summary
# ----------------------------------------------------------------------------
def _continuous_row(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {"n": 0, "mean": None, "median": None, "sd": None,
                "iqr": None, "min": None, "max": None}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n": int(v.shape[0]),
        "mean": round(float(v.mean()), 4),
        "median": round(float(v.median()), 4),
        "sd": round(float(v.std(ddof=1)), 4) if v.shape[0] > 1 else 0.0,
        "iqr": f"{round(float(q1), 3)}–{round(float(q3), 3)}",
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
    }


def _categorical_rows(s):
    v = s.dropna()
    n = len(v)
    vc = v.value_counts()
    rows = []
    for cat, cnt in vc.items():
        rows.append({
            "category": str(cat),
            "count": int(cnt),
            "pct": round(100.0 * cnt / n, 2) if n else 0.0,
        })
    return rows


def summarize(df, coltypes):
    """Build the Data Summary payload.
    coltypes: dict {col_name: type}.
    """
    n_total = len(df)
    continuous, categorical = [], []

    for col in df.columns:
        ctype = coltypes.get(str(col), "categorical")
        s = df[col]
        pct_missing = round(100.0 * s.isna().mean(), 2) if n_total else 0.0
        n_present = int(s.notna().sum())

        if ctype == "continuous":
            row = {"variable": str(col), "n": n_present, "pct_missing": pct_missing}
            row.update(_continuous_row(s))
            continuous.append(row)
        elif ctype in ("binary", "categorical"):
            categorical.append({
                "variable": str(col),
                "n": n_present,
                "pct_missing": pct_missing,
                "type": ctype,
                "categories": _categorical_rows(s),
            })
        # date columns are skipped in the variable tables (used for the card only)

    # Top card -------------------------------------------------------------
    complete_cases = int(df.dropna().shape[0])
    pct_complete = round(100.0 * complete_cases / n_total, 2) if n_total else 0.0

    date_range = None
    for col in df.columns:
        if coltypes.get(str(col)) == "date":
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                date_range = {
                    "column": str(col),
                    "start": str(parsed.min().date()),
                    "end": str(parsed.max().date()),
                }
                break

    return {
        "card": {
            "total_n": n_total,
            "n_columns": df.shape[1],
            "complete_cases": complete_cases,
            "pct_complete_cases": pct_complete,
            "date_range": date_range,
        },
        "continuous": continuous,
        "categorical": categorical,
    }


def apply_missing_handling(df, decisions):
    """decisions: dict {col: 'include' | 'zero' | 'remove'}.
       'remove'  -> drop the column entirely
       'zero'    -> fill NaN with 0
       'include' -> leave as-is (rows may be dropped later by complete-case)
    Returns the cleaned DataFrame.
    """
    df = df.copy()
    drop = [c for c, d in decisions.items() if d == "remove" and c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    for c, d in decisions.items():
        if d == "zero" and c in df.columns:
            df[c] = df[c].fillna(0)
    return df
