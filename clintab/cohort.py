"""cohort.py
Study cohort selection with a recorded flow, so the inclusion and exclusion
steps can be reported (STROBE / TRIPOD flow diagram, methods paragraph).
Pure pandas/numpy, no Flask. Generic, nothing NSQIP specific: the same filters
work on NSQIP (CPT plus OTHERCPT1 to OTHERCPT10), a hospital extract (DX1 to
DX25), a provincial administrative file or a single-centre cohort.

  1. parse_code_list()  -> pasted text ('27130-27132, 27447; M16*') to a clean
     list of codes.
  2. apply_filters()    -> run an ordered list of filters, returning the cohort
     and the counts at every step.
  3. summarize_flow()   -> plain text lines for a methods paragraph or the boxes
     of a flow diagram.

Code matching (ops 'in' / 'not_in') compares a normalised key: case, spaces,
dots and hyphens are ignored, so M16.1 equals m161 and 1.VA.53.LA-PN-N equals
1VA53LAPNN. Numbers match whatever their storage, 27447, 27447.0, '27447' and
' 27447 ' are the same code. Leading zeros in text are kept (ICD-9 0389 is not
389); a column read as numbers has already lost them, so for numeric cells an
all-digit code also matches without its leading zeros (00100 matches 100).
A trailing * is a prefix wildcard (M16* matches M16.0, M161, M16.12).
"""
import difflib
import re

import numpy as np
import pandas as pd

OPS = ("in", "not_in", "between", "gte", "lte", "eq", "neq", "not_missing")
_CODE_OPS = ("in", "not_in")
_SINGLE_COLUMN_OPS = ("between", "gte", "lte", "eq", "neq")
_INCLUDE_OPS = ("in", "eq")
# ops whose removed rows include the missing / non-numeric ones (for the notes)
_DROPS_MISSING = ("in", "eq", "between", "gte", "lte")
_DROPS_NON_NUMERIC = ("eq", "between", "gte", "lte")

MAX_RANGE_CODES = 10000     # a range wider than this is almost surely a typo
_LABEL_CODES = 8            # codes spelled out in an auto label before "and N more"
_LABEL_COLUMNS = 4          # columns spelled out in an auto label
_EXAMPLES = 5               # distinct odd values kept per step

_SPLIT = re.compile(r"[,;\s]+")
_SPACED_HYPHEN = re.compile(r"\s*-\s*")
_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2212]")  # dash look-alikes pasted from word processors
_FLOAT_TAIL = re.compile(r"^(\d{4,})\.0+$")   # 27447.0 is a float artefact, 81.0 and 250.00 are ICD-9
_KEY_STRIP = re.compile(r"[\s.\-]+")
_TOP_CODED = r"^(\d+(?:\.\d+)?)\s*\+$"        # '90+' style top coding (de-identified ages)


class CohortFilterError(ValueError):
    """Bad filter list. .problems holds every problem found, one string each."""

    def __init__(self, problems):
        self.problems = list(problems)
        if len(self.problems) == 1:
            msg = f"Cohort filter problem: {self.problems[0]}"
        else:
            msg = (f"{len(self.problems)} problems with the cohort filters:\n"
                   + "\n".join(f"- {p}" for p in self.problems))
        super().__init__(msg)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _is_digits(s):
    return s.isascii() and s.isdigit()


def _has_digit(s):
    return any(ch.isdigit() for ch in s)


def _num_text(x):
    """18.0 -> '18', 2.5 -> '2.5'."""
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def _plain_num(x):
    """number for the json flow: int when whole, else float."""
    x = float(x)
    return int(x) if x.is_integer() else x


def _to_float(x):
    """number or numeric string -> finite float, else None."""
    if isinstance(x, (bool, np.bool_)) or x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
    elif isinstance(x, str):
        try:
            v = float(x.strip())
        except ValueError:
            return None
    else:
        return None
    return v if np.isfinite(v) else None


def _text_key(t):
    """comparison key for a code written as text."""
    t = str(t).strip().upper()
    m = _FLOAT_TAIL.match(t)
    if m:
        t = m.group(1)
    return _KEY_STRIP.sub("", t)


def _cell_key(v):
    """(key, is_number) for one data cell. key is None when the cell is blank."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).upper(), False
    if isinstance(v, (int, np.integer)):
        return str(int(v)), True
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return None, True
        return _num_text(v).replace(".", ""), True
    k = _text_key(v)
    return (k or None), False


def _count(n, noun):
    """3, 'value' -> '3 values'."""
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def _quote_list(items, limit=None):
    items = [f"'{x}'" for x in items]
    if limit is not None and len(items) > limit:
        return ", ".join(items[:limit]) + f" and {len(items) - limit} more"
    return ", ".join(items)


# ----------------------------------------------------------------------------
# code lists
# ----------------------------------------------------------------------------
def _expand_range(lo, hi, raw):
    a, b = int(lo), int(hi)
    if a > b:
        return [], f"range '{raw}' is reversed, the start is above the end"
    n = b - a + 1
    if n > MAX_RANGE_CODES:
        return [], f"range '{raw}' would expand to {n:,} codes, the limit is {MAX_RANGE_CODES:,}"
    width = len(lo) if lo.startswith("0") else 0      # keep zero padding, 00100-00199
    return [str(i).zfill(width) for i in range(a, b + 1)], None


def _parse_token(tok):
    """one token -> (codes, problem). wildcards keep their trailing *."""
    if isinstance(tok, (bool, np.bool_)):
        return [str(bool(tok)).upper()], None
    if isinstance(tok, (int, np.integer)):
        return [str(int(tok))], None
    if isinstance(tok, (float, np.floating)):
        if not np.isfinite(tok):
            return [], None
        return [_num_text(tok)], None
    if tok is None:
        return [], None
    if not isinstance(tok, str):
        return [], f"could not read {tok!r} as a code, give text or a number"
    raw = tok.strip()
    t = _DASHES.sub("-", raw).strip("'\"").strip().upper()
    t = _SPACED_HYPHEN.sub("-", t)
    if not t:
        return [], None
    if "*" in t:
        base = t.rstrip("*")
        if "*" in base:
            return [], (f"'{raw}' has a * inside the code, only a trailing * works "
                        "(a prefix wildcard such as M16*)")
        if not _text_key(base):
            return [], f"'{raw}' would match every code, put at least one character before the *"
        return [" ".join(base.split()) + "*"], None
    if t.count("-") == 1 and not re.search(r"\s", t):
        lo, hi = t.split("-")
        if not lo or not hi:
            return [], f"'{raw}' is an incomplete range, write it as 27130-27132"
        if _is_digits(lo) and _is_digits(hi):
            return _expand_range(lo, hi, raw)
        if _has_digit(lo) and _has_digit(hi):
            return [], (f"'{raw}' looks like a range but only whole-number ranges can be "
                        "expanded, use a prefix wildcard such as M16* for a code family")
        # one hyphen without digits on both sides is a plain value (Non-Hispanic)
    m = _FLOAT_TAIL.match(t)
    if m:
        t = m.group(1)
    return [" ".join(t.split())], None


def _split_text(text):
    t = _DASHES.sub("-", text)
    t = _SPACED_HYPHEN.sub("-", t)        # '27130 - 27132' is one range
    return [x for x in _SPLIT.split(t) if x]


def _parse_tokens(tokens):
    """tokens -> (deduplicated codes in input order, problems)."""
    codes, seen, problems = [], set(), []
    for tok in tokens:
        got, problem = _parse_token(tok)
        if problem:
            problems.append(problem)
            continue
        for c in got:
            if c not in seen:
                seen.add(c)
                codes.append(c)
    return codes, problems


def _parse_text(text):
    if text is None:
        return [], []
    if isinstance(text, str):
        return _parse_tokens(_split_text(text))
    if isinstance(text, (list, tuple)):
        tokens = []
        for item in text:
            tokens.extend(_split_text(item) if isinstance(item, str) else [item])
        return _parse_tokens(tokens)
    return _parse_tokens([text])


def parse_code_list(text):
    """Pasted codes -> list of normalised code strings, in input order, no repeats.

    Separators: comma, semicolon, spaces, tabs or newlines. '27130-27132'
    expands to 27130, 27131, 27132 (zero padding kept, '00100-00102' gives
    00100, 00101, 00102). Other codes are kept as written, stripped and
    uppercased ('m16.1' -> 'M16.1'). A trailing * is a prefix wildcard and is
    kept on the code ('M16*'). A list of strings is also accepted, each item
    is split the same way.

    Raises ValueError, naming every bad token, for ranges whose ends are not
    whole numbers ('M16.0-M16.9'), reversed ranges, ranges over
    MAX_RANGE_CODES codes, incomplete ranges ('27130-') and a * anywhere but
    the end.
    """
    codes, problems = _parse_text(text)
    if problems:
        raise ValueError("Could not read the code list: " + "; ".join(problems))
    return codes


def _parse_values(values):
    """filter values for in / not_in -> (codes, problems). a string is parsed
    like parse_code_list, a list is one value per item (so category values
    with spaces, such as 'Black or African American', stay whole)."""
    if isinstance(values, str):
        return _parse_text(values)
    if isinstance(values, (list, tuple)):
        return _parse_tokens(values)
    return _parse_tokens([values])


def _compact(codes):
    """['27130', '27131', '27132', '27447'] -> ['27130-27132', '27447'].
    only runs of 3+ consecutive whole numbers of the same width are merged,
    so the result parses back to the same codes."""
    out, i = [], 0
    while i < len(codes):
        c, j = codes[i], i
        if _is_digits(c):
            while (j + 1 < len(codes) and _is_digits(codes[j + 1])
                   and len(codes[j + 1]) == len(c)
                   and int(codes[j + 1]) == int(codes[j]) + 1):
                j += 1
        if j - i >= 2:
            out.append(f"{c}-{codes[j]}")
        else:
            out.extend(codes[i:j + 1])
        i = j + 1
    return out


def _match_sets(codes):
    """display codes -> (exact keys, exact keys for numeric cells, prefixes)."""
    exact, prefixes = set(), []
    for c in codes:
        if c.endswith("*"):
            prefixes.append(_text_key(c[:-1]))
        else:
            exact.add(_text_key(c))
    exact_num = set(exact)
    exact_num.update(k.lstrip("0") or "0" for k in exact if _is_digits(k))
    return exact, exact_num, tuple(prefixes)


# ----------------------------------------------------------------------------
# column evaluation (each returns boolean numpy arrays over the whole frame)
# ----------------------------------------------------------------------------
def _code_hits(series, exact, exact_num, prefixes):
    """(matches, blank) per row. works on the distinct values only, so a few
    thousand distinct codes in 300k rows is cheap."""
    codes, uniques = pd.factorize(series)
    hit = np.zeros(len(uniques) + 1, dtype=bool)
    blank = np.zeros(len(uniques) + 1, dtype=bool)
    blank[-1] = True                      # factorize marks missing as -1, the last slot
    for j, u in enumerate(uniques):
        key, is_num = _cell_key(u)
        if key is None:
            blank[j] = True
            continue
        if key in (exact_num if is_num else exact) or (prefixes and key.startswith(prefixes)):
            hit[j] = True
    return hit[codes], blank[codes]


def _is_text(series):
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def _blank(series):
    miss = series.isna().to_numpy(dtype=bool)
    if _is_text(series):
        st = series.astype("string").str.strip()
        miss = miss | (st == "").fillna(False).to_numpy(dtype=bool)
    return miss


def _numbers(series):
    """coerce to float. returns (values, missing, non_numeric, top_coded) with
    '90+' style values read as their bound and counted as top coded."""
    n = len(series)
    none = np.zeros(n, dtype=bool)
    if pd.api.types.is_numeric_dtype(series) and not _is_text(series):
        vals = series.to_numpy(dtype=float, na_value=np.nan)
        return vals, np.isnan(vals), none, none
    st = series.astype("string").str.strip()
    missing = (st.isna() | (st == "")).fillna(True).to_numpy(dtype=bool)
    vals = pd.to_numeric(st, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    bad = ~missing & np.isnan(vals)
    top = none
    if bad.any():
        bound = st[bad].str.extract(_TOP_CODED, expand=False)
        bound = pd.to_numeric(bound, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        idx = np.flatnonzero(bad)
        ok = ~np.isnan(bound)
        if ok.any():
            vals = vals.copy()
            vals[idx[ok]] = bound[ok]
            top = np.zeros(n, dtype=bool)
            top[idx[ok]] = True
            bad = bad & ~top
    return vals, missing, bad, top


def _examples(series, mask):
    if not mask.any():
        return []
    vals = series[mask].astype(str).str.strip()
    return [str(v) for v in pd.unique(vals)[:_EXAMPLES]]


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def _column_problem(df, missing):
    """unknown column names -> one message, with a hint for near misses."""
    names = [str(c) for c in df.columns]
    lower = {n.lower(): n for n in names}
    bits = []
    for c in missing:
        s = str(c)
        hint = lower.get(s.lower())
        if hint is None:
            close = difflib.get_close_matches(s, names, n=1, cutoff=0.8)
            hint = close[0] if close else None
        bits.append(f"'{s}' (did you mean '{hint}'?)" if hint else f"'{s}'")
    word = "column" if len(missing) == 1 else "columns"
    return f"unknown {word} " + ", ".join(bits)


def _columns_of(df, f, errs):
    has_col = f.get("column") not in (None, "")
    has_cols = f.get("columns") not in (None, "", [], ())
    if has_col and has_cols:
        errs.append("give either 'column' or 'columns', not both")
        return []
    if has_col:
        cols = [f["column"]]
    elif has_cols:
        c = f["columns"]
        if isinstance(c, str):
            cols = [c]
        elif isinstance(c, (list, tuple)):
            cols = list(c)
        else:
            errs.append("'columns' must be a list of column names")
            return []
    else:
        errs.append("no column given, set 'column' or 'columns'")
        return []
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    unknown = []
    for c in out:
        try:
            known = c in df.columns
        except TypeError:
            known = False
        if not known:
            unknown.append(c)
    if unknown:
        errs.append(_column_problem(df, unknown))
    doubled = set(df.columns[df.columns.duplicated()])
    for c in out:
        if c not in unknown and c in doubled:
            errs.append(f"column '{c}' appears more than once in the data, rename one copy")
    return out


def _one_value(values):
    """scalar or one-item list -> (value, ok)."""
    if isinstance(values, (list, tuple)):
        return (values[0], True) if len(values) == 1 else (None, False)
    if values is None:
        return None, False
    return values, True


def _check_filters(df, filters):
    if filters is None:
        return []
    if not isinstance(filters, (list, tuple)):
        raise CohortFilterError(["filters must be a list of filter dicts"])
    specs, problems = [], []
    for i, f in enumerate(filters, start=1):
        if not isinstance(f, dict):
            problems.append(f"filter {i}: expected a dict, got {type(f).__name__}")
            continue
        label = f.get("label")
        named = isinstance(label, str) and label.strip()
        where = f"filter {i} ({label.strip()})" if named else f"filter {i}"
        errs = []
        if label is not None and not isinstance(label, str):
            errs.append("'label' must be text")

        op = f.get("op")
        op_n = op.strip().lower() if isinstance(op, str) else None
        if op_n not in OPS:
            if op in (None, ""):
                errs.append("no op given, use one of " + ", ".join(OPS))
            else:
                errs.append(f"unknown op '{op}', use one of " + ", ".join(OPS))

        cols = _columns_of(df, f, errs)
        if op_n in _SINGLE_COLUMN_OPS and len(cols) > 1:
            errs.append(f"'{op_n}' works on one column, got {len(cols)}")

        spec = {"op": op_n, "columns": cols, "label": label.strip() if named else None}
        values = f.get("values")
        if op_n in _CODE_OPS:
            codes, perrs = _parse_values(values) if values is not None else ([], [])
            errs.extend(perrs)
            if not codes and not perrs:
                errs.append(f"'{op_n}' needs at least one code in values")
            spec["codes"] = codes
            spec["values"] = _compact(codes)
        elif op_n == "between":
            pair = list(values) if isinstance(values, (list, tuple)) else []
            nums = [_to_float(v) for v in pair]
            if len(pair) != 2 or None in nums:
                errs.append(f"'between' needs two numbers in values, [low, high], got {values!r}")
            elif nums[0] > nums[1]:
                errs.append(f"'between' low value {_num_text(nums[0])} is above "
                            f"high value {_num_text(nums[1])}")
            else:
                spec["values"] = [_plain_num(nums[0]), _plain_num(nums[1])]
        elif op_n in ("gte", "lte"):
            v, ok = _one_value(values)
            x = _to_float(v) if ok else None
            if x is None:
                errs.append(f"'{op_n}' needs one number in values, got {values!r}")
            else:
                spec["values"] = [_plain_num(x)]
        elif op_n in ("eq", "neq"):
            v, ok = _one_value(values)
            if not ok or v is None or (isinstance(v, str) and not v.strip()):
                errs.append(f"'{op_n}' needs one value in values, got {values!r}")
            elif not isinstance(v, (str, int, float, np.integer, np.floating)) or isinstance(v, bool):
                errs.append(f"'{op_n}' value must be text or a number, got {v!r}")
            else:
                x = _to_float(v)
                if x is not None:
                    spec["number"] = x
                    spec["values"] = [_plain_num(x) if not isinstance(v, str) else v.strip()]
                elif isinstance(v, str):
                    spec["text"] = v.strip()
                    spec["values"] = [v.strip()]
                else:
                    errs.append(f"'{op_n}' value must be a finite number, got {v!r}")
        elif op_n == "not_missing":
            spec["values"] = []

        if errs:
            problems.extend(f"{where}: {e}" for e in errs)
        else:
            specs.append(spec)
    if problems:
        raise CohortFilterError(problems)
    return specs


# ----------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------
def _columns_text(cols):
    names = [str(c) for c in cols]
    if len(names) <= _LABEL_COLUMNS:
        return ", ".join(names)
    return f"{names[0]}, {names[1]} ... {names[-1]} ({len(names)} columns)"


def _codes_text(tokens):
    if len(tokens) > _LABEL_CODES:
        return ", ".join(tokens[:_LABEL_CODES]) + f" and {len(tokens) - _LABEL_CODES} more"
    return ", ".join(tokens)


def _auto_label(spec):
    op, cols, vals = spec["op"], spec["columns"], spec["values"]
    col = str(cols[0])
    if op in _CODE_OPS:
        verb = "Included" if op == "in" else "Excluded"
        codes = _codes_text(vals)
        if len(cols) == 1:
            return f"{verb} {col} {codes}"
        return f"{verb} {codes} in any of {_columns_text(cols)}"
    if op == "between":
        return f"Excluded {col} < {_num_text(vals[0])} or > {_num_text(vals[1])}"
    if op == "gte":
        return f"Excluded {col} < {_num_text(vals[0])}"
    if op == "lte":
        return f"Excluded {col} > {_num_text(vals[0])}"
    if op in ("eq", "neq"):
        shown = vals[0] if isinstance(vals[0], str) else _num_text(vals[0])
        verb = "Included" if op == "eq" else "Excluded"
        return f"{verb} {col} = {shown}"
    if len(cols) == 1:
        return f"Excluded missing {col}"
    return f"Excluded missing any of {_columns_text(cols)}"


# ----------------------------------------------------------------------------
# filtering
# ----------------------------------------------------------------------------
def _evaluate(df, spec, current):
    """one filter over the whole frame -> (keep mask, counts among current rows)."""
    op, cols = spec["op"], spec["columns"]
    n = len(df)
    info = {"n_missing": 0, "n_non_numeric": 0, "non_numeric_examples": [],
            "n_top_coded": 0, "top_coded_examples": []}

    if op in _CODE_OPS:
        exact, exact_num, prefixes = _match_sets(spec["codes"])
        any_hit = np.zeros(n, dtype=bool)
        all_blank = np.ones(n, dtype=bool)
        for c in cols:
            hit, blank = _code_hits(df[c], exact, exact_num, prefixes)
            any_hit |= hit
            all_blank &= blank
        info["n_missing"] = int((all_blank & current).sum())
        return (any_hit if op == "in" else ~any_hit), info

    if op == "not_missing":
        any_blank = np.zeros(n, dtype=bool)
        for c in cols:
            any_blank |= _blank(df[c])
        info["n_missing"] = int((any_blank & current).sum())
        return ~any_blank, info

    s = df[cols[0]]
    if op in ("eq", "neq") and "text" in spec:
        st = s.astype("string").str.strip()
        missing = (st.isna() | (st == "")).fillna(True).to_numpy(dtype=bool)
        same = (st.str.casefold() == spec["text"].casefold()).fillna(False).to_numpy(dtype=bool)
        info["n_missing"] = int((missing & current).sum())
        return (same if op == "eq" else ~same), info

    vals, missing, bad, top = _numbers(s)
    info["n_missing"] = int((missing & current).sum())
    info["n_non_numeric"] = int((bad & current).sum())
    info["non_numeric_examples"] = _examples(s, bad & current)
    info["n_top_coded"] = int((top & current).sum())
    info["top_coded_examples"] = _examples(s, top & current)
    v = spec["values"]
    with np.errstate(invalid="ignore"):
        if op == "gte":
            keep = vals >= v[0]
        elif op == "lte":
            keep = vals <= v[0]
        elif op == "between":
            keep = (vals >= v[0]) & (vals <= v[1])
        else:
            same = vals == spec["number"]
            keep = same if op == "eq" else ~same
    return keep, info


def apply_filters(df, filters):
    """Run an ordered list of cohort filters. Returns (filtered_df, flow).

    filters: list of dicts, applied in order, each on the rows left by the
    previous one:
        {"column": "AGE"} or {"columns": ["CPT", "OTHERCPT1", ...]},
        "op":     in | not_in | between | gte | lte | eq | neq | not_missing,
        "values": see below,
        "label":  optional text for the flow diagram (auto label otherwise).

    op meanings (every op keeps some rows and drops the rest):
        in          keep rows where ANY listed column holds one of the codes.
                    values is pasted text (parsed like parse_code_list) or a
                    list with one code or category value per item; ranges and
                    trailing * wildcards work in both.
        not_in      drop rows where ANY listed column holds one of the codes,
                    blank cells never match so they stay.
        between     keep low <= value <= high, values [low, high], one column.
        gte / lte   keep value >= x / value <= x, values [x], one column.
        eq          keep value == x, values [x], one column. numbers compare
                    numerically (1 == 1.0 == '1'), text case-insensitively.
        neq         drop value == x, everything else (blank cells too) stays.
        not_missing keep rows with a value in EVERY listed column (blank text
                    counts as missing).
    numeric ops (between, gte, lte, and eq/neq with a number) coerce the
    column to numbers. Missing and non-numeric cells never match, so between,
    gte, lte and eq drop them; both are counted. Top-coded text such as '90+'
    is read as its bound (90) and counted.

    Every filter is checked before anything runs; CohortFilterError (a
    ValueError) lists all the problems at once. The input df is never
    modified; the returned frame is a new copy with the original index kept.

    flow: one dict per filter, in order:
        {"step": 1, "label": str, "kind": "include" | "exclude",
         "columns": [col, ...], "op": str,
         "values": [...],   # codes compacted ('27130-27132'), numbers, or []
         "n_before": int, "n_after": int, "n_removed": int,
         "n_missing": int,  # rows (of n_before) with no value; for several
                            # columns: all blank (in / not_in), any blank
                            # (not_missing)
         "n_non_numeric": int, "non_numeric_examples": [str, ...],
         "n_top_coded": int, "top_coded_examples": [str, ...]}
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("apply_filters needs a pandas DataFrame")
    specs = _check_filters(df, filters)
    current = np.ones(len(df), dtype=bool)
    flow = []
    for i, spec in enumerate(specs, start=1):
        keep, info = _evaluate(df, spec, current)
        n_before = int(current.sum())
        current = current & keep
        n_after = int(current.sum())
        step = {
            "step": i,
            "label": spec["label"] or _auto_label(spec),
            "kind": "include" if spec["op"] in _INCLUDE_OPS else "exclude",
            "columns": list(spec["columns"]),
            "op": spec["op"],
            "values": list(spec["values"]),
            "n_before": n_before,
            "n_after": n_after,
            "n_removed": n_before - n_after,
        }
        step.update(info)
        flow.append(step)
    # take() rather than a boolean .loc so the result is not flagged as a slice
    return df.take(np.flatnonzero(current)), flow


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def _step_note(step):
    op = step.get("op")
    dropped = []
    if op in _DROPS_MISSING and step.get("n_missing"):
        dropped.append(f"{step['n_missing']:,} missing")
    if op in _DROPS_NON_NUMERIC and step.get("n_non_numeric"):
        ex = step.get("non_numeric_examples") or []
        eg = f", e.g. {_quote_list(ex[:3])}" if ex else ""
        dropped.append(f"{step['n_non_numeric']:,} not numeric{eg}")
    notes = []
    if dropped:
        notes.append("removed rows include " + " and ".join(dropped))
    if step.get("n_top_coded"):
        ex = step.get("top_coded_examples") or []
        if ex:
            bound = re.match(_TOP_CODED, ex[0])
            read = _num_text(bound.group(1)) if bound else "the number"
            notes.append(f"{_count(step['n_top_coded'], 'top-coded value')} such as '{ex[0]}' read as {read}")
        else:
            notes.append(f"{_count(step['n_top_coded'], 'top-coded value')} read as their bound")
    return "; ".join(notes)


def summarize_flow(flow, n_start=None, start_label="Records in file"):
    """Flow from apply_filters -> list of plain text lines, e.g.
        Records in file: 120,345
        Included CPT 27130, 27447: 41,220
        Excluded AGE < 18: 312 removed, 40,908 remaining (removed rows include 15 missing)
        Final cohort: 40,908
    include steps (in, eq) show the count kept, exclude steps the count removed
    and the count left. n_start defaults to the first step's n_before.
    """
    flow = list(flow or [])
    if n_start is None:
        n_start = flow[0]["n_before"] if flow else 0
    lines = [f"{start_label}: {int(n_start):,}"]
    for step in flow:
        kind = step.get("kind") or ("include" if step.get("op") in _INCLUDE_OPS else "exclude")
        if kind == "include":
            line = f"{step['label']}: {int(step['n_after']):,}"
        else:
            line = (f"{step['label']}: {int(step['n_removed']):,} removed, "
                    f"{int(step['n_after']):,} remaining")
        note = _step_note(step)
        if note:
            line += f" ({note})"
        lines.append(line)
    n_final = flow[-1]["n_after"] if flow else n_start
    lines.append(f"Final cohort: {int(n_final):,}")
    return lines
