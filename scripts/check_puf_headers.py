"""check_puf_headers.py
Reads ONLY the column names of your raw NSQIP files (never a data row) and
reports which Y2 (complication) and Y4 (death timing) columns exist.

Usage:
    python scripts/check_puf_headers.py /path/tka.csv /path/tha.csv ...
"""
import sys

import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from build_nsqip_cohort import DEATH_FIELDS, MORBIDITY_COMPONENTS  # noqa: E402

HINTS = ("DEATH", "DOD", "MORT", "DTH", "SSI", "INFEC", "DVT", "PULEMB", "PNEUM", "MI",
         "STROKE", "CVA", "ARREST", "SEPSIS", "SHOCK", "BLEED", "RENAL", "REINTUB", "COMP")

for path in sys.argv[1:]:
    cols = list(pd.read_csv(path, nrows=0).columns) if not path.endswith(".sas7bdat") \
        else list(pd.read_sas(path, chunksize=1, encoding="latin1").__next__().columns)
    print(f"\n{path}: {len(cols)} columns")
    print("  Y2 columns found:", [c for c in MORBIDITY_COMPONENTS if c in cols] or "NONE")
    print("  Y4 columns found:", [c for c in DEATH_FIELDS if c in cols] or "NONE")
    other = [c for c in cols if any(h in c.upper() for h in HINTS)
             and c not in MORBIDITY_COMPONENTS and c not in DEATH_FIELDS]
    print("  other possibly relevant column names:", other or "none")
