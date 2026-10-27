"""Rebuild every manuscript artifact from the results folder, in order.

    python paper/analysis/run_all.py

1. tables.py      Tables 1 to 4 and S1 to S3 as CSV    -> paper/v2/tables/
2. facts.py       facts sheet (facts.md, facts.json)   -> paper/v2/
3. figures.py     Figures 1 to 5, PDF/SVG/PNG 300 dpi  -> paper/v2/figures/
4. build_docx.py  Manuscript, supplement, cover letter -> paper/v2/

Inputs are the aggregate files in paper/analysis/results/. Nothing here
reads the NSQIP PUF. Files not yet in the results folder (inclusion counts,
Experiment 4 scores, test-set predictions) render as [PLACEHOLDER] and are
picked up automatically once added at the paths named in common.py.
"""
import tables
import facts
import figures
import build_docx

if __name__ == "__main__":
    tables.main()
    facts.main()
    figures.main()
    build_docx.main()
