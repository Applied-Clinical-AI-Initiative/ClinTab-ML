"""Render paper/v2/manuscript_v2.txt and cover_letter.txt to Word.

Every {{key}} is looked up in facts.json; an unknown key stops the build.
Bare numbers written directly into the source (outside {{...}}) are listed
at the end so they can be checked by eye as design constants, not results.

Outputs, all Times New Roman, black:
  paper/v2/Manuscript_v2.docx      double-spaced, line-numbered
  paper/v2/Supplementary_v2.docx   Tables S1 to S3
  paper/v2/cover_letter.docx       single page

Run: python paper/analysis/build_docx.py
"""
import json
import re
import sys

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from common import FIGURES, OUT, TABLES
from references import REFS

FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)
FACTS = {}
CITE_ORDER = []
LITERALS = []

# ----------------------------------------------------------------- text ----


def render_facts(text, where=""):
    def sub(m):
        key, _, filt = m.group(1).partition("|")
        if key not in FACTS:
            sys.exit(f"Unknown fact key '{key}' in: {where[:80]}")
        # running text drops the explict plus sign; tables keep it.
        v = FACTS[key].replace("+", "")
        if filt and filt != "nosign":
            sys.exit(f"Unknown filter '{filt}'")
        return v
    bare = re.sub(r"\{\{[^}]+\}\}|\[cite:[^\]]+\]|\[PLACEHOLDER[^\]]*\]|\^[^^]+\^", " ", text)
    for m in re.finditer(r"(?<![A-Za-z])\d[\d,\.]*", bare):
        LITERALS.append((m.group(0), bare[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")))
    return re.sub(r"\{\{([^}]+)\}\}", sub, text)


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}-{nums[j]}"))
        i = j + 1
    return ",".join(out)


def render_cites(text):
    def sub(m):
        nums = []
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in REFS:
                sys.exit(f"Unknown reference key '{k}'")
            if k not in CITE_ORDER:
                CITE_ORDER.append(k)
            nums.append(CITE_ORDER.index(k) + 1)
        return f"^{compress(nums)}^"
    return re.sub(r"\[cite:([^\]]+)\]", sub, text)


def set_run(run, size=12, bold=None, italic=None, sup=False):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.superscript = sup


def add_inline(par, text, size=12, bold=False, italic=False):
    """Write text into par, handling ^superscript^ and [PLACEHOLDER] markup."""
    for part in re.split(r"(\^[^^]+\^|\[PLACEHOLDER[^\]]*\])", text):
        if not part:
            continue
        if part.startswith("^") and part.endswith("^"):
            set_run(par.add_run(part[1:-1]), size, bold, italic, sup=True)
        elif part.startswith("[PLACEHOLDER"):
            set_run(par.add_run(part), size, True, italic)
        else:
            set_run(par.add_run(part), size, bold, italic)


def text(t, where=""):
    return render_cites(render_facts(t, where))

# ------------------------------------------------------------- document ----


def base_document(spacing=WD_LINE_SPACING.DOUBLE):
    doc = Document()
    for s in doc.sections:
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for side in ("left", "right", "top", "bottom"):
            setattr(s, f"{side}_margin", Inches(1))
    for name in ("Normal", "Heading 1", "Heading 2", "Title"):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.color.rgb = BLACK
        rpr = st.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            rfonts.attrib.pop(qn(a), None)
        st.font.size = Pt(12)
        pf = st.paragraph_format
        pf.line_spacing_rule = spacing
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Heading 2"].font.bold = True
    doc.styles["Heading 2"].font.italic = True
    for name in ("Heading 1", "Heading 2"):
        doc.styles[name].paragraph_format.space_before = Pt(12)
        doc.styles[name].paragraph_format.keep_with_next = True
    return doc


def heading(doc, t, level):
    p = doc.add_paragraph(style=f"Heading {level}")
    add_inline(p, t, bold=True, italic=(level == 2))
    return p


def para(doc, t, first_indent=True, align=None, size=12, bold=False, italic=False):
    p = doc.add_paragraph()
    if first_indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    if align:
        p.alignment = align
    add_inline(p, t, size=size, bold=bold, italic=italic)
    return p


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run(run, 12)
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)


def line_numbers(section):
    sp = section._sectPr
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    sp.append(ln)


def new_section(doc, landscape=False):
    s = doc.add_section(WD_SECTION.NEW_PAGE)
    if landscape:
        s.orientation = WD_ORIENT.LANDSCAPE
        s.page_width, s.page_height = Inches(11), Inches(8.5)
    else:
        s.orientation = WD_ORIENT.PORTRAIT
        s.page_width, s.page_height = Inches(8.5), Inches(11)
    return s

# --------------------------------------------------------------- tables ----


def cell_borders(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        val = edges.get(edge)
        el.set(qn("w:val"), "single" if val else "nil")
        if val:
            el.set(qn("w:sz"), str(val))
            el.set(qn("w:color"), "000000")
        b.append(el)


def write_cell(cell, value, bold=False, size=9, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if align:
        p.alignment = align
    add_inline(p, str(value), size=size, bold=bold)


def add_table(doc, df, widths, group_col=None, size=9, blank_repeats=None):
    """Three-rule table. group_col: a column whose value, when it changes,
    becomes a full-width spanning row instead of a column."""
    cols = [c for c in df.columns if c != group_col]
    rows = []
    last = None
    for _, r in df.iterrows():
        if group_col and r[group_col] != last:
            rows.append(("group", r[group_col]))
            last = r[group_col]
        rows.append(("row", [r[c] for c in cols]))
    t = doc.add_table(rows=1 + len(rows), cols=len(cols))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, c in enumerate(cols):
        write_cell(t.cell(0, j), c, bold=True, size=size)
    prev = {}
    for i, (kind, val) in enumerate(rows, start=1):
        if kind == "group":
            m = t.cell(i, 0).merge(t.cell(i, len(cols) - 1))
            write_cell(m, val, bold=True, size=size)
        else:
            for j, v in enumerate(val):
                col = cols[j]
                shown = v
                if blank_repeats and col in blank_repeats and prev.get(col) == v:
                    shown = ""
                prev[col] = v
                write_cell(t.cell(i, j), "" if pd.isna(shown) else shown, size=size)
    n = len(t.rows)
    for i, row in enumerate(t.rows):
        for j, cell in enumerate(row.cells):
            cell.width = Inches(widths[j]) if j < len(widths) else cell.width
            top = 8 if i == 0 else None
            bottom = 4 if i == 0 else (8 if i == n - 1 else None)
            cell_borders(cell, top=top, bottom=bottom)
    return t


def caption(doc, t, size=12):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_after = Pt(6)
    head, _, rest = t.partition(". ")
    add_inline(p, head + ". ", size=size, bold=True)
    add_inline(p, rest, size=size)


def footnote(doc, t, size=9):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_before = Pt(4)
    add_inline(p, t, size=size)

# ------------------------------------------------------------- parse src ----


def parse(path):
    meta, blocks = {}, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("@"):
            k, _, v = line[1:].partition(" ")
            meta.setdefault(k, []).append(v)
        elif line.startswith("## "):
            blocks.append(("h2", line[3:]))
        elif line.startswith("# "):
            blocks.append(("h1", line[2:]))
        else:
            blocks.append(("p", line))
    return meta, blocks


def words(s):
    s = re.sub(r"\^[^^]+\^", "", s)
    return len(s.split())

# ----------------------------------------------------- table definitions ----


def load_tables():
    t1 = pd.read_csv(TABLES / "table1_cohort.csv", dtype=str).fillna("")
    t1.columns = ["Characteristic", "Level or summary", "Overall", "Missing, n", "Missing, %"]
    t2 = pd.read_csv(TABLES / "table2_optimism_auroc.csv", dtype=str)
    t2["Group"] = "Arm " + t2.Arm + ": " + t2.Protocol
    t2 = t2.drop(columns=["Arm", "Protocol"])
    t3 = pd.read_csv(TABLES / "table3_equivalence.csv", dtype=str).fillna("")
    t4 = pd.read_csv(TABLES / "table4_tripod.csv", dtype=str).fillna("")
    return t1, t2, t3, t4


def tables_section(doc):
    t1, t2, t3, t4 = load_tables()
    f = FACTS
    new_section(doc)
    caption(doc, "Table 1. Characteristics of the analytic cohort, with per-variable missingness.")
    add_table(doc, t1, [1.9, 1.9, 1.3, 0.8, 0.6], size=8.5, blank_repeats=["Characteristic"])
    footnote(doc, text(
        f"Values are n (%) unless stated. SD, standard deviation; IQR, interquartile range. N = {{{{cohort_n}}}}. "
        "Race and ethnicity categories labelled Unknown are coded as categories in the PUF and are not counted as missing. "
        "ASA class, functional status and diabetes are shown with the numeric codes exported by the cohort script. "
        "[PLACEHOLDER: code labels for ASA class, functional status and diabetes.]"))

    new_section(doc, landscape=True)
    caption(doc, text("Table 2. AUROC optimism (reported minus held-out AUROC) by development protocol, model and cohort size."))
    add_table(doc, t2, [1.8, 1.7, 1.7, 1.7, 1.7], group_col="Group", size=9)
    footnote(doc, text(
        "Values are mean (95% CI) across {{ex1_seeds}} seeds. All models (mean): optimism averaged over the three "
        "models within each seed before computing the interval. Positive values mean the protocol would report higher "
        "discrimination than the fitted model achieved on held-out patients. Results for AUPRC, Brier score, calibration "
        "slope and Hosmer-Lemeshow P value are in Supplementary Table S1."))

    caption(doc, text("Table 3. Numerical equivalence of ClinTAB-ML components with reference implementations."))
    add_table(doc, t3, [1.9, 2.1, 1.7, 0.9, 0.9, 1.0, 0.8], size=8.5, blank_repeats=["Component", "Reference implementation"])
    footnote(doc, text(
        "Tolerance is {{ex2_tol}}, half a unit in the fourth decimal place to which ClinTAB-ML reports. "
        "The restricted cubic spline row compares AIC and the maximum absolute difference between fitted log-odds curves "
        "on a {{ex2_rcs_grid}}-point grid. HR, hazard ratio; OR, odds ratio; CI, confidence interval. "
        "[PLACEHOLDER: describe the test datasets used for each component.]"))

    new_section(doc, landscape=True)
    caption(doc, text("Table 4. TRIPOD+AI items: automated report status and recoverability from artifacts alone."))
    add_table(doc, t4, [1.6, 3.7, 1.3, 1.3, 1.3], size=8.5, blank_repeats=["Section"])
    footnote(doc, text(
        "Automated report: status assigned by the ClinTAB-ML TRIPOD+AI module from the full-cohort session log. "
        "Recoverable columns: scored by two authors from artifacts alone, without access to the analyst's memory. "
        "The automated report covers a practical subset of TRIPOD+AI and does not replace the published checklist."))


FIG_LEGENDS = [
    ("figure1_architecture.png", 6.5,
     "Figure 1. ClinTAB-ML architecture and enforcement points. The analysis path (left) runs entirely on the "
     "researcher's machine. Black markers E1 to E4 identify where each convention is enforced and which Experiment 1 "
     "comparator arm it prevents. Every step appends an entry to the analysis log (right), from which the methods "
     "paragraph and TRIPOD+AI coverage report are generated deterministically."),
    ("figure2_flow.png", 4.3,
     "Figure 2. Cohort inclusion and partitioning. THA, total hip arthroplasty; TKA, total knee arthroplasty; "
     "CPT, Current Procedural Terminology."),
    ("figure3_optimism.png", 6.5,
     "Figure 3. AUROC optimism by cohort size for each development protocol. Optimism is the AUROC the protocol "
     "would report minus the AUROC of the same fitted pipeline on held-out patients. Points are means across "
     "{{ex1_seeds}} seeds and bars are 95% confidence intervals; models are distinguished by marker and line style, "
     "and are offset horizontally for legibility. The y axis is shared across panels."),
    ("figure4_scaling.png", 6.5,
     "Figure 4. Runtime (A) and peak resident memory (B) of the full analysis path on an {{hw_machine}} laptop with "
     "{{hw_ram}} of memory. Points are medians over {{ex3_repeats}} repeats and bars span the minimum to maximum. "
     "Both axes are logarithmic. Methods text and TRIPOD+AI report timings were not recorded at n = 100,000."),
    ("figure5_roc_calibration.png", 6.5,
     "Figure 5. Discrimination and calibration of the three models on the full-cohort test partition "
     "(n = {{split_test}}). (A) Receiver operating characteristic and precision-recall curves. (B) Calibration by "
     "decile of predicted risk. Models were trained with SMOTE on the training fold, the application default. "
     "[PLACEHOLDER: interim colour figure exported by the application; to be redrawn in black and white from "
     "test-set predictions.]"),
]


def figures_section(doc):
    new_section(doc)
    heading(doc, "Figure Legends", 1)
    for _, _, leg in FIG_LEGENDS:
        head, _, rest = text(leg).partition(". ")
        p = doc.add_paragraph()
        add_inline(p, head + ". ", bold=True)
        add_inline(p, rest)
    for fn, w, leg in FIG_LEGENDS:
        page_break(doc)
        doc.add_picture(str(FIGURES / fn), width=Inches(w))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para(doc, text(leg).partition(". ")[0] + ".", first_indent=False, bold=True)

# ------------------------------------------------------------ manuscript ----


def build_manuscript():
    meta, blocks = parse(OUT / "manuscript_v2.txt")
    doc = base_document()
    sec0 = doc.sections[0]

    # title page.
    p = para(doc, text(meta["title"][0]), first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    para(doc, "", first_indent=False)
    para(doc, text(meta["authors"][0]), first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "", first_indent=False)
    for a in meta["affil"]:
        para(doc, text(a), first_indent=False)
    para(doc, "", first_indent=False)
    para(doc, "Corresponding author: " + text(meta["corresponding"][0]), first_indent=False)
    para(doc, "Running title: " + meta["running"][0], first_indent=False)
    para(doc, "Keywords: " + meta["keywords"][0], first_indent=False)
    counts_par = para(doc, "", first_indent=False)

    # body.
    main_words, abstract_words, lay_words = 0, 0, 0
    mode = None
    in_main = False
    pending_h2 = None
    back_matter = {"Acknowledgments", "Author Contributions", "Funding", "Conflicts of Interest",
                   "Data Availability", "ACS NSQIP Disclaimer", "Use of Generative AI"}
    for kind, raw in blocks:
        if kind == "h1":
            mode = raw
            if raw in ("Abstract", "Background and Significance"):
                page_break(doc)
            in_main = raw not in ("Abstract", "Lay Summary") and raw not in back_matter
            heading(doc, raw, 1)
        elif kind == "h2":
            if mode == "Abstract":
                pending_h2 = raw
            else:
                heading(doc, raw, 2)
        else:
            t = text(raw, raw)
            if mode == "Abstract":
                p = doc.add_paragraph()
                add_inline(p, pending_h2 + ": ", bold=True)
                add_inline(p, t)
                abstract_words += words(t)
                pending_h2 = None
            else:
                para(doc, t)
                if in_main:
                    main_words += words(t)
                if mode == "Lay Summary":
                    lay_words += words(t)

    # references.
    page_break(doc)
    heading(doc, "References", 1)
    for i, k in enumerate(CITE_ORDER, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        add_inline(p, f"{i}.\t{REFS[k]}")
    unused = [k for k in REFS if k not in CITE_ORDER]

    tables_section(doc)
    figures_section(doc)

    add_inline(counts_par, f"Word count: abstract {abstract_words}; main text {main_words}. "
               f"Tables: 4. Figures: {len(FIG_LEGENDS)}. Supplementary tables: 3.")
    for s in doc.sections:
        line_numbers(s)
        add_page_numbers(s)
        s.footer.is_linked_to_previous = True
    doc.save(OUT / "Manuscript_v2.docx")
    return abstract_words, main_words, lay_words, unused

# ----------------------------------------------------------- supplement ----


def build_supplement():
    doc = base_document(WD_LINE_SPACING.SINGLE)
    s = doc.sections[0]
    s.orientation = WD_ORIENT.LANDSCAPE
    s.page_width, s.page_height = Inches(11), Inches(8.5)
    para(doc, "Supplementary material: " + text(parse(OUT / "manuscript_v2.txt")[0]["title"][0]),
         first_indent=False, bold=True)
    s1 = pd.read_csv(TABLES / "tableS1_optimism_all_metrics.csv")
    s1["Model"] = s1.model.map({"LogisticRegression": "Logistic regression", "RandomForest": "Random forest",
                                "GradientBoosting": "Gradient boosting"})
    s1["Group"] = s1.metric.map({"AUROC": "AUROC", "AUPRC": "AUPRC", "Brier": "Brier score",
                                 "calibration_slope": "Calibration slope",
                                 "HL_pvalue": "Hosmer-Lemeshow P value"}) + ", Arm " + s1.arm
    s1["n"] = s1.n.map(lambda v: f"{v:,}")
    for c in ("reported", "honest"):
        s1[c.capitalize()] = s1[c].map(lambda v: f"{v:.3f}")
    s1["Optimism, mean (95% CI)"] = s1.apply(lambda r: f"{r['mean']:+.3f} ({r.lo:+.3f} to {r.hi:+.3f})".replace("-", "\u2212"), axis=1)
    s1 = s1[["Group", "Model", "n", "Reported", "Honest", "Optimism, mean (95% CI)"]]
    caption(doc, "Supplementary Table S1. Optimism for every metric, protocol, model and cohort size.")
    add_table(doc, s1, [2.0, 1.2, 1.0, 1.0, 2.4], group_col="Group", size=8, blank_repeats=["Model"])
    footnote(doc, text("Mean across {{ex1_seeds}} seeds with 95% t-interval. Arm A, preprocessing fitted before "
                       "splitting; Arm B, SMOTE before splitting; Arm C, tuning scored on the test partition; "
                       "Arm D, ClinTAB-ML default pipeline."))
    doc.add_section(WD_SECTION.NEW_PAGE)
    s2 = pd.read_csv(TABLES / "tableS2_methods_clauses.csv", dtype=str)
    caption(doc, "Supplementary Table S2. Generated methods paragraph, by clause, with author scoring.")
    add_table(doc, s2, [0.6, 5.6, 0.9, 0.9, 1.0], size=8.5)
    footnote(doc, "Text reproduced verbatim from the ClinTAB-ML methods generator for the full-cohort session. "
                  "Scores: accurate, inaccurate or missing, against what the session log shows was done.")
    doc.add_section(WD_SECTION.NEW_PAGE)
    s3 = pd.read_csv(TABLES / "tableS3_y1_models.csv", dtype=str)
    caption(doc, "Supplementary Table S3. Full-cohort model performance for 30-day unplanned readmission.")
    add_table(doc, s3, [1.5, 1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.8, 0.8, 0.8], size=9, blank_repeats=["Model"])
    footnote(doc, "Sensitivity, specificity, PPV and NPV at a probability threshold of 0.5. PPV, positive predictive "
                  "value; NPV, negative predictive value.")
    for s in doc.sections:
        add_page_numbers(s)
    doc.save(OUT / "Supplementary_v2.docx")

# --------------------------------------------------------- cover letter ----


def build_cover_letter():
    meta, blocks = parse(OUT / "cover_letter.txt")
    doc = base_document(WD_LINE_SPACING.SINGLE)
    for st in ("Normal",):
        doc.styles[st].font.size = Pt(11)
    s = doc.sections[0]
    s.top_margin = s.bottom_margin = Inches(0.8)
    n_words = 0
    for kind, raw in blocks:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        t = text(raw, raw)
        add_inline(p, t, size=11)
        n_words += words(t)
    doc.save(OUT / "cover_letter.docx")
    return n_words


def main():
    FACTS.clear()
    FACTS.update(json.load(open(OUT / "facts.json")))
    CITE_ORDER.clear()
    LITERALS.clear()
    aw, mw, lw, unused = build_manuscript()
    build_supplement()
    cw = build_cover_letter() if (OUT / "cover_letter.txt").exists() else 0
    print(f"Manuscript_v2.docx: abstract {aw} words, main text {mw} words, lay summary {lw} words")
    print(f"References cited: {len(CITE_ORDER)}; unused: {unused}")
    print(f"cover_letter.docx: {cw} words")
    print("\nBare numbers in source (check these are design constants, not results):")
    seen = set()
    for num, ctx in LITERALS:
        if (num, ctx) not in seen:
            seen.add((num, ctx))
            print(f"  {num:>10}  ...{ctx}...")


if __name__ == "__main__":
    main()
