# Paper analysis

Scripts behind the ClinTAB-ML manuscript (validation experiments 1 to 4).

## Data use

The experiments used the ACS NSQIP Participant Use Data Files for 2021 to 2023. The data use agreement does not allow the data to be shared, so this repository holds no NSQIP data and no results computed from it. Only the scripts are here. Their inputs and outputs stay on the authors' machines in `paper/analysis/results/` and `paper/v2/`, which `.gitignore` excludes, and a CI check fails if anything like a data file or a results file is ever committed.

To reproduce the analysis you need your own PUF access from the ACS. The cohort and the experiment runs produce the files below; put them in `paper/analysis/results/` and run the scripts.

## Contents

| File | What it is |
|------|------------|
| `common.py` | Shared paths, loaders and plotting style. Lists every input file the scripts read. |
| `tables.py` | Manuscript Tables 1 to 4 and S1 to S3 as CSV, written to `paper/v2/tables/`. |
| `facts.py` | Facts sheet (`paper/v2/facts.md` and `facts.json`). Every number the manuscript states comes from here. |
| `figures.py` | Figures 1 to 5, written to `paper/v2/figures/`. |
| `references.py` | Reference list for the manuscript. |
| `build_docx.py` | Builds the manuscript, supplement and cover letter in `paper/v2/`. |
| `run_all.py` | Runs the four scripts above in order. |

Inputs expected in `results/` (not in the repository):

| File | From |
|------|------|
| `optimism_results(EX1).csv` | Experiment 1, one row per arm, n, seed, model and metric |
| `equivalence_table(ex2).csv` | Experiment 2, ClinTAB-ML against the R reference implementations |
| `scaling_combined (ex3).csv` | Experiment 3, runtime and peak memory |
| `hardware (ex2).txt` | Machine and library versions |
| `section7_outputs/` | Experiment 4 cohort table, model comparison tables and figures |
| `section7_outputs/inclusion_flow.csv` | Figure 2 counts: columns `step,n`, one row per step in order, raw pooled PUF first and the analytic cohort last |
| `experiment4/` | Methods paragraph and TRIPOD+AI report generated for the full cohort |

## Rebuilding

```bash
pip install -r paper/analysis/requirements.txt
python paper/analysis/run_all.py
```

This overwrites everything in `paper/v2/`, including `Manuscript_v2.docx`, so copy any hand edits out of the Word files first. Missing inputs show up as `[PLACEHOLDER]` in the manuscript.

## Spline results

Before 1.0.0 the restricted cubic spline output showed fitted probabilities labelled as log-odds, and its design matrix was rank deficient. Spline figures and the RCS rows of the Experiment 2 table made with earlier versions have to be regenerated.
