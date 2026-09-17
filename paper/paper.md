---
title: 'ClinTAB-ML: A local application for clinical machine learning on tabular registry data'
tags:
  - Python
  - clinical prediction models
  - machine learning
  - epidemiology
  - surgical registries
  - reproducibility
authors:
  - name: Dharsan Ravindran
    orcid: 0009-0002-5544-9895
    affiliation: "1, 2"
  - name: Kisaan Ravindran
    orcid: 0009-0005-2716-5408
    affiliation: 3
  - name: Aazad Abbas
    orcid: 0000-0001-7414-1701
    corresponding: true
    affiliation: "2, 4"
affiliations:
  - name: Institute of Biomedical Engineering, University of Toronto, Toronto, Canada
    index: 1
  - name: Sunnybrook Research Institute, Toronto, Canada
    index: 2
  - name: Applied Clinical AI Initiative, Canada
    index: 3
  - name: Division of Orthopaedic Surgery, Department of Surgery, University of Toronto, Toronto, Canada
    index: 4
date: 17 September 2026
bibliography: paper.bib
---
# Summary

ClinTAB-ML is a Python application for building and testing clinical prediction models on tabular data, for example an extract from a surgical or trauma registry or a single-centre cohort. The user uploads a CSV file and, from one web page, can profile the columns, split the data into training, validation and test sets, train and tune up to 15 models, evaluate them with discrimination and calibration plots, fit restricted cubic splines, and run common epidemiological calculations such as odds ratios, number needed to treat, Kaplan and Meier curves and Cox models. The application is a Flask server that runs on the user's own computer, and the same functions can be called from a command-line interface. Each analysis step is saved to a log, and from that log the software writes a draft methods paragraph and a checklist based on the TRIPOD+AI reporting guideline [@collins2024tripodai].

# Statement of need

Surgical and trauma registries such as the American College of Surgeons National Surgical Quality Improvement Program and the Trauma Quality Improvement Program are used in a large number of outcome studies [@ko2015acsnsqip; @shafi2009tqip]. When these studies develop a prediction model, the analysis is usually spread over several tools. Data exploration happens in a notebook, model comparison in custom scikit-learn code, odds ratios in an online calculator, and survival analysis in yet another package. Methodological errors are easy to introduce when work moves between these tools. Tuning hyperparameters on the data used to report performance, oversampling before splitting, and fitting preprocessing steps on the full dataset all leak information and overstate model performance, and this has been found in many published machine learning studies [@kapoor2023leakage]. Calibration is also often missing from clinical prediction papers, although a model with poor calibration gives misleading risk estimates for individual patients [@vancalster2019calibration].

Data governance is a further constraint. Online calculators and hosted machine learning platforms need the data to be uploaded to an outside server. Registry extracts are often not fully de-identified and come with data use agreements that do not allow this.

ClinTAB-ML was written for clinical researchers, including residents, students and clinicians with little programming experience, who need to develop and evaluate a model on data that cannot leave their computer. Leakage-free tuning, oversampling of the training fold only, and calibration testing are the defaults, so a user does not need to know about these problems in advance to avoid them.

# State of the field

ClinTAB-ML does not implement its own statistical methods. Models come from scikit-learn [@pedregosa2011sklearn], SMOTE from imbalanced-learn [@lemaitre2017imblearn], survival models from lifelines [@davidsonpilon2019lifelines], splines and generalized linear models from statsmodels [@seabold2010statsmodels], and data handling and figures from pandas, NumPy, SciPy and Matplotlib [@mckinney2010pandas; @harris2020numpy; @virtanen2020scipy; @hunter2007matplotlib]. Using these libraries directly requires the researcher to write the whole analysis. Orange [@demsar2013orange] offers a visual interface to similar methods, but it is aimed at general data mining and has no epidemiological calculators, Hosmer and Lemeshow test, or reporting support for clinical prediction models. Commercial and cloud automated machine learning services train many models with little effort, but the data are processed on remote servers, and model selection is based on predictive performance with no attention to the reporting that clinical journals require.

Adding these features to Orange or to a cloud service would not have met the main requirement, which was that registry data stay on the researcher's machine. We therefore wrote a small application around the existing libraries that puts model development, the epidemiological analyses that usually accompany it, and a record of what was done in one place.

# Software design

The HTTP layer is kept apart from the analysis code. `app.py` starts the Flask application and `routes.py` holds the endpoints. The analysis code is in the `clintab` package and none of it imports Flask. `stats.py` detects column types and summarizes the data, `ml.py` contains the model catalogue, preprocessing, tuning and metrics, `spline.py` fits restricted cubic splines, `epi.py` contains the epidemiological functions, `plots.py` draws the figures and `store.py` handles files on disk. The command-line interface in `clintab_cli.py` calls the same functions as the web application, which means results are identical in both and the analysis modules can be tested without starting a server.

Hyperparameter search by default uses a predefined split. Each candidate is scored on the validation set, and the chosen settings are then refit on the training set alone, so the reported validation metrics come from data the model was not fit on. On small datasets a single validation split can make this selection unstable, and the user can switch to k-fold cross-validation within the training set. Preprocessing is fitted inside each model pipeline, and SMOTE [@chawla2002smote] is applied to the training fold only. For binary outcomes the software reports AUROC, AUPRC, sensitivity, specificity, positive and negative predictive value, F1 and Brier score, and draws ROC, precision-recall and calibration plots. Calibration of a saved model can also be checked with the Hosmer and Lemeshow test [@hosmer1980goodness]. Splines are fitted as logistic generalized linear models with 3, 4 or 5 knots placed at the quantiles recommended by Harrell [@harrell2015regression]. When a two by two table contains a zero cell, the Haldane and Anscombe correction is applied [@haldane1956estimation].

Every session has an analysis log in JSON Lines format. Entries are only ever appended, and each training run, model test, spline and epidemiological calculation is stored with its inputs and main results. `methods_text.py` turns the log into methods text using fixed sentence templates. No language model is involved, and a given log always produces the same text. `tripod_report.py` checks the log against a subset of the TRIPOD+AI reporting items and labels each one as found, not found or needing manual review. Some items, such as the study rationale and limitations, cannot be confirmed from a log, and the report says that it does not replace the published checklist. Figures are drawn with Matplotlib on the server and can be downloaded as PNG or PDF files, and tables can be downloaded as CSV files.

Tests cover the analysis modules, the log, the methods text and the TRIPOD+AI report. GitHub Actions runs them on Python 3.10 and 3.12 for each push and pull request.

# Research impact statement

# AI usage disclosure

Claude (Anthropic) was used to aid in code review aND CODE creation.

# Acknowledgements

# References
