import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import pytest

import clintab_cli
from clintab import store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "Data", "sample_clinical.csv")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """models and exports go to a temp folder, never into the repo."""
    monkeypatch.setattr(store, "RUNTIME", str(tmp_path / "runtime"))
    monkeypatch.setattr(store, "SESSIONS", str(tmp_path / "runtime" / "sessions"))
    monkeypatch.setattr(store, "EXPORTS", str(tmp_path / "runtime" / "exports"))
    monkeypatch.setattr(store, "MODELS", str(tmp_path / "models"))
    return tmp_path


def _run(capsys, *argv):
    assert clintab_cli.main(list(argv)) == 0
    return capsys.readouterr()


def _saved(home):
    return sorted(f[:-4] for f in os.listdir(home / "models") if f.endswith(".pkl"))


# ---------------------------------------------------------------------------
# table formatting
# ---------------------------------------------------------------------------
def test_table_folds_the_ci_into_the_estimate_and_drops_the_ci_column(capsys):
    rows = [{"model": "m", "AUROC": 0.71, "Brier": 0.2,
             "CI": {"AUROC": [0.65, 0.77], "Brier": [None, None]}}]
    clintab_cli.table(rows, ["model", "AUROC", "Brier", "CI"])
    out = capsys.readouterr().out
    assert "0.71 (0.65 to 0.77)" in out
    assert "{" not in out and " CI" not in out.splitlines()[0]
    assert "0.2 " in out or out.rstrip().endswith("0.2")      # no interval when it is missing


# ---------------------------------------------------------------------------
# train / test
# ---------------------------------------------------------------------------
TRAIN = ["train", "--csv", SAMPLE, "--outcome", "mortality_30d",
         "--exclude", "surv_time", "death_event", "--no-grid"]


def test_train_chooses_the_threshold_recalibrates_and_prints_odds_ratios(home, capsys):
    out = _run(capsys, *TRAIN, "--models", "LogisticRegression",
               "--threshold-method", "specificity", "--threshold-target", "0.8",
               "--recalibrate").out
    assert "Adjusted odds ratios" in out and "sex=M" in out
    assert "(specificity, on the validation set), recalibrated" in out
    assert "{" not in out.split("Validation metrics:")[1]        # no raw CI dict
    name = _saved(home)[0]
    mdl = joblib.load(store.model_path(name))
    assert mdl["threshold_method"] == "specificity" and mdl["threshold_target"] == 0.8
    assert mdl["recalibrated"] is True and set(mdl["recalibration"]) == {"a", "b"}
    meta = json.load(open(store.model_meta_path(name)))
    assert meta["odds_ratios"]["rows"] and meta["threshold"] == mdl["threshold"]


def test_train_bare_threshold_still_means_a_fixed_threshold(home, capsys):
    _run(capsys, *TRAIN, "--models", "DecisionTree", "--threshold", "0.35")
    mdl = joblib.load(store.model_path(_saved(home)[0]))
    assert mdl["threshold"] == 0.35 and mdl["threshold_method"] == "fixed"


def test_train_sensitivity_without_a_target_exits(home):
    with pytest.raises(SystemExit) as e:
        clintab_cli.main([*TRAIN, "--models", "LogisticRegression",
                          "--threshold-method", "sensitivity"])
    assert "--threshold-target" in str(e.value)


def test_test_uses_the_saved_threshold_unless_overridden(home, capsys):
    _run(capsys, *TRAIN, "--models", "LogisticRegression")
    name = _saved(home)[0]
    saved = joblib.load(store.model_path(name))["threshold"]
    out = _run(capsys, "test", "--models", name, "--csv", SAMPLE).out
    row = [line for line in out.splitlines() if line.strip().startswith(name)][0]
    assert " saved " in row and str(saved)[:6] in row
    out = _run(capsys, "test", "--models", name, "--csv", SAMPLE, "--threshold", "0.3").out
    row = [line for line in out.splitlines() if line.strip().startswith(name)][0]
    assert " user " in row and " 0.3 " in row


# ---------------------------------------------------------------------------
# epi
# ---------------------------------------------------------------------------
def test_epi_km_prints_a_number_at_risk_table(capsys):
    out = _run(capsys, "epi", "km", "--csv", SAMPLE, "--time", "surv_time",
               "--event", "death_event", "--group", "sex", "--time-points", "0", "12", "24").out
    assert "Number at risk" in out
    header = [line for line in out.splitlines() if line.strip().startswith("group")][0]
    assert header.split() == ["group", "0", "12", "24"]


def test_epi_cox_categorical_and_ph_test(capsys):
    out = _run(capsys, "epi", "cox", "--csv", SAMPLE, "--time", "surv_time",
               "--event", "death_event", "--cov", "age", "asa_class",
               "--categorical", "asa_class", "--ph-transform", "km").out
    assert "asa_class: 2 vs 1" in out
    assert "proportional hazards test (km): global" in out


def test_epi_cif_prints_curves_gray_test_and_saves_a_plot(tmp_path, capsys):
    df = pd.read_csv(SAMPLE)
    # recode some deaths as a competing event (2)
    df.loc[(df["death_event"] == 1) & (df["age"] > 75), "death_event"] = 2
    csv = tmp_path / "cr.csv"
    df.to_csv(csv, index=False)
    png = tmp_path / "cif.png"
    out = _run(capsys, "epi", "cif", "--csv", str(csv), "--time", "surv_time",
               "--event", "death_event", "--event-code", "1", "--group", "sex",
               "--out", str(png)).out
    assert "competing=" in out and "Gray's test" in out
    assert "overstates" in out
    assert png.exists() and png.stat().st_size > 0


# ---------------------------------------------------------------------------
# sample size, teaching, cohort
# ---------------------------------------------------------------------------
def test_samplesize_matches_pmsampsize_and_compares_with_data(capsys):
    out = _run(capsys, "samplesize", "--prevalence", "0.2", "--parameters", "10",
               "--cstat", "0.75", "--n", "600", "--events", "114").out
    assert "minimum n = 666, events = 134" in out
    assert "your data: n = 600, events = 114" in out and "adequate = False" in out


def test_samplesize_needs_exactly_one_r2_measure():
    with pytest.raises(SystemExit):
        clintab_cli.main(["samplesize", "--prevalence", "0.2", "--parameters", "10"])
    with pytest.raises(SystemExit):
        clintab_cli.main(["samplesize", "--prevalence", "0.2", "--parameters", "10",
                          "--cstat", "0.7", "--csr2", "0.1"])


def test_samplesize_bad_value_exits_with_a_plain_message():
    with pytest.raises(SystemExit) as e:
        clintab_cli.main(["samplesize", "--prevalence", "1.5", "--parameters", "10",
                          "--cstat", "0.7"])
    assert "Prevalence" in str(e.value)


def test_teach_progress_on_stderr_table_on_stdout_and_files(tmp_path, capsys):
    js, png = tmp_path / "teach.json", tmp_path / "teach.png"
    res = _run(capsys, "teach", "--csv", SAMPLE, "--outcome", "mortality_30d",
               "--exclude", "surv_time", "death_event", "--model", "logistic",
               "--seeds", "1", "--max-rows", "400", "--json", str(js), "--png", str(png))
    assert "Repeat 1 of 1" in res.err and "Done" in res.err
    assert "Repeat 1 of 1" not in res.out
    assert "Done correctly" in res.out and "optimism" in res.out
    data = json.load(open(js))
    assert data["seeds"] == [1] and data["n_used"] == 400
    assert png.exists() and png.stat().st_size > 0


def test_teach_rejects_a_continuous_outcome():
    with pytest.raises(SystemExit) as e:
        clintab_cli.main(["teach", "--csv", SAMPLE, "--outcome", "length_of_stay",
                          "--seeds", "1"])
    assert "binary outcome" in str(e.value)


def test_cohort_prints_the_flow_and_writes_the_cohort(tmp_path, capsys):
    filters = tmp_path / "filters.json"
    filters.write_text(json.dumps([{"column": "age", "op": "gte", "values": [30]},
                                   {"column": "asa_class", "op": "in", "values": "1-3"}]))
    out_csv, flow = tmp_path / "cohort.csv", tmp_path / "flow.json"
    out = _run(capsys, "cohort", "--csv", SAMPLE, "--filters", str(filters),
               "--out", str(out_csv), "--flow", str(flow)).out
    df = pd.read_csv(SAMPLE)
    keep = df[(df["age"] >= 30) & (df["asa_class"] <= 3)]
    assert "Records in file: 600" in out and f"Final cohort: {len(keep)}" in out
    assert len(pd.read_csv(out_csv)) == len(keep)
    saved = json.load(open(flow))
    assert saved["n_start"] == 600 and saved["n_final"] == len(keep)
    assert len(saved["flow"]) == 2 and saved["lines"][-1] == f"Final cohort: {len(keep)}"


def test_cohort_accepts_a_filters_object_and_reports_bad_filters(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"filters": [{"column": "sex", "op": "eq", "values": ["F"]}]}))
    out = _run(capsys, "cohort", "--csv", SAMPLE, "--filters", str(good)).out
    assert "Final cohort:" in out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"column": "nope", "op": "gte", "values": [1]}]))
    with pytest.raises(SystemExit) as e:
        clintab_cli.main(["cohort", "--csv", SAMPLE, "--filters", str(bad)])
    assert "unknown column 'nope'" in str(e.value)


def test_cli_runs_as_a_script():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "clintab_cli.py"), "samplesize",
                        "--prevalence", "0.2", "--parameters", "10", "--csr2", "0.1"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "minimum n =" in r.stdout
