"""demo.py
A made-up hip and knee replacement cohort for trying every tab without real
patient data. Generated from a fixed seed, so the same call always gives the
same file. Nothing here is derived from any real dataset.

Columns:
  cpt_code          27130 total hip, 27447 total knee, plus some 27125 hip
                    hemiarthroplasty and 27446 partial knee rows to exclude
                    with the cohort builder
  joint             Hip or Knee
  age, sex, bmi, asa_class, diabetes, smoker
  creatinine        umol/L
  albumin           g/dL, about 40% missing (as in many registries)
  hemoglobin        g/L, about 5% missing
  readmission_30d   binary outcome, about 12%
  length_of_stay    days, continuous outcome
  years_followup    follow-up time in years
  revision_status   0 censored, 1 revision, 2 death before revision
                    (competing risks)
  revised           1 if revised, else 0 (death counts as censored)

Pure Python, no Flask.
"""
import numpy as np
import pandas as pd

DEMO_FILENAME = "demo_hip_knee_synthetic.csv"


def make_demo_dataset(n=10000, seed=2026):
    """Return the synthetic cohort as a DataFrame (deterministic for a seed)."""
    if n < 200:
        raise ValueError("The demo dataset needs at least 200 rows.")
    rng = np.random.default_rng(seed)

    cpt = rng.choice([27130, 27447, 27125, 27446], n, p=[0.40, 0.50, 0.05, 0.05])
    joint = np.where(np.isin(cpt, [27130, 27125]), "Hip", "Knee")
    hemi = cpt == 27125
    age = np.where(hemi, rng.normal(81, 7, n), rng.normal(67, 9.5, n)).clip(30, 99).round(0)
    sex = rng.choice(["Female", "Male"], n, p=[0.58, 0.42])
    bmi = rng.normal(31, 6, n).clip(16, 60).round(1)
    asa = rng.choice([1, 2, 3, 4], n, p=[0.04, 0.47, 0.43, 0.06])
    asa = np.where(hemi & (asa < 3), asa + 1, asa)
    diabetes = rng.binomial(1, 0.20, n)
    smoker = rng.binomial(1, 0.11, n)
    creat = rng.lognormal(np.log(82), 0.28, n).round(0)
    alb_true = rng.normal(4.0, 0.45, n) - 0.15 * (asa - 2)
    albumin = np.where(rng.random(n) < 0.40, np.nan, alb_true.round(1))
    hgb_true = rng.normal(135, 13, n) - 8 * (sex == "Female")
    hemoglobin = np.where(rng.random(n) < 0.05, np.nan, hgb_true.round(0))

    # 30-day readmission: logistic model on the true (not the missing) values
    lp = (-3.15 + 0.045 * (age - 67) + 0.05 * (bmi - 31) + 0.50 * (asa - 2)
          + 0.55 * diabetes + 0.45 * smoker + 0.30 * (joint == "Hip") + 0.9 * hemi
          + 0.008 * (creat - 82) - 0.70 * (alb_true - 4.0) - 0.012 * (hgb_true - 135))
    readmit = rng.binomial(1, 1 / (1 + np.exp(-lp)))

    # length of stay in days
    los = (1.2 + 0.03 * (age - 67) + 0.35 * (asa - 2) + 0.6 * hemi + 1.8 * readmit * 0.3
           + rng.gamma(2.0, 0.6, n)).clip(0.5, 30).round(1)

    # revision with death as a competing event, administrative censoring 1 to 12 years
    rev_rate = 0.010 * np.exp(0.35 * (joint == "Hip") + 0.30 * smoker + 0.02 * (bmi - 31)
                              - 0.02 * (age - 67))
    death_rate = 0.012 * np.exp(0.08 * (age - 67) + 0.40 * (asa - 2) + 0.30 * diabetes)
    t_rev = rng.exponential(1 / rev_rate)
    t_death = rng.exponential(1 / death_rate)
    t_cens = rng.uniform(1, 12, n)
    t = np.minimum.reduce([t_rev, t_death, t_cens])
    status = np.where(t == t_rev, 1, np.where(t == t_death, 2, 0))

    return pd.DataFrame({
        "cpt_code": cpt, "joint": joint, "age": age, "sex": sex, "bmi": bmi,
        "asa_class": asa, "diabetes": diabetes, "smoker": smoker,
        "creatinine": creat, "albumin": albumin, "hemoglobin": hemoglobin,
        "readmission_30d": readmit, "length_of_stay": los,
        "years_followup": t.round(2).clip(0.01), "revision_status": status,
        "revised": (status == 1).astype(int),
    })


def demo_csv_text(n=10000, seed=2026):
    """The demo dataset as CSV text."""
    return make_demo_dataset(n=n, seed=seed).to_csv(index=False)
