# Data

`sample_clinical.csv` is a synthetic dataset used throughout this repo's
examples and CLI walkthroughs (`clintab_cli.py summarize`, `train`, `spline`,
`epi`, etc.). It does not contain real patient records. The values were
generated to look and behave like a small surgical outcomes registry, so the
app's features (column-type detection, missing-data handling, model
training, survival analysis) have something realistic to run against.

It has 600 rows and 11 columns:

| Column            | Description                                          |
|-------------------|-------------------------------------------------------|
| `age`             | Patient age in years                                  |
| `bmi`             | Body mass index                                        |
| `sex`             | `M` / `F`                                              |
| `asa_class`       | ASA physical status classification (1-5)               |
| `length_of_stay`  | Hospital length of stay, in days                       |
| `diabetes`        | Diabetes diagnosis (0/1)                               |
| `mortality_30d`   | 30-day mortality outcome (0/1)                         |
| `surv_time`       | Follow-up / survival time, for the KM and Cox examples |
| `death_event`     | Event indicator paired with `surv_time` (0/1)          |
| `admit_date`      | Admission date                                         |
| `rare_lab`        | A lab value populated for only ~42% of rows            |

`rare_lab` is deliberately sparse — it's there so the missing-data handling
step (include / zero-fill / drop column) has a real column to demonstrate on.
Every other column is fully populated.

If you're evaluating or extending this project, feel free to drop in your
own CSV alongside this one; nothing in the app is hard-coded to this file.
