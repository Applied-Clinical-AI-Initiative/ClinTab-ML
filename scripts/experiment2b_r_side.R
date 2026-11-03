# experiment2b_r_side.R
# Protocol Section 3: computes the same quantities as experiment2a_python_side.py,
# independently, using R's epitools / survival / ResourceSelection / rms, on
# the exact same intermediate data that script exported. Writes
# r_side_results.json into the same directory.
#
# Usage:
#   Rscript scripts/experiment2b_r_side.R /path/to/intermediate/dir

suppressMessages({
  library(epitools)
  library(survival)
  library(ResourceSelection)
  library(rms)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript experiment2b_r_side.R <out_dir>")
out_dir <- args[1]

results <- list()

# ---------------------------------------------------------------------------
# Row 2/3: 2x2 panel, with and without a zero cell -- epitools
# ---------------------------------------------------------------------------
run_2x2 <- function(path) {
  tbl <- read.csv(path)
  a <- tbl$a[1]; b <- tbl$b[1]; c <- tbl$c[1]; d <- tbl$d[1]
  # epitools' oddsratio()/riskratio() do NOT apply a Haldane-Anscombe
  # continuity correction on their own -- a zero cell just produces OR=0
  # and an undefined CI. clintab.epi.two_by_two() adds 0.5 to every cell
  # when any cell is zero; match that here so this is actually comparing
  # the corrected estimate, which is the whole point of this row.
  if (a == 0 || b == 0 || c == 0 || d == 0) {
    a <- a + 0.5; b <- b + 0.5; c <- c + 0.5; d <- d + 0.5
  }
  # epitools' oddsratio()/riskratio() take a table with exposure as rows
  # (unexposed, exposed) and disease as columns (no, yes). clintab's
  # convention is a=exposed+outcome+, b=exposed+outcome-, c=unexposed+
  # outcome+, d=unexposed+outcome-, so row1 (Unexposed) = [d, c] and
  # row2 (Exposed) = [b, a].
  m <- matrix(c(d, c, b, a), nrow = 2, byrow = TRUE,
              dimnames = list(c("Unexposed", "Exposed"), c("Outcome-", "Outcome+")))
  or_res <- oddsratio(m, method = "wald", correction = TRUE)
  rr_res <- riskratio(m, method = "wald", correction = TRUE)
  # NOTE: or_res$p.value[, "chi.square"] is actually the chi-square test's
  # P-VALUE despite the column name, not the statistic -- confirmed against
  # a plain chisq.test() on the same table. Pull the real statistic from
  # chisq.test() directly instead.
  chisq_res <- chisq.test(m, correct = TRUE)
  list(
    odds_ratio = unname(or_res$measure["Exposed", "estimate"]),
    or_ci = c(unname(or_res$measure["Exposed", "lower"]), unname(or_res$measure["Exposed", "upper"])),
    risk_ratio = unname(rr_res$measure["Exposed", "estimate"]),
    rr_ci = c(unname(rr_res$measure["Exposed", "lower"]), unname(rr_res$measure["Exposed", "upper"])),
    chi2 = unname(chisq_res$statistic),
    chi2_p_value = unname(chisq_res$p.value)
  )
}
results$twobytwo <- run_2x2(file.path(out_dir, "twobytwo_table.csv"))
results$twobytwo_zero_cell <- run_2x2(file.path(out_dir, "twobytwo_zero_table.csv"))

# ---------------------------------------------------------------------------
# Row 4: Cox PH -- survival::coxph
# ---------------------------------------------------------------------------
cox_data <- read.csv(file.path(out_dir, "cox_data.csv"))
cox_fit <- coxph(Surv(surv_time, death_event) ~ age + diabetes, data = cox_data)
cox_summary <- summary(cox_fit)
results$cox <- list(
  hazard_ratio_age = unname(cox_summary$coefficients["age", "exp(coef)"]),
  hr_ci_age = unname(cox_summary$conf.int["age", c("lower .95", "upper .95")]),
  hazard_ratio_diabetes = unname(cox_summary$coefficients["diabetes", "exp(coef)"]),
  hr_ci_diabetes = unname(cox_summary$conf.int["diabetes", c("lower .95", "upper .95")]),
  concordance = unname(cox_summary$concordance["C"])
)

# ---------------------------------------------------------------------------
# Row 5: Hosmer-Lemeshow -- ResourceSelection::hoslem.test
# ---------------------------------------------------------------------------
hl_data <- read.csv(file.path(out_dir, "hl_data.csv"))
hl_fit <- hoslem.test(hl_data$y_true, hl_data$y_prob, g = 10)
results$hosmer_lemeshow <- list(
  statistic = unname(hl_fit$statistic),
  dof = unname(hl_fit$parameter),
  p_value = unname(hl_fit$p.value)
)

# ---------------------------------------------------------------------------
# Row 6: RCS -- rms::lrm at matched knots
# ---------------------------------------------------------------------------
rcs_data <- read.csv(file.path(out_dir, "rcs_data.csv"))
knots <- fromJSON(file.path(out_dir, "rcs_knots.json"))
rcs_data$sex_bin <- as.integer(as.factor(rcs_data$sex)) - 1  # matches clintab's 0/1 mapping by sorted category

dd <- datadist(rcs_data)
old_opt <- options(datadist = "dd")
rcs_fit <- lrm(sex_bin ~ rcs(age, knots), data = rcs_data, x = TRUE, y = TRUE)
options(old_opt)

python_curve <- read.csv(file.path(out_dir, "rcs_python_curve.csv"))
pred_r <- predict(rcs_fit, newdata = data.frame(age = python_curve$x))
results$rcs <- list(
  aic = unname(AIC(rcs_fit)),
  logodds_at_grid = as.numeric(pred_r)
)

write(toJSON(results, auto_unbox = TRUE, digits = 8), file.path(out_dir, "r_side_results.json"))
cat("Wrote R-side results to", file.path(out_dir, "r_side_results.json"), "\n")
