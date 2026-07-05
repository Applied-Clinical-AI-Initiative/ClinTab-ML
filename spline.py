"""spline.py
Restricted cubic spline fit of a binary outcome on a single continuous
predictor, using patsy for the basis and statsmodels for the logistic fit.
Returns the fitted log-odds curve (with 95% CI) plus knot positions so the
front end can draw log-odds vs predictor with knots as vertical lines.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


# Harrell's recommended knot quantiles by number of knots.
_KNOT_QUANTILES = {
    3: [0.10, 0.50, 0.90],
    4: [0.05, 0.35, 0.65, 0.95],
    5: [0.05, 0.275, 0.50, 0.725, 0.95],
    6: [0.05, 0.23, 0.41, 0.59, 0.77, 0.95],
    7: [0.025, 0.1833, 0.3417, 0.50, 0.6583, 0.8167, 0.975],
}


def fit_rcs(df, predictor, outcome, n_knots=4):
    """Fit logit(outcome) ~ rcs(predictor, knots).
    Returns dict with curve coordinates, knot positions and model stats.
    """
    n_knots = int(n_knots)
    if n_knots not in _KNOT_QUANTILES:
        n_knots = 4

    data = df[[predictor, outcome]].copy()
    data[predictor] = pd.to_numeric(data[predictor], errors="coerce")
    # encode outcome to 0/1
    y_raw = data[outcome]
    if pd.api.types.is_numeric_dtype(y_raw) and set(pd.unique(y_raw.dropna())) <= {0, 1}:
        data["_y"] = y_raw
    else:
        cats = sorted(y_raw.dropna().astype(str).unique())
        if len(cats) != 2:
            raise ValueError("Spline outcome must be binary (2 levels).")
        mapping = {cats[0]: 0, cats[1]: 1}
        data["_y"] = y_raw.astype(str).map(mapping)
    data = data.dropna(subset=[predictor, "_y"])
    if data.empty:
        raise ValueError("No complete rows for the chosen predictor/outcome.")

    x = data[predictor].values
    knots = list(np.quantile(x, _KNOT_QUANTILES[n_knots]))
    knots = sorted(set(round(float(k), 6) for k in knots))

    # patsy natural cubic spline (restricted) basis at the chosen knots
    knot_str = ", ".join(str(k) for k in knots)
    formula = f"cr(x, knots=[{knot_str}])"
    basis = patsy.dmatrix(formula, {"x": x}, return_type="dataframe")
    design = sm.add_constant(basis, has_constant="add")

    model = sm.GLM(data["_y"].values, design, family=sm.families.Binomial())
    res = model.fit()

    # predict the log-odds curve over a dense grid
    grid = np.linspace(np.min(x), np.max(x), 200)
    gbasis = patsy.build_design_matrices([basis.design_info], {"x": grid})[0]
    gdesign = sm.add_constant(np.asarray(gbasis), has_constant="add")
    pred = res.get_prediction(gdesign)
    mean = pred.predicted_mean
    # linear predictor CI: GLM predicted_mean is on the link (log-odds) scale
    ci = pred.conf_int()
    lower, upper = ci[:, 0], ci[:, 1]

    return {
        "predictor": predictor,
        "outcome": outcome,
        "n_knots": n_knots,
        "knots": knots,
        "n": int(len(data)),
        "x": grid.tolist(),
        "logodds": mean.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "aic": round(float(res.aic), 3),
        "deviance": round(float(res.deviance), 3),
    }
