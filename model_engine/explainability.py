"""
Model explainability via SHAP.

SHAP values are computed for the gradient-boosted and random-forest
forecasters. Two output artefacts are produced:

1. A **global feature importance table** (mean ``|SHAP|``) saved to
   ``results/metrics/shap_importance.csv``.
2. A **per-prediction explanation** returned by :func:`explain_prediction`,
   serialised by the ``/api/explain/`` endpoint and rendered on the
   dashboard as a horizontal bar chart for the chosen (station, date).

Rationale: the problem formulation commits to surfacing SHAP values in
the UI (responsible-ML criterion RM1, explainability). A post-hoc
attribution alone is not enough — per-prediction explanations let users
see *why* a given station-month was flagged as deteriorating.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("bekaasense")


def _select_explainer(model):
    """Pick the right SHAP explainer for the underlying estimator."""
    import shap
    # Tree models (RF, XGBoost) -> TreeExplainer (fast, exact)
    underlying = getattr(model, "model_", model)
    return shap.TreeExplainer(underlying)


def compute_global_importance(model, X_sample: pd.DataFrame,
                              out_path: Path | str | None = None
                              ) -> pd.DataFrame:
    """Mean ``|SHAP|`` per feature, sorted descending.

    Parameters
    ----------
    model
        A fitted :class:`RandomForestForecaster` or :class:`XGBoostForecaster`.
    X_sample
        A representative feature matrix — typically the validation set.
    out_path
        If given, the table is written as CSV.
    """
    explainer = _select_explainer(model)
    X = X_sample[model.feature_names_]
    shap_values = explainer.shap_values(X.values)
    # For multi-output classifiers SHAP may return a list; we only call this
    # on regressors here, so a 2-D array is expected.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    imp = pd.DataFrame({
        "feature": model.feature_names_,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        imp.to_csv(out_path, index=False)
        log.info("Wrote global SHAP importance -> %s", out_path)
    return imp


def explain_prediction(model, x_row: pd.DataFrame,
                       top_k: int = 8) -> dict:
    """Return a JSON-friendly per-row SHAP explanation.

    Output schema::

        {
          "base_value": 7.42,
          "prediction": 5.11,
          "top_features": [
            {"feature": "precip_roll3", "value": 12.3, "shap": -1.85},
            ...
          ]
        }
    """
    explainer = _select_explainer(model)
    X = x_row[model.feature_names_]
    shap_values = explainer.shap_values(X.values)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    row_shap = shap_values[0]
    exp_val = getattr(explainer, "expected_value", 0.0)
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = np.asarray(exp_val).ravel()
        base = float(exp_val[0]) if exp_val.size else 0.0
    else:
        base = float(exp_val)

    feats = [
        {
            "feature": f,
            "value": float(X.iloc[0][f]),
            "shap": float(row_shap[i]),
        }
        for i, f in enumerate(model.feature_names_)
    ]
    feats.sort(key=lambda d: abs(d["shap"]), reverse=True)

    return {
        "base_value": base,
        "prediction": float(model.predict(X)[0]),
        "top_features": feats[:top_k],
    }
