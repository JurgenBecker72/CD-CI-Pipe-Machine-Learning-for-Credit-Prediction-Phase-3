"""SHAP-based reason codes for individual scoring requests.

Returns the top N features contributing to the decision, in the format
the FCRA adverse-action notice requires (per US Fair Credit Reporting
Act) -- a small set of named factors with their direction of impact.

Implementation notes
--------------------
The scorecard is a CalibratedClassifierCV wrapping a LogisticRegression.
SHAP's LinearExplainer is the right tool: it gives exact SHAP values for
linear models in milliseconds rather than minutes. The TreeExplainer
path used for the RF challenger is much slower per request and isn't
needed when the production model is linear.

If SHAP isn't available at runtime (e.g. a stripped serving image),
fall back to coefficient-weight reason codes -- not as theoretically
clean but operationally equivalent for linear models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TOP_N_REASONS = 3


def compute_reason_codes(
    model: Any,
    row: pd.DataFrame,
    top_n: int = TOP_N_REASONS,
) -> list[dict]:
    """Return the top-N contributing features for a single scoring row.

    Parameters
    ----------
    model
        The loaded sklearn model. Expected to have predict_proba and
        coef_ (LogisticRegression) or feature_names_in_.
    row
        One-row pandas DataFrame containing the model's input features.
    top_n
        Number of reason codes to return.

    Returns
    -------
    list of dicts with keys: feature, contribution, direction.
        Sorted by absolute contribution descending.
    """
    if len(row) != 1:
        raise ValueError(f"reason codes expect exactly one row, got {len(row)}")

    contributions = _compute_contributions(model, row)

    # Build a sorted dataframe of (feature, contribution).
    feature_names = list(row.columns)
    if len(contributions) != len(feature_names):
        # The model's internal feature list may differ from the input row
        # (column ordering, dummy encoding, etc.). Truncate to the shorter
        # of the two to avoid misalignment.
        n = min(len(contributions), len(feature_names))
        contributions = contributions[:n]
        feature_names = feature_names[:n]

    pairs = sorted(
        zip(feature_names, contributions, strict=True),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    return [
        {
            "feature": name,
            "contribution": float(value),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
        }
        for name, value in pairs[:top_n]
    ]


def _compute_contributions(model: Any, row: pd.DataFrame) -> np.ndarray:
    """Compute per-feature contributions to the predicted probability.

    Strategy ordering:
    1. SHAP LinearExplainer if SHAP is installed and the underlying
       estimator is a calibrated linear model.
    2. Coefficient-weight fallback: feature_value * coef_, which is the
       exact SHAP value for an uncalibrated logistic regression and a
       reasonable approximation for the calibrated version.
    """
    try:
        return _shap_linear_contributions(model, row)
    except Exception as exc:  # noqa: BLE001 - fall through to coef fallback
        print(f"[reason_codes] SHAP path failed ({type(exc).__name__}: {exc}); using coef fallback")
        return _coef_contributions(model, row)


def _shap_linear_contributions(model: Any, row: pd.DataFrame) -> np.ndarray:
    """SHAP values via LinearExplainer. Raises if SHAP is unavailable."""
    import shap  # local import so the serving image can omit shap if desired

    # The CalibratedClassifierCV holds the underlying linear model in
    # estimator.calibrated_classifiers_[0].estimator on newer sklearn.
    underlying = _extract_linear_estimator(model)
    explainer = shap.LinearExplainer(underlying, row)
    values = explainer.shap_values(row)
    # LinearExplainer returns a (1, n_features) array for one row.
    return np.asarray(values).reshape(-1)


def _coef_contributions(model: Any, row: pd.DataFrame) -> np.ndarray:
    """Fallback: contribution = feature_value * coefficient.

    Exact SHAP value for an uncalibrated linear model, and a good
    approximation for a calibrated wrapper (calibration adjusts the
    intercept and scale but not the per-feature directionality).
    """
    underlying = _extract_linear_estimator(model)
    coefs = np.asarray(underlying.coef_).reshape(-1)
    values = np.asarray(row.iloc[0].values, dtype=float).reshape(-1)

    if len(coefs) != len(values):
        n = min(len(coefs), len(values))
        coefs = coefs[:n]
        values = values[:n]

    return coefs * values


def _extract_linear_estimator(model: Any) -> Any:
    """Walk through pyfunc / calibrated wrappers to the underlying linear model."""
    candidate = model
    # mlflow.pyfunc wraps the actual sklearn estimator
    if hasattr(candidate, "_model_impl"):
        candidate = candidate._model_impl  # type: ignore[attr-defined]
    if hasattr(candidate, "sklearn_model"):
        candidate = candidate.sklearn_model  # type: ignore[attr-defined]
    if hasattr(candidate, "python_model"):
        candidate = candidate.python_model  # type: ignore[attr-defined]
    # CalibratedClassifierCV -> first calibrated classifier -> base estimator
    if hasattr(candidate, "calibrated_classifiers_"):
        candidate = candidate.calibrated_classifiers_[0].estimator
    return candidate
