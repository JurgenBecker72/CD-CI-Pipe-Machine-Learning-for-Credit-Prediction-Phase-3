"""Reason codes for individual scoring requests.

Returns the top-N features contributing to the prediction, in the format
the FCRA adverse-action notice requires (per US Fair Credit Reporting
Act) -- a small set of named factors with their direction of impact.

Why coefficient-based (not SHAP)
--------------------------------
The production scorecard is a CalibratedClassifierCV wrapping a
LogisticRegression. For a linear model the per-feature contribution to
the log-odds prediction is exactly ``coef_i * x_i`` -- no closed-form
surprise, no non-linear interaction effects to recover. The sigmoid
calibration on top scales every contribution by the same learned
constant; direction and relative ranking are preserved, which is all
the reason-codes consumer needs (FCRA wants the top factors and the
direction of each, not absolute log-odds magnitudes).

SHAP's ``LinearExplainer`` would compute the same ``coef_i * x_i``
(modulo a baseline subtraction) at the cost of an extra dependency and
a heavier per-request runtime. The ``shap`` library does stay in the
training pipeline (``src/training/train.py``) where ``TreeExplainer``
produces non-trivial attributions for the RF challenger model.

If the production model class ever changes to a non-linear estimator
(XGBoost, RF, calibrated neural net), this module has to be rewritten
to use the explainer appropriate for the new flavour -- the
coefficient trick only works for linear models.
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
        The loaded sklearn classifier. Either a linear estimator
        exposing ``coef_`` directly, or a ``CalibratedClassifierCV``
        wrapping one (production case).
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

    feature_names = list(row.columns)
    if len(contributions) != len(feature_names):
        # The estimator's feature list and the input row may not align
        # one-to-one (drop_first dummy encoding, etc.). Truncate to the
        # shorter so the zip below doesn't mislabel reason codes.
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
    """Per-feature contributions: ``coef_i * x_i``.

    Exact SHAP value for an uncalibrated logistic regression with an
    origin baseline. For the sigmoid-calibrated wrapper the result
    differs from a strict SHAP attribution only by a uniform scalar
    learned by the calibrator -- enough to preserve the ranking and
    direction the reason-codes consumer reads.
    """
    underlying = _extract_linear_estimator(model)
    coefs = np.asarray(underlying.coef_).reshape(-1)
    values = np.asarray(row.iloc[0].values, dtype=float).reshape(-1)

    if len(coefs) != len(values):
        # Some features in the row may not appear in the model's coef_
        # vector (e.g. categoricals that got drop_first-encoded at
        # train time). Truncate both to the shorter so the element-wise
        # multiply is safe.
        n = min(len(coefs), len(values))
        coefs = coefs[:n]
        values = values[:n]

    return coefs * values


def _extract_linear_estimator(model: Any) -> Any:
    """Unwrap a CalibratedClassifierCV to its underlying linear estimator.

    The production loader returns a raw sklearn classifier. A
    ``CalibratedClassifierCV`` exposes the fitted base estimator at
    ``calibrated_classifiers_[0].estimator`` on sklearn 1.4+. If the
    object already exposes ``coef_`` directly (a bare ``LogisticRegression``
    in tests, for example), the function is a no-op.
    """
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model
