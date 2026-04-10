"""
models.py – Train LR / RF / XGB, calibrate to PD, and persist artefacts.

PD Calibration:
  Raw model probabilities are not necessarily well-calibrated.
  We use sklearn's CalibratedClassifierCV (on the validation set)
  so that predicted probabilities genuinely approximate P(default).
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from config import (
    RF_PARAMS, XGB_PARAMS, LR_PARAMS,
    CALIBRATION_METHOD, MODEL_DIR, RANDOM_STATE,
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False


# ── Scaling (fit on train only) ────────────────────────────
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    cols = X_train.columns

    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=cols, index=X_val.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=cols, index=X_test.index)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print("[scale] scaler fitted and saved")

    return X_train_s, X_val_s, X_test_s, scaler


# ── Train base models ─────────────────────────────────────
def _train_lr(X, y):
    m = LogisticRegression(**LR_PARAMS)
    m.fit(X, y)
    return m

def _train_rf(X, y):
    m = RandomForestClassifier(**RF_PARAMS)
    m.fit(X, y)
    return m

def _train_xgb(X, y):
    if not _XGB:
        print("[models] XGBoost not installed – skipping")
        return None
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X, y)
    return m


# ── PD Calibration ─────────────────────────────────────────
def calibrate_model(base_model, X_val, y_val, method=CALIBRATION_METHOD):
    """
    Wrap a fitted classifier with Platt (sigmoid) or Isotonic calibration.
    Uses the VALIDATION set – never the training data.
    """
    cal = CalibratedClassifierCV(
        base_model, method=method, cv="prefit"
    )
    cal.fit(X_val, y_val)
    return cal


# ── Public API ─────────────────────────────────────────────
def train_all(X_train, X_val, X_test, y_train, y_val):
    """
    Returns dict  { name: { "base": model, "calibrated": cal_model } }
    and the scaler object.
    """
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    builders = {
        "Logistic":     _train_lr,
        "RandomForest": _train_rf,
        "XGBoost":      _train_xgb,
    }

    results = {}
    for name, builder in builders.items():
        print(f"\n[train] {name}")
        base = builder(X_train_s, y_train)
        if base is None:
            continue

        cal = calibrate_model(base, X_val_s, y_val)
        results[name] = {"base": base, "calibrated": cal}

        # Persist
        joblib.dump(base, MODEL_DIR / f"{name}_base.pkl")
        joblib.dump(cal,  MODEL_DIR / f"{name}_calibrated.pkl")
        print(f"   saved {name}_base.pkl  &  {name}_calibrated.pkl")

    return results, scaler, X_train_s, X_val_s, X_test_s
