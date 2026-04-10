"""
features.py – Feature engineering applied AFTER the train/val/test split.

Quantile thresholds are computed on TRAIN only and then applied to val/test
to prevent information leakage through feature construction.
"""

import pandas as pd
import numpy as np


def _quantile_flag(series, q, direction="above"):
    """Binary flag: 1 if above (or below) the q-th percentile."""
    threshold = series.quantile(q)
    if direction == "above":
        return (series > threshold).astype(int)
    return (series < threshold).astype(int)


def engineer_features(X_train, X_val, X_test):
    """
    Creates engineered features using thresholds learned from X_train only.
    Returns modified copies (does not mutate originals).
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    datasets = [X_train, X_val, X_test]

    # ── Risk flags (thresholds from train) ─────────────────
    if "total_risk_score" in X_train.columns:
        hi_risk_q70 = X_train["total_risk_score"].quantile(0.70)
        lo_risk_q30 = X_train["total_risk_score"].quantile(0.30)
        for df in datasets:
            df["high_risk_flag"] = (df["total_risk_score"] > hi_risk_q70).astype(int)
            df["low_risk_flag"]  = (df["total_risk_score"] < lo_risk_q30).astype(int)

    # ── Emotional flag ─────────────────────────────────────
    if "r_ho_em2_co" in X_train.columns:
        em_q70 = X_train["r_ho_em2_co"].quantile(0.70)
        for df in datasets:
            df["high_emotional_flag"] = (df["r_ho_em2_co"] > em_q70).astype(int)

    # ── Stability flags ────────────────────────────────────
    if "r_ho_vi4_st" in X_train.columns:
        st_lo = X_train["r_ho_vi4_st"].quantile(0.30)
        st_hi = X_train["r_ho_vi4_st"].quantile(0.70)
        for df in datasets:
            df["low_stability_flag"]  = (df["r_ho_vi4_st"] < st_lo).astype(int)
            df["high_stability_flag"] = (df["r_ho_vi4_st"] > st_hi).astype(int)

    # ── Interaction terms ──────────────────────────────────
    if "high_risk_flag" in X_train.columns and "high_emotional_flag" in X_train.columns:
        for df in datasets:
            df["risk_x_emotional"] = df["high_risk_flag"] * df["high_emotional_flag"]

    if "high_risk_flag" in X_train.columns and "low_stability_flag" in X_train.columns:
        for df in datasets:
            df["risk_x_low_stability"] = df["high_risk_flag"] * df["low_stability_flag"]

    # ── Net-risk & ratio ───────────────────────────────────
    if "risk_drivers" in X_train.columns and "risk_mitigators" in X_train.columns:
        for df in datasets:
            df["net_risk"]   = df["risk_drivers"] - df["risk_mitigators"]
            df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)

    # ── DRA dimension interactions ─────────────────────────
    if "total_risk_score" in X_train.columns and "dim_emotional_understanding" in X_train.columns:
        for df in datasets:
            df["risk_x_dim_emo"] = df["total_risk_score"] * df["dim_emotional_understanding"]

    if "dim_judgement" in X_train.columns and "dim_core_traits" in X_train.columns:
        for df in datasets:
            df["judgement_x_traits"] = df["dim_judgement"] * df["dim_core_traits"]

    new_feats = [c for c in X_train.columns if c not in X_val.columns or c in X_train.columns]
    print(f"[features] {X_train.shape[1]} total features (incl. engineered)")

    return X_train, X_val, X_test
