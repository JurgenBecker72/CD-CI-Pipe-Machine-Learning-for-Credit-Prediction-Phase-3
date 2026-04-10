# src/features/features.py
# Optional engineered features. Every block is guarded so missing source
# columns never raise — the original version assumed they always existed.

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "total_risk_score" in df.columns:
        q_hi = df["total_risk_score"].quantile(0.70)
        q_lo = df["total_risk_score"].quantile(0.30)
        df["high_risk_flag"] = (df["total_risk_score"] > q_hi).astype(int)
        df["low_risk_flag"] = (df["total_risk_score"] < q_lo).astype(int)

    if {"risk_drivers", "risk_mitigators"}.issubset(df.columns):
        df["net_risk"] = df["risk_drivers"] - df["risk_mitigators"]
        df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)

    if {"total_risk_score", "r_ho_em2_co"}.issubset(df.columns):
        df["risk_x_emotional"] = df["total_risk_score"] * df["r_ho_em2_co"]

    if {"r_ho_em2_co", "r_ho_vi4_st"}.issubset(df.columns):
        df["emotional_x_stability"] = df["r_ho_em2_co"] * df["r_ho_vi4_st"]

    return df
