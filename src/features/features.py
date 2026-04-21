# ============================================================
# FEATURE ENGINEERING
# ============================================================


def create_features(df):

    # ===== FLAGS =====
    if "total_risk_score" in df.columns:
        df["high_risk_flag"] = (
            df["total_risk_score"] > df["total_risk_score"].quantile(0.7)
        ).astype(int)
        df["low_risk_flag"] = (
            df["total_risk_score"] < df["total_risk_score"].quantile(0.3)
        ).astype(int)

    if "r_ho_em2_co" in df.columns:
        df["high_emotional_flag"] = (df["r_ho_em2_co"] > df["r_ho_em2_co"].quantile(0.7)).astype(
            int
        )

    if "r_ho_vi4_st" in df.columns:
        df["low_stability_flag"] = (df["r_ho_vi4_st"] < df["r_ho_vi4_st"].quantile(0.3)).astype(int)
        df["high_stability_flag"] = (df["r_ho_vi4_st"] > df["r_ho_vi4_st"].quantile(0.7)).astype(
            int
        )

    # ===== RISK STRUCTURE =====
    if "risk_drivers" in df.columns and "risk_mitigators" in df.columns:
        df["net_risk"] = df["risk_drivers"] - df["risk_mitigators"]
        df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)

    # ===== SHAP INTERACTIONS =====
    df["emotional_x_stability"] = df["r_ho_em2_co"] * df["r_ho_vi4_st"]
    df["risk_x_emotional"] = df["total_risk_score"] * df["r_ho_em2_co"]
    df["drivers_x_mitigators"] = df["risk_drivers"] * df["risk_mitigators"]
    df["risk_x_mitigators"] = df["total_risk_score"] * df["risk_mitigators"]

    return df
