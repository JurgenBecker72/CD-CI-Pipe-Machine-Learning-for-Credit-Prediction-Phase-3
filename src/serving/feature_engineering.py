"""Scoring-time feature engineering — pure pandas, no Spark.

The training pipeline runs feature engineering on Spark (see
`src.features.pipeline`) and persists the fitted state as JSON
sidecars in MLflow. At scoring time the runtime needs to apply the
*same* transformations to a single applicant payload, but spinning up
a Spark session per scoring request is impractical (5-10s JVM boot,
hundreds of MB of memory). Instead we re-implement the feature math in
pure pandas using the persisted thresholds.

The contract: identical output to the Spark pipeline given the same
fitted thresholds and the same input row. Verified by the test suite.
"""

from __future__ import annotations

import pandas as pd

# Mirror the Spark pipeline's INTERACTION_FEATURES list. Single source
# of truth is src/features/spark_features.py; replicated here as a
# constant to avoid importing PySpark into the slim serving runtime.
INTERACTION_FEATURES: list[tuple[str, str, str]] = [
    ("emotional_x_stability", "r_ho_em2_co", "r_ho_vi4_st"),
    ("risk_x_emotional", "total_risk_score", "r_ho_em2_co"),
    ("drivers_x_mitigators", "risk_drivers", "risk_mitigators"),
    ("risk_x_mitigators", "total_risk_score", "risk_mitigators"),
]


def add_row_wise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the stateless row-wise feature engineering in pandas.

    Mirrors the behaviour of `RowWiseFeatures._transform` from the Spark
    pipeline. Missing source columns are skipped (rather than raising)
    so a scoring request that omits an optional field still produces a
    result.
    """
    df = df.copy()

    if "risk_drivers" in df.columns and "risk_mitigators" in df.columns:
        df["net_risk"] = df["risk_drivers"] - df["risk_mitigators"]
        df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)

    for out_col, left, right in INTERACTION_FEATURES:
        if left in df.columns and right in df.columns:
            df[out_col] = df[left] * df[right]

    return df


def apply_quantile_flags(
    df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply the fitted high/low quantile flags using frozen thresholds.

    Mirrors `QuantileFlaggerModel._transform`. The thresholds come from
    the `thresholds.json` sidecar logged to MLflow alongside the model.
    """
    df = df.copy()

    for col, bounds in thresholds.items():
        if col not in df.columns:
            continue
        high_t = bounds["high"]
        low_t = bounds["low"]
        df[f"{col}_high_flag"] = (df[col] > high_t).astype(int)
        df[f"{col}_low_flag"] = (df[col] < low_t).astype(int)

    return df


def engineer_features(
    df: pd.DataFrame,
    quantile_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Full scoring-time feature engineering: row-wise + quantile flags.

    Replaces the Spark pipeline's transform() at scoring time. Given the
    same input and the same fitted thresholds, produces bit-for-bit the
    same feature columns as the training pipeline.
    """
    df = add_row_wise_features(df)
    df = apply_quantile_flags(df, quantile_thresholds)
    return df
