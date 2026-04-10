"""
pipelines/run_pipeline.py

End-to-end credit risk pipeline:
    ingest -> preprocess -> feature engineering -> split ->
    scorecard (calibrated logistic) + random forest benchmark ->
    persist processed data, metrics and band summary.

Run with:
    python -m pipelines.run_pipeline
"""

import json

import pandas as pd

from src.config import TARGET
from src.paths import PROCESSED_DIR, REPORTS_DIR
from src.data.ingest import load_credit_data
from src.data.preprocess import preprocess_data
from src.data.split import split_data
from src.features.features import create_features
from src.models.train_scorecard import train_scorecard_model
from src.models.train_rf import train_random_forest


def _save_processed(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)


def run_pipeline() -> dict:
    print("===== CREDIT RISK PIPELINE =====")

    df = load_credit_data()
    df = preprocess_data(df)
    df = create_features(df)

    if TARGET not in df.columns:
        raise RuntimeError(f"Target '{TARGET}' missing after feature engineering")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    _save_processed(X_train, X_val, X_test, y_train, y_val, y_test)

    _, scores_df, band_summary, sc_metrics = train_scorecard_model(
        X_train, y_train, X_test, y_test
    )

    _, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    scores_df.to_csv(REPORTS_DIR / "scorecard_test_scores.csv", index=False)
    band_summary.to_csv(REPORTS_DIR / "scorecard_band_summary.csv", index=False)

    report = {"scorecard": sc_metrics, "random_forest": rf_metrics}
    with open(REPORTS_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n===== PIPELINE COMPLETE =====")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_pipeline()
