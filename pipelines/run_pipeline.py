# =========================================================
# FINAL PIPELINE (SCORECARD + RF)
# =========================================================

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ID_COLUMNS, LEAKAGE_COLUMNS, RANDOM_STATE, TARGET
from src.models.train_scorecard import train_scorecard_model


# =========================================================
# LOAD DATA
# =========================================================
def load_data(path=None):
    """Load training data.

    Phase C onwards: by default the pipeline reads from the DuckDB warehouse
    (marts.applicant_features). The warehouse is refreshed from the source
    Excel on every call, gated by the Great Expectations data contract.

    Pass a `path` (or use --data-path on the CLI) to bypass the warehouse
    and load directly from an Excel file. Useful for ad-hoc testing on a
    different dataset without touching the warehouse state.
    """
    if path is None:
        # Default: warehouse-backed read.
        from src.data.warehouse import read_marts, refresh

        row_counts = refresh()
        print(
            f"\nWarehouse refreshed: raw={row_counts['raw']:,} -> "
            f"staging={row_counts['staging']:,} -> marts={row_counts['marts']:,}"
        )
        df = read_marts()
        print(f"Loaded data from marts.applicant_features: {df.shape}")
        return df

    # Override: load directly from a file.
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"\nLoaded data from file ({path}): {df.shape}")
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def create_features(df):

    if "risk_drivers" in df.columns and "risk_mitigators" in df.columns:
        df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)

    return df


# =========================================================
# CLEANUP — IDs and LEAKAGE
# =========================================================
def drop_ids_and_leakage(df):
    """
    Drop true identifier columns and all known performance-window
    (leakage) columns. Uses the exact lists in config — previously this
    used `"id" in c` which was fragile and was not catching
    num_accounts_perf / age_oldest_perf.
    """
    drop = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    print(f"Dropping IDs + leakage: {drop}")
    return df.drop(columns=drop)


# =========================================================
# IMPUTE (fit on TRAIN only)
# =========================================================
def impute_train_test(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    medians = X_train[numeric_cols].median()

    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(medians[col])
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(medians[col])

    return X_train, X_test


# =========================================================
# METRICS
# =========================================================
def evaluate(y_true, probs):
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    ks = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
    return auc, gini, ks


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline(path=None):
    """End-to-end training pipeline.

    Without arguments, reads from the DuckDB warehouse (Phase C default).
    Pass a path to override and read directly from an Excel file.
    """
    print("\n===== START PIPELINE =====")

    # ---------------------------------------------
    # LOAD + CLEAN
    # ---------------------------------------------
    df = load_data(path)
    df = create_features(df)
    df = drop_ids_and_leakage(df)

    # ---------------------------------------------
    # SPLIT FIRST — then impute. No fillna on the
    # full frame (that was leakage).
    # ---------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_test = impute_train_test(X_train, X_test)

    # ---------------------------------------------
    # SCORECARD
    # ---------------------------------------------
    model, scores_df, summary = train_scorecard_model(X_train, y_train, X_test, y_test)

    # ---------------------------------------------
    # RF BENCHMARK
    # ---------------------------------------------
    print("\n===== TRAINING RANDOM FOREST =====")

    # One-hot encode using TRAIN columns as the source of truth,
    # then re-align TEST to the same columns.
    X_train_e = pd.get_dummies(X_train, drop_first=True)
    X_test_e = pd.get_dummies(X_test, drop_first=True)
    X_test_e = X_test_e.reindex(columns=X_train_e.columns, fill_value=0)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
    rf.fit(X_train_e, y_train)

    probs_rf = rf.predict_proba(X_test_e)[:, 1]
    auc, gini, ks = evaluate(y_test, probs_rf)

    print(f"RF: AUC={auc:.4f} | Gini={gini:.4f} | KS={ks:.4f}")

    print("\n===== PIPELINE COMPLETE =====")


# =========================================================
# RUN
# =========================================================
def main() -> None:
    """CLI entry point.

    Default behaviour (Phase C onwards): refresh the DuckDB warehouse from
    the source dataset, then read marts.applicant_features. The Great
    Expectations contract runs at the raw -> staging boundary; a failed
    contract aborts the run with a clear error.

        python -m pipelines.run_pipeline                                # warehouse-backed
        python -m pipelines.run_pipeline --data-path /path/other.xlsx   # bypass warehouse

    The override is useful for ad-hoc testing on a different sample
    without polluting warehouse state.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the credit scorecard + RF training pipeline.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional Excel path that bypasses the warehouse "
        "(default: read from DuckDB marts.applicant_features).",
    )
    args = parser.parse_args()
    run_pipeline(args.data_path)


if __name__ == "__main__":
    main()
