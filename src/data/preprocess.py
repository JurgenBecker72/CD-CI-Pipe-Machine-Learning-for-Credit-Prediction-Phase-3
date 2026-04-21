# =========================================================
# CREDIT RISK PREPROCESSING PIPELINE
# =========================================================

# =========================================================
# STEP 1: IMPORTS
# =========================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

from src.config import (
    TARGET,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    RANDOM_STATE,
)


# =========================================================
# STEP 2: LOAD DATA
# =========================================================
def load_data():
    """
    Loads raw dataset from Excel.
    """
    df = pd.read_excel("data/raw/DRA_with_simulated_credit.xlsx")
    return df


# =========================================================
# STEP 3: CLEAN COLUMN NAMES
# =========================================================
def clean_column_names(df):
    """
    Standardizes column names:
    - strip whitespace
    - lowercase
    - replace spaces with underscores
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# =========================================================
# STEP 4: REMOVE ID / IDENTIFIER COLUMNS
# =========================================================
def remove_ids(df):
    """
    Removes TRUE identifier columns only (explicit list from config).

    NOTE: The previous version did substring matching on "id", "account",
    etc. That was a bug — it dropped legitimate features like
    `num_accounts_assess` because "account" is a substring of the name.
    We now match identifiers exactly using config.ID_COLUMNS.
    """
    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    print(f"Removing ID columns: {id_cols}")
    return df.drop(columns=id_cols)


# =========================================================
# STEP 5: REMOVE LEAKAGE VARIABLES
# =========================================================
def remove_leakage(df):
    """
    Removes performance-window features that contain FUTURE / outcome
    information. Critical for credit modelling — keeping any of these
    would inflate AUC and invalidate the model in production.

    The list is centralised in config.LEAKAGE_COLUMNS and now covers:
        - num_accounts_perf
        - highest_arrears_perf
        - age_oldest_perf
    """
    existing = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    print(f"Removing leakage columns: {existing}")
    return df.drop(columns=existing)


# =========================================================
# STEP 6: FEATURE ENGINEERING (row-wise only, no leakage)
# =========================================================
def create_features(df):
    """
    Row-wise engineered features. Safe to compute before or after split
    because no cross-row statistics are used.
    """
    if "income" in df.columns and "expenses" in df.columns:
        df["affordability_ratio"] = df["income"] / (df["expenses"] + 1)

    if "total_risk_score" in df.columns and "dim_emotional_understanding" in df.columns:
        df["risk_x_emotional"] = (
            df["total_risk_score"] * df["dim_emotional_understanding"]
        )

    if "risk_drivers" in df.columns and "risk_mitigators" in df.columns:
        df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)
        df["net_risk"] = df["risk_drivers"] - df["risk_mitigators"]

    return df


# =========================================================
# STEP 7: ENCODE CATEGORICAL VARIABLES
# =========================================================
def encode_categorical(df):
    """
    One-hot encode categoricals on the full feature frame.
    Done BEFORE the split so train/val/test share the same columns.
    (One-hot encoding uses no target information → not leakage.)
    """
    df = pd.get_dummies(df, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# =========================================================
# STEP 8: FINAL NUMERIC SAFETY CHECK
# =========================================================
def ensure_numeric(df):
    non_numeric = df.select_dtypes(exclude=[np.number]).columns

    if len(non_numeric) > 0:
        print("\n🚨 ERROR: Non-numeric columns detected BEFORE modeling:")
        print(list(non_numeric))
        raise ValueError("Non-numeric data detected — fix preprocessing")

    return df


# =========================================================
# STEP 9: SPLIT FEATURES AND TARGET
# =========================================================
def split_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# =========================================================
# STEP 10: TRAIN / VALIDATION / TEST SPLIT
# =========================================================
def split_data(X, y):
    """
    Splits data into:
    - Train (70%)
    - Validation (10%)
    - Test  (20%)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# STEP 11: HANDLE MISSING VALUES (fit on TRAIN only)
# =========================================================
def handle_missing(X_train, X_val, X_test):
    """
    Numeric → median imputation, medians computed from TRAIN only.
    Categorical → 'missing'.

    Computing imputation statistics on the full dataset before the split
    leaks information from val/test into train. We now fit on X_train
    only and apply the same fill values to X_val and X_test.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[numeric_cols].median()

    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(train_medians[col])
        if col in X_val.columns:
            X_val[col] = X_val[col].fillna(train_medians[col])
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(train_medians[col])

    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric:
        X_train[col] = X_train[col].fillna("missing")
        if col in X_val.columns:
            X_val[col] = X_val[col].fillna("missing")
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna("missing")

    return X_train, X_val, X_test


# =========================================================
# STEP 12: SCALE FEATURES (fit on TRAIN only)
# =========================================================
def scale_features(X_train, X_val, X_test):
    """
    StandardScaler fit on TRAIN, applied to VAL and TEST.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_val_scaled, X_test_scaled


# =========================================================
# STEP 13: LEAKAGE DIAGNOSTIC
# =========================================================
def check_single_feature_auc(X, y):
    """
    Flags single features with suspiciously high AUC — helps catch
    any leakage that slipped through the explicit drop list.
    """
    print("\nChecking for suspicious predictors...")

    for col in X.columns:
        try:
            auc = roc_auc_score(y, X[col])
            if auc > 0.90 or auc < 0.10:
                print(f"WARNING: {col} has extreme single-feature AUC ({auc:.3f})")
        except Exception:
            pass


# =========================================================
# STEP 14: SAVE OUTPUT DATA
# =========================================================
def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_val.to_csv("data/processed/X_val.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)

    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)


# =========================================================
# STEP 15: MAIN PIPELINE FUNCTION
# =========================================================
def main():

    print("\nLoading data...")
    df = load_data()

    print("\nCleaning column names...")
    df = clean_column_names(df)

    print("\nRemoving ID columns...")
    df = remove_ids(df)

    print("\nRemoving leakage...")
    df = remove_leakage(df)

    print("\nCreating features...")
    df = create_features(df)

    print("\nEncoding categorical variables...")
    df = encode_categorical(df)

    print("\nEnsuring numeric-only data...")
    df = ensure_numeric(df)

    print("\nSplitting features and target...")
    X, y = split_xy(df)

    print("\nSplitting data (train/val/test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ---- All fit-on-train steps happen AFTER the split ----
    print("\nHandling missing values (fit on train)...")
    X_train, X_val, X_test = handle_missing(X_train, X_val, X_test)

    print("\nScaling features (fit on train)...")
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    print("\nRunning leakage diagnostics...")
    check_single_feature_auc(X_train, y_train)

    print("\nSaving processed data...")
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\nPreprocessing complete.")


# =========================================================
# PIPELINE ENTRY POINT
# =========================================================
def preprocess_data():
    main()


# =========================================================
# RUN STANDALONE
# =========================================================
if __name__ == "__main__":
    main()
