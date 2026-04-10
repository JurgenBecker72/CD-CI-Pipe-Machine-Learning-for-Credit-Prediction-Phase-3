"""
data_prep.py – Ingest, clean, split, THEN impute & encode.

Key fixes over the original pipeline:
  1. ID removal uses an explicit list, not substring matching on "account"
     (which was dropping legitimate credit features).
  2. ALL performance-period columns are flagged as leakage.
  3. Imputation and encoding happen AFTER train/val/test split
     so statistics (medians, dummy levels) are learned only from train.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_RAW, TARGET, RANDOM_STATE,
    ID_COLUMNS, LEAKAGE_COLUMNS,
    TEST_SIZE, VAL_SIZE,
)


# ── 1. Load ────────────────────────────────────────────────
def load_data(path=DATA_RAW):
    df = pd.read_excel(path)
    print(f"[load] shape = {df.shape}")
    return df


# ── 2. Clean column names ─────────────────────────────────
def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_", regex=False)
    )
    return df


# ── 3. Drop IDs (explicit list) ───────────────────────────
def drop_ids(df):
    to_drop = [c for c in ID_COLUMNS if c in df.columns]
    print(f"[drop_ids] removing {to_drop}")
    return df.drop(columns=to_drop)


# ── 4. Drop leakage (performance-period columns) ──────────
def drop_leakage(df):
    to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    print(f"[drop_leakage] removing {to_drop}")
    return df.drop(columns=to_drop)


# ── 5. Train / Val / Test split  ──────────────────────────
def split_data(df):
    """
    70 / 10 / 20 stratified split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp,
        random_state=RANDOM_STATE
    )
    for name, s in [("train", X_train), ("val", X_val), ("test", X_test)]:
        print(f"[split] {name:>5s}  {s.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 6. Impute (fit on train only) ─────────────────────────
def impute(X_train, X_val, X_test):
    """
    Numeric  → median (from train)
    Categorical → 'missing'
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

    medians = X_train[num_cols].median()

    for df in [X_train, X_val, X_test]:
        df[num_cols] = df[num_cols].fillna(medians)
        df[cat_cols] = df[cat_cols].fillna("missing")

    return X_train, X_val, X_test


# ── 7. Encode categoricals (fit on train only) ────────────
def encode(X_train, X_val, X_test):
    """
    One-hot encode using categories from TRAIN so val/test
    never introduce unseen dummies.
    """
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    if not cat_cols:
        return X_train, X_val, X_test

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val   = pd.get_dummies(X_val,   columns=cat_cols, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)

    # align columns to train
    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # convert any remaining bool columns
    for df in [X_train, X_val, X_test]:
        bool_cols = df.select_dtypes(include=["bool"]).columns
        df[bool_cols] = df[bool_cols].astype(int)

    return X_train, X_val, X_test


# ── 8. Quick leakage sanity check ─────────────────────────
def leakage_check(X_train, y_train, threshold=0.85):
    from sklearn.metrics import roc_auc_score
    warnings = []
    for col in X_train.columns:
        try:
            auc = roc_auc_score(y_train, X_train[col])
            if auc > threshold:
                warnings.append((col, auc))
        except Exception:
            pass
    if warnings:
        print("[leakage_check] SUSPICIOUS features:")
        for col, auc in sorted(warnings, key=lambda x: -x[1]):
            print(f"   {col}  AUC={auc:.4f}")
    else:
        print("[leakage_check] no single-feature AUC > {:.2f}".format(threshold))
    return warnings


# ── Public API ─────────────────────────────────────────────
def prepare_data():
    """
    Full prep pipeline.  Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    df = load_data()
    df = clean_columns(df)
    df = drop_ids(df)
    df = drop_leakage(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test = impute(X_train, X_val, X_test)
    X_train, X_val, X_test = encode(X_train, X_val, X_test)

    leakage_check(X_train, y_train)

    feature_names = X_train.columns.tolist()
    print(f"\n[prepare_data] {len(feature_names)} features ready")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


if __name__ == "__main__":
    prepare_data()
