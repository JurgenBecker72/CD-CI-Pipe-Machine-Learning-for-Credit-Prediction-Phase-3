# src/data/preprocess.py
# Preprocessing for the credit risk pipeline.
#
# Fixes applied versus the earlier auto-generated version:
#   1. Accepts an in-memory DataFrame (no hidden re-loading from disk).
#   2. Uses exact ID/leakage column lists from config — no substring matching.
#   3. Vectorised missing-value imputation (no per-column Python loop).
#   4. Runs the numeric safety check exactly once.
#   5. No hard-coded output paths; callers decide what to do with the frame.

import numpy as np
import pandas as pd

from src.config import ID_COLS, LEAKAGE_COLS, TARGET


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return df


def remove_ids(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in ID_COLS if c in df.columns]
    if to_drop:
        print(f"Dropping ID columns: {to_drop}")
    return df.drop(columns=to_drop)


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if to_drop:
        print(f"Dropping leakage columns: {to_drop}")
    return df.drop(columns=to_drop)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if len(cat_cols):
        df[cat_cols] = df[cat_cols].fillna("missing")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encode everything that isn't numeric, then force bools to ints
    # so downstream sklearn/xgboost never chokes on dtype=object.
    df = pd.get_dummies(df, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)
    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns survived preprocessing: {non_numeric}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing: clean -> drop IDs -> drop leakage ->
    impute -> encode -> numeric safety check.
    Target column is preserved untouched.
    """
    df = clean_column_names(df)
    df = remove_ids(df)
    df = remove_leakage(df)
    df = handle_missing(df)
    df = encode_categorical(df)
    df = ensure_numeric(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing after preprocessing")

    print(f"Preprocessed shape: {df.shape}")
    return df
