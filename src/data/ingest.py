# src/data/ingest.py

import pandas as pd
from src.paths import RAW_DIR


def load_credit_data(filename: str = "DRA_with_simulated_credit.xlsx") -> pd.DataFrame:
    """
    Load raw credit dataset from the data/raw directory.
    """

    file_path = RAW_DIR / filename

    print(f"📂 Loading data from: {file_path}")

    df = pd.read_excel(file_path)

    print(f"✅ Data loaded successfully. Shape: {df.shape}")

    return df
