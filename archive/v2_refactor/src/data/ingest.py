# src/data/ingest.py
# Loads the raw Excel file. Path resolution is centralised in src.paths
# so the loader works regardless of where python is invoked from.

import pandas as pd
from src.paths import RAW_DIR
from src.config import RAW_FILENAME


def load_credit_data(filename: str = RAW_FILENAME) -> pd.DataFrame:
    file_path = RAW_DIR / filename
    print(f"Loading data from: {file_path}")
    df = pd.read_excel(file_path)
    print(f"Loaded shape: {df.shape}")
    return df
