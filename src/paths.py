# src/paths.py

from pathlib import Path

# Root of the project (Phase-3 folder)
ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Models
MODELS_DIR = ROOT / "models"
TRAINED_DIR = MODELS_DIR / "trained"
CALIBRATED_DIR = MODELS_DIR / "calibrated"

# Reports
REPORTS_DIR = ROOT / "reports"
