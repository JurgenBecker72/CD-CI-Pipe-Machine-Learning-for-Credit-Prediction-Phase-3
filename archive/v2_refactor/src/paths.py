# src/paths.py
# Central path resolution so scripts can run from any working directory.

from pathlib import Path

# ROOT = the archive/v2_refactor/ folder (two levels up from this file).
# src/paths.py lives at {ROOT}/src/paths.py -> parents[1] is {ROOT}
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
