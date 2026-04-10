"""
config.py – Single source of truth for the credit-risk pipeline.
All magic numbers, column lists and paths live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT.parent / "data" / "raw" / "DRA_with_simulated_credit.xlsx"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
MODEL_DIR = OUT_DIR / "models"
REPORT_DIR = OUT_DIR / "reports"

# ── Target ─────────────────────────────────────────────────
TARGET = "bad"

# ── Reproducibility ────────────────────────────────────────
RANDOM_STATE = 42

# ── Columns to drop ───────────────────────────────────────
# TRUE identifiers only – never substring-match on "account"
ID_COLUMNS = ["dummy_id"]

# Performance-period columns leak future information
LEAKAGE_COLUMNS = [
    "num_accounts_perf",
    "highest_arrears_perf",
    "age_oldest_perf",
]

# ── Assessment-period credit features (safe) ──────────────
CREDIT_FEATURES = [
    "num_accounts_assess",
    "worst_arrears_assess",
    "age_oldest_assess",
]

# ── Split ratios ──────────────────────────────────────────
TEST_SIZE = 0.20
VAL_SIZE = 0.10   # carved from the remaining 80 %

# ── Model hyper-parameters ────────────────────────────────
RF_PARAMS = dict(n_estimators=300, max_depth=6, min_samples_leaf=20,
                 random_state=RANDOM_STATE, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=1, reg_lambda=1,
                  eval_metric="logloss", random_state=RANDOM_STATE)
LR_PARAMS = dict(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)

# ── Scorecard ─────────────────────────────────────────────
SCORECARD_BASE_SCORE = 600
SCORECARD_PDO = 20         # points to double the odds
SCORECARD_BASE_ODDS = 50   # odds at base score (50:1 = ~2% PD)

# ── PD Calibration ────────────────────────────────────────
CALIBRATION_METHOD = "isotonic"   # "sigmoid" or "isotonic"
PD_BINS = 10                      # for calibration-plot buckets
