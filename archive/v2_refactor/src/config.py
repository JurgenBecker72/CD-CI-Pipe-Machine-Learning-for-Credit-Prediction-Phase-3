# src/config.py
# Single source of truth for configuration.
# Everything downstream imports from here so names never drift.

TARGET = "bad"

# Identifier columns: dropped before modelling but NOT treated as features.
# Use exact names, not substring patterns. Substring matching caused the
# original pipeline to strip legitimate columns like `num_accounts_assess`.
ID_COLS = ["dummy_id"]

# Post-outcome (performance window) variables. These know the future and
# must be removed to avoid target leakage.
LEAKAGE_COLS = [
    "highest_arrears_perf",
    "num_accounts_perf",
    "age_oldest_perf",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # of the full dataset

# Source file (lives in data/raw relative to project root)
RAW_FILENAME = "DRA_with_simulated_credit.xlsx"
