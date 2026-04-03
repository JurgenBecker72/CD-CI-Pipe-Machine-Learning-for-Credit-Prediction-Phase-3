# src/config.py

# =============================
# PROJECT CONFIGURATION
# =============================

# Target variable (what we are predicting)
TARGET = "bad"   # <-- update if your column name differs

# ID column (if available)
ID_COL = "customer_id"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.30

# Calibration method
CALIBRATION_METHOD = "sigmoid"

# =============================
# FEATURE CONFIGURATION
# =============================

# Psychometric (DRA) features
DRA_FEATURES = [
    "factor_1", "factor_2", "factor_3", "factor_4", "factor_5",
    "factor_6", "factor_7", "factor_8", "factor_9", "factor_10",
    "factor_11", "factor_12", "factor_13"
]

# Credit / behavioral features (adjust to your dataset)
CREDIT_FEATURES = [
    "accounts",
    "arrears",
    "age_on_book"
]

# Combine all features
BASE_FEATURES = DRA_FEATURES + CREDIT_FEATURES