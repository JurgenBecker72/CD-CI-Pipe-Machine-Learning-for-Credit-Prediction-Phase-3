# src/config.py

# =============================
# PROJECT CONFIGURATION
# =============================

# Target variable (what we are predicting)
TARGET = "bad"

# ID column (true identifier — not a feature)
ID_COL = "dummy_id"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.30

# Calibration method
CALIBRATION_METHOD = "sigmoid"

# =============================
# FEATURE CONFIGURATION
# =============================

# Psychometric (DRA) dimension scores
DRA_DIMENSIONS = [
    "dim_judgement",
    "dim_core_traits",
    "dim_emotional_understanding",
    "dim_principles",
]

# Psychometric item-level features (HO / SF blocks)
DRA_ITEMS = [
    "r_ho_tc1_ag", "r_ho_em2_co", "r_ho_tc3_di", "r_ho_tc4_so",
    "r_ho_vi1_cn", "r_ho_vi5_ss", "r_ho_em3_sp", "r_ho_vi3_ac",
    "r_ho_em1_es", "r_ho_tc2_oc", "r_ho_vi4_st", "r_ho_vi2_cv",
    "r_ho_jd_cm",
    "r_sf_em1_ad", "r_sf_em2_af", "r_sf_tc1_al", "r_sf_tc11_as",
    "r_sf_em3_bo", "r_sf_tc2_cm", "r_sf_em6_cb", "r_sf_em7_cp",
    "r_sf_tc4_du", "r_sf_tc3_ex", "r_sf_tc5_el", "r_sf_vi1_fv",
    "r_sf_vi7_he", "r_sf_em8_im", "r_sf_vi5_ma", "r_sf_em4_mi",
    "r_sf_tc6_or", "r_sf_tc7_pe", "r_sf_vi6_po", "r_sf_dm1_ps",
    "r_sf_tc8_pr", "r_sf_vi4_rn", "r_sf_tc12_re", "r_sf_em5_sc",
    "r_sf_tc9_sd", "r_sf_vi2_tv", "r_sf_tc10_wl", "r_sf_vi3_av",
    "r_sf_dm2_cs",
]

# Composite risk features (psychometric)
RISK_COMPOSITES = [
    "total_risk_score",
    "risk_mitigators",
    "risk_drivers",
]

# Credit bureau features captured at ASSESSMENT time (safe to use)
CREDIT_ASSESS_FEATURES = [
    "num_accounts_assess",
    "worst_arrears_assess",
    "age_oldest_assess",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "product_type",
]

# Combine all modelling-safe features
BASE_FEATURES = (
    DRA_DIMENSIONS
    + DRA_ITEMS
    + RISK_COMPOSITES
    + CREDIT_ASSESS_FEATURES
    + CATEGORICAL_FEATURES
)

# =============================
# LEAKAGE CONFIGURATION
# =============================

# Performance-window credit features — contain OUTCOME information.
# These MUST be dropped before modelling.
LEAKAGE_COLUMNS = [
    "num_accounts_perf",
    "highest_arrears_perf",
    "age_oldest_perf",
]

# Columns that are true identifiers (not features).
# Listed explicitly instead of substring matching to avoid dropping
# legitimate features like `num_accounts_assess`.
ID_COLUMNS = [
    "dummy_id",
]
