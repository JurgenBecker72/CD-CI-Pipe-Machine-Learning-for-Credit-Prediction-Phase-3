# ============================================================
# RUN FULL CREDIT RISK PIPELINE
# ============================================================

# Data steps
from src.data.ingest import load_credit_data
from src.data.preprocess import preprocess_data
from src.data.split import split_data

# Models
from src.models.train_rf import train_random_forest

# Logistic PD model


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline():

    print("\n===== STARTING CREDIT RISK PIPELINE =====")

    # --------------------------------------------------------
    # 1. INGEST DATA
    # --------------------------------------------------------
    df = load_credit_data()
    print(f"Data loaded: {df.shape}")

    # --------------------------------------------------------
    # 2. PREPROCESS DATA
    # --------------------------------------------------------
    df = preprocess_data(df)
    print(f"Data after preprocessing: {df.shape}")

    # --------------------------------------------------------
    # 3. SPLIT DATA
    # --------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("\nData split complete:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # --------------------------------------------------------
    # 4. TRAIN RANDOM FOREST
    # --------------------------------------------------------
    train_random_forest(X_train, y_train)

    # --------------------------------------------------------
    # 5. SHAP ANALYSIS (use sample for speed)
    # --------------------------------------------------------
    # X_sample = X_test.sample(min(500, len(X_test)), random_state=42)

    # run_shap(model, X_sample)
    # run_shap_interactions(model, X_sample)

    # --------------------------------------------------------
    # 6. LOGISTIC PD MODEL (on full dataset)
    # --------------------------------------------------------
    # train_logistic(df)   # TODO: function not implemented yet

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------
    print("\n===== PIPELINE COMPLETE =====")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()
