#!/usr/bin/env python3
"""
run_pipeline.py – Master orchestrator for the credit-risk pipeline.

Usage:
    python run_pipeline.py

Outputs land in  archive/v1_initial/outputs/{plots, models, reports}/.
"""

import sys
import warnings
import time

warnings.filterwarnings("ignore")

# ── ensure archive/v1_initial/ is on the path ──────────────
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PLOT_DIR, MODEL_DIR, REPORT_DIR
from data_prep import prepare_data
from features import engineer_features
from models import train_all
from evaluation import evaluate_all
from scorecard import run_scorecard
from shap_explain import explain_model


def main():
    t0 = time.time()

    # ── 1. Data preparation ────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 : DATA PREPARATION")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = prepare_data()

    # ── 2. Feature engineering ─────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 : FEATURE ENGINEERING")
    print("=" * 60)
    X_train, X_val, X_test = engineer_features(X_train, X_val, X_test)
    feat_names = X_train.columns.tolist()

    # ── 3. Model training + PD calibration ─────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 : MODEL TRAINING + PD CALIBRATION")
    print("=" * 60)
    model_dict, scaler, X_tr_s, X_val_s, X_te_s = train_all(
        X_train, X_val, X_test, y_train, y_val
    )

    # ── 4. Evaluation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 : EVALUATION")
    print("=" * 60)
    metrics_df = evaluate_all(model_dict, X_te_s, y_test)
    print("\n--- Model comparison ---")
    print(metrics_df.to_string(index=False))

    # ── 5. Scorecard (from calibrated LR) ──────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 : SCORECARD + DECISION LOGIC")
    print("=" * 60)
    if "Logistic" in model_dict:
        lr_base = model_dict["Logistic"]["base"]
        sc_df, scores, decisions, iv_df = run_scorecard(
            lr_base, X_te_s, y_test, feat_names,
            X_train, y_train, scaler
        )
    else:
        print("  [skip] Logistic model not available for scorecard.")

    # ── 6. SHAP on best tree model ─────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 : SHAP EXPLAINABILITY")
    print("=" * 60)
    # Prefer XGBoost, fall back to RF
    if "XGBoost" in model_dict:
        shap_model = model_dict["XGBoost"]["base"]
        shap_name = "XGBoost"
    elif "RandomForest" in model_dict:
        shap_model = model_dict["RandomForest"]["base"]
        shap_name = "RandomForest"
    else:
        shap_model = None

    if shap_model:
        explain_model(shap_model, X_te_s, feat_names, model_name=shap_name)
    else:
        print("  [skip] No tree model available for SHAP.")

    # ── Done ───────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  Outputs → {PLOT_DIR.relative_to(Path(__file__).resolve().parent)}")
    print(f"          → {MODEL_DIR.relative_to(Path(__file__).resolve().parent)}")
    print(f"          → {REPORT_DIR.relative_to(Path(__file__).resolve().parent)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
