"""
shap_explain.py – SHAP-based model explainability.

Produces:
  • Global summary (beeswarm) plot
  • Bar importance plot
  • Top-feature dependence plots
  • SHAP values CSV for downstream use
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import PLOT_DIR, REPORT_DIR


def explain_model(model, X_test, feature_names, model_name="best",
                  max_samples=2000):
    """
    Run TreeExplainer (RF/XGB) or LinearExplainer (LR) on the test set.
    Saves plots and SHAP values CSV.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test

    # Choose explainer
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classifiers RF returns list of 2 arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # class-1 (default)
    else:
        # Logistic / other – use KernelExplainer on a small background
        bg = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_raw = explainer.shap_values(X_sample)
        shap_values = shap_raw[1] if isinstance(shap_raw, list) else shap_raw

    # ── 1. Summary beeswarm ────────────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"shap_summary_{model_name}.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  [shap] saved shap_summary_{model_name}.png")

    # ── 2. Bar importance ──────────────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"shap_bar_{model_name}.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  [shap] saved shap_bar_{model_name}.png")

    # ── 3. Dependence plots for top 4 features ─────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:4]

    for i, idx in enumerate(top_idx):
        fname = feature_names[idx]
        plt.figure()
        shap.dependence_plot(idx, shap_values, X_sample,
                             feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"shap_dep_{model_name}_{i}_{fname}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [shap] saved 4 dependence plots")

    # ── 4. Save SHAP values ────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(REPORT_DIR / f"shap_values_{model_name}.csv", index=False)

    # Mean absolute SHAP importance
    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(REPORT_DIR / f"shap_importance_{model_name}.csv", index=False)
    print(f"  [shap] saved shap_values + importance CSV")

    return shap_values, imp
