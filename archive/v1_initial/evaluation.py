"""
evaluation.py – Metrics, calibration plots, and model comparison.

Produces:
  • AUC, Gini, KS per model (base vs calibrated)
  • Calibration plot (predicted PD vs observed default rate)
  • ROC curves
  • Summary CSV
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss,
    classification_report, confusion_matrix,
)
from sklearn.calibration import calibration_curve

from config import PLOT_DIR, REPORT_DIR, PD_BINS


# ── Core metrics ───────────────────────────────────────────
def compute_metrics(y_true, probs, label="model"):
    auc  = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, probs)
    ks   = np.max(tpr - fpr)
    brier = brier_score_loss(y_true, probs)
    print(f"  [{label}]  AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}  Brier={brier:.4f}")
    return {"model": label, "AUC": auc, "Gini": gini, "KS": ks, "Brier": brier}


# ── Calibration plot ───────────────────────────────────────
def plot_calibration(y_true, probs_dict, title="PD Calibration", fname="calibration.png"):
    """
    probs_dict = { "LR_base": probs_array, "LR_cal": probs_array, ... }
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, probs in probs_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=PD_BINS)
        ax.plot(mean_pred, frac_pos, "s-", label=name)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: histogram of predicted PDs
    ax2 = axes[1]
    for name, probs in probs_dict.items():
        ax2.hist(probs, bins=50, alpha=0.4, label=name)
    ax2.set_xlabel("Predicted PD")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of PD Scores")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname, dpi=150)
    plt.close()
    print(f"  [plot] saved {fname}")


# ── ROC curves ─────────────────────────────────────────────
def plot_roc(y_true, probs_dict, fname="roc_curves.png"):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves – Test Set")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname, dpi=150)
    plt.close()
    print(f"  [plot] saved {fname}")


# ── Evaluate all models ───────────────────────────────────
def evaluate_all(model_dict, X_test, y_test):
    """
    model_dict = { name: {"base": ..., "calibrated": ...} }
    Returns DataFrame of metrics and saves plots.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    probs_for_cal_plot = {}
    probs_for_roc = {}

    for name, models in model_dict.items():
        for variant in ["base", "calibrated"]:
            m = models[variant]
            probs = m.predict_proba(X_test)[:, 1]
            label = f"{name}_{variant}"
            row = compute_metrics(y_test, probs, label)
            rows.append(row)
            probs_for_roc[label] = probs
            if variant == "calibrated":
                probs_for_cal_plot[label] = probs

    # Also add base for calibration comparison
    for name, models in model_dict.items():
        probs_for_cal_plot[f"{name}_base"] = models["base"].predict_proba(X_test)[:, 1]

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    print(f"\n  [eval] metrics saved to model_metrics.csv")

    plot_calibration(y_test, probs_for_cal_plot)
    plot_roc(y_test, probs_for_roc)

    return df_metrics
