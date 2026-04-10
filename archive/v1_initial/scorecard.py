"""
scorecard.py – WoE/IV calculation, scorecard points, and decision logic.

Converts a calibrated logistic regression into a traditional credit scorecard
with interpretable point allocations per feature bin.

Decision regions:
  - APPROVE   (score >= approve_cutoff)
  - REFER     (decline_cutoff <= score < approve_cutoff)
  - DECLINE   (score < decline_cutoff)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    SCORECARD_BASE_SCORE, SCORECARD_PDO, SCORECARD_BASE_ODDS,
    PLOT_DIR, REPORT_DIR,
)


# ── WoE / IV ──────────────────────────────────────────────
def woe_iv_table(X_train, y_train, feature, bins=10):
    """
    Compute Weight-of-Evidence and Information Value for one feature.
    Returns a summary DataFrame.
    """
    df = pd.DataFrame({"feature": X_train[feature], "target": y_train.values})

    # try quantile bins; fall back to equal-width for low-cardinality
    try:
        df["bin"] = pd.qcut(df["feature"], q=bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["feature"], bins=min(bins, df["feature"].nunique()))

    grouped = df.groupby("bin", observed=False)["target"]
    stats = grouped.agg(["count", "sum"]).rename(columns={"count": "total", "sum": "bads"})
    stats["goods"] = stats["total"] - stats["bads"]

    # Smooth to avoid log(0)
    total_goods = stats["goods"].sum()
    total_bads  = stats["bads"].sum()

    stats["dist_good"] = (stats["goods"] + 0.5) / (total_goods + 0.5 * len(stats))
    stats["dist_bad"]  = (stats["bads"]  + 0.5) / (total_bads  + 0.5 * len(stats))
    stats["woe"]       = np.log(stats["dist_good"] / stats["dist_bad"])
    stats["iv_bin"]    = (stats["dist_good"] - stats["dist_bad"]) * stats["woe"]

    stats["iv_total"] = stats["iv_bin"].sum()
    stats["feature"]  = feature

    return stats.reset_index()


def compute_iv_all(X_train, y_train, bins=10):
    """IV for every numeric feature. Returns sorted DataFrame."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    rows = []
    for col in num_cols:
        tbl = woe_iv_table(X_train, y_train, col, bins)
        iv = tbl["iv_total"].iloc[0]
        rows.append({"feature": col, "IV": iv})
    iv_df = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)
    return iv_df


# ── Scorecard points ──────────────────────────────────────
def build_scorecard(lr_model, feature_names, scaler=None):
    """
    Convert a fitted Logistic Regression into scorecard points.

    Formula (standard):
      factor = PDO / ln(2)
      offset = base_score - factor * ln(base_odds)
      points_i = -(coef_i * (x_i - mean_i) / std_i) * factor / n_features
                 + offset / n_features

    For simplicity we report the contribution per unit of scaled feature.
    """
    factor = SCORECARD_PDO / np.log(2)
    offset = SCORECARD_BASE_SCORE - factor * np.log(SCORECARD_BASE_ODDS)

    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    n = len(coefs)

    rows = []
    for fname, c in zip(feature_names, coefs):
        pts = -c * factor
        rows.append({"feature": fname, "coefficient": c, "points_per_unit": round(pts, 2)})

    df = pd.DataFrame(rows).sort_values("points_per_unit", ascending=True)
    df["base_score_share"] = round(offset / n, 2)

    return df, offset, factor


# ── Score a dataset ────────────────────────────────────────
def score_dataset(lr_model, X_scaled, feature_names, factor, offset):
    """
    Returns array of integer scorecard scores.
    """
    log_odds = X_scaled @ lr_model.coef_[0] + lr_model.intercept_[0]
    scores = offset - factor * log_odds
    return np.round(scores).astype(int)


# ── Decision regions ───────────────────────────────────────
def assign_decisions(scores, approve_cutoff=620, decline_cutoff=560):
    """
    Maps scores to APPROVE / REFER / DECLINE.
    Returns a Series of decision labels.
    """
    decisions = pd.Series("REFER", index=range(len(scores)))
    decisions[scores >= approve_cutoff] = "APPROVE"
    decisions[scores < decline_cutoff]  = "DECLINE"
    return decisions


# ── Plots ──────────────────────────────────────────────────
def plot_score_distribution(scores, y_true, fname="score_distribution.png"):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[y_true == 0], bins=50, alpha=0.6, label="Good", color="steelblue")
    ax.hist(scores[y_true == 1], bins=50, alpha=0.6, label="Bad",  color="tomato")
    ax.axvline(620, color="green", ls="--", label="Approve cutoff (620)")
    ax.axvline(560, color="red",   ls="--", label="Decline cutoff (560)")
    ax.set_xlabel("Scorecard Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Outcome")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname, dpi=150)
    plt.close()
    print(f"  [plot] saved {fname}")


def plot_iv_chart(iv_df, top_n=20, fname="iv_chart.png"):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    top = iv_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d32f2f" if v > 0.3 else "#ff9800" if v > 0.1 else "#4caf50" for v in top["IV"]]
    ax.barh(top["feature"], top["IV"], color=colors)
    ax.set_xlabel("Information Value")
    ax.set_title(f"Top {top_n} Features by IV")
    ax.axvline(0.1, ls="--", color="gray", alpha=0.5, label="Weak (0.02-0.1)")
    ax.axvline(0.3, ls="--", color="orange", alpha=0.5, label="Medium (0.1-0.3)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname, dpi=150)
    plt.close()
    print(f"  [plot] saved {fname}")


# ── Full scorecard pipeline ───────────────────────────────
def run_scorecard(lr_model, X_test_scaled, y_test, feature_names, X_train, y_train, scaler=None):
    """
    End-to-end: build scorecard → score test → decision → persist.
    Returns scorecard_df, scores, decisions, iv_df.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # IV analysis
    print("\n[scorecard] Computing IV...")
    iv_df = compute_iv_all(X_train, y_train)
    iv_df.to_csv(REPORT_DIR / "iv_table.csv", index=False)
    plot_iv_chart(iv_df)

    # Build scorecard
    print("[scorecard] Building point allocation...")
    sc_df, offset, factor = build_scorecard(lr_model, feature_names, scaler)
    sc_df.to_csv(REPORT_DIR / "scorecard_points.csv", index=False)

    # Score test set
    scores = score_dataset(lr_model, X_test_scaled, feature_names, factor, offset)
    decisions = assign_decisions(scores)

    print(f"\n[scorecard] Decision distribution:")
    print(decisions.value_counts().to_string())

    # Plots
    plot_score_distribution(scores, y_test.values)

    # Save scored test set
    scored = pd.DataFrame({
        "score": scores,
        "decision": decisions,
        "actual_bad": y_test.values,
    })
    scored.to_csv(REPORT_DIR / "scored_test_set.csv", index=False)

    return sc_df, scores, decisions, iv_df
