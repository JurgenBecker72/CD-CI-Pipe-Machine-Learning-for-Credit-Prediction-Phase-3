# ============================================================
# RANDOM FOREST TRAINING WRAPPER
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def train_random_forest(X_train, y_train):

    print("\n===== TRAINING RANDOM FOREST =====")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    probs = model.predict_proba(X_train)[:, 1]

    # Metrics
    auc = roc_auc_score(y_train, probs)
    gini = 2 * auc - 1
    ks = ks_2samp(probs[y_train == 1], probs[y_train == 0]).statistic

    print(f"\nRF: AUC={auc:.4f} | Gini={gini:.4f} | KS={ks:.4f}")

    return model