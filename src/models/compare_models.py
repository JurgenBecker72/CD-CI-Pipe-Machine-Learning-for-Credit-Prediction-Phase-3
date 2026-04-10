import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False


# -------------------------------
# Evaluate function
# -------------------------------
def evaluate_auc(y_true, probs, name):
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1

    print(f"\n{name}:")
    print(f"AUC:  {auc:.4f}")
    print(f"Gini: {gini:.4f}")

    return auc


# -------------------------------
# Model comparison
# -------------------------------
def compare_models(X_train, y_train, X_test, y_test):

    results = {}

    # -------------------------------
    # Logistic Regression
    # -------------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    results["Logistic"] = evaluate_auc(y_test, lr_probs, "Logistic Regression")

    # -------------------------------
    # Random Forest
    # -------------------------------
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    results["RandomForest"] = evaluate_auc(y_test, rf_probs, "Random Forest")

    # -------------------------------
    # XGBoost (if installed)
    # -------------------------------
    if xgb_available:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]

        results["XGBoost"] = evaluate_auc(y_test, xgb_probs, "XGBoost")

    else:
        print("\nXGBoost not installed")

    return results