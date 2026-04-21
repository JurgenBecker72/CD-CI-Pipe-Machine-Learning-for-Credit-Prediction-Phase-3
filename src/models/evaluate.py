import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def get_feature_importance(X, y, top_n=20):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = pd.Series(model.coef_[0], index=X.columns)
    importance = coefs.abs().sort_values(ascending=False)

    print("\nTop predictive features:\n")
    print(importance.head(top_n))

    return importance


def calculate_auc_gini(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    gini = 2 * auc - 1

    print(f"\nAUC: {auc:.4f}")
    print(f"Gini: {gini:.4f}")

    return auc, gini


def calculate_ks(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ks = np.max(tpr - fpr)

    print(f"KS: {ks:.4f}")

    return ks


def evaluate_model(model, X_test, y_test):
    import torch

    X_test_tensor = torch.tensor(X_test.values.astype("float32"))

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()

    y_true = y_test.values

    print("\nModel Evaluation:\n")

    auc, gini = calculate_auc_gini(y_true, probs)
    ks = calculate_ks(y_true, probs)

    return {"AUC": auc, "Gini": gini, "KS": ks}
