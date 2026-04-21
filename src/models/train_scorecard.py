import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def evaluate(y_true, probs):
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    ks = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
    return auc, gini, ks


def build_score(pd, base_score=600, pdo=50, base_odds=20):
    pd = np.clip(pd, 1e-6, 1 - 1e-6)
    odds = (1 - pd) / pd
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)


def train_scorecard_model(X_train, y_train, X_test, y_test):

    print("\n===== TRAINING FINAL SCORECARD =====")

    # SAFE feature selection
    features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in features if f != "bad"]

    if len(features) == 0:
        raise ValueError("No numeric features available")

    features = features[:10]

    print("\nUsing features:", features)

    X_train = X_train[features].copy()
    X_test = X_test[features].copy()

    # Model
    base_model = LogisticRegression(max_iter=1000)

    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc, gini, ks = evaluate(y_test, probs)

    print(f"AUC: {auc:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"KS: {ks:.4f}")

    # Score
    scores = build_score(probs)

    df_scores = pd.DataFrame({"pd": probs, "score": scores, "target": y_test.values})

    df_scores["band"] = pd.qcut(
        df_scores["score"], 5, labels=["E", "D", "C", "B", "A"], duplicates="drop"
    )

    summary = df_scores.groupby("band", observed=False).agg(
        count=("target", "count"),
        avg_score=("score", "mean"),
        avg_pd=("pd", "mean"),
        bad_rate=("target", "mean"),
    )

    print("\n===== SCORE BAND SUMMARY =====")
    print(summary)

    return model, df_scores, summary
