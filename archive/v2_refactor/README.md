# Credit Risk Pipeline — Cleaned Build

A minimal, reproducible credit risk pipeline: ingest the DRA + simulated credit dataset, strip identifiers and outcome-window leakage, engineer a handful of features, split into train/val/test, and train two models — a calibrated logistic scorecard and a random-forest benchmark.

This folder (`archive/v2_refactor/`) is a corrected version of the original auto-generated code that used to live under `../../src/` and `../../pipelines/`. The folder shape is intentionally kept identical so a side-by-side `diff` between the two trees remains meaningful. A separate parallel rewrite lives in `../v1_initial/` and is out of scope here.

A full list of the defects that were fixed is in [`DIGEST.md`](./DIGEST.md).

## Project layout

```
archive/v2_refactor/
├── DIGEST.md                 # Defects in the earlier auto-generated build
├── README.md                 # This file
├── requirements.txt
├── data/
│   ├── raw/                  # DRA_with_simulated_credit.xlsx
│   └── processed/            # written by the pipeline
├── models/                   # fitted artefacts (if persisted)
├── reports/                  # metrics.json, score bands, CSV dumps
├── pipelines/
│   └── run_pipeline.py       # end-to-end entry point
└── src/
    ├── config.py             # single source of truth for column names
    ├── paths.py              # path resolution anchored to repo root
    ├── data/
    │   ├── ingest.py
    │   ├── preprocess.py
    │   └── split.py
    ├── features/
    │   └── features.py
    └── models/
        ├── evaluate.py
        ├── train_rf.py
        └── train_scorecard.py
```

## Run it

```bash
cd archive/v2_refactor
pip install -r requirements.txt
python -m pipelines.run_pipeline
```

Outputs land in:

- `data/processed/` — `X_train.csv`, `X_val.csv`, `X_test.csv` and the matching `y_*.csv`
- `reports/metrics.json` — scorecard and random-forest AUC / Gini / KS on the test set
- `reports/scorecard_band_summary.csv` — count, average score, average PD, bad rate per band A–E

## Configuration

Every column name the pipeline cares about is declared in one place. No substring matching, no magic strings scattered across files.

```python
# src/config.py
TARGET = "bad"

# Exact-match identifier columns to drop before modelling.
ID_COLS = ["dummy_id"]

# Post-outcome (performance window) columns. These look into the future and
# must be removed to avoid target leakage.
LEAKAGE_COLS = [
    "highest_arrears_perf",
    "num_accounts_perf",
    "age_oldest_perf",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10

RAW_FILENAME = "DRA_with_simulated_credit.xlsx"
```

The original pipeline used substring matching (`if "id" in col`) to drop identifiers, which silently stripped legitimate predictors such as `num_accounts_assess`. The leakage list was also incomplete — only `highest_arrears_perf` was being dropped, so `num_accounts_perf` and `age_oldest_perf` were leaking the outcome into the training set.

## Path resolution

```python
# src/paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
```

Anchoring every directory to the module file's location means `python -m pipelines.run_pipeline` behaves the same whether invoked from the project root, from a scheduler, or from a CI job. The original code used relative strings like `"data/raw/..."` and `"models/scaler.pkl"`, which silently broke whenever the working directory changed.

## Ingest

```python
# src/data/ingest.py
import pandas as pd
from src.paths import RAW_DIR
from src.config import RAW_FILENAME


def load_credit_data(filename: str = RAW_FILENAME) -> pd.DataFrame:
    file_path = RAW_DIR / filename
    print(f"Loading data from: {file_path}")
    df = pd.read_excel(file_path)
    print(f"Loaded shape: {df.shape}")
    return df
```

One responsibility: read the Excel file. The loader does not clean, rename, or split anything — those concerns live in their own modules so they can be unit-tested independently.

## Preprocessing

```python
# src/data/preprocess.py
import numpy as np
import pandas as pd

from src.config import ID_COLS, LEAKAGE_COLS, TARGET


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return df


def remove_ids(df):
    to_drop = [c for c in ID_COLS if c in df.columns]
    if to_drop:
        print(f"Dropping ID columns: {to_drop}")
    return df.drop(columns=to_drop)


def remove_leakage(df):
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if to_drop:
        print(f"Dropping leakage columns: {to_drop}")
    return df.drop(columns=to_drop)


def handle_missing(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if len(cat_cols):
        df[cat_cols] = df[cat_cols].fillna("missing")
    return df


def encode_categorical(df):
    df = pd.get_dummies(df, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)
    return df


def ensure_numeric(df):
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns survived preprocessing: {non_numeric}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    df = remove_ids(df)
    df = remove_leakage(df)
    df = handle_missing(df)
    df = encode_categorical(df)
    df = ensure_numeric(df)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing after preprocessing")
    print(f"Preprocessed shape: {df.shape}")
    return df
```

Three things to call out against the original version. First, `preprocess_data` now takes a `DataFrame` rather than silently re-loading from disk — that was the bug that broke every caller that tried `preprocess_data(df)`. Second, imputation is vectorised on the numeric and categorical blocks instead of a per-column Python loop. Third, `ensure_numeric` runs exactly once, at the end of the pipeline, rather than twice in two adjacent stanzas.

## Split

```python
# src/data/split.py
from sklearn.model_selection import train_test_split
from src.config import TARGET, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def split_data(
    df,
    target_col: str = TARGET,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp,
        random_state=random_state,
    )

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
```

Default `target_col` now points at the real target column (`"bad"`). The original defaulted to `"default"`, which did not exist in the dataset, so the split crashed whenever a caller did not pass it explicitly. The two-step split keeps `val_size` expressed relative to the full dataset, not relative to whatever is left after the test split.

## Feature engineering

```python
# src/features/features.py
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "total_risk_score" in df.columns:
        q_hi = df["total_risk_score"].quantile(0.70)
        q_lo = df["total_risk_score"].quantile(0.30)
        df["high_risk_flag"] = (df["total_risk_score"] > q_hi).astype(int)
        df["low_risk_flag"] = (df["total_risk_score"] < q_lo).astype(int)

    if {"risk_drivers", "risk_mitigators"}.issubset(df.columns):
        df["net_risk"] = df["risk_drivers"] - df["risk_mitigators"]
        df["risk_ratio"] = df["risk_drivers"] / (df["risk_mitigators"] + 1)

    if {"total_risk_score", "r_ho_em2_co"}.issubset(df.columns):
        df["risk_x_emotional"] = df["total_risk_score"] * df["r_ho_em2_co"]

    if {"r_ho_em2_co", "r_ho_vi4_st"}.issubset(df.columns):
        df["emotional_x_stability"] = df["r_ho_em2_co"] * df["r_ho_vi4_st"]

    return df
```

Every block is guarded on column existence. The original `features.py` referenced hard-coded lowercase column names and raised `KeyError` whenever the source column was missing or was spelled differently upstream.

## Evaluation helper

```python
# src/models/evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_gini_ks(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))
    return {"AUC": float(auc), "Gini": float(gini), "KS": ks}
```

One function, one job. The original `evaluate.py` mixed a PyTorch inference path with the sklearn metric calls — the torch side was dead code that nothing in the pipeline ever imported.

## Scorecard

```python
# src/models/train_scorecard.py
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.models.evaluate import auc_gini_ks


def probability_to_score(
    prob_default, base_score: int = 600, pdo: int = 50, base_odds: int = 20
):
    prob_default = np.clip(np.asarray(prob_default), 1e-6, 1 - 1e-6)
    odds = (1 - prob_default) / prob_default
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)


def train_scorecard_model(X_train, y_train, X_test, y_test):
    print("\n[Scorecard] training...")

    features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not features:
        raise ValueError("No numeric features available for scorecard")

    X_train = X_train[features].copy()
    X_test = X_test[features].copy()

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])
    model = CalibratedClassifierCV(base_pipe, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    metrics = auc_gini_ks(y_test, probs)
    print(f"[Scorecard] {metrics}")

    scores = probability_to_score(probs)
    scores_df = pd.DataFrame({
        "pd": probs, "score": scores, "target": np.asarray(y_test)
    })
    scores_df["band"] = pd.qcut(
        scores_df["score"], 5, labels=["E", "D", "C", "B", "A"],
        duplicates="drop",
    )

    summary = (
        scores_df.groupby("band", observed=False)
        .agg(
            count=("target", "count"),
            avg_score=("score", "mean"),
            avg_pd=("pd", "mean"),
            bad_rate=("target", "mean"),
        )
        .reset_index()
    )
    print("\n[Scorecard] band summary")
    print(summary.to_string(index=False))

    return model, scores_df, summary, metrics
```

Four changes matter here. The scorecard now uses every numeric feature rather than `features[:10]`, which had effectively turned feature selection into "whichever columns happen to come first out of `select_dtypes`". Scaling moved inside a `Pipeline` so the standardisation that CalibratedClassifierCV sees on each CV fold is identical to the one applied at inference. The `pd` parameter name in `build_score` was renamed `prob_default` so it no longer shadows `pandas as pd`. And `qcut` gets `duplicates="drop"` so tied score edges do not crash the banding.

## Random forest benchmark

```python
# src/models/train_rf.py
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE
from src.models.evaluate import auc_gini_ks


def train_random_forest(X_train, y_train, X_eval=None, y_eval=None):
    print("\n[Random Forest] training...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    if X_eval is not None and y_eval is not None:
        probs = model.predict_proba(X_eval)[:, 1]
        metrics = auc_gini_ks(y_eval, probs)
        print(f"[Random Forest] {metrics}")
        return model, metrics

    return model, None
```

The original `train_random_forest` reported metrics on the training set — it called `predict_proba(X_train)` after fitting on `X_train`. That is not a benchmark, it is an overfitting check with the answer already known. The rewrite takes an explicit evaluation set and uses production-sane hyperparameters (shallower trees, a minimum leaf size to stop the model from memorising rare combinations).

## End-to-end runner

```python
# pipelines/run_pipeline.py
import json
import pandas as pd

from src.config import TARGET
from src.paths import PROCESSED_DIR, REPORTS_DIR
from src.data.ingest import load_credit_data
from src.data.preprocess import preprocess_data
from src.data.split import split_data
from src.features.features import create_features
from src.models.train_scorecard import train_scorecard_model
from src.models.train_rf import train_random_forest


def _save_processed(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)


def run_pipeline() -> dict:
    print("===== CREDIT RISK PIPELINE =====")

    df = load_credit_data()
    df = preprocess_data(df)
    df = create_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    _save_processed(X_train, X_val, X_test, y_train, y_val, y_test)

    _, scores_df, band_summary, sc_metrics = train_scorecard_model(
        X_train, y_train, X_test, y_test
    )
    _, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    scores_df.to_csv(REPORTS_DIR / "scorecard_test_scores.csv", index=False)
    band_summary.to_csv(REPORTS_DIR / "scorecard_band_summary.csv", index=False)

    report = {"scorecard": sc_metrics, "random_forest": rf_metrics}
    with open(REPORTS_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n===== PIPELINE COMPLETE =====")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_pipeline()
```

Only one pipeline. The original repo shipped two divergent implementations — `pipelines/run_pipeline.py` with its own inline cleaning code, and `src/models/train_logistical.py` calling undefined symbols — which made reproducibility impossible because the two files cleaned the data differently. Everything here routes through `src/data/*` so there is exactly one code path from raw Excel to fitted model.

## Verification

- `python3 -m py_compile` across every module is clean.
- Ingest, preprocess and feature engineering were executed on the real Excel file (44,998 × 58). Result: 44,998 × 61, all numeric, no NaN, target preserved.
- A numpy-only logistic smoke test (no sklearn in the sandbox) produced test-set AUC = 0.6329. Plausible for this dataset once the three performance-window leakage columns are removed; the full scikit-learn calibrated scorecard should land slightly higher.
- Running the full scikit-learn pipeline requires `pip install -r requirements.txt`. On a machine with the dependencies installed, `python -m pipelines.run_pipeline` writes `reports/metrics.json`, `reports/scorecard_band_summary.csv`, and the six processed CSV splits.

## Comparison plan

This tree intentionally mirrors the original `src/` + `pipelines/` shape so the comparison against the separate rewrite under `../v1_initial/` can be done module-for-module. Suggested workflow:

1. Run the pipeline here, capture `reports/metrics.json` and `reports/scorecard_band_summary.csv`.
2. Run the parallel build under `../v1_initial/`, capture its metrics artefacts.
3. Compare AUC / Gini / KS on the same test split (same `RANDOM_STATE`, same leakage set) to isolate the effect of architectural differences from cleaning differences.
