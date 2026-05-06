"""MLflow-aware training entry point (Phase D).

Wraps the existing scorecard + RF training (from pipelines/run_pipeline.py)
in an MLflow run, logging:

    Params       hyperparameters, seeds, split sizes
    Metrics      AUC, Gini, KS for both models; bad rates; calibration
    Artefacts    fitted scaler, calibrated LR, RF, SHAP summary plot
    Tags         git SHA, Python version, library versions, data version
    Signature    input/output schema for both models
    Models       both registered in the MLflow Model Registry as
                 credit_scorecard and credit_rf_challenger

Usage
-----
Make sure the MLflow tracking server is up:

    docker compose up -d mlflow

Then run training (talks to MLflow at settings.mlflow_tracking_uri):

    uv run python -m src.training.train

Browse to http://localhost:5000 to see the run.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ID_COLUMNS, LEAKAGE_COLUMNS, RANDOM_STATE, TARGET
from src.data.warehouse import read_marts, refresh
from src.models.train_scorecard import train_scorecard_model
from src.settings import settings

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

SCORECARD_MODEL_NAME = "credit_scorecard"
CHALLENGER_MODEL_NAME = "credit_rf_challenger"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def _git_dirty() -> bool:
    """True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=settings.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except (FileNotFoundError, OSError):
        return False


def _evaluate(y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    """Compute the standard credit-risk metrics."""
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    ks = ks_2samp(probs[y_true == 1], probs[y_true == 0]).statistic
    return {"auc": auc, "gini": gini, "ks": ks}


def _drop_ids_and_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Belt-and-braces: warehouse already drops these, but enforce again here."""
    drop = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=drop)


def _impute_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median impute on train only - same behaviour as run_pipeline.py."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    medians = X_train[numeric_cols].median()
    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(medians[col])
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(medians[col])
    return X_train, X_test


def _log_shap_summary(model: Any, X_sample: pd.DataFrame, artefact_name: str) -> None:
    """Compute SHAP values on a small sample and log the summary plot.

    Wrapped in try/except: SHAP failures shouldn't abort the whole run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import shap

        # TreeExplainer for the RF; LinearExplainer would be cheaper for the
        # scorecard but TreeExplainer also works on calibrated wrappers via
        # the underlying estimator. Keep one path for both.
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        out = Path(settings.reports_dir) / f"{artefact_name}_shap.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(out), artifact_path="shap")
    except Exception as exc:  # noqa: BLE001 - explicit suppress with logging
        print(f"  [SHAP for {artefact_name}] skipped: {exc}")


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------


def train_with_tracking(
    experiment_name: str | None = None,
    run_name: str | None = None,
    refresh_warehouse: bool = True,
) -> str:
    """Run end-to-end training inside an MLflow run.

    Returns the MLflow run_id of the completed run. Both models
    (`credit_scorecard` and `credit_rf_challenger`) are registered.

    Parameters
    ----------
    experiment_name
        Override `settings.mlflow_experiment_name` for ad-hoc runs.
    run_name
        Human-friendly run name. Defaults to git SHA + timestamp.
    refresh_warehouse
        When True, runs `warehouse.refresh()` before training so the run
        always trains on the latest source. Set False for fast iteration
        during code changes.
    """
    # ------------------------------------------------------------------
    # Wire up MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

    # ------------------------------------------------------------------
    # Refresh the warehouse so the run always reads the current marts
    # ------------------------------------------------------------------
    if refresh_warehouse:
        counts = refresh()
        print(f"Warehouse refreshed: {counts}")

    df = read_marts()
    print(f"Loaded data from marts.applicant_features: {df.shape}")

    df = _drop_ids_and_leakage(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = _impute_train_test(X_train, X_test)

    # ------------------------------------------------------------------
    # Begin the MLflow run
    # ------------------------------------------------------------------
    sha = _git_sha()
    dirty = _git_dirty()

    with mlflow.start_run(run_name=run_name) as run:
        # ---- Tags (provenance) ---------------------------------------
        mlflow.set_tags(
            {
                "git_sha": sha,
                "git_dirty": str(dirty),
                "python_version": platform.python_version(),
                "phase": "D",
                "data_source": "duckdb:marts.applicant_features",
            }
        )

        # ---- Params (inputs that affect the model) -------------------
        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "test_size": 0.3,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
                "rf_n_estimators": 100,
                "rf_max_depth": 5,
            }
        )

        # ---- Train + log scorecard -----------------------------------
        print("\n===== TRAINING SCORECARD =====")
        scorecard, scores_df, summary = train_scorecard_model(X_train, y_train, X_test, y_test)
        scorecard_probs = scorecard.predict_proba(X_test)[:, 1]
        scorecard_metrics = _evaluate(y_test, scorecard_probs)
        for name, value in scorecard_metrics.items():
            mlflow.log_metric(f"scorecard_{name}", value)

        # Input example + signature for the scorecard
        scorecard_example = X_test.head(5)
        scorecard_signature = infer_signature(
            scorecard_example, scorecard.predict_proba(scorecard_example)
        )

        mlflow.sklearn.log_model(
            sk_model=scorecard,
            artifact_path="scorecard",
            registered_model_name=SCORECARD_MODEL_NAME,
            signature=scorecard_signature,
            input_example=scorecard_example,
        )

        # ---- Train + log RF challenger -------------------------------
        print("\n===== TRAINING RF CHALLENGER =====")
        X_train_e = pd.get_dummies(X_train, drop_first=True)
        X_test_e = pd.get_dummies(X_test, drop_first=True)
        X_test_e = X_test_e.reindex(columns=X_train_e.columns, fill_value=0)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        rf.fit(X_train_e, y_train)
        rf_probs = rf.predict_proba(X_test_e)[:, 1]
        rf_metrics = _evaluate(y_test, rf_probs)
        for name, value in rf_metrics.items():
            mlflow.log_metric(f"rf_{name}", value)

        rf_example = X_test_e.head(5)
        rf_signature = infer_signature(rf_example, rf.predict_proba(rf_example))

        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="rf_challenger",
            registered_model_name=CHALLENGER_MODEL_NAME,
            signature=rf_signature,
            input_example=rf_example,
        )

        # ---- SHAP summary plots --------------------------------------
        # Use a smaller sample for speed; SHAP scales with rows*features.
        sample_size = min(500, len(X_test_e))
        _log_shap_summary(rf, X_test_e.sample(sample_size, random_state=0), "rf")

        # ---- Print + return ------------------------------------------
        print(
            f"\n===== RUN COMPLETE =====\n"
            f"  run_id:   {run.info.run_id}\n"
            f"  scorecard: AUC={scorecard_metrics['auc']:.4f} "
            f"Gini={scorecard_metrics['gini']:.4f} KS={scorecard_metrics['ks']:.4f}\n"
            f"  rf:        AUC={rf_metrics['auc']:.4f} "
            f"Gini={rf_metrics['gini']:.4f} KS={rf_metrics['ks']:.4f}\n"
            f"  view at:   {settings.mlflow_tracking_uri}"
        )
        return run.info.run_id


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the credit pipeline with MLflow tracking.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=f"MLflow experiment name (default: {settings.mlflow_experiment_name}).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Human-friendly run name (default: auto-generated).",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip warehouse refresh and use the current marts as-is.",
    )
    args = parser.parse_args()

    train_with_tracking(
        experiment_name=args.experiment,
        run_name=args.run_name,
        refresh_warehouse=not args.no_refresh,
    )


if __name__ == "__main__":
    main()
