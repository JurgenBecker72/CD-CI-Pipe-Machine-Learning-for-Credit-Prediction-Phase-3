"""Smoke tests for src/training/train.py.

Runs against an ephemeral MLflow tracking URI (a tmp folder) so the test
suite stays hermetic. Confirms the training entry point completes, both
models register, and run metadata is populated.

Marked `integration` because it does real training on a small synthetic
sample. Excluded from the default `pytest -m "not integration"` sweep;
run explicitly with `pytest -m integration` or remove the marker if you
want it in the default set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")  # only run if mlflow extras installed


@pytest.fixture
def ephemeral_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a tmp dir for tracking + artefacts."""
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    # Importing settings AFTER setting env var so it picks up the override
    from src.settings import settings

    monkeypatch.setattr(settings, "mlflow_tracking_uri", tracking_uri)
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test-credit-scorecard")
    yield tracking_uri


def _make_warehouse_with_sample(tmp_path, monkeypatch, n: int = 5_000) -> Path:
    """Build a minimal warehouse so train_with_tracking has data to read."""
    import numpy as np

    from src.data import warehouse
    from src.settings import settings

    # Point the warehouse at a tmp DuckDB
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setattr(settings, "duckdb_path", db_path)
    monkeypatch.setattr(warehouse.settings, "duckdb_path", db_path)

    # Build a sample frame and stage it as Excel
    rng = np.random.default_rng(0)
    sample = pd.DataFrame(
        {
            "dummy_id": range(n),
            "bad": rng.choice([0, 1], size=n, p=[0.76, 0.24]),
            "dim_judgement": rng.normal(0, 1, n),
            "dim_core_traits": rng.normal(0, 1, n),
            "dim_emotional_understanding": rng.normal(0, 1, n),
            "dim_principles": rng.normal(0, 1, n),
            "num_accounts_assess": rng.integers(0, 10, n),
            "product_type": rng.choice(["A", "B", "C"], size=n),
        }
    )
    excel_path = tmp_path / "sample.xlsx"
    sample.to_excel(excel_path, index=False)

    # Refresh warehouse from this sample
    warehouse.refresh(source=excel_path)
    return excel_path


@pytest.mark.integration
def test_train_with_tracking_returns_run_id(ephemeral_mlflow, tmp_path, monkeypatch):
    """End-to-end: train_with_tracking must complete and return a run_id."""
    _make_warehouse_with_sample(tmp_path, monkeypatch, n=5_000)

    from src.training.train import train_with_tracking

    run_id = train_with_tracking(
        experiment_name="test-experiment",
        run_name="test-run",
        refresh_warehouse=False,  # we built it above
    )

    assert isinstance(run_id, str)
    assert len(run_id) > 0


@pytest.mark.integration
def test_run_logs_metrics_and_params(ephemeral_mlflow, tmp_path, monkeypatch):
    """Inspect the completed run; both scorecard and RF metrics must be logged."""
    _make_warehouse_with_sample(tmp_path, monkeypatch, n=5_000)

    from src.training.train import train_with_tracking

    run_id = train_with_tracking(experiment_name="test-experiment", refresh_warehouse=False)

    client = mlflow.tracking.MlflowClient(tracking_uri=ephemeral_mlflow)
    run = client.get_run(run_id)

    # All four metrics must be present
    for name in ("scorecard_auc", "scorecard_gini", "scorecard_ks", "rf_auc", "rf_gini", "rf_ks"):
        assert name in run.data.metrics, f"metric {name!r} missing from run"
        assert 0.0 <= run.data.metrics[name] <= 1.0

    # Params must include random_state
    assert run.data.params.get("random_state") == "42"

    # Tags must include git_sha (even if 'unknown')
    assert "git_sha" in run.data.tags


@pytest.mark.integration
def test_both_models_registered(ephemeral_mlflow, tmp_path, monkeypatch):
    """Both scorecard and RF challenger must appear in the registry."""
    _make_warehouse_with_sample(tmp_path, monkeypatch, n=5_000)

    from src.training.train import (
        CHALLENGER_MODEL_NAME,
        SCORECARD_MODEL_NAME,
        train_with_tracking,
    )

    train_with_tracking(experiment_name="test-experiment", refresh_warehouse=False)

    client = mlflow.tracking.MlflowClient(tracking_uri=ephemeral_mlflow)
    registered = {m.name for m in client.search_registered_models()}

    assert SCORECARD_MODEL_NAME in registered
    assert CHALLENGER_MODEL_NAME in registered
