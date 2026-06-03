"""Smoke tests for src/data/warehouse.py.

Runs against a temporary DuckDB file (not the real warehouse) so the test
suite stays hermetic. Confirms the module's public API does what the
docstrings claim.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import warehouse
from src.data.warehouse import SCHEMAS


@pytest.fixture
def temp_warehouse(tmp_path, monkeypatch):
    """Point settings.duckdb_path at a fresh per-test DuckDB file."""
    db_path = tmp_path / "test_credit.duckdb"
    monkeypatch.setattr(warehouse.settings, "duckdb_path", db_path)
    yield db_path
    # Connections are context-managed inside warehouse.py, so tmp_path's
    # cleanup handles file removal automatically.


def _make_sample_frame(n: int = 50_000) -> pd.DataFrame:
    """Tiny synthetic frame shaped like the real application table.

    DRA dimensions are drawn from a uniform distribution over [0, 100] to
    match the contract's expected range (a percentile-style 0-100 score).
    Sampling from `rng.normal(0, 1, ...)` puts roughly half the values
    below zero, which the data contract correctly rejects -- the fixture
    has to mirror the production data shape, not a generic normal.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "dummy_id": range(n),
            "bad": rng.choice([0, 1], size=n, p=[0.76, 0.24]),
            "dim_judgement": rng.uniform(0, 100, n),
            "dim_core_traits": rng.uniform(0, 100, n),
            "dim_emotional_understanding": rng.uniform(0, 100, n),
            "dim_principles": rng.uniform(0, 100, n),
            "num_accounts_assess": rng.integers(0, 10, n),
            "product_type": rng.choice(["A", "B", "C"], size=n),
        }
    )


def test_bootstrap_creates_three_schemas(temp_warehouse):
    """bootstrap() should leave a DB with raw, staging, marts."""
    warehouse.bootstrap()

    with warehouse.connect(read_only=True) as con:
        rows = con.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
    schema_names = {r[0] for r in rows}

    for s in SCHEMAS:
        assert s in schema_names, f"schema {s!r} missing after bootstrap()"


def test_bootstrap_is_idempotent(temp_warehouse):
    """Calling bootstrap() twice must not raise."""
    warehouse.bootstrap()
    warehouse.bootstrap()  # should be a no-op


def test_promote_to_staging_writes_table(temp_warehouse, tmp_path, monkeypatch):
    """End-to-end: ingest a sample frame, promote, count rows."""
    # Stage a tiny Excel file the ingest step can read.
    sample = _make_sample_frame()
    excel_path = tmp_path / "sample.xlsx"
    sample.to_excel(excel_path, index=False)

    warehouse.bootstrap()
    raw_n = warehouse.ingest_excel(excel_path)
    assert raw_n == len(sample)

    staging_n = warehouse.promote_to_staging()
    assert staging_n == len(sample)


def test_build_marts_drops_id_and_leakage_columns(temp_warehouse, tmp_path):
    """marts.applicant_features must exclude ID and leakage columns."""
    sample = _make_sample_frame()
    # Inject leakage columns to confirm they get stripped.
    sample["num_accounts_perf"] = 0
    sample["highest_arrears_perf"] = 0
    sample["age_oldest_perf"] = 0

    excel_path = tmp_path / "sample.xlsx"
    sample.to_excel(excel_path, index=False)

    warehouse.bootstrap()
    warehouse.ingest_excel(excel_path)
    warehouse.promote_to_staging()
    warehouse.build_marts()

    marts_df = warehouse.read_marts()
    for col in ("dummy_id", "num_accounts_perf", "highest_arrears_perf", "age_oldest_perf"):
        assert col not in marts_df.columns, f"{col} should be excluded from marts view"

    # Target must survive the projection.
    assert "bad" in marts_df.columns


def test_refresh_returns_row_counts(temp_warehouse, tmp_path):
    """refresh() one-shot must return row counts at each layer."""
    sample = _make_sample_frame(n=45_000)
    excel_path = tmp_path / "sample.xlsx"
    sample.to_excel(excel_path, index=False)

    counts = warehouse.refresh(source=excel_path)

    assert counts["raw"] == 45_000
    assert counts["staging"] == 45_000
    assert counts["marts"] == 45_000


def test_ingest_raises_on_missing_file(temp_warehouse, tmp_path):
    """Friendly error if the source Excel doesn't exist."""
    warehouse.bootstrap()
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        warehouse.ingest_excel(tmp_path / "does_not_exist.xlsx")
