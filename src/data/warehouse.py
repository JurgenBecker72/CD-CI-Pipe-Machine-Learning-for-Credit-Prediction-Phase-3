"""DuckDB warehouse for the credit pipeline.

This module is the *only* place in the project that talks to DuckDB. Everything
else goes through these helpers, so when we eventually swap DuckDB for
Snowflake (Phase C.5) only this file changes.

Three logical schemas — the lakehouse / medallion pattern:

    raw       Landing zone for vendor files; never modified after load
    staging   Typed, deduplicated, lightly cleaned; gated by data contract
    marts     Wide, model-ready table consumed by training (a VIEW)

Usage
-----
    from src.data.warehouse import bootstrap, ingest_excel, promote_to_staging
    from src.data.warehouse import build_marts, read_sql

    bootstrap()                                    # creates schemas
    ingest_excel(settings.raw_dataset_path)        # raw.application
    promote_to_staging()                           # validates + INSERTs into staging.application
    build_marts()                                  # CREATE OR REPLACE VIEW marts.applicant_features
    df = read_sql("SELECT * FROM marts.applicant_features")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from src.config import ID_COLUMNS, LEAKAGE_COLUMNS, TARGET
from src.settings import settings

# ----------------------------------------------------------------------------
# Connection
# ----------------------------------------------------------------------------


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection scoped to settings.duckdb_path.

    Use as a context manager:
        with connect() as con:
            con.execute("SELECT * FROM ...")

    The parent directory is created on demand so a fresh checkout doesn't
    fail with FileNotFoundError on first run.
    """
    db_path = Path(settings.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=str(db_path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


# ----------------------------------------------------------------------------
# Schema bootstrap
# ----------------------------------------------------------------------------

SCHEMAS = ("raw", "staging", "marts")


def bootstrap() -> None:
    """Create raw / staging / marts schemas if they don't already exist.

    Idempotent: safe to call on every pipeline run.
    """
    with connect() as con:
        for schema in SCHEMAS:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


# ----------------------------------------------------------------------------
# Ingest
# ----------------------------------------------------------------------------


def ingest_excel(path: Path | str, *, table: str = "raw.application") -> int:
    """Load the raw Excel into raw.application.

    Drops and recreates the raw table on each call (the raw layer is a
    snapshot, not an accumulator — this matches Snowflake's `CREATE OR
    REPLACE TABLE` pattern). Returns the row count loaded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place the source Excel there "
            f"or override RAW_DATASET_FILENAME via .env."
        )

    df = pd.read_excel(path)
    # Normalise column names same way the legacy loader did.
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    with connect() as con:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return int(row_count)


# ----------------------------------------------------------------------------
# raw -> staging promotion (gated by the data contract)
# ----------------------------------------------------------------------------


def promote_to_staging() -> int:
    """Move data from raw.application to staging.application.

    Currently a near-passthrough: types come from DuckDB's inference. The
    data contract (src/data/contracts/application.py) runs *before* this
    promotes, raising on any failure.
    """
    # Lazy import to avoid making great_expectations a hard dep when
    # someone only needs the warehouse for read.
    from src.data.contracts.application import validate_application

    with connect() as con:
        df_raw = con.execute("SELECT * FROM raw.application").fetchdf()

    # Run the data contract first. Fails LOUD if any expectation breaks.
    validate_application(df_raw)

    with connect() as con:
        con.execute("DROP TABLE IF EXISTS staging.application")
        con.execute("CREATE TABLE staging.application AS SELECT * FROM df_raw")
        row_count = con.execute("SELECT COUNT(*) FROM staging.application").fetchone()[0]
    return int(row_count)


# ----------------------------------------------------------------------------
# Marts — the model-ready view
# ----------------------------------------------------------------------------


def build_marts() -> None:
    """Define marts.applicant_features as a VIEW over staging.

    Drops ID columns and leakage columns at the warehouse layer, so models
    that read marts can never accidentally pull them in. This is belt-and-
    braces with src/config.py — the warehouse layer is the harder gate.
    """
    drop = ID_COLUMNS + LEAKAGE_COLUMNS

    with connect() as con:
        # Get the columns currently in staging.
        columns = [
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='staging' AND table_name='application'"
            ).fetchall()
        ]
        kept = [c for c in columns if c not in drop]
        if TARGET not in kept:
            raise RuntimeError(
                f"Target column '{TARGET}' missing from staging.application. "
                f"Did the source data change?"
            )

        select_list = ", ".join(f'"{c}"' for c in kept)
        con.execute(
            f"CREATE OR REPLACE VIEW marts.applicant_features AS "
            f"SELECT {select_list} FROM staging.application"
        )


# ----------------------------------------------------------------------------
# Read helpers
# ----------------------------------------------------------------------------


def read_sql(sql: str, params: list | None = None) -> pd.DataFrame:
    """Execute a query and return a pandas DataFrame.

    Read-only — opens the connection in read_only mode so concurrent
    readers can coexist without locking issues.
    """
    with connect(read_only=True) as con:
        if params:
            return con.execute(sql, params).fetchdf()
        return con.execute(sql).fetchdf()


def read_marts() -> pd.DataFrame:
    """The standard model-input read. One-liner for the training pipeline."""
    return read_sql("SELECT * FROM marts.applicant_features")


# ----------------------------------------------------------------------------
# End-to-end orchestration
# ----------------------------------------------------------------------------


def refresh(source: Path | str | None = None) -> dict[str, int]:
    """Bootstrap + ingest + validate + promote + build marts, all in one.

    Used by the training pipeline as a one-call "make sure the warehouse
    is current with the source file before I read from it" entry point.
    Returns a dict of row counts at each layer for logging.
    """
    if source is None:
        source = settings.raw_dataset_path

    bootstrap()
    raw_n = ingest_excel(source)
    staging_n = promote_to_staging()
    build_marts()
    with connect(read_only=True) as con:
        marts_n = con.execute("SELECT COUNT(*) FROM marts.applicant_features").fetchone()[0]
    return {"raw": raw_n, "staging": staging_n, "marts": int(marts_n)}
