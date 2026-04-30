"""Environment-aware runtime settings for the credit pipeline.

This module is the single source of truth for values that change between
environments (local development, container, cloud cluster). It complements
the two existing static-knowledge modules:

    src/config.py  — what the project knows about its data
                     (feature lists, target name, leakage columns).
    src/paths.py   — where files live on disk
                     (project-relative directories).
    src/settings.py — runtime configuration that varies by environment
                     (URIs, secrets, environment name, MLflow tracking URI,
                     log level, random seed).

Settings are loaded from environment variables, with .env file support for
local development. Real .env files are gitignored; .env.example documents
the full list of variables and their defaults.

Usage
-----
    from src.settings import settings

    print(settings.env)                  # Environment.LOCAL
    print(settings.raw_dataset_path)     # /…/data/raw/DRA_with_simulated_credit.xlsx
    print(settings.is_local)             # True

The `settings` object is a module-level singleton — import it directly rather
than instantiating Settings() yourself, so reads are consistent across the
process.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.paths import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    ROOT,
)


class Environment(str, Enum):
    """Deployment environment. Used to gate environment-specific behavior."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Runtime configuration loaded from env vars and .env.

    Variable names are matched case-insensitively. Anything not declared
    here that appears in the environment is silently ignored (extra="ignore"),
    so this object stays a focused contract.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Environment & logging
    # ------------------------------------------------------------------
    env: Environment = Environment.LOCAL
    log_level: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Standard library logging level",
    )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random_seed: int = Field(default=42, description="Seed for numpy / sklearn / torch")

    # ------------------------------------------------------------------
    # Filesystem layout
    # Defaults come from src/paths.py (project-relative). Override via
    # env var to relocate, e.g. when running in a container that mounts
    # data elsewhere.
    # ------------------------------------------------------------------
    project_root: Path = ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR

    raw_dataset_filename: str = Field(
        default="DRA_with_simulated_credit.xlsx",
        description="Filename of the source dataset inside raw_dir",
    )

    # ------------------------------------------------------------------
    # MLflow (Phase C onwards) — placeholders so callers can already
    # depend on the contract; Phase C wires them up to a running server.
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "credit-scorecard"
    mlflow_artefact_root: str = ""  # empty = MLflow default

    # ------------------------------------------------------------------
    # DuckDB warehouse (Phase B onwards) — placeholder.
    # ------------------------------------------------------------------
    duckdb_path: Path = ROOT / "warehouse" / "credit.duckdb"

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def is_local(self) -> bool:
        """True when running in a developer's local environment."""
        return self.env == Environment.LOCAL

    @property
    def is_production(self) -> bool:
        """True when running in production. Use to gate side-effects."""
        return self.env == Environment.PRODUCTION

    @property
    def raw_dataset_path(self) -> Path:
        """Absolute path to the source Excel dataset."""
        return self.raw_dir / self.raw_dataset_filename


# Module-level singleton. Import this, do NOT instantiate Settings() elsewhere.
settings = Settings()
