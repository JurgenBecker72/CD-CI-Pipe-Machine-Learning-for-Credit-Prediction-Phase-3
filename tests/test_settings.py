"""Smoke tests for src/settings.py.

These are deliberately minimal — they confirm the Settings object loads
without raising and that the contract (env enum, paths, properties) is what
downstream code expects. Heavier behavioral tests live with the modules
that actually consume settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.settings import Environment, Settings, settings


def test_settings_singleton_loads():
    """The module-level singleton must be importable without raising."""
    assert settings is not None
    assert isinstance(settings, Settings)


def test_default_environment_is_local():
    """Default env must be LOCAL — production gates depend on this."""
    s = Settings()
    assert s.env == Environment.LOCAL
    assert s.is_local is True
    assert s.is_production is False


def test_random_seed_default_matches_config():
    """Settings.random_seed must default to the same value as config.RANDOM_STATE."""
    from src.config import RANDOM_STATE

    s = Settings()
    assert s.random_seed == RANDOM_STATE


def test_paths_are_absolute():
    """All exposed paths must resolve absolutely."""
    s = Settings()
    for attr in (
        "project_root",
        "data_dir",
        "raw_dir",
        "processed_dir",
        "models_dir",
        "reports_dir",
    ):
        path = getattr(s, attr)
        assert isinstance(path, Path)
        assert path.is_absolute(), f"{attr} should be absolute, got {path}"


def test_raw_dataset_path_combines_dir_and_filename():
    """raw_dataset_path = raw_dir / raw_dataset_filename."""
    s = Settings()
    assert s.raw_dataset_path == s.raw_dir / s.raw_dataset_filename


def test_log_level_validation_rejects_garbage():
    """log_level must match the regex; bad values raise ValidationError."""
    with pytest.raises(ValueError):
        Settings(log_level="LOUD")


def test_environment_enum_round_trip():
    """All four declared environments must be constructible from strings."""
    for value in ("local", "dev", "staging", "production"):
        s = Settings(env=value)
        assert s.env.value == value
