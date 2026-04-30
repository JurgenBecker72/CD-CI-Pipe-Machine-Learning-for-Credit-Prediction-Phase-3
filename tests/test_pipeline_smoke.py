"""Smoke tests for the pipeline entry point.

These don't run the full pipeline (that needs the raw Excel file mounted
and is slow). They confirm the module is importable, the CLI parses args
correctly, and the function signatures haven't drifted. This is the cheap
sanity check that catches "did Docker break the import path" issues
during Phase B build cycles.
"""

from __future__ import annotations

import sys

import pytest


def test_pipeline_module_imports():
    """The module must import cleanly — catches missing deps, syntax errors."""
    import pipelines.run_pipeline  # noqa: F401


def test_run_pipeline_callable_with_path_arg():
    """run_pipeline(path) must still be callable for downstream consumers."""
    from pipelines.run_pipeline import run_pipeline

    assert callable(run_pipeline)
    # It takes one positional arg called `path` — verified via signature.
    import inspect

    sig = inspect.signature(run_pipeline)
    assert list(sig.parameters.keys()) == ["path"]


def test_main_uses_settings_default_when_no_args(monkeypatch):
    """`python -m pipelines.run_pipeline` (no args) should default to settings.raw_dataset_path."""
    from pipelines.run_pipeline import main
    from src.settings import settings

    captured: dict[str, str] = {}

    def fake_run_pipeline(path):
        captured["path"] = path

    # Patch run_pipeline within the pipelines.run_pipeline namespace
    monkeypatch.setattr("pipelines.run_pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(sys, "argv", ["run_pipeline"])

    main()

    assert captured["path"] == str(settings.raw_dataset_path)


def test_main_respects_data_path_flag(monkeypatch):
    """--data-path /custom/path must be honoured."""
    from pipelines.run_pipeline import main

    captured: dict[str, str] = {}

    def fake_run_pipeline(path):
        captured["path"] = path

    monkeypatch.setattr("pipelines.run_pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--data-path", "/tmp/custom.xlsx"])

    main()

    assert captured["path"] == "/tmp/custom.xlsx"


@pytest.mark.integration
def test_run_pipeline_end_to_end_smoke(tmp_path):
    """Full end-to-end smoke — only runs when -m integration is selected.

    This is intentionally NOT in the default `pytest -m "not integration"`
    set — it requires the raw Excel and takes seconds. CI runs it as a
    separate job from Phase D onwards.
    """
    pytest.skip("Full integration test wired up in Phase D when MLflow is available")
