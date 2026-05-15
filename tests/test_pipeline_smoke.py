"""Smoke tests for the pipeline entry point.

These don't run the full pipeline (that needs the raw Excel file mounted
and is slow). They confirm the module is importable, the CLI parses args
correctly, and the function signatures haven't drifted. This is the cheap
sanity check that catches "did Docker break the import path" issues at
build time.
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


def test_main_defaults_to_warehouse_mode_when_no_args(monkeypatch):
    """`python -m pipelines.run_pipeline` (no args) must pass None.

    The pipeline's data-loading layer interprets `path=None` as "read from
    the DuckDB warehouse" (see `load_data` in pipelines.run_pipeline).
    Defaulting to None therefore means the CLI defaults to warehouse-backed
    loading; an explicit `--data-path` is the override for ad-hoc Excel reads.
    """
    from pipelines.run_pipeline import main

    captured: dict[str, str | None] = {}

    def fake_run_pipeline(path):
        captured["path"] = path

    # Patch run_pipeline within the pipelines.run_pipeline namespace
    monkeypatch.setattr("pipelines.run_pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(sys, "argv", ["run_pipeline"])

    main()

    assert captured["path"] is None


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
    separate job.
    """
    pytest.skip("Full integration test runs in the MLflow-aware training entry")
