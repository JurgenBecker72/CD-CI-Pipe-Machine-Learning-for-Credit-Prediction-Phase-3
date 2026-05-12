"""Contract tests for the scoring API.

Validates the request/response shape of every endpoint without requiring
a real MLflow server -- the model bundle is monkey-patched with an
in-memory stub. Run as part of the default fast test suite.

A separate integration test (marked `integration`) hits the real MLflow
endpoint and is excluded from default runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------


class _StubModel:
    """Stub model that returns a constant 0.3 probability of default.

    Exposes feature_names_in_ + coef_ so reason-code logic has something
    to walk through. Keeps tests deterministic and decoupled from MLflow.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.coef_ = np.linspace(-0.5, 0.5, len(feature_names)).reshape(1, -1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Mimic mlflow.pyfunc 2D output: shape (n_rows, n_classes)
        return np.array([[0.7, 0.3] for _ in range(len(df))])


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Build a TestClient against the app with a stubbed model bundle."""
    from src.serving import app as app_module
    from src.serving.loader import ModelBundle

    stub = _StubModel(feature_names=["total_risk_score", "risk_drivers", "risk_mitigators"])
    bundle = ModelBundle(
        model=stub,
        model_name="credit_scorecard",
        model_version="1",
        model_run_id="stub-run-id",
        quantile_thresholds={
            "total_risk_score": {"low": 20.0, "high": 80.0},
        },
        feature_names=["total_risk_score", "risk_drivers", "risk_mitigators"],
    )
    monkeypatch.setattr(app_module, "_bundle", bundle)
    return TestClient(app_module.app)


@pytest.fixture
def sample_payload() -> dict:
    """Minimal applicant payload that satisfies the schema."""
    return {
        "dummy_id": "applicant-001",
        "total_risk_score": 55.0,
        "risk_drivers": 28.0,
        "risk_mitigators": 22.0,
        "product_type": "personal_loan",
    }


# ----------------------------------------------------------------------------
# Health & readiness
# ----------------------------------------------------------------------------


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_ready_when_bundle_loaded(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["thresholds_loaded"] is True


def test_readyz_returns_503_when_bundle_missing(monkeypatch) -> None:
    from src.serving import app as app_module

    monkeypatch.setattr(app_module, "_bundle", None)
    client = TestClient(app_module.app)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


# ----------------------------------------------------------------------------
# Metrics & model info
# ----------------------------------------------------------------------------


def test_metrics_is_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "score_requests_total" in response.text
    assert "score_errors_total" in response.text
    assert "score_latency_ms_avg" in response.text


def test_model_info_returns_metadata(client: TestClient) -> None:
    response = client.get("/model_info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "credit_scorecard"
    assert body["model_version"] == "1"
    assert body["model_run_id"] == "stub-run-id"
    assert body["feature_count"] == 3
    assert "total_risk_score" in body["quantile_thresholds"]


# ----------------------------------------------------------------------------
# Scoring contract
# ----------------------------------------------------------------------------


def test_score_returns_expected_shape(client: TestClient, sample_payload: dict) -> None:
    response = client.post("/v1/score", json=sample_payload)
    assert response.status_code == 200, response.text
    body = response.json()

    # Required fields and types.
    assert isinstance(body["probability_of_default"], float)
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert isinstance(body["score"], int | float)
    assert body["band"] in {"A", "B", "C", "D", "E"}
    assert isinstance(body["reason_codes"], list)
    assert len(body["reason_codes"]) <= 3

    # Reason code shape
    for reason in body["reason_codes"]:
        assert {"feature", "contribution", "direction"} <= set(reason.keys())
        assert reason["direction"] in {"increases_risk", "decreases_risk"}

    # Audit fields
    assert body["model_name"] == "credit_scorecard"
    assert body["model_version"] == "1"
    assert body["model_run_id"] == "stub-run-id"


def test_score_rejects_unknown_field(client: TestClient, sample_payload: dict) -> None:
    """Extra fields must be rejected by the strict Pydantic schema."""
    bad = dict(sample_payload)
    bad["mystery_feature"] = 42
    response = client.post("/v1/score", json=bad)
    assert response.status_code == 422


def test_score_rejects_negative_credit_count(client: TestClient, sample_payload: dict) -> None:
    """num_accounts_assess must be non-negative per the schema constraint."""
    bad = dict(sample_payload)
    bad["num_accounts_assess"] = -5
    response = client.post("/v1/score", json=bad)
    assert response.status_code == 422


def test_score_returns_503_when_bundle_missing(monkeypatch, sample_payload: dict) -> None:
    from src.serving import app as app_module

    monkeypatch.setattr(app_module, "_bundle", None)
    client = TestClient(app_module.app)
    response = client.post("/v1/score", json=sample_payload)
    assert response.status_code == 503
