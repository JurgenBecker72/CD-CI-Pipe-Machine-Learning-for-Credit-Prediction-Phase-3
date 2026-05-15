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
    """Stub sklearn-style classifier that returns a constant 0.3 PD.

    Exposes feature_names_in_ + coef_ so reason-code logic has something
    to walk through. Mirrors the raw-sklearn-estimator contract the
    serving loader now produces (predict_proba returns 2D probabilities,
    not class labels). Keeps tests deterministic and decoupled from MLflow.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.coef_ = np.linspace(-0.5, 0.5, len(feature_names)).reshape(1, -1)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(class=0), P(class=1) per row -- shape (n_rows, 2)."""
        return np.array([[0.7, 0.3] for _ in range(len(df))])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return the argmax of predict_proba -- shape (n_rows,)."""
        return np.zeros(len(df), dtype=int)


_DEFAULT_BAND_THRESHOLDS = {
    "labels_low_to_high": ["E", "D", "C", "B", "A"],
    "cut_points": [430.0, 460.0, 490.0, 520.0],
}


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
        band_thresholds=_DEFAULT_BAND_THRESHOLDS,
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
    # Band thresholds are exposed for client/auditor introspection.
    assert body["band_thresholds"]["labels_low_to_high"] == ["E", "D", "C", "B", "A"]
    assert body["band_thresholds"]["cut_points"] == [430.0, 460.0, 490.0, 520.0]


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


def test_band_lookup_uses_bundle_thresholds() -> None:
    """`_band_from_score` reads cut points from the bundle, not hardcoded values.

    Verifies the lookup walks ascending and uses the last label for any
    score above the highest cut. Future regradings (5-band -> 7-band, or
    a recalibration of the existing cuts) need only re-train and re-deploy
    the model -- no code change in the serving layer.
    """
    from src.serving.app import _band_from_score

    thresholds = {
        "labels_low_to_high": ["E", "D", "C", "B", "A"],
        "cut_points": [430.0, 460.0, 490.0, 520.0],
    }
    # Below the first cut -> bottom label.
    assert _band_from_score(400.0, thresholds) == "E"
    # On the boundary -> next label up (strict `<` in the lookup).
    assert _band_from_score(430.0, thresholds) == "D"
    # Mid-distribution.
    assert _band_from_score(475.0, thresholds) == "C"
    # Just below the top cut.
    assert _band_from_score(519.99, thresholds) == "B"
    # At or above the top cut -> top label.
    assert _band_from_score(520.0, thresholds) == "A"
    assert _band_from_score(900.0, thresholds) == "A"
