"""FastAPI scoring service.

Endpoints
---------
GET  /healthz       Liveness probe -- "is the process alive?"
GET  /readyz        Readiness probe -- "is the model loaded and ready to score?"
GET  /metrics       Prometheus scrape target (request count, latency, errors)
GET  /model_info    Current model name, version, run_id, threshold metadata
POST /v1/score      Score a single applicant; returns PD, score, band, reasons
POST /admin/reload  Hot-reload the model bundle from MLflow (manual cache bust)

Run locally
-----------
    docker compose up -d mlflow                  # MLflow must be reachable
    uv run uvicorn src.serving.app:app --reload  # dev mode, port 8000

Then POST a payload to http://localhost:8000/v1/score and read the score back.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status

from src.serving.feature_engineering import engineer_features
from src.serving.loader import ModelBundle, load_bundle
from src.serving.reason_codes import compute_reason_codes
from src.serving.schemas import (
    ApplicantPayload,
    HealthResponse,
    ModelInfoResponse,
    ReadinessResponse,
    ReasonCode,
    ScoreResponse,
)

logger = logging.getLogger("serving")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ----------------------------------------------------------------------------
# Lifespan: load the model once at startup, keep it in memory
# ----------------------------------------------------------------------------


_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - FastAPI signature
    """Load the model bundle at startup so the first request is fast."""
    global _bundle
    try:
        _bundle = load_bundle()
        logger.info(
            "model_loaded name=%s version=%s run_id=%s features=%d",
            _bundle.model_name,
            _bundle.model_version,
            _bundle.model_run_id,
            len(_bundle.feature_names),
        )
    except Exception as exc:  # noqa: BLE001 - log and continue with /readyz reporting unhealthy
        logger.exception("model_load_failed: %s", exc)
        _bundle = None
    yield
    # No teardown needed; pyfunc model is just memory.


app = FastAPI(
    title="Credit Scoring API",
    description="Real-time scoring service for the credit pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# Prometheus-style in-memory counters
# (replaced by `prometheus_client` registry in the governance phase)
# ----------------------------------------------------------------------------


class _Counters:
    def __init__(self) -> None:
        self.score_requests = 0
        self.score_errors = 0
        self.latency_ms_sum = 0.0
        self.latency_ms_count = 0

    def record_request(self, latency_ms: float) -> None:
        self.score_requests += 1
        self.latency_ms_sum += latency_ms
        self.latency_ms_count += 1

    def record_error(self) -> None:
        self.score_errors += 1


_counters = _Counters()


# ----------------------------------------------------------------------------
# Health & readiness
# ----------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    """Liveness probe. Returns 200 OK if the process is alive."""
    return HealthResponse()


@app.get("/readyz", response_model=ReadinessResponse, tags=["health"])
def readyz(response: Response) -> ReadinessResponse:
    """Readiness probe. Returns 200 if the model loaded; 503 otherwise."""
    is_ready = _bundle is not None
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_loaded=False,
            thresholds_loaded=False,
        )
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        thresholds_loaded=bool(_bundle.quantile_thresholds),
    )


# ----------------------------------------------------------------------------
# Metrics (text/plain Prometheus format)
# ----------------------------------------------------------------------------


@app.get("/metrics", tags=["health"])
def metrics() -> Response:
    """Prometheus scrape target. Text-format counters and latencies."""
    avg_latency = (
        _counters.latency_ms_sum / _counters.latency_ms_count if _counters.latency_ms_count else 0.0
    )
    body = (
        "# HELP score_requests_total Total scoring requests received.\n"
        "# TYPE score_requests_total counter\n"
        f"score_requests_total {_counters.score_requests}\n"
        "# HELP score_errors_total Total scoring errors.\n"
        "# TYPE score_errors_total counter\n"
        f"score_errors_total {_counters.score_errors}\n"
        "# HELP score_latency_ms_avg Average scoring latency in milliseconds.\n"
        "# TYPE score_latency_ms_avg gauge\n"
        f"score_latency_ms_avg {avg_latency:.3f}\n"
    )
    return Response(content=body, media_type="text/plain")


# ----------------------------------------------------------------------------
# Model info
# ----------------------------------------------------------------------------


@app.get("/model_info", response_model=ModelInfoResponse, tags=["info"])
def model_info() -> ModelInfoResponse:
    """Return current model metadata + the fitted quantile thresholds."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return ModelInfoResponse(
        model_name=_bundle.model_name,
        model_version=_bundle.model_version,
        model_run_id=_bundle.model_run_id,
        feature_count=len(_bundle.feature_names),
        quantile_thresholds=_bundle.quantile_thresholds,
    )


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def _band_from_score(score: float) -> str:
    """Map a credit score to A-E risk bands."""
    if score >= 700:
        return "A"
    if score >= 620:
        return "B"
    if score >= 560:
        return "C"
    if score >= 500:
        return "D"
    return "E"


def _row_to_dataframe(payload: ApplicantPayload, feature_names: list[str]) -> pd.DataFrame:
    """Convert the Pydantic payload into a one-row pandas DataFrame.

    Fills missing numeric fields with column-median 0 (the model was
    trained on median-imputed data; for an unknown applicant feature
    the median is the safest fallback). In production we'd persist the
    training medians as a sidecar; here we use 0 as a stand-in.
    """
    raw: dict[str, Any] = payload.model_dump()
    df = pd.DataFrame([raw])

    # Replace None with 0 for numeric columns (medianless approximation).
    df = df.fillna(0)

    # Reindex to whatever the model expects; missing -> 0.
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    return df


@app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
def score(payload: ApplicantPayload) -> ScoreResponse:
    """Score a single applicant.

    Returns probability of default, a credit score (300-900), a risk
    band (A-E), top-3 SHAP reason codes, and the model version that
    produced the decision.
    """
    if _bundle is None:
        _counters.record_error()
        raise HTTPException(status_code=503, detail="model not loaded")

    start = time.perf_counter()

    try:
        # 1. Pydantic payload -> one-row DataFrame
        raw_df = pd.DataFrame([payload.model_dump()]).fillna(0)

        # 2. Apply scoring-time feature engineering (row-wise + quantile flags)
        engineered = engineer_features(raw_df, _bundle.quantile_thresholds)

        # 3. Reindex to the model's expected feature schema
        if _bundle.feature_names:
            engineered = engineered.reindex(columns=_bundle.feature_names, fill_value=0)

        # 4. Predict probability of default
        probas = _bundle.model.predict(engineered)
        # mlflow.pyfunc returns a 2D ndarray for predict_proba-like outputs
        probas_arr = np.asarray(probas)
        if probas_arr.ndim == 2:
            pd_value = float(probas_arr[0, 1])
        else:
            pd_value = float(probas_arr[0])
        pd_value = max(min(pd_value, 1.0 - 1e-6), 1e-6)

        # 5. Convert to a credit score (FICO-style scaling: 600 ± factor*log(odds))
        odds = (1 - pd_value) / pd_value
        factor = 50.0 / np.log(2)
        offset = 600.0 - factor * np.log(20)
        score_value = float(offset + factor * np.log(odds))

        # 6. Reason codes (top-3 SHAP)
        reasons = compute_reason_codes(_bundle.model, engineered, top_n=3)

        latency_ms = (time.perf_counter() - start) * 1000
        _counters.record_request(latency_ms)

        return ScoreResponse(
            probability_of_default=pd_value,
            score=score_value,
            band=_band_from_score(score_value),
            reason_codes=[ReasonCode(**r) for r in reasons],
            model_name=_bundle.model_name,
            model_version=_bundle.model_version,
            model_run_id=_bundle.model_run_id,
        )
    except HTTPException:
        _counters.record_error()
        raise
    except Exception as exc:  # noqa: BLE001
        _counters.record_error()
        logger.exception("scoring_failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc


# ----------------------------------------------------------------------------
# Admin
# ----------------------------------------------------------------------------


@app.post("/admin/reload", tags=["admin"])
def admin_reload() -> dict:
    """Hot-reload the model bundle from MLflow without restarting the service.

    Useful after promoting a new version to the `production` alias --
    the running service can pick up the new model with a single curl.
    No auth in this version; lock down behind an ingress or service mesh
    rule in production deployment.
    """
    global _bundle
    try:
        _bundle = load_bundle()
        return {
            "status": "reloaded",
            "model_name": _bundle.model_name,
            "model_version": _bundle.model_version,
            "model_run_id": _bundle.model_run_id,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("reload_failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"reload failed: {exc}") from exc
