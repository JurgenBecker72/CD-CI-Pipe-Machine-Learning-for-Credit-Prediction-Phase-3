"""Pydantic request and response models for the scoring API.

The request schema is built dynamically from `src/config.py` so the API
contract cannot drift from the canonical feature list. Adding a feature
to `BASE_FEATURES` automatically extends the API; the alternative —
duplicating the list here — is the single biggest source of bugs in
production scoring services.

The response shape mirrors the FCRA "adverse action notice" pattern:
probability of default, a credit score, top reason codes, and the
model version that produced the decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.config import (
    CATEGORICAL_FEATURES,
    CREDIT_ASSESS_FEATURES,
    DRA_DIMENSIONS,
    DRA_ITEMS,
    ID_COL,
    RISK_COMPOSITES,
)

# ----------------------------------------------------------------------------
# Field groups -- each group has a different type and validation contract
# ----------------------------------------------------------------------------

_NUMERIC_FIELDS = DRA_DIMENSIONS + DRA_ITEMS + RISK_COMPOSITES
_INTEGER_FIELDS = CREDIT_ASSESS_FEATURES
_CATEGORICAL_FIELDS = CATEGORICAL_FEATURES


# ----------------------------------------------------------------------------
# Dynamic request model
# ----------------------------------------------------------------------------


class _StrictApplicantBase(BaseModel):
    """Base for the dynamic ApplicantPayload that locks down the validation contract.

    `extra="forbid"` rejects requests containing fields the schema doesn't
    know about -- catches typos in client payloads at the door rather
    than silently dropping them. `str_strip_whitespace` trims accidental
    leading/trailing whitespace on categorical inputs.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _build_applicant_payload_model() -> type[BaseModel]:
    """Build the ApplicantPayload Pydantic model from config.py feature lists.

    Numeric features (DRA dimensions, items, risk composites) accept any
    float. Integer features (assessment-time credit bureau counts) accept
    non-negative ints. Categorical features accept any string.

    The dummy_id is optional -- present for batch scoring against historical
    data, absent for real-time scoring of a fresh applicant where no
    identifier exists yet.
    """
    fields: dict[str, tuple] = {
        ID_COL: (str | None, Field(default=None, description="Applicant identifier (optional).")),
    }

    for name in _NUMERIC_FIELDS:
        fields[name] = (float | None, Field(default=None, description=f"Numeric feature {name}."))

    for name in _INTEGER_FIELDS:
        fields[name] = (
            int | None,
            Field(default=None, ge=0, description=f"Credit bureau count {name} (non-negative)."),
        )

    for name in _CATEGORICAL_FIELDS:
        fields[name] = (
            str | None,
            Field(default=None, description=f"Categorical feature {name}."),
        )

    # Build with the strict base so `extra="forbid"` is applied. Setting
    # model_config after create_model() is a no-op in Pydantic v2.
    return create_model("ApplicantPayload", __base__=_StrictApplicantBase, **fields)


ApplicantPayload = _build_applicant_payload_model()


# ----------------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------------


class ReasonCode(BaseModel):
    """One contributing factor in the scoring decision."""

    feature: str = Field(description="Feature name (matches the input schema).")
    contribution: float = Field(
        description="SHAP value: positive raises default probability, negative lowers it.",
    )
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        description="Human-readable direction of the contribution.",
    )


class ScoreResponse(BaseModel):
    """Real-time scoring response.

    Maps to the FCRA adverse-action notice format: the decision (PD +
    score), the top contributing factors, and the model version that
    produced the decision (for auditability).
    """

    model_config = ConfigDict(protected_namespaces=())

    probability_of_default: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated probability of default in the next 12 months.",
    )
    score: float = Field(
        description="Credit score (300-900 range, higher = lower risk).",
    )
    band: Literal["A", "B", "C", "D", "E"] = Field(
        description="Risk band derived from score; A = best, E = worst.",
    )
    reason_codes: list[ReasonCode] = Field(
        description="Top contributing features for this decision (FCRA adverse-action format).",
    )
    model_name: str = Field(description="Registered model name in MLflow.")
    model_version: str = Field(description="Numeric version of the model that produced the score.")
    model_run_id: str = Field(description="MLflow run_id that produced this model version.")


# ----------------------------------------------------------------------------
# Health / readiness / info models
# ----------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    thresholds_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    model_run_id: str
    feature_count: int
    quantile_thresholds: dict[str, dict[str, float]]
