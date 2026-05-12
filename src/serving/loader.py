"""Loads the scoring model and its fitted feature state from MLflow.

The serving service is stateless across requests but holds the model
and the fitted feature thresholds in memory between requests. This
module owns the load -- it talks to MLflow once at startup (or on a
manual reload) and exposes the loaded objects via a `ModelBundle`
dataclass.

Resolved at load time
---------------------
1. The scorecard sklearn model (loaded via `mlflow.pyfunc.load_model`)
2. The version metadata (number, run_id, registered name)
3. The fitted QuantileFlagger thresholds (downloaded from the run's
   `feature_pipeline/` artefact directory as `thresholds.json`)

The thresholds are extracted at load time and held as a plain dict, so
the serving runtime doesn't need PySpark on its import path.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient

from src.settings import settings

SCORECARD_MODEL_NAME = "credit_scorecard"
PRODUCTION_ALIAS = "production"
FEATURE_PIPELINE_ARTIFACT_PATH = "feature_pipeline"


# ----------------------------------------------------------------------------
# Bundle returned by load()
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBundle:
    """Everything the serving runtime needs to score a request."""

    model: Any  # sklearn model loaded via mlflow.pyfunc
    model_name: str
    model_version: str
    model_run_id: str
    quantile_thresholds: dict[str, dict[str, float]]
    feature_names: list[str]  # columns the model expects at predict time


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


def _find_thresholds_json(local_artefact_dir: Path) -> dict[str, dict[str, float]]:
    """Locate the QuantileFlagger thresholds.json inside the downloaded artefact.

    The Spark ML Pipeline save format nests the QuantileFlaggerModel under
    `stages/<n>_QuantileFlaggerModel_<uid>/`. We don't know the index or
    UID at load time, so glob for any `thresholds.json` under the artefact
    directory and use the first match. If none is found, return an empty
    dict and log a warning -- scoring still works, just without quantile
    flags.
    """
    candidates = list(local_artefact_dir.rglob("thresholds.json"))
    if not candidates:
        print(
            f"[loader] WARNING: no thresholds.json found under {local_artefact_dir}; "
            "quantile flags will be absent from scoring."
        )
        return {}

    with candidates[0].open() as f:
        payload = json.load(f)

    return payload.get("thresholds", {})


def load_bundle(
    tracking_uri: str | None = None,
    model_name: str = SCORECARD_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> ModelBundle:
    """Load the production model bundle from MLflow.

    Parameters
    ----------
    tracking_uri
        MLflow tracking server URI. Defaults to `settings.mlflow_tracking_uri`.
    model_name
        Registered model name. Defaults to `credit_scorecard`.
    alias
        Alias to resolve. Defaults to `production`.
    """
    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri or settings.mlflow_tracking_uri)

    model_uri = f"models:/{model_name}@{alias}"
    print(f"[loader] resolving {model_uri} ...")

    # Resolve the alias to a concrete version so we can also fetch the run_id.
    version_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    run_id = version_info.run_id

    # Load the sklearn model itself.
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"[loader] loaded model {model_name}@{alias} (version {version_info.version})")

    # Pull the fitted feature pipeline down to a tmp dir, find thresholds.json.
    with tempfile.TemporaryDirectory() as tmpdir:
        local = client.download_artifacts(
            run_id=run_id,
            path=FEATURE_PIPELINE_ARTIFACT_PATH,
            dst_path=tmpdir,
        )
        quantile_thresholds = _find_thresholds_json(Path(local))
    print(f"[loader] loaded quantile thresholds for {len(quantile_thresholds)} columns")

    # Resolve the model's input feature schema. Source-of-truth is the
    # MLflow model signature logged via `infer_signature(...)` at training
    # time. Walking the pyfunc wrapper is brittle across mlflow versions
    # (the attribute path changes between sklearn-flavor and python-model
    # flavor), but the signature is part of the registered model contract.
    feature_names: list[str] = []
    try:
        info = get_model_info(model_uri)
        if info.signature is not None and info.signature.inputs is not None:
            # Schema.input_names() returns the list of column names in order.
            feature_names = [n for n in info.signature.inputs.input_names() if n]
    except Exception as exc:  # noqa: BLE001 - log and fall back
        print(f"[loader] WARNING: could not read model signature: {exc}")

    # Fallback: try to reach the underlying sklearn estimator for its
    # feature_names_in_ attribute. Path differs across mlflow versions.
    if not feature_names:
        for attr_path in (
            ("_model_impl", "sklearn_model"),
            ("_model_impl", "model"),
            ("_model_impl", "python_model"),
        ):
            try:
                obj: Any = model
                for attr in attr_path:
                    obj = getattr(obj, attr)
                names = getattr(obj, "feature_names_in_", None)
                if names is not None and len(names) > 0:
                    feature_names = list(names)
                    break
            except AttributeError:
                continue

    if not feature_names:
        print(
            "[loader] WARNING: feature_names could not be resolved from signature "
            "or underlying estimator; scoring will pass payloads through unchanged."
        )

    return ModelBundle(
        model=model,
        model_name=model_name,
        model_version=str(version_info.version),
        model_run_id=run_id,
        quantile_thresholds=quantile_thresholds,
        feature_names=feature_names,
    )
