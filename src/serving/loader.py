"""Loads the scoring model and its fitted feature state from MLflow.

The serving service is stateless across requests but holds the model
and the fitted feature thresholds in memory between requests. This
module owns the load -- it talks to MLflow once at startup (or on a
manual reload) and exposes the loaded objects via a `ModelBundle`
dataclass.

Resolved at load time
---------------------
1. The scorecard sklearn estimator (loaded via `mlflow.sklearn.load_model`
   so `.predict_proba()` is available; the pyfunc flavor's `.predict()`
   returns class labels for sklearn classifiers, which is the wrong
   contract for a credit scoring service).
2. The version metadata (number, run_id, registered name)
3. The fitted QuantileFlagger thresholds (downloaded from the run's
   `feature_pipeline/` artefact directory as `thresholds.json`)

The thresholds are extracted at load time and held as a plain dict, so
the serving runtime doesn't need PySpark on its import path.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient

from src.settings import settings

SCORECARD_MODEL_NAME = "credit_scorecard"
PRODUCTION_ALIAS = "production"
FEATURE_PIPELINE_ARTIFACT_PATH = "feature_pipeline"
BAND_THRESHOLDS_ARTIFACT_PATH = "band_thresholds"
MEDIANS_ARTIFACT_PATH = "medians"

# Offline mode -- when this env var points at a directory produced by
# `scripts/export_model_bundle.py`, the loader reads model + sidecars
# from local files instead of contacting an MLflow tracking server.
# Used by the cloud serving image (Phase G cloud deploy) which bakes a
# sealed snapshot of one model version into the container at build time.
MODEL_BUNDLE_PATH_ENV = "MODEL_BUNDLE_PATH"

# Used when the registered model predates the band_thresholds.json artefact
# (older versions, rollback to an earlier model). FICO-tradition cut points
# -- low-default-rate calibration, so they will band most applicants in a
# high-bad-rate population as D/E. The loader warns loudly when this
# fallback is hit.
_DEFAULT_BAND_THRESHOLDS: dict[str, list[Any]] = {
    "labels_low_to_high": ["E", "D", "C", "B", "A"],
    "cut_points": [500.0, 560.0, 620.0, 700.0],
}


# ----------------------------------------------------------------------------
# Bundle returned by load()
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBundle:
    """Everything the serving runtime needs to score a request."""

    model: Any  # raw sklearn estimator (CalibratedClassifierCV) from mlflow.sklearn
    model_name: str
    model_version: str
    model_run_id: str
    quantile_thresholds: dict[str, dict[str, float]]
    feature_names: list[str]  # columns the model expects at predict time
    band_thresholds: dict[str, list[Any]]  # {"labels_low_to_high": [...], "cut_points": [...]}
    medians: dict[str, float]  # per-column training medians; empty dict if absent


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


def _download_medians(client: MlflowClient, run_id: str) -> dict[str, float]:
    """Download the medians.json artefact for this run, or fall back.

    Returns the per-column training medians used to fill missing values
    at scoring time. If the artefact is absent (e.g. the registered
    model version was logged before the medians sidecar contract was
    introduced), logs a warning and returns an empty dict; the serving
    layer interprets that as "fill missing payload values with zero",
    which is the original behaviour and preserves backward compatibility.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = client.download_artifacts(
                run_id=run_id,
                path=MEDIANS_ARTIFACT_PATH,
                dst_path=tmpdir,
            )
            candidates = list(Path(local).rglob("medians.json"))
            if not candidates:
                print(
                    f"[loader] WARNING: no medians.json under {local}; "
                    "serving will fillna(0) for missing payload values."
                )
                return {}
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            # Make sure values are native Python floats.
            return {str(k): float(v) for k, v in payload.items()}
    except Exception as exc:  # noqa: BLE001 - artefact may simply not exist
        print(
            f"[loader] WARNING: could not download medians artefact "
            f"({exc.__class__.__name__}: {exc}); serving will fillna(0)."
        )
        return {}


def _download_band_thresholds(client: MlflowClient, run_id: str) -> dict[str, list[Any]]:
    """Download the band_thresholds.json artefact for this run, or fall back.

    Returns the persisted train-time band cut points. If the artefact is
    absent (e.g. the registered model version was logged before the band
    thresholds sidecar contract was introduced), logs a warning and returns
    the FICO-tradition defaults so the service still starts. The default
    cut points produce a wildly miscalibrated banding for a high-bad-rate
    population -- log volume is intentional.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = client.download_artifacts(
                run_id=run_id,
                path=BAND_THRESHOLDS_ARTIFACT_PATH,
                dst_path=tmpdir,
            )
            candidates = list(Path(local).rglob("band_thresholds.json"))
            if not candidates:
                print(
                    f"[loader] WARNING: no band_thresholds.json under {local}; "
                    "falling back to FICO defaults."
                )
                return dict(_DEFAULT_BAND_THRESHOLDS)
            return json.loads(candidates[0].read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - artefact may simply not exist
        print(
            f"[loader] WARNING: could not download band_thresholds artefact "
            f"({exc.__class__.__name__}: {exc}); falling back to FICO defaults."
        )
        return dict(_DEFAULT_BAND_THRESHOLDS)


def _load_bundle_offline(bundle_dir: Path) -> ModelBundle:
    """Read a sealed model bundle from a local directory.

    The bundle layout is the one produced by
    ``scripts/export_model_bundle.py`` -- one subdirectory per artefact
    path (``scorecard``, ``feature_pipeline``, ``band_thresholds``,
    ``medians``) plus a ``bundle_metadata.json`` capturing model name /
    version / run_id at export time. No MLflow tracking server is
    contacted.
    """
    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"MODEL_BUNDLE_PATH is set to {bundle_dir} but the directory does not exist."
        )

    # Metadata captured at export time -- source of truth for the
    # name/version/run_id surfaced via /model_info and /v1/score responses.
    metadata_path = bundle_dir / "bundle_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"bundle_metadata.json not found in {bundle_dir}; "
            "re-run scripts/export_model_bundle.py to regenerate the bundle."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    # `mlflow.sklearn.load_model` accepts a local path -- no tracking
    # server required when the model files are already on disk.
    scorecard_dir = bundle_dir / "scorecard"
    model = mlflow.sklearn.load_model(str(scorecard_dir))
    print(
        f"[loader] loaded model {metadata['model_name']}@{metadata['alias']} "
        f"(version {metadata['model_version']}) from local bundle"
    )

    # Quantile thresholds live inside the feature_pipeline artefact tree;
    # same finder as the online path uses.
    quantile_thresholds = _find_thresholds_json(bundle_dir / "feature_pipeline")
    print(f"[loader] loaded quantile thresholds for {len(quantile_thresholds)} columns")

    # Feature schema comes from the sklearn estimator's feature_names_in_
    # attribute -- no MLflow model signature lookup needed here because
    # we're not going through the registry.
    feature_names: list[str] = []
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) > 0:
        feature_names = list(names)
    if not feature_names:
        print(
            "[loader] WARNING: feature_names could not be resolved from the "
            "estimator; scoring will pass payloads through unchanged."
        )

    # Band thresholds and medians are flat JSON files in their respective
    # subdirectories. Same fallbacks as the online path so behaviour is
    # equivalent for older bundles missing one of these sidecars.
    band_thresholds_files = list(
        (bundle_dir / BAND_THRESHOLDS_ARTIFACT_PATH).rglob("band_thresholds.json")
    )
    if band_thresholds_files:
        band_thresholds = json.loads(band_thresholds_files[0].read_text(encoding="utf-8"))
    else:
        print(
            "[loader] WARNING: no band_thresholds.json in bundle; " "falling back to FICO defaults."
        )
        band_thresholds = dict(_DEFAULT_BAND_THRESHOLDS)
    print(f"[loader] loaded band cut points: {band_thresholds['cut_points']}")

    medians_files = list((bundle_dir / MEDIANS_ARTIFACT_PATH).rglob("medians.json"))
    if medians_files:
        payload = json.loads(medians_files[0].read_text(encoding="utf-8"))
        medians = {str(k): float(v) for k, v in payload.items()}
    else:
        print(
            "[loader] WARNING: no medians.json in bundle; "
            "serving will fillna(0) for missing payload values."
        )
        medians = {}
    print(f"[loader] loaded training medians for {len(medians)} columns")

    return ModelBundle(
        model=model,
        model_name=metadata["model_name"],
        model_version=str(metadata["model_version"]),
        model_run_id=metadata["model_run_id"],
        quantile_thresholds=quantile_thresholds,
        feature_names=feature_names,
        band_thresholds=band_thresholds,
        medians=medians,
    )


def load_bundle(
    tracking_uri: str | None = None,
    model_name: str = SCORECARD_MODEL_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> ModelBundle:
    """Load the production model bundle.

    Two modes, dispatched by the presence of the ``MODEL_BUNDLE_PATH``
    environment variable:

    * **Offline (cloud)** -- when ``MODEL_BUNDLE_PATH`` is set, read the
      model and sidecars from a local directory baked into the container
      image at build time. Used in cloud deployments where the pod has
      no live connection to an MLflow tracking server.
    * **Online (default, local dev)** -- contact the MLflow tracking
      server, resolve the registered model alias, download the model and
      sidecar artefacts on demand. The behaviour everywhere else.

    Parameters
    ----------
    tracking_uri
        MLflow tracking server URI. Defaults to `settings.mlflow_tracking_uri`.
        Ignored in offline mode.
    model_name
        Registered model name. Defaults to `credit_scorecard`.
        Ignored in offline mode (the bundle's metadata is the source of truth).
    alias
        Alias to resolve. Defaults to `production`.
        Ignored in offline mode.
    """
    bundle_path = os.environ.get(MODEL_BUNDLE_PATH_ENV)
    if bundle_path:
        print(f"[loader] offline mode -- reading bundle from {bundle_path}")
        return _load_bundle_offline(Path(bundle_path))

    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri or settings.mlflow_tracking_uri)

    model_uri = f"models:/{model_name}@{alias}"
    print(f"[loader] resolving {model_uri} ...")

    # Resolve the alias to a concrete version so we can also fetch the run_id.
    version_info = client.get_model_version_by_alias(name=model_name, alias=alias)
    run_id = version_info.run_id

    # Load the raw sklearn estimator (not the pyfunc wrapper) so the
    # serving layer can call `.predict_proba(...)` directly. The pyfunc
    # wrapper's `.predict()` returns class labels for sklearn classifiers,
    # which would silently produce 0/1 outputs that get clamped instead
    # of real probabilities.
    model = mlflow.sklearn.load_model(model_uri)
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

    # Resolve the model's input feature schema. Primary source is the
    # MLflow model signature logged via `infer_signature(...)` at training
    # time -- the signature is part of the registered model contract and
    # is the auditable record of what the model expects. Fall back to the
    # estimator's `feature_names_in_` attribute (set by sklearn during
    # `.fit()`) if the signature is unavailable for any reason.
    feature_names: list[str] = []
    try:
        info = get_model_info(model_uri)
        if info.signature is not None and info.signature.inputs is not None:
            feature_names = [n for n in info.signature.inputs.input_names() if n]
    except Exception as exc:  # noqa: BLE001 - log and fall back
        print(f"[loader] WARNING: could not read model signature: {exc}")

    if not feature_names:
        names = getattr(model, "feature_names_in_", None)
        if names is not None and len(names) > 0:
            feature_names = list(names)

    if not feature_names:
        print(
            "[loader] WARNING: feature_names could not be resolved from signature "
            "or estimator; scoring will pass payloads through unchanged."
        )

    # Band cut points. Falls back to FICO defaults if the registered
    # model predates the persisted artefact.
    band_thresholds = _download_band_thresholds(client, run_id)
    print(f"[loader] loaded band cut points: {band_thresholds['cut_points']}")

    # Per-feature training medians. Falls back to an empty dict if the
    # registered model predates the artefact; serving fillna defaults to
    # zero in that case (the previous behaviour).
    medians = _download_medians(client, run_id)
    print(f"[loader] loaded training medians for {len(medians)} columns")

    return ModelBundle(
        model=model,
        model_name=model_name,
        model_version=str(version_info.version),
        model_run_id=run_id,
        quantile_thresholds=quantile_thresholds,
        feature_names=feature_names,
        band_thresholds=band_thresholds,
        medians=medians,
    )
