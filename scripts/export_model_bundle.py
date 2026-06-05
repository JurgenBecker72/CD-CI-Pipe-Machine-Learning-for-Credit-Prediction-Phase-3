"""Extract a self-contained model bundle from MLflow for offline serving.

The serving image deployed to a cloud Kubernetes cluster has no live
connection to the local MLflow tracking server. Rather than running an
MLflow server inside the cluster, we resolve a registered model alias
once at build time, download every artefact the loader needs to a
flat directory, and COPY that directory into the serving image.

The result is a sealed snapshot of one specific model version --
predictable, fast to start, no runtime registry dependency.

Output layout under <dst>/:

    scorecard/                 -- mlflow.sklearn artefact directory
        MLmodel, model.pkl, conda.yaml, ...
    feature_pipeline/          -- Spark ML pipeline + thresholds.json
    band_thresholds/
        band_thresholds.json
    medians/
        medians.json
    bundle_metadata.json       -- model name, version, run_id, captured_at

The loader's offline mode reads from this layout. Re-run this script
whenever you want a fresh bundle (after promoting a new model version).

Usage
-----
    uv run python scripts/export_model_bundle.py
    uv run python scripts/export_model_bundle.py --dst path/to/bundle
    uv run python scripts/export_model_bundle.py --alias staging
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.settings import settings  # noqa: E402

DEFAULT_MODEL_NAME = "credit_scorecard"
DEFAULT_ALIAS = "production"
DEFAULT_DST = PROJECT_ROOT / "model-bundle"

# Artefact paths within the MLflow run that the loader needs.
ARTEFACT_PATHS = ["scorecard", "feature_pipeline", "band_thresholds", "medians"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Registered model to export (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help=f"Alias to resolve (default: {DEFAULT_ALIAS}).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination directory (default: {DEFAULT_DST}).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=("MLflow tracking URI override. Defaults to " "settings.mlflow_tracking_uri."),
    )
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolve alias -> concrete version + run_id
    print(f"Resolving {args.model_name}@{args.alias} via {tracking_uri} ...")
    version_info = client.get_model_version_by_alias(name=args.model_name, alias=args.alias)
    run_id = version_info.run_id
    version = version_info.version
    print(f"  -> version {version}, run_id {run_id}")

    # Clean destination so we don't mix bundles from different runs.
    if args.dst.exists():
        print(f"Removing existing {args.dst} ...")
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    # Download each artefact sub-tree the loader needs.
    for artefact_path in ARTEFACT_PATHS:
        print(f"Downloading artefact '{artefact_path}' ...")
        local = client.download_artifacts(
            run_id=run_id,
            path=artefact_path,
            dst_path=str(args.dst),
        )
        print(f"  -> {local}")

    # Write a small metadata file the loader reads to populate model_name /
    # model_version / model_run_id in API responses.
    metadata = {
        "model_name": args.model_name,
        "model_version": str(version),
        "model_run_id": run_id,
        "alias": args.alias,
        "captured_at": datetime.now(UTC).isoformat(),
        "source_tracking_uri": tracking_uri,
    }
    metadata_path = args.dst / "bundle_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")

    print()
    print("Bundle ready at:", args.dst)
    print("Contents:")
    for child in sorted(args.dst.iterdir()):
        print(f"  {child.name}")


if __name__ == "__main__":
    main()
