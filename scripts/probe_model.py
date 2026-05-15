"""Direct model invocation, bypassing the FastAPI layer.

Loads the production model bundle, runs the smoke payload through the
serving-time feature pipeline, and calls ``predict_proba`` on the raw
sklearn estimator. Prints every input value the model sees plus the
raw probability output. Use this to isolate whether unexpected scoring
behaviour originates in the model itself or in the request path
(Pydantic validation, feature engineering, reindex, response shaping).

Prerequisites
-------------
* MLflow is running at ``settings.mlflow_tracking_uri`` (loader needs it).
* ``scripts/smoke_payload.json`` exists — generate via
  ``scripts/refresh_smoke_payload.py`` if not.

Usage
-----
    uv run python scripts/probe_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.feature_engineering import engineer_features  # noqa: E402
from src.serving.loader import load_bundle  # noqa: E402

PAYLOAD_PATH = Path(__file__).parent / "smoke_payload.json"


def main() -> None:
    if not PAYLOAD_PATH.exists():
        raise FileNotFoundError(
            f"{PAYLOAD_PATH} not found. Run scripts/refresh_smoke_payload.py first."
        )

    print("=" * 70)
    print("Loading model bundle from MLflow")
    print("=" * 70)
    bundle = load_bundle()
    print(f"  model_name    : {bundle.model_name}")
    print(f"  model_version : {bundle.model_version}")
    print(f"  model class   : {type(bundle.model).__name__}")
    print(f"  feature_count : {len(bundle.feature_names)}")
    print(f"  features      : {bundle.feature_names}")

    print()
    print("=" * 70)
    print(f"Loading payload from {PAYLOAD_PATH.name}")
    print("=" * 70)
    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    raw_df = pd.DataFrame([payload])
    print(f"  payload columns: {len(raw_df.columns)}")

    print()
    print("=" * 70)
    print("Running serving-side feature engineering")
    print("=" * 70)
    engineered = engineer_features(raw_df, bundle.quantile_thresholds)
    print(f"  engineered columns: {len(engineered.columns)}")

    print()
    print("=" * 70)
    print("Reindexing to model schema (what the model actually sees)")
    print("=" * 70)
    model_input = engineered.reindex(columns=bundle.feature_names, fill_value=0)
    print(f"  shape: {model_input.shape}")
    print()
    print("  feature                              value")
    print("  ------------------------------------ ----------------")
    for col in bundle.feature_names:
        value = model_input.iloc[0][col]
        in_payload = "       " if col in raw_df.columns else " [FILL]"
        print(f"  {col:<36s} {value:>16.6f}{in_payload}")

    print()
    print("=" * 70)
    print("Model invocation")
    print("=" * 70)
    probas = bundle.model.predict_proba(model_input)
    label = bundle.model.predict(model_input)
    print(f"  predict_proba raw output : {probas}")
    print(f"  shape                    : {probas.shape}")
    print(f"  P(no default) [class 0]  : {probas[0, 0]:.12f}")
    print(f"  P(default)    [class 1]  : {probas[0, 1]:.12f}")
    print(f"  predict (class label)    : {label}")


if __name__ == "__main__":
    main()
