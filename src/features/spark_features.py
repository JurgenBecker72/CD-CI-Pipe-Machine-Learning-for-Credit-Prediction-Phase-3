"""Row-wise feature engineering for the credit pipeline.

Stateless Spark ML Transformer that adds engineered columns to an input
dataframe. Every output column depends only on values within the same
row, so there is no fitted state and no train/test leakage surface.

Output columns
--------------
Risk structure (added when both source columns are present)::

    net_risk    = risk_drivers - risk_mitigators
    risk_ratio  = risk_drivers / (risk_mitigators + 1)

Interaction terms (added when both inputs are present)::

    emotional_x_stability = r_ho_em2_co * r_ho_vi4_st
    risk_x_emotional      = total_risk_score * r_ho_em2_co
    drivers_x_mitigators  = risk_drivers * risk_mitigators
    risk_x_mitigators     = total_risk_score * risk_mitigators

Missing input columns are skipped with a printed warning rather than
raising, so an upstream contract change degrades gracefully and a
scoring request that omits an optional field still produces a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyspark.ml import Transformer
from pyspark.ml.util import MLReadable, MLReader, MLWritable, MLWriter
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Interaction terms declared at module level for greppable governance audit.
INTERACTION_FEATURES: list[tuple[str, str, str]] = [
    # (output_col, left_col, right_col)
    ("emotional_x_stability", "r_ho_em2_co", "r_ho_vi4_st"),
    ("risk_x_emotional", "total_risk_score", "r_ho_em2_co"),
    ("drivers_x_mitigators", "risk_drivers", "risk_mitigators"),
    ("risk_x_mitigators", "total_risk_score", "risk_mitigators"),
]


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------


class RowWiseFeatures(Transformer, MLReadable, MLWritable):
    """Add the row-wise engineered features to a Spark dataframe.

    Stateless. No fit step. Sits inside a `pyspark.ml.Pipeline` alongside
    fittable transformers (e.g. QuantileFlagger).
    """

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, dataset: DataFrame) -> DataFrame:
        df = dataset

        # Risk structure: requires both risk_drivers and risk_mitigators.
        if "risk_drivers" in df.columns and "risk_mitigators" in df.columns:
            df = df.withColumn(
                "net_risk", F.col("risk_drivers") - F.col("risk_mitigators")
            ).withColumn(
                # +1 in the denominator preserves the divide-by-zero guard.
                "risk_ratio",
                F.col("risk_drivers") / (F.col("risk_mitigators") + F.lit(1)),
            )
        else:
            print(
                "[RowWiseFeatures] skipping net_risk + risk_ratio "
                "(missing risk_drivers and/or risk_mitigators)"
            )

        # Interaction terms: each requires both source columns.
        for out_col, left, right in INTERACTION_FEATURES:
            if left in df.columns and right in df.columns:
                df = df.withColumn(out_col, F.col(left) * F.col(right))
            else:
                missing = [c for c in (left, right) if c not in df.columns]
                print(
                    f"[RowWiseFeatures] skipping {out_col!r} " f"(missing input columns: {missing})"
                )

        return df

    # ---- save / load ---------------------------------------------------
    # Stateless transformer; on-disk layout is just metadata so the
    # Spark ML Pipeline save/load protocol round-trips cleanly.

    def write(self) -> RowWiseFeaturesWriter:
        return RowWiseFeaturesWriter(self)

    @classmethod
    def read(cls) -> RowWiseFeaturesReader:
        return RowWiseFeaturesReader()


class RowWiseFeaturesWriter(MLWriter):
    def __init__(self, instance: RowWiseFeatures) -> None:
        super().__init__()
        self._instance = instance

    def saveImpl(self, path: str) -> None:
        meta_dir = Path(path) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "class": "src.features.spark_features.RowWiseFeatures",
            "timestamp": 0,
            "sparkVersion": "3.5",
            "uid": self._instance.uid,
            "paramMap": {},
            "defaultParamMap": {},
        }
        (meta_dir / "part-00000").write_text(json.dumps(metadata))
        (meta_dir / "_SUCCESS").write_text("")


class RowWiseFeaturesReader(MLReader):
    def load(self, path: str) -> RowWiseFeatures:  # noqa: ARG002
        return RowWiseFeatures()
