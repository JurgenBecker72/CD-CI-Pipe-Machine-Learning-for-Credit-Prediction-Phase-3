"""QuantileFlagger — Spark ML Estimator/Transformer for quantile-based binary flags.

Computes high and low quantile thresholds for each input column at fit
time and produces a Transformer that emits two binary flag columns per
input::

    {col}_high_flag  -- 1 where value > learned high threshold else 0
    {col}_low_flag   -- 1 where value < learned low  threshold else 0

Behaviour
---------
The Estimator's `fit` is the only method that reads training data. The
returned Model carries the frozen thresholds; subsequent `transform`
calls — on test data, holdout, or incoming scoring requests — never
recompute them. Train-time and inference-time transformations are
therefore bit-for-bit identical by construction, which avoids the
distribution leak that arises when quantiles are computed across train
and test rows together.

Persistence
-----------
The fitted Model serialises through the standard Spark ML save/load
protocol. The thresholds travel as a JSON sidecar (`thresholds.json`)
alongside the metadata directory, so a model risk reviewer can audit
the cut-offs for a given training run without loading Spark.

Why a custom Estimator instead of `pyspark.ml.feature.QuantileDiscretizer`
-------------------------------------------------------------------------
QuantileDiscretizer emits a single bucketed integer per input. The
scoring contract here requires two separate binary indicators (high /
low) per input, plus an explicit JSON sidecar of thresholds for
governance review. A custom Estimator hits both requirements with no
post-processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import (
    DefaultParamsReadable,
    DefaultParamsWritable,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------


class QuantileFlagger(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    """Estimator that learns quantile thresholds on a training dataframe.

    Parameters
    ----------
    inputCols : list[str]
        Columns to compute thresholds for.
    highQuantile : float, default 0.7
        Upper percentile (0 < q < 1). One threshold per input column.
    lowQuantile : float, default 0.3
        Lower percentile (0 < q < 1). One threshold per input column.

    Output (after fit + transform)
    ------------------------------
    For each `col` in `inputCols`::

        f"{col}_high_flag"  -- integer; 1 if value > high threshold else 0
        f"{col}_low_flag"   -- integer; 1 if value < low  threshold else 0
    """

    inputCols: Param[list[str]] = Param(
        Params._dummy(),
        "inputCols",
        "Columns to compute high/low quantile flags for.",
        typeConverter=TypeConverters.toListString,
    )
    highQuantile: Param[float] = Param(
        Params._dummy(),
        "highQuantile",
        "Upper percentile (0 < q < 1) used to compute the high flag threshold.",
        typeConverter=TypeConverters.toFloat,
    )
    lowQuantile: Param[float] = Param(
        Params._dummy(),
        "lowQuantile",
        "Lower percentile (0 < q < 1) used to compute the low flag threshold.",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(
        self,
        inputCols: list[str] | None = None,
        highQuantile: float = 0.7,
        lowQuantile: float = 0.3,
    ) -> None:
        super().__init__()
        self._setDefault(highQuantile=0.7, lowQuantile=0.3)
        if inputCols is not None:
            self.setInputCols(inputCols)
        self.setHighQuantile(highQuantile)
        self.setLowQuantile(lowQuantile)

    # ---- setters / getters ---------------------------------------------

    def setInputCols(self, value: list[str]) -> QuantileFlagger:
        return self._set(inputCols=value)

    def getInputCols(self) -> list[str]:
        return self.getOrDefault(self.inputCols)

    def setHighQuantile(self, value: float) -> QuantileFlagger:
        return self._set(highQuantile=float(value))

    def getHighQuantile(self) -> float:
        return self.getOrDefault(self.highQuantile)

    def setLowQuantile(self, value: float) -> QuantileFlagger:
        return self._set(lowQuantile=float(value))

    def getLowQuantile(self) -> float:
        return self.getOrDefault(self.lowQuantile)

    # ---- fit -----------------------------------------------------------

    def _fit(self, dataset: DataFrame) -> QuantileFlaggerModel:
        cols = self.getInputCols()
        high_pct = self.getHighQuantile()
        low_pct = self.getLowQuantile()

        if not (0.0 < low_pct < high_pct < 1.0):
            raise ValueError(
                f"Quantiles must satisfy 0 < lowQuantile < highQuantile < 1; "
                f"got low={low_pct}, high={high_pct}"
            )

        # Skip columns that are absent from the training frame so an
        # upstream contract change degrades gracefully rather than
        # aborting the whole run. The skip is logged so silent omission
        # cannot occur.
        present = [c for c in cols if c in dataset.columns]
        missing = [c for c in cols if c not in dataset.columns]
        if missing:
            print(f"[QuantileFlagger] skipping missing columns: {missing}")

        thresholds: dict[str, dict[str, float]] = {}
        for col in present:
            low_t, high_t = dataset.approxQuantile(col, [low_pct, high_pct], 0.001)
            thresholds[col] = {"low": float(low_t), "high": float(high_t)}

        model = QuantileFlaggerModel(
            inputCols=list(thresholds.keys()),
            thresholds=thresholds,
        )
        model._resetUid(self.uid)
        return model


# --------------------------------------------------------------------------
# Fitted Model
# --------------------------------------------------------------------------


class QuantileFlaggerModel(Model, MLReadable, MLWritable):
    """Fitted Transformer carrying the frozen quantile thresholds.

    Produced by `QuantileFlagger.fit()`. Apply with `.transform(df)`.
    Thresholds are immutable after fit; refitting requires a new
    QuantileFlagger Estimator.
    """

    def __init__(
        self,
        inputCols: list[str] | None = None,
        thresholds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self._inputCols = inputCols or []
        self._thresholds = thresholds or {}

    @property
    def inputCols(self) -> list[str]:
        return list(self._inputCols)

    @property
    def thresholds(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._thresholds.items()}

    def _transform(self, dataset: DataFrame) -> DataFrame:
        for col in self._inputCols:
            if col not in dataset.columns:
                # Optional column absent at scoring time; emit no flags
                # rather than reject the request.
                print(f"[QuantileFlaggerModel] skipping flag for missing column {col!r}")
                continue
            high_t = self._thresholds[col]["high"]
            low_t = self._thresholds[col]["low"]
            dataset = dataset.withColumn(
                f"{col}_high_flag",
                F.when(F.col(col) > F.lit(high_t), 1).otherwise(0).cast("integer"),
            ).withColumn(
                f"{col}_low_flag",
                F.when(F.col(col) < F.lit(low_t), 1).otherwise(0).cast("integer"),
            )
        return dataset

    # ---- save / load ---------------------------------------------------

    def write(self) -> QuantileFlaggerModelWriter:
        return QuantileFlaggerModelWriter(self)

    @classmethod
    def read(cls) -> QuantileFlaggerModelReader:
        return QuantileFlaggerModelReader()


class QuantileFlaggerModelWriter(MLWriter):
    """Serialises a fitted QuantileFlaggerModel.

    Layout written::

        <path>/
            metadata/         # Spark ML class identification
            thresholds.json   # fitted threshold values (governance audit)
    """

    def __init__(self, instance: QuantileFlaggerModel) -> None:
        super().__init__()
        self._instance = instance

    def saveImpl(self, path: str) -> None:
        meta_dir = Path(path) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "class": "src.features.quantile_flagger.QuantileFlaggerModel",
            "timestamp": 0,
            "sparkVersion": "3.5",
            "uid": self._instance.uid,
            "paramMap": {},
            "defaultParamMap": {},
        }
        (meta_dir / "part-00000").write_text(json.dumps(metadata))
        (meta_dir / "_SUCCESS").write_text("")

        sidecar = {
            "inputCols": self._instance.inputCols,
            "thresholds": self._instance.thresholds,
        }
        (Path(path) / "thresholds.json").write_text(json.dumps(sidecar, indent=2))


class QuantileFlaggerModelReader(MLReader):
    """Deserialises a QuantileFlaggerModel."""

    def load(self, path: str) -> QuantileFlaggerModel:
        sidecar_path = Path(path) / "thresholds.json"
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"Expected fitted state at {sidecar_path!s}; "
                "save the model with QuantileFlaggerModel.write().save(...) first."
            )
        sidecar = json.loads(sidecar_path.read_text())
        return QuantileFlaggerModel(
            inputCols=sidecar["inputCols"],
            thresholds=sidecar["thresholds"],
        )
