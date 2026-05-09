"""Spark ML Pipeline for credit pipeline feature engineering.

Composes the row-wise feature transformations and the quantile-flag
estimator into a single fittable, serialisable Spark ML Pipeline. The
pipeline is fit once on training data and then transforms train, test,
and inference-time payloads using the same frozen state.

Stages
------
1. RowWiseFeatures   -- stateless row-wise expressions (interactions, ratios)
2. QuantileFlagger   -- learns high/low thresholds; emits binary flags

Usage
-----
    from src.features.pipeline import build_feature_pipeline

    pipeline = build_feature_pipeline()
    fitted = pipeline.fit(train_sdf)
    train_with_features = fitted.transform(train_sdf)
    test_with_features  = fitted.transform(test_sdf)

    # Persist for reuse at scoring time
    fitted.write().overwrite().save("models/feature_pipeline")

    # Reload later
    from pyspark.ml import PipelineModel
    fitted = PipelineModel.load("models/feature_pipeline")
"""

from __future__ import annotations

from pyspark.ml import Pipeline

from src.features.quantile_flagger import QuantileFlagger
from src.features.spark_features import RowWiseFeatures

# Columns that receive high/low quantile flags. Declared at module level
# so the set of fitted thresholds is greppable for governance review.
QUANTILE_FLAG_COLUMNS: list[str] = [
    "total_risk_score",
    "r_ho_em2_co",
    "r_ho_vi4_st",
]

# Default percentile cut-offs. Symmetric around the median.
DEFAULT_HIGH_QUANTILE: float = 0.7
DEFAULT_LOW_QUANTILE: float = 0.3


def build_feature_pipeline(
    quantile_input_cols: list[str] | None = None,
    high_quantile: float = DEFAULT_HIGH_QUANTILE,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
) -> Pipeline:
    """Build the unfitted feature engineering Pipeline.

    Parameters
    ----------
    quantile_input_cols
        Columns to compute high/low quantile flags for. Defaults to the
        module-level `QUANTILE_FLAG_COLUMNS`.
    high_quantile, low_quantile
        Percentile cut-offs (0 < low < high < 1). Same threshold pair
        applies to every input column.

    Returns
    -------
    pyspark.ml.Pipeline
        Unfitted Pipeline ready to receive `.fit(train_df)`. Caller is
        responsible for fitting on training data only and persisting the
        resulting PipelineModel.
    """
    cols = quantile_input_cols if quantile_input_cols is not None else QUANTILE_FLAG_COLUMNS

    return Pipeline(
        stages=[
            RowWiseFeatures(),
            QuantileFlagger(
                inputCols=cols,
                highQuantile=high_quantile,
                lowQuantile=low_quantile,
            ),
        ]
    )
