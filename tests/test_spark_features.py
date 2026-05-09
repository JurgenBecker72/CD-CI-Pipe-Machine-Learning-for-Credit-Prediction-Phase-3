"""Integration tests for the Spark feature engineering layer.

Validates the three building blocks (RowWiseFeatures, QuantileFlagger,
build_feature_pipeline) end-to-end against a small synthetic dataset.

These tests are marked `integration` because each test pays the
~5-second SparkSession startup cost. Excluded from the default fast
sweep; run explicitly with::

    uv run pytest -m integration tests/test_spark_features.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip the whole module unless the spark extras are installed.
pyspark = pytest.importorskip("pyspark")

# Pin the Python executable Spark workers use on Windows.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark():
    """Module-scoped SparkSession; one JVM boot per test module."""
    from src.features.spark_session import get_spark

    session = get_spark("test-spark-features", shuffle_partitions=2)
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()


@pytest.fixture
def synthetic_pdf():
    """1,000-row synthetic credit-applicant frame with all expected inputs."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(1_000),
            "total_risk_score": rng.normal(50, 15, size=1_000),
            "risk_drivers": rng.normal(20, 5, size=1_000),
            "risk_mitigators": rng.normal(15, 4, size=1_000),
            "r_ho_em2_co": rng.normal(0, 1, size=1_000),
            "r_ho_vi4_st": rng.normal(0, 1, size=1_000),
        }
    )


@pytest.fixture
def synthetic_sdf(spark, synthetic_pdf):
    """SparkDataFrame view of the synthetic pdf."""
    return spark.createDataFrame(synthetic_pdf)


# --------------------------------------------------------------------------
# RowWiseFeatures
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_row_wise_features_adds_expected_columns(synthetic_sdf):
    """All six engineered columns appear after transform()."""
    from src.features.spark_features import RowWiseFeatures

    out = RowWiseFeatures().transform(synthetic_sdf)
    new_cols = set(out.columns) - set(synthetic_sdf.columns)

    expected = {
        "net_risk",
        "risk_ratio",
        "emotional_x_stability",
        "risk_x_emotional",
        "drivers_x_mitigators",
        "risk_x_mitigators",
    }
    assert expected.issubset(new_cols), f"missing columns: {expected - new_cols}"


@pytest.mark.integration
def test_row_wise_features_match_pandas_arithmetic(spark, synthetic_pdf):
    """The Spark-side feature outputs equal the pure-pandas equivalents."""
    from src.features.spark_features import RowWiseFeatures

    sdf = spark.createDataFrame(synthetic_pdf)
    spark_out = RowWiseFeatures().transform(sdf).toPandas()

    # Sort both by id so row order matches.
    spark_out = spark_out.sort_values("id").reset_index(drop=True)
    pdf = synthetic_pdf.sort_values("id").reset_index(drop=True)

    np.testing.assert_allclose(
        spark_out["net_risk"].to_numpy(),
        (pdf["risk_drivers"] - pdf["risk_mitigators"]).to_numpy(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        spark_out["risk_ratio"].to_numpy(),
        (pdf["risk_drivers"] / (pdf["risk_mitigators"] + 1)).to_numpy(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        spark_out["emotional_x_stability"].to_numpy(),
        (pdf["r_ho_em2_co"] * pdf["r_ho_vi4_st"]).to_numpy(),
        rtol=1e-6,
    )


@pytest.mark.integration
def test_row_wise_features_skips_missing_inputs(spark):
    """Missing source columns => skip with warning, don't raise."""
    from src.features.spark_features import RowWiseFeatures

    # Frame missing risk_drivers + risk_mitigators
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "total_risk_score": [10.0, 20.0, 30.0],
            "r_ho_em2_co": [0.1, 0.2, 0.3],
            "r_ho_vi4_st": [-0.1, -0.2, -0.3],
        }
    )
    sdf = spark.createDataFrame(pdf)
    out = RowWiseFeatures().transform(sdf).toPandas()

    # net_risk + risk_ratio should NOT be added (their inputs are missing).
    assert "net_risk" not in out.columns
    assert "risk_ratio" not in out.columns
    # emotional_x_stability still works (its inputs are present).
    assert "emotional_x_stability" in out.columns


# --------------------------------------------------------------------------
# QuantileFlagger
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_quantile_flagger_fits_and_transforms(synthetic_sdf):
    """Fit produces a Model with thresholds; transform emits flag columns."""
    from src.features.quantile_flagger import QuantileFlagger

    est = QuantileFlagger(
        inputCols=["total_risk_score", "r_ho_em2_co"],
        highQuantile=0.7,
        lowQuantile=0.3,
    )
    model = est.fit(synthetic_sdf)

    # Thresholds learned for both columns.
    assert set(model.thresholds.keys()) == {"total_risk_score", "r_ho_em2_co"}
    for col in model.thresholds:
        assert "low" in model.thresholds[col]
        assert "high" in model.thresholds[col]
        assert model.thresholds[col]["low"] < model.thresholds[col]["high"]

    # Flag columns appear after transform.
    out = model.transform(synthetic_sdf).toPandas()
    for col in ("total_risk_score", "r_ho_em2_co"):
        assert f"{col}_high_flag" in out.columns
        assert f"{col}_low_flag" in out.columns
        # Flags are 0 / 1 integers.
        assert set(out[f"{col}_high_flag"].unique()).issubset({0, 1})
        assert set(out[f"{col}_low_flag"].unique()).issubset({0, 1})


@pytest.mark.integration
def test_quantile_flagger_prevents_leakage(spark, synthetic_pdf):
    """Fitting on train-only vs train+test produces DIFFERENT thresholds.

    The whole point of the Estimator/Transformer pattern is that fit only
    learns from rows it sees. If we feed it the full frame, the threshold
    must shift because the population is different. A non-zero delta is
    the structural property we depend on.
    """
    from src.features.quantile_flagger import QuantileFlagger

    rng = np.random.default_rng(1)
    train_pdf = synthetic_pdf.iloc[:700].copy()
    # Test rows with a deliberately different distribution to amplify
    # the threshold drift (mean shifted by +20).
    test_pdf = synthetic_pdf.iloc[700:].copy()
    test_pdf["total_risk_score"] = test_pdf["total_risk_score"] + 20 + rng.normal(0, 1, size=300)

    train_sdf = spark.createDataFrame(train_pdf)
    full_sdf = spark.createDataFrame(pd.concat([train_pdf, test_pdf], ignore_index=True))

    est = QuantileFlagger(inputCols=["total_risk_score"], highQuantile=0.7, lowQuantile=0.3)
    train_only_model = est.fit(train_sdf)
    leaky_model = est.fit(full_sdf)

    train_high = train_only_model.thresholds["total_risk_score"]["high"]
    leaky_high = leaky_model.thresholds["total_risk_score"]["high"]

    # The shifted test rows should pull the threshold up; if the threshold
    # didn't move at all, fit() saw the test rows by mistake.
    assert leaky_high > train_high, (
        f"Threshold did not move when test rows were added "
        f"(train-only={train_high}, train+test={leaky_high}); "
        "QuantileFlagger may not be fit-isolating correctly."
    )


@pytest.mark.integration
def test_quantile_flagger_save_load_roundtrip(synthetic_sdf, tmp_path: Path):
    """Save then load must round-trip the fitted thresholds exactly."""
    from src.features.quantile_flagger import QuantileFlagger, QuantileFlaggerModel

    est = QuantileFlagger(inputCols=["total_risk_score"], highQuantile=0.7, lowQuantile=0.3)
    model = est.fit(synthetic_sdf)

    save_path = tmp_path / "qf_model"
    model.write().save(str(save_path))

    loaded = QuantileFlaggerModel.read().load(str(save_path))

    assert loaded.inputCols == model.inputCols
    assert loaded.thresholds == model.thresholds


# --------------------------------------------------------------------------
# Pipeline composition
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_fit_transform_end_to_end(synthetic_sdf):
    """The composed Pipeline produces both row-wise features and quantile flags."""
    from src.features.pipeline import build_feature_pipeline

    pipeline = build_feature_pipeline(
        quantile_input_cols=["total_risk_score"],
        high_quantile=0.7,
        low_quantile=0.3,
    )
    fitted = pipeline.fit(synthetic_sdf)
    out = fitted.transform(synthetic_sdf).toPandas()

    # Row-wise features
    assert "net_risk" in out.columns
    assert "risk_ratio" in out.columns
    # Quantile flags
    assert "total_risk_score_high_flag" in out.columns
    assert "total_risk_score_low_flag" in out.columns
    # Same row count preserved
    assert len(out) == 1_000


@pytest.mark.integration
def test_pipeline_save_load_roundtrip(synthetic_sdf, tmp_path: Path):
    """Save the whole fitted Pipeline; reload it; outputs must match."""
    from pyspark.ml import PipelineModel

    from src.features.pipeline import build_feature_pipeline

    pipeline = build_feature_pipeline(
        quantile_input_cols=["total_risk_score"],
        high_quantile=0.7,
        low_quantile=0.3,
    )
    fitted = pipeline.fit(synthetic_sdf)

    save_path = tmp_path / "feature_pipeline"
    fitted.write().overwrite().save(str(save_path))

    reloaded = PipelineModel.load(str(save_path))

    original_out = (
        fitted.transform(synthetic_sdf).toPandas().sort_values("id").reset_index(drop=True)
    )
    reloaded_out = (
        reloaded.transform(synthetic_sdf).toPandas().sort_values("id").reset_index(drop=True)
    )

    np.testing.assert_array_equal(
        original_out["total_risk_score_high_flag"].to_numpy(),
        reloaded_out["total_risk_score_high_flag"].to_numpy(),
    )
    np.testing.assert_allclose(
        original_out["net_risk"].to_numpy(),
        reloaded_out["net_risk"].to_numpy(),
        rtol=1e-9,
    )
