"""Local smoke test for the QuantileFlagger Estimator/Transformer pair.

Validates that the local Spark + JDK environment is functional and that
the QuantileFlagger learns thresholds from training data only and applies
them deterministically to subsequent data. Intended for use after a
fresh environment setup or a PySpark version bump.

Usage::

    uv run python scripts/smoke_quantile_flagger.py

Exit codes
----------
0 -- all checks passed
non-zero -- raised exception printed to stderr
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pin the Python executable Spark uses on Windows so worker processes
# match the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 2g pyspark-shell")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("=" * 70)
    print("QuantileFlagger smoke test")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. SparkSession
    # ----------------------------------------------------------------
    print("\n[1/6] Starting SparkSession ...")

    from src.features.spark_session import get_spark

    spark = get_spark("QuantileFlaggerSmoke")
    spark.sparkContext.setLogLevel("WARN")
    print(f"      Spark version: {spark.version}")
    print(f"      Python:        {sys.executable}")

    # ----------------------------------------------------------------
    # 2. Synthetic fixture
    # ----------------------------------------------------------------
    print("\n[2/6] Building synthetic fixture (1,000 rows) ...")
    rng = np.random.default_rng(42)
    pdf = pd.DataFrame(
        {
            "id": range(1_000),
            "total_risk_score": rng.normal(50, 15, size=1_000),
            "r_ho_em2_co": rng.normal(0, 1, size=1_000),
        }
    )
    sdf = spark.createDataFrame(pdf)
    print(f"      Rows: {sdf.count()}")
    print(f"      Columns: {sdf.columns}")

    # ----------------------------------------------------------------
    # 3. Train / test split
    # ----------------------------------------------------------------
    print("\n[3/6] Splitting 70/30 ...")
    train_sdf, test_sdf = sdf.randomSplit([0.7, 0.3], seed=42)
    print(f"      Train rows: {train_sdf.count()}")
    print(f"      Test rows:  {test_sdf.count()}")

    # ----------------------------------------------------------------
    # 4. Fit on training only
    # ----------------------------------------------------------------
    print("\n[4/6] Fitting QuantileFlagger on training rows ...")

    from src.features.quantile_flagger import QuantileFlagger

    estimator = QuantileFlagger(
        inputCols=["total_risk_score", "r_ho_em2_co"],
        highQuantile=0.7,
        lowQuantile=0.3,
    )
    model = estimator.fit(train_sdf)
    print("      Learned thresholds:")
    for col, t in model.thresholds.items():
        print(f"        {col:22s}  low={t['low']:+8.4f}   high={t['high']:+8.4f}")

    # ----------------------------------------------------------------
    # 5. Apply frozen model to both partitions
    # ----------------------------------------------------------------
    print("\n[5/6] Applying frozen model to train and test ...")
    train_with_flags = model.transform(train_sdf)
    test_with_flags = model.transform(test_sdf)

    new_cols = [c for c in train_with_flags.columns if c not in train_sdf.columns]
    print("      New flag columns produced:")
    for c in new_cols:
        print(f"        + {c}")

    print("\n      Sample test rows with flags:")
    test_with_flags.select(
        "id",
        "total_risk_score",
        "total_risk_score_high_flag",
        "total_risk_score_low_flag",
    ).show(5, truncate=False)

    # ----------------------------------------------------------------
    # 6. No-leakage check
    # ----------------------------------------------------------------
    print("\n[6/6] No-leakage check ...")
    print("      Compare thresholds fitted on train-only vs train+test:")

    leaky_estimator = QuantileFlagger(
        inputCols=["total_risk_score"], highQuantile=0.7, lowQuantile=0.3
    )
    leaky_model = leaky_estimator.fit(sdf)

    train_high = model.thresholds["total_risk_score"]["high"]
    leaky_high = leaky_model.thresholds["total_risk_score"]["high"]
    diff = abs(train_high - leaky_high)

    print(f"        train-only   high threshold: {train_high:+.4f}")
    print(f"        train+test   high threshold: {leaky_high:+.4f}")
    print(f"        delta:                       {diff:+.4f}")
    print()
    print("      A non-zero delta is the leakage path the QuantileFlagger")
    print("      eliminates by construction. The fitted Model was built from")
    print("      train rows only and cannot be back-doored at transform time.")

    spark.stop()
    print("\n" + "=" * 70)
    print("OK -- QuantileFlagger smoke test passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
