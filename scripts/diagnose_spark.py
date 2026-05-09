"""Minimal Spark diagnostic - isolates whether PySpark works at all on this machine.

No project code is imported. If this script crashes the same way the
test suite does, the issue is environmental (Java / Python / path /
antivirus). If this script succeeds but the test suite fails, the issue
is somewhere in our code.

Usage::

    uv run python scripts/diagnose_spark.py

Each test prints either OK or the error before the next stage runs, so
the failing stage is obvious in the output.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _short_path_if_windows(p: str) -> str:
    if platform.system() != "Windows":
        return p
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(1024)
        return buffer.value if get_short(p, buffer, 1024) > 0 else p
    except (OSError, AttributeError):
        return p


def main() -> None:
    print("=" * 70)
    print("Spark diagnostic — isolating the failing stage")
    print("=" * 70)

    print(f"\nPython:           {sys.version}")
    print(f"Python exec:      {sys.executable}")
    print(f"Path has spaces:  {' ' in sys.executable}")
    print(f"JAVA_HOME:        {os.environ.get('JAVA_HOME', '(unset)')}")

    python_for_spark = _short_path_if_windows(sys.executable)
    print(f"Resolved for Spark: {python_for_spark}")
    print(f"  spaces in resolved: {' ' in python_for_spark}")

    os.environ["PYSPARK_PYTHON"] = python_for_spark
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_for_spark

    print("\n[Stage 1] Import pyspark ...")
    from pyspark.sql import SparkSession

    print("           OK")

    print("\n[Stage 2] Build SparkSession ...")
    java_opens = " ".join(
        f"--add-opens=java.base/{m}=ALL-UNNAMED"
        for m in (
            "java.lang",
            "java.lang.invoke",
            "java.lang.reflect",
            "java.io",
            "java.net",
            "java.nio",
            "java.util",
            "java.util.concurrent",
            "java.util.concurrent.atomic",
            "sun.nio.ch",
            "sun.nio.cs",
            "sun.security.action",
            "sun.util.calendar",
        )
    )
    spark = (
        SparkSession.builder.appName("DiagnoseSpark")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.driver.extraJavaOptions", java_opens)
        .config("spark.executor.extraJavaOptions", java_opens)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"           OK -- Spark {spark.version}")

    print("\n[Stage 3] Pure JVM action: range(10).count() ...")
    df = spark.range(10)
    n = df.count()
    print(f"           OK -- count = {n}")

    print("\n[Stage 4] Triggers Python worker: range(10).show() ...")
    df.show()
    print("           OK -- workers responded")

    print("\n[Stage 5] createDataFrame from pandas ...")
    import pandas as pd

    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = spark.createDataFrame(pdf)
    print(f"           OK -- created Spark DataFrame with {sdf.count()} rows")

    print("\n[Stage 6] toPandas round-trip ...")
    back_to_pdf = sdf.toPandas()
    print(f"           OK -- got back {len(back_to_pdf)} rows in pandas")

    print("\n[Stage 7] approxQuantile ...")
    quantiles = sdf.approxQuantile("x", [0.3, 0.7], 0.001)
    print(f"           OK -- quantiles = {quantiles}")

    spark.stop()
    print("\n" + "=" * 70)
    print("All stages passed -- Spark works on this machine.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - we want the failure visible
        print("\n\nFAILED at stage above with:")
        print(f"  {type(exc).__name__}: {exc}")
        sys.exit(1)
