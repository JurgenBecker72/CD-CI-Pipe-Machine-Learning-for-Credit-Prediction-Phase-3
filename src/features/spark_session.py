"""Shared SparkSession factory for the credit pipeline.

PySpark 3.5 was released against Java 8/11/17 but does not auto-configure
the JDK module-access flags that Java 17 requires for its Arrow and
Unsafe-based serialisation paths. PySpark 4.x sets these automatically;
in 3.5 the caller has to opt in via `spark.driver.extraJavaOptions` and
`spark.executor.extraJavaOptions`. Without them, Python workers crash
with `java.io.EOFException` the moment any action is triggered.

Windows-specific concerns also handled here:

  * Spaces in the venv path break PYSPARK_PYTHON on Windows. Spark passes
    the executable to its worker spawn unquoted; a path like
    `C:\\Users\\Project Name\\.venv\\Scripts\\python.exe` is interpreted
    as `python.exe` somewhere wrong. Fix: convert to Windows' 8.3 short
    path (`PROJEC~1`) which never has spaces.

  * If the user has both conda and a uv venv activated, conda may have
    set PYSPARK_PYTHON in an activation hook to point at conda's base
    Python (which doesn't have pyspark). Fix: force the env var rather
    than using `setdefault`.

This module centralises the workaround so every entry point that needs a
SparkSession (training, tests, scripts) gets the same configuration.
"""

from __future__ import annotations

import os
import platform
import sys

# Module-access flags required by PySpark 3.5 on Java 17. Applied to
# both the driver and the executor JVMs.
_JAVA17_OPEN_FLAGS = " ".join(
    f"--add-opens=java.base/{module}=ALL-UNNAMED"
    for module in (
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


def _windows_short_path(long_path: str) -> str:
    """Convert a Windows long path to its 8.3 short form.

    Returns the original path on non-Windows platforms or if the
    Win32 call fails. The short form (e.g. `PROGRA~1`) never contains
    spaces, which avoids worker-spawn failures when PYSPARK_PYTHON is
    interpolated into a subprocess command line.
    """
    if platform.system() != "Windows":
        return long_path
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD

        buffer = ctypes.create_unicode_buffer(1024)
        size = get_short(long_path, buffer, 1024)
        return buffer.value if size > 0 else long_path
    except (OSError, AttributeError):
        return long_path


def _resolve_python_exec() -> str:
    """Return the Python executable Spark should use, safely on Windows."""
    return _windows_short_path(sys.executable)


def get_spark(
    app_name: str,
    *,
    driver_memory: str = "2g",
    shuffle_partitions: int = 4,
    arrow: bool = False,
):
    """Return a configured SparkSession.

    Parameters
    ----------
    app_name
        Identifier shown in the Spark UI / logs.
    driver_memory
        JVM heap for the driver. Bump to "4g" or "8g" for larger datasets.
    shuffle_partitions
        Sane default for laptop-scale data. Increase on a cluster.
    arrow
        Toggle pandas <-> Spark Arrow conversion. Disabled by default
        because Arrow's reflective access is the most fragile surface
        on Windows; the fallback path uses plain Python serialisation
        which is slower but predictable.
    """
    # Force these to point at the project's Python regardless of any
    # outer environment (e.g. conda). Use the 8.3 short path on Windows
    # so the worker spawn command doesn't choke on spaces.
    python_exec = _resolve_python_exec()
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.extraJavaOptions", _JAVA17_OPEN_FLAGS)
        .config("spark.executor.extraJavaOptions", _JAVA17_OPEN_FLAGS)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", str(arrow).lower())
        .config("spark.ui.showConsoleProgress", "false")
        # Pin worker reuse off; helps surface worker init errors early.
        .config("spark.python.worker.reuse", "false")
    )
    session = builder.getOrCreate()
    print(f"[spark_session] PYSPARK_PYTHON = {python_exec}")
    return session
