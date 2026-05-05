"""Data contract for the application table.

Five expectations enforced at the raw -> staging boundary. A failure here
blocks the pipeline with a useful error rather than letting bad data drift
into the model.

Why these five:
    1. Row count band       catches truncated / partial loads
    2. Primary key uniqueness  catches duplicate insert / botched join upstream
    3. Target distribution   catches a flipped target encoding or gross sample bias
    4. Leakage column absence forces compliance with src/config.py LEAKAGE_COLUMNS
    5. DRA range bounds     catches a vendor change in the psychometric scoring scale

Adding a sixth: keep them ordered most-likely-to-fail first so the alarm
fires early. Don't add expectations that aren't actionable — if the team
won't pause the pipeline on a violation, don't write the expectation.
"""

from __future__ import annotations

import great_expectations as gx
import pandas as pd

from src.config import DRA_DIMENSIONS, LEAKAGE_COLUMNS, TARGET

# ----------------------------------------------------------------------------
# Suite parameters — tweak only with model owner sign-off
# ----------------------------------------------------------------------------

# Acceptable population size band. Historical: 44,998 rows. Tolerate ±5k.
EXPECTED_ROW_COUNT_MIN = 40_000
EXPECTED_ROW_COUNT_MAX = 50_000

# Bad rate tolerance. Historical: 24.2%. Tolerate ±4 pp.
EXPECTED_BAD_RATE_MIN = 0.20
EXPECTED_BAD_RATE_MAX = 0.28

# Primary key column.
PRIMARY_KEY = "dummy_id"

# DRA dimension scores are normalised; declare a generous range so the
# contract catches "vendor changed scoring scale" not minor distribution drift.
DRA_MIN = -10.0
DRA_MAX = 10.0


# ----------------------------------------------------------------------------
# The contract
# ----------------------------------------------------------------------------


def validate_application(df: pd.DataFrame) -> None:
    """Run the application suite. Raises ContractViolation on any failure.

    Idempotent and stateless — uses an ephemeral GE context, so it leaves
    no metadata on disk. Suitable for running on every pipeline invocation.
    """
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("application_validator")
    data_asset = data_source.add_dataframe_asset("application")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("snapshot")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = [
        # 1. Row count band
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=EXPECTED_ROW_COUNT_MIN,
            max_value=EXPECTED_ROW_COUNT_MAX,
        ),
        # 2. Primary key uniqueness + non-null
        gx.expectations.ExpectColumnValuesToBeUnique(column=PRIMARY_KEY),
        gx.expectations.ExpectColumnValuesToNotBeNull(column=PRIMARY_KEY),
        # 3. Target column: in {0, 1} and bad rate within tolerance
        gx.expectations.ExpectColumnValuesToBeInSet(
            column=TARGET,
            value_set=[0, 1],
        ),
        gx.expectations.ExpectColumnMeanToBeBetween(
            column=TARGET,
            min_value=EXPECTED_BAD_RATE_MIN,
            max_value=EXPECTED_BAD_RATE_MAX,
        ),
    ]

    # 4. Leakage columns must NOT be present in the validated frame.
    # GE's "expect column to not exist" is the cleanest way to enforce this.
    for leakage_col in LEAKAGE_COLUMNS:
        if leakage_col in df.columns:
            # Defer to ExpectColumnToNotExist would mark this as a graceful
            # failure; we want the LOUDEST possible signal because leakage
            # invalidates the model. Fail fast here.
            raise ContractViolation(
                f"Leakage column '{leakage_col}' is present in the staging "
                f"frame. Source data must be cleaned before validation."
            )

    # 5. DRA dimension range bounds.
    for dim_col in DRA_DIMENSIONS:
        if dim_col in df.columns:
            expectations.append(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=dim_col,
                    min_value=DRA_MIN,
                    max_value=DRA_MAX,
                    mostly=0.999,  # tolerate <0.1% out-of-band, fail above
                )
            )

    # Run the suite.
    failures: list[str] = []
    for exp in expectations:
        result = batch.validate(exp)
        if not result.success:
            failures.append(_summarise_failure(exp, result))

    if failures:
        msg = "Application data contract failed:\n" + "\n".join(f"  - {f}" for f in failures)
        raise ContractViolation(msg)


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class ContractViolation(RuntimeError):
    """Raised when the application data contract fails.

    Catching this anywhere except the orchestrator is forbidden — the
    pipeline must abort on a contract failure, never continue.
    """


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _summarise_failure(expectation, result) -> str:
    """Produce a one-line, human-readable summary of a failed expectation."""
    name = type(expectation).__name__
    kwargs = getattr(expectation, "configuration", None)
    observed = result.result.get("observed_value", "N/A") if hasattr(result, "result") else "N/A"
    return f"{name} | observed={observed} | config={kwargs}"
