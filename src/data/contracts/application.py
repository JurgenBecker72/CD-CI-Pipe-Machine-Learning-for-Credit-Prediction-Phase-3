"""Data contract for the application table.

Five expectations enforced at the raw -> staging boundary. A failure here
blocks the pipeline with a useful error rather than letting bad data drift
into the model.

Why these five:
    1. Row count band       catches truncated / partial loads
    2. Primary key uniqueness catches duplicate insert / botched join upstream
    3. Target distribution   catches a flipped target encoding or gross sample bias
    4. Leakage column absence forces compliance with src/config.py LEAKAGE_COLUMNS
    5. DRA range bounds     catches a vendor change in the psychometric scoring scale

Implementation note
-------------------
This module was originally written against Great Expectations. GE 1.5.x pins
pandas<2.2, which conflicts with our pandas>=2.2 requirement; GE 1.6+ has
unrelated wheel-packaging bugs on Windows. Rather than fight the dependency
graph, the same five contracts are now expressed as plain pandas assertions
collected through a small `_check` helper. The external API is unchanged:

    from src.data.contracts.application import validate_application
    validate_application(df)        # raises ContractViolation if any check fails

If GE's situation improves we can swap back; the contract values live as
module-level constants either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import DRA_DIMENSIONS, LEAKAGE_COLUMNS, TARGET

# ----------------------------------------------------------------------------
# Suite parameters - tweak only with model owner sign-off
# ----------------------------------------------------------------------------

# Acceptable population size band. Historical: 44,998 rows. Tolerate +/- 5k.
EXPECTED_ROW_COUNT_MIN = 40_000
EXPECTED_ROW_COUNT_MAX = 50_000

# Bad rate tolerance. Historical: 24.2%. Tolerate +/- 4 pp.
EXPECTED_BAD_RATE_MIN = 0.20
EXPECTED_BAD_RATE_MAX = 0.28

# Primary key column.
PRIMARY_KEY = "dummy_id"

# DRA dimension scores are normalised; declare a generous range so the
# contract catches "vendor changed scoring scale" not minor distribution drift.
DRA_MIN = -10.0
DRA_MAX = 10.0


# ----------------------------------------------------------------------------
# Errors and result types
# ----------------------------------------------------------------------------


class ContractViolation(RuntimeError):
    """Raised when the application data contract fails.

    Catching this anywhere except the orchestrator is forbidden - the
    pipeline must abort on a contract failure, never continue.
    """


@dataclass
class CheckResult:
    """One expectation's outcome - useful for structured logging later."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class SuiteResult:
    """Aggregate result for a full suite run."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ----------------------------------------------------------------------------
# Individual check functions
# ----------------------------------------------------------------------------


def _check_row_count(df: pd.DataFrame) -> CheckResult:
    n = len(df)
    passed = EXPECTED_ROW_COUNT_MIN <= n <= EXPECTED_ROW_COUNT_MAX
    return CheckResult(
        name="row_count_in_band",
        passed=passed,
        detail=(
            f"observed {n:,} rows; expected {EXPECTED_ROW_COUNT_MIN:,}-"
            f"{EXPECTED_ROW_COUNT_MAX:,}"
        ),
    )


def _check_primary_key(df: pd.DataFrame) -> CheckResult:
    if PRIMARY_KEY not in df.columns:
        return CheckResult(
            name="primary_key_present",
            passed=False,
            detail=f"column {PRIMARY_KEY!r} missing from frame",
        )

    pk = df[PRIMARY_KEY]
    n_null = pk.isna().sum()
    n_dup = int(pk.duplicated().sum())

    passed = n_null == 0 and n_dup == 0
    detail = f"nulls={n_null}, duplicates={n_dup} (both must be 0)"
    return CheckResult(name="primary_key_unique_and_non_null", passed=passed, detail=detail)


def _check_target(df: pd.DataFrame) -> CheckResult:
    if TARGET not in df.columns:
        return CheckResult(
            name="target_present",
            passed=False,
            detail=f"column {TARGET!r} missing from frame",
        )

    target_values = set(df[TARGET].dropna().unique().tolist())
    expected_values = {0, 1}

    if not target_values.issubset(expected_values):
        return CheckResult(
            name="target_in_set",
            passed=False,
            detail=(
                f"observed values {sorted(target_values)}; expected subset of "
                f"{sorted(expected_values)}"
            ),
        )

    bad_rate = float(df[TARGET].mean())
    passed = EXPECTED_BAD_RATE_MIN <= bad_rate <= EXPECTED_BAD_RATE_MAX
    return CheckResult(
        name="target_distribution",
        passed=passed,
        detail=(
            f"bad rate {bad_rate:.4f}; expected {EXPECTED_BAD_RATE_MIN:.2f}-"
            f"{EXPECTED_BAD_RATE_MAX:.2f}"
        ),
    )


def _check_no_leakage(df: pd.DataFrame) -> CheckResult:
    present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    passed = len(present) == 0
    detail = (
        "no leakage columns present"
        if passed
        else f"leakage columns FOUND in frame: {present} - must be dropped before validation"
    )
    return CheckResult(name="no_leakage_columns", passed=passed, detail=detail)


def _check_dra_ranges(df: pd.DataFrame) -> CheckResult:
    out_of_range: dict[str, Any] = {}
    for col in DRA_DIMENSIONS:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        n_below = int((values < DRA_MIN).sum())
        n_above = int((values > DRA_MAX).sum())
        # Tolerate <0.1% out-of-band as a robustness margin
        tolerance = max(1, int(len(values) * 0.001))
        n_outside = n_below + n_above
        if n_outside > tolerance:
            out_of_range[col] = {
                "below_min": n_below,
                "above_max": n_above,
                "tolerance": tolerance,
            }

    passed = len(out_of_range) == 0
    detail = (
        f"all DRA dimensions within [{DRA_MIN}, {DRA_MAX}]"
        if passed
        else f"out-of-range DRA columns: {out_of_range}"
    )
    return CheckResult(name="dra_in_range", passed=passed, detail=detail)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def validate_application(df: pd.DataFrame) -> SuiteResult:
    """Run the application suite. Raises ContractViolation on any failure.

    Returns the SuiteResult on success so callers (or tests) can inspect
    individual check details. On any failure, raises with a multi-line
    error message listing every failure.
    """
    checks: list[CheckResult] = [
        _check_row_count(df),
        _check_primary_key(df),
        _check_target(df),
        _check_no_leakage(df),
        _check_dra_ranges(df),
    ]
    result = SuiteResult(checks=checks)

    if not result.all_passed:
        msg_lines = ["Application data contract failed:"]
        for c in result.failures:
            msg_lines.append(f"  - {c.name}: {c.detail}")
        raise ContractViolation("\n".join(msg_lines))

    return result
