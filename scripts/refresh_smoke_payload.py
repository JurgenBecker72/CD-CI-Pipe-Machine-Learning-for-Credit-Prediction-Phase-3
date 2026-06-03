"""Generate ``scripts/smoke_payload.json`` from a real staging.application row.

The serving smoke test in ``scripts/smoke_score.ps1`` needs an applicant
payload that mirrors what the model was actually trained on. Sending a
hand-picked sparse payload produces a degenerate score (the model fills
missing features with 0 and reads that as an outlier). This generator
pulls one real, complete row from the warehouse and writes it to disk
as JSON so the smoke test is fully self-contained at runtime.

The chosen row is deterministic across runs and machines:

* The lowest ``dummy_id`` whose
* target is ``0`` (a non-defaulter — a sensible smoke baseline) and
* every column in ``BASE_FEATURES`` is non-null.

The output file is committed to the repository. Re-run this script
whenever ``BASE_FEATURES`` or the warehouse schema changes.

Usage
-----
    uv run python scripts/refresh_smoke_payload.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# Standalone scripts don't see the project root on sys.path; add it so
# `from src.* import ...` resolves the same way it does in the test
# suite and the training entry point.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASE_FEATURES, ID_COL, TARGET  # noqa: E402
from src.data.warehouse import read_sql  # noqa: E402

# scripts/smoke_payload.json — sits next to the smoke script so the
# PowerShell wrapper can resolve it via $PSScriptRoot without any
# repo-relative path arithmetic.
OUTPUT_PATH = Path(__file__).parent / "smoke_payload.json"

# Placeholder applied to the payload's dummy_id field. We deliberately
# do not echo the source row's real identifier into the committed
# fixture — the data is simulated, but the discipline is the discipline.
SMOKE_DUMMY_ID = "smoke-applicant-001"


def _to_jsonable(value: Any) -> Any:
    """Convert pandas / numpy scalars to JSON-native types.

    DuckDB returns numpy.int64, numpy.float64, etc. The standard json
    encoder rejects these. ``.item()`` unwraps any numpy scalar to its
    Python equivalent; NaN floats are normalised to ``None`` so the
    JSON file stays valid (JSON has no NaN literal).
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_query() -> str:
    """SQL for the lowest-id, complete, non-defaulting applicant."""
    columns = ", ".join(f'"{col}"' for col in [ID_COL, TARGET, *BASE_FEATURES])
    not_null = " AND ".join(f'"{col}" IS NOT NULL' for col in BASE_FEATURES)
    return (
        f"SELECT {columns} "
        f"FROM staging.application "
        f"WHERE {TARGET} = 0 AND {not_null} "
        f'ORDER BY "{ID_COL}" '
        f"LIMIT 1"
    )


def main() -> None:
    df = read_sql(_build_query())
    if df.empty:
        raise RuntimeError(
            "No staging.application row found with target=0 and all BASE_FEATURES "
            "populated. Check the warehouse refresh and the BASE_FEATURES list."
        )

    row = df.iloc[0].to_dict()
    source_id = row[ID_COL]

    payload: dict[str, Any] = {ID_COL: SMOKE_DUMMY_ID}
    for col in BASE_FEATURES:
        payload[col] = _to_jsonable(row[col])

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"  source dummy_id : {source_id}")
    print(f"  target (bad)    : {row[TARGET]}")
    print(f"  feature count   : {len(BASE_FEATURES)}")
    print(f"  payload dummy_id: {SMOKE_DUMMY_ID}")


if __name__ == "__main__":
    main()
