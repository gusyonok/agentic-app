"""Validation tools for actuarial input payloads."""

from __future__ import annotations

from collections.abc import Iterable

from core.models import ValidationReport


REQUIRED_TRIANGLE_KEYS = {"accident_year", "development_period", "value"}


def validate_triangle_records(records: Iterable[dict]) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    flags: list[str] = []
    record_list = list(records)

    if not record_list:
        return ValidationReport(valid=False, errors=["No triangle records provided."])

    for idx, row in enumerate(record_list):
        missing = REQUIRED_TRIANGLE_KEYS - set(row.keys())
        if missing:
            errors.append(f"Row {idx} missing keys: {sorted(missing)}")
            continue
        if not isinstance(row["accident_year"], int):
            errors.append(f"Row {idx} accident_year must be an integer.")
        if not isinstance(row["development_period"], int):
            errors.append(f"Row {idx} development_period must be an integer.")
        if not isinstance(row["value"], (int, float)):
            errors.append(f"Row {idx} value must be numeric.")
            continue
        if row["value"] < 0:
            errors.append(f"Row {idx} has negative value.")
        if row["development_period"] <= 0:
            errors.append(f"Row {idx} has invalid development period.")

    seen_cells: set[tuple[int, int]] = set()
    by_ay: dict[int, dict[int, float]] = {}
    for idx, row in enumerate(record_list):
        if not REQUIRED_TRIANGLE_KEYS.issubset(row.keys()):
            continue
        if not isinstance(row["accident_year"], int) or not isinstance(row["development_period"], int):
            continue
        cell = (row["accident_year"], row["development_period"])
        if cell in seen_cells:
            errors.append(f"Duplicate triangle cell at row {idx}: {cell}.")
        seen_cells.add(cell)
        by_ay.setdefault(row["accident_year"], {})[row["development_period"]] = float(row["value"])

    for ay, dev_map in by_ay.items():
        ordered = [dev_map[d] for d in sorted(dev_map)]
        for left, right in zip(ordered, ordered[1:]):
            if right + 1e-9 < left:
                flags.append(f"AY {ay} is not monotonic cumulative by development period.")
                break

    ay_values = [row.get("accident_year") for row in record_list if "accident_year" in row]
    if len(set(ay_values)) < 3:
        warnings.append("Very few accident years provided; diagnostics may be unstable.")

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, data_quality_flags=flags)

