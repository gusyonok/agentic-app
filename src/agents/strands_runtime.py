"""Optional Strands runtime adapter.

This module keeps the app runnable even when Strands is not configured yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrandsAvailability:
    installed: bool
    detail: str


def check_strands_available() -> StrandsAvailability:
    try:
        import strands  # type: ignore # noqa: F401

        return StrandsAvailability(installed=True, detail="Strands import succeeded.")
    except Exception as exc:  # pragma: no cover
        return StrandsAvailability(installed=False, detail=f"Strands not active yet: {exc}")

