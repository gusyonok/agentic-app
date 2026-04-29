"""Deterministic actuarial calculations with Chain-Ladder style projection."""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.models import MockActuarialOutput


SCENARIO_MULTIPLIER = {
    "base": 1.0,
    "optimistic": 0.92,
    "stress": 1.15,
}


@dataclass
class MockParams:
    scenario: str = "base"
    triangle_size: int = 8
    seed: int = 11


def _deterministic_rng(seed: int, scenario: str) -> random.Random:
    return random.Random(seed + sum(ord(c) for c in scenario))


def generate_mock_triangle(params: MockParams) -> list[dict]:
    rng = _deterministic_rng(params.seed, params.scenario)
    triangle: list[dict] = []
    multiplier = SCENARIO_MULTIPLIER.get(params.scenario, 1.0)
    base_ay = 2015
    cumulative_pattern = [0.45, 0.66, 0.8, 0.9, 0.96, 0.985, 0.995, 1.0]

    for i in range(params.triangle_size):
        ay = base_ay + i
        ultimate = (2100 + i * 175) * multiplier * (1 + rng.uniform(-0.03, 0.03))
        max_dev = params.triangle_size - i
        for dev in range(1, params.triangle_size - i + 1):
            pattern_idx = min(dev - 1, len(cumulative_pattern) - 1)
            pct = cumulative_pattern[pattern_idx]
            prev_pct = cumulative_pattern[pattern_idx - 1] if pattern_idx > 0 else 0.0
            # Keep cumulative strictly increasing while introducing mild noise.
            pct = max(prev_pct + 0.01, min(1.0, pct + rng.uniform(-0.01, 0.01)))
            if dev == max_dev:
                pct = min(1.0, pct + 0.005)
            value = ultimate * pct
            triangle.append(
                {
                    "accident_year": ay,
                    "development_period": dev,
                    "value": round(value, 2),
                }
            )
    return triangle


def compute_mock_reserving(params: MockParams) -> MockActuarialOutput:
    triangle = generate_mock_triangle(params)
    by_ay: dict[int, dict[int, float]] = {}
    for row in triangle:
        ay = int(row["accident_year"])
        dev = int(row["development_period"])
        by_ay.setdefault(ay, {})[dev] = float(row["value"])

    max_dev = max(dev for devs in by_ay.values() for dev in devs.keys())
    factors: list[float] = []
    for dev in range(1, max_dev):
        numerator = 0.0
        denominator = 0.0
        for dev_map in by_ay.values():
            if dev in dev_map and (dev + 1) in dev_map:
                numerator += dev_map[dev + 1]
                denominator += dev_map[dev]
        factors.append(round(numerator / denominator, 6) if denominator else 1.0)

    cdf: list[float] = []
    for idx in range(len(factors)):
        tail = 1.0
        for later in factors[idx:]:
            tail *= later
        cdf.append(round(tail, 6))

    latest_by_ay: dict[str, float] = {}
    ultimate_by_ay: dict[str, float] = {}
    ibnr_by_ay: dict[str, float] = {}
    reserve_by_ay: dict[str, float] = {}

    for ay in sorted(by_ay):
        dev_map = by_ay[ay]
        latest_dev = max(dev_map)
        latest_val = dev_map[latest_dev]
        latest_by_ay[str(ay)] = round(latest_val, 2)

        # If no future development remains, ultimate = latest.
        if latest_dev > len(cdf):
            ultimate = latest_val
        else:
            ultimate = latest_val * cdf[latest_dev - 1]
        ibnr = max(0.0, ultimate - latest_val)
        ultimate_by_ay[str(ay)] = round(ultimate, 2)
        ibnr_by_ay[str(ay)] = round(ibnr, 2)
        reserve_by_ay[str(ay)] = round(ibnr, 2)

    total = round(sum(reserve_by_ay.values()), 2)

    return MockActuarialOutput(
        triangle_records=triangle,
        factors=factors,
        cdf=cdf,
        latest_by_ay=latest_by_ay,
        ultimate_by_ay=ultimate_by_ay,
        ibnr_by_ay=ibnr_by_ay,
        reserve_by_ay=reserve_by_ay,
        total_reserve=total,
        diagnostics={
            "scenario": params.scenario,
            "triangle_size": params.triangle_size,
            "seed": params.seed,
            "method": "deterministic_chain_ladder",
            "ldf_selection": "volume_weighted",
        },
    )

