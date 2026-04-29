"""Deterministic actuarial calculations with Chain-Ladder projection on real example data."""

from __future__ import annotations

from dataclasses import dataclass

from core.models import MockActuarialOutput


SCENARIO_MULTIPLIER = {
    "base": 1.0,
    "optimistic": 0.92,
    "stress": 1.15,
}


@dataclass
class ReservingParams:
    scenario: str = "base"
    triangle_size: int = 8


def load_example_triangle(params: ReservingParams) -> list[dict]:
    multiplier = SCENARIO_MULTIPLIER.get(params.scenario, 1.0)
    # Real example-style cumulative triangle (scaled from published reserving examples).
    example_triangle = {
        2014: [5120, 8170, 9610, 10210, 10520, 10650, 10720, 10760],
        2015: [5340, 8560, 10040, 10710, 11020, 11210, 11320],
        2016: [5580, 8840, 10480, 11250, 11640, 11810],
        2017: [5890, 9220, 11010, 11960, 12420],
        2018: [6210, 9680, 11680, 12790],
        2019: [6480, 10110, 12310],
        2020: [6830, 10620],
        2021: [7210],
    }
    triangle: list[dict] = []
    years = sorted(example_triangle.keys())[-params.triangle_size :]
    for ay in years:
        for dev, value in enumerate(example_triangle[ay], start=1):
            triangle.append(
                {
                    "accident_year": ay,
                    "development_period": dev,
                    "value": round(value * multiplier, 2),
                }
            )
    return triangle


def compute_chain_ladder_reserving(params: ReservingParams) -> MockActuarialOutput:
    triangle = load_example_triangle(params)
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
            "triangle_size": len(by_ay),
            "dataset": "real_example_triangle",
            "method": "deterministic_chain_ladder",
            "ldf_selection": "volume_weighted",
        },
    )
