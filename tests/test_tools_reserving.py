import json
from pathlib import Path

from tools.reserving import ReservingParams, compute_chain_ladder_reserving, load_example_triangle
from tools.validation import validate_triangle_records


def test_example_triangle_and_validation():
    triangle = load_example_triangle(ReservingParams(scenario="base", triangle_size=6))
    report = validate_triangle_records(triangle)
    assert report.valid is True
    assert len(triangle) > 0


def test_chain_ladder_reserving_is_deterministic():
    p = ReservingParams(scenario="stress", triangle_size=7)
    out1 = compute_chain_ladder_reserving(p)
    out2 = compute_chain_ladder_reserving(p)
    assert out1.total_reserve == out2.total_reserve
    assert out1.reserve_by_ay == out2.reserve_by_ay
    assert len(out1.factors) > 0
    assert out1.total_reserve > 0
    assert all(f >= 1.0 for f in out1.factors)


def test_chain_ladder_against_fixture():
    fixture = json.loads(
        Path("tests/fixtures/triangle_sample.json").read_text(encoding="utf-8")
    )
    params = ReservingParams(**fixture["params"])
    output = compute_chain_ladder_reserving(params)

    assert output.total_reserve == fixture["expected_total_reserve"]
    assert output.factors == fixture["expected_factors"]
    assert output.cdf == fixture["expected_cdf"]
    assert output.ibnr_by_ay == fixture["expected_ibnr_by_ay"]
